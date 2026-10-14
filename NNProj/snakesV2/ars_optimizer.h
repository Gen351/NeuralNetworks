#pragma once

// Augmented Random Search (ARS V1-t, Mania et al. 2018) for snakesV2
// neuroevolution. Replaces tournament-selection + sparse Gaussian mutation
// with antithetic finite-difference updates on the flattened weight vector.
//
// Per generation, with mean parameters theta and N directions:
//   sample d_k ~ N(0, I)  (regenerated from a stored uint32 seed on demand,
//                          so a population costs bytes, not megabytes)
//   evaluate r+ = F(theta + nu*d_k), r- = F(theta - nu*d_k) on IDENTICAL
//            game seeds (common random numbers -> fair +/- comparison)
//   keep top-b directions by max(r+, r-), then:
//     theta += (alpha / (b * sigma_R)) * sum_top-b (r+ - r-) * d_k
// where sigma_R is the std of the 2N returns (reward-scale invariance).
//
// State (input) normalization from ARS-V2 is skipped: V2 board inputs are
// already in [-1, 1]. Everything here is pure vector math -- game evals,
// threading, logging and saving stay in main.cpp.

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>

#include "../nn_engine/simple_net/simple_net.h"
#include "../nn_engine/simple_net/simple_layer.h"

namespace ARS {

// Number of trainable floats (weights + biases) in a net. Layer order and
// shapes must stay fixed for flatten/unflatten to round-trip.
inline size_t paramDim(const SimpleNet& net) {
    size_t dim = 0;
    for(const auto& l : net.layers) {
        const SimpleLayer* dense = dynamic_cast<const SimpleLayer*>(l.get());
        if(!dense) continue; // activation layer
        dim += dense->weights.data.size() + dense->biases.size();
    }
    return dim;
}

// Weights (row-major) then biases per dense layer, in layer order.
inline void flattenInto(const SimpleNet& net, std::vector<float>& out) {
    out.clear();
    out.reserve(paramDim(net));
    for(const auto& l : net.layers) {
        const SimpleLayer* dense = dynamic_cast<const SimpleLayer*>(l.get());
        if(!dense) continue;
        out.insert(out.end(), dense->weights.data.begin(), dense->weights.data.end());
        out.insert(out.end(), dense->biases.begin(), dense->biases.end());
    }
}

// Inverse of flattenInto. net must have identical shapes (throws otherwise).
inline void unflattenInto(const std::vector<float>& v, SimpleNet& net) {
    size_t pos = 0;
    for(const auto& l : net.layers) {
        SimpleLayer* dense = dynamic_cast<SimpleLayer*>(l.get());
        if(!dense) continue;
        const size_t nw = dense->weights.data.size();
        const size_t nb = dense->biases.size();
        if(pos + nw + nb > v.size())
            throw std::runtime_error("ARS::unflattenInto: size mismatch");
        std::copy(v.begin() + pos, v.begin() + pos + nw, dense->weights.data.begin());
        pos += nw;
        std::copy(v.begin() + pos, v.begin() + pos + nb, dense->biases.begin());
        pos += nb;
    }
    if(pos != v.size())
        throw std::runtime_error("ARS::unflattenInto: trailing params");
}

// One antithetic direction ~ N(0, I), deterministically from its seed.
inline void sampleDirection(uint32_t seed, size_t dim, std::vector<float>& dir) {
    dir.resize(dim);
    std::mt19937 engine(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    for(float& x : dir) x = gauss(engine);
}

inline float rewardStd(const std::vector<float>& returns) {
    if(returns.size() < 2) return 0.0f;
    double mean = 0.0;
    for(float r : returns) mean += r;
    mean /= (double)returns.size();
    double var = 0.0;
    for(float r : returns) { const double d = r - mean; var += d * d; }
    return (float)std::sqrt(var / (double)returns.size());
}

// ARS V1-t update. dir_seeds/r_plus/r_minus have N entries; top_b <= N.
// theta is updated in place and clamped to [-clamp, clamp] (same convention
// as the ES mutate() clamp, keeps TANH nets out of saturation).
// sigma_r_out receives the return std (for logging/diagnostics).
inline void update(std::vector<float>& theta
        , const std::vector<uint32_t>& dir_seeds
        , const std::vector<float>& r_plus
        , const std::vector<float>& r_minus
        , int top_b
        , float alpha
        , float clamp
        , float* sigma_r_out = nullptr) {
    const size_t n = dir_seeds.size();
    if(n == 0) throw std::runtime_error("ARS::update: no directions");
    if(r_plus.size() != n || r_minus.size() != n)
        throw std::runtime_error("ARS::update: returns/directions mismatch");
    if(top_b <= 0 || top_b > (int)n)
        throw std::runtime_error("ARS::update: bad top_b");

    std::vector<float> all;
    all.reserve(2 * n);
    all.insert(all.end(), r_plus.begin(), r_plus.end());
    all.insert(all.end(), r_minus.begin(), r_minus.end());
    const float sigma_r = rewardStd(all);
    if(sigma_r_out) *sigma_r_out = sigma_r;
    if(!(sigma_r > 1e-8f)) return; // all returns identical: no signal

    // Rank directions by max(r+, r-), keep top-b.
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return std::max(r_plus[a], r_minus[a]) > std::max(r_plus[b], r_minus[b]);
    });

    const float scale = alpha / ((float)top_b * sigma_r);
    std::vector<float> dir(theta.size());
    for(int t = 0; t < top_b; ++t) {
        const size_t k = order[(size_t)t];
        const float w = (r_plus[k] - r_minus[k]) * scale;
        if(w == 0.0f) continue;
        sampleDirection(dir_seeds[k], theta.size(), dir);
        for(size_t i = 0; i < theta.size(); ++i) theta[i] += w * dir[i];
    }

    for(float& x : theta) x = std::clamp(x, -clamp, clamp);
}

} // namespace ARS
