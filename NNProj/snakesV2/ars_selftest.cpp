// ARS self-test: maximize -sphere(x) so the optimum (0) is approached.
// Verifies flatten/update math without burning CPU on snake games.
// Pass criteria: best return improves monotonically-ish and ends far above
// start. Deterministic (fixed seeds). Exit 0 = PASS, 1 = FAIL.

#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "./ars_optimizer.h"

static float negSphere(const std::vector<float>& x) {
    double s = 0.0;
    for(float v : x) s += (double)v * v;
    return (float)-s;
}

int main() {
    const size_t dim = 30;
    const int dirs = 32;
    const int top_b = 16;
    const float alpha = 0.02f;
    const float noise = 0.05f;
    const int iters = 300;

    std::mt19937 init(12345);
    std::uniform_real_distribution<float> unif(-3.0f, 3.0f);
    std::vector<float> theta(dim);
    for(float& v : theta) v = unif(init);

    const float start = negSphere(theta);
    float best = start;
    std::cout << "sphere dim=" << dim << " start=" << start << "\n";

    std::vector<uint32_t> seeds((size_t)dirs);
    std::vector<float> r_plus((size_t)dirs), r_minus((size_t)dirs);
    std::vector<float> dir(dim), cand(dim);
    std::mt19937 seeder(999);

    for(int t = 0; t < iters; ++t) {
        for(int k = 0; k < dirs; ++k) seeds[(size_t)k] = seeder();
        for(int k = 0; k < dirs; ++k) {
            ARS::sampleDirection(seeds[(size_t)k], dim, dir);
            for(size_t i = 0; i < dim; ++i) cand[i] = theta[i] + noise * dir[i];
            r_plus[(size_t)k] = negSphere(cand);
            for(size_t i = 0; i < dim; ++i) cand[i] = theta[i] - noise * dir[i];
            r_minus[(size_t)k] = negSphere(cand);
        }
        float sigma_r = 0.0f;
        ARS::update(theta, seeds, r_plus, r_minus, top_b, alpha, 8.0f, &sigma_r);
        const float cur = negSphere(theta);
        if(cur > best) best = cur;
        if(t % 50 == 0 || t == iters - 1)
            std::cout << "iter " << t << " cur=" << cur << " best=" << best
                      << " sigma_r=" << sigma_r << "\n";
    }

    // Start was about -270 (30 dims * ~9); must get much closer to 0.
    const bool pass = (best > start) && (best > -5.0f);
    std::cout << (pass ? "ARS_SELFTEST: PASS" : "ARS_SELFTEST: FAIL")
              << " start=" << start << " best=" << best << "\n";
    return pass ? 0 : 1;
}
