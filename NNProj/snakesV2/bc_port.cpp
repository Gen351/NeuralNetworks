// Port a trained behavioral-cloning net (.backprop_net) to .simple_net for
// test.exe replay and ES CONTINUE_TRAINING fine-tuning.
//
// The two formats store identical dense payloads (row-major weights then
// biases); only the magic + dense tag differ, so this is an exact rewrite:
//   BACKPROP_NET 1 -> SIMPLE_NET 1,  DEN in out -> LAY in out,  ACT TAG kept.
// SOFT activations are rejected (SimpleNet cannot load them); the BC nets
// are all-TANH by construction.
//
// Verification is built in: both files are re-loaded with their real readers
// and 200 random inputs must agree to < 1e-5. Exit 0 = ported + verified.
//
// Usage: bc_port IN=dataset/bc_long OUT=dataset/bc_ported
//   (writes dataset/bc_ported.simple_net)

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <stdexcept>

#include "../nn_engine/backprop_net/backprop_net.h"
#include "../nn_engine/backprop_net/backprop_net_reader.h"
#include "../nn_engine/simple_net/simple_net.h"
#include "../nn_engine/net_reader/simple_net_reader.h"

namespace {

bool isSimpleAct(const std::string& tag) {
    return tag == "SIGM" || tag == "RELU" || tag == "LEAK" || tag == "TANH";
}

} // namespace

int main(int argc, char* argv[]) {
    std::string in_name = "dataset/bc_net";
    std::string out_name = "dataset/bc_ported";

    for(int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const size_t eq = arg.find('=');
        if(eq == std::string::npos) continue;
        const std::string key = arg.substr(0, eq);
        std::string val = arg.substr(eq + 1);
        if(val.size() >= 2 && val.front() == '"' && val.back() == '"')
            val = val.substr(1, val.size() - 2);
        if(key == "IN") in_name = val;
        else if(key == "OUT") out_name = val;
        else throw std::runtime_error("bc_port: unknown arg '" + key + "'");
    }

    // Parse source tokens (whitespace-separated; newlines irrelevant).
    std::string in_path = in_name;
    if(in_path.size() < 13 || in_path.substr(in_path.size() - 13) != ".backprop_net")
        in_path += ".backprop_net";
    std::ifstream in(in_path);
    if(!in.is_open()) throw std::runtime_error("bc_port: cannot open " + in_path);
    std::vector<std::string> toks;
    {
        std::string t;
        while(in >> t) toks.push_back(t);
    }
    if(toks.size() < 3 || toks[0] != "BACKPROP_NET" || toks[1] != "1")
        throw std::runtime_error("bc_port: not a v1 .backprop_net file: " + in_path);

    size_t pos = 2;
    const size_t layers = (size_t)std::stoull(toks[pos++]);
    std::string out_text = "SIMPLE_NET 1\n" + std::to_string(layers) + "\n";
    for(size_t l = 0; l < layers; ++l) {
        if(pos >= toks.size()) throw std::runtime_error("bc_port: truncated layer " + std::to_string(l));
        const std::string tag = toks[pos++];
        if(tag == "DEN") {
            if(pos + 1 >= toks.size()) throw std::runtime_error("bc_port: truncated dims");
            const size_t inn = (size_t)std::stoull(toks[pos++]);
            const size_t outn = (size_t)std::stoull(toks[pos++]);
            if(inn == 0 || outn == 0) throw std::runtime_error("bc_port: bad dims");
            out_text += "LAY " + std::to_string(inn) + " " + std::to_string(outn) + "\n";
            const size_t count = inn * outn + outn;
            for(size_t k = 0; k < count; ++k) {
                if(pos >= toks.size()) throw std::runtime_error("bc_port: truncated payload");
                out_text += toks[pos++] + " ";
            }
            out_text += "\n";
        } else if(tag == "ACT") {
            if(pos >= toks.size()) throw std::runtime_error("bc_port: truncated act");
            const std::string act = toks[pos++];
            if(!isSimpleAct(act))
                throw std::runtime_error("bc_port: activation '" + act + "' has no SimpleNet equivalent");
            out_text += "ACT " + act + "\n";
        } else {
            throw std::runtime_error("bc_port: unknown tag '" + tag + "'");
        }
    }

    std::string out_path = out_name;
    if(out_path.size() < 11 || out_path.substr(out_path.size() - 11) != ".simple_net")
        out_path += ".simple_net";
    {
        std::ofstream out(out_path);
        if(!out.is_open()) throw std::runtime_error("bc_port: cannot open " + out_path);
        out << out_text;
        out.flush();
        if(!out) throw std::runtime_error("bc_port: failed writing " + out_path);
    }

    // Verify: reload both with the real readers, compare predictions.
    BackpropNet::Net src = BackpropNetReader::load(in_name);
    SimpleNet dst = SimpleNetReader::load(out_name);
    size_t in_dim = 0;
    for(const auto& l : dst.layers) {
        if(const SimpleLayer* d = dynamic_cast<const SimpleLayer*>(l.get())) {
            in_dim = d->weights.cols();
            break;
        }
    }
    if(in_dim == 0) throw std::runtime_error("bc_port: ported net has no dense layer");

    std::mt19937 rng(2024);
    std::uniform_real_distribution<float> unif(-1.0f, 1.0f);
    float max_diff = 0.0f;
    for(int t = 0; t < 200; ++t) {
        std::vector<float> x(in_dim);
        for(float& v : x) v = unif(rng);
        const std::vector<float> a = src.predict(x);
        const std::vector<float> b = dst.predict(x);
        if(a.size() != b.size()) throw std::runtime_error("bc_port: output size mismatch");
        for(size_t i = 0; i < a.size(); ++i)
            max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    std::cout << "bc_port: wrote " << out_path << " max_pred_diff=" << max_diff << "\n";
    const bool pass = max_diff < 1e-5f;
    std::cout << (pass ? "BC_PORT: PASS" : "BC_PORT: FAIL") << "\n";
    return pass ? 0 : 1;
}
