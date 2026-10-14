// Behavioral-cloning trainer for snakesV2.
// Architecture mirrors the ES net (IN->H1->H2->3, all TANH, XAVIER) so the
// trained weights port 1:1 into .simple_net for test.exe / CONTINUE_TRAINING.
// Targets are one-hot move vectors; the engine's MSE gradient handles them.
//
// Usage (from snakesV2/):
//   bc_train TRAIN=dataset/bc_train.csv VAL=dataset/bc_val.csv OUT=dataset/bc_net EPOCHS=40 BATCH=32 LR=0.001
// Writes <OUT>.backprop_net + prints train/val argmax-match accuracy.
// Exit 0 + "PASS" when val accuracy >= 0.85, else exit 1 ("FAIL").
//
// New files only (dataset/ is git-ignored scratch); both trainers untouched.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>

#include "../nn_engine/backprop_net/backprop_net.h"
#include "../nn_engine/backprop_net/backprop_net_reader.h"

namespace {

struct LabeledSet {
    std::vector<std::vector<float>> x;
    std::vector<std::vector<float>> y; // one-hot over 3 moves
};

LabeledSet loadCsv(const std::string& path) {
    std::ifstream f(path);
    if(!f.is_open()) throw std::runtime_error("bc_train: cannot open " + path);
    LabeledSet ds;
    std::string line;
    while(std::getline(f, line)) {
        if(!line.empty() && line.back() == '\r') line.pop_back();
        if(line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;
        while(std::getline(ss, cell, ',')) row.push_back(std::stof(cell));
        if(row.size() < 2) throw std::runtime_error("bc_train: bad row in " + path);
        const int label = (int)row.back();
        if(label < 0 || label > 2) throw std::runtime_error("bc_train: bad label in " + path);
        row.pop_back();
        ds.x.push_back(std::move(row));
        ds.y.push_back({0.0f, 0.0f, 0.0f});
        ds.y.back()[(size_t)label] = 1.0f;
    }
    if(ds.x.empty()) throw std::runtime_error("bc_train: empty dataset " + path);
    return ds;
}

int argmax(const std::vector<float>& v) {
    int best = 0;
    for(size_t i = 1; i < v.size(); ++i)
        if(v[i] > v[best]) best = (int)i;
    return best;
}

float accuracy(BackpropNet::Net& net, const LabeledSet& ds) {
    size_t hit = 0;
    for(size_t i = 0; i < ds.x.size(); ++i) {
        const std::vector<float> out = net.predict(ds.x[i]);
        int want = 0;
        for(int k = 1; k < 3; ++k)
            if(ds.y[i][(size_t)k] > ds.y[i][(size_t)want]) want = k;
        if(argmax(out) == want) ++hit;
    }
    return (float)hit / (float)ds.x.size();
}

} // namespace

int main(int argc, char* argv[]) {
    std::string train_path = "dataset/bc_train.csv";
    std::string val_path = "dataset/bc_val.csv";
    std::string out_name = "dataset/bc_net";
    size_t epochs = 40, batch = 32, patience = 8;
    float lr = 0.001f;
    size_t h1 = 56, h2 = 24;

    for(int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const size_t eq = arg.find('=');
        if(eq == std::string::npos) continue;
        const std::string key = arg.substr(0, eq);
        std::string val = arg.substr(eq + 1);
        if(val.size() >= 2 && val.front() == '"' && val.back() == '"')
            val = val.substr(1, val.size() - 2);
        if(key == "TRAIN") train_path = val;
        else if(key == "VAL") val_path = val;
        else if(key == "OUT") out_name = val;
        else if(key == "EPOCHS") epochs = (size_t)std::stoull(val);
        else if(key == "BATCH") batch = (size_t)std::stoull(val);
        else if(key == "PATIENCE") patience = (size_t)std::stoull(val);
        else if(key == "LR") lr = std::stof(val);
        else if(key == "H1") h1 = (size_t)std::stoull(val);
        else if(key == "H2") h2 = (size_t)std::stoull(val);
        else throw std::runtime_error("bc_train: unknown arg '" + key + "'");
    }

    const LabeledSet train = loadCsv(train_path);
    const LabeledSet val = loadCsv(val_path);
    const size_t in_dim = train.x[0].size();
    std::cout << "bc_train: train=" << train.x.size() << " val=" << val.x.size()
              << " in=" << in_dim << " arch=" << in_dim << "->" << h1 << "->" << h2 << "->3 (TANH)\n";

    BackpropNet::Net net;
    net.add_layer(std::make_unique<BackpropNet::DenseLayer>(in_dim, h1, BackpropNet::DenseLayer::InitType::XAVIER));
    net.add_layer(std::make_unique<BackpropNet::ActivationLayer>(BackpropNet::ActivationLayer::ActType::TANH));
    net.add_layer(std::make_unique<BackpropNet::DenseLayer>(h1, h2, BackpropNet::DenseLayer::InitType::XAVIER));
    net.add_layer(std::make_unique<BackpropNet::ActivationLayer>(BackpropNet::ActivationLayer::ActType::TANH));
    net.add_layer(std::make_unique<BackpropNet::DenseLayer>(h2, 3, BackpropNet::DenseLayer::InitType::XAVIER));
    net.add_layer(std::make_unique<BackpropNet::ActivationLayer>(BackpropNet::ActivationLayer::ActType::TANH));

    net.train_v2(epochs, batch, patience, lr, train.x, train.y, &val.x, &val.y);

    BackpropNetReader::save(out_name, net);
    const float train_acc = accuracy(net, train);
    const float val_acc = accuracy(net, val);
    std::cout << "bc_train: train_acc=" << train_acc << " val_acc=" << val_acc << "\n";

    const bool pass = val_acc >= 0.85f;
    std::cout << (pass ? "BC_TRAIN: PASS" : "BC_TRAIN: FAIL (val < 0.85)") << "\n";
    return pass ? 0 : 1;
}
