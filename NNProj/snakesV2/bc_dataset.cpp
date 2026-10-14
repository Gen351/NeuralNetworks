// Behavioral-cloning dataset generator for snakesV2.
// Rolls out the BCExpert teacher and records EXACT runtime features
// (SnakesGame::getState(), the same vector the net sees when playing).
//
// Output: <OUT>_train.csv / <OUT>_val.csv, no header, one sample per line:
//   <64 floats...,><label 0/1/2 = left/straight/right>
// Classes are trimmed to equal counts (else the net learns "always go
// straight"), shuffled with a fixed seed, split 90/10.
//
// Usage (from snakesV2/):
//   bc_dataset SAMPLES=100000 BOARD_SIZE=6 OUT=dataset/bc SEED=123
//
// New files only (dataset/ is git-ignored scratch); both trainers untouched.

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <stdexcept>
#include <algorithm>

#include "./snake_environment.h"
#include "./bc_expert.h"

namespace {

struct Sample {
    std::vector<float> x;
    int label = 0; // 0 = left, 1 = straight, 2 = right
};

void writeCsv(const std::string& path, const std::vector<Sample>& samples) {
    std::ofstream f(path);
    if(!f.is_open()) throw std::runtime_error("bc_dataset: cannot open " + path);
    for(const auto& s : samples) {
        for(size_t i = 0; i < s.x.size(); ++i) f << s.x[i] << ',';
        f << s.label << '\n';
    }
    f.flush();
    if(!f) throw std::runtime_error("bc_dataset: failed writing " + path);
}

} // namespace

int main(int argc, char* argv[]) {
    size_t samples = 100000;
    int board_size = 6;
    std::string out_prefix = "dataset/bc";
    uint32_t seed = 123;
    const size_t max_games = 40000;
    const size_t tick_cap = 5000;

    for(int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const size_t eq = arg.find('=');
        if(eq == std::string::npos) continue;
        const std::string key = arg.substr(0, eq);
        std::string val = arg.substr(eq + 1);
        if(val.size() >= 2 && val.front() == '"' && val.back() == '"')
            val = val.substr(1, val.size() - 2);
        if(key == "SAMPLES") samples = (size_t)std::stoull(val);
        else if(key == "BOARD_SIZE") board_size = std::stoi(val);
        else if(key == "OUT") out_prefix = val;
        else if(key == "SEED") seed = (uint32_t)std::stoul(val);
        else throw std::runtime_error("bc_dataset: unknown arg '" + key + "'");
    }

    std::mt19937 rng(seed);
    SnakesGame probe(board_size);
    const size_t feat = probe.getStateSize();
    std::cout << "bc_dataset: board=" << board_size << " features=" << feat
              << " target_samples=" << samples << "\n";

    // Per-class pools (0 = left, 1 = straight, 2 = right).
    std::vector<Sample> pools[3];
    const size_t quota = samples / 3 + 1;

    size_t games = 0, ticks = 0;
    long long apples = 0;
    size_t min_ever = 0;
    while(games < max_games) {
        SnakesGame g(board_size, rng());
        size_t t = 0;
        while(!g.isGameOver() && t < tick_cap) {
            const std::vector<float> state = g.getState();
            if(state.size() != feat)
                throw std::runtime_error("bc_dataset: feature size changed mid-run");
            const int move = BCExpert::suggestMove(g); // 1/2/3
            if(move < 1 || move > 3)
                throw std::runtime_error("bc_dataset: expert returned invalid move");
            const int apples_before = (int)g.snake.size;
            g.step(move);
            apples += (int)g.snake.size - apples_before;
            Sample s;
            s.x = state;
            s.label = move - 1;
            if(pools[(size_t)s.label].size() < quota)
                pools[(size_t)s.label].push_back(std::move(s));
            ++t;
        }
        ++games;
        ticks += t;
        size_t m = pools[0].size();
        for(int c = 1; c < 3; ++c) m = std::min(m, pools[c].size());
        min_ever = m;
        if(games % 500 == 0)
            std::cout << "games=" << games << " pooled=" << (pools[0].size() + pools[1].size() + pools[2].size())
                      << " minclass=" << m << " apples=" << apples << "\n";
        if(min_ever >= quota) break;
    }

    std::cout << "games=" << games << " apples=" << apples
              << " avg_score=" << (apples / (double)games) << "\n";
    if(games > 0 && apples / (double)games < 1.0) {
        std::cerr << "bc_dataset: WARNING: expert barely eats (avg "
                  << (apples / (double)games) << " apples/game); labels suspect\n";
    }

    // Trim to equal class counts for a balanced set.
    size_t keep = std::min({pools[0].size(), pools[1].size(), pools[2].size()});
    if(keep == 0) throw std::runtime_error("bc_dataset: empty class, no data");
    std::vector<Sample> all;
    all.reserve(keep * 3);
    for(int c = 0; c < 3; ++c)
        all.insert(all.end(), pools[c].begin(), pools[c].begin() + keep);
    std::shuffle(all.begin(), all.end(), std::mt19937(seed + 1));

    const size_t n_val = all.size() / 10;
    std::vector<Sample> val(all.end() - n_val, all.end());
    std::vector<Sample> train(all.begin(), all.end() - n_val);

    writeCsv(out_prefix + "_train.csv", train);
    writeCsv(out_prefix + "_val.csv", val);
    std::cout << "bc_dataset: wrote " << train.size() << " train + " << val.size()
              << " val -> " << out_prefix + "_{train,val}.csv\n";
    return 0;
}
