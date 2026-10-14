// BC on-policy diagnosis: plays the PORTED student net and asks the expert
// what it would have done in every visited state. Reports on-policy
// agreement, first-divergence tick, death causes and scores.
//
// Usage: bc_diagnose NET=dataset/bc_ported.simple_net BOARD_SIZE=6 GAMES=20
// Read-only w.r.t. training (no files written); both trainers untouched.

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

#include "../nn_engine/simple_net/simple_net.h"
#include "../nn_engine/net_reader/simple_net_reader.h"
#include "./snake_environment.h"
#include "./snake_trainer.h"
#include "./bc_expert.h"

int main(int argc, char* argv[]) {
    std::string net_path = "dataset/bc_ported.simple_net";
    int board_size = 6;
    int games = 20;

    for(int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const size_t eq = arg.find('=');
        if(eq == std::string::npos) continue;
        const std::string key = arg.substr(0, eq);
        std::string val = arg.substr(eq + 1);
        if(val.size() >= 2 && val.front() == '"' && val.back() == '"')
            val = val.substr(1, val.size() - 2);
        if(key == "NET") net_path = val;
        else if(key == "BOARD_SIZE") board_size = std::stoi(val);
        else if(key == "GAMES") games = std::stoi(val);
        else throw std::runtime_error("bc_diagnose: unknown arg '" + key + "'");
    }

    SimpleNet net;
    try { net = SimpleNetReader::load(net_path); }
    catch(const std::exception& e) { throw std::runtime_error("bc_diagnose: load failed: " + std::string(e.what())); }

    long long agree = 0, total = 0, first_div_sum = 0;
    int nodiv_games = 0;
    long long score_sum = 0;
    int deaths[6] = {0, 0, 0, 0, 0, 0};

    for(int g = 0; g < games; ++g) {
        SnakesGame game(board_size);
        game.reset();
        int first_div = -1, ticks = 0;
        while(!game.isGameOver() && game.game_ticks < game.max_game_tick) {
            const std::vector<float> state = game.getState();
            const int student = SnakeTrainer::pickMove(game, net.predict(state));
            const int teacher = BCExpert::suggestMove(game);
            if(student == teacher) ++agree;
            else if(first_div < 0) first_div = ticks;
            ++total;
            ++ticks;
            game.step(student);
            if(ticks > 5000) break; // safety; expert games end far earlier
        }
        if(first_div < 0) { ++nodiv_games; }
        else first_div_sum += first_div;
        score_sum += (int)game.snake.size - 1;
        const int cause = game.deathCause();
        if(cause >= 0 && cause <= 5) ++deaths[cause];
        std::cout << "game " << g << " ticks=" << ticks << " score=" << ((int)game.snake.size - 1)
                  << " cause=" << cause << " first_div=" << first_div << "\n";
    }

    std::cout << "on_policy_agree=" << (double)agree / (double)total
              << " nodivert_games=" << nodiv_games << "/" << games
              << " avg_first_div_tick=" << (games - nodiv_games > 0 ? (double)first_div_sum / (games - nodiv_games) : -1)
              << " avg_score=" << (double)score_sum / games << "\n"
              << "deaths: alive=" << deaths[0] << " wall=" << deaths[1] << " body=" << deaths[2]
              << " win=" << deaths[3] << " tickcap=" << deaths[4] << " starved=" << deaths[5] << "\n";
    return 0;
}
