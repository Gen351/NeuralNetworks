#include <iostream>
#include <random>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <future>
#include <mutex>
#include <atomic>
#include <queue>
#include <functional>
#include <condition_variable>
#include <cctype>

#include "../nn_engine/simple_net/simple_net.h"
#include "../nn_engine/net_reader/simple_net_reader.h"
#include "./snake_environment.h"
#include "./snake_trainer.h"
#include "./ars_optimizer.h"

void mutate(SimpleNet& net
        , const float mutation_chance
        , const float sigma);

struct RunSnapshot {
    float fitness;
    size_t gen;
    int score;
    std::string screenshot;
};

std::string renderScreenshot(const SnakesGame& game);
int findNextRunVersion();
RunSnapshot makeSnapshot(const SnakesGame& game, const float fitness, const size_t gen);
void save(const std::string& run_dir
        , const SimpleNet& best
        , const SimpleNet& current
        , const RunSnapshot& best_snap
        , const RunSnapshot& curr_snap
        , const size_t board_size);
void initGenLog(const std::string& run_dir
        , int argc
        , char* argv[]
        , int board_size
        , int pop_size
        , int evals_per_net
        , float mutation_chance
        , float sigma
        , int max_gen
        , int batch_save
        , const std::string& continue_training);
void appendGenLog(const std::string& run_dir
        , int gen
        , float best_fitness
        , float pop_best_fitness
        , float pop_avg_fitness
        , float pop_sd_fitness
        , int pop_best_score);

// Thread pool for parallel evaluation
class ThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop = false;

public:
    explicit ThreadPool(size_t threads) {
        for (size_t i = 0; i < threads; ++i)
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
    }

    template<class F>
    auto enqueue(F&& f) -> std::future<decltype(f())> {
        using return_type = decltype(f());
        auto task = std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) worker.join();
    }
};

int main(int argc, char* argv[]) {
    std::string continue_training = "";
    int max_gen = 1'000'000;
    int batch_save = -1;
    float mutation_chance = 0.18f;
    float sigma = 0.05f;
    int pop_size = 1000;
    int evals_per_net = 5;

    std::string optimizer = "ars"; // "ars" (ARS V1-t antithetic updates) or "es" (tournament + mutate)
    int ars_dirs = 0;              // N directions; 0 => pop_size / 2 (matches ES eval budget)
    int ars_topb = 0;              // top-b directions; 0 => N / 2
    float ars_step = 0.02f;        // alpha: ARS update step size
    float ars_noise = 0.02f;       // nu: antithetic perturbation scale
    int threads_arg = 0;           // THREADS override for the pool (0 = auto 75% HW)

    int suggested_board_size = 35;


    // Read User Args ============== //
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        size_t eq = arg.find('=');
        if (eq == std::string::npos) continue;

        std::string key = arg.substr(0, eq);
        std::string val = arg.substr(eq + 1);

        if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
            val = val.substr(1, val.size() - 2);
        }

        if (key == "CONTINUE_TRAINING") continue_training = val;
        else if (key == "MAX_GEN") max_gen = std::stoi(val);
        else if (key == "BATCH_SAVE") batch_save = std::stoi(val);
        else if (key == "MUTATION_RATE") mutation_chance = std::stof(val);
        else if (key == "SIGMA") sigma = std::stof(val);
        else if (key == "POP_SIZE") pop_size = std::stoi(val);
        else if (key == "EVALS_PER_NET") evals_per_net = std::stoi(val);
        else if (key == "BOARD_SIZE") suggested_board_size = std::stoi(val);
        else if (key == "OPTIMIZER") optimizer = val;
        else if (key == "ARS_DIRECTIONS") ars_dirs = std::stoi(val);
        else if (key == "ARS_TOPB") ars_topb = std::stoi(val);
        else if (key == "ARS_STEP") ars_step = std::stof(val);
        else if (key == "ARS_NOISE") ars_noise = std::stof(val);
        else if (key == "THREADS") threads_arg = std::stoi(val);
        else throw std::runtime_error("Unknown Arg" +  std::string(" '") + key + std::string("' "));
    }
    // ============================= //

    // Path resolution for CONTINUE_TRAINING
    if (!continue_training.empty()) {
        if (!std::filesystem::path(continue_training).is_absolute()) {
            continue_training = "trained_networks/" + continue_training;
        }
        if (continue_training.size() < 11 || continue_training.substr(continue_training.size() - 11) != ".simple_net") {
            continue_training += ".simple_net";
        }
    }

    // Optimizer selection + ARS hyperparameter resolution
    for(char& c : optimizer) c = (char)std::tolower((unsigned char)c);
    const bool use_ars = (optimizer == "ars");
    if(!use_ars && optimizer != "es")
        throw std::runtime_error("OPTIMIZER must be 'ars' or 'es', got '" + optimizer + "'");
    int ars_n = ars_dirs;
    if(ars_n <= 0) ars_n = pop_size / 2;       // equal eval budget to ES: 2*N == POP
    if(ars_n <= 0) ars_n = 1;
    int ars_b = ars_topb;
    if(ars_b <= 0) ars_b = ars_n / 2;          // ARS V1-t: top half of directions
    if(ars_b <= 0) ars_b = 1;
    if(ars_b > ars_n) ars_b = ars_n;

    // Train Variables ============= //
    const int board_size = suggested_board_size;

    // Population ES settings
    const int num_elites = (pop_size * 0.01) < 1 ? 2 : (pop_size * 0.01);     // Top networks preserved unchanged
    const int tournament_k = num_elites / 2;    // Tournament selection size

    float best_fitness = -999999999.0f;
    SimpleNet bestNet;
    RunSnapshot best_snapshot;
    // ============================= //

    // Need to create now for getStateSize();
    SnakesGame game(board_size);

    // Network Variables =========== //
    const size_t input = game.getStateSize();
    const size_t output = 3;
    // ============================= //


    // Network Design (V2) ========= //
    // V2 input is the whole normalized board with wall ring ((B+2)^2 cells
    // in [-1, 1]), not hand-crafted rays, so the first layer is the spatial
    // combiner. TANH everywhere (monotonic, so argmax move selection is
    // unchanged). NOTE: weight count grows with board size (6x6 -> 68->56->24
    // ->3 ~= 5k weights; 20x20 -> 484 inputs ~= 28k) -- mutation-only ES
    // degrades on big nets, so start V2 trials at BOARD_SIZE=10-12.
	size_t hidden1 = 56;
	size_t hidden2 = 24;

	auto make_net = [&](void) -> SimpleNet {
        SimpleNet n;

        n.addLayer(std::make_unique<SimpleLayer>(input, hidden1));
        n.addLayer(std::make_unique<SimpleActivationLayer>(SimpleActivationLayer::TANH));

        n.addLayer(std::make_unique<SimpleLayer>(hidden1, hidden2));
        n.addLayer(std::make_unique<SimpleActivationLayer>(SimpleActivationLayer::TANH));

        n.addLayer(std::make_unique<SimpleLayer>(hidden2, output));
        n.addLayer(std::make_unique<SimpleActivationLayer>(SimpleActivationLayer::TANH));

        return n;
    };

    SimpleNet net;
    if(continue_training.empty()) {
        net = make_net();
    } else {
        try {
            net = SimpleNetReader::load(continue_training);

            const SimpleLayer* first = dynamic_cast<const SimpleLayer*>(net.layers[0].get());
            if (!first) throw std::runtime_error("First layer not SimpleLayer");
            if (first->weights.cols() != input)
                throw std::runtime_error("Input size mismatch: V2 needs (board_size+2)^2 = " + std::to_string(input) +
                                         ", got " + std::to_string(first->weights.cols()) +
                                         " (V1 24-ray nets cannot run on V2; train fresh)");

            const SimpleLayer* last = nullptr;
            for (auto it = net.layers.rbegin(); it != net.layers.rend(); ++it) {
                last = dynamic_cast<const SimpleLayer*>(it->get());
                if (last) break;
            }
            if (!last || last->weights.rows() != output)
                throw std::runtime_error("Output size mismatch: expected " + std::to_string(output));

        } catch(const std::exception& e) {
            throw std::runtime_error("Network Design: Invalid Model Load: " + std::string(e.what()));
        }
    }
    // ============================= //



// Train Loop ================== //
    struct Individual {
        SimpleNet net;
        float fitness = -99999.0f;
        int score = 0;
        SnakesGame best_game;   // final state of this net's best single run
    };

    // Optimizer state (only one branch is used; the other stays empty)
    std::vector<Individual> population; // ES mode
    std::vector<float> theta;           // ARS mode: mean parameter vector
    if(use_ars) {
        ARS::flattenInto(net, theta);
        std::cout << "OPTIMIZER=ars N=" << ars_n << " top_b=" << ars_b
                  << " step=" << ars_step << " noise=" << ars_noise
                  << " dim=" << theta.size() << "\n";
    } else {
        // Initialize population
        population.reserve(pop_size);

        // Seed with loaded net + mutations
        population.push_back({net, -99999.0f, 0});
        for (int i = 1; i < pop_size; ++i) {
            SimpleNet mutant = net;
            mutate(mutant, mutation_chance, sigma);
            population.push_back({std::move(mutant), -99999.0f, 0});
        }
    }

    // Pin the run version once per session
    const std::string run_dir = "trained_networks/run_" + std::to_string(findNextRunVersion());

    // Per-generation CSV log (header once, one row per gen). Never fatal.
    initGenLog(run_dir, argc, argv, board_size, pop_size, evals_per_net,
        mutation_chance, sigma, max_gen, batch_save, continue_training);

    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist_idx(0, pop_size - 1);

    // Thread pool for parallel evaluation (THREADS=0 keeps the 75% default;
    // set explicitly when sharing the box with another trainer).
    const int num_threads = std::thread::hardware_concurrency();
    const int pool_size = (threads_arg > 0) ? threads_arg
        : (num_threads > 0 ? int(num_threads * 0.75) : 4);
    ThreadPool pool(pool_size > 0 ? pool_size : 1);
    std::cout << "Thread Count: " << num_threads << "\nUsed: " << (pool_size > 0 ? pool_size : 1) << "\n";

    auto eval_individual = [&](Individual& ind) -> float {
        float total_fitness = 0.0f;
        int max_score = 0;
        float best_single = -1e30f;
        SnakesGame best_game(board_size);
        for (int e = 0; e < evals_per_net; ++e) {
            SnakesGame local_game(board_size);
            const float f = SnakeTrainer::run_net(local_game, ind.net, false, 0).fitness;
            total_fitness += f;
            if (f > best_single) { best_single = f; best_game = local_game; }
            max_score = std::max(max_score, (int)local_game.snake.size);
        }
        ind.fitness = total_fitness / evals_per_net;
        ind.score = max_score;
        ind.best_game = std::move(best_game);
        return ind.fitness;
    };

    auto eval_population = [&](std::vector<Individual>& pop) {
        std::vector<std::future<float>> futures;
        futures.reserve(pop.size());

        for (auto& ind : pop) {
            futures.push_back(pool.enqueue([&ind, &eval_individual]() -> float {
                return eval_individual(ind);
            }));
        }

        // Wait for all evaluations to complete
        for (auto& fut : futures) fut.get();
    };

    // Result of evaluating one parameter vector on fixed game seeds.
    // (Common random numbers: + / - pairs and the mean all play identical
    // games, so antithetic comparisons are apples-to-apples.)
    struct ARSEval {
        float fitness = -1e30f;
        int score = 0;
        SnakesGame game;
        ARSEval() : game(10) {}
    };

    // Evaluate params on evals_per_net games seeded from seed_base.
    // Builds a local net from the template shapes; thread-safe (all state
    // is local) so pool workers can call it concurrently.
    auto eval_params = [&](const std::vector<float>& params, uint32_t seed_base) -> ARSEval {
        ARSEval out;
        SimpleNet local = net; // template shapes
        ARS::unflattenInto(params, local);
        float total = 0.0f;
        float best_single = -1e30f;
        SnakesGame best_game(board_size);
        for(int e = 0; e < evals_per_net; ++e) {
            SnakesGame g(board_size, seed_base + (uint32_t)e);
            const float f = SnakeTrainer::run_net(g, local, false, 0).fitness;
            total += f;
            if(f > best_single) { best_single = f; best_game = g; }
            out.score = std::max(out.score, (int)g.snake.size);
        }
        out.fitness = total / (float)evals_per_net;
        out.game = best_game;
        return out;
    };

    // Sort by fitness descending (ES-only helper, defined at function scope
    // because the ES loop body below calls it).
    auto sort_pop = [&](void) {
        std::sort(population.begin(), population.end(),
            [](const Individual& a, const Individual& b) { return a.fitness > b.fitness; });
    };

    if(!use_ars) {
        // Initial evaluation
        eval_population(population);
        sort_pop();

        bestNet = population[0].net;
        best_fitness = population[0].fitness;
        best_snapshot = makeSnapshot(population[0].best_game, best_fitness, 0);
    }
    // ARS evaluates the mean theta inside the loop (gen 0 included), so the
    // global best is seeded on the first iteration.

    int gen_count = 0;

    while(max_gen == -1 || gen_count <= max_gen) {
        if(use_ars) {
            // ---- ARS generation: antithetic pairs around the mean ----
            std::vector<uint32_t> dir_seeds((size_t)ars_n);
            for(int k = 0; k < ars_n; ++k) dir_seeds[(size_t)k] = static_cast<uint32_t>(rng());
            const uint32_t game_base = static_cast<uint32_t>(rng());

            std::vector<std::future<ARSEval>> afuts;
            afuts.reserve((size_t)(2 * ars_n + 1));
            for(int k = 0; k < ars_n; ++k) {
                for(int s = 0; s < 2; ++s) {
                    const float sign = (s == 0) ? 1.0f : -1.0f;
                    afuts.push_back(pool.enqueue([&, k, sign]() -> ARSEval {
                        thread_local std::vector<float> tdir;
                        thread_local std::vector<float> tparams;
                        ARS::sampleDirection(dir_seeds[(size_t)k], theta.size(), tdir);
                        tparams = theta; // mean is frozen until all futures resolve
                        for(size_t i = 0; i < tparams.size(); ++i)
                            tparams[i] += sign * ars_noise * tdir[i];
                        return eval_params(tparams, game_base);
                    }));
                }
            }
            // Mean-theta eval on the same seeds (stats anchor + bestNet candidate).
            afuts.push_back(pool.enqueue([&]() -> ARSEval {
                return eval_params(theta, game_base);
            }));

            std::vector<float> r_plus((size_t)ars_n), r_minus((size_t)ars_n);
            std::vector<float> ars_all;
            ars_all.reserve((size_t)(2 * ars_n + 1));
            float ars_pop_best = -1e30f;
            int ars_pop_best_score = 0;
            auto absorb = [&](const ARSEval& e) {
                ars_all.push_back(e.fitness);
                if(e.fitness > ars_pop_best) { ars_pop_best = e.fitness; ars_pop_best_score = e.score; }
            };
            size_t fi = 0;
            for(int k = 0; k < ars_n; ++k) {
                const ARSEval ep = afuts[fi++].get();
                const ARSEval em = afuts[fi++].get();
                r_plus[(size_t)k] = ep.fitness;
                r_minus[(size_t)k] = em.fitness;
                absorb(ep);
                absorb(em);
            }
            const ARSEval t0 = afuts[fi++].get();
            absorb(t0);

            double ars_sum = 0.0;
            for(float r : ars_all) ars_sum += r;
            const float ars_avg = (float)(ars_sum / (double)ars_all.size());
            float ars_sd = 0.0f;
            for(float r : ars_all) { const float d = r - ars_avg; ars_sd += d * d; }
            ars_sd = std::sqrt(ars_sd / (float)ars_all.size());

            // Current-mean net + snapshot (pre-update theta).
            SimpleNet cur_net = net;
            ARS::unflattenInto(theta, cur_net);
            const RunSnapshot cur_snap = makeSnapshot(t0.game, t0.fitness, gen_count);

            if(t0.fitness > best_fitness) {
                bestNet = cur_net;
                best_fitness = t0.fitness;
                best_snapshot = cur_snap;

                if(batch_save == -1) {
                    std::cout << "Saving...\n";
                    save(run_dir, bestNet, cur_net, best_snapshot, cur_snap, board_size);
                    std::cout << "Saved: " << run_dir << "\n";
                }
            }

            // Display (same line format as ES so logs stay comparable).
            std::cout << "Gen " << gen_count
                      << " | Best: " << best_fitness
                      << " | Pop Best: " << ars_pop_best
                      << " | Pop Avg: " << ars_avg
                      << " | Best Score: " << best_snapshot.score
                      << " | Pop Best Score: " << ars_pop_best_score
                      << "\n";

            appendGenLog(run_dir, gen_count, best_fitness, ars_pop_best,
                ars_avg, ars_sd, ars_pop_best_score);

            if(batch_save != -1 && gen_count != 0 && gen_count % batch_save == 0) {
                std::cout << "Saving...\n";
                save(run_dir, bestNet, cur_net, best_snapshot, cur_snap, board_size);
                std::cout << "Saved: " << run_dir << "\n";
            }

            // ARS V1-t mean update (pairs were evaluated pre-update).
            ARS::update(theta, dir_seeds, r_plus, r_minus, ars_b, ars_step, 8.0f);

            gen_count++;
        } else {
        // Evaluate all in parallel (re-evaluate elites too for noise handling)
        eval_population(population);
        sort_pop();

        // Update global best
        if (population[0].fitness > best_fitness) {
            bestNet = population[0].net;
            best_fitness = population[0].fitness;
            best_snapshot = makeSnapshot(population[0].best_game, best_fitness, gen_count);

            if(batch_save == -1) {
                std::cout << "Saving...\n";
                save(run_dir
                    , bestNet
                    , population[0].net
                    , best_snapshot
                    , makeSnapshot(population[0].best_game, population[0].fitness, gen_count)
                    , board_size);
                std::cout << "Saved: " << run_dir << "\n";
            }
        }

        // Display
        std::cout << "Gen " << gen_count
                  << " | Best: " << best_fitness
                  << " | Pop Best: " << population[0].fitness
                  << " | Pop Avg: ";
        float avg_fit = 0; for (auto& p : population) avg_fit += p.fitness; avg_fit /= pop_size;
        float sd_fit = 0;
        for (auto& p : population) { const float d = p.fitness - avg_fit; sd_fit += d * d; }
        sd_fit = (pop_size > 0) ? std::sqrt(sd_fit / pop_size) : 0.0f;
        std::cout << avg_fit
                  << " | Best Score: " << best_snapshot.score
                  << " | Pop Best Score: " << population[0].score
                  << "\n";

        // Per-generation CSV log (append + flush each gen, never fatal).
        appendGenLog(run_dir, gen_count, best_fitness, population[0].fitness,
            avg_fit, sd_fit, population[0].score);

        // Save
        if(batch_save != -1 && gen_count != 0 && gen_count % batch_save == 0) {
            std::cout << "Saving...\n";
            save(run_dir
                , bestNet
                , population[0].net
                , best_snapshot
                , makeSnapshot(population[0].best_game, population[0].fitness, gen_count)
                , board_size);
            std::cout << "Saved: " << run_dir << "\n";
        }
        gen_count++;

        // Selection & Reproduction
        std::vector<Individual> next_gen;
        next_gen.reserve(pop_size);

        // Elitism: keep top num_elites unchanged
        for (int i = 0; i < num_elites; ++i) {
            next_gen.push_back(population[i]);
        }

        // Generate offspring via tournament selection + mutation
        while ((int)next_gen.size() < pop_size) {
            // Tournament selection
            int best_idx = dist_idx(rng);
            for (int t = 1; t < tournament_k; ++t) {
                int idx = dist_idx(rng);
                if (population[idx].fitness > population[best_idx].fitness) best_idx = idx;
            }

            SimpleNet child = population[best_idx].net;
            mutate(child, mutation_chance, sigma);
            next_gen.push_back({std::move(child), -99999.0f, 0});
        }

            population = std::move(next_gen);
        } // end ES branch
    }

    // ============================= //




    return 0;
}



static bool shouldMutate(std::mt19937& engine, const float chance) {
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(engine) < chance;
}

static float gauss(std::mt19937& engine, const float sigma) {
    static std::normal_distribution<float> dist(0.0f, 1.0f);
    return dist(engine) * sigma;
}

void mutate(SimpleNet& net, const float mutation_chance, const float sigma) {
    static thread_local std::mt19937 engine(std::random_device{}());

    for(auto& l : net.layers) {
        SimpleLayer* layer = dynamic_cast<SimpleLayer*>(l.get());
        if(!layer) continue; // activation layer

        for(float& w : layer->weights.data) {
            const float specific_mutation_chance = (w == 0.0f) ? mutation_chance * 1.75f : mutation_chance;
            if(shouldMutate(engine, specific_mutation_chance)) {
                w += gauss(engine, sigma);
                // Clamp weights to prevent explosion and activation saturation
                w = std::clamp(w, -8.0f, 8.0f);
            }
        }

        for(float& b : layer->biases) {
            const float specific_mutation_chance = (b == 0.0f) ? mutation_chance * 1.75f : mutation_chance;
            if(shouldMutate(engine, specific_mutation_chance)) {
                b += gauss(engine, sigma);
                // Clamp biases as well
                b = std::clamp(b, -8.0f, 8.0f);
            }
        }
    }
}


std::string renderScreenshot(const SnakesGame& game) {
    std::string out;
    const size_t n = game.board_size + 2;
    for(size_t row = 0; row < n; row++) {
        for(size_t col = 0; col < n; col++) {
            if(row == 0 || col == 0 || row == n - 1 || col == n - 1) {
                out += (row == 0 || row == n - 1) ? '-' : '|';
            }
            else if(col - 1 == (size_t)game.snake.head_x && row - 1 == (size_t)game.snake.head_y) {
                out += 'S';
            }
            else if(game.apple.x != -1
            && col - 1 == (size_t)game.apple.x
            && row - 1 == (size_t)game.apple.y) {
                out += 'A';
            }
            else if(game.board(row - 1, col - 1) == 1) {
                out += 's';
            }
            else {
                out += ' ';
            }
        }
        out += "\n";
    }
    return out;
}


int findNextRunVersion() {
    int max_ver = 0;
    if(!std::filesystem::exists("trained_networks")) return 1;

    for(const std::filesystem::directory_entry& entry
            : std::filesystem::directory_iterator("trained_networks")) {
        const std::string name = entry.path().filename().string();
        if(name.rfind("run_", 0) != 0) continue;
        try {
            const int ver = std::stoi(name.substr(4));
            if(ver > max_ver) max_ver = ver;
        } catch(...) {
            // Not a run_<N> folder, skip it
        }
    }
    return max_ver + 1;
}


RunSnapshot makeSnapshot(const SnakesGame& game, const float fitness, const size_t gen) {
    return RunSnapshot{fitness, gen, (int)game.snake.size, renderScreenshot(game)};
}


void save(const std::string& run_dir
        , const SimpleNet& best
        , const SimpleNet& current
        , const RunSnapshot& best_snap
        , const RunSnapshot& curr_snap
        , const size_t board_size) {
    std::filesystem::create_directories(run_dir);

    // Atomic: write to a temp file, then rename over the target.
    // A Ctrl+C mid-save can no longer leave a half-written .simple_net behind.
    // Retries against transient locks (antivirus/editor handles);
    // on persistent failure keep the tmp file and warn, never kill training.
    auto replaceFile = [&](const std::string& tmp, const std::string& target) {
        for(int attempt = 0; attempt < 10; attempt++) {
            std::error_code ec;
            std::filesystem::rename(tmp, target, ec);
            if(!ec) return;
            if(ec != std::errc::permission_denied) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        std::cerr << "save: WARNING: could not replace " << target
                  << " (kept " << tmp << "); next save will retry\n";
    };

    SimpleNetReader::save(run_dir + "/best.tmp", best);
    replaceFile(run_dir + "/best.tmp.simple_net", run_dir + "/best.simple_net");
    SimpleNetReader::save(run_dir + "/current.tmp", current);
    replaceFile(run_dir + "/current.tmp.simple_net", run_dir + "/current.simple_net");

    std::ofstream perf(run_dir + "/performance.txt");
    if(!perf.is_open())
        throw std::runtime_error("save: failed to open " + run_dir + "/performance.txt");

    perf << "BOARD_SIZE: " << board_size << "\n";
    perf << "Best;\n";
    perf << "    Fitness         : " << best_snap.fitness << "\n";
    perf << "    Gen             : " << best_snap.gen << "\n";
    perf << "    Score           : " << best_snap.score << "\n";
    perf << "Game Over Screenshot:\n";
    perf << best_snap.screenshot;
    perf << "Current;\n";
    perf << "    Fitness         : " << curr_snap.fitness << "\n";
    perf << "    Gen             : " << curr_snap.gen << "\n";
    perf << "    Score           : " << curr_snap.score << "\n";
    perf << "Game Over Screenshot:\n";
    perf << curr_snap.screenshot;

    if(!perf)
        throw std::runtime_error("save: failed writing " + run_dir + "/performance.txt");
}


void initGenLog(const std::string& run_dir
        , int argc
        , char* argv[]
        , int board_size
        , int pop_size
        , int evals_per_net
        , float mutation_chance
        , float sigma
        , int max_gen
        , int batch_save
        , const std::string& continue_training) {
    std::error_code ec;
    std::filesystem::create_directories(run_dir, ec);
    if(ec) {
        std::cerr << "initGenLog: WARNING: could not create " << run_dir
                  << " (" << ec.message() << "); gen logging disabled\n";
        return;
    }

    std::ofstream log(run_dir + "/log.txt", std::ios::trunc);
    if(!log.is_open()) {
        std::cerr << "initGenLog: WARNING: could not open " << run_dir
                  << "/log.txt; gen logging disabled\n";
        return;
    }

    log << "# run_dir: " << run_dir << "\n";
    log << "# cmd:";
    for(int i = 0; i < argc; ++i) log << (i == 0 ? " " : " ") << argv[i];
    log << "\n";
    log << "# BOARD_SIZE=" << board_size
        << " POP_SIZE=" << pop_size
        << " EVALS_PER_NET=" << evals_per_net
        << " MUTATION_RATE=" << mutation_chance
        << " SIGMA=" << sigma
        << " MAX_GEN=" << max_gen
        << " BATCH_SAVE=" << batch_save
        << " CONTINUE_TRAINING=" << (continue_training.empty() ? "(none)" : continue_training)
        << "\n";
    log << "gen,curr_best_fitness,pop_best_fitness,pop_avg_fitness,pop_sd_fitness,pop_best_score\n";
    log.flush();
    if(!log) {
        std::cerr << "initGenLog: WARNING: failed writing " << run_dir
                  << "/log.txt header; gen logging may be incomplete\n";
    }
}


void appendGenLog(const std::string& run_dir
        , int gen
        , float best_fitness
        , float pop_best_fitness
        , float pop_avg_fitness
        , float pop_sd_fitness
        , int pop_best_score) {
    std::ofstream log(run_dir + "/log.txt", std::ios::app);
    if(!log.is_open()) {
        std::cerr << "appendGenLog: WARNING: could not open " << run_dir
                  << "/log.txt for gen " << gen << "\n";
        return;
    }

    log << gen << ","
        << best_fitness << ","
        << pop_best_fitness << ","
        << pop_avg_fitness << ","
        << pop_sd_fitness << ","
        << pop_best_score << "\n";
    log.flush();
    if(!log) {
        std::cerr << "appendGenLog: WARNING: failed writing gen " << gen
                  << " to " << run_dir << "/log.txt\n";
    }
}
