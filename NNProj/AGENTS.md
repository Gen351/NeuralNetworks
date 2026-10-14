# NNProj — header-only C++17 NN + game AIs, no build system

Each binary compiles one `.cpp` directly with `g++`. Keep `-DNDEBUG -O3`
(`-DNDEBUG` disables `Matrix` bounds asserts in hot loops).
Toolchain: `g++.exe -std=c++17 -DNDEBUG -O3 -pthread` (+ `-mavx2 [-mfma]` for SIMD chess builds).
Ignore root `.vscode/launch.json` (stale `snakes/build/Debug` path).

## Layout
- `nn_engine/` (`simple_net/`, `backprop_net/`, `matrix_op/`, `net_reader/`, `nnue/`): header-only, included via relative paths. Never built alone.
- `snakes/` (V1, 24 hand-crafted ray inputs) and `snakesV2/` (full-board `(BOARD_SIZE+2)^2` inputs, board-size locked): neuroevolution trainers. Nets are NOT interchangeable across versions/sizes.
- `chess/`: UCI engine + supervised trainer. Net contract `781->1` (see `chess/howto/createAI.md`); first Dense must take 781, last must emit 1, else load throws.
- Scratch/ignored per `.gitignore` — never commit: `dev/`, `trained_networks/`, `evolved_networks/`, `dataset/`, `*.exe`, `snakes/with_gui/`.

## Build (from the named subdir)
- snakes: `g++.exe -std=c++17 -DNDEBUG -O3 -pthread main.cpp -o train` (`test.cpp -o test`, `visualize_layer.cpp -o visualize_layer`)
- snakesV2: same; `./test <model> [speed_ms] [board_sz] <show?>` must use the training `BOARD_SIZE`
- chess: `uci_main.cpp -o chess_engine`, `train.cpp -o chess_trainer`, `test.cpp -o chess_test`, `self_play.cpp -o self_play` (all need `-pthread`)
- GUI (`snakes/with_gui/`, vendored ImGui+GLFW): `g++ main.cpp imgui/*.cpp ... -I. -Iimgui -L. -lglfw3 -lopengl32 -lgdi32 -o app.exe`

## Run (all trainers use `KEY=VALUE` args)
- snakes: `./train EVALS_PER_NET=3 BOARD_SIZE=10 POP_SIZE=1000 SIGMA=0.02 MUTATION_RATE=0.1 BATCH_SAVE=10`; resume with `CONTINUE_TRAINING="run_N/best"`
- chess: `./chess_trainer SAMPLES=50000 EPOCHS=40 BATCH=32`; resume `CONTINUE=<path>/net.backprop_net`; CSV finetune `DATASET=<csv>`; `./chess_trainer --help`, `--probe NET=<path>`
- Never cross `*.simple_net` (snakes) / `*.backprop_net` (chess); resume keys differ (`CONTINUE_TRAINING=` vs `CONTINUE=`).
- chess checks: `./chess_test` must print `ALL TESTS PASSED` before trusting `DATASET=` training; `./chess_engine --selftest`, `./chess_engine --perft 5`
- self_play: `NET1=`/`NET2=` (`PST` = built-in eval); if both `DEPTH=` and `MOVETIME=` given, the later one wins
