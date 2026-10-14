#pragma once

#include <vector>
#include <deque>
#include <algorithm>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>
#include <chrono>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#endif

#include "../nn_engine/matrix_op/matrix.hpp"

namespace SnakesColors {
    // 24-bit (true color) RGB values
    inline constexpr int BLACK_R = 0,   BLACK_G = 0,   BLACK_B = 0;
    inline constexpr int WHITE_R = 255, WHITE_G = 255, WHITE_B = 255;
    inline constexpr int HEAD_R = 0,    HEAD_G = 255,  HEAD_B = 0;
    inline constexpr int BODY_R = 0,    BODY_G = 128,  BODY_B = 0;
    inline constexpr int BODY_TAIL_R = 0, BODY_TAIL_G = 80, BODY_TAIL_B = 255;
    inline constexpr int APPLE_R = 255, APPLE_G = 0,   APPLE_B = 0;
    inline constexpr int IND_R = 60,      IND_G = 60,      IND_B = 60;
    inline constexpr int IND_DARK_R = 10, IND_DARK_G = 10, IND_DARK_B = 10;
}

inline void enableVt() {
    static bool enabled = false;
    if(enabled) return;
    enabled = true;
#ifdef _WIN32
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if(GetConsoleMode(out, &mode))
        SetConsoleMode(out, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
}

// Pure game environment. It knows nothing about neural networks, fitness,
// or training loops -- callers step() it and read getState() themselves.
// Move encoding: step() takes a relative move (1 = turn left, 2 = straight,
// 3 = turn right) relative to the snake's current heading. The heading is
// tracked internally (snake.direction) as an absolute direction:
// 1 = left, 2 = up, 3 = right, 4 = down.
struct SnakesGame {
    struct Pos {
        int x, y;
        bool operator==(const Pos& o) const { return x == o.x && y == o.y; }
    };

    // Death causes returned by step() / run() / deathCause():
    //   0 = still alive, 1 = wall, 2 = body, 3 = win (board full).
    // The trainer adds its own out-of-band causes (tick cap, starvation)
    // via terminate().
    struct StepResult {
        int cause = 0;
        bool ate = false;
    };

    struct Snake {
        int direction = 0;       // 1..4, 0 = not moving yet
        int prev_direction = 0;
        std::deque<Pos> body;    // front = head; single source of truth
        size_t size = 1;         // == body.size(), kept in sync
        int head_x = 0, head_y = 0;   // == body.front(), kept in sync
        int tail_x = 0, tail_y = 0;   // == body.back(), kept in sync

        Snake(int board_size) {
            head_x = rand() % (board_size - 1);
            head_y = rand() % (board_size - 1);
            tail_x = head_x;
            tail_y = head_y;
        }

        Snake() = default;
    };

    struct Apple {
        int x = -1, y = -1;
    };

    size_t game_ticks = 0;       // completed (non-terminal) moves only
    size_t max_game_tick;        // tick cap / state[4] normalizer (set by reset)
    size_t board_size;
    Matrix<int> board;           // 1 = body cell, always matches snake.body
    Snake snake;
    Apple apple;

    // Constructs a fresh game on a square board. seed lets you replay a run
    // deterministically.
    SnakesGame() : SnakesGame(10) {
        snake = Snake(10);
    }

    SnakesGame(size_t _board_size, unsigned seed = std::random_device{}())
        : board_size(_board_size)
        , board(_board_size, _board_size, 0)
        , rng(seed) {
        snake = Snake(board_size);
        reset();
    }

    // Fresh game: centered size-1 snake + a randomly placed apple.
    // Tick cap is 10x the trainer starvation base (1.5 * B^2): immortal
    // circlers die by starvation long before this; the old B^4 cap (160k
    // ticks on 20x20) only burned wall-clock on hopeless games.
    void reset() {
        game_ticks = 0;
        game_over = 0;
        max_game_tick = board_size * board_size * 15;
        snake = Snake(board_size);
        const int mid = (int)(board_size / 2);
        snake.body = { Pos{snake.head_x, snake.head_y} };
        syncGeometry();
        apple = Apple();
        spawnApple();
    }

    // Test helper: replace the snake body (front = head), repaint the board,
    // and re-sync the cached head/tail/size fields.
    void setBody(const std::deque<Pos>& new_body) {
        snake.body = new_body;
        game_over = 0;
        game_ticks = 0;
        syncGeometry();
    }

    bool isGameOver() const { return game_over != 0; }
    int deathCause() const { return game_over; }
    int score() const { return (int)snake.size; }

    // End the episode without a step (trainer-side causes: 4 = tick cap,
    // 5 = starvation). No-op once the game is already over.
    void terminate(int cause) {
        if(game_over == 0) game_over = cause;
    }

    // Execute one relative move: 1 = turn left, 2 = straight, 3 = turn right,
    // relative to the current heading. The heading is tracked internally as an
    // absolute direction (1 = left,  = up, 3 = right, 4 = down), so a
    // reversal is impossible by construction. Throws on an invalid move so a
    // bad caller fails loudly instead of silently freezing the snake.
    // Returns cause 0 (alive), 1 (wall), 2 (body), 3 (win) + whether it ate.
    StepResult step(int move) {
        if(move < 1 || move > 3)
            throw std::runtime_error("SnakesGame::step: invalid move " + std::to_string(move));
        if(game_over != 0) return {game_over, false};

        // Resolve the relative input against the current heading. Before the
        // snake has moved (direction 0) it is treated as facing up, matching
        // getState().
        const int base = (snake.direction == 0) ? 2 : snake.direction;
        const int dir = (move == 1) ? (base == 1 ? 4 : base - 1)
                      : (move == 3) ? (base == 4 ? 1 : base + 1)
                      : base;

        const int dx[5] = {0, -1, 0, 1, 0};
        const int dy[5] = {0, 0, -1, 0, 1};
        const int nx = snake.head_x + dx[dir];
        const int ny = snake.head_y + dy[dir];

        // Wall
        if(nx < 0 || nx >= (int)board_size || ny < 0 || ny >= (int)board_size) {
            game_over = 1;
            return {1, false};
        }

        const bool eating = (apple.x == nx && apple.y == ny);

        // Body collision. The tail cell vacates this tick, so the head may
        // enter it -- unless the snake is eating (then the tail stays).
        // This one uniform rule handles tail-chase (size 2) and size-1
        // reversal with no special cases.
        for(size_t i = 0; i < snake.body.size(); ++i) {
            if(snake.body[i].x == nx && snake.body[i].y == ny) {
                const bool is_tail = (i == snake.body.size() - 1);
                if(eating || !is_tail) {
                    game_over = 2;
                    return {2, false};
                }
                break;
            }
        }

        // Commit the move.
        snake.prev_direction = snake.direction;
        snake.direction = dir;
        snake.body.push_front(Pos{nx, ny});
        if(!eating) snake.body.pop_back();
        syncGeometry();

        game_ticks++;
        if(eating) {
            if(!spawnApple()) { game_over = 3; return {3, true}; }
            return {0, true};
        }
        return {0, false};
    }

    // Places the apple on a random empty cell, never inside the body.
    // Apples 1-2 cells ahead of the head are very unlikely, so the snake
    // isn't handed free apples. Returns false when the board is full (win).
    bool spawnApple() {
        const float ahead_spawn_chance = 0.005f; // 0.5%: apple may appear in the snake's path

        // Cells the head is facing (within the board, currently empty)
        std::vector<size_t> ahead;
        if(snake.direction != 0) {
            const int dx[5] = {0, -1, 0, 1, 0};
            const int dy[5] = {0, 0, -1, 0, 1};
            for(int steps = 1; steps <= 2; steps++) {
                const int ax = snake.head_x + dx[snake.direction] * steps;
                const int ay = snake.head_y + dy[snake.direction] * steps;
                if(ax < 0 || ax >= (int)board_size || ay < 0 || ay >= (int)board_size) continue;
                const size_t idx = (size_t)ay * board_size + (size_t)ax;
                if(board.data[idx] == 0) ahead.push_back(idx);
            }
        }

        std::vector<size_t> empty;
        for(size_t i = 0; i < board.data.size(); i++) {
            if(board.data[i] != 0) continue;
            if(std::find(ahead.begin(), ahead.end(), i) != ahead.end()) continue;
            empty.push_back(i);
        }
        if(empty.empty() && ahead.empty()) { apple = Apple(); return false; }

        std::uniform_real_distribution<float> coin(0.0f, 1.0f);
        if(!ahead.empty() && coin(rng) < ahead_spawn_chance) {
            std::uniform_int_distribution<size_t> dist(0, ahead.size() - 1);
            const size_t cell = ahead[dist(rng)];
            apple.x = (int)(cell % board_size);
            apple.y = (int)(cell / board_size);
            return true;
        }

        std::vector<size_t>& pool = empty.empty() ? ahead : empty;
        std::uniform_int_distribution<size_t> dist(0, pool.size() - 1);
        const size_t cell = pool[dist(rng)];
        apple.x = (int)(cell % board_size);
        apple.y = (int)(cell / board_size);
        return true;
    }

    float getDistance(int h_x, int h_y, int a_x, int a_y) const {
        const int x_delta = (h_x - a_x);
        const int y_delta = (h_y - a_y);
        return std::sqrt((float)(x_delta * x_delta) + (float)(y_delta * y_delta));
    }

    // Free cells from the head to the nearest wall along (dx, dy).
    float wallDistance(int dx, int dy) const {
        float d = 0.0f;
        while(true) {
            const int x = snake.head_x + dx * ((int)d + 1);
            const int y = snake.head_y + dy * ((int)d + 1);
            if(x < 0 || x >= (int)board_size || y < 0 || y >= (int)board_size) return d;
            d += 1.0f;
        }
    }

    // V2 full-board input with a padded wall ring. (B+2)^2 features, row-major
    // over the padded grid; inner cell (x, y) sits at (y+1) * (B+2) + (x+1).
    // Raw values normalized by board_size^2 into [-1, 1]:
    //   wall    = -1 / B^2           (border ring raw -1, constant every tick;
    //                                 tiny negative, distinct from empty 0)
    //   apple   = -1.0               (raw -B^2 / B^2, the unique large negative)
    //   + 4 heading bits [up, right, down, left] (one-hot, facing-up when the
    //     snake hasn't moved yet, mirroring step()). WITHOUT these the input
    //     is heading-blind while labels/moves are heading-relative, so the
    //     same board with different headings demands different moves -- an
    //     unlearnable contradiction (behavioral cloning stalls near chance).
    //   empty   = 0
    //   head    = 1 / B^2            (small but nonzero, never confused with empty)
    //   body[k] = (B^2 - k) / B^2    (k = 0 neck ... k = L-2 tail; neck ~= 1.0,
    //                                 tail >= 2/B^2 even when the board is full)
    // Unlike V1 rays, the tail is visibly the smallest body value and its motion
    // is observable across ticks, so coiled states are distinguishable; and the
    // constant -1 ring tells the net exactly where the walls are.
    std::vector<float> getState() const {
        const size_t stride = board_size + 2;
        const float norm = (float)(board_size * board_size);
        std::vector<float> state(stride * stride, -1.0f / norm); // wall ring (raw -1)

        for(size_t y = 0; y < board_size; ++y)
            for(size_t x = 0; x < board_size; ++x)
                state[(y + 1) * stride + (x + 1)] = 0.0f; // inner empty

        for(size_t i = 0; i < snake.body.size(); ++i) {
            const int x = snake.body[i].x, y = snake.body[i].y;
            if(x < 0 || x >= (int)board_size || y < 0 || y >= (int)board_size) continue;
            float raw;
            if(i == 0) raw = 1.0f;                    // head
            else raw = norm - (float)(i - 1);         // neck = B^2, ..., tail >= 2
            state[((size_t)y + 1) * stride + ((size_t)x + 1)] = raw / norm;
        }

        if(apple.x >= 0 && apple.y >= 0
        && apple.x < (int)board_size && apple.y < (int)board_size) {
            state[((size_t)apple.y + 1) * stride + ((size_t)apple.x + 1)] = -1.0f;
        }

        // Heading one-hot [up, right, down, left] = absolute dirs [2,3,4,1].
        const int base = (snake.direction == 0) ? 2 : snake.direction;
        state.push_back(base == 2 ? 1.0f : 0.0f);
        state.push_back(base == 3 ? 1.0f : 0.0f);
        state.push_back(base == 4 ? 1.0f : 0.0f);
        state.push_back(base == 1 ? 1.0f : 0.0f);

        return state;
    }

    // V2 nets are board-size locked: a net trained on B x B only runs on B x B
    // (input is (B+2)^2 board cells + 4 heading bits). V1 24-ray nets are
    // rejected by the input-size check at load time.
    size_t getStateSize() const { return (board_size + 2) * (board_size + 2) + 4; }

    // Manual/testing play: keep moving straight in the current direction
    // until death.
    int run(bool display = false, int frame_delay_ms = 0) {
        if(display) display_game(true);
        while(game_over == 0) {
            if(snake.direction == 0) break;
            const StepResult r = step(2);
            if(display) {
                display_game(false); // overwrite the previous frame, no flicker
                std::this_thread::sleep_for(std::chrono::milliseconds(frame_delay_ms));
            }
            if(r.cause != 0) return r.cause;
        }
        return game_over;
    }

    // Screenshot only: renders the current game state once.
    void display_game(bool clear_scr=true) {
        enableVt();

        std::ostringstream frame;
        if(clear_scr) {frame << "\x1b[H\x1b[2J";}   // first frame: clear screen
        else {frame << "\x1b[H";}                   // later frames: overwrite in place

        frame << "\n\n\n\n";

        // Body gradient: per-cell index along the snake (0 = head, size-1 = tail).
        std::vector<int> body_index(board_size * board_size, -1);
        for(size_t i = 0; i < snake.body.size(); ++i)
            body_index[(size_t)snake.body[i].y * board_size + (size_t)snake.body[i].x] = (int)i;

        // Direction ray from the head to the wall (or until obstructed).
        // Stores the distance (in cells) from the head for each lit cell.
        std::vector<int> dir_dist(board_size * board_size, -1);
        if(snake.direction != 0) {
            const int dx[5] = {0, -1, 0, 1, 0};
            const int dy[5] = {0, 0, -1, 0, 1};
            int s = 1;
            while(true) {
                const int x = snake.head_x + dx[snake.direction] * s;
                const int y = snake.head_y + dy[snake.direction] * s;
                if(x < 0 || x >= (int)board_size || y < 0 || y >= (int)board_size) break;
                if(apple.x == x && apple.y == y) break;      // obstructed
                if(board(y, x) == 1) break;                  // obstructed
                dir_dist[(size_t)y * board_size + (size_t)x] = s;
                s++;
            }
        }

        const size_t n = board_size + 2;
        for(size_t row = 0; row < n; row++) {
            for(size_t col = 0; col < n; col++) {
                int r = SnakesColors::BLACK_R;
                int g = SnakesColors::BLACK_G;
                int b = SnakesColors::BLACK_B;

                if(row == 0 || col == 0 || row == n - 1 || col == n - 1) {
                    r = SnakesColors::WHITE_R;
                    g = SnakesColors::WHITE_G;
                    b = SnakesColors::WHITE_B;
                }
                else {
                    size_t br = row - 1;
                    size_t bc = col - 1;
                    if(bc == (size_t)snake.head_x && br == (size_t)snake.head_y) {
                        r = SnakesColors::HEAD_R;
                        g = SnakesColors::HEAD_G;
                        b = SnakesColors::HEAD_B;
                    }
                    else if(apple.x != -1 && bc == (size_t)apple.x && br == (size_t)apple.y) {
                        r = SnakesColors::APPLE_R;
                        g = SnakesColors::APPLE_G;
                        b = SnakesColors::APPLE_B;
                    }
                    else if(board(br, bc) == 1) {
                        const int i = body_index[br * board_size + bc];
                        const float t = (snake.size > 1) ? (float)i / (float)(snake.size - 1) : 0.0f;
                        r = SnakesColors::BODY_R;
                        g = (int)(SnakesColors::BODY_TAIL_G + (255 - SnakesColors::BODY_TAIL_G) * (1.0f - t));
                        b = (int)(255.0f * t);
                    }
                    else if(dir_dist[br * board_size + bc] >= 0) {
                        const float t = (float)dir_dist[br * board_size + bc] / (float)board_size;
                        const int v = (int)(SnakesColors::IND_R + (SnakesColors::IND_DARK_R - SnakesColors::IND_R) * t);
                        r = v; g = v; b = v;
                    }
                }

                frame << "\x1b[48;2;" << r << ";" << g << ";" << b << "m  \x1b[0m";
            }
            frame << "\n";
        }
        std::cout << frame.str() << "\n\n> Score: " << snake.size - 1 << std::flush;
    }

private:
    int game_over = 0;
    std::mt19937 rng;

    // Rebuild the board from snake.body and refresh the cached head/tail/size.
    void syncGeometry() {
        if(snake.body.empty()) return;
        snake.head_x = snake.body.front().x;
        snake.head_y = snake.body.front().y;
        snake.tail_x = snake.body.back().x;
        snake.tail_y = snake.body.back().y;
        snake.size = snake.body.size();
        board = Matrix<int>(board_size, board_size, 0);
        for(const auto& p : snake.body) board(p.y, p.x) = 1;
    }
};
