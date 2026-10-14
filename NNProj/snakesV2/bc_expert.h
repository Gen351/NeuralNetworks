#pragma once

// Behavioral-cloning teacher for snakesV2. Three tiers, cheapest first:
//   1. Sightline: head shares a row/column with the apple and nothing is
//      between -> step straight toward it (no search).
//   2. Blocked -> BFS shortest safe path to the apple, treating the tail
//      cell as free (it vacates next tick -- same rule as step()). If the
//      first step eats the apple the tail does NOT vacate, so that case is
//      re-searched with the tail blocked.
//   3. No safe path -> greedy: among the 3 expressible relative moves,
//      pick a non-suicidal one minimizing Manhattan distance to the apple.
// Returns a RELATIVE move (1 = left, 2 = straight, 3 = right), matching
// SnakesGame::step() and the 3-output nets. Absolute reversal is never
// returned: BFS may not start into the reverse cell (unexpressible), and
// greedy only evaluates the 3 relative options.

#include <queue>
#include <vector>
#include <cstdlib>

#include "./snake_environment.h"

namespace BCExpert {

// Absolute dirs: 1 = left(-1,0), 2 = up(0,-1), 3 = right(1,0), 4 = down(0,1).
inline int baseHeading(const SnakesGame& g) {
    return (g.snake.direction == 0) ? 2 : g.snake.direction;
}

inline int reverseOf(int d) {
    return (d == 1) ? 3 : (d == 2) ? 4 : (d == 3) ? 1 : 2;
}

inline int leftOf(int d) {
    return (d == 1) ? 4 : d - 1;
}

// Absolute -> relative under base heading (d must not be reverseOf(base)).
inline int toRelative(int base, int d) {
    if(d == base) return 2;
    if(d == leftOf(base)) return 1;
    return 3;
}

inline void absStep(int d, int& dx, int& dy) {
    dx = (d == 1) ? -1 : (d == 3) ? 1 : 0;
    dy = (d == 2) ? -1 : (d == 4) ? 1 : 0;
}

inline bool inside(const SnakesGame& g, int x, int y) {
    return x >= 0 && x < (int)g.board_size && y >= 0 && y < (int)g.board_size;
}

// True if the cell kills on entry (tail treated as free when tail_free).
inline bool lethal(const SnakesGame& g, int x, int y, bool tail_free) {
    if(!inside(g, x, y)) return true;
    if(g.board((size_t)y, (size_t)x) != 1) return false;
    if(tail_free && x == g.snake.tail_x && y == g.snake.tail_y) return false;
    return true;
}

// Tier 1: clear straight-line run at the apple? Returns absolute dir or 0.
inline int sightlineDir(const SnakesGame& g) {
    const int hx = g.snake.head_x, hy = g.snake.head_y;
    const int base = baseHeading(g);
    for(int d = 1; d <= 4; ++d) {
        int dx, dy;
        absStep(d, dx, dy);
        int x = hx + dx, y = hy + dy;
        bool blocked = false, found = false;
        while(inside(g, x, y)) {
            if(g.board((size_t)y, (size_t)x) == 1) { blocked = true; break; }
            if(x == g.apple.x && y == g.apple.y) { found = true; break; }
            // Sightline only counts along the ray: stop at first apple check
            // per cell; cells past the apple are irrelevant.
            x += dx;
            y += dy;
        }
        if(found && !blocked) {
            // Only useful if expressible: must not be the reverse cell, and
            // the head must be able to turn toward it in one relative move.
            // (toRelative maps anything non-reverse; reverse handled by BFS.)
            if(d == reverseOf(base)) continue;
            return d;
        }
    }
    return 0;
}

// Tier 2: BFS first-step absolute dir toward the apple (0 = none). The
// opening step may not be the reverse cell (unexpressible relatively).
inline int bfsDir(const SnakesGame& g, bool tail_free) {
    const size_t n = g.board_size;
    const int hx = g.snake.head_x, hy = g.snake.head_y;
    const int base = baseHeading(g);
    const int rev = reverseOf(base);

    std::vector<int> parent(n * n, -2); // -2 unvisited, else parent index
    std::queue<int> q;
    const int start = hy * (int)n + hx;
    parent[(size_t)start] = -1;
    q.push(start);

    static const int dxs[4] = {-1, 0, 1, 0};
    static const int dys[4] = {0, -1, 0, 1};
    static const int dirs[4] = {1, 2, 3, 4};

    int found = -1;
    while(!q.empty()) {
        const int cur = q.front();
        q.pop();
        const int cx = cur % (int)n, cy = cur / (int)n;
        if(cx == g.apple.x && cy == g.apple.y) { found = cur; break; }
        for(int k = 0; k < 4; ++k) {
            const int nx = cx + dxs[k], ny = cy + dys[k];
            if(!inside(g, nx, ny)) continue;
            if(cur == start && dirs[k] == rev) continue; // unexpressible
            if(lethal(g, nx, ny, tail_free)) continue;
            const int ni = ny * (int)n + nx;
            if(parent[(size_t)ni] != -2) continue;
            parent[(size_t)ni] = cur;
            q.push(ni);
        }
    }
    if(found < 0) return 0;

    // Walk back to the first step after start.
    int cur = found, prev = -1;
    while(cur != start) { prev = cur; cur = parent[(size_t)cur]; }
    const int fx = prev % (int)n, fy = prev / (int)n;
    const int sdx = fx - hx, sdy = fy - hy;
    if(sdx == -1) return 1;
    if(sdx == 1) return 3;
    if(sdy == -1) return 2;
    return 4;
}

// Tier 3: greedy-safe relative move (1..3). Never suicidal if any safe
// move exists; otherwise returns straight (doomed either way).
inline int greedyMove(const SnakesGame& g) {
    const int base = baseHeading(g);
    const int rels[3] = {2, 1, 3}; // prefer straight on ties
    int best_rel = 2, best_dist = 1 << 30;
    for(int r = 0; r < 3; ++r) {
        const int d = (rels[r] == 2) ? base
            : (rels[r] == 1) ? leftOf(base)
            : (base == 4) ? 1 : base + 1; // rightOf(base)
        int dx, dy;
        absStep(d, dx, dy);
        const int nx = g.snake.head_x + dx, ny = g.snake.head_y + dy;
        const bool eating = (nx == g.apple.x && ny == g.apple.y);
        if(lethal(g, nx, ny, !eating)) continue; // tail vacates unless eating
        const int dist = std::abs(nx - g.apple.x) + std::abs(ny - g.apple.y);
        if(dist < best_dist) { best_dist = dist; best_rel = rels[r]; }
    }
    return best_rel;
}

// Full teacher: relative move 1/2/3 for the current game state.
inline int suggestMove(const SnakesGame& g) {
    const int base = baseHeading(g);

    // Tier 1: sightline.
    const int sight = sightlineDir(g);
    if(sight != 0) return toRelative(base, sight);

    // Tier 2: BFS with tail free; re-search tail-blocked if it eats.
    int d = bfsDir(g, true);
    if(d != 0) {
        int dx, dy;
        absStep(d, dx, dy);
        const int nx = g.snake.head_x + dx, ny = g.snake.head_y + dy;
        if(nx == g.apple.x && ny == g.apple.y) {
            const int d2 = bfsDir(g, false); // eating: tail stays
            if(d2 != 0) return toRelative(base, d2);
            return greedyMove(g);
        }
        return toRelative(base, d);
    }

    // Tier 3: greedy-safe.
    return greedyMove(g);
}

} // namespace BCExpert
