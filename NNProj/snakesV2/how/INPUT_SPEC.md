# snakesV2 Input Spec (full-board, normalized)

V1 used 24 hand-crafted ray sensors. V2 feeds the whole board so the net can
see coils, the tail, and escape routes directly.

## Encoding

One feature per cell over the board **plus a padded wall ring**, row-major
over the `(B+2) x (B+2)` grid (inner cell `(x, y)` is at index
`(y+1)*(B+2)+(x+1)`), **plus 4 heading bits** `[up, right, down, left]`
(one-hot; up when the snake hasn't moved yet). Total `(board_size+2)^2+4`
inputs (68 on 6x6). The heading bits are load-bearing: moves are
heading-relative, so without them the same board under different headings
demands different moves from identical inputs. Raw cell values are divided
by `board_size^2`, so the state vector is in `[-1, 1]`:

| Cell  | Raw value         | Normalized      |
|-------|-------------------|-----------------|
| wall  | `-1` (ring)       | `-1 / B^2`      |
| apple | `-(B^2)`          | `-1.0`          |
| empty | `0`               | `0`             |
| head  | `1`               | `1 / B^2`       |
| body  | `B^2 - k`         | `(B^2-k) / B^2` |

Every value is distinct: wall (`-1/B^2`, tiny negative) vs empty (`0`) vs
head (`+1/B^2`) vs body (larger positives) vs apple (`-1.0`, unique large
negative). The ring is also constant every tick, so the net learns the
border as fixed-position features.

`k` = body index from the head (`k = 0` neck … `k = L-2` tail, `L` = length).

- Neck (most dangerous) is always the largest value (`~+1.0`).
- Tail is the smallest body value but never `0`: even with the board nearly
  full (`L = B^2 - 1`), tail = `3 / B^2`. Never confused with empty.
- Head is small but nonzero (`1 / B^2`), distinct from empty.
- Apple is the unique large negative (`-1.0`).

Magnitude rule of thumb: far from zero = care about.

## Consequences

- Nets are **board-size locked**: a net trained on 10x10 (148 inputs) only
  runs on 10x10. `getStateSize() == (board_size+2) * (board_size+2) + 4`.
- V1 nets (24 inputs) are rejected at load with an explicit error.
- Hidden layers are currently fixed at `56 -> 24` (TANH), so e.g. 6x6 is
  `68->56->24->3` (~5.4k weights) and 10x10 is `148->56->24->3` (~9.8k).
  Mutation-only ES struggles as weights grow, so start V2 trials at
  `BOARD_SIZE=10-12`.
- `visualize_layer` renders the input as a board grid on the TANH
  (`[-1, 1]`) color scale so the negative apple is visible.

## Optimizer (ARS vs ES)

`OPTIMIZER=ars` (default) runs Augmented Random Search V1-t: antithetic
pairs `theta +/- noise*d` evaluated on identical game seeds, top-b update
scaled by return std. `OPTIMIZER=es` keeps the old tournament + mutation
path for A/B comparison (same `log.txt` columns).

```
./train OPTIMIZER=ars EVALS_PER_NET=3 BOARD_SIZE=10 POP_SIZE=1000 ARS_DIRECTIONS=500 ARS_TOPB=250 ARS_STEP=0.02 ARS_NOISE=0.05 BATCH_SAVE=10
```

- `ARS_DIRECTIONS` (default `POP_SIZE/2`, so eval budget matches ES),
  `ARS_TOPB` (default half the directions), `ARS_STEP`, `ARS_NOISE`.
- NOTE: the policy is argmax-discrete, so tiny `ARS_NOISE` often changes
  no behavior at all (all pairs tie) — prefer `0.05-0.1` over `0.02`.
- `log.txt` format is identical in both modes; `ars_optimizer.h` holds the
  vector math, `ars_selftest.cpp` sphere-checks it.

## Suggested first run (from snakesV2/)

```
g++.exe -std=c++17 -DNDEBUG -O3 -pthread main.cpp -o train
./train EVALS_PER_NET=3 BOARD_SIZE=10 POP_SIZE=1000 SIGMA=0.02 MUTATION_RATE=0.1 BATCH_SAVE=10
```
