# 03 — DeepSeek Optimizations: What Changes vs the Baseline (KV cache & compute)

**Part 3 of 3.** This is the actual **architecture** of DeepSeek-V3 / DeepSeek-R1. It takes
the baseline decoder from part 01 and replaces the two expensive parts:

1. **Attention** → **MLA (Multi-head Latent Attention)** — the KV-cache change.
2. **FFN** → **DeepSeekMoE** — sparse experts.
3. Plus **aux-loss-free load balancing**, **Multi-Token Prediction**, and **FP8**.

Read 01 → 02 → 03. This is the end of the ladder.

---

## 3.1 The changes at a glance

| Baseline (01) | DeepSeek (03) | Wins |
|---------------|---------------|------|
| Multi-Head Attention, full K/V cache | **MLA** (low-rank latent + decoupled RoPE) | KV cache ↓ **>90%** |
| Dense FFN (4×H) | **DeepSeekMoE** (256 routed + 1 shared, top-8) | same FLOPs, far more capacity |
| Aux loss for load balance | **Aux-loss-free bias balancing** | no quality penalty |
| Next-token only | **Multi-Token Prediction (MTP)** | richer signal + speculative decode |
| BF16 | **FP8** (fine-grained, blockwise) | ~2× memory/throughput |

**Full DeepSeek-V3/R1 config (verified from the official `config.json`):**
```
vocab_size 129280        hidden_size 7168        num_hidden_layers 61
num_attention_heads 128  num_key_value_heads 128
q_lora_rank 1536         kv_lora_rank 512
qk_nope_head_dim 128     qk_rope_head_dim 64    v_head_dim 128
n_routed_experts 256     n_shared_experts 1     num_experts_per_tok 8
moe_intermediate_size 2048   intermediate_size 18432 (dense layers)
first_k_dense_replace 3  routed_scaling_factor 2.5
topk_method "noaux_tc"   n_group 8   topk_group 4   scoring_func "sigmoid"
rope_theta 10000         rope_scaling yarn factor 40, original_max_pos 4096
rms_norm_eps 1e-6        tie_word_embeddings false
fp8: e4m3, weight_block_size [128,128]    num_nextn_predict_layers 1
```
→ **671B total params, 37B activated per token.**

---

## 3.2 MLA — the KV-cache optimization (the big one)

### The problem
Baseline MHA caches, per token per layer: `2 * n_h * head_dim = 2*128*128 = 32768` values.
For long contexts this dominates memory. MLA stores a **compressed latent** instead and
**reconstructs** K/V on the fly.

### The idea (equations)
Let `h_t` be the input, `d_c` the latent dim:
```
c_KV = W_DKV · h_t                         # compress to latent  (d_c = kv_lora_rank = 512)
[kC_1..kC_nh] = W_UK · c_KV                 # reconstruct "content" keys (up-projection)
vC            = W_UV · c_KV                 # reconstruct values
kR            = RoPE(W_KR · h_t)            # decoupled key carrying position (d_h^R = 64)
k_i           = [ kC_i ; kR ]               # concat content + rope per head
```
For queries, low-rank too:
```
c_Q = W_DQ · h_t                            # q_lora_rank = 1536
[qC; qR] = [ W_UQ · c_Q ; RoPE(W_QR · c_Q) ]
```

### Decoupled RoPE (why it's needed)
RoPE is applied to `c_KV`… but you **cannot** rotate a compressed latent, because the same
latent feeds both K and V — rotating it would corrupt V. So MLA carries a **separate small
RoPE dim** (`kR`, 64) that is *not* compressed, while the content part stays rotation-free.
This is the "decoupled" trick.

### What gets cached (inference)
Only the **blue-boxed** vectors:
```
cache per token per layer = c_KV (512) + kR (64) = 576 values
```
vs full MHA `32768`. That's **~57× smaller** (well over 90% reduction) at V3 dimensions.
(The V2 paper reports a 93.3% KV-cache reduction vs its baseline.)

### Absorption trick (optional speedup)
At inference you can fold `W_UK` and `W_UV` into the query/output projections so you never
materialize K/V — you attend directly in the latent space. Saves compute, costs a
matmul-reassociation. Implement only after the naive version works.

### Implementation checklist (MLA)
```
q_a_proj -> q_a_layernorm -> q_b_proj -> Q      (then split nope/rope)
kv_a_proj -> kv_a_layernorm -> c_KV, k_rope
k_rope -> RoPE
kv_b_proj(c_KV) -> K_nope, V
scores = Q_nope·K_nope + Q_rope·k_rope  (scale by 1/sqrt(qk_nope+qk_rope))
causal softmax -> out = A @ V -> o_proj
```

---

## 3.3 DeepSeekMoE — sparse experts

Replaces the dense FFN with many small experts, activating only a few per token.

- **Router:** `scores = sigmoid(x @ W_router)` over `n_routed_experts = 256` (V3 uses
  `scoring_func="sigmoid"`, not softmax, for routing; softmax only for probabilities).
- **Group routing** (`topk_method="noaux_tc"`): split the 256 experts into `n_group=8`
  groups, pick the top `topk_group=4` groups, then take the top `num_experts_per_tok=8`
  experts within those. Grouping stabilizes routing.
- **Shared expert:** `n_shared_experts=1` always-on SwiGLU expert added to every token
  (captures common knowledge, reduces redundancy across routed experts).
- **Combine:** `out = Σ_{i in top8} g_i * Expert_i(x) + SharedExpert(x)`, scaled by
  `routed_scaling_factor=2.5`.
- **Each expert** = SwiGLU: `(silu(x@W_gate) * (x@W_up)) @ W_down`, `moe_inter=2048`.
- **First 3 layers are dense** (`first_k_dense_replace=3`); the rest are MoE.

### Aux-loss-free load balancing (change vs classic MoE)
Classic MoE adds an auxiliary loss to force equal expert usage — this **hurts** model
quality. V3 instead keeps a per-expert **bias** `b_e` used *only for routing selection*:
```
g = sigmoid(scores) + b_e          # b_e is added for top-k selection only
g = g / sum(g)                     # probabilities are unaffected by b_e
```
After each training step: experts that are **overloaded** get `b_e -= γ`; underloaded get
`b_e += γ`. No gradient flows through `b_e`. This balances load without a loss term.

---

## 3.4 Multi-Token Prediction (MTP)

Besides predicting token `t+1`, add `num_nextn_predict_layers=1` head that predicts `t+2`
(as a training objective and for speculative decoding). It gives a denser training signal
and lets inference propose multiple tokens at once. Optional but part of V3.

---

## 3.5 FP8

V3 is trained in **FP8 (e4m3)** with **fine-grained blockwise scaling** (`weight_block_size
[128,128]`): each 128×128 weight block has its own scale, so precision loss stays small.
Activations use dynamic per-token scaling. Result: ~half the memory and much higher
throughput vs BF16, with negligible quality loss. For your C++ rebuild, start in BF16 and
add FP8 as a later optimization once correctness is proven.

---

## 3.6 Build order (do these one at a time, measuring after each)

Build on top of the working part-01 model. For each step log: **params, peak memory,
FLOPs/token, tokens/sec, loss.**

```
01 baseline GPT ................................ done first
 └─ 02 (training/RL) ........................... can reuse baseline model
03 optimizations:
 ├─ 1. RMSNorm + RoPE + SwiGLU .................. swap-in modern defaults
 ├─ 2. GQA (num_kv_heads < num_heads) ........... warm-up for MLA cache idea
 ├─ 3. MLA (latent + decoupled RoPE) ............ measure KV-cache bytes/token
 ├─ 4. DeepSeekMoE (router + top-k + shared) ..... measure active params / FLOPs
 ├─ 5. Aux-loss-free bias balancing ............. measure expert load histogram
 ├─ 6. MTP head .................................. measure loss + decode speed
 └─ 7. FP8 blockwise quant ...................... measure memory + throughput
```

Rule of the game: **every optimization swaps one resource for another.** Measure both sides.
That is how you'll eventually invent your own.

---

## 3.7 C++ / g++ notes for 03

- MLA needs `reshape/transpose` (heads and the nope/rope split). Keep them as views when
  possible.
- MoE needs **token dispatch**: group tokens per selected expert so each expert runs one
  batched matmul (or you waste most FLOPs on padding).
- Weight loading: DeepSeek ships **FP8 safetensors** — you need dequant (e4m3 + block scale)
  → BF16 before compute. `llama.cpp`/GGUF is a good reference format.
- Cache clarity: baseline MHA cache is `[layer][token][2][n_h][head_dim]`; MLA cache is
  `[layer][token][kv_lora=512 + rope=64]` — a completely different shape. Make the cache a
  virtual interface so you can A/B test them.
- Build exactly as in 01/02 (plain `g++`, `-O2 -march=native`, `-flto` when splitting TUs).

---

## 3.8 End of the ladder

You now have the full picture:

- **01** = the plain decoder everything builds on.
- **02** = the training/token change that creates "thinking".
- **03** = the architectural optimizations (MLA, MoE, MTP, FP8) that make it cheap.

Next moves: get 01 compiling and correct against a PyTorch reference, then add features from
03 one at a time — and keep the benchmark harness from 3.6 open, because that is where your
own optimization ideas will come from.
