# 01 — The Baseline: How to Make a ChatGPT-style LLM (no "thinking")

This is **part 1 of 3**. It builds the simplest possible decoder-only transformer — the
architecture behind GPT-2 / GPT-3 / early ChatGPT. **Everything in parts 02 and 03 is a
modification of this.** Build and understand this first.

- **01 (this file):** baseline GPT. Dense attention + dense FFN. **No reasoning trace.**
- **02:** what changes for DeepSeek-R1's "thinking" (training + reasoning tokens).
- **03:** DeepSeek's architectural optimizations (MLA / KV-cache, MoE, MTP, FP8).

> Build/compile rules for this project are the same everywhere: plain `g++`, no CMake.
> Choose your own file layout — it does **not** affect runtime speed. The only thing that
> affects speed is *translation-unit boundaries* (use `inline` in headers, or `-flto`).

---

## 1.1 What "ChatGPT with no thinking" means

It is a **causal language model**: given tokens `t0..tn`, it predicts the probability of the
next token. A response is generated one token at a time, directly, with **no separate
reasoning section**. (ChatGPT's chat *behavior* was added by RLHF later, but the
architecture is exactly this.)

```
prompt ─► [ model ] ─► P(next token) ─► sample ─► append ─► repeat
No <think> block. No chain-of-thought. Answer starts immediately.
```

---

## 1.2 The parts (top to bottom)

| # | Part | Purpose | Connects to |
|---|------|---------|-------------|
| 1 | **Tokenizer** | text ↔ token IDs (BPE, e.g. GPT-2 vocab 50257) | in: embeddings; out: sampler |
| 2 | **Token embedding** | ID → vector `[V, H]` | → + position |
| 3 | **Positional embedding** | inject order info (GPT-2: **learned** table `[ctx, H]`) | add to token emb |
| 4 | **Transformer block ×N** | the repeated compute unit | stacked |
| 5 | **Final LayerNorm** | normalize last hidden state | → LM head |
| 6 | **LM head** | hidden → logits `[H, V]` (tied to embedding) | → sampler |
| 7 | **Sampler** | logits → next token (greedy / temp / top-p) | → loop to #1 |

---

## 1.3 Inside one block (the baseline unit)

```
x ──► LayerNorm ──► MultiHeadAttention ──►(+)──► LayerNorm ──► FFN ──►(+)──┐
      └──────────────────────────────────┘     └───────────────────────┘  │
                                                                          └─► next block
```

**a) LayerNorm (pre-norm)** — normalize over the feature dim, learn `gamma,beta`.
```
y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
```

**b) Multi-Head Attention (MHA)** — full attention, every head has its own K and V.
```
Q = x @ Wq      # [B,T,H]
K = x @ Wk
V = x @ Wv
split into n_heads → [B, n_heads, T, head_dim]
scores = Q @ K^T / sqrt(head_dim)
scores = scores + causal_mask(-inf for future)      # <-- causal
A = softmax(scores, dim=-1)
O = A @ V          # [B, n_heads, T, head_dim]
out = concat_heads(O) @ Wo
```

**c) FFN** — a 2-layer MLP, usually 4× wider.
```
h = gelu(x @ W1 + b1)     # [B,T,4H]
out = h @ W2 + b2         # [B,T,H]
```

**d) Residual adds** — `x = x + sublayer(LN(x))` around both sublayers. This is what makes
deep stacks trainable.

---

## 1.4 Full data flow (shapes)

```
ids        [B, T]
emb        = E[ids]              [B, T, H]
x          = emb + P[0:T]        [B, T, H]
for block in 0..N-1:  x = block(x)
x          = LayerNorm(x)
logits     = x @ E^T             [B, T, V]     # weight tying
```

---

## 1.5 Connecting the parts (assembly)

1. Tokenize prompt → `ids`.
2. Look up `E[ids]`, add learned positional rows `P[0:T]`.
3. Pass through `N` identical blocks (weights are **not** shared between blocks).
4. Final norm, then LM head → logits at the **last** position for generation.
5. Sample one token, append, repeat.

Everything is just `matmul`, `softmax`, `gelu`, `layernorm`, plus a causal mask. That's the
whole model.

---

## 1.6 Inference loop (with KV cache)

Naively you re-run the whole sequence each step — wasteful. Cache K,V per layer:
```
prefill: run whole prompt, store K,V for every layer -> sample token 1
decode : for each new token, compute q,k,v for THIS token only,
         append k,v to cache, attend over full cache, sample next
```
KV cache size per token = `2 * n_heads * head_dim * n_layers * dtype_bytes`.
**This number is exactly what part 03 (MLA) attacks.**

---

## 1.7 Training loop (next-token prediction)

```
for step:
    ids, labels = next_batch()            # labels = ids shifted by 1
    logits = model(ids)                   # [B,T,V]
    loss   = cross_entropy(logits, labels)
    grad   = backward(loss)
    clip_grad_norm(grad, 1.0)
    adamw.step(grad); scheduler.step()
```
- Loss: softmax cross-entropy over the vocab at every position.
- Optimizer: AdamW, cosine schedule + warmup, weight decay 0.1 on matrices only.
- This is **pure imitation** — the model learns to predict text, nothing more. No RL.

---

## 1.8 C++ / g++ implementation notes

- Header-only ops with `inline` (or `template`) is fine and fast:
  `matmul`, `layernorm`, `gelu`, `softmax`, `add`, `embedding_gather`, `causal_mask`.
- Keep all fingerprints (shapes, strides) in one `Tensor` struct.
- Build (PowerShell, MinGW-w64 + OpenBLAS):
  ```powershell
  g++ -std=c++20 -O2 -march=native -fopenmp -Iinclude `
      (Get-ChildItem src -Filter *.cpp).FullName -o gpt.exe -lopenblas
  ```
  If you split hot code across `.cpp` files, add `-flto` to compile **and** link.
- Recommended first milestone: a GPT-2-small equivalent.

**Reference config (GPT-2 small):** `H=768`, `layers=12`, `n_heads=12`, `head_dim=64`,
`ctx=1024`, `V=50257`, learned positions, GELU FFN, pre-norm, tied embeddings.

---

## 1.9 What this baseline is missing (bridge to 02 & 03)

| Missing thing | Added in | Why |
|---------------|----------|-----|
| Reasoning behavior | **02** | RL/GRPO + `<think>` tokens = "thinking" |
| Cheap KV cache | **03** | MLA compression |
| Cheap compute, big capacity | **03** | DeepSeekMoE |
| Better training signal | **03** | Multi-Token Prediction |
| Low memory/precision | **03** | FP8 |

Build 01 fully, then move to 02.
