# Building a DeepSeek-R1-style Transformer in C++ (Inference + Distillation Training)

This is a build guide for a from-scratch C++ implementation that:

1. **Runs inference** for a DeepSeek-R1-style decoder (MLA + DeepSeekMoE).
2. **Runs a training loop** so you can **distill** a large "teacher" (DeepSeek-R1, or its
   outputs) into a **small "student"** dense model (e.g. **Qwen3-0.6B**).

> Note on naming: there is no "Qwen 3.5:0.8B". The realistic small students are
> **Qwen3-0.6B** / **Qwen3-1.7B** (dense) or **Qwen2.5-0.5B**. This guide uses
> **Qwen3-0.6B** as the concrete student example. Swap the config numbers to target another.

> Important mental model: **"thinking"/reasoning is a behavior, not an architectural part.**
> The DeepSeek-R1 *architecture* is DeepSeek-V3. The reasoning comes from RL (GRPO) + reward
> models applied on top of that decoder. What follows separates (a) architecture, (b) inference,
> (c) training, (d) the distillation recipe that actually transfers the reasoning.

---

## 0. The two models in this project

| Role | Model | Architecture | Params | Where it runs |
|------|-------|--------------|--------|----------------|
| **Teacher** | DeepSeek-R1 (V3 arch) | MLA + DeepSeekMoE (MoE decoder) | 671B total / 37B active | API, or your C++ engine on big HW |
| **Student** | Qwen3-0.6B | Dense GQA + SwiGLU decoder | ~0.6B | Your C++ engine (infer + train) |

Distillation = train the **student** to match the **teacher**'s outputs (hard text traces
and/or soft logits). You do **not** need to train the teacher.

---

## 1. Terminology / tensor conventions

- `B` batch, `T` sequence length, `H` hidden dim, `V` vocab.
- Weight matrices stored **row-major**, `Linear(x, W)` computes `x @ W^T` (PyTorch convention).
- Accumulate in `fp32`; store activations/weights in `bf16` (or `fp8` for the teacher).
- Every block is **pre-norm** with **residual connections**.
- No biases anywhere in DeepSeek/Qwen3 (except MoE router when needed — DeepSeek uses
  a bias for *load balancing*, not in the linear math).

---

## 2. DeepSeek-R1 (V3) architecture — part by part

Config (DeepSeek-V3 / R1): `H=7168`, `layers=61`, `V=129280`, MLA heads `n_h=128`,
`q_lora_rank=1536`, `kv_lora_rank=512`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`,
`v_head_dim=128`; MoE `n_routed=256`, `n_shared=1`, `top_k=8`, `moe_inter=2048`;
first 3 layers are dense FFN; RoPE `theta=10000` with scaling.

### 2.1 Tokenizer
- Byte-level BPE, vocab ~129,280.
- Special tokens matter for R1 reasoning, e.g.:
  `＜｜begin▁of▁sentence｜＞`, `<｜｜User｜｜>`, `<｜｜Assistant｜｜>`,
  `＜｜end▁of▁thinking｜＞`, `｜｜DSML｜｜`, `＜｜end▁of▁sentence｜＞`.
- **Use:** text ↔ token IDs. The `begin/end thinking` tokens are how the model switches
  between the reasoning trace and the final answer.
- **Connects to:** input of the embedding table; output of the sampler.

### 2.2 Token Embedding
- Table `[V, H]`. `x = E[token_id]` → shape `[B, T, H]`.
- **Connects to:** block 0.

### 2.3 Final RMSNorm
- `y = x / sqrt(mean(x^2) + eps) * weight`, learned `weight[H]`.
- **Use:** stabilizes the last representation before logits.
- **Connects to:** LM head.

### 2.4 LM Head (unembed) + Sampler
- `logits = h @ Emb^T` (weight-tied with the embedding table in V3/R1).
- Sampler: temperature, top-p / top-k, repetition penalty; greedy for eval.
- **Connects to:** loop back to the tokenizer.

### 2.5 The Transformer Block (repeated ×61)

```
x ──► RMSNorm ─► MLA ─►(+)──► RMSNorm ─► DeepSeekMoE ─►(+)──► next block
 └─────────────────────┘     └──────────────────────────┘
```

**a) RMSNorm (pre-norm)** — as above.

**b) MLA — Multi-head Latent Attention** (the defining DeepSeek innovation).
Instead of storing full K and V per head, it stores a **low-rank latent** `c_KV`
and reconstructs K/V on the fly. It also **decouples RoPE**: position info is applied
to a *separate* small head dimension so the latent stays uncompressed by rotation.

Data flow:
```
x ─┬─ q_a_proj ─► q_a_layernorm ─► q_b_proj ─► Q  [n_h*(128+64)]
   │
   └─ kv_a_proj ─► kv_a_layernorm ─┬─► kv_lora (512)  ──┐ (cache this + k_rope)
                                   └─► k_rope (64)      │
         k_rope ─► RoPE ─────────────┘                  │
         kv_lora ─► kv_b_proj ─► K_nope[n_h,128], V[n_h,128]
   Q_nope = Q[:, :128]; Q_rope = Q[:, 128:]
   Q_rope ─► RoPE
   scores = (Q_nope·K_nope + Q_rope·K_rope) / sqrt(192)
   attn = softmax(scores) (causal) ; out = attn @ V ; o_proj
```
- **KV cache:** store only `kv_lora` (512) + `k_rope` (64) per token → tiny cache.
- **Use:** attention with 128 heads at a fraction of the memory of full MHA.
- **Connects to:** residual add.

**c) Residual add** — `x = x + MLA(RMSNorm(x))`.

**d) DeepSeekMoE FFN** — replaces the dense FFN.
- **Router:** `gate = x @ W_router` (`[H, 256]`), softmax, take **top-8** experts by
  `score + bias` (bias is the *aux-loss-free load-balancing* term, updated during training).
- **Routed experts:** 256 SwiGLU MLPs; each token uses 8. Each expert:
  `SwiGLU(x) = (silu(x @ W_gate) * (x @ W_up)) @ W_down`, `moe_inter=2048`.
- **Shared experts:** 1 always-on SwiGLU expert (`n_shared`), added to every token.
- **Combine:** `out = Σ (router_prob_i * expert_i(x)) + shared(x)`.
- **Use:** 671B params but only ~37B active per token → cheap compute, huge capacity.
- **Connects to:** residual add → next block.

---

## 3. Qwen3-0.6B architecture (student) — part by part

Config: `H=1024`, `layers=28`, `V=151936`, `n_heads=16`, `n_kv_heads=8` (GQA),
`head_dim=128` (note: `H ≠ n_heads*head_dim`), `intermediate=3072`, `RMSNorm eps=1e-6`,
`tie_word_embeddings=true`, **QK-Norm** on q/k per head, RoPE `theta=1e6`, no bias.

Block flow:
```
x ─► RMSNorm ─► GQA+QK-Norm+RoPE ─►(+)──► RMSNorm ─► SwiGLU ─►(+)──► next
```

Parts:
1. **Embedding** `[V, H]`.
2. **RMSNorm** (pre-norm, ×2 per block).
3. **GQA attention**:
   - `q = x@Wq` → reshape `[T, 16, 128]`; `k = x@Wk` → `[T, 8, 128]`; `v = x@Wv` → `[T, 8, 128]`.
   - **QK-Norm:** per-head RMSNorm applied to `q` and `k` (stabilizer).
   - **RoPE** applied to `q` and `k`.
   - `k`/`v` heads are **repeated** to 16 (GQA group size = 2) for attention.
   - causal softmax `scores/sqrt(128)` → `@v` → `o = out@Wo`.
4. **SwiGLU** dense FFN (`silu(gate)*up → down`, `intermediate=3072`).
5. **Final RMSNorm** → **LM head** (tied to embedding).
6. **Sampler**.

This is a *standard dense decoder*. It's much easier to implement and train than R1's MoE+MLA.

---

## 4. Inference engine (C++)

### 4.1 Tensors & kernels
- `Tensor` = `{data*, shape, strides, dtype}`.
- Kernels: `matmul`, `rmsnorm`, `rope`, `softmax`, `silu`, `mul`, `add`, `topk`, `argmax`,
  `embedding_gather`.
- Backend: **OpenBLAS** (CPU) for sgemm; **cuBLAS** if GPU. Keep dtype conversions in one place.

### 4.2 Weight loading
- **GGUF** (llama.cpp format) or **safetensors**. Map each tensor name to your layer structs.
- For the teacher, easiest path is API calls; local 671B is not practical.

### 4.3 KV cache & decode loop
```
prompt tokens ─► prefill (full seq, fill cache) ─► sample ─► decode 1 token at a time
each new token: append to cache; recompute only current step's attention.
```
- **MLA cache:** `kv_lora[512] + k_rope[64]` per token per layer.
- **GQA cache:** `k,v[8,128]` per token per layer.
- Stop at `end_of_sequence` / `end_thinking` tokens.

### 4.4 Sampling
- Greedy (argmax) for deterministic eval; temperature + top-p for generation.

---

## 5. Training loop (C++) — needed for distillation

### 5.1 Autograd options
- **Option A (recommended start):** hand-written backprop for the *student only* (dense
  decoder is tractable). Derive grads layer by layer; no ML framework needed.
- **Option B:** tiny reverse-mode autograd on top of your `Tensor` (a `grad` pointer +
  `backward()` per op). More general, more work.
- **Option C:** link libtorch for training while using your C++ kernels for inference.

### 5.2 Optimizer
- **AdamW** with cosine schedule + warmup, weight decay `0.1` (not on norms/biases), grad clip 1.0.

### 5.3 Forward/backward skeleton
```
for step in steps:
    batch = load(prompts)            # input_ids, labels, teacher_signals
    logits = student.forward(batch.input_ids)
    loss   = distill_loss(logits, batch.teacher_signals, batch.labels)
    grads  = backward(loss, student.params)
    clip_grad_norm_(grads, 1.0)
    adamw.step(grads)
    scheduler.step()
```

### 5.4 Losses
- **Hard distillation (SFT on traces):** cross-entropy on teacher-generated reasoning
  + answer tokens. Simplest and very effective for R1-style traces.
- **Soft distillation (KD):** match student to teacher logits:
  `L_kd = T^2 * KL( softmax(t_logits/T) || softmax(s_logits/T) )`, `T≈2–4`.
- **Combined:** `L = α*L_ce + (1-α)*L_kd` (e.g. α=0.5).
- **Length/reasoning shaping (optional):** reward-length, format reward if doing your own RL.

---

## 6. Distillation recipe that actually transfers "thinking"

1. **Generate traces** with the teacher on a task set (math/code/logic). Capture the full
   `…` reasoning plus the final answer.
2. **SFT the student** on those traces (hard distillation) with the R1 chat template.
3. **Add soft KD** if you can capture teacher logits (needs teacher in your engine; with an
   API you only get text → use hard distillation).
4. **Optional GRPO** on the student with verifiable rewards (answer correctness) to push
   reasoning further — this is the R1 "zero"/RL stage, now at small scale.
5. Over time, **length control** and **rejection sampling** (keep only correct traces) mimic
   R1's cold-start + rejection-sampling SFT stages.

What transfers: reasoning *format* and *heuristics* via imitation. What doesn't fully:
raw reasoning *capability* — a 0.6B student plateaus below the 671B teacher. Expect
"good student", not "R1-quality".

---

## 7. Build order (1 by 1)

```
Backend / Tensor
 └─ 1. Tensor + matmul (OpenBLAS)
 └─ 2. RMSNorm
 └─ 3. RoPE (rotate_half)
Student (Qwen3-0.6B) — do this first; it's your trainable target
 └─ 4. GQA attention + QK-Norm
 └─ 5. Dense block + residual
 └─ 6. Embedding + LM head + sampler
 └─ 7. End-to-end greedy decode; validate against HF outputs
Inference (teacher arch, for study/optional logits)
 └─ 8. MLA  (q/kv low-rank, decoupled RoPE)
 └─ 9. DeepSeekMoE (router, top-k, shared expert, aux-loss-free bias)
 └─ 10. Stack 61 blocks; weight loader (GGUF/safetensors)
Training / Distillation
 └─ 11. Backward for student block (hand-derived)
 └─ 12. AdamW + scheduler + grad clip
 └─ 13. Dataset + tokenized traces + dataloader
 └─ 14. CE loss → then KL soft-distill loss
 └─ 15. (optional) GRPO with verifiable reward
```

**Milestone checkpoint for every step:** unit-test the op against a NumPy/PyTorch reference
before moving on.

---

## 8. Suggested C++ project layout

```
transformer/
├─ build.ps1            # g++-only build script (PowerShell)
├─ build.bat            # g++-only build script (cmd.exe)
├─ include/
│  ├─ tensor.hpp        # Tensor, shape, dtype
│  ├─ ops.hpp           # matmul, rmsnorm, rope, softmax, silu, topk
│  ├─ layer.hpp         # Linear, RMSNorm, Attention, MLP
│  ├─ mla.hpp           # DeepSeek MLA
│  ├─ moe.hpp           # DeepSeekMoE router + experts
│  ├─ block.hpp         # TransformerBlock (inference + backward)
│  ├─ model.hpp         # Model (student/teacher), config
│  ├─ tokenizer.hpp     # BPE
│  ├─ sampler.hpp       # temperature/top-p/greedy
│  ├─ optimizer.hpp     # AdamW + scheduler
│  ├─ loss.hpp          # cross_entropy, kl_distill
│  └─ autograd.hpp      # optional tape
├─ src/                 # .cpp implementations
├─ tests/               # op-level reference tests
├─ weights/             # GGUF / safetensors
└─ data/                # tokenized traces
```

---

## 8b. Building with GCC only (no CMake)

On Windows your `gcc`/`g++` is almost certainly **MinGW-w64** (from MSYS2 or w64devkit).
Verify with `g++ --version`. You need an **OpenBLAS** for MinGW (MSYS2 package
`mingw-w64-x86_64-openblas`, or a prebuilt `libopenblas.a` + headers).

### 8b.1 Compile everything in one shot

PowerShell (globs expand, so this is the cleanest):
```powershell
g++ -std=c++20 -O2 -march=native -fopenmp -I include `
    (Get-ChildItem src -Filter *.cpp).FullName `
    -o transformer.exe -lopenblas
```

cmd.exe (does **not** expand globs — list files or use `build.bat` below):
```
g++ -std=c++20 -O2 -march=native -fopenmp -Iinclude src\*.cpp -o transformer.exe -lopenblas
```

If OpenBLAS is not on the default search path, point at it explicitly:
```
g++ ... -IC:\openblas\include -LC:\openblas\lib -lopenblas -o transformer.exe
```

### 8b.2 build.ps1 (save this in the repo root)
```powershell
$ErrorActionPreference = "Stop"
$cxx   = "g++"
$flags = @("-std=c++20","-O2","-march=native","-fopenmp","-Iinclude")
$src   = (Get-ChildItem src -Filter *.cpp).FullName
& $cxx @flags @src "-o" "transformer.exe" "-lopenblas"
if ($LASTEXITCODE -eq 0) { Write-Host "OK -> transformer.exe" }
```

Run with: `powershell -ExecutionPolicy Bypass -File build.ps1`

### 8b.3 Faster incremental builds (objects, no CMake)

Compile each `.cpp` to `.o`, then link. Only changed files recompile.

PowerShell:
```powershell
New-Item -ItemType Directory -Force build | Out-Null
foreach ($f in Get-ChildItem src -Filter *.cpp) {
    $o = "build\" + $f.BaseName + ".o"
    if (-not (Test-Path $o) -or $f.LastWriteTime -gt (Get-Item $o).LastWriteTime) {
        g++ -std=c++20 -O2 -march=native -fopenmp -Iinclude -c $f.FullName -o $o
    }
}
g++ build\*.o -o transformer.exe -lopenblas -fopenmp
```

### 8b.4 Debug / sanitizer build
```
g++ -std=c++20 -O0 -g -fsanitize=address,undefined -Iinclude src\*.cpp -o transformer_dbg.exe -lopenblas
```
(If MinGW lacks `-fsanitize=address`, use MSYS2's `ucrt64` toolchain or Valgrind on WSL.)

### 8b.5 Tips
- Build a **separate test exe** from `tests/` so you can run op-level checks without the full model:
  `g++ -std=c++20 -O2 -Iinclude tests\test_ops.cpp src\ops.cpp -o test_ops.exe -lopenblas`
- Keep `-march=native` only for local runs; drop it for distributable binaries.
- If linking fails with `undefined reference to cblas_*`, you're missing `-lopenblas`
  (or the `-L` path to it). If it's `__imp_*` errors, you mixed static/dynamic imports —
  match MEH/SEH and link the same `libopenblas` flavor.
- `-fopenmp` is optional; drop the flag and the pragmas if a plain-threaded build is easier.
- A `Makefile` is allowed here too (`mingw32-make`), it is **not** CMake — but the scripts
  above stay dependency-free.

---

## 9. Precision & performance notes

- **Student:** `bf16` compute, `fp32` master weights (mixed precision).
- **Teacher (V3):** trained/produced in **fp8** (fine-grained block scaling); if you run it
  locally you'll need fp8 dequant → bf16.
- **FlashAttention-style tiling** for long context; **KV-cache reuse** is mandatory for speed.
- **MoE** must gather tokens per expert (batched dispatch) or you waste most FLOPs.
- Memory: 671B in fp8 ≈ 671 GB + cache — not consumer-feasible. Plan on API teacher.

---

## 10. Key references

- DeepSeek-V3 Technical Report (MLA, DeepSeekMoE, aux-loss-free balancing, MTP).
- DeepSeek-R1 paper (GRPO, cold-start SFT, rejection sampling, reward models).
- Qwen3 Technical Report (GQA, QK-Norm, dense small models, long RoPE).
- llama.cpp (GGUF, CPU kernels, `ggml`) — best reference for a C/C++ inference engine.
- FlashAttention papers for tiled attention.

---

## 11. TL;DR

- Build a **dense Qwen3-style student first** — it is both your inference target and your
  trainable/ distillable model.
- Implement **MLA + DeepSeekMoE** separately as the **teacher architecture** (inference only
  if you have the weights; otherwise use the R1 API for training data).
- Distill via **hard traces (SFT) → soft KL (if logits available) → optional GRPO**.
- "Thinking" is trained, not architected: the architecture is just the decoder; the
  reasoning tokens are the interface.
