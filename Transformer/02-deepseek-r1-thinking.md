# 02 — DeepSeek-R1: What Changes for "Thinking"

**Part 2 of 3.** Read part 01 first. This file explains the **smallest possible delta** that
turns a plain GPT (part 01) into a reasoning model like DeepSeek-R1: a **token format** plus
a **reinforcement-learning post-training stage**.

> Crucial framing: **"thinking" is not an architectural block.** DeepSeek-R1's *architecture*
> is DeepSeek-V3 (part 03). R1's reasoning comes from **post-training**, not from new layers.
> So "how to make R1" = "make a V3 decoder (03), then apply the training recipe below."

---

## 2.1 Change #1: the reasoning token format

The model outputs a **reasoning section** before its answer. Format (special tokens, ids):

| Token | Example id | Meaning |
|-------|-----------|---------|
| `<｜begin▁of▁sentence｜>` | 0 | BOS |
| `<｜end▁of▁sentence｜>` | 1 | EOS |
| `<think>` | 128798 | open reasoning |
| `</think>` | 128799 | close reasoning |
| `<｜User｜>` | 128803 | user turn marker |
| `<｜Assistant｜>` | 128804 | assistant turn marker |

A turn looks like:
```
<｜Assistant｜><think>
... long chain of thought ...
</think>
... final answer ... <｜end▁of▁sentence｜>
```

Nothing about this requires new math — it's just tokens the model learns to emit. R1's own
model card notes R1 sometimes skips thinking (`<think>\n\n</think>`); they recommend forcing
the output to start with `<think>\n`.

---

## 2.2 Change #2: train with RL (GRPO), not just imitative CE

The baseline (01) is trained by cross-entropy. R1 is post-trained with **GRPO — Group
Relative Policy Optimization** (Shao et al. 2024), which removes PPO's value/critic network.

**Per question `q`, sample a group of `G` outputs** `{o1..oG}` from the old policy, score
each with a reward `r_i`, and build **group-relative advantages**:

```
A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)
```

The policy is updated to maximize the clipped PPO-style objective plus a KL penalty to a
frozen **reference policy** `pi_ref`:

```
J(θ) = E[ (1/G) Σ_i  min( ρ_i * A_i , clip(ρ_i, 1-ε, 1+ε) * A_i )  -  β * KL(π_θ ‖ π_ref) ]
where ρ_i = π_θ(o_i|q) / π_θ_old(o_i|q)
```

- **No critic/value model** — the group mean/std *is* the baseline. Huge memory savings.
- Hyperparameters (R1-Zero): lr `3e-6`, KL coefficient `β=0.001`, rollout temperature `1`,
  clip `ε=0.2` typical.
- Advantage `A_i` is the **same for every token** in output `i`.

### Reward design (the actual signal)
```
Reward = Reward_accuracy + Reward_format
```
- **Accuracy reward** — rule-based, verifiable:
  - math: check the final answer (boxed / exact match) against ground truth.
  - code: run a compiler + predefined test cases.
  - logic: pattern/answer matching.
- **Format reward** — only requires the reasoning to be wrapped in the `<think>...</think>`
  tags. Same weight as accuracy.
- **No neural reward model** is used for reasoning (process/outcome reward models were tried
  and rejected: reward hacking, hard to define "a step").

---

## 2.3 Change #3: the full R1 training pipeline

```
                 (V3-Base, from part 03)
                        │
   R1-Zero:  pure RL (GRPO)  ──────────────────────────────►  reasoning emerges
                        │                                     (messy: language mixing,
                        │                                      poor readability)
   R1:  cold-start SFT (few thousand long-CoT examples)
        → RL (GRPO, reasoning)
        → rejection sampling: keep only correct traces (≈600K reasoning)
          + ≈200K non-reasoning examples → SFT again
        → RL again (all scenarios)
                        │
                        ▼
                  DeepSeek-R1
```

- **R1-Zero** proved reasoning can emerge from RL alone on a base model — the "aha moment".
- **R1** fixes R1-Zero's problems (readability, language mixing, weak general ability) with a
  cold-start SFT and a final general-purpose RL stage.
- **Distillation** then transfers R1's traces into small dense students (Llama/Qwen). That is
  exactly the teacher→student plan if you want to distil.

---

## 2.4 What you actually implement in C++ (delta over part 01)

| Piece | Baseline (01) | R1 (02) |
|-------|---------------|---------|
| Tokenizer | BPE | add `<think>`/`</think>`/`<｜User｜>` etc. |
| Forward pass | same | same (no new layers!) |
| Generation | greedy/top-p | sampling with temperature 1 for rollouts |
| Loss | cross-entropy | **GRPO surrogate loss** |
| Extra models | none | **reference policy** (frozen copy) |
| Extra code | none | **reward functions** (answer check / code exec) |

GRPO loop skeleton:
```
for each batch of questions:
    for each question q:
        rollouts = sample_G_outputs(policy, q, T=1)         # text incl. <think>...
        r_i      = accuracy_reward(rollouts_i, answer) + format_reward(rollouts_i)
        A_i      = (r_i - mean(r)) / (std(r) + eps)
        logp_new = policy.logprob(rollouts_i)
        logp_old = old_policy.logprob(rollouts_i)           # frozen at step start
        logp_ref = ref_policy.logprob(rollouts_i)           # frozen forever
        loss    += grpo_loss(logp_new, logp_old, logp_ref, A_i, eps, beta)
    loss.backward(); adamw.step(); ema_update(old_policy)
```
You need `logprob` for a token sequence = sum of `log softmax(logits)[token]` along the
sequence — get this right; it's the crux.

---

## 2.5 Contrast: baseline vs R1

| | 01 Baseline (GPT/ChatGPT) | 02 DeepSeek-R1 |
|--|---------------------------|----------------|
| Output | answer only | `<think>` reasoning + answer |
| Training | pretrain CE (+ optional RLHF) | pretrain CE + **GRPO RL** |
| Reward | human preference (RLHF) | **rule-based accuracy + format** |
| Critic | (PPO) uses a value model | **none (group baseline)** |
| New architecture | none | **none — it's V3 (see 03)** |
| Cost driver | generation quality | **long reasoning sequences** |

---

## 2.6 g++ build notes

- This stage is dominated by **generation throughput** (rollouts) and **logprob**-heavy
  backward passes. Keep the sampler and `log_softmax` fast.
- You can develop 02 on the **part-01 model** first (a small GPT + GRPO on a toy math task)
  before adding DeepSeek's architecture from 03 — much easier to debug.
- Same build command as 01 (plain `g++`, `-O2 -march=native`, `-flto` if split).

**Milestone:** a 10–50M-param GPT trained with GRPO on a synthetic arithmetic task that
learns to emit `<think>...</think>` then a correct answer. That tiny system *is* R1's recipe
in miniature.
