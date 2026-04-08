# Multiscreen Architecture

This is a brief technical overview of the Multiscreen architecture.
For the full motivation and experimental results, see the paper:
["Screening Is Enough"](https://arxiv.org/abs/2604.01178) (Nakanishi, 2026).

## Why screening?

Standard softmax attention defines **relative** query-key relevance: weights are normalized
across keys, so a key receives high weight only when its score exceeds that of competing keys.
This has two consequences:

1. There is no notion of *absolute* relevance — every key always receives some weight,
   even when no key is genuinely relevant.
2. As context length grows, attention is diluted across more keys, weakening contributions
   from genuinely relevant tokens.

**Screening** evaluates each key independently against an explicit threshold. Irrelevant keys are
discarded; relevant keys are aggregated without normalization across keys. This removes global
competition and allows the model to represent the absence of relevant context.

## Screening unit

Given query `q`, key `k`, value `v` (all in `R^d_K` or `R^d_V`):

1. **Unit-length normalization** of Q, K, V:
   ```
   q_bar = q / ||q||,  k_bar = k / ||k||,  v_bar = v / ||v||
   ```
   This bounds similarity to `[-1, 1]` and removes value-norm dominance.

2. **Minimal Positional Encoding (MiPE)**: a RoPE-like rotation applied **only to the first 2
   coordinates** of Q and K, and **only when the learned screening window `w` is small**:
   ```
   theta(i, w) = pi * i * gamma(w) / w
   gamma(w) = 0.5 * (cos(pi * w / w_th) + 1)   if w < w_th
            = 0                                  otherwise
   ```
   Long-range tiles use the identity (no positional encoding), so they don't depend on
   length-extrapolation tricks.

3. **Bounded similarity**:
   ```
   s_ij = q_bar_i . k_bar_j^T  in [-1, 1]
   ```

4. **Trim-and-Square** (the key step):
   ```
   alpha_ij = max(1 - r * (1 - s_ij), 0)^2
   ```
   `r > 1` is a per-head learned acceptance sharpness. The clamp sets relevance exactly to zero
   when `s_ij <= 1 - 1/r`. The square emphasizes high-similarity keys.

5. **Softmask** (causal + distance-aware):
   ```
   m_ij(w) = 0.5 * (cos(pi * (j - i) / w) + 1)   for -w < j - i <= 0
           = 0                                     otherwise
   alpha_d_ij = alpha_ij * m_ij(w)
   ```
   `w` is a per-head learned window width.

6. **Aggregation** (no softmax!):
   ```
   h_i = sum_j alpha_d_ij * v_bar_j
   ```

7. **TanhNorm** (bounds output norm):
   ```
   u_i = tanh(||h_i||) / ||h_i|| * h_i
   ```
   Preserves direction, smoothly bounds norm by 1.

## Gated screening tile

A tile wraps a screening unit with GLU-style gating:

```
q_i, k_i, v_i, g_i = x_i @ W_Q, W_K, W_V, W_G
u_i = Screening(q, k, v)
g_hat_i = tanh(silu(g_i))
h_i = u_i * g_hat_i
delta_x_i = exp(s_O) * h_i @ W_O
```

This unifies attention and FFN into a single operation, reducing parameter count.

## Model

```
x = exp(s_E) * normalize(W_E)[input_ids]                    # input embedding
for layer in 1..N_L:
    x = x + sum_h delta_x^(layer, h)                        # parallel tiles per layer
logits = x @ (exp(s_F) * normalize(W_E))^T                   # tied output (same W_E)
```

`s_E`, `s_F` are learned input/output scales. `W_E` is shared (weight tying).

## Scaling rule

A single supraparameter `Psi` controls the model:
```
N_L = N_H = Psi
d_E = Psi^2
```

`d_K`, `d_V`, `w_th` are kept constant across scales (paper Table 1: d_K=16, d_V=64, w_th=256).
Use `MultiscreenConfig.from_psi(psi=8)` to build a paper-style config.

| Psi | N_L | N_H | d_E | params (with GPT-2 vocab) |
|-----|-----|-----|------|---------------------------|
| 8   | 8   | 8   | 64   | ~6.5M                     |
| 12  | 12  | 12  | 144  | ~16M                      |
| 16  | 16  | 16  | 256  | ~30M                      |
| 18  | 18  | 18  | 1024 | ~154M (custom dV=128)     |

## Key parameters

| Parameter | Symbol | Init | Description |
|-----------|--------|------|-------------|
| `sw`      | s_w    | linspace(0, log w_th) per head | Learned window: w = exp(s_w) + 1 |
| `sr`      | s_r    | 0                              | Learned acceptance: r = exp(s_r) + 1 |
| `sO`      | s_O    | log(1/sqrt(N_H N_L))           | Per-tile output scale |
| `s_E`     |        | 0                              | Input embedding scale |
| `s_F`     |        | log(sqrt(d_E))                 | Output embedding scale |

## Training notes

- **Weight decay**: 0 (paper recommends none — Multiscreen is stable)
- **Gradient clipping**: none
- **Learning rate**: substantially larger than Transformer (paper uses 2^-4 = 0.0625; we use 1e-2 for stability margin)
- **Optimizer**: AdamW(beta1=0.9, beta2=0.95)
- **Schedule**: paper uses constant LR after warmup; we default to cosine decay
