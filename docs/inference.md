# Inference: KV cache for Multiscreen

This document describes how Multiscreen decodes autoregressively with a
per-layer KV cache, and how the implementation in `multiscreen/model.py`
differs from a standard Transformer cache.

## The basic idea

Autoregressive decoding generates one token at a time. A naive loop
re-runs the entire forward pass on the growing sequence each step:

```python
for _ in range(max_new):
    logits, _ = model(input_ids)
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    input_ids = torch.cat([input_ids, next_id], dim=1)
```

This does `O(T²)` work per step (because the screening matmul is
`T × T`), giving `O(T³)` total work.

A KV cache avoids reprocessing the prompt and previously generated
tokens: we store everything per-layer that only depends on past
positions, and on each new step we only compute the new-token projections
and a cross-attention between the new query and the cached keys / values.

```python
# 1. Prefill: one forward pass over the whole prompt
logits, kv_caches = model(prompt_ids)

# 2. Incremental decode
for step in range(max_new):
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    logits, kv_caches = model(
        next_id, start_pos=prompt_len + step, kv_caches=kv_caches,
    )
```

## What is cached

For each `GatedScreeningBlock`, the cache is a tuple `(K, V)` with shape:

- `K: (B, N_H, T_cached, d_K)` — **post-MiPE**, unit-normalized
- `V: (B, N_H, T_cached, d_V)` — unit-normalized

The reason we cache the post-MiPE K (not the raw projection) is that MiPE
rotation only depends on the absolute position of the token, so the
rotated K at position `j` is fixed once computed. Caching it post-rotation
means we never recompute it.

For V we cache after unit normalization — again, this value is fixed once
the token is seen.

We do **not** cache Q, because Q is only needed at the current query
position and is recomputed each step anyway.

## The softmask in the cached path

The non-cached softmask is a square `(T, T)` matrix:

```
m[i, j] = 0.5 · (cos(π · (j - i) / w_h) + 1)   if -w_h < j - i ≤ 0
        = 0                                      otherwise
```

In the cached path we have a row of length `T_total = T_cached + T_new`
for each new query position, so the mask becomes a non-square
`(1, N_H, T_new, T_total)` tensor. The formula is the same, with
`rel[i, j] = k_pos[j] - q_pos[i]` where `q_pos` runs over the new
positions `[start_pos, start_pos + T_new)` and `k_pos` runs over
`[0, T_total)`. See `GatedScreeningBlock._softmask` in `model.py`.

Because the mask is recomputed from scratch each step we don't need to
keep any per-step position buffers around; only the (K, V) tensors.

## API

```python
def MultiscreenModel.forward(
    input_ids: torch.Tensor,              # (B, T_new)
    start_pos: int = 0,                   # absolute position of input_ids[:, 0]
    kv_caches: list[ScreeningCache] | None = None,
) -> tuple[torch.Tensor, list[ScreeningCache]]
```

- `start_pos=0` and `kv_caches=None` is the training / naive forward path.
- `start_pos=k, kv_caches=None, T_new=k` corresponds to an initial
  *prefill* of the prompt — the same as training forward, but the
  returned cache is populated.
- Subsequent calls pass `kv_caches` from the previous step and advance
  `start_pos`.

The model returns an empty cache list when `self.training == True` so
that training does not pay the cost of materializing the K/V tensors.

Chunked decoding is supported: you can feed multiple new tokens per step
(e.g. `T_new=4`) and the cache grows accordingly.

## Correctness

`tests/test_kv_cache.py` verifies six scenarios that all must match a
full forward pass bit-for-bit (within fp32 tolerances):

1. Train-mode and eval-mode prefill produce identical logits.
2. Prefill a prefix, then decode the rest one token at a time.
3. Decode one token at a time from an empty cache.
4. Same as (2) but with batch size > 1.
5. Cache tensor shapes grow correctly per step.
6. Chunked decoding (multiple new tokens per step) matches a full forward.

## Performance

KV cache delivers the biggest wins when combined with `torch.compile`
and applied to **long-context** decoding. Benchmarked on RTX 5070 Ti,
200M parameters (`hidden_dim=1024, N_L=N_H=18, d_K=32, d_V=128`), bf16,
prompt length 256, generate length 128:

| Configuration | per-token (ms) | tok/s | speedup |
|---------------|---------------:|------:|--------:|
| eager + full re-forward    | 16.83 |  59 | 1.0x |
| eager + KV cache           | 16.85 |  59 | 1.0x |
| compile + full re-forward  |  4.77 | 210 | 3.5x |
| **compile + KV cache**     |  **3.41** | **293** | **5.0x** |

Reproduce with:

```bash
python scripts/bench_inference.py --prompt 256 --generate 128 \
    --max-seq-len 512 --compile --warmup 5 --repeats 10
```

### Where KV cache matters most

On this architecture, the dominant cost is element-wise ops (normalize,
gate, tanh-norm), not matmul — the screening matmul accounts for only
~4% of CUDA time in training profiles. As a result:

- **Eager mode**: per-step kernel launch overhead dominates and KV cache
  is roughly a tie with full re-forward. On the table above, both sit
  at ~17 ms/token.
- **Compile mode**: the fixed kernel cost shrinks by ~3.5x, so the
  actual matmul savings from the cache become visible. The cache adds
  another **~40%** throughput on top of `compile + full re-forward`
  (210 → 293 tok/s).
- **Long prompts / long generations** favor the cache further because
  the avoided work grows with sequence length.

In short: enable both `torch.compile` and the KV cache path together
and you will get ~5x over eager, with ~40% of that coming from the
cache.

## Limitations and caveats

- **`max_seq_len` is a hard bound**: the cache grows monotonically. If
  your application needs unbounded streaming, add a sliding window on
  the cache yourself (drop oldest positions once full). The underlying
  model still respects causal and per-head window constraints inside
  `w ≤ mipe_threshold`.
- **bf16 position precision**: `_apply_mipe` creates the position tensor
  in the same dtype as Q. For sequences ≤ 256 tokens this is exact in
  bf16; beyond that you will start losing integer precision on position
  indices. Either cast positions to fp32 locally or keep `max_seq_len`
  small enough.
- **`torch.compile` + `reduce-overhead` + KV cache**: CUDA graphs assume
  static tensor addresses, but our returned cache tensors are fed back
  to the next call. Use `mode="default"` instead (benchmarked above).
  If you need CUDA graphs, you need to call
  `torch.compiler.cudagraph_mark_step_begin()` between steps and clone
  the cache outputs.
- **Windows build toolchain**: inductor invokes `cl.exe` to build the
  CPU-side kernel shim even when the heavy kernels run on CUDA. Use
  `multiscreen.setup_compile_env()` to load the full MSVC environment
  (including INCLUDE paths so `omp.h` is found); without it,
  `torch.compile` will fail with `fatal error C1083: include file
  'omp.h': No such file or directory` on Windows.
