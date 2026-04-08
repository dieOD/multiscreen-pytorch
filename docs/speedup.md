# Optimization Journey: 15.7k -> 41.1k tok/s

This doc describes how we sped up Multiscreen training 2.6x without changing the math.

## Setup

- Hardware: NVIDIA RTX 5070 Ti 16GB
- Model: 154M parameters (N_L=18, N_H=18, d_E=1024, d_K=32, d_V=128)
- Sequence length: 256
- Precision: bf16 with `torch.amp.autocast`

## Baseline profile

Naive PyTorch implementation, B=16:

| Metric | Value |
|--------|-------|
| Throughput | **15,666 tok/s** |
| Peak VRAM | 12,346 MB |

CUDA-time breakdown (Self CUDA, 5 training steps):

| Operation | CUDA % | Notes |
|-----------|--------|-------|
| `aten::mul` (element-wise) | **24.7%** | rho * mask, gate, sO scaling |
| `aten::mm` (linear projections) | 23.3% | q/k/v/g/o_proj |
| `aten::copy_` (dtype cast) | **13.5%** | bf16 <-> fp32 in F.normalize |
| AdamW.step | 7.0% | optimizer |
| LinalgVectorNorm backward | 5.8% | normalize backward |
| `aten::div` | 5.7% | normalize forward |
| PowBackward (square backward) | 5.1% | trim-and-square |
| `aten::bmm` (q@k^T, rho_d@v) | **4.3%** | the actual matmul |

**Critical insight**: matmul is only 4.3%. The bottleneck is element-wise ops and dtype casts,
not the screening matmul itself. This means kernel fusion (rather than a Triton matmul kernel)
will give the biggest win.

## PyTorch-level optimizations (Phase 1-3)

These are direct edits to `_screening`, `_softmask`, `_apply_mipe`. Numerical equivalence
verified by unit tests.

### Phase 1: Softmask cache

The relative position tensor `rel = pos[None] - pos[:, None]` is constant for fixed T but
was being recomputed every forward pass. Cached as a `register_buffer`.

Also replaced `torch.where(valid, cosine, zeros)` with `cosine * valid` to avoid GPU branch
overhead.

### Phase 2: MiPE in-place rotation

```python
# Before (3 tensors, torch.cat)
q_rot = torch.cat([
    (q0 * cos_a - q1 * sin_a).unsqueeze(-1),
    (q0 * sin_a + q1 * cos_a).unsqueeze(-1),
    q[..., 2:],
], dim=-1)

# After (1 tensor, in-place index assign)
q_rot = torch.empty_like(q)
q_rot[..., 0] = q0 * cos_a - q1 * sin_a
q_rot[..., 1] = q0 * sin_a + q1 * cos_a
q_rot[..., 2:] = q[..., 2:]
```

### Phase 3: Fused trim-square-mask

Reduced T x T intermediates from 3 (sim, rho, rho_d) to 2 (sim, rho_d):

```python
# Before
rho = torch.clamp(1.0 - r * (1.0 - sim), min=0.0).square()
mask = self._softmask(...)
rho_d = rho * mask  # 3 T x T tensors alive

# After
mask = self._softmask(...)
rho_d = torch.clamp(1.0 - r * (1.0 - sim), min=0.0).square_().mul_(mask)
# square_() and mul_() are in-place on the clamp() output (a temporary), so autograd-safe
```

Saves ~36 MB per layer at B=16, bf16.

**Result**: ~16,400 tok/s (~5% improvement). Expected to be small because the bottleneck
is kernel launch overhead, which only `torch.compile` can fix.

## Phase 4: Gradient checkpointing

```python
for layer in self.layers:
    if self.training and self.gradient_checkpointing:
        x = checkpoint(layer, x, use_reentrant=False)
    else:
        x = layer(x)
```

Results:

| Setting | tok/s | VRAM |
|---------|-------|------|
| no-ckpt B=16 | 15,666 | 12,346 MB |
| ckpt B=16 | 12,471 | **3,096 MB** |
| ckpt B=32 | 12,806 | 3,824 MB |
| ckpt B=48 | 12,882 | 4,830 MB |
| ckpt B=64 | 12,860 | 5,830 MB |

VRAM dropped 75% (12.3GB -> 3.1GB), but tok/s stayed flat. This proves we're **compute-bound**:
batch size scaling doesn't help when each step has its own ceiling. The 33% recompute overhead
wipes out the parallelism gain.

**Conclusion**: gradient checkpointing is useful for VRAM (e.g. enables larger batch on smaller GPUs)
but is not the answer to speed.

## Phase 5: torch.compile (the big win)

```python
model = torch.compile(model, mode="default")
```

Results:

| Setting | tok/s | VRAM | vs Baseline |
|---------|-------|------|-------------|
| compile B=16 | **37,767** | 6,158 MB | **2.4x** |
| compile B=24 | **39,145** | 8,236 MB | **2.5x** |
| compile B=32 | **41,126** | 10,204 MB | **2.6x** |

Note that compile *also* reduces VRAM (12.3GB -> 6.2GB at B=16) because the inductor backend
fuses element-wise ops, eliminating intermediate tensor materializations.

### What inductor fused

The 24.7% `aten::mul` and 13.5% `aten::copy_` (dtype casts inside `F.normalize`) almost
disappeared. inductor generated fused kernels covering:

- normalize -> MiPE -> matmul -> trim-and-square -> mask -> matmul -> tanhnorm
- gate computation (`tanh(silu(g)) * u * sO`)
- output projection chain

### Caveats

- **`torch.compile` + gradient checkpointing is incompatible**: the checkpoint boundary
  prevents inductor from fusing across layers. Result: 12.4k tok/s (slower than baseline).
  Don't combine them.
- **`mode='max-autotune'` fails on consumer GPUs**: the autotuner picks templates that need
  >100KB shared memory, but RTX 5070 Ti only has 99KB per SM. Use `mode='default'`.
- **First run is slow**: kernel compilation takes ~1-2 minutes. Subsequent runs use cache.
- **Windows requires MSVC**: the inductor codegen path uses cl.exe. See [setup.md](setup.md).

## Phase 6: Custom Triton kernel (deferred)

The original paper authors use a custom Triton kernel for inference (Section 4.5). Since
torch.compile gave us 2.6x and matches our target, a hand-written Triton screening kernel
is currently deferred. The estimated additional speedup is 30-100% based on Flash Attention
analogies, but the complexity (forward + backward kernel, gradient flow through MiPE and softmask)
is significant.

If you want to try, the design constraint is:

- Keep MiPE and softmask in PyTorch (autograd handles `sw` gradients through them)
- Triton kernel handles only: `Q@K^T -> trim-and-square -> mul by precomputed mask -> aggregate -> tanhnorm`
- Backward kernel must output `grad_Q, grad_K, grad_V, grad_r, grad_M` (where M is the precomputed softmask)

The trim-and-square derivative is:
```
d_rho/d_s = 2r * max(1 - r(1-s), 0)   (zero outside the clamp)
```

## Summary

| Phase | tok/s | micro_batch | VRAM | Cumulative |
|-------|-------|-------------|------|------------|
| Baseline | 15,666 | 16 | 12,346 MB | 1.0x |
| +Phase 1-3 (PyTorch) | ~16,400 | 16 | ~12,300 MB | 1.05x |
| +Phase 4 (ckpt only) | 12,860 | 64 | 5,830 MB | 0.8x (slower!) |
| **+Phase 5 (compile)** | **41,126** | **32** | **10,204 MB** | **2.6x** |

The lesson: **profile first**. Our intuition said "no Flash Attention -> write a Triton kernel",
but profiling revealed the bottleneck was kernel launch overhead on element-wise ops, not the
matmul. `torch.compile` solved that for free.
