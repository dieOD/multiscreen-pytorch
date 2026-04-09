# multiscreen-pytorch

A PyTorch implementation of **Multiscreen**, the screening-based language model architecture from
["Screening Is Enough"](https://arxiv.org/abs/2604.01178) (Nakanishi, 2026).

Multiscreen replaces softmax attention with **screening**: each key is evaluated independently
against a learned threshold, removing global competition between keys. This enables absolute
query-key relevance, stable training at large learning rates, and strong long-context retrieval.

## Highlights

- **Pure PyTorch reference implementation** of the Multiscreen model
- **2.6x faster training** than naive PyTorch via `torch.compile` (~41k tok/s on RTX 5070 Ti for a 154M model)
- **KV cache inference** with up to **5x faster decoding** via `torch.compile` + cache
- **Generic training script** using HuggingFace `datasets` + `tokenizers`
- **Gradient checkpointing** for low-VRAM training (-75% VRAM)
- **CPU-friendly tests** (20 unit tests, no GPU required)

## Installation

```bash
git clone https://github.com/dieOD/multiscreen-pytorch
cd multiscreen-pytorch
pip install -e ".[train]"
```

For the optional `torch.compile` speedup:

```bash
pip install -e ".[train,perf]"
```

On Windows, you also need MSVC (Visual Studio Build Tools with the C++ workload).
See [docs/setup.md](docs/setup.md) for details.

## Quick start

Train a tiny Multiscreen model on TinyStories with the GPT-2 tokenizer:

```bash
python scripts/train.py \
    --dataset roneneldan/TinyStories \
    --psi 8 \
    --max-steps 1000 \
    --micro-batch 16
```

This builds an ~8M parameter model (Psi=8 -> 8 layers, 8 heads, hidden_dim=64).

For a paper-comparable 154M run on Wikitext-103:

```bash
python scripts/train.py \
    --dataset wikitext --config wikitext-103-raw-v1 \
    --hidden-dim 1024 --num-layers 18 --num-heads 18 \
    --key-dim 32 --value-dim 128 --seq-len 256 \
    --max-steps 17000 --peak-lr 1e-2 \
    --micro-batch 32 --grad-accum 16 \
    --compile
```

## Inference

Greedy generation with the KV cache (see [`examples/generation.py`](examples/generation.py)):

```python
import torch
from multiscreen import MultiscreenConfig, MultiscreenModel

model = MultiscreenModel(MultiscreenConfig(...)).eval()

# 1. Prefill the prompt in one forward pass
prompt_ids = torch.tensor([[1, 2, 3, 4]])
logits, kv_caches = model(prompt_ids)
next_logits = logits[:, -1, :]

# 2. Incremental decode
for step in range(max_new_tokens):
    next_id = next_logits.argmax(dim=-1, keepdim=True)
    logits, kv_caches = model(
        next_id,
        start_pos=prompt_ids.shape[1] + step,
        kv_caches=kv_caches,
    )
    next_logits = logits[:, -1, :]
```

See [docs/inference.md](docs/inference.md) for the design (what's cached, how
the softmask is rebuilt per step, caveats).

## Profiling

Training throughput / VRAM:

```bash
# Default 154M config, B=16
python scripts/benchmark.py

# With torch.compile (~2.6x faster)
python scripts/benchmark.py --compile --batch-size 32

# Export Chrome trace for kernel-level inspection
python scripts/benchmark.py --trace
```

Inference throughput:

```bash
# Compare eager/compile × full-reforward/KV-cache on 200M model
python scripts/bench_inference.py --prompt 256 --generate 128 --compile
```

## Architecture

Each Multiscreen layer contains N_H parallel **gated screening tiles**. A tile:

1. Projects input into Q, K, V, G
2. Normalizes Q, K, V to unit length
3. Applies **MiPE** (RoPE-like rotation, only the first 2 dims, only when window is short)
4. Computes bounded similarity: `s = q . k^T` in `[-1, 1]`
5. **Trim-and-Square**: `rho = max(1 - r(1-s), 0)^2`
6. **Softmask**: causal + distance-aware cosine window of width `w`
7. Aggregates: `h = sum_j rho_d_ij * v_j`
8. **TanhNorm**: `tanh(||h||) / ||h|| * h` (bounds output norm by 1)
9. Gates with `tanh(silu(g))` and projects back to model dim

`r`, `w` are per-head learned parameters. See [docs/architecture.md](docs/architecture.md) for the math.

## Project layout

```
multiscreen-pytorch/
├── multiscreen/
│   ├── config.py         # MultiscreenConfig
│   ├── model.py          # MultiscreenModel + GatedScreeningBlock (with KV cache)
│   ├── data.py           # PackedTextDataset (HF datasets loader)
│   ├── trainer.py        # Trainer with AMP, grad accum, checkpointing
│   └── compile_utils.py  # torch.compile / MSVC environment helpers
├── scripts/
│   ├── train.py              # End-to-end training script
│   ├── benchmark.py          # Training throughput / VRAM benchmark
│   └── bench_inference.py    # Inference throughput benchmark
├── examples/
│   ├── quickstart.py    # Minimal forward + backward
│   └── generation.py    # Greedy decode with KV cache
├── tests/
│   ├── test_model.py    # 14 unit tests for model / gradients / MiPE
│   └── test_kv_cache.py #  6 unit tests for incremental decode correctness
└── docs/
    ├── architecture.md
    ├── inference.md     # KV cache design and performance
    ├── setup.md
    └── speedup.md
```

## Optimizations applied

The default model implementation includes several optimizations beyond the naive paper transcription.

**Training** (154M model, batch=32, seq_len=256, bf16, RTX 5070 Ti):

| Optimization | What changed | Speedup |
|--------------|--------------|---------|
| MiPE in-place rotation | Replace `torch.cat` with index assignment | ~2-3% |
| Fused trim-square-mask | Reduce T×T intermediates from 3 to 2 via in-place ops | ~10-15% |
| `torch.compile` | inductor backend fuses element-wise ops | **2.4x** |
| Gradient checkpointing | Trade compute for VRAM (-75% VRAM, enables larger batch) | (compute-bound) |

See [docs/speedup.md](docs/speedup.md) for the full training optimization journey, including a CUDA-time profile.

**Inference** (200M model, bf16, prompt=256, generate=128, RTX 5070 Ti):

| Configuration | per-token (ms) | tok/s | Speedup |
|---------------|---------------:|------:|--------:|
| eager + full re-forward    | 16.83 |  59 | 1.0x |
| eager + KV cache           | 16.85 |  59 | 1.0x |
| compile + full re-forward  |  4.77 | 210 | 3.5x |
| **compile + KV cache**     |  **3.41** | **293** | **5.0x** |

KV cache alone is a tie in eager mode (the screening matmul is only ~4% of
total work) but stacks cleanly on top of `torch.compile`, adding another
~40% throughput for a total 5x speedup over the eager baseline. See
[docs/inference.md](docs/inference.md) for the design and when each
configuration matters.

## Status

This is an unofficial third-party implementation. The original paper authors have a custom Triton
implementation (Section 4.5) which is not yet publicly available as far as we know. This repo aims
to be the most complete pure-PyTorch reference for researchers wanting to experiment with screening.

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{nakanishi2026screening,
  title={Screening Is Enough},
  author={Nakanishi, Ken M.},
  journal={arXiv preprint arXiv:2604.01178},
  year={2026}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
