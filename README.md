# multiscreen-pytorch

A PyTorch implementation of **Multiscreen**, the screening-based language model architecture from
["Screening Is Enough"](https://arxiv.org/abs/2604.01178) (Nakanishi, 2026).

Multiscreen replaces softmax attention with **screening**: each key is evaluated independently
against a learned threshold, removing global competition between keys. This enables absolute
query-key relevance, stable training at large learning rates, and strong long-context retrieval.

## Highlights

- **Pure PyTorch reference implementation** of the Multiscreen model
- **2.6x faster training** than naive PyTorch via `torch.compile` (~41k tok/s on RTX 5070 Ti for a 154M model)
- **Generic training script** using HuggingFace `datasets` + `tokenizers`
- **Gradient checkpointing** for low-VRAM training (-75% VRAM)
- **CPU-friendly tests** (15 unit tests, no GPU required)

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

## Profiling

Benchmark throughput and VRAM:

```bash
# Default 154M config, B=16
python scripts/benchmark.py

# With torch.compile (~2.6x faster)
python scripts/benchmark.py --compile --batch-size 32

# Export Chrome trace for kernel-level inspection
python scripts/benchmark.py --trace
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
│   ├── config.py       # MultiscreenConfig
│   ├── model.py        # MultiscreenModel + GatedScreeningBlock
│   ├── data.py         # PackedTextDataset (HF datasets loader)
│   └── trainer.py      # Trainer with AMP, grad accum, checkpointing
├── scripts/
│   ├── train.py        # End-to-end training script
│   └── benchmark.py    # Throughput / VRAM benchmark
├── tests/
│   └── test_model.py   # 15 unit tests (CPU-only by default)
└── docs/
    ├── architecture.md
    ├── setup.md
    └── speedup.md
```

## Optimizations applied

The default model implementation includes several optimizations beyond the naive paper transcription:

| Optimization | What changed | Speedup |
|--------------|--------------|---------|
| Softmask cache | Cache `rel` tensor (constant for fixed T), drop `torch.where` | ~3-5% |
| MiPE in-place rotation | Replace `torch.cat` with index assignment | ~2-3% |
| Fused trim-square-mask | Reduce T x T intermediates from 3 to 2 via in-place ops | ~10-15% |
| `torch.compile` | inductor backend fuses element-wise ops | **2.4x** |
| Gradient checkpointing | Trade compute for VRAM (-75% VRAM, enables larger batch) | (compute-bound) |

See [docs/speedup.md](docs/speedup.md) for the full optimization journey, including a CUDA-time profile.

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
