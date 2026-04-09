# Setup Guide

## Basic install

```bash
pip install -e ".[train]"
```

This installs:
- `torch >= 2.1`
- `numpy`, `tqdm`
- `transformers` and `datasets` for the training script

## Verify install (CPU only)

```bash
pytest tests/ -k "not CUDA"
```

15 tests should pass without a GPU.

## torch.compile setup (optional, big speedup)

`torch.compile` with the inductor backend gives ~2.4x training throughput on the
default 154M model. It requires both:

1. `triton` (or `triton-windows` on Windows)
2. A C compiler

### Linux

```bash
pip install -e ".[train,perf]"
# Triton ships its own dependencies. GCC is usually already installed.
```

Test:
```bash
python scripts/benchmark.py --compile --batch-size 16 --steps 30
```

### Windows

This is the gnarly path. You need:

1. **Visual Studio Build Tools 2022** (NOT 2019; cl.exe in 2019 fails on the C11 code Triton generates)
   - Download: https://aka.ms/vs/17/release/vs_BuildTools.exe
   - During install, check **"Desktop development with C++"** workload
   - Installation is several GB

2. **triton-windows**:
   ```powershell
   pip install -e ".[train,perf]"
   # This pulls triton-windows automatically
   ```

3. **Verify cl.exe is found**. The training and profiling scripts auto-detect MSVC at:
   ```
   C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\<version>\bin\Hostx64\x64\cl.exe
   ```
   You can also set `CC` manually:
   ```powershell
   $env:CC = "C:\path\to\cl.exe"
   ```

4. **Test**:
   ```powershell
   python scripts\benchmark.py --compile --batch-size 16 --steps 30
   ```
   The first run takes a couple of minutes (kernel compilation). Subsequent runs are cached.

## Known issues

### "RuntimeError: Failed to find C compiler"

You need a C compiler installed. See above.

### "out of resource: triton_mm Required: 131072 Hardware limit: 101376"

This happens with `mode='max-autotune'` on consumer GPUs (e.g. RTX 5070 Ti). Use the default
mode instead — `scripts/train.py --compile` already does this:

```python
torch.compile(model, mode="default")  # not "max-autotune"
```

### "Missing key(s) in state_dict: \"_orig_mod...\""

`torch.compile` wraps the model in `OptimizedModule` which adds `_orig_mod.` prefix to
state dict keys. The `Trainer` class in `multiscreen/trainer.py` handles this transparently,
but if you compile *before* loading a checkpoint you'll hit this. The training script
applies `--compile` *after* `--resume` for this reason.

### Power management / scheduler crashes

If you run a heavy `torch.compile` benchmark on a system that's also training another model,
the GPU/system memory pressure can crash both jobs. We learned this the hard way. Don't run
two GPU jobs concurrently on a single 16GB card.
