"""Benchmark Multiscreen training throughput and VRAM.

Examples:
    python scripts/benchmark.py
    python scripts/benchmark.py --compile --batch-size 32
    python scripts/benchmark.py --psi 12 --batch-size 16 --trace

Note: this file is named ``benchmark.py`` (not ``profile.py``) because ``profile``
collides with Python's standard library, which breaks ``cProfile`` imports when
run as a script.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from multiscreen import MultiscreenConfig, MultiscreenModel, setup_compile_env


def make_dummy_batch(B, T, V, device):
    return (
        torch.randint(0, V, (B, T), device=device),
        torch.randint(0, V, (B, T), device=device),
        torch.ones(B, T, device=device),
    )


def profile_throughput(model, B, T, V, warmup, steps, device, dtype):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for _ in range(warmup):
        ids, lab, mask = make_dummy_batch(B, T, V, device)
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=dtype):
            logits, _ = model(ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, V), lab.view(-1), reduction="none"
            )
            loss = (loss * mask.view(-1)).sum() / mask.sum()
        loss.backward()
        opt.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    total_tokens = 0
    t0 = time.perf_counter()
    for _ in range(steps):
        ids, lab, mask = make_dummy_batch(B, T, V, device)
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=dtype):
            logits, _ = model(ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, V), lab.view(-1), reduction="none"
            )
            loss = (loss * mask.view(-1)).sum() / mask.sum()
        loss.backward()
        opt.step()
        total_tokens += B * T
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return total_tokens / elapsed, peak, elapsed


def profile_with_trace(model, B, T, V, device, dtype, steps=5):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(3):
        ids, lab, mask = make_dummy_batch(B, T, V, device)
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=dtype):
            logits, _ = model(ids)
            loss = nn.functional.cross_entropy(logits.view(-1, V), lab.view(-1))
        loss.backward()
        opt.step()

    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        for _ in range(steps):
            ids, lab, mask = make_dummy_batch(B, T, V, device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=dtype):
                logits, _ = model(ids)
                loss = nn.functional.cross_entropy(logits.view(-1, V), lab.view(-1))
            loss.backward()
            opt.step()

    torch.cuda.synchronize()
    print("\n=== CUDA Time by Kernel ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    trace_path = Path("logs/profile_trace.json")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))
    print(f"\nChrome trace exported: {trace_path}")


def main():
    parser = argparse.ArgumentParser(description="Profile Multiscreen training")
    parser.add_argument("--psi", type=int, default=None,
                        help="Use Psi scaling (otherwise default 154M-style config)")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=18)
    parser.add_argument("--num-heads", type=int, default=18)
    parser.add_argument("--key-dim", type=int, default=32)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--trace", action="store_true", help="Export Chrome trace")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    if args.psi:
        config = MultiscreenConfig.from_psi(
            psi=args.psi, vocab_size=args.vocab_size, max_seq_len=args.seq_len,
            key_dim=args.key_dim, value_dim=args.value_dim,
            gradient_checkpointing=args.gradient_checkpointing,
        )
    else:
        config = MultiscreenConfig(
            vocab_size=args.vocab_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            key_dim=args.key_dim,
            value_dim=args.value_dim,
            max_seq_len=args.seq_len,
            gradient_checkpointing=args.gradient_checkpointing,
        )

    model = MultiscreenModel(config).to(device)

    if args.compile:
        cl = setup_compile_env()
        if cl:
            print(f"CC: {cl}")
        import logging
        logging.disable(logging.ERROR)
        model = torch.compile(model, mode="default")
        print("torch.compile: enabled")

    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    n_params = raw.count_parameters()

    print(f"Parameters: {n_params:,}")
    print(f"B={args.batch_size}, T={args.seq_len}, dtype={dtype}")
    print()

    if args.trace:
        profile_with_trace(model, args.batch_size, args.seq_len, args.vocab_size, device, dtype)
    else:
        tok_s, peak, elapsed = profile_throughput(
            model, args.batch_size, args.seq_len, args.vocab_size,
            args.warmup, args.steps, device, dtype,
        )
        print("=" * 50)
        print(f"  Throughput: {tok_s:>10,.0f} tok/s")
        print(f"  Peak VRAM:  {peak:>10,.0f} MB")
        print(f"  Time:       {elapsed:>10.1f} s ({args.steps} steps)")
        print("=" * 50)


if __name__ == "__main__":
    main()
