"""Benchmark Multiscreen inference throughput and per-token latency.

Compares four configurations:
    - eager + full re-forward   (reprocess the whole sequence each step)
    - eager + KV cache          (O(T) per step)
    - compile + full re-forward (torch.compile, no cache)
    - compile + KV cache        (torch.compile + cache)

Runs on synthetic random IDs so no dataset is needed.

Examples:
    # Default 154M config, prompt=32, generate=128
    python scripts/bench_inference.py

    # Longer generation, with compile
    python scripts/bench_inference.py --generate 256 --compile

    # Only compare the two compile variants
    python scripts/bench_inference.py --compile --skip-eager
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from multiscreen import MultiscreenConfig, MultiscreenModel, setup_compile_env


@dataclass
class BenchResult:
    name: str
    prompt_len: int
    generate_len: int
    total_ms: float
    per_token_ms: float
    tokens_per_sec: float


@torch.no_grad()
def decode_full_reforward(
    model: MultiscreenModel,
    prompt_ids: torch.Tensor,
    generate_len: int,
    max_seq_len: int,
) -> None:
    """Full re-forward decoding: concatenate each new token to input and
    re-run the whole forward pass. O(T^2) total work."""
    input_ids = prompt_ids
    for _ in range(generate_len):
        if input_ids.shape[1] >= max_seq_len:
            break
        logits, _ = model(input_ids)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=1)


@torch.no_grad()
def decode_kv_cache(
    model: MultiscreenModel,
    prompt_ids: torch.Tensor,
    generate_len: int,
    max_seq_len: int,
) -> None:
    """KV cache decoding: prefill once, then feed one token per step."""
    logits, kv_caches = model(prompt_ids)
    next_logits = logits[:, -1, :]
    start = prompt_ids.shape[1]

    for step in range(generate_len):
        if start + step >= max_seq_len:
            break
        next_id = next_logits.argmax(dim=-1, keepdim=True)
        logits, kv_caches = model(
            next_id, start_pos=start + step, kv_caches=kv_caches,
        )
        next_logits = logits[:, -1, :]


def bench_one(
    name: str,
    model: MultiscreenModel,
    prompt_ids: torch.Tensor,
    generate_len: int,
    decode_fn,
    warmup: int,
    repeats: int,
    max_seq_len: int,
) -> BenchResult:
    # Warmup
    for _ in range(warmup):
        decode_fn(model, prompt_ids, generate_len, max_seq_len)
    torch.cuda.synchronize()

    # Timed runs
    total = 0.0
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        decode_fn(model, prompt_ids, generate_len, max_seq_len)
        torch.cuda.synchronize()
        total += (time.perf_counter() - t0) * 1000  # ms

    avg_ms = total / repeats
    per_tok = avg_ms / generate_len
    tok_s = generate_len / (avg_ms / 1000)
    return BenchResult(
        name=name,
        prompt_len=prompt_ids.shape[1],
        generate_len=generate_len,
        total_ms=avg_ms,
        per_token_ms=per_tok,
        tokens_per_sec=tok_s,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark Multiscreen inference")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=18)
    parser.add_argument("--num-heads", type=int, default=18)
    parser.add_argument("--key-dim", type=int, default=32)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--prompt", type=int, default=32,
                        help="Prompt length (prefill size)")
    parser.add_argument("--generate", type=int, default=128,
                        help="Number of tokens to generate")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compile", action="store_true",
                        help="Also benchmark torch.compile variants")
    parser.add_argument("--skip-eager", action="store_true",
                        help="Skip eager benchmarks (only run compile variants)")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: running on CPU, numbers will not be representative.")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    config = MultiscreenConfig(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        max_seq_len=args.max_seq_len,
    )

    print(f"Config: hidden_dim={config.hidden_dim}, NL={config.num_layers}, "
          f"NH={config.num_heads}, dK={config.key_dim}, dV={config.value_dim}")
    print(f"Prompt: {args.prompt} tokens, Generate: {args.generate} tokens")
    print(f"Device: {device}, dtype: {dtype}")
    print()

    model = MultiscreenModel(config).to(device=device, dtype=dtype)
    model.eval()
    n_params = model.count_parameters()
    print(f"Parameters: {n_params:,}")

    torch.manual_seed(0)
    prompt_ids = torch.randint(0, config.vocab_size, (1, args.prompt), device=device)

    results: list[BenchResult] = []

    if not args.skip_eager:
        results.append(bench_one(
            "eager + full re-forward", model, prompt_ids, args.generate,
            decode_full_reforward, args.warmup, args.repeats, args.max_seq_len,
        ))
        results.append(bench_one(
            "eager + KV cache", model, prompt_ids, args.generate,
            decode_kv_cache, args.warmup, args.repeats, args.max_seq_len,
        ))

    if args.compile:
        cl = setup_compile_env()
        if cl:
            print(f"CC: {cl}")
        import logging
        logging.disable(logging.ERROR)
        compiled = torch.compile(model, mode="default", dynamic=True)
        print("torch.compile: enabled (mode=default, dynamic=True)")
        print()

        results.append(bench_one(
            "compile + full re-forward", compiled, prompt_ids, args.generate,
            decode_full_reforward, args.warmup, args.repeats, args.max_seq_len,
        ))
        results.append(bench_one(
            "compile + KV cache", compiled, prompt_ids, args.generate,
            decode_kv_cache, args.warmup, args.repeats, args.max_seq_len,
        ))

    # Report
    print()
    print("=" * 78)
    print(f"{'Configuration':<30} {'Total (ms)':>12} {'per-token (ms)':>16} {'tok/s':>12}")
    print("-" * 78)
    for r in results:
        print(f"{r.name:<30} {r.total_ms:>12.1f} {r.per_token_ms:>16.2f} {r.tokens_per_sec:>12.0f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
