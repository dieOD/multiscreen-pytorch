"""Train a Multiscreen language model.

Quick start (TinyStories, GPT-2 tokenizer):
    python scripts/train.py --dataset roneneldan/TinyStories --psi 8 --max-steps 1000

Wikitext-2:
    python scripts/train.py --dataset wikitext --config wikitext-2-raw-v1 --psi 8

Custom config:
    python scripts/train.py --dataset wikitext --config wikitext-103-raw-v1 \\
        --hidden-dim 1024 --num-layers 18 --num-heads 18 \\
        --key-dim 32 --value-dim 128 --seq-len 256 \\
        --max-steps 17000 --peak-lr 1e-2 --compile
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from multiscreen import MultiscreenConfig, MultiscreenModel
from multiscreen.data import PackedTextDataset
from multiscreen.trainer import Trainer, TrainConfig


def find_msvc_cl() -> str | None:
    """Find MSVC cl.exe for Triton/torch.compile on Windows."""
    if os.environ.get("CC"):
        return os.environ["CC"]
    bases = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio"),
        Path(r"C:\Program Files\Microsoft Visual Studio"),
    ]
    for base in bases:
        if not base.exists():
            continue
        for vs in sorted(base.iterdir(), reverse=True):
            pattern = str(vs / "BuildTools" / "VC" / "Tools" / "MSVC" / "*" / "bin" / "Hostx64" / "x64" / "cl.exe")
            matches = sorted(glob.glob(pattern))
            if matches:
                return matches[-1]
    return None


def main():
    parser = argparse.ArgumentParser(description="Train a Multiscreen language model")

    # Dataset
    parser.add_argument("--dataset", default="roneneldan/TinyStories",
                        help="HuggingFace dataset name")
    parser.add_argument("--config", default=None, help="Dataset subconfig name (e.g. wikitext-2-raw-v1)")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer", default="gpt2", help="HuggingFace tokenizer name")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Cap dataset size for quick experiments")
    parser.add_argument("--eval-split", default=None,
                        help="Eval split name (e.g. validation, test). Optional.")

    # Model: either --psi or explicit dims
    parser.add_argument("--psi", type=int, default=None,
                        help="Supraparameter Psi (N_L=N_H=Psi, d_E=Psi^2). Overrides hidden-dim/num-layers/num-heads.")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--key-dim", type=int, default=16)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--mipe-threshold", type=float, default=256.0)

    # Training
    parser.add_argument("--peak-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))

    # Optimizations
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Trade compute for VRAM (~75%% reduction)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for ~2.6x speedup (requires triton + C compiler)")

    # Other
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--weights-only", action="store_true",
                        help="Load only model weights from checkpoint (reset step/optimizer)")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Build config
    if args.psi is not None:
        config = MultiscreenConfig.from_psi(
            psi=args.psi,
            vocab_size=tokenizer.vocab_size,
            max_seq_len=args.seq_len,
            key_dim=args.key_dim,
            value_dim=args.value_dim,
            mipe_threshold=args.mipe_threshold,
            gradient_checkpointing=args.gradient_checkpointing,
        )
    else:
        config = MultiscreenConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            key_dim=args.key_dim,
            value_dim=args.value_dim,
            max_seq_len=args.seq_len,
            mipe_threshold=args.mipe_threshold,
            gradient_checkpointing=args.gradient_checkpointing,
        )

    print(f"Config: hidden_dim={config.hidden_dim}, layers={config.num_layers}, heads={config.num_heads}")
    print(f"Estimated parameters: {config.num_params_estimate:,}")

    # Datasets
    print(f"Loading dataset: {args.dataset}")
    train_dataset = PackedTextDataset.from_hf_dataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        split="train",
        text_column=args.text_column,
        config_name=args.config,
        max_tokens=args.max_tokens,
    )
    print(f"Train: {len(train_dataset):,} chunks")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    eval_loader = None
    if args.eval_split:
        eval_dataset = PackedTextDataset.from_hf_dataset(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            split=args.eval_split,
            text_column=args.text_column,
            config_name=args.config,
        )
        eval_loader = DataLoader(
            eval_dataset, batch_size=args.micro_batch,
            shuffle=False, num_workers=args.num_workers,
            pin_memory=True, drop_last=True,
        )
        print(f"Eval: {len(eval_dataset):,} chunks")

    # Build model
    model = MultiscreenModel(config)
    print(f"Actual parameters: {model.count_parameters():,}")

    # Setup CC env for compile (auto-detect MSVC on Windows)
    if args.compile and not os.environ.get("CC"):
        cl_path = find_msvc_cl()
        if cl_path:
            os.environ["CC"] = cl_path
            print(f"CC auto-detected: {cl_path}")

    # Trainer
    train_config = TrainConfig(
        peak_lr=args.peak_lr,
        min_lr=args.min_lr,
        warmup_ratio=args.warmup_ratio,
        micro_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=args.device,
        dtype=args.dtype,
    )
    trainer = Trainer(model, train_loader, train_config, eval_loader=eval_loader)

    # Resume BEFORE compile (avoids _orig_mod prefix in state_dict keys)
    if args.resume:
        trainer.load_checkpoint(args.resume, weights_only=args.weights_only)

    if args.compile:
        try:
            trainer.model = torch.compile(trainer.model, mode="default")
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile failed: {e}, continuing without compilation")

    trainer.train()


if __name__ == "__main__":
    main()
