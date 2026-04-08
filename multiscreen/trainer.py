"""Generic training loop for Multiscreen models."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def cosine_with_warmup(step: int, max_steps: int, peak_lr: float, min_lr: float, warmup_steps: int) -> float:
    """Cosine schedule with linear warmup."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Optimization
    peak_lr: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 0.0   # paper recommends 0 for Multiscreen
    warmup_ratio: float = 0.02
    max_grad_norm: float = 0.0  # paper recommends no clipping for Multiscreen

    # Batch
    micro_batch_size: int = 16
    gradient_accumulation_steps: int = 8

    # Training
    max_steps: int = 10000
    log_interval: int = 10
    checkpoint_interval: int = 1000
    eval_interval: int = 500

    # Paths
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))

    # Device
    device: str = "cuda"
    dtype: str = "bfloat16"

    @property
    def warmup_steps(self) -> int:
        return int(self.max_steps * self.warmup_ratio)


def _unwrap(model: nn.Module) -> nn.Module:
    """Strip torch.compile's _orig_mod wrapper if present."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


class Trainer:
    """Training loop for Multiscreen models.

    Supports mixed precision (bf16), gradient accumulation, gradient checkpointing,
    and torch.compile. Saves checkpoints with optimizer state.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: TrainConfig,
        eval_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config

        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        self.model = self.model.to(self.device)

        # Split params: no weight decay on biases and norms
        decay_params, no_decay_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "norm" in name or "bias" in name or p.ndim < 2:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config.peak_lr,
            betas=(0.9, 0.95),
        )

        self.scaler = torch.amp.GradScaler("cuda", enabled=(config.dtype != "float32"))

        self.step = 0
        self.tokens_processed = 0

        config.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = config.log_dir / "train_log.csv"
        self._log_file = open(self._log_path, "a", newline="", encoding="utf-8")
        self._log_writer = csv.writer(self._log_file)
        if self._log_path.stat().st_size == 0:
            self._log_writer.writerow(["step", "loss", "lr", "tok_s", "tokens", "eval_loss"])
            self._log_file.flush()

    def _get_lr(self) -> float:
        return cosine_with_warmup(
            self.step, self.config.max_steps,
            self.config.peak_lr, self.config.min_lr, self.config.warmup_steps,
        )

    def _set_lr(self, lr: float):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def train(self):
        self.model.train()
        train_iter = iter(self.train_loader)
        accumulated_loss = 0.0
        start_time = time.time()

        while self.step < self.config.max_steps:
            self._set_lr(self._get_lr())
            self.optimizer.zero_grad()

            for _ in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                with torch.amp.autocast("cuda", dtype=self.dtype):
                    logits = self.model(input_ids)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        reduction="none",
                    )
                    if attention_mask is not None:
                        loss = (loss * attention_mask.view(-1).float()).sum()
                        denom = attention_mask.sum().clamp(min=1)
                    else:
                        loss = loss.sum()
                        denom = torch.tensor(loss.numel(), device=self.device)
                    loss = loss / denom
                    loss = loss / self.config.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
                accumulated_loss += loss.item()
                tokens_in_batch = (
                    attention_mask.sum().item() if attention_mask is not None
                    else input_ids.numel()
                )
                self.tokens_processed += tokens_in_batch

            self.scaler.unscale_(self.optimizer)
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.step += 1

            if self.step % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                tok_s = self.tokens_processed / elapsed
                lr = self._get_lr()
                print(
                    f"step {self.step:>6d} | loss {accumulated_loss:.4f} | "
                    f"lr {lr:.2e} | tok/s {tok_s:.0f} | tokens {self.tokens_processed:,}"
                )
                self._log_writer.writerow([
                    self.step, f"{accumulated_loss:.6f}", f"{lr:.2e}",
                    f"{tok_s:.0f}", self.tokens_processed, "",
                ])
                self._log_file.flush()

            if self.step % self.config.checkpoint_interval == 0:
                self.save_checkpoint(self.config.checkpoint_dir / f"step_{self.step}.pt", accumulated_loss)

            if self.eval_loader and self.step % self.config.eval_interval == 0:
                eval_loss = self.evaluate()
                print(f"step {self.step:>6d} | eval_loss {eval_loss:.4f}")
                self._log_writer.writerow([self.step, "", "", "", "", f"{eval_loss:.6f}"])
                self._log_file.flush()
                self.model.train()

            accumulated_loss = 0.0

        self.save_checkpoint(self.config.checkpoint_dir / "final.pt", accumulated_loss)
        print("Training complete.")

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss, total_tokens = 0.0, 0
        for batch in self.eval_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            with torch.amp.autocast("cuda", dtype=self.dtype):
                logits = self.model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction="none",
                )
                if attention_mask is not None:
                    loss = (loss * attention_mask.view(-1).float()).sum()
                    n = attention_mask.sum().item()
                else:
                    loss = loss.sum()
                    n = labels.numel()

            total_loss += loss.item()
            total_tokens += n

        return total_loss / max(total_tokens, 1)

    def save_checkpoint(self, path: Path, loss: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_model = _unwrap(self.model)
        torch.save({
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.step,
            "loss": loss,
            "config": raw_model.config if hasattr(raw_model, "config") else None,
        }, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: Path, weights_only: bool = False):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        raw_model = _unwrap(self.model)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        if not weights_only and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.step = ckpt.get("step", 0)
        print(f"Loaded checkpoint: {path} (step {self.step})")
