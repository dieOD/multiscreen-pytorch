"""Multiscreen model configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace


@dataclass
class MultiscreenConfig:
    """Configuration for the Multiscreen language model.

    Following "Screening Is Enough" (Nakanishi, 2026; arXiv:2604.01178).

    The supraparameter Psi (Ψ) controls model scale: N_L = N_H = Ψ, d_E = Ψ².
    Use ``MultiscreenConfig.from_psi(psi=8)`` for paper-comparable scaling.
    """

    # Vocabulary
    vocab_size: int = 50257  # GPT-2 default

    # Architecture
    hidden_dim: int = 256       # d_E (embedding dimension)
    num_layers: int = 8         # N_L (number of layers)
    num_heads: int = 8          # N_H (number of screening tiles per layer)
    key_dim: int = 16           # d_K (query/key projection dim)
    value_dim: int = 64         # d_V (value/gate projection dim)

    # Sequence length
    max_seq_len: int = 256

    # MiPE: minimal positional encoding active when window < threshold
    mipe_threshold: float = 256.0

    # Training optimization
    gradient_checkpointing: bool = False

    @classmethod
    def from_psi(cls, psi: int, vocab_size: int = 50257, max_seq_len: int = 256, **overrides) -> "MultiscreenConfig":
        """Build a config using the supraparameter Psi.

        From paper Table 1: N_L = N_H = Psi, d_E = Psi².
        """
        return cls(
            vocab_size=vocab_size,
            hidden_dim=psi * psi,
            num_layers=psi,
            num_heads=psi,
            max_seq_len=max_seq_len,
            **overrides,
        )

    def __post_init__(self):
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.key_dim <= 0:
            raise ValueError("key_dim must be positive")
        if self.value_dim <= 0:
            raise ValueError("value_dim must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.mipe_threshold <= 0:
            raise ValueError("mipe_threshold must be positive")

    def clone(self, **updates) -> "MultiscreenConfig":
        return replace(self, **updates)

    @property
    def num_params_estimate(self) -> int:
        """Approximate parameter count (tied input/output embedding)."""
        embed = self.vocab_size * self.hidden_dim
        per_tile = self.hidden_dim * (2 * self.key_dim + 3 * self.value_dim)
        total_tiles = self.num_layers * self.num_heads
        return embed + total_tiles * per_tile
