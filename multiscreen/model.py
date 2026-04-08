"""Multiscreen language model.

Implements the screening mechanism from "Screening Is Enough"
(Nakanishi, 2026; arXiv:2604.01178).

Key differences from Transformer:
- No softmax attention: uses absolute query-key relevance via screening
- No FFN: gated screening tiles replace both attention and FFN
- No layer normalization: uses TanhNorm and unit-length normalization
- Normalized + tied embeddings with learned scales
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from multiscreen.config import MultiscreenConfig


class MultiscreenModel(nn.Module):
    """Multiscreen language model.

    Architecture: Normalized Embedding -> N_L Screening Layers -> Tied Output.
    Each layer contains N_H parallel Gated Screening Tiles.
    """

    def __init__(self, config: MultiscreenConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing
        dE = config.hidden_dim

        # Raw embedding (normalized before use)
        self.embed = nn.Embedding(config.vocab_size, dE)

        # Learned scalars for input/output scaling (paper Table 3)
        self.s_E = nn.Parameter(torch.tensor(0.0))           # exp(0) = 1
        self.s_F = nn.Parameter(torch.tensor(math.log(math.sqrt(dE))))  # exp = sqrt(dE)

        # Stack of screening layers
        self.layers = nn.ModuleList([
            MultiscreenLayer(config, layer_idx=l)
            for l in range(config.num_layers)
        ])

        # Initialize embedding (paper Table 3)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.1 / math.sqrt(dE))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs.

        Returns:
            logits: (batch, seq_len, vocab_size).
        """
        # Normalize embedding to unit length
        W_norm = F.normalize(self.embed.weight, dim=-1)

        # Embed with learned input scale
        x = F.embedding(input_ids, W_norm) * self.s_E.exp()

        # Apply screening layers (with optional gradient checkpointing)
        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                x = grad_checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        # Output logits via tied normalized embedding with learned scale
        logits = F.linear(x, W_norm) * self.s_F.exp()
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiscreenLayer(nn.Module):
    """Single Multiscreen layer: residual connection around N_H screening tiles."""

    def __init__(self, config: MultiscreenConfig, layer_idx: int):
        super().__init__()
        self.block = GatedScreeningBlock(config, layer_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class GatedScreeningBlock(nn.Module):
    """All N_H gated screening tiles in one layer, batched for efficiency.

    Each tile: project -> screen -> gate -> project back.
    Replaces both attention and FFN from Transformer.
    """

    def __init__(self, config: MultiscreenConfig, layer_idx: int):
        super().__init__()
        dE = config.hidden_dim
        dK = config.key_dim
        dV = config.value_dim
        NH = config.num_heads
        NL = config.num_layers
        wth = config.mipe_threshold

        self.NH = NH
        self.dK = dK
        self.dV = dV
        self.wth = wth

        # Batched linear projections across all heads (no bias)
        self.q_proj = nn.Linear(dE, NH * dK, bias=False)
        self.k_proj = nn.Linear(dE, NH * dK, bias=False)
        self.v_proj = nn.Linear(dE, NH * dV, bias=False)
        self.g_proj = nn.Linear(dE, NH * dV, bias=False)
        self.o_proj = nn.Linear(NH * dV, dE, bias=False)

        # Per-head scalar parameters (paper Table 3)
        # sw: window parameter, linearly spaced from 0 to log(wth) per layer
        self.sw = nn.Parameter(torch.linspace(0, math.log(wth), NH))
        # sr: acceptance width, initialized to 0 -> r = exp(0) + 1 = 2
        self.sr = nn.Parameter(torch.zeros(NH))
        # sO: output scale, initialized so total contribution is ~1
        self.sO = nn.Parameter(
            torch.full((NH,), math.log(1.0 / math.sqrt(NH * NL)))
        )

        # Cached relative position tensor (filled lazily on first call)
        self.register_buffer("_rel_cache", None, persistent=False)
        self._rel_cache_T: int = 0

        # Initialize projections (paper Table 3)
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.1 / math.sqrt(dK))
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.1 / math.sqrt(dK))
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.1 / math.sqrt(dV))
        nn.init.normal_(self.g_proj.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=0.1 / math.sqrt(dE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, dE)
        Returns:
            (B, T, dE)
        """
        B, T, _ = x.shape

        # Project to Q, K, V, G for all heads
        q = self.q_proj(x).view(B, T, self.NH, self.dK)
        k = self.k_proj(x).view(B, T, self.NH, self.dK)
        v = self.v_proj(x).view(B, T, self.NH, self.dV)
        g = self.g_proj(x).view(B, T, self.NH, self.dV)

        # Screening unit
        u = self._screening(q, k, v)  # (B, T, NH, dV)

        # Gate: tanh(silu(g)) - bounded in (-1, 1)
        g_hat = torch.tanh(F.silu(g))

        # Element-wise gating
        h = u * g_hat  # (B, T, NH, dV)

        # Per-head output scaling: (1, 1, NH, 1) broadcasts over (B, T, NH, dV)
        h = h * self.sO.exp().view(1, 1, self.NH, 1)

        # Flatten heads and project to model dim
        h = h.reshape(B, T, self.NH * self.dV)
        return self.o_proj(h)

    def _screening(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Screening unit: absolute query-key relevance.

        1. Normalize Q, K, V to unit length
        2. Apply MiPE (positional encoding on first 2 dims)
        3. Compute bounded similarity s_ij in [-1, 1]
        4. Trim-and-Square: rho = max(1 - r(1-s), 0)^2
        5. Softmask: causal + distance-aware window
        6. Aggregate: h = sum_j rho_d_ij * v_j
        7. TanhNorm

        Args:
            q: (B, T, NH, dK)
            k: (B, T, NH, dK)
            v: (B, T, NH, dV)
        Returns:
            u: (B, T, NH, dV)
        """
        # 1. Normalize to unit length
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        v = F.normalize(v, dim=-1)

        # 2. Screening parameters
        w = self.sw.exp() + 1  # (NH,)  screening window
        r = self.sr.exp() + 1  # (NH,)  acceptance sharpness

        # 3. Apply MiPE to Q, K
        q, k = self._apply_mipe(q, k, w)

        # 4. Rearrange for batched matmul: (B, NH, T, d)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 5. Similarity: s_ij = q_i . k_j^T  in [-1, 1]
        sim = torch.matmul(q, k.transpose(-2, -1))  # (B, NH, T, T)

        # 6-7. Fused Trim-and-Square + Softmask (avoids separate rho allocation)
        mask = self._softmask(sim.shape[-1], w, sim.device, sim.dtype)
        rho_d = torch.clamp(
            1.0 - r.view(1, -1, 1, 1) * (1.0 - sim), min=0.0
        ).square_().mul_(mask)  # (B, NH, T, T)

        # 8. Weighted aggregation
        h = torch.matmul(rho_d, v)  # (B, NH, T, dV)

        # 9. TanhNorm: preserves direction, bounds norm by 1
        h_norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        u = (torch.tanh(h_norm) / h_norm) * h

        return u.transpose(1, 2)  # (B, T, NH, dV)

    def _apply_mipe(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Minimal Positional Encoding: RoPE-like rotation on first 2 dims.

        Active only when learned window w < wth; disabled for long-range tiles.

        Args:
            q: (B, T, NH, dK) unit-normalized
            k: (B, T, NH, dK) unit-normalized
            w: (NH,) screening window widths
        Returns:
            q_rot, k_rot: same shapes, still unit-length.
        """
        T = q.shape[1]

        # phi(w): smoothly 1 -> 0 as w -> wth, then 0 for w >= wth
        phi = torch.where(
            w < self.wth,
            0.5 * (torch.cos(math.pi * w / self.wth) + 1.0),
            torch.zeros_like(w),
        )  # (NH,)

        # Rotation angle: theta(i, w) = i * phi(w) / w
        positions = torch.arange(T, device=q.device, dtype=q.dtype)  # (T,)
        angles = positions.unsqueeze(1) * (phi / w).unsqueeze(0)     # (T, NH)

        cos_a = torch.cos(angles)  # (T, NH)
        sin_a = torch.sin(angles)  # (T, NH)

        # Rotate first 2 coordinates of Q and K (in-place copy, avoids torch.cat)
        q0, q1 = q[..., 0], q[..., 1]  # (B, T, NH)
        k0, k1 = k[..., 0], k[..., 1]

        q_rot = torch.empty_like(q)
        q_rot[..., 0] = q0 * cos_a - q1 * sin_a
        q_rot[..., 1] = q0 * sin_a + q1 * cos_a
        q_rot[..., 2:] = q[..., 2:]

        k_rot = torch.empty_like(k)
        k_rot[..., 0] = k0 * cos_a - k1 * sin_a
        k_rot[..., 1] = k0 * sin_a + k1 * cos_a
        k_rot[..., 2:] = k[..., 2:]

        return q_rot, k_rot

    def _softmask(
        self,
        T: int,
        w: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute causal distance-aware softmask.

        m_ij = 0.5(cos(pi(j-i)/w) + 1)  for -w < j-i <= 0
             = 0                         otherwise

        Args:
            T: sequence length
            w: (NH,) per-head window widths
        Returns:
            mask: (1, NH, T, T)
        """
        # Cache relative positions (constant for fixed T)
        if self._rel_cache_T != T or self._rel_cache is None:
            pos = torch.arange(T, device=device, dtype=dtype)
            self._rel_cache = (pos.unsqueeze(0) - pos.unsqueeze(1)).unsqueeze(0)
            self._rel_cache_T = T
        rel = self._rel_cache  # (1, T, T)

        w_exp = w.view(-1, 1, 1)  # (NH, 1, 1)

        # Valid region: causal (j <= i -> rel <= 0) AND within window (rel > -w)
        valid = (rel <= 0) & (rel > -w_exp)

        # Smooth cosine mask (multiply by valid instead of torch.where to avoid branching)
        mask = (0.5 * (torch.cos(math.pi * rel / w_exp) + 1.0)) * valid

        return mask.unsqueeze(0)  # (1, NH, T, T)
