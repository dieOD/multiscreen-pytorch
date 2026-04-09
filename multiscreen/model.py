"""Multiscreen language model.

Implements the screening mechanism from "Screening Is Enough"
(Nakanishi, 2026; arXiv:2604.01178).

Key differences from Transformer:
- No softmax attention: uses absolute query-key relevance via screening
- No FFN: gated screening tiles replace both attention and FFN
- No layer normalization: uses TanhNorm and unit-length normalization
- Normalized + tied embeddings with learned scales

KV cache: for incremental decode, K is cached **after MiPE rotation and unit
normalization**, V is cached **after unit normalization**. The causal/window
softmask is recomputed per step from absolute query positions.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from multiscreen.config import MultiscreenConfig


# Per-layer screening cache: (K, V)
#   K: (B, NH, T_cached, dK) post-MiPE, unit-normalized
#   V: (B, NH, T_cached, dV) unit-normalized
ScreeningCache = tuple[torch.Tensor, torch.Tensor]


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

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_caches: Optional[list[ScreeningCache]] = None,
    ) -> tuple[torch.Tensor, list[ScreeningCache]]:
        """Forward pass.

        Args:
            input_ids: (B, T_new) token IDs. T_new is the full prompt length at
                prefill, 1 (or more) at incremental decode.
            start_pos: absolute position of ``input_ids[:, 0]`` in the full
                sequence. Use 0 for prefill / training.
            kv_caches: per-layer screening caches from the previous step. None
                for training or the first prefill call.

        Returns:
            logits: (B, T_new, vocab_size).
            new_kv_caches: per-layer caches for the next decode step. Empty
                list during training (``self.training == True``), populated
                during eval/inference.
        """
        # Normalize embedding to unit length
        W_norm = F.normalize(self.embed.weight, dim=-1)

        # Embed with learned input scale
        x = F.embedding(input_ids, W_norm) * self.s_E.exp()

        use_cache = not self.training
        new_kv_caches: list[ScreeningCache] = []

        for i, layer in enumerate(self.layers):
            past = kv_caches[i] if kv_caches is not None else None

            if self.training and self.gradient_checkpointing:
                # Checkpoint path: no cache (saves activation memory).
                # Layer returns (x_new, None); keep only x_new.
                x, _ = grad_checkpoint(layer, x, use_reentrant=False)
            else:
                x, new_kv = layer(x, start_pos, past, use_cache)
                if use_cache:
                    new_kv_caches.append(new_kv)

        # Output logits via tied normalized embedding with learned scale
        logits = F.linear(x, W_norm) * self.s_F.exp()

        return logits, new_kv_caches

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiscreenLayer(nn.Module):
    """Single Multiscreen layer: residual connection around N_H screening tiles."""

    def __init__(self, config: MultiscreenConfig, layer_idx: int):
        super().__init__()
        self.block = GatedScreeningBlock(config, layer_idx)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        past_kv: Optional[ScreeningCache] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[ScreeningCache]]:
        block_out, new_kv = self.block(x, start_pos, past_kv, use_cache)
        return x + block_out, new_kv


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

        # Initialize projections (paper Table 3)
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.1 / math.sqrt(dK))
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.1 / math.sqrt(dK))
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.1 / math.sqrt(dV))
        nn.init.normal_(self.g_proj.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=0.1 / math.sqrt(dE))

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        past_kv: Optional[ScreeningCache] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[ScreeningCache]]:
        """
        Args:
            x: (B, T_new, dE)
            start_pos: absolute position of ``x[:, 0]``
            past_kv: optional (past_k, past_v) - cached post-MiPE K, normalized V
            use_cache: if True, return updated cache for the next step

        Returns:
            out: (B, T_new, dE)
            new_kv: (full_k, full_v) if use_cache else None
        """
        B, T_new, _ = x.shape

        # Project to Q, K, V, G for all heads
        q = self.q_proj(x).view(B, T_new, self.NH, self.dK)
        k_new = self.k_proj(x).view(B, T_new, self.NH, self.dK)
        v_new = self.v_proj(x).view(B, T_new, self.NH, self.dV)
        g = self.g_proj(x).view(B, T_new, self.NH, self.dV)

        # Screening unit (with optional cache)
        u, new_kv = self._screening(q, k_new, v_new, start_pos, past_kv, use_cache)

        # Gate: tanh(silu(g)) - bounded in (-1, 1)
        g_hat = torch.tanh(F.silu(g))

        # Element-wise gating
        h = u * g_hat  # (B, T_new, NH, dV)

        # Per-head output scaling: (1, 1, NH, 1) broadcasts over (B, T_new, NH, dV)
        h = h * self.sO.exp().view(1, 1, self.NH, 1)

        # Flatten heads and project to model dim
        h = h.reshape(B, T_new, self.NH * self.dV)
        return self.o_proj(h), new_kv

    def _screening(
        self,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        start_pos: int = 0,
        past_kv: Optional[ScreeningCache] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[ScreeningCache]]:
        """Screening unit with KV cache support.

        Steps:
            1. Normalize Q, K_new, V_new to unit length
            2. MiPE on Q and K_new with absolute positions (start_pos offset)
            3. Concat with past K, V if provided
            4. Similarity Q @ K_full^T  -> (B, NH, T_new, T_total)
            5. Softmask between new queries and full keys
            6. Trim-and-Square fused with mask
            7. Aggregation rho @ V_full -> (B, NH, T_new, dV)
            8. TanhNorm

        Args:
            q, k_new, v_new: (B, T_new, NH, d)
            start_pos: absolute position of ``q[:, 0]``
            past_kv: cached (post-MiPE K, normalized V) for positions [0, start_pos)
            use_cache: if True, return the concatenated cache for the next step

        Returns:
            u: (B, T_new, NH, dV)
            new_kv: (full_k, full_v) if use_cache else None
        """
        # 1. Normalize to unit length
        q = F.normalize(q, dim=-1)
        k_new = F.normalize(k_new, dim=-1)
        v_new = F.normalize(v_new, dim=-1)

        # 2. Screening parameters
        w = self.sw.exp() + 1  # (NH,)  screening window
        r = self.sr.exp() + 1  # (NH,)  acceptance sharpness

        # 3. Apply MiPE to Q and new K with absolute positions
        q, k_new = self._apply_mipe(q, k_new, w, start_pos)

        # 4. Rearrange for batched matmul: (B, NH, T_new, d)
        q = q.transpose(1, 2)
        k_new = k_new.transpose(1, 2)
        v_new = v_new.transpose(1, 2)

        # 5. Concatenate new K/V with past cache
        if past_kv is not None:
            past_k, past_v = past_kv
            full_k = torch.cat([past_k, k_new], dim=2)  # (B, NH, T_total, dK)
            full_v = torch.cat([past_v, v_new], dim=2)  # (B, NH, T_total, dV)
        else:
            full_k = k_new
            full_v = v_new

        T_new = q.shape[2]
        T_total = full_k.shape[2]

        # 6. Similarity: s_ij = q_i . k_j^T  in [-1, 1]
        sim = torch.matmul(q, full_k.transpose(-2, -1))  # (B, NH, T_new, T_total)

        # 7-8. Fused Trim-and-Square + Softmask
        mask = self._softmask(T_new, T_total, start_pos, w, sim.device, sim.dtype)
        rho_d = torch.clamp(
            1.0 - r.view(1, -1, 1, 1) * (1.0 - sim), min=0.0
        ).square_().mul_(mask)  # (B, NH, T_new, T_total)

        # 9. Weighted aggregation
        h = torch.matmul(rho_d, full_v)  # (B, NH, T_new, dV)

        # 10. TanhNorm: preserves direction, bounds norm by 1
        h_norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        u = (torch.tanh(h_norm) / h_norm) * h

        new_kv = (full_k, full_v) if use_cache else None
        return u.transpose(1, 2), new_kv  # u: (B, T_new, NH, dV)

    def _apply_mipe(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Minimal Positional Encoding: RoPE-like rotation on first 2 dims.

        Active only when learned window w < wth; disabled for long-range tiles.

        Args:
            q: (B, T_new, NH, dK) unit-normalized
            k: (B, T_new, NH, dK) unit-normalized
            w: (NH,) screening window widths
            start_pos: absolute position of ``q[:, 0]`` (for incremental decode)
        Returns:
            q_rot, k_rot: same shapes, still unit-length.
        """
        T_new = q.shape[1]

        # phi(w): smoothly 1 -> 0 as w -> wth, then 0 for w >= wth
        phi = torch.where(
            w < self.wth,
            0.5 * (torch.cos(math.pi * w / self.wth) + 1.0),
            torch.zeros_like(w),
        )  # (NH,)

        # Rotation angle: theta(i, w) = (start_pos + i) * phi(w) / w
        positions = torch.arange(
            start_pos, start_pos + T_new, device=q.device, dtype=q.dtype
        )  # (T_new,)
        angles = positions.unsqueeze(1) * (phi / w).unsqueeze(0)  # (T_new, NH)

        cos_a = torch.cos(angles)  # (T_new, NH)
        sin_a = torch.sin(angles)  # (T_new, NH)

        # Rotate first 2 coordinates of Q and K (index copy avoids torch.cat)
        q0, q1 = q[..., 0], q[..., 1]  # (B, T_new, NH)
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
        T_new: int,
        T_total: int,
        start_pos: int,
        w: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Causal distance-aware softmask between new queries and full keys.

        Query positions: [start_pos, start_pos + T_new)
        Key positions:   [0, T_total)

        rel[i, j] = k_pos[j] - q_pos[i]
        m[h, i, j] = 0.5(cos(pi * rel / w_h) + 1)  for  -w_h < rel <= 0
                   = 0                             otherwise

        Args:
            T_new: number of new queries
            T_total: total number of keys (past + new)
            start_pos: absolute position of the first new query
            w: (NH,) per-head window widths
        Returns:
            mask: (1, NH, T_new, T_total)
        """
        q_pos = torch.arange(
            start_pos, start_pos + T_new, device=device, dtype=dtype
        )
        k_pos = torch.arange(T_total, device=device, dtype=dtype)
        rel = k_pos.unsqueeze(0) - q_pos.unsqueeze(1)  # (T_new, T_total)

        w_exp = w.view(-1, 1, 1)  # (NH, 1, 1)

        # Valid region: causal (rel <= 0) AND within window (rel > -w)
        valid = (rel <= 0) & (rel > -w_exp)

        # Smooth cosine mask (multiply by valid to avoid branching)
        mask = (0.5 * (torch.cos(math.pi * rel / w_exp) + 1.0)) * valid

        return mask.unsqueeze(0)  # (1, NH, T_new, T_total)
