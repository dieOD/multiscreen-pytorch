"""Tests for Multiscreen model. CPU-only by default."""

import math

import pytest
import torch
import torch.nn as nn

from multiscreen import MultiscreenConfig, MultiscreenModel
from multiscreen.model import GatedScreeningBlock


SMALL = MultiscreenConfig(
    vocab_size=100,
    hidden_dim=64,
    num_layers=2,
    num_heads=4,
    key_dim=16,
    value_dim=32,
    max_seq_len=32,
    mipe_threshold=32.0,
)


def _make_block(config=SMALL) -> GatedScreeningBlock:
    return GatedScreeningBlock(config, layer_idx=0)


class TestConfig:
    def test_psi_scaling(self):
        c = MultiscreenConfig.from_psi(psi=8, vocab_size=100, max_seq_len=128)
        assert c.num_layers == 8
        assert c.num_heads == 8
        assert c.hidden_dim == 64

    def test_validation(self):
        with pytest.raises(ValueError):
            MultiscreenConfig(hidden_dim=0)
        with pytest.raises(ValueError):
            MultiscreenConfig(key_dim=0)
        with pytest.raises(ValueError):
            MultiscreenConfig(num_layers=-1)

    def test_param_estimate_close_to_actual(self):
        config = SMALL
        model = MultiscreenModel(config)
        actual = model.count_parameters()
        estimate = config.num_params_estimate
        # Estimate excludes scalar params (s_E, s_F, sw, sr, sO), so it's slightly lower
        # but they're tiny. Should be within 1% for non-trivial models.
        rel_diff = abs(actual - estimate) / actual
        assert rel_diff < 0.05, f"actual={actual}, estimate={estimate}"


class TestForwardBackward:
    def test_forward_shape(self):
        model = MultiscreenModel(SMALL)
        x = torch.randint(0, 100, (2, 32))
        logits = model(x)
        assert logits.shape == (2, 32, 100)

    def test_backward_all_grads(self):
        model = MultiscreenModel(SMALL)
        x = torch.randint(0, 100, (2, 32))
        logits = model(x)
        logits.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_sr_gradient_nonzero(self):
        """sr (acceptance width) must receive a gradient via trim-and-square."""
        model = MultiscreenModel(SMALL)
        x = torch.randint(0, 100, (2, 32))
        logits = model(x)
        logits.sum().backward()
        for name, p in model.named_parameters():
            if "sr" in name:
                assert p.grad.abs().sum() > 0, f"sr gradient is zero for {name}"


class TestGradientCheckpointing:
    def test_output_equivalence(self):
        torch.manual_seed(42)
        model_ref = MultiscreenModel(SMALL)
        model_ckpt = MultiscreenModel(SMALL.clone(gradient_checkpointing=True))
        model_ckpt.load_state_dict(model_ref.state_dict())

        x = torch.randint(0, 100, (2, 32))
        model_ref.train()
        model_ckpt.train()
        out_ref = model_ref(x)
        out_ckpt = model_ckpt(x)
        assert torch.allclose(out_ref, out_ckpt, atol=1e-6)

    def test_gradient_equivalence(self):
        torch.manual_seed(42)
        model_ref = MultiscreenModel(SMALL)
        model_ckpt = MultiscreenModel(SMALL.clone(gradient_checkpointing=True))
        model_ckpt.load_state_dict(model_ref.state_dict())

        x = torch.randint(0, 100, (2, 32))
        model_ref.train()
        model_ckpt.train()

        model_ref(x).sum().backward()
        model_ckpt(x).sum().backward()

        for (n_r, p_r), (n_c, p_c) in zip(
            model_ref.named_parameters(), model_ckpt.named_parameters()
        ):
            assert torch.allclose(p_r.grad, p_c.grad, atol=1e-5), \
                f"Grad mismatch at {n_r}"


class TestSoftmask:
    def test_causal_mask(self):
        block = _make_block()
        w = block.sw.exp() + 1
        mask = block._softmask(8, w, torch.device("cpu"), torch.float32)
        for h in range(SMALL.num_heads):
            for i in range(8):
                for j in range(i + 1, 8):
                    assert mask[0, h, i, j] == 0

    def test_rel_cache_reuse(self):
        block = _make_block()
        w = block.sw.exp() + 1
        _ = block._softmask(8, w, torch.device("cpu"), torch.float32)
        cached = block._rel_cache
        _ = block._softmask(8, w, torch.device("cpu"), torch.float32)
        assert block._rel_cache is cached

    def test_rel_cache_invalidation(self):
        block = _make_block()
        w = block.sw.exp() + 1
        _ = block._softmask(8, w, torch.device("cpu"), torch.float32)
        old = block._rel_cache
        _ = block._softmask(16, w, torch.device("cpu"), torch.float32)
        assert block._rel_cache is not old
        assert block._rel_cache_T == 16

    def test_softmask_in_range(self):
        block = _make_block()
        w = block.sw.exp() + 1
        mask = block._softmask(32, w, torch.device("cpu"), torch.float32)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0 + 1e-6


class TestMiPE:
    def test_unit_length_preserved(self):
        block = _make_block()
        B, T, NH, dK = 2, 32, SMALL.num_heads, SMALL.key_dim
        q = nn.functional.normalize(torch.randn(B, T, NH, dK), dim=-1)
        k = nn.functional.normalize(torch.randn(B, T, NH, dK), dim=-1)
        w = block.sw.exp() + 1
        q_rot, k_rot = block._apply_mipe(q, k, w)
        assert torch.allclose(q_rot.norm(dim=-1), torch.ones(B, T, NH), atol=1e-5)
        assert torch.allclose(k_rot.norm(dim=-1), torch.ones(B, T, NH), atol=1e-5)


class TestScreening:
    def test_output_shape(self):
        block = _make_block()
        B, T = 2, 32
        q = torch.randn(B, T, SMALL.num_heads, SMALL.key_dim)
        k = torch.randn(B, T, SMALL.num_heads, SMALL.key_dim)
        v = torch.randn(B, T, SMALL.num_heads, SMALL.value_dim)
        u = block._screening(q, k, v)
        assert u.shape == (B, T, SMALL.num_heads, SMALL.value_dim)

    def test_tanhnorm_bounds(self):
        block = _make_block()
        B, T = 2, 32
        q = torch.randn(B, T, SMALL.num_heads, SMALL.key_dim)
        k = torch.randn(B, T, SMALL.num_heads, SMALL.key_dim)
        v = torch.randn(B, T, SMALL.num_heads, SMALL.value_dim)
        u = block._screening(q, k, v)
        assert u.norm(dim=-1).max() <= 1.0 + 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCUDA:
    """GPU smoke tests. Skipped on CPU-only systems."""

    def test_forward_backward_cuda(self):
        model = MultiscreenModel(SMALL).cuda()
        x = torch.randint(0, 100, (2, 32), device="cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        logits.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None
