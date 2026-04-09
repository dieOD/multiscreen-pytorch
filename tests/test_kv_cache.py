"""KV cache correctness tests for Multiscreen.

Verifies that incremental decode using the per-layer screening cache produces
numerically identical logits to a full forward pass on the entire sequence.

CPU-only by default; no CUDA required.
"""

from __future__ import annotations

import pytest
import torch

from multiscreen import MultiscreenConfig, MultiscreenModel


TINY = MultiscreenConfig(
    vocab_size=64,
    hidden_dim=32,
    num_layers=3,
    num_heads=4,
    key_dim=8,
    value_dim=16,
    max_seq_len=32,
    mipe_threshold=16.0,
)


@pytest.fixture
def tiny_model() -> MultiscreenModel:
    torch.manual_seed(0)
    model = MultiscreenModel(TINY)
    model.eval()
    return model


def test_train_and_eval_prefill_match(tiny_model):
    """Training-mode forward and eval-mode (cache-returning) prefill must
    produce identical logits. The only difference is whether the cache is
    kept around."""
    torch.manual_seed(1)
    input_ids = torch.randint(0, TINY.vocab_size, (2, 8))

    tiny_model.train()
    logits_train, kv_train = tiny_model(input_ids)
    assert kv_train == []

    tiny_model.eval()
    logits_eval, kv_eval = tiny_model(input_ids)

    assert len(kv_eval) == TINY.num_layers
    torch.testing.assert_close(logits_train, logits_eval, rtol=1e-5, atol=1e-5)


def test_prefill_then_incremental_matches_full_forward(tiny_model):
    """Prefill a prefix, then decode the rest one token at a time. The
    resulting logits must match the full forward pass."""
    torch.manual_seed(2)
    seq_len = 8
    input_ids = torch.randint(0, TINY.vocab_size, (1, seq_len))

    # Reference: full forward
    ref_logits, _ = tiny_model(input_ids)

    # Incremental: prefill first 4 tokens, then 4 single-token steps
    prefill_len = 4
    prefill_ids = input_ids[:, :prefill_len]
    pre_logits, kv_caches = tiny_model(prefill_ids, start_pos=0)

    torch.testing.assert_close(
        pre_logits, ref_logits[:, :prefill_len], rtol=1e-5, atol=1e-5,
    )

    inc_pieces = []
    for t in range(prefill_len, seq_len):
        next_id = input_ids[:, t:t + 1]
        step_logits, kv_caches = tiny_model(
            next_id, start_pos=t, kv_caches=kv_caches,
        )
        inc_pieces.append(step_logits)

    inc_logits = torch.cat(inc_pieces, dim=1)
    torch.testing.assert_close(
        inc_logits, ref_logits[:, prefill_len:], rtol=1e-5, atol=1e-5,
    )


def test_token_by_token_from_empty_cache(tiny_model):
    """Decoding one token at a time from an empty cache must match a full
    forward pass."""
    torch.manual_seed(3)
    seq_len = 6
    input_ids = torch.randint(0, TINY.vocab_size, (1, seq_len))

    ref_logits, _ = tiny_model(input_ids)

    kv_caches = None
    inc_pieces = []
    for t in range(seq_len):
        next_id = input_ids[:, t:t + 1]
        step_logits, kv_caches = tiny_model(
            next_id, start_pos=t, kv_caches=kv_caches,
        )
        inc_pieces.append(step_logits)

    inc_logits = torch.cat(inc_pieces, dim=1)
    torch.testing.assert_close(inc_logits, ref_logits, rtol=1e-5, atol=1e-5)


def test_batched_incremental_matches_full(tiny_model):
    """Batch size > 1 must also work correctly."""
    torch.manual_seed(4)
    seq_len = 7
    input_ids = torch.randint(0, TINY.vocab_size, (3, seq_len))

    ref_logits, _ = tiny_model(input_ids)

    prefill_len = 3
    pre_logits, kv_caches = tiny_model(input_ids[:, :prefill_len])
    torch.testing.assert_close(
        pre_logits, ref_logits[:, :prefill_len], rtol=1e-5, atol=1e-5,
    )

    for t in range(prefill_len, seq_len):
        next_ids = input_ids[:, t:t + 1]
        step_logits, kv_caches = tiny_model(
            next_ids, start_pos=t, kv_caches=kv_caches,
        )
        torch.testing.assert_close(
            step_logits, ref_logits[:, t:t + 1], rtol=1e-5, atol=1e-5,
        )


def test_kv_cache_shapes(tiny_model):
    """KV cache tensors should grow by the new-token count each step."""
    torch.manual_seed(5)
    B, T = 2, 5
    input_ids = torch.randint(0, TINY.vocab_size, (B, T))
    _, kv_caches = tiny_model(input_ids)

    assert len(kv_caches) == TINY.num_layers
    for k_cache, v_cache in kv_caches:
        assert k_cache.shape == (B, TINY.num_heads, T, TINY.key_dim)
        assert v_cache.shape == (B, TINY.num_heads, T, TINY.value_dim)

    # Append one more token -> cache should grow by 1
    next_id = torch.randint(0, TINY.vocab_size, (B, 1))
    _, kv_caches_2 = tiny_model(next_id, start_pos=T, kv_caches=kv_caches)
    for k_cache, v_cache in kv_caches_2:
        assert k_cache.shape == (B, TINY.num_heads, T + 1, TINY.key_dim)
        assert v_cache.shape == (B, TINY.num_heads, T + 1, TINY.value_dim)


def test_multi_token_chunk_decode(tiny_model):
    """Feeding multiple new tokens at once (chunk decode) must match the
    full forward pass as well."""
    torch.manual_seed(6)
    seq_len = 10
    input_ids = torch.randint(0, TINY.vocab_size, (1, seq_len))

    ref_logits, _ = tiny_model(input_ids)

    # Prefill 4, then feed 3 + 3 in two chunks
    pre_logits, kv = tiny_model(input_ids[:, :4])
    chunk1, kv = tiny_model(input_ids[:, 4:7], start_pos=4, kv_caches=kv)
    chunk2, kv = tiny_model(input_ids[:, 7:10], start_pos=7, kv_caches=kv)

    torch.testing.assert_close(pre_logits, ref_logits[:, :4], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(chunk1, ref_logits[:, 4:7], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(chunk2, ref_logits[:, 7:10], rtol=1e-5, atol=1e-5)
