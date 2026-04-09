"""Minimal KV-cache generation example.

Shows how to use the Multiscreen model for autoregressive decoding:
    1. Prefill the prompt in one forward pass -> cache is built.
    2. Step-by-step decode: feed the last token + cache, get a new token.

No tokenizer required — the demo uses random prompt IDs so it runs anywhere.
For a real use case, replace ``prompt_ids`` with tokenizer output.
"""

from __future__ import annotations

import torch

from multiscreen import MultiscreenConfig, MultiscreenModel


@torch.no_grad()
def greedy_generate(
    model: MultiscreenModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 32,
    eos_id: int | None = None,
) -> list[int]:
    """Greedy decode with KV cache.

    Args:
        model: a ``MultiscreenModel`` in eval mode.
        prompt_ids: (1, T_prompt) long tensor on the same device as the model.
        max_new_tokens: number of tokens to generate.
        eos_id: if provided, stop when this token is emitted.

    Returns:
        List of generated token IDs (not including the prompt).
    """
    assert prompt_ids.dim() == 2 and prompt_ids.size(0) == 1, "batch size must be 1"
    model.eval()
    device = prompt_ids.device

    # 1. Prefill: one forward pass over the whole prompt builds the cache.
    logits, kv_caches = model(prompt_ids)
    next_logits = logits[:, -1, :]
    start_pos = prompt_ids.shape[1]

    generated: list[int] = []
    for step in range(max_new_tokens):
        if start_pos + step >= model.config.max_seq_len:
            break

        next_id = int(next_logits.argmax(dim=-1).item())
        if eos_id is not None and next_id == eos_id:
            break
        generated.append(next_id)

        # 2. Incremental step: feed only the new token + cache.
        next_input = torch.tensor([[next_id]], device=device)
        logits, kv_caches = model(
            next_input, start_pos=start_pos + step, kv_caches=kv_caches,
        )
        next_logits = logits[:, -1, :]

    return generated


def main():
    torch.manual_seed(0)

    # Build a tiny (untrained!) model and a random "prompt".
    config = MultiscreenConfig(
        vocab_size=1000,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        key_dim=16,
        value_dim=64,
        max_seq_len=64,
    )
    model = MultiscreenModel(config)
    model.eval()
    print(f"Parameters: {model.count_parameters():,}")

    prompt_ids = torch.randint(0, config.vocab_size, (1, 8))
    print(f"Prompt IDs: {prompt_ids[0].tolist()}")

    generated = greedy_generate(model, prompt_ids, max_new_tokens=16)
    print(f"Generated:  {generated}")

    # Sanity check: cached decode must match a full forward on the full sequence.
    full_ids = torch.cat(
        [prompt_ids, torch.tensor([generated], dtype=torch.long)], dim=1
    )
    ref_logits, _ = model(full_ids)
    # Argmax at each position > prompt should match the generated token at position+1
    # (standard greedy autoregression invariant).
    for i, tok in enumerate(generated):
        pred = int(ref_logits[0, prompt_ids.shape[1] + i - 1].argmax().item())
        assert pred == tok, f"mismatch at {i}: ref={pred} cache={tok}"
    print("Sanity check: cached and full-forward decoding agree.")


if __name__ == "__main__":
    main()
