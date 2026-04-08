"""Minimal Multiscreen example: forward + backward in 30 lines.

This script doesn't require any datasets — it just shows how to use the model API.
"""

import torch
import torch.nn as nn

from multiscreen import MultiscreenConfig, MultiscreenModel


def main():
    # Build a tiny model
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
    print(f"Parameters: {model.count_parameters():,}")

    # Fake data
    input_ids = torch.randint(0, 1000, (2, 64))
    labels = torch.randint(0, 1000, (2, 64))

    # Forward
    logits = model(input_ids)
    print(f"Logits shape: {logits.shape}")

    # Loss + backward
    loss = nn.functional.cross_entropy(
        logits.view(-1, 1000), labels.view(-1)
    )
    print(f"Loss: {loss.item():.4f}")

    loss.backward()
    print("Backward pass succeeded.")

    # Check gradients
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
    print(f"All {sum(1 for _ in model.parameters())} parameters have gradients.")


if __name__ == "__main__":
    main()
