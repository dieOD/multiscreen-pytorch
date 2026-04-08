"""Dataset utilities: tokenize, pack, and serve sequences for LM training."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTextDataset(Dataset):
    """In-memory packed dataset for language model training.

    Pre-tokenizes a stream of texts, concatenates with EOS separators,
    then chunks into fixed-size sequences. Labels are next-token (shifted by 1).

    Args:
        texts: Iterable of strings.
        tokenizer: HuggingFace tokenizer (any object with .encode() method).
        seq_len: Sequence length per chunk.
        eos_token_id: Token ID to insert between documents (defaults to tokenizer.eos_token_id).
        max_tokens: Optional cap on total tokens for quick experiments.
    """

    def __init__(
        self,
        texts: Iterable[str],
        tokenizer,
        seq_len: int = 256,
        eos_token_id: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        self.seq_len = seq_len
        if eos_token_id is None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if eos_token_id is None:
                eos_token_id = 0
        self.eos_token_id = eos_token_id

        # Tokenize and concatenate
        all_ids: list[int] = []
        for text in texts:
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
            all_ids.append(eos_token_id)
            if max_tokens is not None and len(all_ids) >= max_tokens:
                all_ids = all_ids[:max_tokens]
                break

        # We need seq_len + 1 tokens per chunk (input + label)
        chunk_size = seq_len + 1
        usable = (len(all_ids) // chunk_size) * chunk_size
        if usable == 0:
            raise ValueError(f"Not enough tokens for one chunk (need {chunk_size}, got {len(all_ids)})")
        self.tokens = np.array(all_ids[:usable], dtype=np.int64)
        self.tokens = self.tokens.reshape(-1, chunk_size)
        self.num_chunks = self.tokens.shape[0]

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> dict:
        chunk = self.tokens[idx]
        input_ids = torch.from_numpy(chunk[:-1].copy())
        labels = torch.from_numpy(chunk[1:].copy())
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.float32),
        }

    @classmethod
    def from_hf_dataset(
        cls,
        dataset_name: str,
        tokenizer,
        seq_len: int = 256,
        split: str = "train",
        text_column: str = "text",
        config_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> "PackedTextDataset":
        """Load a HuggingFace dataset and pack it.

        Examples:
            PackedTextDataset.from_hf_dataset("wikitext", tokenizer, config_name="wikitext-2-raw-v1")
            PackedTextDataset.from_hf_dataset("roneneldan/TinyStories", tokenizer)
        """
        from datasets import load_dataset
        ds = load_dataset(dataset_name, config_name, split=split)
        return cls(
            texts=(row[text_column] for row in ds),
            tokenizer=tokenizer,
            seq_len=seq_len,
            max_tokens=max_tokens,
        )
