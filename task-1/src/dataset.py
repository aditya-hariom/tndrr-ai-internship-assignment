"""
src/dataset.py

Dataset loader with ChatML Prompt Loss Masking for instruction fine-tuning.
Ensures Cross-Entropy loss is computed strictly on assistant JSON tokens (labels = -100 on prompt).
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

IGNORE_INDEX = -100

class JsonActionDataset(Dataset):
    """
    PyTorch Dataset for ChatML JSON action extraction.
    Formats conversation turns and applies prompt-loss masking so only assistant
    response tokens contribute to the training loss.
    Pre-tokenizes all samples in memory for zero DataLoader latency.
    """
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 256
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        raw_samples = self._load_data(jsonl_path)
        print(f"Tokenizing and caching {len(raw_samples)} samples from: {jsonl_path}...")

        self.encoded_samples = []
        for sample in raw_samples:
            messages = sample["messages"]
            prompt_messages = [m for m in messages if m["role"] in ("system", "user")]

            # Encode full conversation using ChatML template
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            # Encode prompt only with add_generation_prompt=True to find exact assistant start boundary
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize full conversation
            full_encoding = self.tokenizer(
                full_text,
                max_length=self.max_seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            # Tokenize prompt only to measure prompt token length
            prompt_encoding = self.tokenizer(
                prompt_text,
                max_length=self.max_seq_length,
                truncation=True,
                add_special_tokens=False,
                return_tensors="pt"
            )

            input_ids = full_encoding["input_ids"].squeeze(0)
            attention_mask = full_encoding["attention_mask"].squeeze(0)
            labels = input_ids.clone()

            prompt_len = min(prompt_encoding["input_ids"].shape[1], self.max_seq_length)

            # PROMPT LOSS MASKING:
            # Mask out prompt tokens so loss is computed ONLY on assistant JSON
            labels[:prompt_len] = IGNORE_INDEX

            # Also mask padding tokens
            labels[attention_mask == 0] = IGNORE_INDEX

            self.encoded_samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

        print(f"Cached {len(self.encoded_samples)} samples.")

    def _load_data(self, path: str) -> List[Dict[str, Any]]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def __len__(self) -> int:
        return len(self.encoded_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.encoded_samples[idx]
