import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase
import re
import logging

import os
import json
import gzip
import random
from typing import Any
from pathlib import Path

class PackedSFTDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | os.PathLike,
        seq_length: int,
        shuffle: bool,
    ):
        self.seq_length = seq_length
        docs = []
        open_fn = gzip.open if str(dataset_path).endswith(".gz") else open
        with open_fn(dataset_path, "rt", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompt = obj["prompt"]
                response = obj["response"]
                text = (
                    "Below is an instruction that describes a task. "
                    "Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{prompt}\n\n"
                    f"### Response:\n{response}"
                )
                docs.append(text)
        if shuffle:
            random.shuffle(docs)

        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        full_ids = []
        for doc in docs:
            if bos_id is not None:
                full_ids.append(bos_id)
            ids = tokenizer.encode(doc, add_special_tokens=False)
            full_ids.extend(ids)
            if eos_id is not None:
                full_ids.append(eos_id)

        total_seqs = len(full_ids) // seq_length
        self.full_ids = full_ids
        self.token_ids = full_ids[: total_seqs * seq_length]
        self.num_sequences = total_seqs

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError

        start = idx * self.seq_length
        end = start + self.seq_length
        input_ids = self.token_ids[start:end]
        labels    = self.full_ids[start+1 : end+1]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        } 

def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Constructs a Dataset that packs instruction-response pairs into fixed-length sequences.
    Each example has 'input_ids' (length seq_length) and 'labels' (next-token labels).
    """
    return PackedSFTDataset(tokenizer, dataset_path, seq_length, shuffle)


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Returns a DataLoader that yields batches of size batch_size over the dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)