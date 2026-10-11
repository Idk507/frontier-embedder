
"""Retrieval dataset, splitting, tokenization and dynamic collation."""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from frontier_embedder.training_step import RetrievalBatch


def normalize_text(text: str) -> str:
    """Normalize whitespace for validation and duplicate detection."""
    return re.sub(r"\s+", " ", text).strip().casefold()


@dataclass(frozen=True)
class RetrievalExample:
    query: str
    positive: str

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        if not isinstance(self.positive, str):
            raise TypeError("positive must be a string")
        if not normalize_text(self.query):
            raise ValueError("query cannot be empty")
        if not normalize_text(self.positive):
            raise ValueError("positive cannot be empty")


class RetrievalDataset(Dataset):
    """Validated query-positive-document dataset."""

    def __init__(
        self,
        examples: Sequence[RetrievalExample],
    ) -> None:
        self.examples = list(examples)

        if not self.examples:
            raise ValueError("Dataset cannot be empty")

        for example in self.examples:
            if not isinstance(example, RetrievalExample):
                raise TypeError(
                    "Dataset entries must be RetrievalExample"
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> RetrievalExample:
        return self.examples[index]

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
    ) -> "RetrievalDataset":
        examples = []

        with open(
            path,
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)

            if not reader.fieldnames or not {
                "query", "positive"
            }.issubset(reader.fieldnames):
                raise ValueError(
                    "CSV must contain query and positive columns"
                )

            for row_number, row in enumerate(reader, start=2):
                try:
                    examples.append(
                        RetrievalExample(
                            query=row["query"],
                            positive=row["positive"],
                        )
                    )
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"Invalid CSV row {row_number}: {exc}"
                    ) from exc

        return cls(examples)


def split_dataset(
    dataset: RetrievalDataset,
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[RetrievalDataset, RetrievalDataset]:
    """Deterministic split grouped by normalized positive text.

    All examples sharing the same positive document remain in
    the same split, preventing exact-document train/validation
    leakage.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction must be between 0 and 1"
        )

    groups: dict[str, list[RetrievalExample]] = {}

    for example in dataset.examples:
        key = normalize_text(example.positive)
        groups.setdefault(key, []).append(example)

    keys = sorted(groups)

    if len(keys) < 2:
        raise ValueError(
            "At least two distinct positive documents "
            "are required for splitting"
        )

    random.Random(seed).shuffle(keys)

    validation_count = max(
        1,
        min(
            len(keys) - 1,
            round(len(keys) * validation_fraction),
        ),
    )

    validation_keys = set(keys[:validation_count])

    train = []
    validation = []

    for key, examples in groups.items():
        if key in validation_keys:
            validation.extend(examples)
        else:
            train.extend(examples)

    return (
        RetrievalDataset(train),
        RetrievalDataset(validation),
    )


class RetrievalCollator:
    """Tokenize and dynamically pad paired retrieval examples."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_query_length: int = 32,
        max_document_length: int = 128,
        reject_duplicate_positives: bool = True,
    ) -> None:
        if max_query_length < 1:
            raise ValueError("max_query_length must be positive")

        if max_document_length < 1:
            raise ValueError(
                "max_document_length must be positive"
            )

        if getattr(tokenizer, "pad_token_id", None) is None:
            raise ValueError(
                "Tokenizer must define pad_token_id"
            )

        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_document_length = max_document_length
        self.reject_duplicate_positives = (
            reject_duplicate_positives
        )

    def _encode(
        self,
        texts: list[str],
        max_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        ids = encoded["input_ids"]
        mask = encoded["attention_mask"]

        if ids.ndim != 2 or ids.shape != mask.shape:
            raise ValueError(
                "Tokenizer returned invalid tensor shapes"
            )

        if ids.shape[1] > max_length:
            raise ValueError(
                "Tokenizer exceeded maximum sequence length"
            )

        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError(
                "Tokenizer returned a nonbinary mask"
            )

        if torch.any(mask.sum(dim=1) == 0):
            raise ValueError(
                "Tokenization produced an empty sequence"
            )

        return ids.long(), mask.long()

    def __call__(
        self,
        examples: list[RetrievalExample],
    ) -> RetrievalBatch:
        if len(examples) < 2:
            raise ValueError(
                "Contrastive batches need at least two examples"
            )

        if not all(
            isinstance(example, RetrievalExample)
            for example in examples
        ):
            raise TypeError(
                "Expected RetrievalExample entries"
            )

        queries = [example.query for example in examples]
        positives = [
            example.positive for example in examples
        ]

        if self.reject_duplicate_positives:
            normalized = [
                normalize_text(text) for text in positives
            ]

            if len(set(normalized)) != len(normalized):
                raise ValueError(
                    "Batch contains duplicate positive documents"
                )

        query_ids, query_mask = self._encode(
            queries,
            self.max_query_length,
        )

        document_ids, document_mask = self._encode(
            positives,
            self.max_document_length,
        )

        batch = RetrievalBatch(
            query_input_ids=query_ids,
            query_attention_mask=query_mask,
            document_input_ids=document_ids,
            document_attention_mask=document_mask,
        )

        batch.validate()
        return batch


def create_dataloader(
    dataset: RetrievalDataset,
    collator: RetrievalCollator,
    *,
    batch_size: int = 2,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    """Build a reproducible DataLoader for contrastive training.

    Duplicate positive documents must be handled before batching.
    A shuffled dataset with repeated positives may still generate
    invalid batches, which the collator rejects.
    """
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collator,
        generator=generator,
    )
