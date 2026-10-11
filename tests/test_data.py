
"""Tests for retrieval data preparation."""

import csv
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier_embedder.data import (
    RetrievalExample,
    RetrievalDataset,
    RetrievalCollator,
    normalize_text,
    split_dataset,
    create_dataloader,
)


class TinyTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        texts,
        *,
        truncation,
        max_length,
        padding,
        return_attention_mask,
        return_tensors,
    ):
        sequences = []

        for text in texts:
            tokens = [
                (sum(word.encode("utf-8")) % 99) + 1
                for word in text.split()
            ]

            if truncation:
                tokens = tokens[:max_length]

            sequences.append(tokens)

        longest = max(map(len, sequences))

        ids = []
        masks = []

        for sequence in sequences:
            count = len(sequence)
            pad_count = longest - count

            ids.append(
                sequence + [self.pad_token_id] * pad_count
            )

            masks.append(
                [1] * count + [0] * pad_count
            )

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                masks, dtype=torch.long
            ),
        }


@pytest.fixture
def examples():
    return [
        RetrievalExample(
            "What is Python?",
            "Python is a programming language",
        ),
        RetrievalExample(
            "What is PyTorch?",
            "PyTorch is a deep learning framework",
        ),
        RetrievalExample(
            "What is FAISS?",
            "FAISS supports similarity search",
        ),
        RetrievalExample(
            "What is ColBERT?",
            "ColBERT uses late interaction",
        ),
        RetrievalExample(
            "What is LoRA?",
            "LoRA enables parameter efficient fine tuning",
        ),
        RetrievalExample(
            "What is Qwen?",
            "Qwen is a family of language models",
        ),
    ]


def test_normalize_text():
    assert normalize_text("  Hello   WORLD ") == "hello world"


def test_valid_example():
    example = RetrievalExample("question", "answer")
    assert example.query == "question"


def test_empty_query_rejected():
    with pytest.raises(ValueError):
        RetrievalExample("  ", "answer")


def test_empty_positive_rejected():
    with pytest.raises(ValueError):
        RetrievalExample("question", "\n\t")


def test_invalid_example_type():
    with pytest.raises(TypeError):
        RetrievalExample(123, "answer")


def test_dataset_length(examples):
    dataset = RetrievalDataset(examples)
    assert len(dataset) == 6


def test_dataset_index(examples):
    dataset = RetrievalDataset(examples)
    assert dataset[0] == examples[0]


def test_empty_dataset_rejected():
    with pytest.raises(ValueError):
        RetrievalDataset([])


def test_csv_loading(tmp_path):
    path = tmp_path / "pairs.csv"

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "positive"])
        writer.writerow(["What is AI?", "Artificial intelligence"])

    dataset = RetrievalDataset.from_csv(path)

    assert len(dataset) == 1
    assert dataset[0].query == "What is AI?"


def test_csv_missing_columns(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_text(
        "question,answer\nhello,world\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        RetrievalDataset.from_csv(path)


def test_split_is_reproducible(examples):
    dataset = RetrievalDataset(examples)

    train_a, val_a = split_dataset(dataset, seed=42)
    train_b, val_b = split_dataset(dataset, seed=42)

    assert train_a.examples == train_b.examples
    assert val_a.examples == val_b.examples


def test_split_has_no_exact_positive_leakage(examples):
    dataset = RetrievalDataset(examples)

    train, validation = split_dataset(dataset)

    train_docs = {
        normalize_text(e.positive) for e in train
    }

    val_docs = {
        normalize_text(e.positive) for e in validation
    }

    assert train_docs.isdisjoint(val_docs)


def test_split_keeps_duplicate_documents_together():
    dataset = RetrievalDataset([
        RetrievalExample("q1", "shared document"),
        RetrievalExample("q2", "shared document"),
        RetrievalExample("q3", "unique document"),
    ])

    train, validation = split_dataset(dataset)

    assert len(train) + len(validation) == 3

    shared_train = sum(
        e.positive == "shared document" for e in train
    )
    shared_val = sum(
        e.positive == "shared document" for e in validation
    )

    assert (shared_train, shared_val) in [(2, 0), (0, 2)]


def test_collator_shapes(examples):
    collator = RetrievalCollator(TinyTokenizer())
    batch = collator(examples[:3])

    assert batch.query_input_ids.shape[0] == 3
    assert batch.document_input_ids.shape[0] == 3
    assert batch.query_attention_mask.shape == (
        batch.query_input_ids.shape
    )


def test_dynamic_padding(examples):
    collator = RetrievalCollator(TinyTokenizer())
    batch = collator(examples[:2])

    assert batch.query_input_ids.shape[1] == 3
    assert batch.document_input_ids.shape[1] == 6


def test_query_truncation(examples):
    collator = RetrievalCollator(
        TinyTokenizer(),
        max_query_length=2,
    )

    batch = collator(examples[:2])

    assert batch.query_input_ids.shape[1] == 2


def test_document_truncation(examples):
    collator = RetrievalCollator(
        TinyTokenizer(),
        max_document_length=3,
    )

    batch = collator(examples[:2])

    assert batch.document_input_ids.shape[1] == 3


def test_positive_alignment(examples):
    collator = RetrievalCollator(TinyTokenizer())

    batch = collator(examples[:3])

    assert batch.query_input_ids.shape[0] == (
        batch.document_input_ids.shape[0]
    )

    # Order is preserved independently in both branches.
    first_query = collator._encode(
        [examples[0].query], 32
    )[0]

    first_document = collator._encode(
        [examples[0].positive], 128
    )[0]

    assert torch.equal(
        batch.query_input_ids[0, :first_query.shape[1]],
        first_query[0],
    )

    assert torch.equal(
        batch.document_input_ids[
            0, :first_document.shape[1]
        ],
        first_document[0],
    )


def test_duplicate_positives_rejected():
    collator = RetrievalCollator(TinyTokenizer())

    examples = [
        RetrievalExample("first question", "Same document"),
        RetrievalExample("second question", " same  DOCUMENT "),
    ]

    with pytest.raises(ValueError):
        collator(examples)


def test_single_example_batch_rejected(examples):
    collator = RetrievalCollator(TinyTokenizer())

    with pytest.raises(ValueError):
        collator(examples[:1])


def test_invalid_max_length():
    with pytest.raises(ValueError):
        RetrievalCollator(
            TinyTokenizer(),
            max_query_length=0,
        )


def test_dataloader_returns_retrieval_batch(examples):
    dataset = RetrievalDataset(examples)
    collator = RetrievalCollator(TinyTokenizer())

    loader = create_dataloader(
        dataset,
        collator,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(loader))

    batch.validate()
    assert batch.query_input_ids.shape[0] == 2


def test_dataloader_reproducible_shuffle(examples):
    dataset = RetrievalDataset(examples)
    collator = RetrievalCollator(TinyTokenizer())

    loader_a = create_dataloader(
        dataset, collator, batch_size=2, seed=123
    )

    loader_b = create_dataloader(
        dataset, collator, batch_size=2, seed=123
    )

    a = [
        batch.query_input_ids[:, 0].tolist()
        for batch in loader_a
    ]

    b = [
        batch.query_input_ids[:, 0].tolist()
        for batch in loader_b
    ]

    assert a == b


def test_invalid_batch_size(examples):
    dataset = RetrievalDataset(examples)
    collator = RetrievalCollator(TinyTokenizer())

    with pytest.raises(ValueError):
        create_dataloader(
            dataset,
            collator,
            batch_size=1,
        )
