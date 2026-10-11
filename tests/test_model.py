
"""Unit tests for the unified embedding model."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier_embedder.config import ModelConfig
from frontier_embedder.model import UnifiedEmbeddingModel


class TinyBackbone(nn.Module):
    """Trainable synthetic encoder for integration tests."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(100, hidden_size)

    def forward(self, input_ids, attention_mask):
        return self.embedding(input_ids)


@pytest.fixture
def model():
    config = replace(
        ModelConfig(),
        hidden_size=32,
        dense_dim=32,
        matryoshka_dims=(8, 16, 32),
        multi_vector_dim=8,
    )
    return UnifiedEmbeddingModel(
        TinyBackbone(32),
        config,
    )


@pytest.fixture
def batch():
    ids = torch.tensor([
        [1, 2, 3, 0],
        [4, 5, 0, 0],
    ])
    mask = torch.tensor([
        [1, 1, 1, 0],
        [1, 1, 0, 0],
    ])
    return ids, mask


def test_dense_shape(model, batch):
    output = model(*batch)
    assert output.dense.shape == (2, 32)


def test_dense_unit_norm(model, batch):
    output = model(*batch)
    norms = output.dense.norm(dim=-1)
    assert torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-5,
    )


def test_matryoshka_shapes(model, batch):
    output = model(*batch)
    assert set(output.matryoshka) == {8, 16, 32}
    for dim, embedding in output.matryoshka.items():
        assert embedding.shape == (2, dim)


def test_matryoshka_unit_norm(model, batch):
    output = model(*batch)
    for embedding in output.matryoshka.values():
        norms = embedding.norm(dim=-1)
        assert torch.allclose(
            norms,
            torch.ones_like(norms),
            atol=1e-5,
        )


def test_multi_vector_shape(model, batch):
    output = model(*batch)
    assert output.multi_vector.shape == (2, 4, 8)


def test_padding_vectors_are_zero(model, batch):
    output = model(*batch)
    invalid = batch[1] == 0
    assert torch.count_nonzero(
        output.multi_vector[invalid]
    ) == 0


def test_valid_token_vectors_normalized(model, batch):
    output = model(*batch)
    valid = batch[1].bool()
    norms = output.multi_vector[valid].norm(dim=-1)
    assert torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-5,
    )


def test_dense_padding_invariance(model):
    original = torch.tensor([[1, 2, 3, 4]])
    modified = torch.tensor([[1, 2, 3, 5]])
    mask = torch.tensor([[1, 1, 1, 0]])

    first = model(original, mask)
    second = model(modified, mask)

    assert torch.allclose(
        first.dense,
        second.dense,
        atol=1e-6,
    )


def test_dense_scores_shape(model, batch):
    output = model(*batch)
    scores = model.score(output, output)
    assert scores.shape == (2, 2)


def test_dense_self_similarity(model, batch):
    output = model(*batch)
    scores = model.score(output, output)

    assert torch.allclose(
        scores.diag(),
        torch.ones(2),
        atol=1e-5,
    )


def test_matryoshka_scores(model, batch):
    output = model(*batch)
    scores = model.score(
        output,
        output,
        method="dense",
        dimension=8,
    )
    assert scores.shape == (2, 2)


def test_multi_vector_scores(model, batch):
    output = model(*batch)
    scores = model.score(
        output,
        output,
        method="multi_vector",
    )
    assert scores.shape == (2, 2)
    assert torch.isfinite(scores).all()


def test_gradient_flow(model, batch):
    output = model(*batch)

    loss = (
        output.dense.square().sum()
        + output.multi_vector.square().sum()
    )

    loss.backward()

    assert model.token_projection.weight.grad is not None
    assert model.backbone.embedding.weight.grad is not None


def test_freeze_backbone(model):
    model.freeze_backbone()

    assert all(
        not p.requires_grad
        for p in model.backbone.parameters()
    )
    assert all(
        p.requires_grad
        for p in model.token_projection.parameters()
    )


def test_unfreeze_backbone(model):
    model.freeze_backbone()
    model.unfreeze_backbone()

    assert all(
        p.requires_grad
        for p in model.backbone.parameters()
    )


def test_invalid_mask(model):
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[1, 2, 0]])

    with pytest.raises(ValueError):
        model(ids, mask)


def test_all_padding_rejected(model):
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.zeros_like(ids)

    with pytest.raises(ValueError):
        model(ids, mask)


def test_invalid_scoring_method(model, batch):
    output = model(*batch)

    with pytest.raises(ValueError):
        model.score(
            output,
            output,
            method="unknown",
        )


def test_invalid_matryoshka_dimension(model, batch):
    output = model(*batch)

    with pytest.raises(ValueError):
        model.score(
            output,
            output,
            dimension=7,
        )
