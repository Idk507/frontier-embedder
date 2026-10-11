
"""Tests for end-to-end retrieval training."""

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
from frontier_embedder.training_step import (
    RetrievalBatch,
    RetrievalTrainingObjective,
    train_one_step,
)


class TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(128, hidden_size)

    def forward(self, input_ids, attention_mask):
        return self.embedding(input_ids)


@pytest.fixture
def model():
    torch.manual_seed(42)

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
    return RetrievalBatch(
        query_input_ids=torch.tensor([
            [1, 2, 3, 0],
            [4, 5, 6, 0],
            [7, 8, 9, 0],
        ]),
        query_attention_mask=torch.tensor([
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
        ]),
        document_input_ids=torch.tensor([
            [1, 2, 10, 11, 0],
            [4, 5, 12, 13, 0],
            [7, 8, 14, 15, 0],
        ]),
        document_attention_mask=torch.tensor([
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
        ]),
    )


def test_batch_validation(batch):
    batch.validate()


def test_batch_requires_multiple_pairs(batch):
    batch.query_input_ids = batch.query_input_ids[:1]
    batch.query_attention_mask = batch.query_attention_mask[:1]
    batch.document_input_ids = batch.document_input_ids[:1]
    batch.document_attention_mask = batch.document_attention_mask[:1]

    with pytest.raises(ValueError):
        batch.validate()


def test_batch_rejects_mismatched_pair_counts(batch):
    batch.document_input_ids = batch.document_input_ids[:2]
    batch.document_attention_mask = batch.document_attention_mask[:2]

    with pytest.raises(ValueError):
        batch.validate()


def test_joint_loss_is_finite(model, batch):
    objective = RetrievalTrainingObjective()
    output = objective(model, batch)

    assert torch.isfinite(output.total_loss)
    assert torch.isfinite(output.dense_loss)
    assert torch.isfinite(output.matryoshka_loss)
    assert torch.isfinite(output.multi_vector_loss)


def test_joint_loss_weighted_sum(model, batch):
    objective = RetrievalTrainingObjective(
        dense_weight=1.0,
        matryoshka_weight=0.5,
        multi_vector_weight=0.25,
    )

    output = objective(model, batch)

    expected = (
        output.dense_loss
        + 0.5 * output.matryoshka_loss
        + 0.25 * output.multi_vector_loss
    )

    assert torch.allclose(output.total_loss, expected)


def test_backward_reaches_backbone_and_head(model, batch):
    objective = RetrievalTrainingObjective()
    output = objective(model, batch)

    output.total_loss.backward()

    backbone_grad = model.backbone.embedding.weight.grad
    projection_grad = model.token_projection.weight.grad

    assert backbone_grad is not None
    assert projection_grad is not None

    assert backbone_grad.abs().sum() > 0
    assert projection_grad.abs().sum() > 0


def test_optimizer_updates_projection(model, batch):
    objective = RetrievalTrainingObjective()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    before = model.token_projection.weight.detach().clone()

    train_one_step(
        model,
        batch,
        objective,
        optimizer,
    )

    after = model.token_projection.weight.detach()

    assert not torch.equal(before, after)


def test_frozen_backbone_is_unchanged(model, batch):
    model.freeze_backbone()

    objective = RetrievalTrainingObjective()

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=1e-3,
    )

    backbone_before = (
        model.backbone.embedding.weight.detach().clone()
    )

    projection_before = (
        model.token_projection.weight.detach().clone()
    )

    train_one_step(
        model,
        batch,
        objective,
        optimizer,
    )

    assert torch.equal(
        backbone_before,
        model.backbone.embedding.weight.detach(),
    )

    assert not torch.equal(
        projection_before,
        model.token_projection.weight.detach(),
    )


def test_multi_vector_only_training(model, batch):
    objective = RetrievalTrainingObjective(
        dense_weight=0.0,
        matryoshka_weight=0.0,
        multi_vector_weight=1.0,
    )

    output = objective(model, batch)
    output.total_loss.backward()

    assert (
        model.token_projection.weight.grad.abs().sum() > 0
    )


def test_invalid_temperature():
    with pytest.raises(ValueError):
        RetrievalTrainingObjective(
            dense_temperature=0.0
        )


def test_invalid_loss_weights():
    with pytest.raises(ValueError):
        RetrievalTrainingObjective(
            dense_weight=-1.0
        )


def test_all_zero_loss_weights():
    with pytest.raises(ValueError):
        RetrievalTrainingObjective(
            dense_weight=0.0,
            matryoshka_weight=0.0,
            multi_vector_weight=0.0,
        )


def test_invalid_gradient_clip(model, batch):
    objective = RetrievalTrainingObjective()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    with pytest.raises(ValueError):
        train_one_step(
            model,
            batch,
            objective,
            optimizer,
            max_grad_norm=0.0,
        )
