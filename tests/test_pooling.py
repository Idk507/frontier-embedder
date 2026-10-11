
"""Tests for dense pooling and Matryoshka embeddings."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier_embedder.config import ModelConfig
from frontier_embedder.pooling import (
    DensePooling,
    MatryoshkaEmbedding,
)


@pytest.fixture
def config():
    return ModelConfig()


def test_dense_pooling_shape(config):
    pooling = DensePooling(config)

    hidden = torch.randn(2, 8, 1024)
    mask = torch.ones(2, 8)

    output = pooling(hidden, mask)

    assert output.shape == (2, 1024)


def test_dense_pooling_normalization(config):
    pooling = DensePooling(config)

    hidden = torch.randn(3, 5, 1024)
    mask = torch.ones(3, 5)

    output = pooling(hidden, mask)
    norms = torch.linalg.vector_norm(output, dim=-1)

    assert torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-5,
    )


def test_padding_is_excluded(config):
    pooling = DensePooling(config)

    hidden = torch.zeros(1, 3, 1024)
    hidden[:, 0, 0] = 2.0
    hidden[:, 1, 0] = 4.0
    hidden[:, 2, 1] = 1000.0

    mask = torch.tensor([[1, 1, 0]])

    output = pooling(hidden, mask)

    assert torch.isclose(
        output[0, 0],
        torch.tensor(1.0),
    )

    assert torch.isclose(
        output[0, 1],
        torch.tensor(0.0),
    )


def test_all_padding_rejected(config):
    pooling = DensePooling(config)

    hidden = torch.randn(1, 4, 1024)
    mask = torch.zeros(1, 4)

    with pytest.raises(ValueError):
        pooling(hidden, mask)


def test_matryoshka_dimensions(config):
    module = MatryoshkaEmbedding(config)

    embeddings = torch.randn(2, 1024)
    outputs = module(embeddings)

    assert set(outputs.keys()) == {
        128, 256, 512, 1024
    }

    for dimension, tensor in outputs.items():
        assert tensor.shape == (2, dimension)


def test_matryoshka_prefix_normalization(config):
    module = MatryoshkaEmbedding(config)

    embeddings = torch.randn(2, 1024)
    output = module(embeddings, dimensions=256)

    norms = torch.linalg.vector_norm(output, dim=-1)

    assert torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-5,
    )


def test_invalid_matryoshka_dimension(config):
    module = MatryoshkaEmbedding(config)

    embeddings = torch.randn(2, 1024)

    with pytest.raises(ValueError):
        module(embeddings, dimensions=300)


def test_matryoshka_prefix_consistency(config):
    module = MatryoshkaEmbedding(config)

    embeddings = torch.randn(2, 1024)
    outputs = module(embeddings)

    expected = torch.nn.functional.normalize(
        embeddings[:, :128],
        p=2,
        dim=-1,
    )

    assert torch.allclose(
        outputs[128],
        expected,
        atol=1e-6,
    )
