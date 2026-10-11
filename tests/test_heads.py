
"""Unit tests for the multi-vector embedding head."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier_embedder.config import ModelConfig
from frontier_embedder.heads import MultiVectorHead


@pytest.fixture
def config():
    return ModelConfig()


@pytest.fixture
def head(config):
    return MultiVectorHead(config)


def test_output_shape(head):
    hidden = torch.randn(2, 10, 1024)
    mask = torch.ones(2, 10)

    output = head(hidden, mask)

    assert output.shape == (2, 10, 128)


def test_token_normalization(head):
    hidden = torch.randn(2, 6, 1024)
    mask = torch.ones(2, 6)

    output = head(hidden, mask)
    norms = torch.linalg.vector_norm(output, dim=-1)

    assert torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-5,
    )


def test_padding_is_zero(head):
    hidden = torch.randn(1, 5, 1024)
    mask = torch.tensor([[1, 1, 1, 0, 0]])

    output = head(hidden, mask)

    assert torch.count_nonzero(output[:, 3:, :]) == 0


def test_valid_tokens_remain_nonzero(head):
    hidden = torch.randn(1, 4, 1024)
    mask = torch.tensor([[1, 1, 0, 0]])

    output = head(hidden, mask)

    assert torch.count_nonzero(output[:, :2, :]) > 0


def test_projection_parameter_count(head):
    count = sum(
        parameter.numel()
        for parameter in head.parameters()
    )

    assert count == 1024 * 128


def test_gradient_flow(head):
    hidden = torch.randn(
        2, 5, 1024,
        requires_grad=True,
    )
    mask = torch.ones(2, 5)

    output = head(hidden, mask)
    loss = output.square().sum()
    loss.backward()

    assert hidden.grad is not None
    assert head.projection.weight.grad is not None

    assert torch.isfinite(hidden.grad).all()
    assert torch.isfinite(
        head.projection.weight.grad
    ).all()


def test_invalid_hidden_size(head):
    hidden = torch.randn(1, 4, 512)
    mask = torch.ones(1, 4)

    with pytest.raises(ValueError):
        head(hidden, mask)


def test_invalid_attention_mask_shape(head):
    hidden = torch.randn(1, 4, 1024)
    mask = torch.ones(1, 3)

    with pytest.raises(ValueError):
        head(hidden, mask)


def test_all_padding_rejected(head):
    hidden = torch.randn(1, 4, 1024)
    mask = torch.zeros(1, 4)

    with pytest.raises(ValueError):
        head(hidden, mask)


def test_nonbinary_mask_rejected(head):
    hidden = torch.randn(1, 4, 1024)
    mask = torch.tensor([[1, 1, 0.5, 0]])

    with pytest.raises(ValueError):
        head(hidden, mask)


def test_invalid_hidden_dtype(head):
    hidden = torch.ones(
        1, 4, 1024,
        dtype=torch.long,
    )
    mask = torch.ones(1, 4)

    with pytest.raises(TypeError):
        head(hidden, mask)


def test_custom_projection_dimension():
    config = ModelConfig(multi_vector_dim=64)
    head = MultiVectorHead(config)

    hidden = torch.randn(2, 3, 1024)
    mask = torch.ones(2, 3)

    output = head(hidden, mask)

    assert output.shape == (2, 3, 64)
