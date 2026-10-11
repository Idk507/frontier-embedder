
"""Integration tests for the Qwen3 embedding backbone."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier_embedder.backbone import BidirectionalQwenBackbone
from frontier_embedder.config import ModelConfig


@pytest.fixture(scope="module")
def backbone():
    """Load the backbone once for integration tests."""
    config = ModelConfig()
    model = BidirectionalQwenBackbone.from_pretrained(
        config,
        dtype=torch.float32,
    )
    model.eval()
    return model


def test_output_shape(backbone):
    input_ids = torch.tensor([[10, 20, 30, 40]])
    mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        output = backbone(input_ids, mask)

    assert output.shape == (1, 4, 1024)
    assert torch.isfinite(output).all()


def test_future_token_influences_earlier_token(backbone):
    """A noncausal model must let a future token affect position zero."""
    first = torch.tensor([[10, 20, 30, 40]])
    second = torch.tensor([[10, 20, 30, 41]])
    mask = torch.ones_like(first)

    with torch.inference_mode():
        a = backbone(first, mask)
        b = backbone(second, mask)

    difference = torch.max(
        torch.abs(a[:, 0, :] - b[:, 0, :])
    ).item()

    assert difference > 1e-6, (
        "The earlier token did not respond to a future-token "
        "change. Attention may still be causal."
    )


def test_invalid_input_shape(backbone):
    with pytest.raises(ValueError):
        backbone(
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 1, 1]),
        )


def test_invalid_mask_shape(backbone):
    with pytest.raises(ValueError):
        backbone(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 1]]),
        )

def test_future_token_changes_earlier_representation(backbone):
    """Changing the final token must influence position zero."""
    original = torch.tensor([[10, 20, 30, 40]])
    modified = torch.tensor([[10, 20, 30, 41]])

    mask = torch.ones_like(original)

    with torch.inference_mode():
        first = backbone(original, mask)
        second = backbone(modified, mask)

    difference = (
        first[:, 0, :] - second[:, 0, :]
    ).abs().max().item()

    assert difference > 1e-6


def test_padding_token_identity_does_not_affect_valid_tokens(
    backbone,
):
    """Masked token IDs must not influence valid representations."""
    original = torch.tensor([[10, 20, 30, 40]])
    modified = torch.tensor([[10, 20, 30, 41]])

    mask = torch.tensor([[1, 1, 1, 0]])

    with torch.inference_mode():
        first = backbone(original, mask)
        second = backbone(modified, mask)

    assert torch.allclose(
        first[:, :3, :],
        second[:, :3, :],
        atol=1e-5,
        rtol=1e-5,
    )


def test_bidirectional_forward_is_deterministic(backbone):
    input_ids = torch.tensor([[10, 20, 30, 40]])
    mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        first = backbone(input_ids, mask)
        second = backbone(input_ids, mask)

    assert torch.allclose(
        first,
        second,
        atol=1e-6,
        rtol=1e-6,
    )


def test_nonbinary_mask_rejected(backbone):
    input_ids = torch.tensor([[10, 20, 30]])

    mask = torch.tensor([[1.0, 0.5, 0.0]])

    with pytest.raises(ValueError):
        backbone(input_ids, mask)


def test_all_padding_rejected(backbone):
    input_ids = torch.tensor([[10, 20, 30]])
    mask = torch.zeros_like(input_ids)

    with pytest.raises(ValueError):
        backbone(input_ids, mask)


def test_invalid_token_dtype_rejected(backbone):
    input_ids = torch.tensor([[10.0, 20.0, 30.0]])
    mask = torch.ones_like(input_ids)

    with pytest.raises(TypeError):
        backbone(input_ids, mask)
