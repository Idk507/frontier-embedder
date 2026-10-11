
"""Tests for backbone behavioral diagnostics."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier_embedder.backbone_diagnostics import (
    check_bidirectionality,
    check_right_padding_invariance,
    profile_backbone_memory,
)



class ToyAttentionBackbone(nn.Module):
    def __init__(self, causal=False):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.causal = causal

        torch.manual_seed(7)

        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.randn_like(self.embedding.weight)
            )

    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids)

        batch_size, length, _ = hidden.shape

        # Every query position can initially attend to
        # every valid key position.
        allowed = attention_mask.bool()[:, None, :].expand(
            batch_size, length, length
        )

        if self.causal:
            causal_mask = torch.ones(
                length,
                length,
                device=hidden.device,
                dtype=torch.bool,
            ).tril()

            allowed = allowed & causal_mask.unsqueeze(0)

        weights = allowed.to(hidden.dtype)

        weights = weights / weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1)

        return torch.matmul(weights, hidden)


@pytest.fixture
def inputs():
    return {
        "original_ids": torch.tensor([[1, 2, 3, 4]]),
        "modified_ids": torch.tensor([[1, 2, 3, 5]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "earlier_position": 0,
        "later_position": 3,
    }


def test_detects_bidirectional_attention(inputs):
    model = ToyAttentionBackbone(causal=False)

    result = check_bidirectionality(model, **inputs)

    assert result.earlier_token_responds
    assert result.later_token_responds


def test_detects_causal_attention(inputs):
    model = ToyAttentionBackbone(causal=True)

    result = check_bidirectionality(model, **inputs)

    assert not result.earlier_token_responds
    assert result.later_token_responds


def test_rejects_multiple_changed_tokens(inputs):
    model = ToyAttentionBackbone()
    inputs["modified_ids"] = torch.tensor([[9, 2, 3, 5]])

    with pytest.raises(ValueError):
        check_bidirectionality(model, **inputs)


def test_rejects_invalid_positions(inputs):
    model = ToyAttentionBackbone()
    inputs["earlier_position"] = 3

    with pytest.raises(ValueError):
        check_bidirectionality(model, **inputs)


def test_rejects_unchanged_tokens(inputs):
    model = ToyAttentionBackbone()
    inputs["modified_ids"] = inputs["original_ids"].clone()

    with pytest.raises(ValueError):
        check_bidirectionality(model, **inputs)


def test_right_padding_invariance():
    model = ToyAttentionBackbone(causal=False)

    result = check_right_padding_invariance(
        model,
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        pad_token_id=0,
    )

    assert result.invariant
    assert result.max_valid_token_difference < 1e-5


def test_rejects_invalid_padding_length():
    model = ToyAttentionBackbone()

    with pytest.raises(ValueError):
        check_right_padding_invariance(
            model,
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            pad_token_id=0,
            extra_padding=0,
        )


def test_cpu_memory_profile():
    model = ToyAttentionBackbone()

    result = profile_backbone_memory(
        model,
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
    )

    assert result.device == "cpu"
    assert result.output_shape == (1, 3, 8)
    assert result.elapsed_seconds >= 0
    assert result.peak_allocated_mb is None


def test_diagnostic_restores_training_mode(inputs):
    model = ToyAttentionBackbone()
    model.train()

    check_bidirectionality(model, **inputs)

    assert model.training


def test_rejects_negative_tolerance(inputs):
    model = ToyAttentionBackbone()

    with pytest.raises(ValueError):
        check_bidirectionality(
            model,
            **inputs,
            tolerance=-1,
        )
