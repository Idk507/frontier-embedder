
"""Unit tests for real-backbone verification orchestration."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier_embedder.qwen3_verification import (
    build_probe,
    load_factory,
    resolve_device,
    verify_backbone,
)


class TinyTokenizer:
    pad_token_id = 0

    def __len__(self):
        return 64

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        return_tensors="pt",
    ):
        return {
            "input_ids": torch.tensor(
                [[1, 2, 3, 4]],
                dtype=torch.long,
            )
        }


class ToyBackbone(nn.Module):
    def __init__(self, *, causal=False):
        super().__init__()
        self.embedding = nn.Embedding(64, 16)
        self.causal = causal

    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids)

        batch_size, length, _ = hidden.shape

        allowed = attention_mask.bool()[
            :, None, :
        ].expand(batch_size, length, length)

        if self.causal:
            causal_mask = torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=hidden.device,
            ).tril()

            allowed = allowed & causal_mask.unsqueeze(0)

        weights = allowed.to(hidden.dtype)

        weights = weights / weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1)

        return torch.matmul(weights, hidden)


def test_auto_device_is_supported():
    device = resolve_device("auto")
    assert device.type in ("cpu", "cuda")


def test_cpu_device():
    assert resolve_device("cpu").type == "cpu"


def test_unsupported_device_rejected():
    with pytest.raises(ValueError):
        resolve_device("meta")


def test_invalid_factory_specification():
    with pytest.raises(ValueError):
        load_factory("invalid_factory_name")


def test_build_probe_shapes():
    original, modified, mask = build_probe(
        TinyTokenizer()
    )

    assert original.shape == (1, 4)
    assert modified.shape == original.shape
    assert mask.shape == original.shape


def test_build_probe_changes_only_last_token():
    original, modified, _ = build_probe(
        TinyTokenizer()
    )

    changed = original != modified

    assert changed.sum().item() == 1
    assert changed[0, -1].item()


def test_bidirectional_backbone_detected():
    torch.manual_seed(123)

    report = verify_backbone(
        ToyBackbone(causal=False),
        TinyTokenizer(),
    )

    assert report.bidirectional
    assert report.padding_invariant
    assert report.earlier_token_difference > 0


def test_causal_backbone_detected():
    torch.manual_seed(123)

    report = verify_backbone(
        ToyBackbone(causal=True),
        TinyTokenizer(),
        mode="causal",
    )

    assert not report.bidirectional
    assert report.padding_invariant


def test_cpu_memory_report():
    report = verify_backbone(
        ToyBackbone(),
        TinyTokenizer(),
    )

    assert report.device == "cpu"
    assert report.peak_allocated_mb is None
    assert report.peak_reserved_mb is None
    assert report.output_shape == (1, 4, 16)


def test_missing_pad_token_rejected():
    tokenizer = TinyTokenizer()
    tokenizer.pad_token_id = None

    with pytest.raises(ValueError):
        verify_backbone(
            ToyBackbone(),
            tokenizer,
        )
