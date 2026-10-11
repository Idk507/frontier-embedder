"""Tests for embedding model configuration."""

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from frontier_embedder.config import (  # noqa: E402
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)


def test_default_model_configuration():
    config = ModelConfig()

    assert config.hidden_size == 1024
    assert config.dense_dim == 1024
    assert config.multi_vector_dim == 128
    assert config.matryoshka_dims == (
        128, 256, 512, 1024
    )


def test_invalid_matryoshka_dimension():
    with pytest.raises(ValueError):
        ModelConfig(matryoshka_dims=(128, 2048))


def test_duplicate_matryoshka_dimensions():
    with pytest.raises(ValueError):
        ModelConfig(matryoshka_dims=(128, 128, 512))


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        TrainingConfig(per_device_batch_size=0)


def test_negative_loss_weight():
    with pytest.raises(ValueError):
        TrainingConfig(dense_loss_weight=-1.0)


def test_all_losses_disabled():
    with pytest.raises(ValueError):
        TrainingConfig(
            dense_loss_weight=0.0,
            matryoshka_loss_weight=0.0,
            multi_vector_loss_weight=0.0,
        )


def test_experiment_configuration():
    config = ExperimentConfig()

    assert config.model.backbone_name == (
        "Qwen/Qwen3-0.6B-Base"
    )
    assert config.training.freeze_backbone is True