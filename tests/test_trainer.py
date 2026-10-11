
"""Unit tests for the reproducible retrieval trainer."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier_embedder.config import ModelConfig
from frontier_embedder.model import UnifiedEmbeddingModel
from frontier_embedder.trainer import (
    RetrievalTrainer,
    TrainerConfig,
    set_seed,
)
from frontier_embedder.training_step import (
    RetrievalBatch,
    RetrievalTrainingObjective,
)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(128, 32)

    def forward(self, input_ids, attention_mask):
        return self.embedding(input_ids)


def make_batch(offset=0):
    return RetrievalBatch(
        query_input_ids=torch.tensor([
            [1 + offset, 2 + offset, 0],
            [3 + offset, 4 + offset, 0],
        ]),
        query_attention_mask=torch.tensor([
            [1, 1, 0],
            [1, 1, 0],
        ]),
        document_input_ids=torch.tensor([
            [1 + offset, 2 + offset, 5 + offset],
            [3 + offset, 4 + offset, 6 + offset],
        ]),
        document_attention_mask=torch.ones(2, 3, dtype=torch.long),
    )


def make_trainer(
    tmp_path,
    *,
    epochs=2,
    accumulation=1,
    batches=3,
    scheduler=False,
):
    set_seed(42)

    config = replace(
        ModelConfig(),
        hidden_size=32,
        dense_dim=32,
        matryoshka_dims=(8, 16, 32),
        multi_vector_dim=8,
    )

    model = UnifiedEmbeddingModel(
        TinyBackbone(),
        config,
    )

    objective = RetrievalTrainingObjective()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    lr_scheduler = None

    if scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.9,
        )

    train_batches = [
        make_batch(i * 6)
        for i in range(batches)
    ]

    validation_batches = [
        make_batch(30),
    ]

    # A list of already-collated RetrievalBatch objects.
    train_loader = DataLoader(
        train_batches,
        batch_size=None,
        shuffle=False,
    )

    validation_loader = DataLoader(
        validation_batches,
        batch_size=None,
        shuffle=False,
    )

    return RetrievalTrainer(
        model=model,
        objective=objective,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=TrainerConfig(
            epochs=epochs,
            gradient_accumulation_steps=accumulation,
            checkpoint_dir=str(tmp_path),
        ),
        scheduler=lr_scheduler,
        device="cpu",
    )


def test_trainer_config_rejects_zero_epochs():
    with pytest.raises(ValueError):
        TrainerConfig(epochs=0)


def test_trainer_config_rejects_invalid_accumulation():
    with pytest.raises(ValueError):
        TrainerConfig(gradient_accumulation_steps=0)


def test_trainer_config_rejects_invalid_clipping():
    with pytest.raises(ValueError):
        TrainerConfig(max_grad_norm=0)


def test_training_runs_two_epochs(tmp_path):
    trainer = make_trainer(tmp_path)

    history = trainer.fit()

    assert len(history) == 2
    assert trainer.completed_epochs == 2


def test_training_losses_are_finite(tmp_path):
    trainer = make_trainer(tmp_path)

    history = trainer.fit()

    for record in history:
        assert torch.isfinite(
            torch.tensor(record.train_loss)
        )
        assert torch.isfinite(
            torch.tensor(record.validation_loss)
        )


def test_optimizer_updates_weights(tmp_path):
    trainer = make_trainer(tmp_path)

    before = (
        trainer.model.token_projection.weight
        .detach().clone()
    )

    trainer.fit()

    after = (
        trainer.model.token_projection.weight.detach()
    )

    assert not torch.equal(before, after)


def test_accumulation_step_count(tmp_path):
    trainer = make_trainer(
        tmp_path,
        epochs=2,
        accumulation=2,
        batches=3,
    )

    trainer.fit()

    # ceil(3 / 2) = 2 optimizer steps per epoch.
    assert trainer.global_step == 4


def test_accumulation_larger_than_epoch(tmp_path):
    trainer = make_trainer(
        tmp_path,
        epochs=1,
        accumulation=10,
        batches=3,
    )

    trainer.fit()

    assert trainer.global_step == 1


def test_scheduler_steps_per_optimizer_update(tmp_path):
    trainer = make_trainer(
        tmp_path,
        epochs=1,
        accumulation=2,
        batches=3,
        scheduler=True,
    )

    trainer.fit()

    assert trainer.scheduler.last_epoch == 2


def test_evaluation_does_not_create_gradients(tmp_path):
    trainer = make_trainer(tmp_path)

    trainer.optimizer.zero_grad(set_to_none=True)

    value = trainer.evaluate()

    assert isinstance(value, float)

    assert all(
        parameter.grad is None
        for parameter in trainer.model.parameters()
    )


def test_latest_checkpoint_created(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.fit()

    assert (tmp_path / "latest.pt").is_file()


def test_best_checkpoint_created(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.fit()

    assert (tmp_path / "best.pt").is_file()


def test_checkpoint_restores_weights(tmp_path):
    trainer = make_trainer(tmp_path, epochs=1)
    trainer.fit()

    expected = {
        key: value.detach().clone()
        for key, value in trainer.model.state_dict().items()
    }

    restored = make_trainer(
        tmp_path,
        epochs=2,
    )

    restored.load_checkpoint(tmp_path / "latest.pt")

    for key, value in restored.model.state_dict().items():
        assert torch.equal(value, expected[key])

    assert restored.completed_epochs == 1


def test_resume_continues_remaining_epochs(tmp_path):
    first = make_trainer(tmp_path, epochs=1)
    first.fit()

    resumed = make_trainer(tmp_path, epochs=3)
    resumed.load_checkpoint(tmp_path / "latest.pt")

    history = resumed.fit()

    assert resumed.completed_epochs == 3
    assert len(history) == 3


def test_checkpoint_restores_global_step(tmp_path):
    first = make_trainer(
        tmp_path,
        epochs=1,
        batches=3,
    )
    first.fit()

    resumed = make_trainer(
        tmp_path,
        epochs=2,
        batches=3,
    )

    resumed.load_checkpoint(tmp_path / "latest.pt")

    assert resumed.global_step == 3


def test_checkpoint_rejects_changed_accumulation(tmp_path):
    first = make_trainer(
        tmp_path,
        epochs=1,
        accumulation=1,
    )
    first.fit()

    resumed = make_trainer(
        tmp_path,
        epochs=2,
        accumulation=2,
    )

    with pytest.raises(ValueError):
        resumed.load_checkpoint(tmp_path / "latest.pt")


def test_frozen_backbone_remains_unchanged(tmp_path):
    trainer = make_trainer(tmp_path, epochs=1)

    trainer.model.freeze_backbone()

    before = (
        trainer.model.backbone.embedding.weight
        .detach().clone()
    )

    trainer.fit()

    after = (
        trainer.model.backbone.embedding.weight.detach()
    )

    assert torch.equal(before, after)


def test_seed_reproducibility(tmp_path):
    first = make_trainer(
        tmp_path / "first",
        epochs=1,
    )

    first_history = first.fit()

    second = make_trainer(
        tmp_path / "second",
        epochs=1,
    )

    second_history = second.fit()

    assert (
        first_history[0].train_loss
        == second_history[0].train_loss
    )

    assert (
        first_history[0].validation_loss
        == second_history[0].validation_loss
    )
