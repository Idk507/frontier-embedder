
"""Reproducible epoch-based retrieval trainer.

Features:
- Gradient accumulation, including partial final windows
- Gradient clipping and nonfinite gradient checks
- Optional learning-rate scheduler
- Validation without gradients
- Atomic checkpoint replacement
- Epoch-boundary checkpoint resumption
- PyTorch and DataLoader RNG restoration

Checkpoint files are trusted local artifacts. Do not load
untrusted pickle-based checkpoints.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from frontier_embedder.model import UnifiedEmbeddingModel
from frontier_embedder.training_step import (
    RetrievalBatch,
    RetrievalTrainingObjective,
)


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 3
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    checkpoint_dir: str = "checkpoints"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                "gradient_accumulation_steps must be positive"
            )
        if not math.isfinite(self.max_grad_norm):
            raise ValueError("max_grad_norm must be finite")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float
    optimizer_steps: int
    learning_rate: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(
    batch: RetrievalBatch,
    device: torch.device,
) -> RetrievalBatch:
    return RetrievalBatch(
        query_input_ids=batch.query_input_ids.to(device),
        query_attention_mask=batch.query_attention_mask.to(device),
        document_input_ids=batch.document_input_ids.to(device),
        document_attention_mask=batch.document_attention_mask.to(device),
    )


class RetrievalTrainer:
    """Train and validate a unified retrieval model."""

    def __init__(
        self,
        model: UnifiedEmbeddingModel,
        objective: RetrievalTrainingObjective,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        config: TrainerConfig,
        *,
        scheduler: Any | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.config = config
        self.scheduler = scheduler
        self.device = torch.device(device)

        self.model.to(self.device)
        self.objective.to(self.device)

        if len(train_loader) == 0:
            raise ValueError("Training DataLoader is empty")
        if len(validation_loader) == 0:
            raise ValueError("Validation DataLoader is empty")

        self.completed_epochs = 0
        self.global_step = 0
        self.best_validation_loss = float("inf")
        self.history: list[EpochMetrics] = []

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _train_epoch(self) -> float:
        self.model.train()
        self.objective.train()

        accumulation = self.config.gradient_accumulation_steps
        total_batches = len(self.train_loader)

        weighted_loss_sum = 0.0
        example_count = 0

        self.optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(self.train_loader):
            batch = move_batch(batch, self.device)
            batch.validate()

            window_start = (
                batch_index // accumulation
            ) * accumulation

            window_size = min(
                accumulation,
                total_batches - window_start,
            )

            output = self.objective(self.model, batch)
            loss = output.total_loss

            if not torch.isfinite(loss).all():
                raise FloatingPointError(
                    "Nonfinite training loss"
                )

            # Divide by actual window size, including the
            # smaller final accumulation window.
            (loss / window_size).backward()

            batch_size = batch.query_input_ids.shape[0]
            weighted_loss_sum += (
                loss.detach().item() * batch_size
            )
            example_count += batch_size

            window_end = (
                (batch_index + 1) % accumulation == 0
                or batch_index + 1 == total_batches
            )

            if not window_end:
                continue

            parameters = [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
                and parameter.grad is not None
            ]

            if not parameters:
                raise RuntimeError(
                    "No trainable parameters received gradients"
                )

            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=self.config.max_grad_norm,
                error_if_nonfinite=True,
            )

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1
            self.optimizer.zero_grad(set_to_none=True)

        return weighted_loss_sum / example_count

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        self.objective.eval()

        weighted_loss_sum = 0.0
        example_count = 0

        for batch in self.validation_loader:
            batch = move_batch(batch, self.device)
            batch.validate()

            loss = self.objective(
                self.model,
                batch,
            ).total_loss

            if not torch.isfinite(loss).all():
                raise FloatingPointError(
                    "Nonfinite validation loss"
                )

            batch_size = batch.query_input_ids.shape[0]
            weighted_loss_sum += (
                loss.item() * batch_size
            )
            example_count += batch_size

        return weighted_loss_sum / example_count

    def _rng_state(self) -> dict[str, Any]:
        state = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
        }

        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()

        generator = getattr(
            self.train_loader,
            "generator",
            None,
        )

        if generator is not None:
            state["train_loader_generator"] = (
                generator.get_state()
            )

        return state

    def _restore_rng_state(
        self,
        state: dict[str, Any],
    ) -> None:
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"].cpu())

        if (
            "cuda" in state
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(state["cuda"])

        generator = getattr(
            self.train_loader,
            "generator",
            None,
        )

        if (
            generator is not None
            and "train_loader_generator" in state
        ):
            generator.set_state(
                state["train_loader_generator"]
            )

    def save_checkpoint(
        self,
        filename: str = "latest.pt",
    ) -> Path:
        """Save complete epoch-boundary training state."""
        if Path(filename).name != filename:
            raise ValueError(
                "Checkpoint filename must not contain directories"
            )

        path = self.checkpoint_dir / filename
        temporary = path.with_name(path.name + ".tmp")

        checkpoint = {
            "format_version": 1,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            "completed_epochs": self.completed_epochs,
            "global_step": self.global_step,
            "best_validation_loss": self.best_validation_loss,
            "history": [
                asdict(metric)
                for metric in self.history
            ],
            "rng": self._rng_state(),
            "trainer_config": asdict(self.config),
        }

        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
        return path

    def load_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        """Resume from a trusted local checkpoint.

        Instantiate the same model architecture, optimizer,
        scheduler and DataLoaders before loading.
        """
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if checkpoint.get("format_version") != 1:
            raise ValueError(
                "Unsupported checkpoint format"
            )

        previous = checkpoint["trainer_config"]

        if (
            previous["gradient_accumulation_steps"]
            != self.config.gradient_accumulation_steps
        ):
            raise ValueError(
                "Gradient accumulation configuration changed"
            )

        if (
            previous["seed"] != self.config.seed
        ):
            raise ValueError(
                "Training seed changed"
            )

        has_saved_scheduler = (
            checkpoint["scheduler"] is not None
        )

        if has_saved_scheduler != (
            self.scheduler is not None
        ):
            raise ValueError(
                "Scheduler configuration changed"
            )

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        if self.scheduler is not None:
            self.scheduler.load_state_dict(
                checkpoint["scheduler"]
            )

        self.completed_epochs = checkpoint[
            "completed_epochs"
        ]
        self.global_step = checkpoint["global_step"]
        self.best_validation_loss = checkpoint[
            "best_validation_loss"
        ]

        self.history = [
            EpochMetrics(**record)
            for record in checkpoint["history"]
        ]

        self._restore_rng_state(checkpoint["rng"])

    def fit(self) -> list[EpochMetrics]:
        """Train through the configured total epoch count."""
        for epoch in range(
            self.completed_epochs,
            self.config.epochs,
        ):
            train_loss = self._train_epoch()
            validation_loss = self.evaluate()

            self.completed_epochs = epoch + 1

            improved = (
                validation_loss < self.best_validation_loss
            )

            if improved:
                self.best_validation_loss = validation_loss

            learning_rate = self.optimizer.param_groups[0][
                "lr"
            ]

            metrics = EpochMetrics(
                epoch=self.completed_epochs,
                train_loss=train_loss,
                validation_loss=validation_loss,
                optimizer_steps=self.global_step,
                learning_rate=learning_rate,
            )

            self.history.append(metrics)

            # Save latest after each completed epoch.
            self.save_checkpoint("latest.pt")

            # Best checkpoint uses the same complete state.
            if improved:
                self.save_checkpoint("best.pt")

            print(
                f"Epoch {metrics.epoch}/{self.config.epochs} | "
                f"train={train_loss:.6f} | "
                f"val={validation_loss:.6f} | "
                f"steps={self.global_step} | "
                f"lr={learning_rate:.2e}",
                flush=True,
            )

        return self.history
