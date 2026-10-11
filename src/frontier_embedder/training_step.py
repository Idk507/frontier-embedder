
"""End-to-end training step for unified retrieval embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from frontier_embedder.losses import contrastive_cross_entropy
from frontier_embedder.model import UnifiedEmbeddingModel


@dataclass
class RetrievalBatch:
    """Aligned query and positive-document token batches."""

    query_input_ids: Tensor
    query_attention_mask: Tensor
    document_input_ids: Tensor
    document_attention_mask: Tensor

    def validate(self) -> None:
        pairs = (
            (self.query_input_ids, self.query_attention_mask),
            (self.document_input_ids, self.document_attention_mask),
        )

        for ids, mask in pairs:
            if ids.ndim != 2 or mask.shape != ids.shape:
                raise ValueError(
                    "Token IDs and masks must have matching [B, T] shapes"
                )

        query_count = self.query_input_ids.shape[0]
        document_count = self.document_input_ids.shape[0]

        if query_count < 2:
            raise ValueError(
                "In-batch contrastive training requires at least 2 pairs"
            )

        if query_count != document_count:
            raise ValueError(
                "Each query must have one aligned positive document"
            )


@dataclass
class TrainingStepOutput:
    total_loss: Tensor
    dense_loss: Tensor
    matryoshka_loss: Tensor
    multi_vector_loss: Tensor


class RetrievalTrainingObjective(nn.Module):
    """Joint dense, Matryoshka and MaxSim contrastive objective."""

    def __init__(
        self,
        *,
        dense_temperature: float = 0.05,
        multi_vector_temperature: float = 1.0,
        dense_weight: float = 1.0,
        matryoshka_weight: float = 1.0,
        multi_vector_weight: float = 1.0,
    ) -> None:
        super().__init__()

        if dense_temperature <= 0:
            raise ValueError("dense_temperature must be positive")

        if multi_vector_temperature <= 0:
            raise ValueError(
                "multi_vector_temperature must be positive"
            )

        weights = (
            dense_weight,
            matryoshka_weight,
            multi_vector_weight,
        )

        if any(weight < 0 for weight in weights):
            raise ValueError("Loss weights cannot be negative")

        if not any(weight > 0 for weight in weights):
            raise ValueError(
                "At least one loss weight must be positive"
            )

        self.dense_temperature = dense_temperature
        self.multi_vector_temperature = multi_vector_temperature

        self.dense_weight = dense_weight
        self.matryoshka_weight = matryoshka_weight
        self.multi_vector_weight = multi_vector_weight

    def forward(
        self,
        model: UnifiedEmbeddingModel,
        batch: RetrievalBatch,
    ) -> TrainingStepOutput:
        batch.validate()

        queries = model(
            batch.query_input_ids,
            batch.query_attention_mask,
        )

        documents = model(
            batch.document_input_ids,
            batch.document_attention_mask,
        )

        # The diagonal contains the aligned positives.
        dense_scores = model.score(
            queries,
            documents,
            method="dense",
        )

        dense_loss = contrastive_cross_entropy(
            dense_scores,
            temperature=self.dense_temperature,
        )

        # Train every Matryoshka prefix independently.
        prefix_losses = []

        for dimension in model.matryoshka_dims:
            scores = model.score(
                queries,
                documents,
                method="dense",
                dimension=dimension,
            )

            prefix_losses.append(
                contrastive_cross_entropy(
                    scores,
                    temperature=self.dense_temperature,
                )
            )

        matryoshka_loss = torch.stack(
            prefix_losses
        ).mean()

        # ColBERT MaxSim: [B, B] query-document scores.
        multi_vector_scores = model.score(
            queries,
            documents,
            method="multi_vector",
        )

        multi_vector_loss = contrastive_cross_entropy(
            multi_vector_scores,
            temperature=self.multi_vector_temperature,
        )

        total_loss = (
            self.dense_weight * dense_loss
            + self.matryoshka_weight * matryoshka_loss
            + self.multi_vector_weight * multi_vector_loss
        )

        return TrainingStepOutput(
            total_loss=total_loss,
            dense_loss=dense_loss,
            matryoshka_loss=matryoshka_loss,
            multi_vector_loss=multi_vector_loss,
        )


def train_one_step(
    model: UnifiedEmbeddingModel,
    batch: RetrievalBatch,
    objective: RetrievalTrainingObjective,
    optimizer: torch.optim.Optimizer,
    *,
    max_grad_norm: float = 1.0,
) -> TrainingStepOutput:
    """Perform one full optimization step."""

    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    output = objective(model, batch)

    if not torch.isfinite(output.total_loss):
        raise FloatingPointError(
            "Training loss is not finite"
        )

    output.total_loss.backward()

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and parameter.grad is not None
    ]

    if not parameters:
        raise RuntimeError(
            "No trainable parameters received gradients"
        )

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=max_grad_norm,
        error_if_nonfinite=True,
    )

    if not torch.isfinite(gradient_norm):
        raise FloatingPointError(
            "Gradient norm is not finite"
        )

    optimizer.step()

    return output
