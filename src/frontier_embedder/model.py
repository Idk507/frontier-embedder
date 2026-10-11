
"""Unified dense, Matryoshka and multi-vector embedding model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from frontier_embedder.config import ModelConfig
from frontier_embedder.backbone import BidirectionalQwenBackbone


@dataclass
class EmbeddingOutput:
    """Representations produced by a shared encoder."""

    dense: Tensor
    matryoshka: dict[int, Tensor]
    multi_vector: Tensor
    attention_mask: Tensor


class UnifiedEmbeddingModel(nn.Module):
    """Shared encoder with dense and multi-vector retrieval heads."""

    def __init__(
        self,
        backbone: nn.Module,
        config: ModelConfig,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.config = config

        self.hidden_size = config.hidden_size
        self.dense_dim = config.dense_dim
        self.multi_vector_dim = config.multi_vector_dim
        self.matryoshka_dims = tuple(config.matryoshka_dims)

        if self.dense_dim != self.hidden_size:
            raise ValueError(
                "This phase requires dense_dim == hidden_size. "
                "A separate dense projection can be added later."
            )

        if not self.matryoshka_dims:
            raise ValueError("Matryoshka dimensions cannot be empty")

        if any(
            dim <= 0 or dim > self.dense_dim
            for dim in self.matryoshka_dims
        ):
            raise ValueError("Invalid Matryoshka dimension")

        if len(set(self.matryoshka_dims)) != len(
            self.matryoshka_dims
        ):
            raise ValueError("Matryoshka dimensions must be unique")

        if self.multi_vector_dim <= 0:
            raise ValueError(
                "multi_vector_dim must be positive"
            )

        self.token_projection = nn.Linear(
            self.hidden_size,
            self.multi_vector_dim,
            bias=False,
        )

        nn.init.xavier_uniform_(
            self.token_projection.weight
        )

    @classmethod
    def from_pretrained(
        cls,
        config: ModelConfig,
        *,
        dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
    ) -> "UnifiedEmbeddingModel":
        backbone = BidirectionalQwenBackbone.from_pretrained(
            config,
            dtype=dtype,
            local_files_only=local_files_only,
        )
        return cls(backbone, config)

    def freeze_backbone(self) -> None:
        """Freeze encoder parameters while keeping heads trainable."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        """Restore encoder parameter training."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(True)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> EmbeddingOutput:
        """Generate all retrieval representations.

        Args:
            input_ids: [B, T]
            attention_mask: [B, T], binary

        Returns:
            EmbeddingOutput containing dense [B, D],
            Matryoshka prefixes and token vectors [B, T, V].
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must match input_ids shape"
            )

        if not torch.all(
            (attention_mask == 0) |
            (attention_mask == 1)
        ):
            raise ValueError("attention_mask must be binary")

        if torch.any(attention_mask.sum(dim=1) == 0):
            raise ValueError(
                "Every sequence needs at least one valid token"
            )

        hidden_states = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if hidden_states.ndim != 3:
            raise ValueError(
                "Backbone must return [B, T, H]"
            )

        if hidden_states.shape != (
            *input_ids.shape,
            self.hidden_size,
        ):
            raise ValueError(
                "Backbone hidden-state shape mismatch"
            )

        # Accumulate pooled representations in FP32 for stability.
        hidden_float = hidden_states.float()

        valid_tokens = attention_mask.unsqueeze(-1).to(
            hidden_float.dtype
        )

        summed = (
            hidden_float * valid_tokens
        ).sum(dim=1)

        counts = valid_tokens.sum(dim=1).clamp_min(1.0)

        dense = F.normalize(
            summed / counts,
            p=2,
            dim=-1,
        )

        # Each prefix must be normalized independently.
        matryoshka = {
            dim: F.normalize(
                dense[:, :dim],
                p=2,
                dim=-1,
            )
            for dim in self.matryoshka_dims
        }

        # Project tokens using the head's parameter dtype.
        projected = self.token_projection(
            hidden_states.to(
                self.token_projection.weight.dtype
            )
        )

        multi_vector = F.normalize(
            projected.float(),
            p=2,
            dim=-1,
        )

        # Padded positions must be exactly zero.
        multi_vector = multi_vector * valid_tokens

        return EmbeddingOutput(
            dense=dense,
            matryoshka=matryoshka,
            multi_vector=multi_vector,
            attention_mask=attention_mask,
        )

    @staticmethod
    def score(
        queries: EmbeddingOutput,
        documents: EmbeddingOutput,
        *,
        method: Literal["dense", "multi_vector"] = "dense",
        dimension: int | None = None,
    ) -> Tensor:
        """Compute all-pairs retrieval scores [N, M].

        Dense scoring uses cosine similarity.
        Multi-vector scoring uses masked ColBERT MaxSim.
        """
        if method == "dense":
            if dimension is None:
                q = queries.dense
                d = documents.dense
            else:
                if (
                    dimension not in queries.matryoshka
                    or dimension not in documents.matryoshka
                ):
                    raise ValueError(
                        "Requested Matryoshka dimension unavailable"
                    )

                q = queries.matryoshka[dimension]
                d = documents.matryoshka[dimension]

            return q.float() @ d.float().T

        if method != "multi_vector":
            raise ValueError(
                "method must be 'dense' or 'multi_vector'"
            )

        q = queries.multi_vector.float()
        d = documents.multi_vector.float()

        if q.ndim != 3 or d.ndim != 3:
            raise ValueError(
                "Multi-vector embeddings must be 3D"
            )

        if q.shape[-1] != d.shape[-1]:
            raise ValueError(
                "Multi-vector dimensions must match"
            )

        if q.device != d.device:
            raise ValueError(
                "Query and document tensors must share a device"
            )

        qmask = queries.attention_mask.bool()
        dmask = documents.attention_mask.bool()

        if qmask.shape != q.shape[:2]:
            raise ValueError("Invalid query mask shape")

        if dmask.shape != d.shape[:2]:
            raise ValueError("Invalid document mask shape")

        if not dmask.any(dim=1).all():
            raise ValueError(
                "Documents must contain valid tokens"
            )

        # [N, M, Q, D]
        similarities = torch.einsum(
            "nqh,mdh->nmqd",
            q,
            d,
        )

        minimum = torch.finfo(
            similarities.dtype
        ).min

        similarities = similarities.masked_fill(
            ~dmask[None, :, None, :],
            minimum,
        )

        # Maximum document-token similarity for each query token.
        token_scores = similarities.max(dim=-1).values

        # Sum over valid query tokens only.
        return (
            token_scores
            * qmask[:, None, :].to(token_scores.dtype)
        ).sum(dim=-1)
