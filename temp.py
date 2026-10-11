
"""Trainable token-level embedding heads.

Provides a ColBERT-style multi-vector projection head for
late-interaction retrieval.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from frontier_embedder.config import ModelConfig


class MultiVectorHead(nn.Module):
    """Project contextual token states into compact token embeddings.

    Input:
        hidden_states: [batch, tokens, hidden_size]
        attention_mask: [batch, tokens]

    Output:
        token_embeddings: [batch, tokens, multi_vector_dim]

    Valid nonzero token vectors are L2-normalized.
    Padding positions are exactly zero.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        self.projection = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.multi_vector_dim,
            bias=False,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the projection with Xavier initialization."""
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Produce masked, normalized token embeddings."""
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [B, T, H]"
            )

        batch_size, sequence_length, hidden_size = (
            hidden_states.shape
        )

        if hidden_size != self.config.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.config.hidden_size}, "
                f"received {hidden_size}"
            )

        if attention_mask.shape != (
            batch_size,
            sequence_length,
        ):
            raise ValueError(
                "attention_mask must have shape [B, T]"
            )

        if hidden_states.device != attention_mask.device:
            raise ValueError(
                "hidden_states and attention_mask must "
                "be on the same device"
            )

        if not hidden_states.is_floating_point():
            raise TypeError(
                "hidden_states must be a floating-point tensor"
            )

        if not torch.all(
            (attention_mask == 0) | (attention_mask == 1)
        ):
            raise ValueError(
                "attention_mask must contain only 0 and 1"
            )

        # A document with no valid tokens should be rejected.
        if torch.any(attention_mask.sum(dim=1) == 0):
            raise ValueError(
                "Each input must contain at least one valid token"
            )

        # Learned shared token projection.
        projected = self.projection(hidden_states)

        # Normalize in float32 for numerical stability.
        normalized = F.normalize(
            projected.float(),
            p=2,
            dim=-1,
        )

        # Zero padding positions after projection and normalization.
        mask = attention_mask.unsqueeze(-1).to(
            dtype=normalized.dtype
        )

        return normalized * mask
