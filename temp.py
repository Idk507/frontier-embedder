
"""Dense pooling and Matryoshka embedding utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
 
from frontier_embedder.config import ModelConfig


class DensePooling(nn.Module):
    """Mask-aware mean pooling with optional L2 normalization."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        if config.dense_dim != config.hidden_size:
            raise ValueError(
                "DensePooling currently requires dense_dim == hidden_size. "
                "A learned projection is needed for other dimensions."
            )

        self.config = config

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Convert token representations into document embeddings.

        Args:
            hidden_states: Tensor of shape [B, T, H].
            attention_mask: Tensor of shape [B, T].
                1 indicates a valid token; 0 indicates padding.

        Returns:
            Dense embeddings of shape [B, dense_dim].
        """
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

        if attention_mask.device != hidden_states.device:
            raise ValueError(
                "attention_mask and hidden_states must "
                "be on the same device"
            )

        if not torch.all(
            (attention_mask == 0) | (attention_mask == 1)
        ):
            raise ValueError(
                "attention_mask must contain only 0 and 1"
            )

        valid_counts = attention_mask.sum(dim=1)

        if torch.any(valid_counts == 0):
            raise ValueError(
                "Each input must contain at least one valid token"
            )

        mask = attention_mask.unsqueeze(-1).to(
            dtype=hidden_states.dtype
        )

        masked_states = hidden_states * mask
        summed = masked_states.sum(dim=1)

        counts = mask.sum(dim=1).clamp(min=1)
        embeddings = summed / counts

        if self.config.normalize_embeddings:
            embeddings = F.normalize(
                embeddings.float(),
                p=2,
                dim=-1,
            )

        return embeddings


class MatryoshkaEmbedding(nn.Module):
    """Extract normalized prefix embeddings at configured dimensions."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        embeddings: Tensor,
        dimensions: int | None = None,
    ) -> Tensor | dict[int, Tensor]:
        """Return one prefix or all configured Matryoshka prefixes.

        A prefix is normalized independently after truncation.
        """
        if embeddings.ndim != 2:
            raise ValueError(
                "embeddings must have shape [B, dense_dim]"
            )

        if embeddings.shape[-1] != self.config.dense_dim:
            raise ValueError(
                f"Expected dense dimension {self.config.dense_dim}, "
                f"received {embeddings.shape[-1]}"
            )

        if dimensions is not None:
            if dimensions not in self.config.matryoshka_dims:
                raise ValueError(
                    f"Unsupported dimension: {dimensions}"
                )

            return F.normalize(
                embeddings[:, :dimensions].float(),
                p=2,
                dim=-1,
            )

        return {
            dim: F.normalize(
                embeddings[:, :dim].float(),
                p=2,
                dim=-1,
            )
            for dim in self.config.matryoshka_dims
        }
