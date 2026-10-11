
"""ColBERT-style late-interaction MaxSim scoring.

Supports aligned query-document pairs and all-pairs similarity
matrices with padding-aware, differentiable computation.


Query embeddings:       [B, Q, D]
Document embeddings:    [B, T, D]
                               |
                               v
Similarity matrix:      [B, Q, T]
                               |
                      Document padding mask
                               |
                               v
Max over document T:    [B, Q]
                               |
                        Query padding mask
                               |
                               v
Sum over query Q:       [B]


"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MaxSimInteraction(nn.Module):
    """Compute token-level late-interaction relevance scores.

    Inputs must contain L2-normalized token embeddings.
    Padding masks use 1 for valid tokens and 0 for padding.

    Document padding is excluded before max reduction.
    Query padding is excluded before sum reduction.
    """

    def __init__(
        self,
        *,
        document_chunk_size: int = 16,
    ) -> None:
        super().__init__()

        if document_chunk_size <= 0:
            raise ValueError(
                "document_chunk_size must be positive"
            )

        self.document_chunk_size = document_chunk_size

    @staticmethod
    def _validate(
        embeddings: Tensor,
        mask: Tensor,
        name: str,
    ) -> None:
        if embeddings.ndim != 3:
            raise ValueError(
                f"{name} embeddings must have shape [B, T, D]"
            )

        if not embeddings.is_floating_point():
            raise TypeError(
                f"{name} embeddings must be floating point"
            )

        if mask.shape != embeddings.shape[:2]:
            raise ValueError(
                f"{name} mask must have shape [B, T]"
            )

        if mask.device != embeddings.device:
            raise ValueError(
                f"{name} mask and embeddings must share a device"
            )

        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError(
                f"{name} mask must contain only 0 and 1"
            )

        if torch.any(mask.sum(dim=-1) == 0):
            raise ValueError(
                f"{name} contains an all-padding sequence"
            )

    @staticmethod
    def _maxsim(
        similarities: Tensor,
        query_mask: Tensor,
        document_mask: Tensor,
    ) -> Tensor:
        """Reduce similarities [..., Q, T] to [...]."""

        # Exclude invalid document tokens before taking maxima.
        similarities = similarities.masked_fill(
            ~document_mask.bool(),
            float("-inf"),
        )

        # Best document-token match for each query token.
        maximum = similarities.max(dim=-1).values

        # Exclude padded query tokens from the final sum.
        maximum = maximum.masked_fill(
            ~query_mask.bool(),
            0.0,
        )

        return maximum.sum(dim=-1)

    def forward(
        self,
        query_embeddings: Tensor,
        document_embeddings: Tensor,
        query_mask: Tensor,
        document_mask: Tensor,
        *,
        all_pairs: bool = False,
    ) -> Tensor:
        """Compute MaxSim relevance scores.

        Args:
            query_embeddings: [N, Q, D].
            document_embeddings: [M, T, D].
            query_mask: [N, Q].
            document_mask: [M, T].
            all_pairs: If True, score every query against every
                document and return [N, M]. Otherwise require N=M
                and return [N].
        """
        self._validate(
            query_embeddings, query_mask, "query"
        )
        self._validate(
            document_embeddings, document_mask, "document"
        )

        if query_embeddings.device != document_embeddings.device:
            raise ValueError(
                "Query and document embeddings must share a device"
            )

        if query_embeddings.shape[-1] != document_embeddings.shape[-1]:
            raise ValueError(
                "Query and document embedding dimensions must match"
            )

        if not all_pairs:
            if query_embeddings.shape[0] != document_embeddings.shape[0]:
                raise ValueError(
                    "Aligned scoring requires matching batch sizes"
                )

            # [B, Q, D] x [B, D, T] -> [B, Q, T]
            similarities = torch.bmm(
                query_embeddings.float(),
                document_embeddings.float().transpose(1, 2),
            )

            return self._maxsim(
            similarities,
            query_mask,
            document_mask.unsqueeze(1),
        )

        # All-pairs scoring: [N, M].
        # Chunk document batches to limit intermediate memory.
        scores = []

        for start in range(
            0,
            document_embeddings.shape[0],
            self.document_chunk_size,
        ):
            end = start + self.document_chunk_size

            documents = document_embeddings[start:end]
            masks = document_mask[start:end]

            # [N, 1, Q, D] x [1, C, D, T]
            # -> [N, C, Q, T]
            similarities = torch.matmul(
                query_embeddings.float().unsqueeze(1),
                documents.float()
                .transpose(-1, -2)
                .unsqueeze(0),
            )

            chunk_scores = self._maxsim(
                similarities,
                query_mask[:, None, :],
                masks[None, :, None, :],
            )

            scores.append(chunk_scores)

        return torch.cat(scores, dim=1)
