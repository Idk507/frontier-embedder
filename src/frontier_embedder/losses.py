"""Joint contrastive losses for dense and multi-vector retrieval."""

from __future__ import annotations
from dataclasses import dataclass
import torch 
import torch.nn.functional as F
from torch import Tensor,nn

from frontier_embedder.interaction import MaxSimInteraction

@dataclass
class LossOutput:
    total: Tensor
    dense: Tensor
    matryoshka: Tensor
    multi_vector: Tensor


def contrastive_cross_entropy(
        scores: Tensor,
        temperature: float = 0.05,
) -> Tensor:
    """InfoNCE with diagonal query-document positives.
    
        scores[i, i] is the positive score for query i.
        All other columns are treated as negatives.
    """
    if scores.ndim != 2 :
        raise ValueError("scores must have shape [N, M]")

    n_queries, n_documents = scores.shape

    if n_queries < 1 or n_documents < 2:
        raise ValueError(
            "Expected at least one query and two documents"
        )

    if n_documents < n_queries:
        raise ValueError(
            "Each query must have a diagonal positive document"
        )

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")

    targets = torch.arange(
        n_queries,
        device=scores.device,
    )

    return F.cross_entropy(
        scores.float() / temperature,   
        targets ,
        ) # computes the cross entropy loss between the predicted scores and the target labels


class DenseContrastiveLoss(nn.Module):
    """Contrastive loss for dense retrieval.

    Each query has a single positive document.
    All other documents are treated as negatives.
    """

    def __init__(self, temperature: float = 0.05) -> None :
        super().__init__()
        if temperature <=0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(
            self, 
            query_embeddings: Tensor,
            document_embeddings: Tensor,
    ) -> Tensor : 
        if query_embeddings.ndim != 2:
            raise ValueError("query_embeddings must have shape [N, D]")
        if document_embeddings.ndim != 2:
            raise ValueError("document_embeddings must have shape [M, D]")
        if query_embeddings.shape[1] != document_embeddings.shape[1]:
            raise ValueError(
                "query_embeddings and document_embeddings must have the same dimension"
            )

        if query_embeddings.device != document_embeddings.device:
            raise ValueError(
                "query_embeddings and document_embeddings must be on the same device"
            )

        queries = F.normalize(
            query_embeddings.float(),
            p=2, # p is the order of the norm, p=2 corresponds to the Euclidean norm
            dim=-1,  # dim=-1 means that the normalization is applied along the last dimension of the tensor, which corresponds to the embedding dimension
        )

        documents = F.normalize(
            document_embeddings.float(),
            p = 2,
            dim=-1,
        )

        scores = queries @ documents.T # computes the dot product between the normalized query and document embeddings, resulting in a similarity score matrix of shape [N, M]

        return contrastive_cross_entropy(
            scores,
            temperature=self.temperature,
        )


class MatryoshkaContrastiveLoss(nn.Module):
    """Average contrastive loss across embedding prefix dimensions."""
    def __init__(
            self,
            dimensions : tuple[int, ...] = (128,256,512,1024),
            temperature: float = 0.05,
    ) -> None:
        super().__init__()

        if not dimensions:
            raise ValueError("dimensions must be non-empty")

        if len(set(dimensions)) != len(dimensions):
            raise ValueError("dimensions must be unique")

        if any(d <= 0 for d in dimensions):
            raise ValueError("dimensions must be positive")

        self.dimensions = dimensions
        self.dense_loss = DenseContrastiveLoss(temperature=temperature)

    def forward(
            self,
            query_embeddings: Tensor,
            document_embeddings: Tensor,
        ) ->  Tensor:
        if query_embeddings.ndim != 2 or document_embeddings.ndim != 2:
            raise ValueError(
                "query_embeddings and document_embeddings must have shape [N, D] and [M, D]"
            )

        if max(self.dimensions) > min(
            query_embeddings.shape[1], document_embeddings.shape[1]
        ):
            raise ValueError(
                "max(dimensions) must be <= min(embedding dimension) and MRL dimension exceeds embedding dimension width"
            )

        losses = []

        for dimension in self.dimensions:
            loss = self.dense_loss(
                query_embeddings[:, :dimension],
                document_embeddings[:, :dimension],
            )
            losses.append(loss)

        return torch.stack(losses).mean() # returns the average of the contrastive losses computed for each specified dimension, providing a single scalar loss value that reflects the overall performance across all dimensions


class MultiVectorContrastiveLoss(nn.Module):
    """ InfoNCE over ColBERT style all pairs MaxSim scores. """

    def __init__(
            self,
            temperature: float = 0.05,
            document_chunk_size: int = 16,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.temperature = temperature
        self.interaction = MaxSimInteraction(
            document_chunk_size=document_chunk_size
        )

    def forward(
            self,
            query_embeddings: Tensor,
            document_embeddings: Tensor,
            query_mask: Tensor,
            document_mask: Tensor,
    ) -> Tensor : 
        scores = self.interaction(
            query_embeddings,
            document_embeddings,
            query_mask,
            document_mask,
            all_pairs=True,
        )
        return contrastive_cross_entropy(
            scores,
            temperature=self.temperature,
        )



class JointRetrievalLoss(nn.Module):
    """Combine dense, Matryoshka and multi-vector objectives."""

    def __init__(
        self,
        dense_weight: float = 1.0,
        matryoshka_weight: float = 1.0,
        multi_vector_weight: float = 1.0,
        dense_temperature: float = 0.05,
        multi_vector_temperature: float = 1.0,
        matryoshka_dimensions: tuple[int, ...] = (
            128, 256, 512, 1024
        ),
    ) -> None:
        super().__init__()

        weights = (
            dense_weight,
            matryoshka_weight,
            multi_vector_weight,
        )

        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError(
                "Loss weights must be nonnegative "
                "and at least one must be positive"
            )

        self.dense_weight = dense_weight
        self.matryoshka_weight = matryoshka_weight
        self.multi_vector_weight = multi_vector_weight

        self.dense_loss = DenseContrastiveLoss(
            dense_temperature
        )

        self.matryoshka_loss = MatryoshkaContrastiveLoss(
            dimensions=matryoshka_dimensions,
            temperature=dense_temperature,
        )

        self.multi_vector_loss = MultiVectorContrastiveLoss(
            temperature=multi_vector_temperature
        )

    def forward(
        self,
        query_dense: Tensor,
        document_dense: Tensor,
        query_tokens: Tensor,
        document_tokens: Tensor,
        query_mask: Tensor,
        document_mask: Tensor,
    ) -> LossOutput:
        zero = query_dense.sum() * 0.0

        dense = (
            self.dense_loss(query_dense, document_dense)
            if self.dense_weight > 0
            else zero
        )

        matryoshka = (
            self.matryoshka_loss(
                query_dense, document_dense
            )
            if self.matryoshka_weight > 0
            else zero
        )

        multi_vector = (
            self.multi_vector_loss(
                query_tokens,
                document_tokens,
                query_mask,
                document_mask,
            )
            if self.multi_vector_weight > 0
            else zero
        )

        total = (
            self.dense_weight * dense
            + self.matryoshka_weight * matryoshka
            + self.multi_vector_weight * multi_vector
        )

        return LossOutput(
            total=total,
            dense=dense,
            matryoshka=matryoshka,
            multi_vector=multi_vector,
        )

        