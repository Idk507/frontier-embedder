"""Dense pooling and Matryoshka embedding utilities."""

from __future__ import annotations

import torch 
from torch import Tensor, nn
import torch.nn.functional as F

from frontier_embedder.config import ModelConfig

class DensePooling(nn.Module):
    """
    Mask aware mean pooling with optional L2 normalization.
    Understanding why we need **masked pooling** combined with **L2 normalization** in embedding models (like your dense pooling module for Qwen3) comes down to two major challenges in vector representation: variable text lengths and geometric distance scaling.

    Here is a breakdown of why both steps are critical:

    ---

    ### 1. Why Masked Pooling?

    Transformer backbones output a sequence of token representations with the shape **[Batch, Sequence Length, Hidden Size]** (e.g., [2, 8, 1024]). To get a single fixed-size vector for a whole sentence or document, we need to pool these token vectors together (usually via mean pooling).

    However, sentences have different lengths, which requires **padding** to make batching possible in PyTorch.

    * **The Problem with Standard Pooling:** If you take the unweighted average of all hidden states, the padding tokens (which are just empty filler added to make sequences equally long) will be included in the math. Because padding tokens carry meaningless representations, they will **skew the average**, pulling the final sentence embedding off-center and ruining the semantic representation.
    * **The Role of Masking:** Masked pooling multiplies the hidden states by the attention mask (which contains `1` for real tokens and `0` for padding) before averaging. This ensures that **only valid, actual content tokens contribute to the final embedding**, while padding vectors are completely ignored.

    ---

    ### 2. Why L2 Normalization?

    Once you have your pooled 1024-dimensional vector, applying L2 normalization forces the vector's Euclidean length (norm) to exactly `1.0`.

    * **Uniform Scale for Similarity Search:** Embedding models typically measure semantic similarity using **Cosine Similarity** (the angle between two vectors) rather than raw Euclidean distance. When all vectors are L2-normalized to unit length ($\vert{}\vert{}x\vert{}\vert{}_2 = 1$), the cosine similarity between two vectors simplifies mathematically to their **Dot Product** ($A \cdot B$).
    * **Computational Efficiency:** Dot products are vastly faster to compute at scale in vector databases (like Milvus, Qdrant, or Pinecone) than full cosine distance formulas.
    * **Matryoshka Representation Learning (MRL) Compatibility:** As noted in your implementation, if you want to truncate your embeddings to smaller sizes (like 512, 256, or 128 dimensions) to save space and speed up retrieval, **truncation changes the vector's length**. Re-applying L2 normalization right after truncation ensures that the smaller-dimensional prefix vectors remain valid, unit-length embeddings that can still be accurately compared using simple dot products.

    ---

    ### Summary Workflow

    1. **Backbone:** Generates contextual token vectors for all words in the input.
    2. **Masked Pooling:** Averages valid tokens while ignoring padding, yielding a clean document vector.
    3. **L2 Normalization:** Scales the vector to a unit length of 1, aligning it perfectly for fast dot-product similarity search and MRL slicing.    
    """
    def __init__(self, config : ModelConfig)-> None : 
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
            attention_mask: Tensor

        ) -> Tensor : 
        """Convert token representations into document embeddings.
        
                Args:
                    hidden_states: Tensor of shape [B, T, H].
                    attention_mask: Tensor of shape [B, T].
                        1 indicates a valid token; 0 indicates padding.
        
                Returns:
                    Dense embeddings of shape [B, dense_dim].
        """
        if hidden_states.ndim != 3 : # (batch,sequence length, hidden size)
            raise ValueError(
                "hidden_states must have shape [B, T, H]"
            
            )
        
        batch_size, sequence_length, hidden_size = (
            hidden_states.shape
        )

        if hidden_size != self.config.hidden_size : 
            raise ValueError(
                f"Expected hidden size {self.config.hidden_size}, "
                f"received {hidden_size}"
            )

        if attention_mask.shape != (
            batch_size, sequence_length
        ) :
            raise ValueError(
                "attention_mask must have shape [B, T]"
            )

        if attention_mask.device != hidden_states.device : 
            raise ValueError(
                "attention_mask and hidden_states must "
                "be on the same device"
            )

        if not torch.all(
            (attention_mask == 0) | (attention_mask == 1)
        ) :
            raise ValueError(
                "attention_mask must contain only 0s and 1s"
            
        )

        valid_counts = attention_mask.sum(dim=1, keepdim=True)  # dim=1 means sum across the columns (row-by-row).

        if torch.any(valid_counts == 0) :
            raise ValueError(
                "All sequences must have at least one valid token (non-padding)."
            )

        mask = attention_mask.unsqueeze(-1).to(
            dtype=hidden_states.dtype
        )  # Shape: [B, T, H] , unsqueeze(-1) adds a new dimension at the end, and expand_as replicates the mask across the hidden size dimension to match hidden_states shape.

        masked_states = hidden_states * mask  # Element-wise multiplication, zeroing out padding token representations.
        summed = masked_states.sum(dim=1)  # Sum across the sequence length dimension, resulting in shape [B, H].

        counts = mask.sum(dim=1).clamp(min=1)  # Shape: [B, 1], counts of valid tokens for each sequence and clamp ensures no division by zero and min=1 ensures that we don't divide by zero in case of an empty sequence (though we already checked for valid counts above).
        embeddings = summed / counts  # Element-wise division, resulting in shape [B, H]. This is the mean pooling step.

        if self.config.normalize_embeddings:
            embeddings = F.normalize(
                embeddings.float(), p=2, dim= -1
            ) # L2 normalization along the last dimension (hidden size), ensuring that each embedding vector has a unit norm. This is important for similarity search and Matryoshka representation learning, as discussed in the class docstring.

        return embeddings 



class MatryoshkaEmbedding(nn.Module):
    """
    Matryoshka embedding module for hierarchical representation learning.
    This module takes a dense embedding and produces multiple truncated embeddings of varying sizes, allowing for efficient storage and retrieval in vector databases.

    The term "Matryoshka" refers to the Russian nesting dolls, where each smaller doll fits inside a larger one. Similarly, this module generates embeddings of decreasing dimensions that can be used for different levels of granularity in semantic search.

    Args:
        config (ModelConfig): Configuration object containing model parameters.
    """
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
            self, 
            embeddings : Tensor ,
            dimensions : int | None = None 
    ) -> Tensor | dict[int, Tensor] : 
        """
        Return one prefix or all configured Matryoshka prefixes.

        Args:
            embeddings (Tensor): Input dense embeddings of shape [B, H].
            dimensions (int | None): If specified, returns the embedding truncated to this dimension. 
                                     If None, returns a dictionary of all configured Matryoshka prefixes.

        Returns:
            Tensor | dict[int, Tensor]: If dimensions is specified, returns a single truncated embedding of
        shape [B, dimensions]. If dimensions is None, returns a dictionary where keys are the configured dimensions and values are the corresponding truncated embeddings.
        """

        if embeddings.ndim != 2 : 
            raise ValueError(
                "embeddings must have shape [B, H]"
            )
        if embeddings.shape[1] != self.config.dense_dim : 
            raise ValueError(
                f"Expected dense dimension {self.config.dense_dim}, "
                f"received {embeddings.shape[1]}"
            )

        if dimensions is not None : 
            if dimensions not in self.config.matryoshka_dims:
                raise ValueError(
                    f"Unsupported dimension: {dimensions}"
                )

            return F.normalize(
                embeddings[:, :dimensions].float(),
                p=2,
                dim=-1,
            )   # L2 normalization of the truncated embedding to ensure it has unit length, which is important for similarity search and Matryoshka representation learning.

        return {
            dim : F.normalize(
                embeddings[:, :dim].float(), # Truncate the embeddings to the first `dim` dimensions. This is done by slicing the tensor along the last dimension.
                p=2,
                dim=-1,
            ) for dim in self.config.matryoshka_dims
        }  # Dictionary comprehension that generates truncated embeddings for all configured Matryoshka dimensions, each normalized to unit length.
