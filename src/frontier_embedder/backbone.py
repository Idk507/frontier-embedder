
"""Qwen3 backbone with bidirectional attention for embedding learning.

Targets Hugging Face Transformers 4.57.3 with eager attention.

Important:
    The temporary mask override patches a module-level function.
    Concurrent forward passes from other Qwen3 model instances in
    the same Python process are not supported during the override.
"""

from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from typing import Iterator
from unittest.mock import patch

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel

from frontier_embedder.config import ModelConfig


_MASK_PATCH_LOCK = RLock()


class BidirectionalQwenBackbone(nn.Module):
    """Qwen3 representation backbone with noncausal attention."""

    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
    ) -> None:
        super().__init__()

        self.model = model
        self.config = config

        hidden_size = getattr(
            model.config, "hidden_size", None
        )

        if hidden_size != config.hidden_size:
            raise ValueError(
                f"Expected hidden size {config.hidden_size}, "
                f"received {hidden_size}"
            )

        if getattr(
            model.config, "_attn_implementation", None
        ) != "eager":
            raise ValueError(
                "BidirectionalQwenBackbone currently "
                "requires eager attention"
            )

    @classmethod
    def from_pretrained(
        cls,
        config: ModelConfig,
        *,
        dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
    ) -> "BidirectionalQwenBackbone":
        model_config = AutoConfig.from_pretrained(
            config.backbone_name,
            local_files_only=local_files_only,
        )

        if model_config.model_type != "qwen3":
            raise ValueError(
                "Only Qwen3 backbones are supported"
            )

        model = AutoModel.from_pretrained(
            config.backbone_name,
            config=model_config,
            dtype=dtype,
            local_files_only=local_files_only,
            attn_implementation="eager",
        )

        return cls(model, config)

    @staticmethod
    def _build_bidirectional_mask(
        attention_mask: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        """Construct a 4D additive padding-only attention mask.

        Shape: [B, 1, T, T]

        Valid keys receive 0.
        Padding keys receive the minimum finite dtype value.
        No triangular causal restriction is applied.
        """
        batch_size, seq_len = attention_mask.shape

        minimum = torch.finfo(dtype).min

        key_padding = (
            attention_mask[:, None, None, :] == 0
        )

        mask = torch.zeros(
            batch_size,
            1,
            seq_len,
            seq_len,
            dtype=dtype,
            device=attention_mask.device,
        )

        return mask.masked_fill(
            key_padding,
            minimum,
        )

    @contextmanager
    def _bidirectional_attention(
        self,
        attention_mask: Tensor,
    ) -> Iterator[None]:
        """Override Qwen3 model-level causal mask construction."""
        from transformers.models.qwen3 import modeling_qwen3

        if not hasattr(
            modeling_qwen3, "create_causal_mask"
        ):
            raise RuntimeError(
                "Expected create_causal_mask was not found. "
                "Check Transformers version and Qwen3 internals."
            )

        def create_bidirectional_mask(*args, **kwargs):
            return self._build_bidirectional_mask(
                attention_mask=attention_mask,
                dtype=self.model.dtype,
            )

        # This is a temporary compatibility approach.
        # The lock serializes calls using this wrapper, but cannot
        # protect unrelated Qwen3 instances outside the lock.
        with _MASK_PATCH_LOCK:
            with patch.object(
                modeling_qwen3,
                "create_causal_mask",
                create_bidirectional_mask,
            ):
                yield

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [B, T]"
            )

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must match input_ids shape"
            )

        if input_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError(
                "input_ids must contain integer token IDs"
            )

        if input_ids.device != attention_mask.device:
            raise ValueError(
                "input_ids and attention_mask "
                "must share a device"
            )

        if not torch.all(
            (attention_mask == 0) |
            (attention_mask == 1)
        ):
            raise ValueError(
                "attention_mask must be binary"
            )

        if torch.any(
            attention_mask.sum(dim=1) == 0
        ):
            raise ValueError(
                "Every sequence must contain "
                "at least one valid token"
            )

        with self._bidirectional_attention(
            attention_mask
        ):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )

        return outputs.last_hidden_state
