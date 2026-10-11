
"""Qwen3 backbone with verified bidirectional attention support.

The attention-mask override is installed only for the duration of a
forward pass. The model is intended for representation learning,
not autoregressive text generation.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel

from frontier_embedder.config import ModelConfig


class BidirectionalQwenBackbone(nn.Module):
    """Expose contextual Qwen3 hidden states with noncausal attention.

    The model is loaded lazily through ``from_pretrained`` and starts
    on CPU unless the caller explicitly moves it elsewhere.
    """

    def __init__(self, model: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config

        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size != config.hidden_size:
            raise ValueError(
                f"Hidden-size mismatch: expected {config.hidden_size}, "
                f"received {hidden_size}"
            )

    @classmethod
    def from_pretrained(
        cls,
        config: ModelConfig,
        *,
        dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
    ) -> "BidirectionalQwenBackbone":
        """Load pretrained Qwen3 weights without a language-model head."""
        model_config = AutoConfig.from_pretrained(
            config.backbone_name,
            local_files_only=local_files_only,
        )

        if getattr(model_config, "model_type", None) != "qwen3":
            raise ValueError(
                "This backbone currently supports Qwen3 models only."
            )

        model = AutoModel.from_pretrained(
            config.backbone_name,
            config=model_config,
            dtype=dtype,
            local_files_only=local_files_only,
            attn_implementation="eager",
        )
        return cls(model, config)

    @contextmanager
    def _bidirectional_attention(self) -> Iterator[None]:
        """Temporarily disable causality at the attention-module level.

        Qwen3 eager attention reads the causal behavior from the
        attention module. We update those flags and restore them
        afterward. This approach is intentionally restricted to
        eager attention; SDPA and FlashAttention require separate
        implementation-specific handling.
        """
        attention_modules = [
            module
            for module in self.model.modules()
            if module.__class__.__name__ == "Qwen3Attention"
        ]

        if not attention_modules:
            raise RuntimeError(
                "No Qwen3Attention modules found. "
                "Check the installed Transformers implementation."
            )

        previous_states = [
            getattr(module, "is_causal", None)
            for module in attention_modules
        ]

        try:
            for module in attention_modules:
                module.is_causal = False
            yield
        finally:
            for module, previous in zip(
                attention_modules, previous_states
            ):
                if previous is None:
                    delattr(module, "is_causal")
                else:
                    module.is_causal = previous

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Return contextual embeddings with shape [batch, tokens, hidden].

        Args:
            input_ids: Integer token IDs of shape [B, T].
            attention_mask: Binary mask of shape [B, T], where 1
                represents a valid token and 0 represents padding.

        Returns:
            Last hidden states of shape [B, T, hidden_size].
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must match input_ids shape"
            )

        with self._bidirectional_attention():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )

        return outputs.last_hidden_state
