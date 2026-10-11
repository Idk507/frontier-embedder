
"""Qwen3 bidirectional embedding backbone.

Compatible with the eager-attention Qwen3 implementation in
Transformers 4.57.3.

Uses instance-scoped forward pre-hooks rather than global
monkey-patching.

The implementation targets full-sequence embedding inference
and training, not autoregressive generation or KV caching.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel

from frontier_embedder.config import ModelConfig


class BidirectionalQwenBackbone(nn.Module):
    """Qwen3 backbone with bidirectional self-attention."""

    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
    ) -> None:
        super().__init__()

        self.model = model
        self.config = config

        hidden_size = getattr(
            model.config,
            "hidden_size",
            None,
        )

        if hidden_size != config.hidden_size:
            raise ValueError(
                f"Expected hidden size {config.hidden_size}, "
                f"received {hidden_size}"
            )

        implementation = getattr(
            model.config,
            "_attn_implementation",
            None,
        )

        if implementation != "eager":
            raise ValueError(
                "BidirectionalQwenBackbone requires "
                "eager attention"
            )

        model_type = getattr(
            model.config,
            "model_type",
            None,
        )

        if model_type != "qwen3":
            raise ValueError(
                "Only Qwen3 models are supported"
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
            trust_remote_code=False,
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
            trust_remote_code=False,
        )

        return cls(model, config)

    @staticmethod
    def _build_bidirectional_mask(
        attention_mask: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return additive mask of shape [B, 1, T, T]."""
        if attention_mask.ndim != 2:
            raise ValueError(
                "attention_mask must be [B, T]"
            )

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

    def _attention_modules(self) -> list[nn.Module]:
        """Find attention layers owned by this model."""
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3Attention,
        )

        modules = [
            module
            for module in self.model.modules()
            if isinstance(module, Qwen3Attention)
        ]

        if not modules:
            raise RuntimeError(
                "No Qwen3Attention modules found"
            )

        return modules

    @contextmanager
    def _bidirectional_attention(
        self,
        attention_mask: Tensor,
    ) -> Iterator[None]:
        """Install temporary, instance-local mask overrides."""
        mask = self._build_bidirectional_mask(
            attention_mask,
            dtype=self.model.dtype,
        )

        def replace_attention_mask(
            module: nn.Module,
            args: tuple,
            kwargs: dict,
        ):
            # Qwen3Attention receives attention_mask
            # as a keyword argument in the supported
            # Transformers implementation.
            if "attention_mask" not in kwargs:
                raise RuntimeError(
                    "Qwen3 attention signature changed: "
                    "attention_mask keyword missing"
                )

            updated_kwargs = dict(kwargs)
            updated_kwargs["attention_mask"] = mask

            return args, updated_kwargs

        handles = []

        try:
            for module in self._attention_modules():
                handle = module.register_forward_pre_hook(
                    replace_attention_mask,
                    with_kwargs=True,
                )
                handles.append(handle)

            yield

        finally:
            for handle in handles:
                handle.remove()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must be [B, T]"
            )

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask shape mismatch"
            )

        if input_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError(
                "input_ids must contain integer IDs"
            )

        if input_ids.device != attention_mask.device:
            raise ValueError(
                "Input tensors must share a device"
            )

        if not torch.all(
            (attention_mask == 0)
            | (attention_mask == 1)
        ):
            raise ValueError(
                "attention_mask must be binary"
            )

        if torch.any(
            attention_mask.sum(dim=1) == 0
        ):
            raise ValueError(
                "Each sequence needs a valid token"
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
