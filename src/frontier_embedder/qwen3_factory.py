
"""Factory for real Qwen3 backbone verification."""

import torch

from frontier_embedder.backbone import (
    BidirectionalQwenBackbone,
)
from frontier_embedder.config import ModelConfig


def create_backbone(
    *,
    model_name: str,
    device: torch.device,
):
    config = ModelConfig(
        backbone_name=model_name,
        hidden_size=1024,
    )

    dtype = (
        torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    backbone = BidirectionalQwenBackbone.from_pretrained(
        config,
        dtype=dtype,
    )

    return backbone.to(device).eval()
