
"""Behavioral correctness and memory diagnostics for embedding backbones."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BidirectionalityResult:
    earlier_token_max_difference: float
    later_token_max_difference: float
    earlier_token_responds: bool
    later_token_responds: bool


@dataclass(frozen=True)
class PaddingResult:
    max_valid_token_difference: float
    invariant: bool


@dataclass(frozen=True)
class MemoryProfile:
    device: str
    elapsed_seconds: float
    peak_allocated_mb: float | None
    peak_reserved_mb: float | None
    output_shape: tuple[int, ...]


def extract_hidden_states(output: Any) -> Tensor:
    """Accept a hidden-state tensor or Hugging Face model output."""
    if isinstance(output, Tensor):
        hidden = output
    elif hasattr(output, "last_hidden_state"):
        hidden = output.last_hidden_state
    elif (
        isinstance(output, (tuple, list))
        and len(output) > 0
    ):
        hidden = output[0]
    else:
        raise TypeError(
            "Backbone must return a tensor or last_hidden_state"
        )

    if not isinstance(hidden, Tensor) or hidden.ndim != 3:
        raise ValueError(
            "Expected hidden states with shape [B, T, H]"
        )

    return hidden


@torch.inference_mode()
def check_bidirectionality(
    backbone: nn.Module,
    *,
    original_ids: Tensor,
    modified_ids: Tensor,
    attention_mask: Tensor,
    earlier_position: int,
    later_position: int,
    tolerance: float = 1e-5,
) -> BidirectionalityResult:
    """Check whether changing a later token affects an earlier one.

    Inputs must differ only at later_position.
    """
    if tolerance < 0:
        raise ValueError("tolerance cannot be negative")

    if original_ids.ndim != 2:
        raise ValueError("Input IDs must be [B, T]")

    if original_ids.shape != modified_ids.shape:
        raise ValueError("Input shapes must match")

    if attention_mask.shape != original_ids.shape:
        raise ValueError("Attention mask shape mismatch")

    if original_ids.shape[0] != 1:
        raise ValueError("Use batch size 1 for this diagnostic")

    length = original_ids.shape[1]

    if not (0 <= earlier_position < later_position < length):
        raise ValueError(
            "Positions must satisfy earlier < later"
        )

    if (
        attention_mask[0, earlier_position].item() != 1
        or attention_mask[0, later_position].item() != 1
    ):
        raise ValueError("Diagnostic positions must be unmasked")

    changed = original_ids != modified_ids

    if changed.sum().item() != 1:
        raise ValueError(
            "Exactly one token must change"
        )

    if not changed[0, later_position].item():
        raise ValueError(
            "Only later_position may change"
        )

    device = next(backbone.parameters()).device

    original_ids = original_ids.to(device)
    modified_ids = modified_ids.to(device)
    attention_mask = attention_mask.to(device)

    was_training = backbone.training
    backbone.eval()

    try:
        original = extract_hidden_states(
            backbone(original_ids, attention_mask)
        )

        modified = extract_hidden_states(
            backbone(modified_ids, attention_mask)
        )
    finally:
        backbone.train(was_training)

    earlier_difference = (
        original[0, earlier_position]
        - modified[0, earlier_position]
    ).abs().max().item()

    later_difference = (
        original[0, later_position]
        - modified[0, later_position]
    ).abs().max().item()

    return BidirectionalityResult(
        earlier_token_max_difference=earlier_difference,
        later_token_max_difference=later_difference,
        earlier_token_responds=earlier_difference > tolerance,
        later_token_responds=later_difference > tolerance,
    )


@torch.inference_mode()
def check_right_padding_invariance(
    backbone: nn.Module,
    *,
    input_ids: Tensor,
    attention_mask: Tensor,
    pad_token_id: int,
    extra_padding: int = 4,
    tolerance: float = 1e-4,
) -> PaddingResult:
    """Compare valid-token states before and after right padding."""
    if extra_padding < 1:
        raise ValueError("extra_padding must be positive")

    if input_ids.ndim != 2:
        raise ValueError("Input IDs must be [B, T]")

    if input_ids.shape != attention_mask.shape:
        raise ValueError("Input and mask shapes must match")

    if not torch.all(attention_mask == 1):
        raise ValueError(
            "Supply an unpadded reference sequence"
        )

    device = next(backbone.parameters()).device

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    padded_ids = torch.cat(
        [
            input_ids,
            torch.full(
                (input_ids.shape[0], extra_padding),
                pad_token_id,
                dtype=input_ids.dtype,
                device=device,
            ),
        ],
        dim=1,
    )

    padded_mask = torch.cat(
        [
            attention_mask,
            torch.zeros(
                (input_ids.shape[0], extra_padding),
                dtype=attention_mask.dtype,
                device=device,
            ),
        ],
        dim=1,
    )

    was_training = backbone.training
    backbone.eval()

    try:
        reference = extract_hidden_states(
            backbone(input_ids, attention_mask)
        )

        padded = extract_hidden_states(
            backbone(padded_ids, padded_mask)
        )
    finally:
        backbone.train(was_training)

    valid_length = input_ids.shape[1]

    difference = (
        reference
        - padded[:, :valid_length, :]
    ).abs().max().item()

    return PaddingResult(
        max_valid_token_difference=difference,
        invariant=difference <= tolerance,
    )


@torch.inference_mode()
def profile_backbone_memory(
    backbone: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
) -> MemoryProfile:
    """Measure one forward pass, including CUDA peak allocations.

    CUDA peaks are relative to the allocator state after reset.
    They do not include all process, driver or external GPU memory.
    """
    if input_ids.shape != attention_mask.shape:
        raise ValueError("Input and mask shapes must match")

    device = next(backbone.parameters()).device

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    was_training = backbone.training
    backbone.eval()

    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            gc.collect()
            torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()

        hidden = extract_hidden_states(
            backbone(input_ids, attention_mask)
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - start

        allocated = None
        reserved = None

        if device.type == "cuda":
            allocated = (
                torch.cuda.max_memory_allocated(device)
                / (1024 ** 2)
            )

            reserved = (
                torch.cuda.max_memory_reserved(device)
                / (1024 ** 2)
            )

        return MemoryProfile(
            device=str(device),
            elapsed_seconds=elapsed,
            peak_allocated_mb=allocated,
            peak_reserved_mb=reserved,
            output_shape=tuple(hidden.shape),
        )
    finally:
        backbone.train(was_training)
