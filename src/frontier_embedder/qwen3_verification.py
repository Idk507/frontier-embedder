
"""Real-model verification for Frontier Embedder.

Run:
    python -m frontier_embedder.qwen3_verification --mode causal

To test the project backbone, supply a factory:
    python -m frontier_embedder.qwen3_verification ^
        --mode custom ^
        --factory your_module:create_backbone

The factory must accept model_name and device keyword arguments,
and return an initialized torch.nn.Module.
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from typing import Callable

import torch
from torch import nn

from frontier_embedder.backbone_diagnostics import (
    check_bidirectionality,
    check_right_padding_invariance,
    profile_backbone_memory,
)


DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"


@dataclass(frozen=True)
class VerificationReport:
    """Report of the verification results."""
    model_name: str
    mode: str
    device: str
    bidirectional: bool
    earlier_token_difference: float
    later_token_difference: float
    padding_invariant: bool
    padding_difference: float
    elapsed_seconds: float
    peak_allocated_mb: float | None
    peak_reserved_mb: float | None
    output_shape: tuple[int, ...]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(requested)

    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA are supported")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        if device.index is not None:
            if device.index >= torch.cuda.device_count():
                raise ValueError("Invalid CUDA device index")

    return device


def load_factory(
    specification: str,
) -> Callable[..., nn.Module]:
    """Import module:function without evaluating arbitrary strings."""
    module_name, separator, function_name = (
        specification.partition(":")
    )

    if not separator or not module_name or not function_name:
        raise ValueError(
            "Factory must have format module:function"
        )

    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)

    if not callable(factory):
        raise TypeError("Factory is not callable")

    return factory


def load_backbone(
    model_name: str,
    device: torch.device,
    *,
    mode: str,
    factory_spec: str | None = None,
) -> nn.Module:
    if mode == "causal":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            model_name,
            dtype=(
                torch.float16
                if device.type == "cuda"
                else torch.float32
            ),
            attn_implementation="eager", # eager attention is required for bidirectional verification where eager means that the attention is computed in a single pass and not in chunks. This is important for verifying that the model can attend to future tokens.
            trust_remote_code=False, # trust_remote_code=False is used to prevent the execution of arbitrary code from the model's configuration file.
        )
        model.to(device)

    elif mode == "custom":
        if not factory_spec:
            raise ValueError(
                "Custom mode requires --factory"
            )

        factory = load_factory(factory_spec)

        model = factory(
            model_name=model_name,
            device=device,
        )

        if not isinstance(model, nn.Module):
            raise TypeError(
                "Factory must return a torch.nn.Module"
            )

        model.to(device)

    else:
        raise ValueError("Unknown verification mode")

    model.eval()
    return model


def build_probe(
    tokenizer,
    *,
    text: str = "Retrieval models learn useful semantic representations.",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct two equal-length sequences differing at the end."""
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )

    original = encoded["input_ids"].long()

    if original.ndim != 2 or original.shape[0] != 1:
        raise ValueError("Unexpected tokenizer output")

    if original.shape[1] < 3:
        raise ValueError(
            "Probe requires at least three tokens"
        )

    vocabulary_size = len(tokenizer)

    if vocabulary_size < 2:
        raise ValueError("Tokenizer vocabulary is too small")

    modified = original.clone()

    original_last = int(original[0, -1].item())

    # Choose another valid token ID without changing sequence
    # length, attention mask or earlier token positions.
    candidate = (original_last + 1) % vocabulary_size

    if candidate == original_last:
        raise RuntimeError(
            "Could not choose a different token"
        )

    modified[0, -1] = candidate

    mask = torch.ones_like(original)

    return original, modified, mask


def verify_backbone(
    backbone: nn.Module,
    tokenizer,
    *,
    model_name: str = DEFAULT_MODEL,
    mode: str = "custom",
    tolerance: float = 1e-5,
    padding_tolerance: float = 1e-4,
) -> VerificationReport:
    original, modified, mask = build_probe(tokenizer)

    later_position = original.shape[1] - 1

    bidirectional = check_bidirectionality(
        backbone,
        original_ids=original,
        modified_ids=modified,
        attention_mask=mask,
        earlier_position=0,
        later_position=later_position,
        tolerance=tolerance,
    )

    pad_id = tokenizer.pad_token_id

    if pad_id is None:
        raise ValueError(
            "Tokenizer must have a padding token"
        )

    padding = check_right_padding_invariance(
        backbone,
        input_ids=original,
        attention_mask=mask,
        pad_token_id=pad_id,
        extra_padding=4,
        tolerance=padding_tolerance,
    )

    memory = profile_backbone_memory(
        backbone,
        input_ids=original,
        attention_mask=mask,
    )

    return VerificationReport(
        model_name=model_name,
        mode=mode,
        device=memory.device,
        bidirectional=bidirectional.earlier_token_responds,
        earlier_token_difference=(
            bidirectional.earlier_token_max_difference
        ),
        later_token_difference=(
            bidirectional.later_token_max_difference
        ),
        padding_invariant=padding.invariant,
        padding_difference=(
            padding.max_valid_token_difference
        ),
        elapsed_seconds=memory.elapsed_seconds,
        peak_allocated_mb=memory.peak_allocated_mb,
        peak_reserved_mb=memory.peak_reserved_mb,
        output_shape=memory.output_shape,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify Qwen3 embedding backbone behavior"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--mode",
        choices=("causal", "custom"),
        default="causal",
    )

    parser.add_argument(
        "--factory",
        default=None,
        help="Custom backbone factory as module:function",
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--output",
        default="qwen3_verification.json",
    )

    args = parser.parse_args()

    device = resolve_device(args.device)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=False,
    )

    backbone = load_backbone(
        args.model,
        device,
        mode=args.mode,
        factory_spec=args.factory,
    )

    report = verify_backbone(
        backbone,
        tokenizer,
        model_name=args.model,
        mode=args.mode,
    )

    serialized = json.dumps(
        asdict(report),
        indent=2,
    )

    print(serialized)

    from pathlib import Path

    Path(args.output).write_text(
        serialized + "\n",
        encoding="utf-8",
    )

    if not report.padding_invariant:
        print(
            "FAIL: valid-token states changed under padding"
        )
        return 2

    if args.mode == "custom" and not report.bidirectional:
        print(
            "FAIL: custom backbone did not demonstrate "
            "future-token influence"
        )
        return 3

    if args.mode == "causal" and report.bidirectional:
        print(
            "WARNING: causal control demonstrated "
            "future-token influence"
        )
        return 4

    print("Behavioral verification completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
