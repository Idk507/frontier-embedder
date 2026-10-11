"""GPU and CPU memory profiling for Frontier Embedder backbones.

Profiles forward inference and forward/backward activation costs. Does not
create an optimizer; optimizer states and training-step peak are excluded.
"""
from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn


Mode = Literal["inference", "backward"]


@dataclass(frozen=True)
class ProfileResult:
    sequence_length: int
    batch_size: int
    mode: str
    device: str
    status: str
    elapsed_seconds: float | None
    peak_allocated_mb: float | None
    peak_reserved_mb: float | None
    baseline_allocated_mb: float | None
    incremental_peak_mb: float | None
    error: str | None = None


def _device_of(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("Model must contain at least one parameter") from exc


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clear_cuda_after_failure(device: torch.device) -> None:
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()


def _hidden(output: object) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError("Model must return a tensor or hidden-state output")


def profile_sequence(
    model: nn.Module,
    *,
    sequence_length: int,
    batch_size: int = 1,
    mode: Mode = "inference",
    vocab_size: int = 128,
    warmup: bool = False,
) -> ProfileResult:
    """Profile one shape. Model and inputs must fit on the model's device.

    In backward mode, all parameters with requires_grad=True receive gradients.
    No optimizer step is performed; gradients are cleared after measurement.
    """
    if sequence_length <= 0 or batch_size <= 0:
        raise ValueError("sequence_length and batch_size must be positive")
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    if mode not in ("inference", "backward"):
        raise ValueError("mode must be inference or backward")
    if warmup and mode == "backward":
        raise ValueError("Warmup is supported for inference only")

    device = _device_of(model)
    was_training = model.training
    model.eval() if mode == "inference" else model.train()
    input_ids = None
    attention_mask = None
    output = None
    loss = None
    baseline = None
    elapsed = None
    peak_allocated = None
    peak_reserved = None
    status = "ok"
    error = None

    try:
        input_ids = torch.randint(
            1, vocab_size, (batch_size, sequence_length),
            dtype=torch.long, device=device,
        )
        attention_mask = torch.ones_like(input_ids)
        model.zero_grad(set_to_none=True)

        if warmup:
            with torch.inference_mode():
                _hidden(model(input_ids, attention_mask))

        _synchronize(device)
        if device.type == "cuda":
            baseline = torch.cuda.memory_allocated(device) / 2**20
            torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()
        if mode == "inference":
            with torch.inference_mode():
                output = _hidden(model(input_ids, attention_mask))
        else:
            if not any(p.requires_grad for p in model.parameters()):
                raise ValueError("Backward mode needs trainable parameters")
            output = _hidden(model(input_ids, attention_mask))
            loss = output.float().square().mean()
            loss.backward()

        _synchronize(device)
        elapsed = time.perf_counter() - start
        if device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(device) / 2**20
            peak_reserved = torch.cuda.max_memory_reserved(device) / 2**20

    except (torch.cuda.OutOfMemoryError, MemoryError) as exc:
        status = "oom"
        error = str(exc)
    finally:
        model.zero_grad(set_to_none=True)
        del loss, output, input_ids, attention_mask
        model.train(was_training)
        if status == "oom":
            _clear_cuda_after_failure(device)

    return ProfileResult(
        sequence_length=sequence_length,
        batch_size=batch_size,
        mode=mode,
        device=str(device),
        status=status,
        elapsed_seconds=elapsed,
        peak_allocated_mb=peak_allocated,
        peak_reserved_mb=peak_reserved,
        baseline_allocated_mb=baseline,
        incremental_peak_mb=(
            None if peak_allocated is None or baseline is None
            else max(0.0, peak_allocated - baseline)
        ),
        error=error,
    )


def profile_sweep(
    model: nn.Module,
    *,
    sequence_lengths: tuple[int, ...] = (32, 64, 128, 256),
    batch_size: int = 1,
    mode: Mode = "inference",
    vocab_size: int = 128,
) -> list[ProfileResult]:
    """Run sequence lengths in order; stop after the first OOM."""
    if not sequence_lengths:
        raise ValueError("Provide at least one sequence length")
    results = []
    for length in sequence_lengths:
        result = profile_sequence(
            model, sequence_length=length, batch_size=batch_size,
            mode=mode, vocab_size=vocab_size,
        )
        results.append(result)
        if result.status == "oom":
            break
    return results


def results_as_dicts(results: list[ProfileResult]) -> list[dict]:
    return [asdict(result) for result in results]
