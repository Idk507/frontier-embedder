"""Validated configuration for the frontier embedding model."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings for dense and multi-vector embeddings."""

    backbone_name: str = "Qwen/Qwen3-0.6B-Base"

    hidden_size: int = 1024
    dense_dim: int = 1024

    matryoshka_dims: tuple[int, ...] = (
        128,
        256,
        512,
        1024,
    )

    multi_vector_dim: int = 128

    # Small initial limits for a 4 GiB GPU.
    max_query_length: int = 32
    max_document_length: int = 128

    normalize_embeddings: bool = True

    def __post_init__(self) -> None:
        if not self.backbone_name.strip():
            raise ValueError("backbone_name cannot be empty")

        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        if self.dense_dim <= 0:
            raise ValueError("dense_dim must be positive")

        if self.dense_dim > self.hidden_size:
            raise ValueError(
                "dense_dim cannot exceed hidden_size"
            )

        if self.multi_vector_dim <= 0:
            raise ValueError(
                "multi_vector_dim must be positive"
            )

        if not self.matryoshka_dims:
            raise ValueError(
                "matryoshka_dims cannot be empty"
            )

        if any(dim <= 0 for dim in self.matryoshka_dims):
            raise ValueError(
                "Matryoshka dimensions must be positive"
            )

        if any(
            dim > self.dense_dim
            for dim in self.matryoshka_dims
        ):
            raise ValueError(
                "Matryoshka dimensions cannot exceed dense_dim"
            )

        if tuple(sorted(set(self.matryoshka_dims))) != (
            self.matryoshka_dims
        ):
            raise ValueError(
                "Matryoshka dimensions must be unique and ascending"
            )

        if self.max_query_length <= 0:
            raise ValueError(
                "max_query_length must be positive"
            )

        if self.max_document_length <= 0:
            raise ValueError(
                "max_document_length must be positive"
            )


@dataclass(frozen=True)
class TrainingConfig:
    """Memory-conscious settings for initial local experiments."""

    learning_rate: float = 2e-4
    weight_decay: float = 0.01

    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    num_epochs: int = 1
    gradient_checkpointing: bool = True

    # Start by freezing the backbone.
    freeze_backbone: bool = True

    dense_loss_weight: float = 1.0
    matryoshka_loss_weight: float = 1.0
    multi_vector_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative"
            )

        if self.per_device_batch_size <= 0:
            raise ValueError(
                "per_device_batch_size must be positive"
            )

        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be positive"
            )

        if self.num_epochs <= 0:
            raise ValueError(
                "num_epochs must be positive"
            )

        weights = (
            self.dense_loss_weight,
            self.matryoshka_loss_weight,
            self.multi_vector_loss_weight,
        )

        if any(weight < 0 for weight in weights):
            raise ValueError(
                "Loss weights cannot be negative"
            )

        if sum(weights) <= 0:
            raise ValueError(
                "At least one loss weight must be positive"
            )


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete configuration for a local experiment."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(
        default_factory=TrainingConfig
    )