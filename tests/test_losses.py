
"""Unit tests for joint retrieval training losses."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier_embedder.losses import (
    DenseContrastiveLoss,
    JointRetrievalLoss,
    MatryoshkaContrastiveLoss,
    MultiVectorContrastiveLoss,
    contrastive_cross_entropy,
)


def test_contrastive_perfect_matches():
    scores = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    loss = contrastive_cross_entropy(
        scores, temperature=0.05
    )

    assert loss.item() < 1e-6


def test_contrastive_wrong_matches():
    correct = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    wrong = torch.tensor([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    a = contrastive_cross_entropy(correct)
    b = contrastive_cross_entropy(wrong)

    assert a < b


def test_invalid_temperature():
    with pytest.raises(ValueError):
        DenseContrastiveLoss(temperature=0)


def test_dense_loss_finite():
    module = DenseContrastiveLoss()

    q = torch.randn(3, 1024)
    d = torch.randn(3, 1024)

    loss = module(q, d)

    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_dense_gradient_flow():
    module = DenseContrastiveLoss()

    q = torch.randn(
        3, 32, requires_grad=True
    )
    d = torch.randn(
        3, 32, requires_grad=True
    )

    loss = module(q, d)
    loss.backward()

    assert q.grad is not None
    assert d.grad is not None
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(d.grad).all()


def test_matryoshka_loss_finite():
    module = MatryoshkaContrastiveLoss()

    q = torch.randn(3, 1024)
    d = torch.randn(3, 1024)

    loss = module(q, d)

    assert torch.isfinite(loss)


def test_matryoshka_single_dimension_matches_dense():
    q = torch.randn(3, 256)
    d = torch.randn(3, 256)

    mrl = MatryoshkaContrastiveLoss(
        dimensions=(128,)
    )
    dense = DenseContrastiveLoss()

    assert torch.allclose(
        mrl(q, d),
        dense(q[:, :128], d[:, :128]),
    )


def test_invalid_matryoshka_dimension():
    module = MatryoshkaContrastiveLoss(
        dimensions=(128, 512)
    )

    q = torch.randn(2, 256)
    d = torch.randn(2, 256)

    with pytest.raises(ValueError):
        module(q, d)


def test_multi_vector_loss_finite():
    module = MultiVectorContrastiveLoss()

    q = F.normalize(
        torch.randn(3, 4, 128), dim=-1
    )
    d = F.normalize(
        torch.randn(3, 6, 128), dim=-1
    )

    qm = torch.ones(3, 4)
    dm = torch.ones(3, 6)

    loss = module(q, d, qm, dm)

    assert torch.isfinite(loss)


def test_multi_vector_gradient_flow():
    module = MultiVectorContrastiveLoss()

    q = torch.randn(
        3, 4, 16, requires_grad=True
    )
    d = torch.randn(
        3, 5, 16, requires_grad=True
    )

    qm = torch.ones(3, 4)
    dm = torch.ones(3, 5)

    loss = module(q, d, qm, dm)
    loss.backward()

    assert q.grad is not None
    assert d.grad is not None


def test_joint_loss_components():
    module = JointRetrievalLoss()

    q_dense = torch.randn(3, 1024)
    d_dense = torch.randn(3, 1024)

    q_tokens = F.normalize(
        torch.randn(3, 4, 128), dim=-1
    )
    d_tokens = F.normalize(
        torch.randn(3, 6, 128), dim=-1
    )

    qm = torch.ones(3, 4)
    dm = torch.ones(3, 6)

    result = module(
        q_dense, d_dense,
        q_tokens, d_tokens,
        qm, dm,
    )

    assert torch.isfinite(result.total)
    assert torch.isfinite(result.dense)
    assert torch.isfinite(result.matryoshka)
    assert torch.isfinite(result.multi_vector)

    expected = (
        result.dense
        + result.matryoshka
        + result.multi_vector
    )

    assert torch.allclose(result.total, expected)


def test_joint_weighted_sum():
    module = JointRetrievalLoss(
        dense_weight=2.0,
        matryoshka_weight=0.5,
        multi_vector_weight=0.0,
    )

    q = torch.randn(2, 1024)
    d = torch.randn(2, 1024)

    # Multi-vector branch disabled.
    empty = torch.empty(0)

    result = module(
        q, d, empty, empty, empty, empty
    )

    expected = (
        2.0 * result.dense
        + 0.5 * result.matryoshka
    )

    assert torch.allclose(result.total, expected)


def test_invalid_loss_weights():
    with pytest.raises(ValueError):
        JointRetrievalLoss(
            dense_weight=0,
            matryoshka_weight=0,
            multi_vector_weight=0,
        )


def test_nonfinite_scores_rejected():
    scores = torch.tensor([
        [float("nan"), 0.0],
        [0.0, 1.0],
    ])

    with pytest.raises(ValueError):
        contrastive_cross_entropy(scores)


def test_single_document_rejected():
    scores = torch.tensor([[1.0]])

    with pytest.raises(ValueError):
        contrastive_cross_entropy(scores)
