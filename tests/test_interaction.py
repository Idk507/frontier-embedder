
"""Tests for ColBERT-style MaxSim late interaction."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier_embedder.interaction import MaxSimInteraction


@pytest.fixture
def scorer():
    return MaxSimInteraction(document_chunk_size=2)


def test_aligned_output_shape(scorer):
    q = F.normalize(torch.randn(3, 4, 128), dim=-1)
    d = F.normalize(torch.randn(3, 6, 128), dim=-1)

    qm = torch.ones(3, 4)
    dm = torch.ones(3, 6)

    scores = scorer(q, d, qm, dm)

    assert scores.shape == (3,)


def test_all_pairs_output_shape(scorer):
    q = F.normalize(torch.randn(3, 4, 128), dim=-1)
    d = F.normalize(torch.randn(5, 6, 128), dim=-1)

    qm = torch.ones(3, 4)
    dm = torch.ones(5, 6)

    scores = scorer(q, d, qm, dm, all_pairs=True)

    assert scores.shape == (3, 5)


def test_exact_maxsim_score(scorer):
    # Two query tokens; three document tokens.
    q = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]]
    ])
    d = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    ])

    qm = torch.tensor([[1, 1]])
    dm = torch.tensor([[1, 1, 1]])

    score = scorer(q, d, qm, dm)

    assert torch.allclose(
        score,
        torch.tensor([2.0]),
    )


def test_document_padding_excluded(scorer):
    q = torch.tensor([[[1.0, 0.0]]])

    # The second token has the highest similarity but is padding.
    d = torch.tensor([
        [[0.0, 1.0], [1.0, 0.0]]
    ])

    qm = torch.tensor([[1]])
    dm = torch.tensor([[1, 0]])

    score = scorer(q, d, qm, dm)

    assert torch.allclose(
        score,
        torch.tensor([0.0]),
    )


def test_query_padding_excluded(scorer):
    q = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]]
    ])
    d = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]]
    ])

    qm = torch.tensor([[1, 0]])
    dm = torch.tensor([[1, 1]])

    score = scorer(q, d, qm, dm)

    assert torch.allclose(
        score,
        torch.tensor([1.0]),
    )


def test_negative_similarity_preserved(scorer):
    q = torch.tensor([[[1.0, 0.0]]])
    d = torch.tensor([[[-1.0, 0.0]]])

    qm = torch.ones(1, 1)
    dm = torch.ones(1, 1)

    score = scorer(q, d, qm, dm)

    assert torch.allclose(
        score,
        torch.tensor([-1.0]),
    )


def test_all_pairs_diagonal_matches_aligned(scorer):
    q = F.normalize(torch.randn(3, 4, 8), dim=-1)
    d = F.normalize(torch.randn(3, 5, 8), dim=-1)

    qm = torch.ones(3, 4)
    dm = torch.ones(3, 5)

    aligned = scorer(q, d, qm, dm)
    matrix = scorer(q, d, qm, dm, all_pairs=True)

    assert torch.allclose(
        aligned,
        matrix.diagonal(),
        atol=1e-6,
    )


def test_chunk_size_consistency():
    q = F.normalize(torch.randn(2, 3, 8), dim=-1)
    d = F.normalize(torch.randn(5, 4, 8), dim=-1)

    qm = torch.ones(2, 3)
    dm = torch.ones(5, 4)

    small = MaxSimInteraction(document_chunk_size=1)
    large = MaxSimInteraction(document_chunk_size=5)

    a = small(q, d, qm, dm, all_pairs=True)
    b = large(q, d, qm, dm, all_pairs=True)

    assert torch.allclose(a, b, atol=1e-6)


def test_gradient_flow(scorer):
    q = torch.randn(2, 3, 8, requires_grad=True)
    d = torch.randn(2, 4, 8, requires_grad=True)

    qm = torch.ones(2, 3)
    dm = torch.ones(2, 4)

    scores = scorer(q, d, qm, dm)
    scores.sum().backward()

    assert q.grad is not None
    assert d.grad is not None
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(d.grad).all()


def test_all_pairs_gradient_flow(scorer):
    q = torch.randn(2, 3, 8, requires_grad=True)
    d = torch.randn(3, 4, 8, requires_grad=True)

    qm = torch.ones(2, 3)
    dm = torch.ones(3, 4)

    scores = scorer(q, d, qm, dm, all_pairs=True)
    scores.sum().backward()

    assert q.grad is not None
    assert d.grad is not None


def test_all_padding_query_rejected(scorer):
    q = torch.randn(1, 3, 8)
    d = torch.randn(1, 4, 8)

    qm = torch.zeros(1, 3)
    dm = torch.ones(1, 4)

    with pytest.raises(ValueError):
        scorer(q, d, qm, dm)


def test_all_padding_document_rejected(scorer):
    q = torch.randn(1, 3, 8)
    d = torch.randn(1, 4, 8)

    qm = torch.ones(1, 3)
    dm = torch.zeros(1, 4)

    with pytest.raises(ValueError):
        scorer(q, d, qm, dm)


def test_mismatched_dimensions_rejected(scorer):
    q = torch.randn(1, 3, 8)
    d = torch.randn(1, 4, 16)

    qm = torch.ones(1, 3)
    dm = torch.ones(1, 4)

    with pytest.raises(ValueError):
        scorer(q, d, qm, dm)


def test_mismatched_aligned_batches_rejected(scorer):
    q = torch.randn(2, 3, 8)
    d = torch.randn(3, 4, 8)

    qm = torch.ones(2, 3)
    dm = torch.ones(3, 4)

    with pytest.raises(ValueError):
        scorer(q, d, qm, dm)


def test_nonbinary_mask_rejected(scorer):
    q = torch.randn(1, 3, 8)
    d = torch.randn(1, 4, 8)

    qm = torch.tensor([[1.0, 0.5, 0.0]])
    dm = torch.ones(1, 4)

    with pytest.raises(ValueError):
        scorer(q, d, qm, dm)


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        MaxSimInteraction(document_chunk_size=0)
