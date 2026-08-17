"""Unit tests for the benchmark metric helpers."""

import torch

from src.engine.benchmark import auroc_from_scores, average_precision, eer_from_scores


def test_eer_perfect_separation():
    pos = [torch.tensor([0.9, 0.95])]
    neg = [torch.tensor([0.1, 0.05])]
    assert eer_from_scores(pos, neg) < 1e-9


def test_auroc_perfect():
    pos = [torch.tensor([0.9, 0.8])]
    neg = [torch.tensor([0.2, 0.1])]
    assert auroc_from_scores(pos, neg) == 1.0


def test_average_precision_monotonic():
    # good positives clearly above all negatives; bad positives interleaved
    pos_good = [torch.tensor([0.9, 0.85, 0.8, 0.75])]
    pos_bad = [torch.tensor([0.12, 0.06, 0.16, 0.09])]
    neg = [torch.tensor([0.1, 0.05, 0.15, 0.08])]
    assert average_precision(pos_good, neg) > average_precision(pos_bad, neg)
