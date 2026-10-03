"""Small, dependency-light reference implementation of churn identities.

The production parsers compute the same quantities while handling lm-eval
records.  These array functions support documentation and synthetic tests.
"""

from __future__ import annotations

import numpy as np


def _checked_labels(dense_pred, compressed_pred, gold):
    dense = np.asarray(dense_pred)
    compressed = np.asarray(compressed_pred)
    target = np.asarray(gold)
    if dense.shape != compressed.shape or dense.shape != target.shape:
        raise ValueError("dense_pred, compressed_pred, and gold must have identical shapes")
    if dense.ndim != 1:
        raise ValueError("label arrays must be one-dimensional")
    return dense, compressed, target


def decision_churn_rates(dense_pred, compressed_pred, gold):
    """Return per-example rates for decision changes and correctness flows."""
    dense, compressed, target = _checked_labels(dense_pred, compressed_pred, gold)
    if dense.size == 0:
        raise ValueError("at least one example is required")

    dense_correct = dense == target
    compressed_correct = compressed == target
    flipped = dense != compressed
    damage = dense_correct & ~compressed_correct
    repair = ~dense_correct & compressed_correct
    wrong_to_wrong = ~dense_correct & ~compressed_correct & flipped
    correct_to_correct_flip = dense_correct & compressed_correct & flipped

    mean = lambda mask: float(np.mean(mask))
    dense_accuracy = mean(dense_correct)
    compressed_accuracy = mean(compressed_correct)
    pred_flip = mean(flipped)
    accuracy_change = compressed_accuracy - dense_accuracy

    return {
        "dense_accuracy": dense_accuracy,
        "compressed_accuracy": compressed_accuracy,
        "accuracy_change": accuracy_change,
        "pred_flip": pred_flip,
        "damage": mean(damage),
        "repair": mean(repair),
        "wrong_to_wrong": mean(wrong_to_wrong),
        "correct_to_correct_flip": mean(correct_to_correct_flip),
        "hidden_churn": pred_flip - abs(accuracy_change),
    }


def jensen_shannon(p, q, *, axis=-1, epsilon=1e-12):
    """Compute Jensen-Shannon divergence with natural logarithms."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("p and q must have identical shapes")
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    p = p / p.sum(axis=axis, keepdims=True)
    q = q / q.sum(axis=axis, keepdims=True)
    midpoint = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / midpoint), axis=axis)
    kl_qm = np.sum(q * np.log(q / midpoint), axis=axis)
    return 0.5 * (kl_pm + kl_qm)
