"""Metrics for MASD: EER, ECE, accuracy, F1-score."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_eer(labels, scores):
    """
    Compute Equal Error Rate (EER).

    Args:
        labels: Binary ground truth (0=real, 1=fake).
        scores: Fake-class probability scores.
    Returns:
        EER as percentage.
    """
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    try:
        eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0)
    except ValueError:
        eer = 0.5
    return eer * 100


def compute_ece(labels, probs, n_bins=15):
    """
    Compute Expected Calibration Error (ECE).

    Args:
        labels: Binary ground truth.
        probs: Predicted probabilities for positive class.
        n_bins: Number of calibration bins.
    Returns:
        ECE as percentage.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(labels)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += mask.sum() / total * abs(bin_acc - bin_conf)

    return ece * 100


def compute_metrics(labels, preds, scores=None):
    """Compute full evaluation metrics."""
    labels, preds = np.array(labels), np.array(preds)
    metrics = {
        "accuracy": accuracy_score(labels, preds) * 100,
        "precision": precision_score(labels, preds, zero_division=0) * 100,
        "recall": recall_score(labels, preds, zero_division=0) * 100,
        "f1": f1_score(labels, preds, zero_division=0) * 100,
    }
    if scores is not None:
        scores = np.array(scores)
        metrics["eer"] = compute_eer(labels, scores)
        metrics["ece"] = compute_ece(labels, scores)
    return metrics
