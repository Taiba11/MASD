"""Visualization utilities for MASD: reliability diagrams, uncertainty plots."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_reliability_diagram(labels, probs, n_bins=10, save_path=None):
    """Plot calibration reliability diagram (Fig. 3a)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accs.append(labels[mask].mean())
            bin_confs.append(probs[mask].mean())
            bin_counts.append(mask.sum())

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(bin_confs, bin_accs, width=1.0/n_bins, alpha=0.6, edgecolor="black", label="Model")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Reliability Diagram")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_uncertainty_distribution(correct_unc, incorrect_unc, save_path=None):
    """Plot uncertainty distributions for correct vs incorrect (Fig. 3b)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(correct_unc, bins=30, alpha=0.6, label=f"Correct (mu={np.mean(correct_unc):.3f})")
    ax.hist(incorrect_unc, bins=30, alpha=0.6, label=f"Incorrect (mu={np.mean(incorrect_unc):.3f})")
    ax.set_xlabel("Uncertainty (1 - confidence)")
    ax.set_ylabel("Count")
    ax.set_title("Uncertainty Distribution")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_coverage_accuracy(uncertainties, correct, save_path=None):
    """Plot coverage vs accuracy trade-off (Fig. 3c)."""
    thresholds = np.linspace(0, 1, 50)
    coverages, accs = [], []

    for t in thresholds:
        mask = uncertainties <= t
        if mask.sum() > 0:
            coverages.append(mask.mean() * 100)
            accs.append(correct[mask].mean() * 100)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(coverages, accs, "b-o", markersize=3)
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Coverage vs Accuracy Trade-off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
