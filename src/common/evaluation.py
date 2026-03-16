"""
Shared evaluation metrics for scoring models.

Provides Spearman correlation, Precision@K, and a combined evaluation
function used across all model types.
"""

import numpy as np
from scipy.stats import spearmanr


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Fraction of label=1 among the top-k rows ranked by score."""
    if len(scores) < k:
        return float("nan")
    idx = np.argsort(scores)[::-1][:k]
    return float(labels[idx].mean())


def evaluate_scores(scores: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute standard evaluation metrics for a score vector.

    Returns dict with: spearman_rho, spearman_p, precision@25, precision@100.
    """
    rho, p = spearmanr(scores, labels)
    return {
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(p), 4),
        "precision@25": round(precision_at_k(scores, labels, 25), 4),
        "precision@100": round(precision_at_k(scores, labels, 100), 4),
    }
