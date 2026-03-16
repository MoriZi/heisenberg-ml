"""
Shared feature utilities: percentile ranking, normalization.

Used by model-specific pipelines to transform raw feature columns
into percentile-ranked arrays suitable for weighted scoring.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def percentile_rank(arr: np.ndarray) -> np.ndarray:
    """Rank values to [0, 1] using average tie-breaking."""
    return rankdata(arr, method="average") / len(arr)


def normalize_features(
    df: pd.DataFrame,
    features: list[str],
    invert: set[str],
    medians: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Percentile-rank each feature within df (0 = worst, 1 = best).

    Parameters
    ----------
    df : DataFrame containing all feature columns.
    features : ordered list of feature column names.
    invert : set of feature names where higher raw value → lower score.
    medians : optional {feature: value} for columns needing median imputation
              instead of 0. Values should come from the training split.

    Returns
    -------
    X : (n_rows, n_features) float array of percentile ranks.
    """
    _medians = medians or {}
    X = np.empty((len(df), len(features)), dtype=float)
    for i, feat in enumerate(features):
        fill = _medians.get(feat, 0)
        col = df[feat].fillna(fill).values.astype(float)
        if feat in invert:
            col = -col
        X[:, i] = percentile_rank(col)
    return X
