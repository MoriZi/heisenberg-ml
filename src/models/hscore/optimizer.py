"""
H-Score weight optimizer using scipy SLSQP.

Objective: maximize mean Precision@25 across snapshot dates.
Weights are constrained to be non-negative and sum to 100.

This is the v8 production optimizer logic, extracted into the
model-specific module.
"""

import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr

from src.models.hscore.config import HScoreConfig


def load_data(
    features_path: str,
    labels_path: str,
    config: HScoreConfig,
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    Load, join, impute, invert, and percentile-rank all features.

    Returns
    -------
    X : (n_samples, n_features) float array of percentile ranks
    y : (n_samples,) binary label array
    medians : dict of training-data medians for sparse columns
    dates : (n_samples,) string array of snapshot_date per row
    """
    features_df = pd.read_parquet(features_path)
    labels_df = pd.read_parquet(labels_path)[
        ["proxy_wallet", "snapshot_date", "label"]
    ]

    features_df["snapshot_date"] = features_df["snapshot_date"].astype(str)
    labels_df["snapshot_date"] = labels_df["snapshot_date"].astype(str)

    df = features_df.merge(labels_df, on=["proxy_wallet", "snapshot_date"], how="inner")
    n = len(df)
    print(f"Joined rows : {n:,}")
    print(f"Label=1 rate: {df['label'].mean() * 100:.1f}%")
    assert n > 0, "Inner join produced 0 rows — check key alignment"

    missing = [f for f in config.features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    medians: dict = {}
    for feat in config.fillna_median_feats:
        med = float(df[feat].median())
        medians[feat] = med
        null_pct = df[feat].isna().mean() * 100
        print(
            f"  {feat:<25}  median = {med:.4f}  ({null_pct:.1f}% null → imputed)"
        )

    X = np.empty((n, config.n_features), dtype=float)
    for i, feat in enumerate(config.features):
        col = df[feat].copy()
        if feat in config.fillna_median_feats:
            col = col.fillna(medians[feat])
        else:
            col = col.fillna(0)
        col = col.values.astype(float)
        if feat in config.invert:
            col = -col
        X[:, i] = rankdata(col, method="average") / n

    y = df["label"].values.astype(float)
    dates = df["snapshot_date"].values
    return X, y, medians, dates


def build_date_groups(dates: np.ndarray, k: int) -> list[np.ndarray]:
    """Precompute row-index arrays per snapshot_date (dates with >= k wallets)."""
    unique, inverse = np.unique(dates, return_inverse=True)
    groups = []
    for d_idx in range(len(unique)):
        idx = np.where(inverse == d_idx)[0]
        if len(idx) >= k:
            groups.append(idx)
    print(f"  Dates with >= {k} wallets: {len(groups)} / {len(unique)}")
    return groups


def make_p25_objective(
    X: np.ndarray,
    y: np.ndarray,
    date_groups: list[np.ndarray],
    k: int = 25,
):
    """Returns an objective closure: minimise -mean(Precision@K across dates)."""
    n_dates = len(date_groups)

    def objective(w: np.ndarray) -> float:
        scores = X @ w
        total = 0.0
        for idx in date_groups:
            s_d = scores[idx]
            y_d = y[idx]
            top_k = np.argpartition(s_d, -k)[-k:]
            total += y_d[top_k].mean()
        return -(total / n_dates)

    return objective


def compute_p25(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    date_groups: list[np.ndarray],
    k: int = 25,
) -> float:
    """Compute mean Precision@K for reporting."""
    return -make_p25_objective(X, y, date_groups, k)(weights)


def optimize(
    X: np.ndarray,
    y: np.ndarray,
    date_groups: list[np.ndarray],
    config: HScoreConfig,
) -> dict:
    """
    Run SLSQP optimization with multiple random Dirichlet initializations.

    Returns dict with keys: weights, p25, result.
    """
    rng = np.random.default_rng(config.seed)
    obj = make_p25_objective(X, y, date_groups, config.k)

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 100.0}
    bounds = [(0.0, None)] * config.n_features

    best_p25 = -np.inf
    best_result = None

    for i in range(config.n_init):
        w0 = rng.dirichlet(np.ones(config.n_features)) * 100.0
        res = minimize(
            obj,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "ftol": config.ftol,
                "eps": config.eps,
                "maxiter": config.maxiter,
            },
        )

        p25 = -res.fun
        if p25 > best_p25:
            best_p25 = p25
            best_result = res

        if (i + 1) % 10 == 0:
            print(f"  init {i + 1:>3}/{config.n_init}  best P@{config.k} = {best_p25:.4f}")

    return {"weights": best_result.x, "p25": best_p25, "result": best_result}


def validate_spearman(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    """Compute Spearman rho between weighted scores and labels."""
    score = X @ weights
    rho, p = spearmanr(score, y)
    return rho, p


def build_weight_table(
    weights: np.ndarray, config: HScoreConfig
) -> pd.DataFrame:
    """Build a sorted DataFrame of feature weights."""
    rows = [
        {
            "feature": feat,
            "weight": round(w, 4),
            "pct_total": round(100 * w / weights.sum(), 2),
            "inverted": feat in config.invert,
        }
        for feat, w in zip(config.features, weights)
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )


def save_weights(
    weights: np.ndarray,
    medians: dict,
    spearman_rho: float,
    p25_train: float,
    config: HScoreConfig,
    out_path: str,
) -> None:
    """Save optimized weights to JSON."""
    payload = {
        "features": config.features,
        "weights": weights.tolist(),
        "spearman_rho": spearman_rho,
        "p25_train": p25_train,
        "inverted": list(config.invert),
        "medians": medians,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {out_path}")
