"""
weight_optimizer_v7.py

Optimizes feature weights for the H-Score using scipy SLSQP.
Extends the v5 curated feature list with 5 additional 15d features:

    sortino_ratio       — sparse (~55% null) → median imputed
    calmar_ratio        — sparse (~55% null) → median imputed
    gain_to_pain_ratio  — sparse (~52% null) → median imputed
    annualized_return   — sparse (~52% null) → median imputed
    total_trades        — fully populated    → fillna(0)

All 5 are regular (higher = better), not inverted.

Imputation rule (no data leakage):
    Medians are computed from the training dataset only (the full joined
    features × labels dataset passed to this script).  They are saved
    into optimal_weights_v7.json so that evaluate.py can apply the same
    values to test folds without computing medians from test data.

Total: 32 (v5) + 5 = 37 features

Objective  : maximize Spearman correlation between weighted score and label
Constraints: all weights >= 0, weights sum to 100
Search     : 50 random Dirichlet initializations, best result kept

Usage:
    python weight_optimizer_v7.py
    python weight_optimizer_v7.py --features features_multiwindow.parquet
    python weight_optimizer_v7.py --n-init 100
"""

import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr

# ── feature list ──────────────────────────────────────────────────────────────
# v5 curated 32 features, unchanged:

FEATURES_V5 = [
    # regular: higher = better
    "total_pnl",
    "total_pnl_1d",
    "total_pnl_3d",
    "total_pnl_7d",
    "total_invested",
    "total_invested_3d",
    "total_invested_7d",
    "best_trade",
    "best_trade_7d",
    "avg_position_size",
    "stddev_position_size_7d",
    "dominant_market_pnl",
    "dominant_market_pnl_7d",
    "pnl_cat_sports",
    "pnl_cat_other",
    "pnl_cat_other_7d",
    "perfect_entry_count",
    "statistical_confidence",
    "statistical_confidence_1d",
    "markets_traded",
    "pnl_cat_economics",
    # inverted: higher raw value = worse
    "worst_trade",
    "worst_trade_1d",
    "roi_1d",
    "roi_3d",
    "profit_factor",
    "profit_factor_7d",
    "win_rate",
    "win_rate_1d",
    "win_rate_3d",
    "market_concentration_ratio",
    "pnl_cat_crypto",
]

# v7 additions (all regular, higher = better):
FEATURES_NEW = [
    "sortino_ratio",
    "calmar_ratio",
    "gain_to_pain_ratio",
    "annualized_return",
    "total_trades",
]

FEATURES   = FEATURES_V5 + FEATURES_NEW
N_FEATURES = len(FEATURES)

# Inverted features (identical to v5)
INVERT = {
    "worst_trade", "worst_trade_1d",
    "roi_1d", "roi_3d",
    "profit_factor", "profit_factor_7d",
    "win_rate", "win_rate_1d", "win_rate_3d",
    "market_concentration_ratio",
    "pnl_cat_crypto",
}

# Sparse ratio columns — imputed with training-data median (not 0).
# Median is computed once in load_data and returned for downstream use.
FILLNA_MEDIAN_FEATS = {
    "sortino_ratio",
    "calmar_ratio",
    "gain_to_pain_ratio",
    "annualized_return",
}


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(
    features_path: str,
    labels_path: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load, join, impute, invert, and percentile-rank all features.

    Returns:
        X        : (n_samples, N_FEATURES) float array of percentile ranks
        y        : (n_samples,) binary label array
        medians  : dict {feature_name: median_value} for sparse ratio columns
                   Computed from this (training) dataset only — save to JSON
                   so test scoring can apply the same imputation values.
    """
    features_df = pd.read_parquet(features_path)
    labels_df   = pd.read_parquet(labels_path)[
        ["proxy_wallet", "snapshot_date", "label"]
    ]

    features_df["snapshot_date"] = features_df["snapshot_date"].astype(str)
    labels_df["snapshot_date"]   = labels_df["snapshot_date"].astype(str)

    df = features_df.merge(labels_df, on=["proxy_wallet", "snapshot_date"], how="inner")
    n  = len(df)
    print(f"Joined rows : {n:,}")
    print(f"Label=1 rate: {df['label'].mean()*100:.1f}%")
    assert n > 0, "Inner join produced 0 rows — check key alignment"

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # ── compute medians from training data (this dataset) ─────────────────
    # Medians are computed before any imputation so they reflect only
    # the naturally non-null rows — accurate representation of the
    # underlying distribution.
    medians: dict = {}
    for feat in FILLNA_MEDIAN_FEATS:
        med = float(df[feat].median())   # pandas median() ignores NaN
        medians[feat] = med
        null_pct = df[feat].isna().mean() * 100
        print(f"  {feat:<25}  median = {med:.4f}  ({null_pct:.1f}% null → imputed)")

    # ── impute, invert, percentile-rank ───────────────────────────────────
    X = np.empty((n, N_FEATURES), dtype=float)
    for i, feat in enumerate(FEATURES):
        col = df[feat].copy()
        if feat in FILLNA_MEDIAN_FEATS:
            col = col.fillna(medians[feat])
        else:
            col = col.fillna(0)
        col = col.values.astype(float)
        if feat in INVERT:
            col = -col
        X[:, i] = rankdata(col, method="average") / n

    y = df["label"].values.astype(float)
    return X, y, medians


# ── objective ─────────────────────────────────────────────────────────────────

def make_objective(X: np.ndarray, y_rank: np.ndarray):
    """
    Fast Pearson-of-ranks proxy for Spearman.
    X columns are already percentile ranks; Pearson(X@w, y_rank) ≈ Spearman.
    Avoids O(n log n) re-ranking per optimizer call.
    """
    y_c   = y_rank - y_rank.mean()
    y_std = y_rank.std()

    def objective(w: np.ndarray) -> float:
        score = X @ w
        s_c   = score - score.mean()
        denom = s_c.std() * y_std
        if denom == 0:
            return 0.0
        return -(s_c * y_c).mean() / denom

    return objective


# ── optimizer ─────────────────────────────────────────────────────────────────

def optimize(X: np.ndarray, y: np.ndarray, n_init: int, seed: int) -> dict:
    rng      = np.random.default_rng(seed)
    y_rank   = rankdata(y, method="average")
    obj      = make_objective(X, y_rank)

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 100.0}
    bounds      = [(0.0, None)] * N_FEATURES

    best_rho    = -np.inf
    best_result = None

    for i in range(n_init):
        w0 = rng.dirichlet(np.ones(N_FEATURES)) * 100.0
        res = minimize(
            obj, w0, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        rho = -res.fun
        if rho > best_rho:
            best_rho    = rho
            best_result = res

        if (i + 1) % 10 == 0:
            print(f"  init {i+1:>3}/{n_init}  best Spearman rho = {best_rho:.4f}")

    return {"weights": best_result.x, "spearman_rho": best_rho, "result": best_result}


# ── reporting ─────────────────────────────────────────────────────────────────

def build_weight_table(weights: np.ndarray) -> pd.DataFrame:
    rows = [
        {
            "feature":   feat,
            "weight":    round(w, 4),
            "pct_total": round(100 * w / weights.sum(), 2),
            "inverted":  feat in INVERT,
        }
        for feat, w in zip(FEATURES, weights)
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )


def validate_spearman(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple:
    score = X @ weights
    rho, p = spearmanr(score, y)
    return rho, p


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="features_multiwindow.parquet")
    parser.add_argument("--labels",   default="labels.parquet")
    parser.add_argument("--n-init",   type=int, default=50)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--out",      default="optimal_weights_v7.json")
    args = parser.parse_args()

    print(f"\nFeature count : {N_FEATURES}  (v5: {len(FEATURES_V5)}, new: {len(FEATURES_NEW)})")
    print(f"Inverted      : {len(INVERT)}")
    print(f"Median-imputed: {len(FILLNA_MEDIAN_FEATS)}")
    print(f"\nLoading data + computing training medians...")
    X, y, medians = load_data(args.features, args.labels)

    print(f"\nRunning SLSQP optimization ({args.n_init} random inits)...")
    best    = optimize(X, y, n_init=args.n_init, seed=args.seed)
    weights = best["weights"]
    rho_opt, p_opt = validate_spearman(X, y, weights)

    table = build_weight_table(weights)

    print(f"\n{'─'*60}")
    print(f"  Optimized weights  (Spearman rho = {rho_opt:.4f}, p = {p_opt:.2e})")
    print(f"{'─'*60}")

    top15 = table.head(15)
    print(f"\n  Top 15 features by weight:")
    print(f"  {'Feature':<35}  {'Weight':>7}  {'%':>6}  {'Inv':>4}")
    print(f"  {'─'*35}  {'─'*7}  {'─'*6}  {'─'*4}")
    for _, row in top15.iterrows():
        inv = "yes" if row["inverted"] else ""
        print(f"  {row['feature']:<35}  {row['weight']:>7.3f}  {row['pct_total']:>5.1f}%  {inv:>4}")

    w_equal      = np.full(N_FEATURES, 100.0 / N_FEATURES)
    rho_eq, _    = validate_spearman(X, y, w_equal)
    n_gt1pct     = (table["weight"] > 1.0).sum()

    print(f"\n  Baseline (equal weights)  Spearman rho = {rho_eq:.4f}")
    print(f"  Optimized                 Spearman rho = {rho_opt:.4f}  ({rho_opt - rho_eq:+.4f})")
    print(f"  Features with weight > 1%: {n_gt1pct} / {N_FEATURES}")

    # ── save weights + training medians ───────────────────────────────────
    # medians are stored so evaluate.py can impute test data without leakage
    payload = {
        "features":     FEATURES,
        "weights":      weights.tolist(),
        "spearman_rho": rho_opt,
        "inverted":     list(INVERT),
        "medians":      medians,   # training-data medians for sparse columns
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {args.out}  (includes training medians for test imputation)")


if __name__ == "__main__":
    main()
