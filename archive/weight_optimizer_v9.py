"""
weight_optimizer_v9.py

Optimizes feature weights for the H-Score using scipy SLSQP.
Identical to v8 in every way EXCEPT 4 additional features.

v8: 37 features, P@25 objective
v9: 41 features, P@25 objective

New features (all regular, higher = better):
    pnl_per_trade    — total_pnl / total_trades (15d)
    pnl_per_trade_1d — short-window quality: avg PnL per trade over 1d
    pnl_per_trade_3d — avg PnL per trade over 3d
    pnl_per_trade_7d — avg PnL per trade over 7d

Hypothesis: market makers execute many low-value trades; directional
traders take fewer, higher-conviction positions. pnl_per_trade should
discriminate between the two styles and penalize high-frequency noise.

    For each snapshot_date:
        1. Score all wallets: score_i = X_i @ w
        2. Identify top-25 wallets by score
        3. Precision@25 = fraction of top-25 with label=1
    Objective = mean(Precision@25) across all dates with >= 25 wallets
    Optimizer minimizes: -mean(Precision@25)

Why this matters:
    Spearman rho measures rank correlation across the full distribution.
    Precision@25 measures quality of the very top of the ranking —
    exactly the wallets we care about in production.  Optimizing the
    wrong metric produces high rho but poor top-end precision.

Implementation notes:
    P@25 involves a hard top-k cutoff (non-differentiable).  SLSQP uses
    numerical finite differences for gradients.  Key settings:
      eps=1.0   — step size for FD; small enough for accuracy, large enough
                  to detect score changes that flip wallets in/out of top-25
                  (weights are in [0, 100] scale)
      ftol=1e-5 — relaxed from 1e-9; non-smooth objective means tighter
                  tolerances just waste iterations chasing flat regions
      maxiter=500 — reduced from 1000; FD Jacobian is 37x more expensive

    Date index groups are precomputed once.  Each objective call does:
      O(n_total) for matrix multiply, O(n_d) argpartition per date — fast.

Total: 37 features (same as v7)
Save to: optimal_weights_v9.json

Usage:
    python weight_optimizer_v9.py
    python weight_optimizer_v9.py --features features_multiwindow.parquet
    python weight_optimizer_v9.py --n-init 100 --k 25
"""

import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr

# ── feature list (identical to v7) ───────────────────────────────────────────

FEATURES_V5 = [
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

FEATURES_NEW = [
    "sortino_ratio",
    "calmar_ratio",
    "gain_to_pain_ratio",
    "annualized_return",
    "total_trades",
    # v9 additions: pnl per trade — quality-adjusted return signal
    "pnl_per_trade",
    "pnl_per_trade_1d",
    "pnl_per_trade_3d",
    "pnl_per_trade_7d",
]

FEATURES   = FEATURES_V5 + FEATURES_NEW
N_FEATURES = len(FEATURES)

INVERT = {
    "worst_trade", "worst_trade_1d",
    "roi_1d", "roi_3d",
    "profit_factor", "profit_factor_7d",
    "win_rate", "win_rate_1d", "win_rate_3d",
    "market_concentration_ratio",
    "pnl_cat_crypto",
}

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
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    Load, join, impute, invert, and percentile-rank all features.

    Returns:
        X        : (n_samples, N_FEATURES) float array of percentile ranks
        y        : (n_samples,) binary label array
        medians  : dict of training-data medians for sparse columns
        dates    : (n_samples,) string array of snapshot_date per row
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

    # Compute training-data medians for sparse columns
    medians: dict = {}
    for feat in FILLNA_MEDIAN_FEATS:
        med = float(df[feat].median())
        medians[feat] = med
        null_pct = df[feat].isna().mean() * 100
        print(f"  {feat:<25}  median = {med:.4f}  ({null_pct:.1f}% null → imputed)")

    # Impute, invert, percentile-rank
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

    y     = df["label"].values.astype(float)
    dates = df["snapshot_date"].values
    return X, y, medians, dates


# ── P@K objective ─────────────────────────────────────────────────────────────

def build_date_groups(dates: np.ndarray, k: int) -> list[np.ndarray]:
    """
    Precompute row-index arrays per snapshot_date.
    Only includes dates with >= k wallets (others can't have meaningful P@K).
    """
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
    """
    Returns an objective closure: minimise -mean(Precision@K across dates).

    Each call:
      1. One O(n_total) matrix multiply to get all scores.
      2. One O(n_d) argpartition per date to find top-k indices.
      3. Average label rate in top-k across all eligible dates.

    The function is non-differentiable (step function at rank-k boundary).
    SLSQP uses numerical finite differences; eps=1.0 (set in optimizer)
    ensures perturbations are large enough to move wallets across the cutoff.
    """
    n_dates = len(date_groups)

    def objective(w: np.ndarray) -> float:
        scores = X @ w
        total  = 0.0
        for idx in date_groups:
            s_d   = scores[idx]
            y_d   = y[idx]
            # argpartition: O(n_d), gives indices of top-k (unordered)
            top_k = np.argpartition(s_d, -k)[-k:]
            total += y_d[top_k].mean()
        return -(total / n_dates)   # minimise → maximise P@K

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


# ── optimizer ─────────────────────────────────────────────────────────────────

def optimize(
    X: np.ndarray,
    y: np.ndarray,
    date_groups: list[np.ndarray],
    n_init: int,
    seed: int,
    k: int = 25,
) -> dict:
    rng  = np.random.default_rng(seed)
    obj  = make_p25_objective(X, y, date_groups, k)

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 100.0}
    bounds      = [(0.0, None)] * N_FEATURES

    best_p25    = -np.inf
    best_result = None

    for i in range(n_init):
        w0 = rng.dirichlet(np.ones(N_FEATURES)) * 100.0
        res = minimize(
            obj,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "ftol":    1e-5,   # relaxed: non-smooth objective, no point chasing flat regions
                "eps":     1.0,    # FD step size; large enough to flip wallets across top-25 boundary
                "maxiter": 500,    # reduced: FD Jacobian costs 37 extra evals per step
            },
        )

        p25 = -res.fun
        if p25 > best_p25:
            best_p25    = p25
            best_result = res

        if (i + 1) % 10 == 0:
            print(f"  init {i+1:>3}/{n_init}  best P@{k} = {best_p25:.4f}")

    return {"weights": best_result.x, "p25": best_p25, "result": best_result}


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
    parser.add_argument("--k",        type=int, default=25,
                        help="Top-k cutoff for precision objective (default: 25)")
    parser.add_argument("--out",      default="optimal_weights_v9.json")
    args = parser.parse_args()

    print(f"\nFeature count : {N_FEATURES}")
    print(f"Inverted      : {len(INVERT)}")
    print(f"Median-imputed: {len(FILLNA_MEDIAN_FEATS)}")
    print(f"Objective     : Precision@{args.k} (mean across snapshot dates)")
    print(f"\nLoading data + computing training medians...")
    X, y, medians, dates = load_data(args.features, args.labels)

    print(f"\nBuilding date groups (k={args.k})...")
    date_groups = build_date_groups(dates, k=args.k)

    # baseline P@k with equal weights
    w_equal   = np.full(N_FEATURES, 100.0 / N_FEATURES)
    p25_equal = compute_p25(X, y, w_equal, date_groups, args.k)
    rho_equal, _ = validate_spearman(X, y, w_equal)
    print(f"  Equal-weights baseline  P@{args.k} = {p25_equal:.4f}  "
          f"(Spearman rho = {rho_equal:.4f})")

    print(f"\nRunning SLSQP optimization ({args.n_init} random inits)...")
    best    = optimize(X, y, date_groups, n_init=args.n_init, seed=args.seed, k=args.k)
    weights = best["weights"]

    p25_opt      = compute_p25(X, y, weights, date_groups, args.k)
    rho_opt, p_rho = validate_spearman(X, y, weights)

    table = build_weight_table(weights)

    print(f"\n{'─'*60}")
    print(f"  Objective: P@{args.k}  optimized = {p25_opt:.4f}  "
          f"({p25_opt - p25_equal:+.4f} vs equal weights)")
    print(f"  Spearman rho (informational) = {rho_opt:.4f}  "
          f"(p = {p_rho:.2e})")
    print(f"{'─'*60}")

    top15 = table.head(15)
    print(f"\n  Top 15 features by weight:")
    print(f"  {'Feature':<35}  {'Weight':>7}  {'%':>6}  {'Inv':>4}")
    print(f"  {'─'*35}  {'─'*7}  {'─'*6}  {'─'*4}")
    for _, row in top15.iterrows():
        inv = "yes" if row["inverted"] else ""
        print(f"  {row['feature']:<35}  {row['weight']:>7.3f}  {row['pct_total']:>5.1f}%  {inv:>4}")

    n_gt1pct = (table["weight"] > 1.0).sum()
    print(f"\n  Features with weight > 1%: {n_gt1pct} / {N_FEATURES}")
    print(f"  Baseline P@{args.k} (equal weights): {p25_equal:.4f}")
    print(f"  Optimized P@{args.k}               : {p25_opt:.4f}  "
          f"({p25_opt - p25_equal:+.4f})")

    payload = {
        "features":     FEATURES,
        "weights":      weights.tolist(),
        "spearman_rho": rho_opt,
        "p25_train":    p25_opt,
        "inverted":     list(INVERT),
        "medians":      medians,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
