"""
weight_optimizer_v2.py

Optimizes feature weights for the H-Score using scipy SLSQP.
Updated feature set based on feature_audit.py findings (2026-02-26 snapshot):

Changes from v1:
    REMOVED  (constant zero — zero variance):
        trade_size_stdev, max_position_pct, position_size_consistency,
        trade_timing_correlation_max, edge_decay

    ADDED:
        stddev_position_size    — replaces trade_size_stdev; actually populated
        coefficient_of_variation — scale-normalised position size volatility
        dominant_market_pnl     — pnl from single highest-pnl market
        curve_smoothness        — equity curve regularity
        perfect_timing_score    — composite timing quality metric
        perfect_entry_count     — count of well-timed entries
        perfect_exit_count      — count of well-timed exits
        gain_to_pain_ratio      — risk-adjusted return; sparse → median imputed
        annualized_return       — sparse → median imputed

    JSONB categories expanded (6 → 7):
        pnl_cat_sports, pnl_cat_crypto, pnl_cat_world_events,
        pnl_cat_economics, pnl_cat_politics, pnl_cat_entertainment,
        pnl_cat_other

    IMPUTATION:
        sortino_ratio, calmar_ratio, gain_to_pain_ratio, annualized_return
        → fillna(median) instead of fillna(0) — less distorting when ~50% null

Objective : maximize Spearman correlation between weighted score and label
Constraints: all weights >= 0, weights sum to 100
Search     : 50 random Dirichlet initializations, best result kept

Usage:
    python weight_optimizer_v2.py
    python weight_optimizer_v2.py --features features.parquet --labels labels.parquet
    python weight_optimizer_v2.py --n-init 100
"""

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr

# ── feature config ────────────────────────────────────────────────────────────

FEATURES = [
    # core performance
    "total_invested",
    "total_pnl",
    "best_trade",
    "worst_trade",                  # inverted: more-negative = worse
    # position sizing
    "avg_position_size",
    "stddev_position_size",         # replaces trade_size_stdev
    "coefficient_of_variation",     # stddev / mean — scale-normalised volatility
    "dominant_market_pnl",          # pnl from single highest-pnl market
    # risk-adjusted ratios (sparse — fillna(median) in load_data)
    "calmar_ratio",
    "sortino_ratio",
    "gain_to_pain_ratio",
    "annualized_return",
    # activity
    "total_trades",
    "markets_traded",
    "days_active",
    # concentration
    "market_concentration_ratio",   # inverted: higher concentration = worse
    # quality signals
    "statistical_confidence",
    "curve_smoothness",
    "perfect_timing_score",
    "perfect_entry_count",
    "perfect_exit_count",
    # JSONB category pnl
    "pnl_cat_sports",
    "pnl_cat_crypto",
    "pnl_cat_world_events",
    "pnl_cat_economics",
    "pnl_cat_politics",
    "pnl_cat_entertainment",
    "pnl_cat_other",
]

# Multiply by -1 before percentile ranking so high raw value → low score
INVERT = {"worst_trade", "market_concentration_ratio"}

# Sparse ratios — fillna with per-column median (not 0)
FILLNA_MEDIAN_FEATS = {"sortino_ratio", "calmar_ratio", "gain_to_pain_ratio", "annualized_return"}

N_FEATURES = len(FEATURES)


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(features_path: str, labels_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load, join, impute, invert, and percentile-rank all features.
    Returns (X, y) as numpy arrays.
    X shape: (n_samples, n_features)
    y shape: (n_samples,)  — binary label
    """
    features_df = pd.read_parquet(features_path)
    labels_df   = pd.read_parquet(labels_path)[
        ["proxy_wallet", "snapshot_date", "label"]
    ]
    df = features_df.merge(labels_df, on=["proxy_wallet", "snapshot_date"], how="inner")
    print(f"Joined rows : {len(df):,}")
    print(f"Label=1 rate: {df['label'].mean()*100:.1f}%")

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # ── impute, invert, then percentile-rank each feature (0=worst, 1=best) ─
    X = np.empty((len(df), N_FEATURES), dtype=float)
    for i, feat in enumerate(FEATURES):
        col = df[feat].copy()
        # median imputation for sparse ratio columns; 0 for everything else
        if feat in FILLNA_MEDIAN_FEATS:
            col = col.fillna(col.median())
        else:
            col = col.fillna(0)
        col = col.values.astype(float)
        if feat in INVERT:
            col = -col
        # rankdata / n → percentile rank in [1/n, 1]; ties averaged
        X[:, i] = rankdata(col, method="average") / len(col)

    y = df["label"].values.astype(float)
    return X, y


# ── objective ─────────────────────────────────────────────────────────────────

def make_objective(X: np.ndarray, y_rank: np.ndarray):
    """
    Returns a closure that computes -Spearman(score, label).
    Uses pre-ranked y to speed up inner loop (Spearman = Pearson of ranks).
    """
    def objective(w: np.ndarray) -> float:
        score      = X @ w
        score_rank = rankdata(score, method="average")
        corr = np.corrcoef(score_rank, y_rank)[0, 1]
        return -corr  # minimise → maximise Spearman

    return objective


# ── optimizer ─────────────────────────────────────────────────────────────────

def optimize(X: np.ndarray, y: np.ndarray, n_init: int, seed: int) -> dict:
    """
    Run SLSQP from n_init random starting points.
    Returns the best result found.
    """
    rng      = np.random.default_rng(seed)
    y_rank   = rankdata(y, method="average")
    obj      = make_objective(X, y_rank)

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 100.0}
    bounds      = [(0.0, None)] * N_FEATURES

    best_rho    = -np.inf
    best_result = None

    for i in range(n_init):
        # Dirichlet sample naturally sums to 1 → scale to 100
        w0 = rng.dirichlet(np.ones(N_FEATURES)) * 100.0

        res = minimize(
            obj,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
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
    rows = []
    for feat, w in zip(FEATURES, weights):
        rows.append({
            "feature":   feat,
            "weight":    round(w, 4),
            "pct_total": round(100 * w / weights.sum(), 2),
            "inverted":  feat in INVERT,
        })
    return (
        pd.DataFrame(rows)
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )


def validate_spearman(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple:
    """Recompute Spearman rho from scratch for final verification."""
    score = X @ weights
    rho, p = spearmanr(score, y)
    return rho, p


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="features.parquet")
    parser.add_argument("--labels",   default="labels.parquet")
    parser.add_argument("--n-init",   type=int, default=50,
                        help="Number of random initializations (default: 50)")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--out",      default="optimal_weights_v2.json",
                        help="Output path for best weights (default: optimal_weights_v2.json)")
    args = parser.parse_args()

    print(f"\nFeature count : {N_FEATURES}")
    print(f"Loading data...")
    X, y = load_data(args.features, args.labels)

    print(f"\nRunning SLSQP optimization ({args.n_init} random inits)...")
    best = optimize(X, y, n_init=args.n_init, seed=args.seed)

    weights = best["weights"]
    rho_opt, p_opt = validate_spearman(X, y, weights)

    # ── weight table ──────────────────────────────────────────────────────
    table = build_weight_table(weights)
    print(f"\n{'─'*55}")
    print(f"  Optimized weights  (Spearman rho = {rho_opt:.4f}, p = {p_opt:.2e})")
    print(f"{'─'*55}")
    print(table.to_string(index=True))

    # ── baseline: equal weights ───────────────────────────────────────────
    w_equal        = np.full(N_FEATURES, 100.0 / N_FEATURES)
    rho_eq, p_eq   = validate_spearman(X, y, w_equal)
    print(f"\nBaseline (equal weights)  Spearman rho = {rho_eq:.4f}")
    print(f"Optimized                 Spearman rho = {rho_opt:.4f}  "
          f"({rho_opt - rho_eq:+.4f})")

    # ── save ──────────────────────────────────────────────────────────────
    import json
    payload = {
        "features":     FEATURES,
        "weights":      weights.tolist(),
        "spearman_rho": rho_opt,
        "inverted":     list(INVERT),
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
