"""
weight_optimizer_v5.py

Optimizes feature weights for the H-Score using scipy SLSQP.
Uses features_multiwindow.parquet with a strict curated feature list —
no flags, no risk scores, no dead columns, no anomaly detectors.

Changes from v4:
    FEATURES : hard-coded 32-feature curated list (vs auto-detected 97)
    EXCLUDED : stddev_position_size (15d dead/zero), win_rate_z_score,
               sybil_risk_flag/score, timing signals, risk flags,
               curve_smoothness, days_active, sparse ratios (calmar,
               sortino, gain_to_pain, annualized_return), drawdown cols,
               category_diversity_score
    INVERT   : trimmed to the 11 features that semantically make sense

Objective  : maximize Spearman correlation between weighted score and label
Constraints: all weights >= 0, weights sum to 100
Search     : 50 random Dirichlet initializations, best result kept

Usage:
    python weight_optimizer_v5.py
    python weight_optimizer_v5.py --features features_multiwindow.parquet
    python weight_optimizer_v5.py --n-init 100
"""

import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr

# ── curated feature list ──────────────────────────────────────────────────────

FEATURES = [
    # ── regular: higher = better ─────────────────────────────────────────
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
    "stddev_position_size_7d",       # short-window only; 15d version is dead
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
    # ── inverted: higher raw value = worse outcome ────────────────────────
    "worst_trade",                   # more negative = worse
    "worst_trade_1d",
    "roi_1d",                        # extreme short-window roi = noise/sybil
    "roi_3d",
    "profit_factor",                 # extreme values on short windows = luck
    "profit_factor_7d",
    "win_rate",                      # near 1.0 is suspicious in eligible pool
    "win_rate_1d",
    "win_rate_3d",
    "market_concentration_ratio",    # higher = more concentrated = worse
    "pnl_cat_crypto",                # crypto markets: volatile/easier to exploit
]

N_FEATURES = len(FEATURES)

# Multiply by -1 before percentile ranking so high raw value → low score
INVERT = {
    "worst_trade", "worst_trade_1d",
    "roi_1d", "roi_3d",
    "profit_factor", "profit_factor_7d",
    "win_rate", "win_rate_1d", "win_rate_3d",
    "market_concentration_ratio",
    "pnl_cat_crypto",
}

# No sparse ratio columns in this feature set — median imputation not needed


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(features_path: str, labels_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load, join, impute, invert, and percentile-rank all features.
    Returns (X, y) as numpy arrays.
      X shape: (n_samples, N_FEATURES)
      y shape: (n_samples,)  — binary label
    """
    features_df = pd.read_parquet(features_path)
    labels_df   = pd.read_parquet(labels_path)[
        ["proxy_wallet", "snapshot_date", "label"]
    ]

    features_df["snapshot_date"] = features_df["snapshot_date"].astype(str)
    labels_df["snapshot_date"]   = labels_df["snapshot_date"].astype(str)

    df = features_df.merge(labels_df, on=["proxy_wallet", "snapshot_date"], how="inner")
    print(f"Joined rows : {len(df):,}")
    print(f"Label=1 rate: {df['label'].mean()*100:.1f}%")

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = np.empty((len(df), N_FEATURES), dtype=float)
    for i, feat in enumerate(FEATURES):
        col = df[feat].fillna(0).values.astype(float)
        if feat in INVERT:
            col = -col
        X[:, i] = rankdata(col, method="average") / len(df)

    y = df["label"].values.astype(float)
    return X, y


# ── objective ─────────────────────────────────────────────────────────────────

def make_objective(X: np.ndarray, y_rank: np.ndarray):
    """
    Fast Pearson-of-ranks proxy for Spearman.
    X columns are already percentile ranks; weighted sum is a monotone
    proxy for the true score rank — avoids O(n log n) re-ranking per call.
    """
    y_c   = y_rank - y_rank.mean()
    y_std = y_rank.std()

    def objective(w: np.ndarray) -> float:
        score = X @ w
        s_c   = score - score.mean()
        denom = s_c.std() * y_std
        if denom == 0:
            return 0.0
        return -(s_c * y_c).mean() / denom  # minimise → maximise

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
    parser.add_argument("--out",      default="optimal_weights_v5.json")
    args = parser.parse_args()

    print(f"\nFeature count : {N_FEATURES}")
    print(f"Inverted      : {len(INVERT)}")
    print(f"Loading data...")
    X, y = load_data(args.features, args.labels)

    print(f"\nRunning SLSQP optimization ({args.n_init} random inits)...")
    best    = optimize(X, y, n_init=args.n_init, seed=args.seed)
    weights = best["weights"]
    rho_opt, p_opt = validate_spearman(X, y, weights)

    table = build_weight_table(weights)
    print(f"\n{'─'*60}")
    print(f"  Optimized weights  (Spearman rho = {rho_opt:.4f}, p = {p_opt:.2e})")
    print(f"{'─'*60}")
    print(table.to_string(index=True))

    w_equal      = np.full(N_FEATURES, 100.0 / N_FEATURES)
    rho_eq, _    = validate_spearman(X, y, w_equal)
    n_gt1pct     = (table["weight"] > 1.0).sum()

    print(f"\nBaseline (equal weights)  Spearman rho = {rho_eq:.4f}")
    print(f"Optimized                 Spearman rho = {rho_opt:.4f}  ({rho_opt - rho_eq:+.4f})")
    print(f"Features with weight > 1%: {n_gt1pct} / {N_FEATURES}")

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
