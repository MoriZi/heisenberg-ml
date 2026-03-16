"""
weight_optimizer_v4.py

Optimizes feature weights for the H-Score using scipy SLSQP.
Uses features_multiwindow.parquet — the 15d base features plus
short-window (1d, 3d, 7d) variants of 16 key metrics.

Changes from v2:
    INPUT    : features_multiwindow.parquet (103 cols → 101 features)
               rather than features.parquet (55 cols → 53 features)
    FEATURES : auto-detected from parquet (all numeric/bool cols except
               proxy_wallet and snapshot_date); no hard-coded list
    INVERT   : extended with short-window variants of worst_trade, roi,
               profit_factor, market_concentration_ratio, win_rate,
               plus pnl_cat_crypto
    OUTPUT   : optimal_weights_v4.json

Objective  : maximize Spearman correlation between weighted score and label
Constraints: all weights >= 0, weights sum to 100
Search     : 50 random Dirichlet initializations, best result kept

Usage:
    python weight_optimizer_v4.py
    python weight_optimizer_v4.py --features features_multiwindow.parquet --labels labels.parquet
    python weight_optimizer_v4.py --n-init 100
"""

import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr

# ── inversion config ──────────────────────────────────────────────────────────
# Higher raw value = worse outcome for these features.
# Base 15d columns use their original names (no _15d suffix).
# Short-window columns use _1d / _3d / _7d suffixes.
# Only columns actually present in the feature set are applied;
# extras in this set are silently ignored.

INVERT = {
    # worst trade: more negative = worse (original v2)
    "worst_trade",
    "worst_trade_1d", "worst_trade_3d", "worst_trade_7d",
    # roi: very high short-window roi can be noise / sybil signal
    "roi",
    "roi_1d", "roi_3d", "roi_7d",
    # profit_factor: extreme values on short windows often mean luck
    "profit_factor",
    "profit_factor_1d", "profit_factor_3d", "profit_factor_7d",
    # market concentration: higher = more concentrated = worse (original v2)
    "market_concentration_ratio",
    "market_concentration_ratio_1d", "market_concentration_ratio_3d",
    "market_concentration_ratio_7d",
    # win_rate: near 1.0 on short windows is suspicious
    "win_rate",
    "win_rate_1d", "win_rate_3d", "win_rate_7d",
    # pnl_cat_crypto: crypto markets are highly volatile / easier to exploit
    "pnl_cat_crypto",
}

# Sparse ratio columns — fillna with per-column median (only base 15d versions)
FILLNA_MEDIAN_FEATS = {
    "sortino_ratio",
    "calmar_ratio",
    "gain_to_pain_ratio",
    "annualized_return",
}

# Columns to drop entirely (non-features)
DROP_COLS = {"proxy_wallet", "snapshot_date"}


# ── feature detection ─────────────────────────────────────────────────────────

def detect_features(df: pd.DataFrame) -> list[str]:
    """
    Return sorted list of feature columns: all numeric/bool columns
    except proxy_wallet and snapshot_date.
    """
    feats = [
        col for col in df.columns
        if col not in DROP_COLS
        and df[col].dtype.kind in ("f", "i", "u", "b")  # float, int, uint, bool
    ]
    return sorted(feats)


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(
    features_path: str,
    labels_path: str,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
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

    # Align types for join key
    features_df["snapshot_date"] = features_df["snapshot_date"].astype(str)
    labels_df["snapshot_date"]   = labels_df["snapshot_date"].astype(str)

    df = features_df.merge(labels_df, on=["proxy_wallet", "snapshot_date"], how="inner")

    n_joined = len(df)
    print(f"Joined rows : {n_joined:,}")
    print(f"Label=1 rate: {df['label'].mean()*100:.1f}%")

    assert n_joined > 0, "Inner join produced 0 rows — check key alignment"

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # ── impute, invert, percentile-rank each feature (0=worst, 1=best) ───
    n_feat = len(features)
    X = np.empty((n_joined, n_feat), dtype=float)

    for i, feat in enumerate(features):
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
        X[:, i] = rankdata(col, method="average") / n_joined

    y = df["label"].values.astype(float)
    return X, y


# ── objective ─────────────────────────────────────────────────────────────────

def make_objective(X: np.ndarray, y_rank: np.ndarray):
    """
    Returns a closure that computes -Spearman(score, label).

    Speed trick: X columns are already percentile ranks (from load_data).
    X @ w is a weighted sum of per-feature ranks.  For the purpose of
    *ranking* scores to compute Spearman, the weighted sum of ranks is
    a monotone proxy for the true score rank — so we skip re-ranking
    the score and compute Pearson(X @ w, y_rank) directly.  This turns
    the inner loop from O(n log n) to O(n * n_feat), giving ~10-20x
    speedup on large feature sets.

    Note: the final validation in main() still calls spearmanr() for
    the exact reported rho.
    """
    # Centre y_rank once (Pearson of centred vectors = Pearson of originals)
    y_c   = y_rank - y_rank.mean()
    y_std = y_rank.std()

    def objective(w: np.ndarray) -> float:
        score = X @ w
        s_c   = score - score.mean()
        denom = s_c.std() * y_std
        if denom == 0:
            return 0.0
        corr = (s_c * y_c).mean() / denom
        return -corr  # minimise → maximise

    return objective


# ── optimizer ─────────────────────────────────────────────────────────────────

def optimize(X: np.ndarray, y: np.ndarray, n_init: int, seed: int) -> dict:
    """
    Run SLSQP from n_init random starting points.
    Returns the best result found.
    """
    n_feat   = X.shape[1]
    rng      = np.random.default_rng(seed)
    y_rank   = rankdata(y, method="average")
    obj      = make_objective(X, y_rank)

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 100.0}
    bounds      = [(0.0, None)] * n_feat

    best_rho    = -np.inf
    best_result = None

    for i in range(n_init):
        w0 = rng.dirichlet(np.ones(n_feat)) * 100.0

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

def build_weight_table(features: list[str], weights: np.ndarray) -> pd.DataFrame:
    rows = []
    for feat, w in zip(features, weights):
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
    parser.add_argument("--features", default="features_multiwindow.parquet")
    parser.add_argument("--labels",   default="labels.parquet")
    parser.add_argument("--n-init",   type=int, default=50,
                        help="Number of random initializations (default: 50)")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--out",      default="optimal_weights_v4.json",
                        help="Output path for best weights (default: optimal_weights_v4.json)")
    args = parser.parse_args()

    # ── detect features from parquet ──────────────────────────────────────
    print(f"\nScanning features from: {args.features}")
    features_df = pd.read_parquet(args.features)
    features = detect_features(features_df)
    del features_df  # free memory before loading again in load_data

    n_feat = len(features)
    print(f"Feature count : {n_feat}")

    # Report which INVERT entries apply vs are missing
    active_invert = [f for f in features if f in INVERT]
    print(f"Inverted      : {len(active_invert)} features")
    missing_invert = [f for f in INVERT if f not in features]
    if missing_invert:
        print(f"  (INVERT entries not in feature set: {missing_invert})")

    # ── load + join ───────────────────────────────────────────────────────
    print(f"\nLoading data...")
    X, y = load_data(args.features, args.labels, features)

    # ── optimize ──────────────────────────────────────────────────────────
    print(f"\nRunning SLSQP optimization ({args.n_init} random inits, {n_feat} features)...")
    best = optimize(X, y, n_init=args.n_init, seed=args.seed)

    weights = best["weights"]
    rho_opt, p_opt = validate_spearman(X, y, weights)

    # ── weight table ──────────────────────────────────────────────────────
    table = build_weight_table(features, weights)

    print(f"\n{'─'*65}")
    print(f"  Optimized weights  (Spearman rho = {rho_opt:.4f}, p = {p_opt:.2e})")
    print(f"{'─'*65}")

    top20 = table.head(20)
    print(f"\n  Top 20 features by weight:")
    print(f"  {'Feature':<45}  {'Weight':>7}  {'%Total':>7}  {'Inv':>4}")
    print(f"  {'─'*45}  {'─'*7}  {'─'*7}  {'─'*4}")
    for _, row in top20.iterrows():
        inv_tag = "yes" if row["inverted"] else ""
        print(f"  {row['feature']:<45}  {row['weight']:>7.3f}  {row['pct_total']:>6.2f}%  {inv_tag:>4}")

    # features with weight > 1% of total (i.e. weight > 1.0)
    n_gt1pct = (table["weight"] > 1.0).sum()
    print(f"\n  Features with weight > 1%: {n_gt1pct} / {n_feat}")

    # ── baseline: equal weights ───────────────────────────────────────────
    w_equal      = np.full(n_feat, 100.0 / n_feat)
    rho_eq, p_eq = validate_spearman(X, y, w_equal)
    print(f"\n  Baseline (equal weights)  Spearman rho = {rho_eq:.4f}")
    print(f"  Optimized                 Spearman rho = {rho_opt:.4f}  "
          f"({rho_opt - rho_eq:+.4f})")

    # ── save ──────────────────────────────────────────────────────────────
    payload = {
        "features":     features,
        "weights":      weights.tolist(),
        "spearman_rho": rho_opt,
        "inverted":     list(INVERT & set(features)),
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
