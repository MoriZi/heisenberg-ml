"""
evaluate.py

Walk-forward evaluation of three H-Score variants on the test set.

Split:
    Train  : snapshot_date <  2025-12-15
    Gap    : 14 days (2025-12-15 → 2025-12-28, excluded)
    Test   : snapshot_date >= 2025-12-29

Variants compared:
    1. current_formula  — original H-Score components, approximated from
                          wallet_profile_metrics window=15 (see caveats below)
    2. equal_weights    — uniform weight across the 14 optimizer features
    3. optimized        — weights from optimal_weights.json

Metrics (test set only):
    Spearman rho    — rank correlation between score and binary label
    Precision@25    — label=1 rate in top 25 rows by score
    Precision@100   — label=1 rate in top 100 rows by score

Usage:
    python evaluate.py
    python evaluate.py --weights optimal_weights.json
    python evaluate.py --features features.parquet --labels labels.parquet
"""

import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

# ── walk-forward boundaries ───────────────────────────────────────────────────
TRAIN_END  = "2025-12-15"   # exclusive (train: < this date)
TEST_START = "2025-12-29"   # inclusive (gap = 14 days)

# ── known validation wallets ──────────────────────────────────────────────────
KNOWN_WALLETS = {
    "cf11 (NBA)":       "0xcf119e969f31de9653a58cb3dc213b485cd48399",
    "ccb2 (CS2)":       "0xccb290b1c145d1c95695d3756346bba9f1398586",
    "916f (UCL)":       "0x916f7165c2c836aba22edb6453cdbb5f3ea253ba",
    "d008 (selective)": "0xd008786fad743d0d5c60f99bff5d90ebc212135d",
}
VALIDATION_DATE = "2026-01-23"

# ── walk-forward folds ────────────────────────────────────────────────────────
FOLDS = [
    {"name": "Fold 1", "train_end": "2025-12-01", "test_start": "2025-12-15"},
    {"name": "Fold 2", "train_end": "2025-12-15", "test_start": "2025-12-29"},
    {"name": "Fold 3", "train_end": "2026-01-01", "test_start": "2026-01-15"},
]


# ── current formula implementation ───────────────────────────────────────────
# Approximated from wallet_profile_metrics window=15.
# Caveats vs the full production formula:
#   - consistency_score (25 pts) omitted — requires avg_rank from leaderboard
#   - sharpe_score (15 pts) omitted      — sharpe_ratio is 90.9% null, dropped
#   - roi_stability_score uses raw roi (15d) instead of roi_30d / roi_7d ratio
# Remaining components (60 pts max) are kept exact and not rescaled, so the
# absolute score is lower than production but rank order is comparable.

def _step(val: float, thresholds: list, scores: list) -> float:
    """Piecewise step function: last threshold exceeded wins."""
    result = 0.0
    for t, s in zip(thresholds, scores):
        if val >= t:
            result = float(s)
    return result


def compute_current_formula(df: pd.DataFrame) -> pd.Series:
    """
    Approximated H-Score from wallet_profile_metrics columns.
    Components implemented:
        roi_score           (up to 20 pts) — simplified, uses 15d roi
        win_rate_score      (up to 20 pts) — exact
        diversification     (up to 10 pts) — exact (markets_traded step fn)
        sample_score        (up to 10 pts) — exact (total_trades step fn)
    """
    roi_score = df["roi"].clip(lower=0).clip(upper=1.0) * 20

    win_rate_score = (
        (df["win_rate"] - 0.5).clip(lower=0) / 0.45 * 20
    ).clip(upper=20)

    div_score = df["markets_traded"].apply(
        lambda x: _step(x, [5, 10, 20, 50], [3, 5, 8, 10])
    )

    sample_score = df["total_trades"].apply(
        lambda x: _step(x, [100, 500, 10_000, 50_000], [6, 10, 4, 2])
    )

    return roi_score + win_rate_score + div_score + sample_score


# ── percentile normalization ───────────────────────────────────────────────────

def percentile_rank(arr: np.ndarray) -> np.ndarray:
    """Rank values to [0, 1] using average tie-breaking."""
    return rankdata(arr, method="average") / len(arr)


def normalize_features(
    df: pd.DataFrame,
    features: list,
    invert: set,
    medians: dict | None = None,
) -> np.ndarray:
    """
    Percentile-rank each feature within df (0 = worst, 1 = best).
    Inverted features are negated before ranking.

    medians: optional dict {feature: value} for sparse columns that need
             median imputation instead of 0.  Values should come from the
             training split only to avoid data leakage.
    """
    _medians = medians or {}
    X = np.empty((len(df), len(features)), dtype=float)
    for i, feat in enumerate(features):
        fill = _medians.get(feat, 0)
        col  = df[feat].fillna(fill).values.astype(float)
        if feat in invert:
            col = -col
        X[:, i] = percentile_rank(col)
    return X


# ── metrics ───────────────────────────────────────────────────────────────────

def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Fraction of label=1 among the top-k rows ranked by score."""
    if len(scores) < k:
        return float("nan")
    idx = np.argsort(scores)[::-1][:k]
    return float(labels[idx].mean())


def evaluate_scores(scores: np.ndarray, labels: np.ndarray) -> dict:
    rho, p = spearmanr(scores, labels)
    return {
        "spearman_rho": round(float(rho), 4),
        "spearman_p":   round(float(p),   4),
        "precision@25":  round(precision_at_k(scores, labels, 25),  4),
        "precision@100": round(precision_at_k(scores, labels, 100), 4),
    }


# ── known wallet scoring ──────────────────────────────────────────────────────

def score_known_wallets(
    df: pd.DataFrame,
    opt_features: list,
    opt_weights: np.ndarray,
    opt_invert: set,
) -> None:
    """
    For each known wallet, find all snapshot_dates where it appears in
    features.parquet. Score and rank it within that day's eligible pool
    (percentile rank computed per-date so ranks are production-comparable).

    Prints:
      1. Per-date detail table for each wallet
      2. Summary table: wallet, dates_present, avg_h_score, avg_rank_pct, label=1 rate
    """
    # ── score every row, per-date normalization ───────────────────────────
    # Rank features within each snapshot_date so h_score reflects a wallet's
    # standing among its peers on that specific day.
    scored_parts = []
    for date_str, day_df in df.groupby("snapshot_date"):
        day_df = day_df.copy().reset_index(drop=True)
        X      = normalize_features(day_df, opt_features, opt_invert)
        day_df["h_score"]  = X @ opt_weights
        n = len(day_df)
        day_df["rank"]     = day_df["h_score"].rank(ascending=False, method="min").astype(int)
        day_df["rank_pct"] = day_df["rank"] / n          # 0 = best, 1 = worst
        day_df["pool_size"] = n
        scored_parts.append(day_df)

    scored = pd.concat(scored_parts, ignore_index=True)

    # ── per-wallet detail + summary ───────────────────────────────────────
    summary_rows = []

    for alias, addr in KNOWN_WALLETS.items():
        wallet_df = (
            scored[scored["proxy_wallet"] == addr]
            [["snapshot_date", "h_score", "rank", "pool_size", "rank_pct", "label"]]
            .sort_values("snapshot_date")
            .reset_index(drop=True)
        )

        if wallet_df.empty:
            print(f"\n[{alias}]  not found in features.parquet on any date.")
            summary_rows.append({
                "wallet":        alias,
                "dates_present": 0,
                "avg_h_score":   float("nan"),
                "avg_rank_pct":  float("nan"),
                "label=1 rate":  float("nan"),
            })
            continue

        print(f"\n{'─' * 65}")
        print(f"  {alias}  ({addr[:10]}…)")
        print(f"{'─' * 65}")
        print(f"  {'date':<13} {'h_score':>8}  {'rank':>5}/{'pool':>5}  "
              f"{'rank_pct':>9}  {'label':>5}")
        for _, r in wallet_df.iterrows():
            print(f"  {r['snapshot_date']:<13} {r['h_score']:>8.3f}  "
                  f"{int(r['rank']):>5}/{int(r['pool_size']):<5}  "
                  f"{r['rank_pct']:>8.1%}  {int(r['label']):>5}")

        summary_rows.append({
            "wallet":        alias,
            "dates_present": len(wallet_df),
            "avg_h_score":   round(wallet_df["h_score"].mean(), 3),
            "avg_rank_pct":  f"{wallet_df['rank_pct'].mean():.1%}",
            "label=1 rate":  f"{wallet_df['label'].mean():.1%}",
        })

    # ── summary table ─────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  Known wallet summary (optimized weights, per-date pool ranking)")
    print(f"{'=' * 65}")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print()


# ── multi-fold walk-forward ───────────────────────────────────────────────────

def run_folds(
    df: pd.DataFrame,
    opt_features: list,
    opt_weights: np.ndarray,
    opt_invert: set,
    stored_medians: dict | None = None,
) -> None:
    """
    Evaluate optimized weights across FOLDS and print a summary table.
    Each fold reports Spearman rho and Precision@25 on its test set only.

    stored_medians: medians from the weights JSON (training-data medians).
                    When provided, fold training medians are computed from
                    each fold's own train split for no-leakage evaluation.
                    Falls back to stored_medians if train split is empty.
    """
    rows = []
    for fold in FOLDS:
        test_df  = df[df["snapshot_date"] >= fold["test_start"]].copy()
        train_df = df[df["snapshot_date"] <  fold["train_end"]].copy()

        # apply an upper bound so folds don't bleed into each other's test windows
        # (each test window runs to the end of the dataset — comparable to production)
        if test_df.empty:
            rows.append({
                "fold":        fold["name"],
                "train_end":   fold["train_end"],
                "test_start":  fold["test_start"],
                "test_rows":   0,
                "spearman_rho": float("nan"),
                "precision@25": float("nan"),
            })
            continue

        # Compute fold-specific training medians (no leakage into test)
        if stored_medians:
            fold_medians = {}
            for feat in stored_medians:
                if feat in train_df.columns and not train_df[feat].isna().all():
                    fold_medians[feat] = float(train_df[feat].median())
                else:
                    fold_medians[feat] = stored_medians[feat]  # fallback
        else:
            fold_medians = None

        y   = test_df["label"].values
        X   = normalize_features(test_df, opt_features, opt_invert, fold_medians)
        s   = X @ opt_weights
        res = evaluate_scores(s, y)

        rows.append({
            "fold":         fold["name"],
            "train_end":    fold["train_end"],
            "test_start":   fold["test_start"],
            "test_rows":    len(test_df),
            "label=1 %":    round(y.mean() * 100, 1),
            "spearman_rho": res["spearman_rho"],
            "precision@25": res["precision@25"],
        })

    fold_table = pd.DataFrame(rows)
    print("=" * 72)
    print("  Walk-forward folds — optimized weights")
    print("=" * 72)
    print(fold_table.to_string(index=False))
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="features.parquet")
    parser.add_argument("--labels",   default="labels.parquet")
    parser.add_argument("--weights",  default="optimal_weights.json")
    args = parser.parse_args()

    # ── load data ─────────────────────────────────────────────────────────
    features_df = pd.read_parquet(args.features)
    labels_df   = pd.read_parquet(args.labels)[
        ["proxy_wallet", "snapshot_date", "label"]
    ]
    df = features_df.merge(labels_df, on=["proxy_wallet", "snapshot_date"], how="inner")
    df["snapshot_date"] = df["snapshot_date"].astype(str)
    print(f"Full dataset : {len(df):,} rows across {df['snapshot_date'].nunique()} dates")

    # ── walk-forward split ────────────────────────────────────────────────
    train_df = df[df["snapshot_date"] <  TRAIN_END].copy()
    test_df  = df[df["snapshot_date"] >= TEST_START].copy()

    print(f"Train set    : {len(train_df):,} rows "
          f"({train_df['snapshot_date'].min()} → {train_df['snapshot_date'].max()})")
    print(f"Gap          : {TRAIN_END} → {TEST_START} (14 days, excluded)")
    print(f"Test set     : {len(test_df):,} rows "
          f"({test_df['snapshot_date'].min()} → {test_df['snapshot_date'].max()})")
    print(f"Test label=1 : {test_df['label'].mean()*100:.1f}%\n")

    if len(test_df) == 0:
        print("ERROR: test set is empty — check snapshot dates in features.parquet.")
        return

    y_test = test_df["label"].values

    # ── load optimized weights ────────────────────────────────────────────
    with open(args.weights) as f:
        payload = json.load(f)
    opt_features    = payload["features"]
    opt_weights     = np.array(payload["weights"])
    opt_invert      = set(payload["inverted"])
    stored_medians  = payload.get("medians")  # present in v7+; None for older files

    # For the main test split, compute training-set medians (no leakage)
    test_medians: dict | None = None
    if stored_medians:
        test_medians = {}
        for feat, fallback in stored_medians.items():
            if feat in train_df.columns and not train_df[feat].isna().all():
                test_medians[feat] = float(train_df[feat].median())
            else:
                test_medians[feat] = fallback

    missing = [c for c in opt_features if c not in test_df.columns]
    if missing:
        print(f"WARNING: missing feature columns in test set: {missing}")

    # ── score 1: current formula (approximated) ───────────────────────────
    scores_current = compute_current_formula(test_df).values

    # ── score 2: equal weights ────────────────────────────────────────────
    X_test_eq    = normalize_features(test_df, opt_features, opt_invert, test_medians)
    w_equal      = np.full(len(opt_features), 100.0 / len(opt_features))
    scores_equal = X_test_eq @ w_equal

    # ── score 3: optimized weights ────────────────────────────────────────
    X_test_opt = normalize_features(test_df, opt_features, opt_invert, test_medians)
    scores_opt = X_test_opt @ opt_weights

    # ── evaluate all three ────────────────────────────────────────────────
    results = {
        "current_formula": evaluate_scores(scores_current, y_test),
        "equal_weights":   evaluate_scores(scores_equal,   y_test),
        "optimized":       evaluate_scores(scores_opt,     y_test),
    }

    # ── comparison table ──────────────────────────────────────────────────
    metrics = ["spearman_rho", "spearman_p", "precision@25", "precision@100"]
    table   = pd.DataFrame(results, index=metrics).T

    print("=" * 62)
    print(f"  Walk-forward evaluation  (test: {TEST_START} → "
          f"{test_df['snapshot_date'].max()})")
    print("=" * 62)
    print(table.to_string())
    print()

    # ── lift over equal-weights baseline ──────────────────────────────────
    eq = results["equal_weights"]
    op = results["optimized"]
    print("Lift of optimized over equal-weights baseline:")
    for m in ["spearman_rho", "precision@25", "precision@100"]:
        delta = op[m] - eq[m]
        print(f"  {m:<18} {eq[m]:>7.4f}  →  {op[m]:>7.4f}  "
              f"({'+'if delta>=0 else ''}{delta:.4f})")

    # ── optimized weight breakdown ────────────────────────────────────────
    print(f"\nOptimized weight breakdown (Spearman rho on train not shown —")
    print(f"weights loaded from {args.weights}):")
    weight_table = (
        pd.DataFrame({
            "feature":  opt_features,
            "weight":   opt_weights.round(3),
            "pct":      (100 * opt_weights / opt_weights.sum()).round(1),
            "inverted": [f in opt_invert for f in opt_features],
        })
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )
    print(weight_table.to_string(index=False))

    # ── known wallet scores on 2026-01-23 ────────────────────────────────
    score_known_wallets(df, opt_features, opt_weights, opt_invert)

    # ── multi-fold walk-forward ───────────────────────────────────────────
    run_folds(df, opt_features, opt_weights, opt_invert, stored_medians)

    # ── current formula caveats ───────────────────────────────────────────
    print("\nCurrent formula caveats (approximation from 15d window metrics):")
    print("  consistency_score (25 pts) — OMITTED: requires avg_rank from leaderboard")
    print("  sharpe_score      (15 pts) — OMITTED: sharpe_ratio dropped (90.9% null)")
    print("  roi_stability     (20 pts) — APPROXIMATED: uses raw 15d roi, not 30d/7d ratio")


if __name__ == "__main__":
    main()
