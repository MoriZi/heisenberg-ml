"""
Sport H-Score walk-forward evaluation.

Compares three scoring variants on the test set:
    1. sports_pnl_baseline — rank by sports_pnl only
    2. equal_weights       — uniform weight across optimizer features
    3. optimized           — weights from the trained model

Also runs multi-fold walk-forward CV and known-wallet scoring.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.common.evaluation import evaluate_scores, precision_at_k
from src.common.features import normalize_features
from src.models.sport_hscore.config import (
    FOLDS,
    KNOWN_WALLETS,
    TEST_START,
    TRAIN_END,
)


# ── Sports PnL baseline ─────────────────────────────────────────────────────


def compute_sports_pnl_baseline(df: pd.DataFrame) -> pd.Series:
    """Baseline: rank by sports_pnl (higher = better)."""
    return df["sports_pnl"].fillna(0).astype(float)


# ── Known wallet scoring ────────────────────────────────────────────────────


def score_known_wallets(
    df: pd.DataFrame,
    opt_features: list,
    opt_weights: np.ndarray,
    opt_invert: set,
) -> None:
    """Score and rank known wallets within each day's eligible pool."""
    scored_parts = []
    for date_str, day_df in df.groupby("snapshot_date"):
        day_df = day_df.copy().reset_index(drop=True)
        X = normalize_features(day_df, opt_features, opt_invert)
        day_df["sport_h_score"] = X @ opt_weights
        n = len(day_df)
        day_df["rank"] = day_df["sport_h_score"].rank(ascending=False, method="min").astype(int)
        day_df["rank_pct"] = day_df["rank"] / n
        day_df["pool_size"] = n
        scored_parts.append(day_df)

    scored = pd.concat(scored_parts, ignore_index=True)

    summary_rows = []
    for alias, addr in KNOWN_WALLETS.items():
        wallet_df = (
            scored[scored["proxy_wallet"] == addr]
            [["snapshot_date", "sport_h_score", "rank", "pool_size", "rank_pct", "label"]]
            .sort_values("snapshot_date")
            .reset_index(drop=True)
        )

        if wallet_df.empty:
            print(f"\n[{alias}]  not found in features on any date.")
            summary_rows.append({
                "wallet": alias, "dates_present": 0,
                "avg_sport_h_score": float("nan"), "avg_rank_pct": float("nan"),
                "label=1 rate": float("nan"),
            })
            continue

        print(f"\n{'─' * 65}")
        print(f"  {alias}  ({addr[:10]}...)")
        print(f"{'─' * 65}")
        print(f"  {'date':<13} {'score':>8}  {'rank':>5}/{'pool':>5}  {'rank_pct':>9}  {'label':>5}")
        for _, r in wallet_df.iterrows():
            print(
                f"  {r['snapshot_date']:<13} {r['sport_h_score']:>8.3f}  "
                f"{int(r['rank']):>5}/{int(r['pool_size']):<5}  "
                f"{r['rank_pct']:>8.1%}  {int(r['label']):>5}"
            )

        summary_rows.append({
            "wallet": alias,
            "dates_present": len(wallet_df),
            "avg_sport_h_score": round(wallet_df["sport_h_score"].mean(), 3),
            "avg_rank_pct": f"{wallet_df['rank_pct'].mean():.1%}",
            "label=1 rate": f"{wallet_df['label'].mean():.1%}",
        })

    print(f"\n{'=' * 65}")
    print("  Known wallet summary (optimized weights, per-date pool ranking)")
    print(f"{'=' * 65}")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print()


# ── Multi-fold walk-forward ─────────────────────────────────────────────────


def run_folds(
    df: pd.DataFrame,
    opt_features: list,
    opt_weights: np.ndarray,
    opt_invert: set,
    stored_medians: dict | None = None,
) -> None:
    """Evaluate optimized weights across walk-forward folds."""
    rows = []
    for fold in FOLDS:
        test_df = df[df["snapshot_date"] >= fold["test_start"]].copy()
        train_df = df[df["snapshot_date"] < fold["train_end"]].copy()

        if test_df.empty:
            rows.append({
                "fold": fold["name"], "train_end": fold["train_end"],
                "test_start": fold["test_start"], "test_rows": 0,
                "spearman_rho": float("nan"), "precision@25": float("nan"),
            })
            continue

        fold_medians = None
        if stored_medians:
            fold_medians = {}
            for feat in stored_medians:
                if feat in train_df.columns and not train_df[feat].isna().all():
                    fold_medians[feat] = float(train_df[feat].median())
                else:
                    fold_medians[feat] = stored_medians[feat]

        y = test_df["label"].values
        X = normalize_features(test_df, opt_features, opt_invert, fold_medians)
        s = X @ opt_weights
        res = evaluate_scores(s, y)

        rows.append({
            "fold": fold["name"], "train_end": fold["train_end"],
            "test_start": fold["test_start"], "test_rows": len(test_df),
            "label=1 %": round(y.mean() * 100, 1),
            "spearman_rho": res["spearman_rho"], "precision@25": res["precision@25"],
        })

    print("=" * 72)
    print("  Walk-forward folds — optimized weights")
    print("=" * 72)
    print(pd.DataFrame(rows).to_string(index=False))
    print()


# ── Main evaluation ─────────────────────────────────────────────────────────


def run_evaluation(
    features_path: str,
    labels_path: str,
    weights_path: str,
) -> None:
    """Full walk-forward evaluation comparing all scoring variants."""
    features_df = pd.read_parquet(features_path)
    labels_df = pd.read_parquet(labels_path)[["proxy_wallet", "snapshot_date", "label"]]
    df = features_df.merge(labels_df, on=["proxy_wallet", "snapshot_date"], how="inner")
    df["snapshot_date"] = df["snapshot_date"].astype(str)
    print(f"Full dataset : {len(df):,} rows across {df['snapshot_date'].nunique()} dates")

    train_df = df[df["snapshot_date"] < TRAIN_END].copy()
    test_df = df[df["snapshot_date"] >= TEST_START].copy()

    print(f"Train set    : {len(train_df):,} rows")
    print(f"Test set     : {len(test_df):,} rows")
    print(f"Test label=1 : {test_df['label'].mean() * 100:.1f}%\n")

    if len(test_df) == 0:
        print("ERROR: test set is empty.")
        return

    y_test = test_df["label"].values

    with open(weights_path) as f:
        payload = json.load(f)
    opt_features = payload["features"]
    opt_weights = np.array(payload["weights"])
    opt_invert = set(payload["inverted"])
    stored_medians = payload.get("medians")

    test_medians = None
    if stored_medians:
        test_medians = {}
        for feat, fallback in stored_medians.items():
            if feat in train_df.columns and not train_df[feat].isna().all():
                test_medians[feat] = float(train_df[feat].median())
            else:
                test_medians[feat] = fallback

    # Score 1: sports PnL baseline
    scores_baseline = compute_sports_pnl_baseline(test_df).values

    # Score 2: equal weights
    X_test = normalize_features(test_df, opt_features, opt_invert, test_medians)
    w_equal = np.full(len(opt_features), 100.0 / len(opt_features))
    scores_equal = X_test @ w_equal

    # Score 3: optimized
    scores_opt = X_test @ opt_weights

    results = {
        "sports_pnl_baseline": evaluate_scores(scores_baseline, y_test),
        "equal_weights": evaluate_scores(scores_equal, y_test),
        "optimized": evaluate_scores(scores_opt, y_test),
    }

    metrics = ["spearman_rho", "spearman_p", "precision@25", "precision@100"]
    table = pd.DataFrame(results, index=metrics).T
    print("=" * 62)
    print(f"  Walk-forward evaluation  (test: {TEST_START} -> {test_df['snapshot_date'].max()})")
    print("=" * 62)
    print(table.to_string())
    print()

    # Lift
    eq = results["equal_weights"]
    op = results["optimized"]
    print("Lift of optimized over equal-weights baseline:")
    for m in ["spearman_rho", "precision@25", "precision@100"]:
        delta = op[m] - eq[m]
        print(f"  {m:<18} {eq[m]:>7.4f}  ->  {op[m]:>7.4f}  ({'+'if delta >= 0 else ''}{delta:.4f})")

    # Known wallets
    score_known_wallets(df, opt_features, opt_weights, opt_invert)

    # Folds
    run_folds(df, opt_features, opt_weights, opt_invert, stored_medians)
