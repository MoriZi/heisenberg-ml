"""
Train an H-Score model (or future models).

Usage:
    python scripts/train.py --model hscore
    python scripts/train.py --model hscore --features data/hscore/features_multiwindow.parquet
    python scripts/train.py --model hscore --n-init 100 --k 25
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def train_hscore(args):
    from src.models.hscore.config import HScoreConfig
    from src.models.hscore.optimizer import (
        build_date_groups,
        build_weight_table,
        compute_p25,
        load_data,
        optimize,
        save_weights,
        validate_spearman,
    )

    config = HScoreConfig(
        n_init=args.n_init,
        seed=args.seed,
        k=args.k,
    )

    print(f"\nFeature count : {config.n_features}")
    print(f"Inverted      : {len(config.invert)}")
    print(f"Objective     : Precision@{config.k}")
    print(f"\nLoading data...")

    X, y, medians, dates = load_data(args.features, args.labels, config)

    print(f"\nBuilding date groups (k={config.k})...")
    date_groups = build_date_groups(dates, k=config.k)

    # Baseline
    import numpy as np
    w_equal = np.full(config.n_features, 100.0 / config.n_features)
    p25_equal = compute_p25(X, y, w_equal, date_groups, config.k)
    rho_equal, _ = validate_spearman(X, y, w_equal)
    print(f"  Equal-weights baseline  P@{config.k} = {p25_equal:.4f}  (Spearman = {rho_equal:.4f})")

    print(f"\nRunning SLSQP optimization ({config.n_init} random inits)...")
    best = optimize(X, y, date_groups, config)
    weights = best["weights"]

    p25_opt = compute_p25(X, y, weights, date_groups, config.k)
    rho_opt, _ = validate_spearman(X, y, weights)

    table = build_weight_table(weights, config)

    print(f"\n{'─' * 60}")
    print(f"  P@{config.k} optimized = {p25_opt:.4f}  ({p25_opt - p25_equal:+.4f} vs equal)")
    print(f"  Spearman rho = {rho_opt:.4f}")
    print(f"{'─' * 60}")

    top15 = table.head(15)
    print(f"\n  Top 15 features by weight:")
    print(f"  {'Feature':<35}  {'Weight':>7}  {'%':>6}  {'Inv':>4}")
    for _, row in top15.iterrows():
        inv = "yes" if row["inverted"] else ""
        print(f"  {row['feature']:<35}  {row['weight']:>7.3f}  {row['pct_total']:>5.1f}%  {inv:>4}")

    save_weights(weights, medians, rho_opt, p25_opt, config, args.out)


def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--model", required=True, choices=["hscore"],
                        help="Which model to train")
    parser.add_argument("--features", default="data/hscore/features_multiwindow.parquet")
    parser.add_argument("--labels", default="data/hscore/labels.parquet")
    parser.add_argument("--n-init", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--out", default="src/models/hscore/artifacts/optimal_weights.json")
    args = parser.parse_args()

    if args.model == "hscore":
        train_hscore(args)
    else:
        print(f"Unknown model: {args.model}")
        sys.exit(1)


if __name__ == "__main__":
    main()
