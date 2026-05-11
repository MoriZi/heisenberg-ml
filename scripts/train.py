"""
Train an H-Score model.

Usage:
    python scripts/train.py --model hscore
    python scripts/train.py --model sport_hscore
    python scripts/train.py --model hscore --n-init 100 --k 25
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


MODEL_DEFAULTS = {
    "hscore": {
        "features": "data/hscore/features_multiwindow.parquet",
        "labels":   "data/hscore/labels.parquet",
        "out":      "src/models/hscore/artifacts/optimal_weights.json",
    },
    "sport_hscore": {
        "features": "data/sport_hscore/features_multiwindow.parquet",
        "labels":   "data/sport_hscore/labels.parquet",
        "out":      "src/models/sport_hscore/artifacts/optimal_weights.json",
    },
}


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

    config = HScoreConfig(n_init=args.n_init, seed=args.seed, k=args.k)
    _run_train(
        config, args,
        load_data=load_data,
        build_date_groups=build_date_groups,
        compute_pak=compute_p25,
        optimize_fn=optimize,
        validate_spearman_fn=validate_spearman,
        build_weight_table_fn=build_weight_table,
        save_weights_fn=save_weights,
        score_key="p25",
    )


def train_sport_hscore(args):
    from src.models.sport_hscore.config import SportHScoreConfig
    from src.models.sport_hscore.optimizer import (
        build_date_groups,
        build_weight_table,
        compute_p_at_k,
        load_data,
        optimize,
        save_weights,
        validate_spearman,
    )

    config = SportHScoreConfig(n_init=args.n_init, seed=args.seed, k=args.k)
    _run_train(
        config, args,
        load_data=load_data,
        build_date_groups=build_date_groups,
        compute_pak=compute_p_at_k,
        optimize_fn=optimize,
        validate_spearman_fn=validate_spearman,
        build_weight_table_fn=build_weight_table,
        save_weights_fn=save_weights,
        score_key="p_at_k",
    )


def _run_train(
    config, args, *,
    load_data, build_date_groups, compute_pak, optimize_fn,
    validate_spearman_fn, build_weight_table_fn, save_weights_fn, score_key,
):
    import numpy as np

    print(f"\nFeature count : {config.n_features}")
    print(f"Inverted      : {len(config.invert)}")
    print(f"Objective     : Precision@{config.k}")
    print(f"\nLoading data...")

    X, y, medians, dates = load_data(args.features, args.labels, config)

    print(f"\nBuilding date groups (k={config.k})...")
    date_groups = build_date_groups(dates, k=config.k)

    w_equal = np.full(config.n_features, 100.0 / config.n_features)
    p_eq = compute_pak(X, y, w_equal, date_groups, config.k)
    rho_eq, _ = validate_spearman_fn(X, y, w_equal)
    print(f"  Equal-weights baseline  P@{config.k} = {p_eq:.4f}  (Spearman = {rho_eq:.4f})")

    print(f"\nRunning SLSQP optimization ({config.n_init} random inits)...")
    best = optimize_fn(X, y, date_groups, config)
    weights = best["weights"]

    p_opt = compute_pak(X, y, weights, date_groups, config.k)
    rho_opt, _ = validate_spearman_fn(X, y, weights)

    table = build_weight_table_fn(weights, config)

    print(f"\n{'-' * 60}")
    print(f"  P@{config.k} optimized = {p_opt:.4f}  ({p_opt - p_eq:+.4f} vs equal)")
    print(f"  Spearman rho = {rho_opt:.4f}")
    print(f"{'-' * 60}")

    top15 = table.head(15)
    print(f"\n  Top 15 features by weight:")
    print(f"  {'Feature':<35}  {'Weight':>7}  {'%':>6}  {'Inv':>4}")
    for _, row in top15.iterrows():
        inv = "yes" if row["inverted"] else ""
        print(f"  {row['feature']:<35}  {row['weight']:>7.3f}  {row['pct_total']:>5.1f}%  {inv:>4}")

    save_weights_fn(weights, medians, rho_opt, p_opt, config, args.out)


def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--model", required=True, choices=list(MODEL_DEFAULTS))
    parser.add_argument("--features", default=None)
    parser.add_argument("--labels",   default=None)
    parser.add_argument("--out",      default=None)
    parser.add_argument("--n-init", type=int, default=50)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--k",      type=int, default=25)
    args = parser.parse_args()

    defs = MODEL_DEFAULTS[args.model]
    args.features = args.features or defs["features"]
    args.labels   = args.labels   or defs["labels"]
    args.out      = args.out      or defs["out"]

    if args.model == "hscore":
        train_hscore(args)
    elif args.model == "sport_hscore":
        train_sport_hscore(args)
    else:
        print(f"Unknown model: {args.model}")
        sys.exit(1)


if __name__ == "__main__":
    main()
