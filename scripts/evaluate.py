"""
Evaluate a trained model with walk-forward validation.

Usage:
    python scripts/evaluate.py --model hscore
    python scripts/evaluate.py --model sport_hscore
    python scripts/evaluate.py --model hscore --features data/hscore/features_multiwindow.parquet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


MODEL_DEFAULTS = {
    "hscore": {
        "features": "data/hscore/features_multiwindow.parquet",
        "labels":   "data/hscore/labels.parquet",
        "weights":  "src/models/hscore/artifacts/optimal_weights.json",
    },
    "sport_hscore": {
        "features": "data/sport_hscore/features_multiwindow.parquet",
        "labels":   "data/sport_hscore/labels.parquet",
        "weights":  "src/models/sport_hscore/artifacts/optimal_weights.json",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--model", required=True, choices=list(MODEL_DEFAULTS))
    parser.add_argument("--features", default=None)
    parser.add_argument("--labels",   default=None)
    parser.add_argument("--weights",  default=None)
    args = parser.parse_args()

    defs = MODEL_DEFAULTS[args.model]
    args.features = args.features or defs["features"]
    args.labels   = args.labels   or defs["labels"]
    args.weights  = args.weights  or defs["weights"]

    if args.model == "hscore":
        from src.models.hscore.evaluate import run_evaluation
    elif args.model == "sport_hscore":
        from src.models.sport_hscore.evaluate import run_evaluation
    else:
        print(f"Unknown model: {args.model}")
        sys.exit(1)

    run_evaluation(args.features, args.labels, args.weights)


if __name__ == "__main__":
    main()
