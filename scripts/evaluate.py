"""
Evaluate a trained model with walk-forward validation.

Usage:
    python scripts/evaluate.py --model hscore
    python scripts/evaluate.py --model hscore --weights src/models/hscore/artifacts/optimal_weights.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--model", required=True, choices=["hscore"])
    parser.add_argument("--features", default="data/hscore/features.parquet")
    parser.add_argument("--labels", default="data/hscore/labels.parquet")
    parser.add_argument("--weights", default="src/models/hscore/artifacts/optimal_weights.json")
    args = parser.parse_args()

    if args.model == "hscore":
        from src.models.hscore.evaluate import run_evaluation
        run_evaluation(args.features, args.labels, args.weights)
    else:
        print(f"Unknown model: {args.model}")
        sys.exit(1)


if __name__ == "__main__":
    main()
