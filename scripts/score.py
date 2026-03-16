"""
Score wallets using a deployed model formula.

Usage:
    python scripts/score.py --model hscore
    python scripts/score.py --model hscore --sql src/models/hscore/artifacts/deploy_formula.sql
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Score wallets with a model")
    parser.add_argument("--model", required=True, choices=["hscore"])
    parser.add_argument("--sql", default="src/models/hscore/artifacts/deploy_formula.sql")
    parser.add_argument("--output-dir", default="data/hscore")
    args = parser.parse_args()

    if args.model == "hscore":
        from src.models.hscore.deploy import score_and_save
        score_and_save(args.sql, args.output_dir)
    else:
        print(f"Unknown model: {args.model}")
        sys.exit(1)


if __name__ == "__main__":
    main()
