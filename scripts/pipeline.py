"""
Run the data pipeline to build features and labels.

Usage:
    python scripts/pipeline.py --model hscore
    python scripts/pipeline.py --model hscore --forward-days 14
    python scripts/pipeline.py --model hscore --test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Run data pipeline")
    parser.add_argument("--model", required=True, choices=["hscore"])
    parser.add_argument("--forward-days", type=int, default=7)
    parser.add_argument("--test", action="store_true", help="Process only first 5 dates")
    args = parser.parse_args()

    if args.model == "hscore":
        from src.models.hscore.pipeline import run_pipeline
        run_pipeline(forward_days=args.forward_days, test_mode=args.test)
    else:
        print(f"Unknown model: {args.model}")
        sys.exit(1)


if __name__ == "__main__":
    main()
