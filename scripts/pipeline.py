"""
Run the data pipeline to build features and labels.

Usage:
    python scripts/pipeline.py --model hscore
    python scripts/pipeline.py --model sport_hscore
    python scripts/pipeline.py --model hscore --start 2026-03-10 --end 2026-05-04
    python scripts/pipeline.py --model hscore --forward-days 14
    python scripts/pipeline.py --model hscore --test
    python scripts/pipeline.py --model hscore --no-multiwindow
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Run data pipeline")
    parser.add_argument("--model", required=True, choices=["hscore", "sport_hscore"])
    parser.add_argument("--start", default=None, help="Snapshot start date (YYYY-MM-DD); defaults to config PIPELINE_START")
    parser.add_argument("--end", default=None, help="Snapshot end date (YYYY-MM-DD); defaults to config PIPELINE_END")
    parser.add_argument("--forward-days", type=int, default=7)
    parser.add_argument("--test", action="store_true", help="Process only first 5 dates")
    parser.add_argument("--no-multiwindow", action="store_true", help="Skip the 1d/3d/7d join step")
    args = parser.parse_args()

    if args.model == "hscore":
        from src.models.hscore.config import PIPELINE_END, PIPELINE_START
        from src.models.hscore.pipeline import build_multiwindow, run_pipeline
        data_dir = "data/hscore"
    elif args.model == "sport_hscore":
        from src.models.sport_hscore.config import PIPELINE_END, PIPELINE_START
        from src.models.sport_hscore.pipeline import build_multiwindow, run_pipeline
        data_dir = "data/sport_hscore"
    else:
        print(f"Unknown model: {args.model}")
        sys.exit(1)

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    run_pipeline(
        start_date=args.start or PIPELINE_START,
        end_date=args.end or PIPELINE_END,
        forward_days=args.forward_days,
        features_out=f"{data_dir}/features.parquet",
        labels_out=f"{data_dir}/labels.parquet",
        test_mode=args.test,
    )

    if not args.no_multiwindow:
        build_multiwindow(
            base_path=f"{data_dir}/features.parquet",
            out_path=f"{data_dir}/features_multiwindow.parquet",
        )


if __name__ == "__main__":
    main()
