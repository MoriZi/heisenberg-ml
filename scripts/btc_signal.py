"""
Run the BTC Up/Down signal card pipeline.

Usage:
    python scripts/btc_signal.py backtest --last 100
    python scripts/btc_signal.py backtest --days 14
    python scripts/btc_signal.py backtest --last 100 --features streak decisiveness
    python scripts/btc_signal.py backtest --15m --last 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="BTC Up/Down signal card pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── Backtest ────────────────────────────────────────────────────────
    bt = sub.add_parser("backtest", help="Run backtest over markets")
    bt.add_argument("--last", type=int, default=None,
                     help="Evaluate the last N markets")
    bt.add_argument("--days", type=int, default=None,
                     help="Lookback days (default: 7)")
    bt.add_argument("--start", type=str, default=None,
                     help="Start date (YYYY-MM-DD)")
    bt.add_argument("--end", type=str, default=None,
                     help="End date (YYYY-MM-DD)")
    bt.add_argument("--15m", dest="fifteen_min", action="store_true",
                     help="Use 15-minute markets instead of 5-minute")
    bt.add_argument("--features", nargs="+", default=None,
                     help="Feature names to include (default: all)")
    bt.add_argument("--no-save", action="store_true",
                     help="Skip saving CSV output")

    args = parser.parse_args()

    from src.models.btc_signal.config import (
        BTCSignalConfig,
        MARKET_SLUG_PATTERN_5M,
        MARKET_SLUG_PATTERN_15M,
    )

    pattern = MARKET_SLUG_PATTERN_15M if args.fifteen_min else MARKET_SLUG_PATTERN_5M

    if args.command == "backtest":
        from src.models.btc_signal.backtest import run_backtest

        cfg = BTCSignalConfig(slug_pattern=pattern)
        run_backtest(
            days=args.days,
            last=args.last,
            start_date=args.start,
            end_date=args.end,
            features=args.features,
            slug_pattern=pattern,
            config=cfg,
            save_csv=not args.no_save,
        )


if __name__ == "__main__":
    main()
