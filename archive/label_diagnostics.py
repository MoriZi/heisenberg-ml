"""
label_diagnostics.py

Prints the distribution of forward_pnl and forward_rank from labels.parquet
to help calibrate the label=1 threshold.

Usage:
    python label_diagnostics.py
    python label_diagnostics.py --labels labels.parquet
"""

import argparse
import pandas as pd
import numpy as np


def fmt(v: float) -> str:
    """Format a dollar value with commas."""
    return f"${v:>12,.2f}"


def pct(n: int, total: int) -> str:
    return f"{n:>7,}  ({100 * n / total:.1f}%)"


def print_section(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="labels.parquet",
                        help="Path to labels parquet file (default: labels.parquet)")
    args = parser.parse_args()

    df = pd.read_parquet(args.labels)
    total = len(df)

    print(f"\nLoaded: {args.labels}")
    print(f"Rows            : {total:,}")
    print(f"Unique wallets  : {df['proxy_wallet'].nunique():,}")
    print(f"Snapshot dates  : {df['snapshot_date'].nunique()}  "
          f"({df['snapshot_date'].min()} → {df['snapshot_date'].max()})")

    # ── forward_pnl distribution ──────────────────────────────────────────
    print_section("forward_pnl distribution (all wallets)")
    pnl = df["forward_pnl"]
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        print(f"  p{p:<3}  {fmt(np.percentile(pnl, p))}")
    print(f"  mean   {fmt(pnl.mean())}")
    print(f"  std    {fmt(pnl.std())}")
    print(f"  min    {fmt(pnl.min())}")
    print(f"  max    {fmt(pnl.max())}")

    # ── % positive / negative ─────────────────────────────────────────────
    print_section("forward_pnl sign split")
    n_pos  = (pnl > 0).sum()
    n_zero = (pnl == 0).sum()
    n_neg  = (pnl < 0).sum()
    print(f"  positive (> 0)   {pct(n_pos,  total)}")
    print(f"  zero    (= 0)    {pct(n_zero, total)}")
    print(f"  negative (< 0)   {pct(n_neg,  total)}")

    # ── % above absolute forward_pnl thresholds ───────────────────────────
    print_section("% of wallets above forward_pnl thresholds")
    thresholds = [0, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000]
    for t in thresholds:
        n = (pnl > t).sum()
        print(f"  > {fmt(t).strip():>12}   {pct(n, total)}")

    # ── forward_rank distribution ─────────────────────────────────────────
    print_section("forward_rank distribution (all wallets)")
    rank = df["forward_rank"]
    for p in [10, 25, 50, 75, 90]:
        print(f"  p{p:<3}  {np.percentile(rank, p):>8,.0f}")
    print(f"  mean   {rank.mean():>8,.1f}")
    print(f"  max    {rank.max():>8,}")

    # ── % within rank thresholds ──────────────────────────────────────────
    print_section("% of wallets within rank thresholds")
    rank_thresholds = [100, 250, 500, 1_000, 2_500]
    for t in rank_thresholds:
        n = (rank <= t).sum()
        print(f"  rank <= {t:<5}   {pct(n, total)}")

    # ── combined: forward_pnl > 0 AND rank <= N ───────────────────────────
    print_section("label=1 count under different rank thresholds  (pnl > 0 AND rank <= N)")
    for t in rank_thresholds:
        n = ((pnl > 0) & (rank <= t)).sum()
        print(f"  rank <= {t:<5}   {pct(n, total)}")

    # ── current label distribution ────────────────────────────────────────
    print_section("current label distribution (label=1 threshold in labels.parquet)")
    n_label1 = df["label"].sum()
    n_label0 = total - n_label1
    print(f"  label = 1   {pct(n_label1, total)}")
    print(f"  label = 0   {pct(n_label0, total)}")

    # ── by snapshot date ──────────────────────────────────────────────────
    print_section("label=1 rate by snapshot date")
    by_date = (
        df.groupby("snapshot_date")["label"]
        .agg(total="count", positive="sum")
        .assign(pct_pos=lambda x: 100 * x["positive"] / x["total"])
    )
    print(by_date.to_string())

    print()


if __name__ == "__main__":
    main()
