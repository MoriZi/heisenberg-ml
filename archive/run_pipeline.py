"""
run_pipeline.py

Runs the full eligibility → features → labels pipeline for every calendar
day in the snapshot range, accumulates results, and saves combined parquets.

Snapshot range:
    start : 2025-11-24
    end   : 2026-01-23
    cadence: daily

Performance — exactly 2 DB queries total, before the loop:
    1. wallet_profile_metrics (window=15) with eligibility filters in SQL
       → supplies both eligible wallet sets AND feature rows
    2. wallet_daily_pnl for the full forward-label range (T+1 → T+14)
       → in-memory aggregation per snapshot date

The loop itself makes zero DB calls — pure dict lookups.

Outputs:
    features.parquet
    labels.parquet

Usage:
    python run_pipeline.py                        # full run, 7-day forward window
    python run_pipeline.py --forward-days 14      # 14-day forward window
    python run_pipeline.py --test                 # first 5 dates only
"""

import argparse
import pandas as pd
from tqdm import tqdm

from db import get_engine
from build_features import (
    METRIC_COLS,
    FILLNA_ZERO_COLS,
    FILLNA_MEDIAN_COLS,
    parse_dominant_category,
    parse_category_pnl,
)

# ── config ────────────────────────────────────────────────────────────────────
START_DATE     = "2025-11-24"
END_DATE       = "2026-01-23"
RANK_THRESHOLD = 500
FEATURES_OUT   = "features.parquet"
LABELS_OUT     = "labels.parquet"


# ── precompute: features + eligibility ───────────────────────────────────────

def precompute_features_and_eligibility(snapshots: pd.DatetimeIndex, engine) -> tuple:
    """
    Single query on wallet_profile_metrics with all eligibility filters
    pushed into the SQL WHERE clause. The same rows serve as both the
    eligible wallet set and the feature matrix — no separate query needed.

    Eligibility filters applied in SQL:
        roi > 0
        win_rate BETWEEN 0.45 AND 0.95
        total_trades BETWEEN 50 AND 100000
        total_pnl > 5000
        combined_risk_score <= 50

    Returns:
        features_cache    {date_str: DataFrame of eligible wallet features}
        eligibility_cache {date_str: set(proxy_wallet)}
    """
    # Include date in SELECT so we can partition by it after fetch
    col_select = "date, " + ", ".join(METRIC_COLS)

    sql = f"""
        SELECT {col_select}
        FROM polymarket.wallet_profile_metrics
        WHERE date >= %(start)s
          AND date <= %(end)s
          AND calculation_window_days = 15
          AND roi > 0
          AND win_rate BETWEEN 0.45 AND 0.95
          AND total_trades BETWEEN 50 AND 100000
          AND total_pnl > 5000
          AND combined_risk_score <= 50
    """
    print(f"Precomputing features + eligibility "
          f"({snapshots[0].date()} → {snapshots[-1].date()})...")
    df = pd.read_sql(
        sql, engine,
        params={
            "start": snapshots[0].strftime("%Y-%m-%d"),
            "end":   snapshots[-1].strftime("%Y-%m-%d"),
        },
    )
    print(f"  Eligible rows loaded: {len(df):,}")

    # ── JSONB parsing ─────────────────────────────────────────────────────
    df["dominant_category"] = df["performance_by_category"].apply(parse_dominant_category)
    cat_pnl_df = pd.DataFrame(
        df["performance_by_category"].apply(parse_category_pnl).tolist(),
        index=df.index,
    )
    df = pd.concat([df, cat_pnl_df], axis=1)
    df = df.drop(columns=["performance_by_category"])

    # ── null handling ─────────────────────────────────────────────────────
    # FILLNA_ZERO_COLS is now empty; kept for backward-compat.
    for col in FILLNA_ZERO_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Sparse ratio columns: impute with global median across all fetched
    # rows (before date partitioning). Using global median here is correct
    # because the full eligible population is already in memory and median
    # is more representative than 0 when ~50% of values are missing.
    for col in FILLNA_MEDIAN_COLS:
        if col in df.columns:
            median_val = df[col].median()   # pandas median() ignores NaN
            df[col] = df[col].fillna(median_val)
            print(f"  Median-imputed {col}: {median_val:.4f}")

    # ── normalise date column to clean string for dict keys ───────────────
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["snapshot_date"] = df["date"]

    # ── partition by date ─────────────────────────────────────────────────
    features_cache    = {}
    eligibility_cache = {}
    for date_str, group in df.groupby("date"):
        features_cache[date_str] = (
            group.drop(columns=["date"]).reset_index(drop=True)
        )
        eligibility_cache[date_str] = set(group["proxy_wallet"])

    print(f"  Dates with eligible wallets: {len(features_cache)}")
    return features_cache, eligibility_cache


# ── precompute: forward labels ────────────────────────────────────────────────

def precompute_labels(
    snapshots: pd.DatetimeIndex,
    eligibility_cache: dict,
    engine,
    forward_days: int = 7,
) -> dict:
    """
    Single SQL query for all snapshot dates using a date spine.

    Structure:
      1. generate_series produces all snapshot dates in the range.
      2. daily_pnl aggregates wallet_daily_pnl to (wallet, date) totals
         for the entire forward range — one scan of the table.
      3. forward_pnl joins daily_pnl to the snapshot spine; each pnl row
         contributes to every snapshot date T where date BETWEEN T+1 AND T+14.
      4. ranked applies RANK() OVER (PARTITION BY snapshot_date ...) so
         forward_rank is computed against ALL wallets active in that window,
         not just the eligible subset.
      5. Python filters to eligible wallets per date after fetch —
         after global ranking, so rank values are correct.

    Settlement rows (trades=0, invested=0, pnl≠0) are included — valid per spec.

    Returns:
        {date_str: DataFrame with proxy_wallet, snapshot_date,
                   forward_pnl, forward_rank, label}
    """
    start     = snapshots[0].strftime("%Y-%m-%d")
    end       = snapshots[-1].strftime("%Y-%m-%d")
    fwd_start = (snapshots[0]  + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fwd_end   = (snapshots[-1] + pd.Timedelta(days=forward_days)).strftime("%Y-%m-%d")

    sql = """
        WITH
        -- date spine: one row per snapshot date in the pipeline range
        snapshots (snapshot_date) AS (
            SELECT generate_series(
                %(start)s::date,
                %(end)s::date,
                '1 day'::interval
            )::date
        ),
        -- aggregate raw rows to (wallet, date) before the join to reduce
        -- the size of the intermediate result
        daily_pnl AS (
            SELECT proxy_wallet, date, SUM(pnl) AS pnl
            FROM polymarket.wallet_daily_pnl
            WHERE date BETWEEN %(fwd_start)s AND %(fwd_end)s
            GROUP BY proxy_wallet, date
        ),
        -- assign each daily row to all snapshot dates whose forward window
        -- contains it: date BETWEEN T+1 AND T+14  ↔  T BETWEEN date-14 AND date-1
        forward_pnl AS (
            SELECT
                s.snapshot_date,
                d.proxy_wallet,
                SUM(d.pnl) AS forward_pnl
            FROM daily_pnl d
            JOIN snapshots s
              ON d.date BETWEEN s.snapshot_date + 1 AND s.snapshot_date + %(forward_days)s
            GROUP BY s.snapshot_date, d.proxy_wallet
        ),
        -- rank globally within each snapshot date across ALL active wallets
        ranked AS (
            SELECT
                snapshot_date,
                proxy_wallet,
                forward_pnl,
                RANK() OVER (
                    PARTITION BY snapshot_date
                    ORDER BY forward_pnl DESC
                ) AS forward_rank
            FROM forward_pnl
        )
        SELECT
            snapshot_date::text AS snapshot_date,
            proxy_wallet,
            forward_pnl,
            forward_rank::int   AS forward_rank,
            CASE
                WHEN forward_pnl > 0 AND forward_rank <= %(rank_threshold)s
                THEN 1 ELSE 0
            END AS label
        FROM ranked
    """

    print(f"Precomputing labels: one SQL query for all {len(snapshots)} snapshot dates "
          f"(forward range {fwd_start} → {fwd_end})...")
    df = pd.read_sql(
        sql, engine,
        params={
            "start":          start,
            "end":            end,
            "fwd_start":      fwd_start,
            "fwd_end":        fwd_end,
            "rank_threshold": RANK_THRESHOLD,
            "forward_days":   forward_days,
        },
    )
    print(f"  Rows returned (all wallets, all dates): {len(df):,}")

    # Filter to eligible wallets per date in Python — after global ranking,
    # so forward_rank values reflect the full active population.
    labels_cache = {}
    for snap_str, group in df.groupby("snapshot_date"):
        eligible_wallets = eligibility_cache.get(snap_str, set())
        filtered = group[group["proxy_wallet"].isin(eligible_wallets)].copy()
        labels_cache[snap_str] = filtered.reset_index(drop=True)

    print(f"  Labels cache built for {len(labels_cache)} dates.")
    return labels_cache


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Process only the first 5 dates for validation.")
    parser.add_argument("--forward-days", type=int, default=7,
                        help="Forward window in days for label construction (default: 7).")
    args = parser.parse_args()

    snapshots = pd.date_range(start=START_DATE, end=END_DATE, freq="D")

    if args.test:
        snapshots = snapshots[:5]
        print(f"TEST MODE — processing first 5 dates: "
              f"{snapshots[0].date()} → {snapshots[-1].date()}")
    else:
        print(f"Snapshot range: {START_DATE} → {END_DATE}  ({len(snapshots)} dates)")

    engine = get_engine()

    # ── two bulk queries — zero DB calls inside the loop ──────────────────
    features_cache, eligibility_cache = precompute_features_and_eligibility(
        snapshots, engine
    )
    labels_cache = precompute_labels(snapshots, eligibility_cache, engine,
                                     forward_days=args.forward_days)

    all_features = []
    all_labels   = []
    skipped      = []

    for snap_ts in tqdm(snapshots, desc="Pipeline", unit="date"):
        snap_str = snap_ts.strftime("%Y-%m-%d")

        feat_df  = features_cache.get(snap_str)
        label_df = labels_cache.get(snap_str)

        if feat_df is None or feat_df.empty:
            skipped.append(snap_str)
            continue

        all_features.append(feat_df)
        if label_df is not None and not label_df.empty:
            all_labels.append(label_df)

    # ── report skipped dates ──────────────────────────────────────────────
    if skipped:
        print(f"\nSkipped {len(skipped)} dates (no eligible wallets / no window=15 data):")
        for d in skipped:
            print(f"  {d}")

    if not all_features:
        print("\nNo data collected — nothing saved.")
        return

    # ── combine and save ──────────────────────────────────────────────────
    features_df = pd.concat(all_features, ignore_index=True)
    labels_df   = pd.concat(all_labels,   ignore_index=True)

    features_df.to_parquet(FEATURES_OUT, index=False)
    labels_df.to_parquet(LABELS_OUT,   index=False)

    # ── final summary ─────────────────────────────────────────────────────
    n_pos   = labels_df["label"].sum()
    n_total = len(labels_df)
    pos_pct = 100 * n_pos / n_total if n_total > 0 else 0

    print(f"\nfeatures.parquet : {features_df.shape}")
    print(f"labels.parquet   : {labels_df.shape}")
    print(f"label=1          : {n_pos}/{n_total}  ({pos_pct:.1f}%)")
    print(f"\nDone. Saved {FEATURES_OUT} and {LABELS_OUT}.")


if __name__ == "__main__":
    main()
