"""
Sport H-Score data pipeline.

Orchestrates: eligibility → features → multiwindow → labels.
Produces features.parquet and labels.parquet in the data directory.

Uses wallet_profile_metrics_v2 and focuses on sports-active wallets.
"""

import json

import pandas as pd
from tqdm import tqdm
no
from src.common.db import get_engine
from src.models.sport_hscore.config import (
    ELIGIBILITY_SQL_FILTERS,
    FILLNA_MEDIAN_COLS,
    JSONB_CATS,
    METRIC_COLS,
    MIN_SPORTS_PNL,
    MIN_SPORTS_TRADES,
    PIPELINE_END,
    PIPELINE_START,
    RANK_THRESHOLD,
    SPORTS_SUBCATS,
    TABLE,
    WINDOW_METRIC_COLS,
    WINDOWS,
)


# ── JSONB helpers ────────────────────────────────────────────────────────────


def parse_sports_aggregate(jsonb_val) -> dict:
    """Extract aggregate Sports metrics from performance_by_category.

    Looks for the top-level "Sports" entry in the hierarchical category list.
    Returns dict with keys: sports_pnl, sports_roi, sports_win_rate,
    sports_trades, sports_invested.
    """
    result = {
        "sports_pnl": 0.0,
        "sports_roi": 0.0,
        "sports_win_rate": 0.0,
        "sports_trades": 0,
        "sports_invested": 0.0,
    }
    if jsonb_val is None:
        return result
    try:
        items = json.loads(jsonb_val) if isinstance(jsonb_val, str) else jsonb_val
        if not items:
            return result
        for item in items:
            cat = item.get("category", "")
            if cat == "Sports":
                result["sports_pnl"] = float(item.get("total_pnl", 0))
                result["sports_roi"] = float(item.get("roi", 0))
                result["sports_win_rate"] = float(item.get("win_rate", 0))
                result["sports_trades"] = int(item.get("total_trades", 0))
                result["sports_invested"] = float(item.get("total_invested", 0))
                break
    except Exception:
        pass
    return result


def parse_sports_subcategories(jsonb_val) -> dict:
    """Extract per-subcategory PnL from performance_by_category.

    Matches hierarchical categories like "Sports / Basketball / NBA"
    against SPORTS_SUBCATS mapping. Returns dict with keys like
    sports_pnl_nba, sports_pnl_epl, etc.
    """
    result = {f"sports_pnl_{suffix}": 0.0 for suffix in SPORTS_SUBCATS.values()}
    if jsonb_val is None:
        return result
    try:
        items = json.loads(jsonb_val) if isinstance(jsonb_val, str) else jsonb_val
        for item in items:
            cat = item.get("category", "")
            if cat in SPORTS_SUBCATS:
                suffix = SPORTS_SUBCATS[cat]
                result[f"sports_pnl_{suffix}"] = float(item.get("total_pnl", 0))
    except Exception:
        pass
    return result


def parse_window_sports(jsonb_val) -> dict:
    """Extract sports PnL from performance_by_category for multiwindow."""
    result = {"pnl_cat_sports": 0.0}
    if jsonb_val is None:
        return result
    try:
        items = json.loads(jsonb_val) if isinstance(jsonb_val, str) else jsonb_val
        for item in items:
            if item.get("category", "") == "Sports":
                result["pnl_cat_sports"] = float(item.get("total_pnl", 0))
                break
    except Exception:
        pass
    return result


# ── Bulk pipeline ────────────────────────────────────────────────────────────


def precompute_features_and_eligibility(snapshots, engine):
    """Single query for features + eligibility across all dates.

    Applies SQL-side eligibility filters, then Python-side sports filter.
    """
    col_select = "date, " + ", ".join(METRIC_COLS)

    sql = f"""
        SELECT {col_select}
        FROM {TABLE}
        WHERE date >= %(start)s
          AND date <= %(end)s
          AND calculation_window_days = 15
          {ELIGIBILITY_SQL_FILTERS}
    """
    print(f"Precomputing features + eligibility ({snapshots[0].date()} -> {snapshots[-1].date()})...")
    df = pd.read_sql(
        sql, engine,
        params={
            "start": snapshots[0].strftime("%Y-%m-%d"),
            "end": snapshots[-1].strftime("%Y-%m-%d"),
        },
    )
    print(f"  Eligible rows loaded (pre-sports filter): {len(df):,}")

    # Parse sports aggregate from JSONB
    sports_agg = pd.DataFrame(
        df["performance_by_category"].apply(parse_sports_aggregate).tolist(),
        index=df.index,
    )
    df = pd.concat([df, sports_agg], axis=1)

    # Parse sports sub-categories from JSONB
    sports_sub = pd.DataFrame(
        df["performance_by_category"].apply(parse_sports_subcategories).tolist(),
        index=df.index,
    )
    df = pd.concat([df, sports_sub], axis=1)

    # Apply sports eligibility filter
    sports_mask = (df["sports_trades"] >= MIN_SPORTS_TRADES) & (df["sports_pnl"] > MIN_SPORTS_PNL)
    df = df[sports_mask].copy()
    print(f"  Eligible rows after sports filter: {len(df):,}")

    df = df.drop(columns=["performance_by_category"])

    for col in FILLNA_MEDIAN_COLS:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Median-imputed {col}: {median_val:.4f}")

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["snapshot_date"] = df["date"]

    features_cache = {}
    eligibility_cache = {}
    for date_str, group in df.groupby("date"):
        features_cache[date_str] = group.drop(columns=["date"]).reset_index(drop=True)
        eligibility_cache[date_str] = set(group["proxy_wallet"])

    print(f"  Dates with eligible wallets: {len(features_cache)}")
    return features_cache, eligibility_cache


def precompute_labels(snapshots, eligibility_cache, engine, forward_days=7):
    """Single SQL query for labels across all snapshot dates.

    Only scans wallet_daily_pnl for eligible wallets.
    """
    start = snapshots[0].strftime("%Y-%m-%d")
    end = snapshots[-1].strftime("%Y-%m-%d")
    fwd_start = (snapshots[0] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fwd_end = (snapshots[-1] + pd.Timedelta(days=forward_days)).strftime("%Y-%m-%d")

    all_eligible = set()
    for wallets in eligibility_cache.values():
        all_eligible.update(wallets)

    if not all_eligible:
        print("  No eligible wallets — skipping labels.")
        return {}

    print(f"  Eligible wallet universe: {len(all_eligible):,}")

    sql = """
        WITH
        snapshots (snapshot_date) AS (
            SELECT generate_series(%(start)s::date, %(end)s::date, '1 day'::interval)::date
        ),
        daily_pnl AS (
            SELECT proxy_wallet, date, SUM(pnl) AS pnl
            FROM polymarket.wallet_daily_pnl
            WHERE date BETWEEN %(fwd_start)s AND %(fwd_end)s
              AND proxy_wallet = ANY(%(wallets)s)
            GROUP BY proxy_wallet, date
        ),
        forward_pnl AS (
            SELECT s.snapshot_date, d.proxy_wallet, SUM(d.pnl) AS forward_pnl
            FROM daily_pnl d
            JOIN snapshots s ON d.date BETWEEN s.snapshot_date + 1 AND s.snapshot_date + %(forward_days)s
            GROUP BY s.snapshot_date, d.proxy_wallet
        ),
        ranked AS (
            SELECT snapshot_date, proxy_wallet, forward_pnl,
                   RANK() OVER (PARTITION BY snapshot_date ORDER BY forward_pnl DESC) AS forward_rank
            FROM forward_pnl
        )
        SELECT
            snapshot_date::text AS snapshot_date, proxy_wallet,
            forward_pnl, forward_rank::int AS forward_rank,
            CASE WHEN forward_pnl > 0 AND forward_rank <= %(rank_threshold)s THEN 1 ELSE 0 END AS label
        FROM ranked
    """

    print(f"Precomputing labels ({len(snapshots)} dates, forward {fwd_start} -> {fwd_end})...")
    df = pd.read_sql(
        sql, engine,
        params={
            "start": start, "end": end,
            "fwd_start": fwd_start, "fwd_end": fwd_end,
            "rank_threshold": RANK_THRESHOLD,
            "forward_days": forward_days,
            "wallets": list(all_eligible),
        },
    )
    print(f"  Rows returned: {len(df):,}")

    labels_cache = {}
    for snap_str, group in df.groupby("snapshot_date"):
        eligible_wallets = eligibility_cache.get(snap_str, set())
        filtered = group[group["proxy_wallet"].isin(eligible_wallets)].copy()
        labels_cache[snap_str] = filtered.reset_index(drop=True)

    return labels_cache


# ── Multiwindow extension ───────────────────────────────────────────────────


def fetch_window(engine, window_days: int, wallets: list, dates: list) -> pd.DataFrame:
    """Pull short-window features for the given (wallet, date) universe."""
    plain_cols = [c for c in WINDOW_METRIC_COLS if c != "performance_by_category"]
    col_select = ", ".join(["proxy_wallet", "date"] + plain_cols + ["performance_by_category"])

    sql = f"""
        SELECT {col_select}
        FROM {TABLE}
        WHERE calculation_window_days = %(window)s
          AND proxy_wallet = ANY(%(wallets)s)
          AND date = ANY(%(dates)s::date[])
    """
    params = {"window": window_days, "wallets": wallets, "dates": dates}

    print(f"  Fetching window={window_days}d ({len(wallets):,} wallets x {len(dates)} dates)...")
    df = pd.read_sql(sql, engine, params=params)
    print(f"  Rows returned: {len(df):,}")

    if df.empty:
        return pd.DataFrame(columns=["proxy_wallet", "snapshot_date"])

    # Extract sports PnL from JSONB for this window
    cat_pnl = pd.DataFrame(
        df["performance_by_category"].apply(parse_window_sports).tolist(),
        index=df.index,
    )
    df = pd.concat([df.drop(columns=["performance_by_category"]), cat_pnl], axis=1)
    df = df.rename(columns={"date": "snapshot_date"})
    df["snapshot_date"] = df["snapshot_date"].astype(str)

    suffix = f"_{window_days}d"
    rename_map = {
        col: col + suffix
        for col in plain_cols + ["pnl_cat_sports"]
    }
    df = df.rename(columns=rename_map)
    return df


def build_multiwindow(base_path: str, out_path: str) -> None:
    """Extend base features with 1d, 3d, 7d window features."""
    print(f"\nLoading base: {base_path}")
    base = pd.read_parquet(base_path)
    base["snapshot_date"] = base["snapshot_date"].astype(str)
    n_base = len(base)
    print(f"Base rows   : {n_base:,}")

    all_wallets = base["proxy_wallet"].unique().tolist()
    all_dates = base["snapshot_date"].unique().tolist()

    engine = get_engine()
    merged = base.copy()

    for w in WINDOWS:
        print(f"\n-- Window {w}d --")
        win_df = fetch_window(engine, w, all_wallets, all_dates)

        if win_df.empty:
            plain_cols = [c for c in WINDOW_METRIC_COLS if c != "performance_by_category"]
            new_cols = [c + f"_{w}d" for c in plain_cols + ["pnl_cat_sports"]]
            for col in new_cols:
                merged[col] = 0.0
            continue

        win_df = win_df.drop_duplicates(subset=["proxy_wallet", "snapshot_date"])
        merged = merged.merge(win_df, on=["proxy_wallet", "snapshot_date"], how="left")
        assert len(merged) == n_base
        print(f"  Row count after join: {len(merged):,}")

    engine.dispose()

    new_cols = [
        col for col in merged.columns
        if any(col.endswith(f"_{w}d") for w in WINDOWS)
    ]
    merged[new_cols] = merged[new_cols].fillna(0.0)

    assert len(merged) == n_base
    print(f"\nFinal: {len(merged):,} rows x {len(merged.columns)} cols")

    merged.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")


# ── Run pipeline ─────────────────────────────────────────────────────────────


def run_pipeline(
    start_date: str = PIPELINE_START,
    end_date: str = PIPELINE_END,
    forward_days: int = 7,
    features_out: str = "data/sport_hscore/features.parquet",
    labels_out: str = "data/sport_hscore/labels.parquet",
    test_mode: bool = False,
) -> None:
    """Run the full eligibility -> features -> labels pipeline."""
    snapshots = pd.date_range(start=start_date, end=end_date, freq="D")
    if test_mode:
        snapshots = snapshots[:5]

    print(f"Snapshot range: {start_date} -> {end_date}  ({len(snapshots)} dates)")

    engine = get_engine()
    features_cache, eligibility_cache = precompute_features_and_eligibility(snapshots, engine)
    labels_cache = precompute_labels(snapshots, eligibility_cache, engine, forward_days)

    all_features = []
    all_labels = []

    for snap_ts in tqdm(snapshots, desc="Pipeline", unit="date"):
        snap_str = snap_ts.strftime("%Y-%m-%d")
        feat_df = features_cache.get(snap_str)
        label_df = labels_cache.get(snap_str)

        if feat_df is None or feat_df.empty:
            continue

        all_features.append(feat_df)
        if label_df is not None and not label_df.empty:
            all_labels.append(label_df)

    if not all_features:
        print("No data collected.")
        return

    features_df = pd.concat(all_features, ignore_index=True)
    labels_df = pd.concat(all_labels, ignore_index=True)

    features_df.to_parquet(features_out, index=False)
    labels_df.to_parquet(labels_out, index=False)

    n_pos = labels_df["label"].sum()
    n_total = len(labels_df)
    print(f"\nfeatures: {features_df.shape}")
    print(f"labels  : {labels_df.shape}")
    if n_total > 0:
        print(f"label=1 : {n_pos}/{n_total}  ({100 * n_pos / n_total:.1f}%)")
