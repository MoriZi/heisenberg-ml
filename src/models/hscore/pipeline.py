"""
H-Score data pipeline.

Orchestrates: eligibility → features → multiwindow → labels.
Produces features.parquet and labels.parquet in the data directory.
"""

import json

import pandas as pd
from tqdm import tqdm

from src.common.db import get_engine
from src.models.hscore.config import (
    ELIGIBILITY_SQL_FILTERS,
    FILLNA_MEDIAN_COLS,
    JSONB_CATS,
    KNOWN_CATEGORIES,
    METRIC_COLS,
    PIPELINE_END,
    PIPELINE_START,
    RANK_THRESHOLD,
    WINDOW_METRIC_COLS,
    WINDOWS,
)


# ── JSONB helpers ────────────────────────────────────────────────────────────


def _key(cat: str) -> str:
    return "pnl_cat_" + cat.lower().replace(" ", "_").replace("&", "and")


def parse_dominant_category(jsonb_val) -> str | None:
    """Return the category with the highest PnL from performance_by_category."""
    if jsonb_val is None:
        return None
    try:
        items = json.loads(jsonb_val) if isinstance(jsonb_val, str) else jsonb_val
        if not items:
            return None
        best = max(items, key=lambda x: float(x.get("pnl", 0)))
        return best.get("category")
    except Exception:
        return None


def parse_category_pnl(jsonb_val) -> dict:
    """Return per-category PnL as flat columns."""
    result = {_key(c): 0.0 for c in KNOWN_CATEGORIES}
    if jsonb_val is None:
        return result
    try:
        items = json.loads(jsonb_val) if isinstance(jsonb_val, str) else jsonb_val
        for item in items:
            key = _key(item.get("category", ""))
            if key in result:
                result[key] = float(item.get("pnl", 0))
    except Exception:
        pass
    return result


def parse_window_jsonb(jsonb_val) -> dict:
    """Extract pnl_cat_sports and pnl_cat_other from performance_by_category."""
    result = {_key(c): 0.0 for c in JSONB_CATS}
    if jsonb_val is None:
        return result
    try:
        items = json.loads(jsonb_val) if isinstance(jsonb_val, str) else jsonb_val
        for item in items:
            cat = item.get("category", "")
            k = _key(cat)
            if k in result:
                result[k] = float(item.get("pnl", 0))
    except Exception:
        pass
    return result


# ── Eligibility ──────────────────────────────────────────────────────────────


def build_eligibility(snapshot_date: str) -> pd.DataFrame:
    """Reconstruct 30-day eligibility flags for a single snapshot date."""
    snap = pd.Timestamp(snapshot_date).date()
    window_start_30d = pd.Timestamp(snap) - pd.Timedelta(days=30)
    window_start_7d = pd.Timestamp(snap) - pd.Timedelta(days=7)

    engine = get_engine()

    sql_30d = """
        SELECT
            proxy_wallet,
            SUM(pnl)      AS total_pnl_30d,
            SUM(invested)  AS total_volume_30d,
            SUM(trades)    AS total_trades_30d,
            SUM(wins)      AS total_wins_30d,
            SUM(losses)    AS total_losses_30d
        FROM polymarket.wallet_daily_pnl
        WHERE date > %(start)s AND date <= %(snap)s
        GROUP BY proxy_wallet
    """
    df_30d = pd.read_sql(
        sql_30d, engine,
        params={"start": str(window_start_30d.date()), "snap": str(snap)},
    )

    sql_7d = """
        SELECT proxy_wallet, SUM(pnl) AS pnl_7d, SUM(invested) AS volume_7d
        FROM polymarket.wallet_daily_pnl
        WHERE date > %(start)s AND date <= %(snap)s
        GROUP BY proxy_wallet
    """
    df_7d = pd.read_sql(
        sql_7d, engine,
        params={"start": str(window_start_7d.date()), "snap": str(snap)},
    )

    sql_risk = """
        SELECT proxy_wallet, combined_risk_score
        FROM polymarket.wallet_profile_metrics
        WHERE date = %(snap)s AND calculation_window_days = 15
    """
    df_risk = pd.read_sql(sql_risk, engine, params={"snap": str(snap)})

    df = df_30d.merge(df_7d, on="proxy_wallet", how="left")
    df = df.merge(df_risk, on="proxy_wallet", how="left")

    total_decided = df["total_wins_30d"] + df["total_losses_30d"]
    df["win_rate_30d"] = df["total_wins_30d"] / total_decided.replace(0, float("nan"))
    df["roi_30d"] = df["total_pnl_30d"] / df["total_volume_30d"].replace(0, float("nan"))
    df["roi_7d"] = df["pnl_7d"] / df["volume_7d"].replace(0, float("nan"))

    df["trajectory"] = df.apply(
        lambda r: "Decaying"
        if (r.get("roi_7d") or 0) < (r.get("roi_30d") or 0)
        else "Stable",
        axis=1,
    )

    df["eligible"] = (
        (df["roi_30d"] > 0)
        & df["win_rate_30d"].between(0.45, 0.95)
        & df["total_trades_30d"].between(50, 100_000)
        & (df["total_volume_30d"] > 10_000)
        & (df["total_pnl_30d"] > 5_000)
        & (df["trajectory"] != "Decaying")
        & (df["combined_risk_score"].fillna(0) <= 50)
    )

    df["snapshot_date"] = str(snap)
    return df


# ── Feature building ─────────────────────────────────────────────────────────


def build_features(
    snapshot_date: str, eligible_wallets: set | None = None
) -> pd.DataFrame:
    """Pull 15d features from wallet_profile_metrics for eligible wallets."""
    snap = pd.Timestamp(snapshot_date).date()
    engine = get_engine()

    if eligible_wallets is None:
        elig_df = build_eligibility(snapshot_date)
        eligible_wallets = set(elig_df.loc[elig_df["eligible"], "proxy_wallet"])

    col_select = ", ".join(METRIC_COLS)
    sql = f"""
        SELECT {col_select}
        FROM polymarket.wallet_profile_metrics
        WHERE date = %(snap)s
          AND calculation_window_days = 15
          AND proxy_wallet = ANY(%(wallets)s)
    """
    df = pd.read_sql(
        sql, engine, params={"snap": str(snap), "wallets": list(eligible_wallets)}
    )

    if df.empty:
        return df.copy()

    df["dominant_category"] = df["performance_by_category"].apply(parse_dominant_category)
    cat_pnl_df = pd.DataFrame(
        df["performance_by_category"].apply(parse_category_pnl).tolist(),
        index=df.index,
    )
    df = pd.concat([df, cat_pnl_df], axis=1)
    df = df.drop(columns=["performance_by_category"])

    for col in FILLNA_MEDIAN_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df["snapshot_date"] = str(snap)
    df = df.reset_index(drop=True)
    return df


# ── Multiwindow extension ───────────────────────────────────────────────────


def fetch_window(engine, window_days: int, wallets: list, dates: list) -> pd.DataFrame:
    """Pull short-window features for the given (wallet, date) universe."""
    plain_cols = [c for c in WINDOW_METRIC_COLS if c != "performance_by_category"]
    col_select = ", ".join(["proxy_wallet", "date"] + plain_cols + ["performance_by_category"])

    sql = f"""
        SELECT {col_select}
        FROM polymarket.wallet_profile_metrics
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

    cat_pnl = pd.DataFrame(
        df["performance_by_category"].apply(parse_window_jsonb).tolist(),
        index=df.index,
    )
    df = pd.concat([df.drop(columns=["performance_by_category"]), cat_pnl], axis=1)
    df = df.rename(columns={"date": "snapshot_date"})
    df["snapshot_date"] = df["snapshot_date"].astype(str)

    suffix = f"_{window_days}d"
    rename_map = {
        col: col + suffix
        for col in plain_cols + [_key(c) for c in JSONB_CATS]
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
            new_cols = [c + f"_{w}d" for c in plain_cols + [_key(c) for c in JSONB_CATS]]
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

    # Compute pnl_per_trade for all windows
    ppt_windows = [
        ("pnl_per_trade", "total_pnl", "total_trades"),
        ("pnl_per_trade_1d", "total_pnl_1d", "total_trades_1d"),
        ("pnl_per_trade_3d", "total_pnl_3d", "total_trades_3d"),
        ("pnl_per_trade_7d", "total_pnl_7d", "total_trades_7d"),
    ]
    for col_name, pnl_col, trades_col in ppt_windows:
        safe_trades = merged[trades_col].replace(0, float("nan"))
        merged[col_name] = (merged[pnl_col] / safe_trades).fillna(0.0)

    assert len(merged) == n_base
    print(f"\nFinal: {len(merged):,} rows x {len(merged.columns)} cols")

    merged.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")


# ── Bulk pipeline ────────────────────────────────────────────────────────────


def precompute_features_and_eligibility(snapshots, engine):
    """Single query for features + eligibility across all dates."""
    col_select = "date, " + ", ".join(METRIC_COLS)

    sql = f"""
        SELECT {col_select}
        FROM polymarket.wallet_profile_metrics
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
    print(f"  Eligible rows loaded: {len(df):,}")

    df["dominant_category"] = df["performance_by_category"].apply(parse_dominant_category)
    cat_pnl_df = pd.DataFrame(
        df["performance_by_category"].apply(parse_category_pnl).tolist(),
        index=df.index,
    )
    df = pd.concat([df, cat_pnl_df], axis=1)
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

    Only scans wallet_daily_pnl for eligible wallets (from eligibility_cache),
    avoiding a full table scan of the 41GB+ table.
    """
    start = snapshots[0].strftime("%Y-%m-%d")
    end = snapshots[-1].strftime("%Y-%m-%d")
    fwd_start = (snapshots[0] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fwd_end = (snapshots[-1] + pd.Timedelta(days=forward_days)).strftime("%Y-%m-%d")

    # Collect all eligible wallets across all snapshots
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


def run_pipeline(
    start_date: str = PIPELINE_START,
    end_date: str = PIPELINE_END,
    forward_days: int = 7,
    features_out: str = "data/hscore/features.parquet",
    labels_out: str = "data/hscore/labels.parquet",
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
    print(f"label=1 : {n_pos}/{n_total}  ({100 * n_pos / n_total:.1f}%)")
