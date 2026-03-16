"""
build_features_multiwindow.py

Extends features.parquet (15d window, all eligible wallets) with shorter-window
features from the 1d, 3d, and 7d calculation windows.

Strategy:
  - Start from features.parquet as the BASE (never drop rows).
  - For each window in [1, 3, 7]:
      * Query wallet_profile_metrics with calculation_window_days = W,
        filtering to the exact (proxy_wallet, date) pairs in base.
      * Parse JSONB performance_by_category for pnl_cat_sports, pnl_cat_other.
      * Rename all 16 feature columns to feature_Wd (e.g. total_pnl_1d).
      * LEFT JOIN onto base on proxy_wallet + snapshot_date.
  - Wallets with no data in shorter windows get 0 (never dropped).
  - Assert row count unchanged after all joins.
  - Save to features_multiwindow.parquet.

Usage:
    python build_features_multiwindow.py
    python build_features_multiwindow.py --base features.parquet --out features_multiwindow.parquet
"""

import argparse
import json
import pandas as pd
from db import get_connection, get_engine

# ── short-window feature columns to pull from wallet_profile_metrics ──────────

WINDOW_METRIC_COLS = [
    "total_pnl",
    "total_invested",
    "worst_trade",
    "best_trade",
    "total_trades",
    "roi",
    "win_rate",
    "avg_position_size",
    "stddev_position_size",
    "dominant_market_pnl",
    "profit_factor",
    "market_concentration_ratio",
    "perfect_entry_count",
    "statistical_confidence",
    # JSONB — parsed separately; not kept as raw column
    "performance_by_category",
]

# The 2 JSONB-derived columns we extract per window
JSONB_CATS = ["sports", "other"]

WINDOWS = [1, 3, 7]


# ── JSONB helpers (mirrors build_features.py) ─────────────────────────────────

def _key(cat: str) -> str:
    return "pnl_cat_" + cat.lower().replace(" ", "_").replace("&", "and")


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


# ── window fetcher ─────────────────────────────────────────────────────────────

def fetch_window(engine, window_days: int, wallets: list, dates: list) -> pd.DataFrame:
    """
    Pull short-window features for the given (wallet, date) universe.
    No eligibility filters — those are already baked into the base.
    Returns a DataFrame with proxy_wallet, snapshot_date, and 16 feature columns
    renamed to feature_{window_days}d.
    """
    plain_cols = [c for c in WINDOW_METRIC_COLS if c != "performance_by_category"]
    col_select = ", ".join(["proxy_wallet", "date"] + plain_cols + ["performance_by_category"])

    sql = f"""
        SELECT {col_select}
        FROM polymarket.wallet_profile_metrics
        WHERE calculation_window_days = %(window)s
          AND proxy_wallet = ANY(%(wallets)s)
          AND date = ANY(%(dates)s::date[])
    """
    params = {
        "window": window_days,
        "wallets": wallets,
        "dates": dates,
    }

    print(f"  Fetching window={window_days}d ({len(wallets):,} wallets × {len(dates)} dates)...")
    df = pd.read_sql(sql, engine, params=params)
    print(f"  Rows returned: {len(df):,}")

    if df.empty:
        return pd.DataFrame(columns=["proxy_wallet", "snapshot_date"])

    # ── parse JSONB ───────────────────────────────────────────────────────
    cat_pnl = pd.DataFrame(
        df["performance_by_category"].apply(parse_window_jsonb).tolist(),
        index=df.index,
    )
    df = pd.concat([df.drop(columns=["performance_by_category"]), cat_pnl], axis=1)

    # ── rename date → snapshot_date ───────────────────────────────────────
    df = df.rename(columns={"date": "snapshot_date"})
    df["snapshot_date"] = df["snapshot_date"].astype(str)

    # ── rename feature columns to feature_Wd ─────────────────────────────
    suffix = f"_{window_days}d"
    rename_map = {}
    for col in plain_cols + [_key(c) for c in JSONB_CATS]:
        rename_map[col] = col + suffix

    df = df.rename(columns=rename_map)
    return df


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="features.parquet",
                        help="Base features file (default: features.parquet)")
    parser.add_argument("--out",  default="features_multiwindow.parquet",
                        help="Output path (default: features_multiwindow.parquet)")
    args = parser.parse_args()

    # ── 1. load base ──────────────────────────────────────────────────────
    print(f"\nLoading base: {args.base}")
    base = pd.read_parquet(args.base)
    base["snapshot_date"] = base["snapshot_date"].astype(str)
    n_base = len(base)
    print(f"Base rows   : {n_base:,}")
    print(f"Base columns: {len(base.columns)}")

    # Wallet + date universe for DB queries
    all_wallets = base["proxy_wallet"].unique().tolist()
    all_dates   = base["snapshot_date"].unique().tolist()

    # ── 2. open DB engine (reused across all 3 window queries) ───────────
    engine = get_engine()

    # ── 3. LEFT JOIN each short window ────────────────────────────────────
    merged = base.copy()
    for w in WINDOWS:
        print(f"\n── Window {w}d ──────────────────────────────────────────")
        win_df = fetch_window(engine, w, all_wallets, all_dates)

        if win_df.empty:
            print(f"  WARNING: no data for window={w}d — all {w}d columns will be 0")
            # Add zero columns so downstream code doesn't break
            new_cols = [c + f"_{w}d" for c in
                        [c for c in WINDOW_METRIC_COLS if c != "performance_by_category"]
                        + [_key(c) for c in JSONB_CATS]]
            for col in new_cols:
                merged[col] = 0.0
            continue

        # Deduplicate (safety — should not happen, but guard against it)
        before = len(win_df)
        win_df = win_df.drop_duplicates(subset=["proxy_wallet", "snapshot_date"])
        if len(win_df) < before:
            print(f"  Dropped {before - len(win_df)} duplicate rows in window data")

        merged = merged.merge(
            win_df,
            on=["proxy_wallet", "snapshot_date"],
            how="left",
        )

        # Confirm row count preserved
        assert len(merged) == n_base, (
            f"Row count changed after {w}d join: {len(merged):,} != {n_base:,}. "
            "Check for duplicate (proxy_wallet, snapshot_date) in window data."
        )
        print(f"  Row count after join: {len(merged):,} ✓")

    engine.dispose()

    # ── 4. fillna(0) for all new short-window columns ─────────────────────
    new_cols = [
        col for col in merged.columns
        if any(col.endswith(f"_{w}d") for w in WINDOWS)
    ]
    null_before = merged[new_cols].isnull().sum().sum()
    merged[new_cols] = merged[new_cols].fillna(0.0)
    print(f"\nFilled {null_before:,} nulls in short-window columns with 0")

    # ── 4b. compute pnl_per_trade for all 4 windows ───────────────────────
    # total_pnl / total_trades; 0 where trades = 0 in that window.
    # Uses safe division: replace 0-trade denominator with NaN, then fillna(0).
    ppt_windows = [
        ("pnl_per_trade",    "total_pnl",    "total_trades"),
        ("pnl_per_trade_1d", "total_pnl_1d", "total_trades_1d"),
        ("pnl_per_trade_3d", "total_pnl_3d", "total_trades_3d"),
        ("pnl_per_trade_7d", "total_pnl_7d", "total_trades_7d"),
    ]
    for col_name, pnl_col, trades_col in ppt_windows:
        safe_trades = merged[trades_col].replace(0, float("nan"))
        merged[col_name] = (merged[pnl_col] / safe_trades).fillna(0.0)
    ppt_cols = [t[0] for t in ppt_windows]
    print(f"Computed {len(ppt_cols)} pnl_per_trade columns (15d, 1d, 3d, 7d)")

    # ── 5. final assertion ────────────────────────────────────────────────
    assert len(merged) == n_base, (
        f"Final row count {len(merged):,} != base {n_base:,}"
    )
    print(f"\nFinal row count : {len(merged):,}  ✓  (matches base)")
    print(f"Final col count : {len(merged.columns)}")

    # ── 6. diagnostics ────────────────────────────────────────────────────
    diag_cols = sorted(new_cols) + ppt_cols
    print(f"\n{'─'*70}")
    print(f"  New columns ({len(diag_cols)} total: {len(new_cols)} window joins + {len(ppt_cols)} computed)")
    print(f"  {'Column':<45}  {'Nulls':>6}  {'Zeros':>6}  {'Coverage':>9}")
    print(f"  {'─'*45}  {'─'*6}  {'─'*6}  {'─'*9}")
    for col in diag_cols:
        nulls  = merged[col].isnull().sum()
        zeros  = (merged[col] == 0.0).sum()
        pct    = 100 * (1 - zeros / n_base)
        print(f"  {col:<45}  {nulls:>6}  {zeros:>6}  {pct:>8.1f}%")

    # ── 6b. pnl_per_trade stats ───────────────────────────────────────────
    print(f"\n  pnl_per_trade summary (non-zero rows only):")
    print(f"  {'Column':<25}  {'Min':>10}  {'Median':>10}  {'Max':>10}  {'Zero%':>7}")
    print(f"  {'─'*25}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*7}")
    for col in ppt_cols:
        nz  = merged[col][merged[col] != 0]
        zp  = 100 * (merged[col] == 0).mean()
        if len(nz) == 0:
            print(f"  {col:<25}  {'all zero':>10}")
            continue
        print(f"  {col:<25}  {nz.min():>10.2f}  {nz.median():>10.2f}  "
              f"{nz.max():>10.2f}  {zp:>6.1f}%")

    # ── 7. save ───────────────────────────────────────────────────────────
    merged.to_parquet(args.out, index=False)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
