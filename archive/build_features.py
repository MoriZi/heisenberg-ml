"""
build_features.py

Pulls 15d features from wallet_profile_metrics for all eligible wallets
on a given snapshot date, parses JSONB, handles nulls, saves to parquet.

Usage:
    python build_features.py 2026-01-15

Output:
    features_YYYY-MM-DD.parquet
"""

import sys
import json
import pandas as pd
from db import get_engine
from build_eligibility import build_eligibility

# ── columns to select from wallet_profile_metrics ────────────────────────────
METRIC_COLS = [
    # identifiers
    "proxy_wallet",
    # core performance
    "roi",
    "win_rate",
    "total_pnl",
    "total_invested",
    "total_trades",
    "markets_traded",
    # risk-adjusted returns
    # sharpe_ratio excluded — 90.9% null in practice, unusable as a feature
    # edge_decay excluded — constant 0.0 across all rows; zero variance
    # trade_timing_correlation_max excluded — constant 0.0; zero variance
    "sortino_ratio",        # sparse (54.7% null) → fillna(median)
    "calmar_ratio",         # sparse (55.0% null) → fillna(median)
    "gain_to_pain_ratio",   # sparse (52.4% null) → fillna(median)
    "annualized_return",    # sparse (52.4% null) → fillna(median)
    # drawdown / risk
    "max_drawdown",
    "ulcer_index",
    "drawdown_frequency",
    "recovery_time_avg",
    "profit_factor",
    # trend / curve
    "performance_trend",
    "curve_smoothness",
    "equity_curve_pattern",
    # sizing / concentration
    # position_size_consistency excluded — constant 0.0; zero variance
    # trade_size_stdev excluded — constant 0.0; zero variance
    # max_position_pct excluded — constant 0.0; zero variance
    "avg_position_size",
    "stddev_position_size",      # replaces trade_size_stdev; fully populated
    "coefficient_of_variation",  # scale-normalised position size volatility
    "dominant_market_pnl",       # pnl from single most-traded market
    "market_concentration_ratio",
    "num_markets_traded",
    # diversity
    "category_diversity_score",
    # activity
    "days_active",
    "best_trade",
    "worst_trade",
    "win_rate_last_30d",
    "win_rate_z_score",
    # timing / anomaly
    "timing_hit_rate",
    "timing_z_score",
    "timing_anomaly_flag",
    "perfect_timing_score",
    "perfect_entry_count",
    "perfect_exit_count",
    "statistical_confidence",
    # risk flags
    "combined_risk_score",
    "risk_level",
    "suspicious_win_rate_flag",
    "single_market_dependence_flag",
    "sybil_risk_score",
    "sybil_risk_flag",
    # JSONB — parsed in Python, not kept as raw column
    "performance_by_category",
]

# Sparse ratio columns — imputed with per-snapshot median (not 0).
# Rationale: these ratios are undefined when the denominator condition is absent
# (no drawdown, no losing trades). Median is less distorting than 0 for a
# feature where ~50% of values are missing.
FILLNA_MEDIAN_COLS = [
    "sortino_ratio",
    "calmar_ratio",
    "gain_to_pain_ratio",
    "annualized_return",
]

# Previously in FILLNA_ZERO_COLS — now empty; all sparse cols use median.
# Kept for backward-compatibility with run_pipeline.py import.
FILLNA_ZERO_COLS: list = []

KNOWN_CATEGORIES = [
    "Sports", "Crypto", "World Events", "Economics",
    "Science & Tech", "Politics", "Entertainment", "Weather", "Other",
]


# ── JSONB helpers ─────────────────────────────────────────────────────────────

def parse_dominant_category(jsonb_val) -> str | None:
    """Return the category with the highest PnL from performance_by_category JSONB."""
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
    """
    Return per-category PnL as flat columns.
    Keys: pnl_cat_sports, pnl_cat_crypto, pnl_cat_world_events,
          pnl_cat_economics, pnl_cat_science_and_tech, pnl_cat_other
    Missing categories default to 0.
    """
    def _key(cat: str) -> str:
        return "pnl_cat_" + cat.lower().replace(" ", "_").replace("&", "and")

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


# ── main ─────────────────────────────────────────────────────────────────────

def build_features(snapshot_date: str, eligible_wallets: set | None = None) -> pd.DataFrame:
    snap = pd.Timestamp(snapshot_date).date()
    engine = get_engine()

    # ── 1. eligible wallets ───────────────────────────────────────────────
    # Accept a pre-computed set from run_pipeline.py (avoids a redundant
    # build_eligibility call when the pipeline drives the loop).
    if eligible_wallets is None:
        print(f"  Computing eligibility for {snap}...")
        elig_df = build_eligibility(snapshot_date)
        eligible_wallets = set(elig_df.loc[elig_df["eligible"], "proxy_wallet"])
    print(f"  Eligible wallets: {len(eligible_wallets)}")

    # ── 2. fetch wallet_profile_metrics filtered in SQL ───────────────────
    # Pass the eligible wallet list directly into the WHERE clause so the
    # database returns only the rows we need — avoids fetching the full
    # daily snapshot and filtering in Python (~43s → ~2-3s per date).
    col_select = ", ".join(METRIC_COLS)
    sql = f"""
        SELECT {col_select}
        FROM polymarket.wallet_profile_metrics
        WHERE date = %(snap)s
          AND calculation_window_days = 15
          AND proxy_wallet = ANY(%(wallets)s)
    """
    print(f"  Fetching wallet_profile_metrics (window=15) for {snap}...")
    df = pd.read_sql(sql, engine, params={"snap": str(snap), "wallets": list(eligible_wallets)})
    print(f"  Rows from metrics table: {len(df)}")

    if df.empty:
        print("  WARNING: no feature rows — check that wallet_profile_metrics "
              "has data for this date with calculation_window_days=15.")
        return df.copy()

    # ── 4. parse performance_by_category JSONB ────────────────────────────
    df["dominant_category"] = df["performance_by_category"].apply(
        parse_dominant_category
    )
    cat_pnl_df = pd.DataFrame(
        df["performance_by_category"].apply(parse_category_pnl).tolist(),
        index=df.index,
    )
    df = pd.concat([df, cat_pnl_df], axis=1)
    df = df.drop(columns=["performance_by_category"])

    # ── 5. null handling ──────────────────────────────────────────────────
    # Sparse ratio columns: impute with median of non-null values in this
    # snapshot — less distorting than 0 when ~50% of rows are missing.
    for col in FILLNA_MEDIAN_COLS:
        if col in df.columns:
            median_val = df[col].median()   # pandas median() ignores NaN
            df[col] = df[col].fillna(median_val)

    # ── 6. metadata ───────────────────────────────────────────────────────
    df["snapshot_date"] = str(snap)
    df = df.reset_index(drop=True)

    return df


def main():
    if len(sys.argv) != 2:
        print("Usage: python build_features.py YYYY-MM-DD")
        sys.exit(1)

    snapshot_date = sys.argv[1]
    print(f"\nBuilding features for snapshot: {snapshot_date}")

    df = build_features(snapshot_date)

    if df.empty:
        print("No data — nothing saved.")
        sys.exit(0)

    # ── summary ───────────────────────────────────────────────────────────
    print(f"\nShape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  {col}")

    # ── null check ────────────────────────────────────────────────────────
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if cols_with_nulls.empty:
        print("\nNull check: PASSED — no nulls")
    else:
        print(f"\nNull check: {len(cols_with_nulls)} columns still have nulls:")
        print(cols_with_nulls.to_string())

    # ── cf11 validation ───────────────────────────────────────────────────
    cf11 = "0xcf119e969f31de9653a58cb3dc213b485cd48399"
    row = df[df["proxy_wallet"] == cf11]
    if row.empty:
        print(f"\n[VALIDATION] cf11 not found in features for this date.")
    else:
        r = row.iloc[0]
        print(f"\n[VALIDATION] cf11 stats:")
        print(f"  roi                 = {r['roi']}")
        print(f"  win_rate            = {r['win_rate']}")
        print(f"  sortino_ratio       = {r['sortino_ratio']}")
        print(f"  calmar_ratio        = {r['calmar_ratio']}")
        print(f"  total_trades        = {r['total_trades']}")
        print(f"  markets_traded      = {r['markets_traded']}")
        print(f"  dominant_category   = {r['dominant_category']}")
        print(f"  combined_risk_score = {r['combined_risk_score']}")
        print(f"  performance_trend   = {r['performance_trend']}")

    # ── save ──────────────────────────────────────────────────────────────
    out_path = f"features_{snapshot_date}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
