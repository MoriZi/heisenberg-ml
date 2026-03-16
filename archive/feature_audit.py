"""
feature_audit.py

Comprehensive audit of wallet_profile_metrics columns for eligible wallets
on the latest available snapshot with calculation_window_days = 15.

Eligibility filters (matching deploy_formula.sql):
    roi > 0
    win_rate BETWEEN 0.45 AND 0.95
    total_trades BETWEEN 50 AND 100000
    total_pnl > 5000
    combined_risk_score <= 50

Verdicts:
    USABLE      — std >= 0.001 and null % < 30
    SPARSE      — null % >= 30
    CONSTANT    — std < 0.001 (after fillna(0) for numerics)
    CATEGORICAL — non-numeric / boolean

Output:
    Console table + feature_audit.csv
"""

import json
import numpy as np
import pandas as pd
from db import get_engine

# ── columns to skip (identifiers / window param) ─────────────────────────────
SKIP_COLS = {"proxy_wallet", "date", "calculation_window_days"}

# ── JSONB column — handled separately ────────────────────────────────────────
JSONB_COL = "performance_by_category"

# ── threshold constants ───────────────────────────────────────────────────────
SPARSE_NULL_PCT   = 30.0   # null % >= this → SPARSE
CONSTANT_STD      = 0.001  # std < this     → CONSTANT


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_category_pnl(jsonb_val) -> dict:
    """Return {category: pnl} from a performance_by_category JSONB value."""
    if jsonb_val is None:
        return {}
    try:
        items = json.loads(jsonb_val) if isinstance(jsonb_val, str) else jsonb_val
        return {item["category"]: float(item.get("pnl", 0)) for item in items}
    except Exception:
        return {}


def verdict(row: dict) -> str:
    """Assign USABLE / SPARSE / CONSTANT / CATEGORICAL based on audit stats."""
    if row["dtype_kind"] == "categorical":
        return "CATEGORICAL"
    if row["null_pct"] >= SPARSE_NULL_PCT:
        return "SPARSE"
    if row["std"] < CONSTANT_STD:
        return "CONSTANT"
    return "USABLE"


# ── main audit ────────────────────────────────────────────────────────────────

def run_audit() -> None:
    engine = get_engine()

    # ── 1. fetch all columns for eligible wallets on latest date ──────────────
    print("Fetching schema for wallet_profile_metrics...")
    schema_sql = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'polymarket'
          AND table_name   = 'wallet_profile_metrics'
        ORDER BY ordinal_position
    """
    schema_df = pd.read_sql(schema_sql, engine)
    all_cols  = schema_df["column_name"].tolist()

    print(f"Total columns in table: {len(all_cols)}")

    # ── 2. pull eligible rows with all columns ─────────────────────────────────
    print("Pulling eligible rows (window=15, latest date)...")
    data_sql = """
        SELECT *
        FROM polymarket.wallet_profile_metrics
        WHERE calculation_window_days = 15
          AND date = (
              SELECT MAX(date)
              FROM polymarket.wallet_profile_metrics
              WHERE calculation_window_days = 15
          )
          AND roi                 > 0
          AND win_rate            BETWEEN 0.45 AND 0.95
          AND total_trades        BETWEEN 50 AND 100000
          AND total_pnl           > 5000
          AND combined_risk_score <= 50
    """
    df = pd.read_sql(data_sql, engine)
    n  = len(df)
    print(f"Eligible rows fetched: {n:,}")
    print(f"Snapshot date: {df['date'].iloc[0] if n > 0 else 'N/A'}\n")

    if n == 0:
        print("No rows — nothing to audit.")
        return

    # ── 3. audit each column ──────────────────────────────────────────────────
    rows = []

    # Build dtype lookup from schema for display purposes
    dtype_map = dict(zip(schema_df["column_name"], schema_df["data_type"]))

    for col in all_cols:
        if col in SKIP_COLS:
            continue
        if col not in df.columns:
            continue
        if col == JSONB_COL:
            continue  # handled separately below

        series  = df[col]
        db_type = dtype_map.get(col, "unknown")

        null_count = series.isna().sum()
        null_pct   = 100.0 * null_count / n

        # ── detect categorical (object, bool, text enums) ─────────────────
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_bool    = pd.api.types.is_bool_dtype(series)

        # Treat boolean and string columns as categorical for audit purposes
        if not is_numeric or is_bool:
            # convert to str first to handle unhashable types (e.g. JSONB lists)
            unique_vals = series.dropna().apply(lambda x: str(x)).unique().tolist()
            uv_str = str(sorted(unique_vals)[:10])
            rows.append({
                "feature":    col,
                "db_type":    db_type,
                "dtype_kind": "categorical",
                "null_pct":   round(null_pct, 1),
                "zero_pct":   None,
                "min":        None,
                "max":        None,
                "median":     None,
                "std":        0.0,
                "variance":   0.0,
                "unique_vals": uv_str,
            })
            continue

        # ── numeric ──────────────────────────────────────────────────────────
        # fill NaN with 0 for std/variance calculation (same as training)
        filled = series.fillna(0).astype(float)

        zero_count = (filled == 0).sum()
        zero_pct   = 100.0 * zero_count / n

        # Stats on non-null values for min/max/median (more honest)
        non_null = series.dropna().astype(float)
        if len(non_null) > 0:
            col_min    = float(non_null.min())
            col_max    = float(non_null.max())
            col_median = float(non_null.median())
        else:
            col_min = col_max = col_median = np.nan

        col_std = float(filled.std())
        col_var = float(filled.var())

        rows.append({
            "feature":    col,
            "db_type":    db_type,
            "dtype_kind": "numeric",
            "null_pct":   round(null_pct, 1),
            "zero_pct":   round(zero_pct, 1),
            "min":        round(col_min,    4) if not np.isnan(col_min)    else None,
            "max":        round(col_max,    4) if not np.isnan(col_max)    else None,
            "median":     round(col_median, 4) if not np.isnan(col_median) else None,
            "std":        round(col_std,    6),
            "variance":   round(col_var,    6),
            "unique_vals": None,
        })

    # ── assign verdicts ───────────────────────────────────────────────────────
    audit_df = pd.DataFrame(rows)
    audit_df["verdict"] = audit_df.apply(verdict, axis=1)

    # ── 4. print full audit table ─────────────────────────────────────────────
    display_cols = ["feature", "db_type", "null_pct", "zero_pct",
                    "min", "max", "median", "std", "verdict"]
    print("=" * 100)
    print(f"  FEATURE AUDIT  —  {n:,} eligible wallets  (window=15, latest date)")
    print("=" * 100)
    print(audit_df[display_cols].to_string(index=False))

    # ── 5. counts by verdict ──────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Verdict summary")
    print(f"{'─' * 60}")
    for v, cnt in audit_df["verdict"].value_counts().items():
        print(f"  {v:<12}  {cnt:>3}")

    # ── 6. JSONB: performance_by_category ────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  JSONB: performance_by_category")
    print(f"{'─' * 60}")

    if JSONB_COL in df.columns:
        parsed   = df[JSONB_COL].apply(parse_category_pnl)
        # Collect all categories
        all_cats = set()
        for d in parsed:
            all_cats.update(d.keys())
        all_cats = sorted(all_cats)

        print(f"  Unique categories found: {all_cats}\n")
        print(f"  {'Category':<30}  {'wallets w/ non-zero pnl':>22}  {'pct':>6}  {'median_pnl':>12}")
        print(f"  {'─'*30}  {'─'*22}  {'─'*6}  {'─'*12}")

        for cat in all_cats:
            pnls        = [d.get(cat, 0.0) for d in parsed]
            pnl_series  = pd.Series(pnls, dtype=float)
            nonzero_cnt = (pnl_series != 0).sum()
            pct         = 100.0 * nonzero_cnt / n
            med_pnl     = pnl_series[pnl_series != 0].median() if nonzero_cnt > 0 else 0.0
            print(f"  {cat:<30}  {nonzero_cnt:>22,}  {pct:>5.1f}%  {med_pnl:>12,.2f}")
    else:
        print("  Column not present in fetched data.")

    # ── 7. recommended feature list ───────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  RECOMMENDED features  (USABLE, sorted by variance desc)")
    print(f"{'─' * 60}")

    usable = (
        audit_df[audit_df["verdict"] == "USABLE"]
        .sort_values("variance", ascending=False)
        .reset_index(drop=True)
    )
    print(f"  Total USABLE: {len(usable)}\n")
    print(f"  {'#':<4}  {'feature':<35}  {'std':>10}  {'null%':>6}  {'zero%':>6}")
    print(f"  {'─'*4}  {'─'*35}  {'─'*10}  {'─'*6}  {'─'*6}")
    for i, row in usable.iterrows():
        print(f"  {i+1:<4}  {row['feature']:<35}  {row['std']:>10.4f}  "
              f"{row['null_pct']:>5.1f}%  {row['zero_pct']:>5.1f}%")

    # ── 8. save to CSV ────────────────────────────────────────────────────────
    out_path = "feature_audit.csv"
    audit_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run_audit()
