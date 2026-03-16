"""
score_and_save.py

Daily scoring script — runs the H-Score formula against the live DB,
saves full results to CSV, and prints a summary leaderboard.

Usage:
    python score_and_save.py                              # uses deploy_formula_v2.sql
    python score_and_save.py --sql deploy_formula_v2.sql  # explicit formula file

Outputs (in working directory):
    scored_wallets_YYYY-MM-DD.csv   — dated snapshot (never overwritten)
    scored_wallets_latest.csv       — always overwritten; easy reference

The snapshot date in the filename comes from the DB (MAX date with
calculation_window_days=15), not the system clock — so the file is
always named after the data it contains.

Future extensions (not yet implemented):
    - scored_wallets_history.parquet  — append each day for trend tracking
    - alert when a known leader wallet re-enters the eligible pool
"""

import argparse
import re
import shutil
import pandas as pd
from db import get_connection

# ── config ────────────────────────────────────────────────────────────────────
DEFAULT_SQL = "deploy_formula_v2.sql"

TIER_ORDER = ["Elite", "Sharp", "Solid", "Emerging"]

OUTPUT_COLS = [
    "rank", "proxy_wallet", "h_score", "tier",
    "total_invested", "total_pnl", "win_rate",
    "worst_trade", "total_trades", "markets_traded",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_sql_no_limit(path: str) -> str:
    """Read SQL file and strip any trailing LIMIT clause."""
    with open(path) as f:
        sql = f.read()
    # Remove LIMIT N (with optional semicolon / trailing whitespace)
    return re.sub(r"\bLIMIT\s+\d+\s*;?\s*$", "", sql.strip(), flags=re.IGNORECASE)


def get_snapshot_date(conn) -> str:
    """Fetch the latest date with calculation_window_days=15 from the DB."""
    row = pd.read_sql(
        "SELECT MAX(date)::text AS snap "
        "FROM polymarket.wallet_profile_metrics "
        "WHERE calculation_window_days = 15",
        conn,
    ).iloc[0]
    return row["snap"]


def print_summary(df: pd.DataFrame, snap_date: str) -> None:
    n = len(df)

    print(f"\n{'=' * 58}")
    print(f"  H-Score leaderboard — snapshot {snap_date}")
    print(f"  Total eligible wallets scored: {n:,}")
    print(f"{'=' * 58}")

    # ── tier distribution ─────────────────────────────────────────────────
    print(f"\n  Tier distribution")
    print(f"  {'─' * 40}")
    counts = df["tier"].value_counts().reindex(TIER_ORDER, fill_value=0)
    for tier, cnt in counts.items():
        bar = "█" * int(cnt / n * 40)
        print(f"  {tier:<10}  {cnt:>4,}  ({100*cnt/n:>4.1f}%)  {bar}")

    # ── h_score percentiles ───────────────────────────────────────────────
    print(f"\n  h_score percentiles")
    print(f"  {'─' * 30}")
    pctls = [10, 25, 50, 75, 90, 95, 99]
    for p in pctls:
        val = df["h_score"].quantile(p / 100)
        print(f"  p{p:<4}  {val:>7.3f}")
    print(f"  min    {df['h_score'].min():>7.3f}")
    print(f"  max    {df['h_score'].max():>7.3f}")

    # ── top 20 ────────────────────────────────────────────────────────────
    print(f"\n  Top 20 wallets")
    print(f"  {'─' * 100}")
    top20 = df.head(20)[OUTPUT_COLS].copy()
    top20["proxy_wallet"] = top20["proxy_wallet"].str[:12] + "…"
    top20["win_rate"]     = top20["win_rate"].map(lambda x: f"{x:.1%}")
    top20["total_pnl"]    = top20["total_pnl"].map(lambda x: f"${x:>12,.0f}")
    top20["worst_trade"]  = top20["worst_trade"].map(lambda x: f"${x:>10,.0f}")
    top20["total_invested"] = top20["total_invested"].map(lambda x: f"${x:>12,.0f}")
    print(top20.to_string(index=False))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sql", default=DEFAULT_SQL,
        help=f"Path to the formula SQL file (default: {DEFAULT_SQL})",
    )
    args = parser.parse_args()

    conn = get_connection()

    # ── snapshot date (from DB, not system clock) ─────────────────────────
    snap_date = get_snapshot_date(conn)
    print(f"Snapshot date : {snap_date}")
    print(f"Formula file  : {args.sql}")

    # ── run formula (full pool, no LIMIT) ─────────────────────────────────
    sql = load_sql_no_limit(args.sql)
    print("Running formula against live DB...")
    df = pd.read_sql(sql, conn)
    conn.close()

    print(f"Rows returned : {len(df):,}")

    # ── save CSV ──────────────────────────────────────────────────────────
    dated_path  = f"scored_wallets_{snap_date}.csv"
    latest_path = "scored_wallets_latest.csv"

    df.to_csv(dated_path,  index=False)
    shutil.copy(dated_path, latest_path)

    print(f"Saved         : {dated_path}")
    print(f"Saved         : {latest_path}  (copy of above)")

    # ── summary ───────────────────────────────────────────────────────────
    print_summary(df, snap_date)


if __name__ == "__main__":
    main()
