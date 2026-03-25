"""
Sport H-Score deployment utilities.

Runs the production SQL formula against the live DB, saves scored
wallet CSVs, and prints a summary leaderboard.
"""

import re
import shutil
from pathlib import Path

import pandas as pd

from src.common.db import get_connection
from src.models.sport_hscore.config import TABLE, TIERS


TIER_ORDER = [t[0] for t in TIERS]

OUTPUT_COLS = [
    "rank",
    "proxy_wallet",
    "sport_h_score",
    "tier",
    "total_invested",
    "total_pnl",
    "win_rate",
    "worst_market_pnl",
    "total_trades",
    "markets_traded",
    "sports_pnl",
]


def load_sql_no_limit(path: str) -> str:
    """Read SQL file and strip any trailing LIMIT clause."""
    with open(path) as f:
        sql = f.read()
    return re.sub(
        r"\bLIMIT\s+\d+\s*;?\s*$", "", sql.strip(), flags=re.IGNORECASE
    )


def get_snapshot_date(conn) -> str:
    """Fetch the latest date with calculation_window_days=15 from v2."""
    row = pd.read_sql(
        f"SELECT MAX(date)::text AS snap "
        f"FROM {TABLE} "
        f"WHERE calculation_window_days = 15",
        conn,
    ).iloc[0]
    return row["snap"]


def print_summary(df: pd.DataFrame, snap_date: str) -> None:
    n = len(df)

    print(f"\n{'=' * 58}")
    print(f"  Sport H-Score leaderboard — snapshot {snap_date}")
    print(f"  Total eligible wallets scored: {n:,}")
    print(f"{'=' * 58}")

    print(f"\n  Tier distribution")
    print(f"  {'─' * 40}")
    counts = df["tier"].value_counts().reindex(TIER_ORDER, fill_value=0)
    for tier, cnt in counts.items():
        bar = "█" * int(cnt / n * 40)
        print(f"  {tier:<10}  {cnt:>4,}  ({100 * cnt / n:>4.1f}%)  {bar}")

    print(f"\n  sport_h_score percentiles")
    print(f"  {'─' * 30}")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = df["sport_h_score"].quantile(p / 100)
        print(f"  p{p:<4}  {val:>7.3f}")

    print(f"\n  Top 20 wallets")
    print(f"  {'─' * 110}")
    top20 = df.head(20)[OUTPUT_COLS].copy()
    top20["proxy_wallet"] = top20["proxy_wallet"].str[:12] + "…"
    top20["win_rate"] = top20["win_rate"].map(lambda x: f"{x:.1%}")
    top20["total_pnl"] = top20["total_pnl"].map(lambda x: f"${x:>12,.0f}")
    top20["worst_market_pnl"] = top20["worst_market_pnl"].map(lambda x: f"${x:>10,.0f}")
    top20["total_invested"] = top20["total_invested"].map(lambda x: f"${x:>12,.0f}")
    top20["sports_pnl"] = top20["sports_pnl"].map(lambda x: f"${x:>10,.0f}")
    print(top20.to_string(index=False))


def score_and_save(sql_path: str, output_dir: str = "data/sport_hscore") -> None:
    """Run the deploy formula and save scored wallets."""
    conn = get_connection()

    snap_date = get_snapshot_date(conn)
    print(f"Snapshot date : {snap_date}")
    print(f"Formula file  : {sql_path}")

    sql = load_sql_no_limit(sql_path)
    print("Running formula against live DB...")
    df = pd.read_sql(sql, conn)
    conn.close()

    print(f"Rows returned : {len(df):,}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dated_path = out_dir / f"scored_wallets_{snap_date}.csv"
    latest_path = out_dir / "scored_wallets_latest.csv"

    df.to_csv(dated_path, index=False)
    shutil.copy(dated_path, latest_path)

    print(f"Saved         : {dated_path}")
    print(f"Saved         : {latest_path}")

    print_summary(df, snap_date)
