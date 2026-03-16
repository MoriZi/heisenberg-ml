"""
build_eligibility.py

Reconstructs 30-day eligibility flags per wallet for a single snapshot date.

Usage:
    python build_eligibility.py 2026-01-15

Eligibility criteria (all must pass):
    roi_30d          > 0
    win_rate_30d     BETWEEN 0.45 AND 0.95
    total_trades_30d BETWEEN 50 AND 100000
    total_volume_30d > 10000
    total_pnl_30d    > 5000
    trajectory       != 'Decaying'
    combined_risk_score <= 50   (from wallet_profile_metrics, window=15)
"""

import sys
import pandas as pd
from db import get_engine

# ── helpers ─────────────────────────────────────────────────────────────────

def infer_trajectory(roi_7d: float, roi_30d: float) -> str:
    """
    Approximate trajectory from wallet_daily_pnl aggregates.
    We reconstruct 7d and 30d ROI; 'Decaying' means the short window
    underperforms the long window (proxy for rank_1d > rank_7d > rank_30d).
    """
    if roi_7d < roi_30d:
        return "Decaying"
    return "Stable"


# ── main ─────────────────────────────────────────────────────────────────────

def build_eligibility(snapshot_date: str) -> pd.DataFrame:
    snap = pd.Timestamp(snapshot_date).date()
    window_start_30d = pd.Timestamp(snap) - pd.Timedelta(days=30)
    window_start_7d  = pd.Timestamp(snap) - pd.Timedelta(days=7)

    engine = get_engine()

    # ── 1. 30d aggregates from wallet_daily_pnl ──────────────────────────
    sql_30d = """
        SELECT
            proxy_wallet,
            SUM(pnl)                                        AS total_pnl_30d,
            SUM(invested)                                   AS total_volume_30d,
            SUM(trades)                                     AS total_trades_30d,
            SUM(wins)                                       AS total_wins_30d,
            SUM(losses)                                     AS total_losses_30d
        FROM polymarket.wallet_daily_pnl
        WHERE date > %(start)s
          AND date <= %(snap)s
        GROUP BY proxy_wallet
    """
    df_30d = pd.read_sql(
        sql_30d,
        engine,
        params={"start": str(window_start_30d.date()), "snap": str(snap)},
    )

    # ── 2. 7d aggregates (for trajectory proxy) ───────────────────────────
    sql_7d = """
        SELECT
            proxy_wallet,
            SUM(pnl)      AS pnl_7d,
            SUM(invested) AS volume_7d
        FROM polymarket.wallet_daily_pnl
        WHERE date > %(start)s
          AND date <= %(snap)s
        GROUP BY proxy_wallet
    """
    df_7d = pd.read_sql(
        sql_7d,
        engine,
        params={"start": str(window_start_7d.date()), "snap": str(snap)},
    )

    # ── 3. combined_risk_score from wallet_profile_metrics (window=15) ────
    sql_risk = """
        SELECT
            proxy_wallet,
            combined_risk_score
        FROM polymarket.wallet_profile_metrics
        WHERE date = %(snap)s
          AND calculation_window_days = 15
    """
    df_risk = pd.read_sql(sql_risk, engine, params={"snap": str(snap)})

    # ── 4. merge ──────────────────────────────────────────────────────────
    df = df_30d.merge(df_7d, on="proxy_wallet", how="left")
    df = df.merge(df_risk, on="proxy_wallet", how="left")

    # ── 5. derived metrics ────────────────────────────────────────────────
    total_decided = df["total_wins_30d"] + df["total_losses_30d"]
    df["win_rate_30d"] = df["total_wins_30d"] / total_decided.replace(0, float("nan"))

    df["roi_30d"] = df["total_pnl_30d"] / df["total_volume_30d"].replace(0, float("nan"))
    df["roi_7d"]  = df["pnl_7d"]        / df["volume_7d"].replace(0, float("nan"))

    df["trajectory"] = df.apply(
        lambda r: infer_trajectory(
            r["roi_7d"]  if pd.notna(r["roi_7d"])  else 0,
            r["roi_30d"] if pd.notna(r["roi_30d"]) else 0,
        ),
        axis=1,
    )

    # ── 6. eligibility flags ──────────────────────────────────────────────
    df["elig_roi"]        = df["roi_30d"] > 0
    df["elig_win_rate"]   = df["win_rate_30d"].between(0.45, 0.95)
    df["elig_trades"]     = df["total_trades_30d"].between(50, 100_000)
    df["elig_volume"]     = df["total_volume_30d"] > 10_000
    df["elig_pnl"]        = df["total_pnl_30d"] > 5_000
    df["elig_trajectory"] = df["trajectory"] != "Decaying"
    df["elig_risk"]       = df["combined_risk_score"].fillna(0) <= 50

    df["eligible"] = (
        df["elig_roi"]
        & df["elig_win_rate"]
        & df["elig_trades"]
        & df["elig_volume"]
        & df["elig_pnl"]
        & df["elig_trajectory"]
        & df["elig_risk"]
    )

    df["snapshot_date"] = str(snap)
    return df


def main():
    if len(sys.argv) != 2:
        print("Usage: python build_eligibility.py YYYY-MM-DD")
        sys.exit(1)

    snapshot_date = sys.argv[1]
    print(f"Building eligibility for snapshot: {snapshot_date}")

    df = build_eligibility(snapshot_date)
    eligible = df[df["eligible"]].reset_index(drop=True)

    print(f"\nEligible wallets: {len(eligible)}")
    print(eligible[["proxy_wallet", "roi_30d", "win_rate_30d",
                     "total_pnl_30d", "total_trades_30d", "trajectory"]].head())

    # ── validation: cf11 wallet ───────────────────────────────────────────
    cf11 = "0xcf119e969f31de9653a58cb3dc213b485cd48399"
    row = df[df["proxy_wallet"] == cf11]
    if row.empty:
        print(f"\n[VALIDATION] cf11 wallet not found in raw data.")
    else:
        r = row.iloc[0]
        status = "ELIGIBLE" if r["eligible"] else "NOT ELIGIBLE"
        print(f"\n[VALIDATION] cf11 wallet — {status}")
        print(f"  roi_30d      = {r['roi_30d']:.4f}  (expect ~0.278)")
        print(f"  win_rate_30d = {r['win_rate_30d']:.4f}  (expect ~0.736)")
        print(f"  total_pnl    = {r['total_pnl_30d']:,.2f}")
        print(f"  trajectory   = {r['trajectory']}")
        print(f"  risk_score   = {r['combined_risk_score']}")
        for flag in ["elig_roi","elig_win_rate","elig_trades",
                     "elig_volume","elig_pnl","elig_trajectory","elig_risk"]:
            print(f"  {flag:22s} = {r[flag]}")


if __name__ == "__main__":
    main()
