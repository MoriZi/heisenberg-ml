"""
build_labels.py

Constructs binary ground-truth labels for a given snapshot date T.

Forward window: T+1 to T+14 (inclusive) from wallet_daily_pnl.

Label definition (per CONTEXT.md):
    label = 1  if  forward_pnl > 0  AND  forward_rank <= 500
    label = 0  otherwise

forward_rank is reconstructed by ranking ALL active wallets by their
forward_pnl descending — not pulled from the leaderboard (materialized
view only reflects current state, not historical snapshots).

Settlement rows (trades=0, invested=0, pnl≠0) are INCLUDED per spec.

Usage:
    python build_labels.py 2026-01-15

Output:
    labels_YYYY-MM-DD.parquet   — columns: proxy_wallet, snapshot_date,
                                            forward_pnl, forward_rank, label
"""

import sys
import pandas as pd
from db import get_engine

# Rank threshold for positive label
RANK_THRESHOLD = 500


def build_labels(snapshot_date: str, eligible_wallets: set | None = None) -> pd.DataFrame:
    snap = pd.Timestamp(snapshot_date).date()
    fwd_start = pd.Timestamp(snap) + pd.Timedelta(days=1)   # T+1
    fwd_end   = pd.Timestamp(snap) + pd.Timedelta(days=14)  # T+14

    engine = get_engine()

    # ── 1. aggregate forward PnL per wallet from wallet_daily_pnl ────────
    # Include ALL rows in the forward window — settlement rows (pnl≠0,
    # trades=0, invested=0) represent resolved market payouts and are valid.
    # When eligible_wallets is provided, filter in SQL to avoid scanning
    # the full 41GB table and returning rows we'll discard anyway.
    params = {
        "fwd_start": str(fwd_start.date()),
        "fwd_end":   str(fwd_end.date()),
    }
    wallet_filter = ""
    if eligible_wallets is not None:
        wallet_filter = "AND proxy_wallet = ANY(%(wallets)s)"
        params["wallets"] = list(eligible_wallets)

    sql = f"""
        SELECT
            proxy_wallet,
            SUM(pnl)    AS forward_pnl,
            SUM(trades) AS forward_trades
        FROM polymarket.wallet_daily_pnl
        WHERE date >= %(fwd_start)s
          AND date <= %(fwd_end)s
          {wallet_filter}
        GROUP BY proxy_wallet
    """
    print(f"  Querying forward window {fwd_start.date()} → {fwd_end.date()}...")
    df = pd.read_sql(sql, engine, params=params)
    print(f"  Wallets with forward activity: {len(df)}")

    # ── 2. reconstruct forward rank ───────────────────────────────────────
    # Rank all wallets globally by forward_pnl descending.
    # method='min' means ties share the lowest rank (conservative).
    df["forward_rank"] = df["forward_pnl"].rank(
        ascending=False, method="min"
    ).astype(int)

    # ── 3. assign binary label ────────────────────────────────────────────
    df["label"] = (
        (df["forward_pnl"] > 0) & (df["forward_rank"] <= RANK_THRESHOLD)
    ).astype(int)

    # ── 4. metadata ───────────────────────────────────────────────────────
    df["snapshot_date"] = str(snap)

    # Keep only what downstream scripts need
    df = df[["proxy_wallet", "snapshot_date", "forward_pnl",
             "forward_rank", "label"]].reset_index(drop=True)

    return df


def main():
    if len(sys.argv) != 2:
        print("Usage: python build_labels.py YYYY-MM-DD")
        sys.exit(1)

    snapshot_date = sys.argv[1]
    print(f"\nBuilding labels for snapshot: {snapshot_date}")

    df = build_labels(snapshot_date)

    # ── summary ───────────────────────────────────────────────────────────
    n_pos   = df["label"].sum()
    n_total = len(df)
    pos_pct = 100 * n_pos / n_total if n_total > 0 else 0

    print(f"\nTotal wallets with forward activity: {n_total}")
    print(f"Positive labels (label=1):           {n_pos}  ({pos_pct:.1f}%)")
    print(f"Negative labels (label=0):           {n_total - n_pos}")
    print(f"\n(Expected positive rate: ~20-30%)")

    # ── distribution check ────────────────────────────────────────────────
    print(f"\nForward PnL distribution (label=1 wallets):")
    pos = df[df["label"] == 1]["forward_pnl"]
    if not pos.empty:
        print(f"  min    = {pos.min():,.2f}")
        print(f"  median = {pos.median():,.2f}")
        print(f"  max    = {pos.max():,.2f}")

    # ── cf11 validation ───────────────────────────────────────────────────
    cf11 = "0xcf119e969f31de9653a58cb3dc213b485cd48399"
    row = df[df["proxy_wallet"] == cf11]
    if row.empty:
        print(f"\n[VALIDATION] cf11 not found — no forward activity for this window.")
    else:
        r = row.iloc[0]
        print(f"\n[VALIDATION] cf11:")
        print(f"  forward_pnl  = {r['forward_pnl']:,.2f}")
        print(f"  forward_rank = {r['forward_rank']}")
        print(f"  label        = {r['label']}")

    # ── save ──────────────────────────────────────────────────────────────
    out_path = f"labels_{snapshot_date}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
