"""
Shared label construction utilities.

Builds binary ground-truth labels from wallet_daily_pnl for a given
snapshot date. The label definition is:
    label = 1  if  forward_pnl > 0  AND  forward_rank <= rank_threshold
    label = 0  otherwise
"""

import pandas as pd
from src.common.db import get_engine


def build_labels_for_date(
    snapshot_date: str,
    forward_days: int = 14,
    rank_threshold: int = 500,
    eligible_wallets: set[str] | None = None,
) -> pd.DataFrame:
    """
    Build labels for a single snapshot date.

    Parameters
    ----------
    snapshot_date : date string (YYYY-MM-DD) — the observation date T.
    forward_days : number of days in the forward window (T+1 to T+N).
    rank_threshold : wallets must be within this rank to get label=1.
    eligible_wallets : if provided, filter to these wallets in SQL.

    Returns
    -------
    DataFrame with columns: proxy_wallet, snapshot_date, forward_pnl,
                            forward_rank, label.
    """
    snap = pd.Timestamp(snapshot_date).date()
    fwd_start = pd.Timestamp(snap) + pd.Timedelta(days=1)
    fwd_end = pd.Timestamp(snap) + pd.Timedelta(days=forward_days)

    engine = get_engine()

    params: dict = {
        "fwd_start": str(fwd_start.date()),
        "fwd_end": str(fwd_end.date()),
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
    df = pd.read_sql(sql, engine, params=params)

    df["forward_rank"] = df["forward_pnl"].rank(
        ascending=False, method="min"
    ).astype(int)

    df["label"] = (
        (df["forward_pnl"] > 0) & (df["forward_rank"] <= rank_threshold)
    ).astype(int)

    df["snapshot_date"] = str(snap)
    df = df[
        ["proxy_wallet", "snapshot_date", "forward_pnl", "forward_rank", "label"]
    ].reset_index(drop=True)

    return df
