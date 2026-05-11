"""
Shared utilities for the BTC signal pipeline.

Market fetching, outcome resolution, and slug parsing — used by
features, backtest, and monitor.
"""

import json
import re
from datetime import datetime, timezone

import pandas as pd


# ── Slug parsing ────────────────────────────────────────────────────────────


def parse_slug_timestamp(slug: str) -> int | None:
    """Extract the unix timestamp from a btc-updown slug."""
    m = re.search(r"-(\d{10,})$", slug)
    return int(m.group(1)) if m else None


def slug_to_window_start(slug: str) -> datetime | None:
    """Convert a slug to the UTC datetime when the BTC price window opens."""
    ts = parse_slug_timestamp(slug)
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc)


# ── Outcome resolution ──────────────────────────────────────────────────────


def resolve_winner(row: pd.Series) -> str | None:
    """Determine market winner from outcome_prices.

    Settled markets have prices at exactly [1,0] or [0,1].
    """
    prices = row["outcome_prices"]
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except (json.JSONDecodeError, TypeError):
            return None
    if not prices or len(prices) < 2:
        return None
    try:
        p = float(prices[0])
        if p > 0.9:
            return "Up"
        elif p < 0.1:
            return "Down"
        return None
    except (ValueError, TypeError):
        return None


# ── Market fetching ─────────────────────────────────────────────────────────


def fetch_markets(
    conn,
    slug_pattern: str,
    start_date: str | None = None,
    end_date: str | None = None,
    closed_only: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    """Fetch BTC Up/Down markets ordered sequentially.

    Returns a DataFrame with columns: condition_id, slug, end_date, closed,
    active, outcome_prices, outcomes, volume — ordered by end_date ASC.
    """
    where_clauses = ["slug LIKE %(pattern)s"]
    params: dict = {"pattern": slug_pattern}

    if start_date:
        where_clauses.append("end_date >= %(start_date)s")
        params["start_date"] = start_date
    if end_date:
        where_clauses.append("end_date <= %(end_date)s")
        params["end_date"] = end_date
    if closed_only:
        where_clauses.append("closed = true")

    where_sql = " AND ".join(where_clauses)

    if limit:
        sql = f"""
            SELECT * FROM (
                SELECT condition_id, slug, end_date, closed, active,
                       outcome_prices, outcomes, volume
                FROM   polymarket.polymarket_market
                WHERE  {where_sql}
                ORDER  BY end_date DESC, slug DESC
                LIMIT  {int(limit)}
            ) sub
            ORDER BY end_date ASC, slug ASC
        """
    else:
        sql = f"""
            SELECT condition_id, slug, end_date, closed, active,
                   outcome_prices, outcomes, volume
            FROM   polymarket.polymarket_market
            WHERE  {where_sql}
            ORDER  BY end_date ASC, slug ASC
        """
    return pd.read_sql(sql, conn, params=params)


def fetch_markets_with_winners(
    conn,
    slug_pattern: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Fetch closed, settled markets with a 'winner' column resolved."""
    df = fetch_markets(
        conn,
        slug_pattern=slug_pattern,
        start_date=start_date,
        end_date=end_date,
        closed_only=True,
        limit=limit,
    )
    df["winner"] = df.apply(resolve_winner, axis=1)
    # Keep only settled markets
    df = df[df["winner"].notna()].reset_index(drop=True)
    return df
