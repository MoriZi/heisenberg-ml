"""
BTC signal configuration.

All tunable parameters for the multi-feature signal card pipeline.
"""

from dataclasses import dataclass


# ── Market slug patterns ────────────────────────────────────────────────────

MARKET_SLUG_PATTERN_5M = "btc-updown-5m-%"
MARKET_SLUG_PATTERN_15M = "btc-updown-15m-%"


@dataclass
class BTCSignalConfig:
    """Runtime configuration for the BTC signal pipeline."""

    slug_pattern: str = MARKET_SLUG_PATTERN_5M

    # ── Feature: Streak reversal ────────────────────────────────────────
    streak_min_length: int = 3      # consecutive same outcomes to trigger reversal
    streak_lookback: int = 10       # how many prior markets to scan

    # ── Feature: Decisiveness ───────────────────────────────────────────
    # Measured by minute-0 displacement from 0.50 in the prior market.
    # |close_min0 - 0.50| < indecisive_max → indecisive (reversal signal)
    # |close_min0 - 0.50| > decisive_min   → decisive (continuation signal)
    decisive_min_displacement: float = 0.20
    indecisive_max_displacement: float = 0.10

    # ── Feature: Volume trend ───────────────────────────────────────────
    volume_lookback: int = 5        # number of prior markets for trend

    # ── Feature: Price lean ─────────────────────────────────────────────
    # Pre-window candle observation for the current/target market.
    price_lean_pre_window_minutes: int = 5  # how far before window to look

    # ── Feature: Early movement ─────────────────────────────────────────
    early_window_seconds: int = 60  # first N seconds of current market window

    # ── Query batching ──────────────────────────────────────────────────
    batch_size: int = 10

    # ── Live monitor ────────────────────────────────────────────────────
    poll_interval: int = 60
    monitor_lookback: int = 15

    # ── Backtest defaults ───────────────────────────────────────────────
    backtest_days: int = 7

    @property
    def history_depth(self) -> int:
        """Max lookback needed across all features."""
        return max(self.streak_lookback, self.volume_lookback, 5)
