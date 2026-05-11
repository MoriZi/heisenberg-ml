"""
Copy-target scorecard configuration.

This module produces a per-wallet copyability scorecard from v2 metrics.
The scoring SQL lives in query_template.sql and is structured the same way
as the H-Score data agent: a parameterized template suitable for use as
an MCP data agent over the Postgres DB.

Scope of v1
-----------
v1 covers ~70% of the copy-target investigation list using only
polymarket.wallet_profile_metrics_v2 (window=15) + polymarket.wallet_daily_pnl.
Behavioural signals that require trade-level data (CLV, holding-time
distribution, entry-timing, time-of-day clustering, order-book sensitivity)
are NOT in this version. Those require either:
  - a wallet_trade_metrics_v2 materialised view (recommended)
  - a companion MCP tool that wraps trade-level data
"""

from dataclasses import dataclass, field


# ── Source tables ────────────────────────────────────────────────────────────

TABLE = "polymarket.wallet_profile_metrics_v2"
DAILY_PNL_TABLE = "polymarket.wallet_daily_pnl"


# ── Eligibility filters (mirror H-Score so universe is consistent) ───────────

ELIGIBILITY = {
    "min_roi": 0,
    "min_win_rate": 0.45,
    "max_win_rate": 0.95,
    "min_total_trades": 50,
    "max_total_trades": 100000,
    "min_total_pnl": 5000,
    "max_combined_risk_score": 50,
}


# ── Sub-score component weights (each list sums to its sub-score max) ────────
# These are hand-tuned for v1. When we have labelled copy-outcomes (e.g. a
# wallet's forward CLV over a held-out window), retrain like H-Score.

SKILL_WEIGHTS = {
    "profitable_market_rate":  25,  # profitable_markets_count / markets_traded
    "statistical_confidence":  20,
    "gain_to_pain_ratio":      15,
    "sortino_ratio":           10,
    "annualized_return":       10,
    "win_rate_z_score":        10,
    "performance_trend":       10,  # improving=100, stable=50, declining=0
}

SPECIALIZATION_WEIGHTS = {
    "category_hhi":         70,  # Herfindahl across top-level categories
    "top1_category_share":  30,  # top category PnL as % of total PnL
}

COPYABILITY_WEIGHTS = {
    "avg_trade_size_inv":          30,  # smaller size = more retail-copyable
    "trades_per_week_inv":         20,  # fewer/manageable trades = easier
    "position_size_consistency":   20,
    "coefficient_of_variation_inv":15,
    "single_market_dependence":    15,  # boolean → 0/1 (0=good)
}

RISK_WEIGHTS = {
    "combined_risk_score_inv":  25,
    "sybil_risk_score_inv":     20,
    "similar_wallets_inv":      10,
    "max_drawdown":             15,  # less-negative = better; ranked ASC
    "ulcer_index_inv":          10,
    "recovery_time_avg_inv":    10,
    "positive_days_pct":        10,  # trajectory smoothness proxy
}


# ── Composite weights (must sum to 1.0) ──────────────────────────────────────

COMPOSITE_WEIGHTS = {
    "skill_score":          0.35,
    "specialization_score": 0.15,
    "copyability_score":    0.20,
    "risk_score":           0.30,
}


# ── Tier thresholds (composite score 0..100) ─────────────────────────────────

TIERS = [
    ("Strong Copy", 75),
    ("Watch",       50),
    ("Skip",         0),
]


# ── Default trajectory window for wallet_daily_pnl ───────────────────────────

DEFAULT_TRAJECTORY_DAYS = 60
DEFAULT_TOP_N = 50


@dataclass
class CopyTargetConfig:
    """Runtime configuration for the copy-target scorecard."""

    eligibility: dict = field(default_factory=lambda: dict(ELIGIBILITY))
    skill_weights: dict = field(default_factory=lambda: dict(SKILL_WEIGHTS))
    specialization_weights: dict = field(default_factory=lambda: dict(SPECIALIZATION_WEIGHTS))
    copyability_weights: dict = field(default_factory=lambda: dict(COPYABILITY_WEIGHTS))
    risk_weights: dict = field(default_factory=lambda: dict(RISK_WEIGHTS))
    composite_weights: dict = field(default_factory=lambda: dict(COMPOSITE_WEIGHTS))
    tiers: list = field(default_factory=lambda: list(TIERS))
    trajectory_days: int = DEFAULT_TRAJECTORY_DAYS
    top_n: int = DEFAULT_TOP_N
