"""
H-Score model configuration.

All feature lists, eligibility filters, train/test splits, and
hyperparameters for the H-Score weighted scoring model.

Source table: polymarket.wallet_profile_metrics_v2.
"""

from dataclasses import dataclass, field


# ── v2 table reference ───────────────────────────────────────────────────────

TABLE = "polymarket.wallet_profile_metrics_v2"


# ── Feature definitions ──────────────────────────────────────────────────────

FEATURES_BASE = [
    "total_pnl",
    "total_pnl_1d",
    "total_pnl_3d",
    "total_pnl_7d",
    "total_invested",
    "total_invested_3d",
    "total_invested_7d",
    "best_market_pnl",
    "best_market_pnl_7d",
    "avg_market_exposure",
    "stddev_position_size_7d",
    "dominant_market_pnl",
    "dominant_market_pnl_7d",
    "pnl_cat_sports",
    "pnl_cat_other",
    "pnl_cat_other_7d",
    "statistical_confidence",
    "statistical_confidence_1d",
    "markets_traded",
    "pnl_cat_economics",
    "worst_market_pnl",
    "worst_market_pnl_1d",
    "roi_1d",
    "roi_3d",
    "profit_factor",
    "profit_factor_7d",
    "win_rate",
    "win_rate_1d",
    "win_rate_3d",
    "market_concentration_ratio",
    "pnl_cat_crypto",
]

FEATURES_RATIOS = [
    "sortino_ratio",
    "calmar_ratio",
    "gain_to_pain_ratio",
    "annualized_return",
    "total_trades",
]

FEATURES = FEATURES_BASE + FEATURES_RATIOS
N_FEATURES = len(FEATURES)

# Features where higher raw value → lower score (negated before ranking)
INVERT = {
    "worst_market_pnl",
    "worst_market_pnl_1d",
    "roi_1d",
    "roi_3d",
    "profit_factor",
    "profit_factor_7d",
    "win_rate",
    "win_rate_1d",
    "win_rate_3d",
    "market_concentration_ratio",
    "pnl_cat_crypto",
}

# Sparse ratio columns → imputed with training-data median (not 0)
FILLNA_MEDIAN_FEATS = {
    "sortino_ratio",
    "calmar_ratio",
    "gain_to_pain_ratio",
    "annualized_return",
}

# ── Multiwindow feature extraction ──────────────────────────────────────────

WINDOW_METRIC_COLS = [
    "total_pnl",
    "total_invested",
    "worst_market_pnl",
    "best_market_pnl",
    "total_trades",
    "roi",
    "win_rate",
    "avg_market_exposure",
    "stddev_position_size",
    "dominant_market_pnl",
    "profit_factor",
    "market_concentration_ratio",
    "statistical_confidence",
    "performance_by_category",
]

JSONB_CATS = ["sports", "other"]
WINDOWS = [1, 3, 7]

# ── Base feature columns from wallet_profile_metrics_v2 (window=15) ─────────

METRIC_COLS = [
    "proxy_wallet",
    "roi",
    "win_rate",
    "total_pnl",
    "total_invested",
    "total_trades",
    "markets_traded",
    "sortino_ratio",
    "calmar_ratio",
    "gain_to_pain_ratio",
    "annualized_return",
    "max_drawdown",
    "ulcer_index",
    "drawdown_frequency",
    "recovery_time_avg",
    "profit_factor",
    "performance_trend",
    "curve_volatility",
    "equity_curve_pattern",
    "avg_market_exposure",
    "avg_trade_size",
    "stddev_position_size",
    "coefficient_of_variation",
    "dominant_market_pnl",
    "market_concentration_ratio",
    "category_diversity_score",
    "days_active",
    "best_market_pnl",
    "worst_market_pnl",
    "win_rate_last_30day",
    "win_rate_z_score",
    "timing_hit_rate",
    "timing_z_score",
    "timing_anomaly_flag",
    "perfect_timing_score",
    "profitable_markets_count",
    "high_win_rate_markets_count",
    "statistical_confidence",
    "combined_risk_score",
    "risk_level",
    "suspicious_win_rate_flag",
    "single_market_dependence_flag",
    "sybil_risk_score",
    "sybil_risk_flag",
    "num_markets_traded",
    "buy_trade_ratio",
    "sell_trade_ratio",
    "pnl_last_30day",
    "performance_by_category",
]

# Sparse ratio columns for base features → imputed with per-snapshot median
FILLNA_MEDIAN_COLS = [
    "sortino_ratio",
    "calmar_ratio",
    "gain_to_pain_ratio",
    "annualized_return",
]

# Known performance_by_category categories
KNOWN_CATEGORIES = [
    "Sports",
    "Crypto",
    "World Events",
    "Economics",
    "Science & Tech",
    "Politics",
    "Entertainment",
    "Weather",
    "Other",
]

# ── Eligibility filters ─────────────────────────────────────────────────────

ELIGIBILITY_SQL_FILTERS = """
    AND roi > 0
    AND win_rate BETWEEN 0.45 AND 0.95
    AND total_trades BETWEEN 50 AND 100000
    AND total_pnl > 5000
    AND combined_risk_score <= 50
"""

# ── Walk-forward evaluation ──────────────────────────────────────────────────

TRAIN_END = "2026-04-20"
TEST_START = "2026-04-27"

FOLDS = [
    {"name": "Fold 1", "train_end": "2026-04-01", "test_start": "2026-04-08"},
    {"name": "Fold 2", "train_end": "2026-04-10", "test_start": "2026-04-17"},
    {"name": "Fold 3", "train_end": "2026-04-20", "test_start": "2026-04-27"},
]

KNOWN_WALLETS = {
    "cf11 (NBA)": "0xcf119e969f31de9653a58cb3dc213b485cd48399",
    "ccb2 (CS2)": "0xccb290b1c145d1c95695d3756346bba9f1398586",
    "916f (UCL)": "0x916f7165c2c836aba22edb6453cdbb5f3ea253ba",
    "d008 (selective)": "0xd008786fad743d0d5c60f99bff5d90ebc212135d",
}

# ── Tier thresholds ──────────────────────────────────────────────────────────

TIERS = [
    ("Elite", 70),
    ("Sharp", 50),
    ("Solid", 35),
    ("Emerging", 0),
]

# ── Pipeline date range ──────────────────────────────────────────────────────

PIPELINE_START = "2026-03-05"
PIPELINE_END = "2026-05-04"
RANK_THRESHOLD = 500


@dataclass
class HScoreConfig:
    """Runtime configuration for the H-Score model."""

    features: list[str] = field(default_factory=lambda: list(FEATURES))
    n_features: int = N_FEATURES
    invert: set[str] = field(default_factory=lambda: set(INVERT))
    fillna_median_feats: set[str] = field(
        default_factory=lambda: set(FILLNA_MEDIAN_FEATS)
    )
    train_end: str = TRAIN_END
    test_start: str = TEST_START

    # Optimizer settings
    n_init: int = 50
    seed: int = 42
    k: int = 25
    ftol: float = 1e-5
    eps: float = 1.0
    maxiter: int = 500
