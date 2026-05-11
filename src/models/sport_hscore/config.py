"""
Sport H-Score model configuration.

Sports-focused scoring model using wallet_profile_metrics_v2.
All feature lists, eligibility filters, train/test splits, and
hyperparameters for the Sport H-Score weighted scoring model.
"""

from dataclasses import dataclass, field


# ── v2 table reference ───────────────────────────────────────────────────────

TABLE = "polymarket.wallet_profile_metrics_v2"

# ── Sports sub-category mapping ──────────────────────────────────────────────
# Maps JSONB category strings to column suffixes.
# The JSONB uses hierarchical format: "Sports / Basketball / NBA"

SPORTS_SUBCATS = {
    "Sports / Baseball / MLB": "mlb",
    "Sports / Basketball / NBA": "nba",
    "Sports / Basketball / WNBA": "wnba",
    "Sports / Basketball / College": "college_basketball",
    "Sports / Football / NFL": "nfl",
    "Sports / Football / College": "college_football",
    "Sports / Hockey / NHL": "nhl",
    "Sports / Soccer / EPL": "epl",
    "Sports / Soccer / Champions League": "champions_league",
    "Sports / Soccer / La Liga": "la_liga",
    "Sports / Soccer / Ligue 1": "ligue_1",
    "Sports / Soccer / Bundesliga": "bundesliga",
    "Sports / Cricket": "cricket",
    "Sports / Tennis": "tennis",
    "Sports / Golf": "golf",
    "Sports / Rugby": "rugby",
    "Sports / Combat Sports": "combat_sports",
    "Sports / Motorsport / F1": "f1",
    "Sports / Olympics": "olympics",
}

# ── Feature definitions ──────────────────────────────────────────────────────

FEATURES_OVERALL = [
    "total_pnl",
    "best_market_pnl",
    "worst_market_pnl",
    "avg_market_exposure",
    "profitable_markets_count",
    "sortino_ratio",
    "calmar_ratio",
    "annualized_return",
]

FEATURES_SPORTS_AGG = [
    "sports_pnl",
    "sports_trades",
    "sports_invested",
]

FEATURES_SPORTS_SUBCATS = [
    "sports_pnl_mlb",
    "sports_pnl_la_liga",
    "sports_pnl_ligue_1",
    "sports_pnl_bundesliga",
    "sports_pnl_wnba",
    "sports_pnl_golf",
    "sports_pnl_college_football",
    "sports_pnl_f1",
]

FEATURES_MULTIWINDOW = [
    "total_invested_7d",
    "best_market_pnl_7d",
    "worst_market_pnl_7d",
    "profit_factor_7d",
    "win_rate_7d",
    "pnl_cat_sports_7d",
    "total_pnl_3d",
    "total_invested_3d",
]

FEATURES_BASE = FEATURES_OVERALL + FEATURES_SPORTS_AGG + FEATURES_SPORTS_SUBCATS

# Full feature list used by the optimizer
FEATURES = FEATURES_BASE + FEATURES_MULTIWINDOW

N_FEATURES = len(FEATURES)

# Features where higher raw value -> lower score (negated before ranking)
INVERT = {
    "worst_market_pnl",
    "worst_market_pnl_7d",
    "profit_factor_7d",
    "win_rate_7d",
}

# Sparse ratio columns -> imputed with training-data median
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
    "profitable_markets_count",
    "statistical_confidence",
    "performance_by_category",
]

JSONB_CATS = ["sports"]
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

# Sparse ratio columns for base features -> imputed with per-snapshot median
FILLNA_MEDIAN_COLS = [
    "sortino_ratio",
    "calmar_ratio",
    "gain_to_pain_ratio",
    "annualized_return",
]

# ── Eligibility filters ─────────────────────────────────────────────────────

ELIGIBILITY_SQL_FILTERS = """
    AND roi > 0
    AND win_rate BETWEEN 0.40 AND 0.95
    AND total_trades BETWEEN 20 AND 100000
    AND total_pnl > 1000
    AND combined_risk_score <= 60
"""

# Sports-specific eligibility (applied in Python after JSONB parse)
MIN_SPORTS_TRADES = 10
MIN_SPORTS_PNL = 100

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
    "916f (UCL)": "0x916f7165c2c836aba22edb6453cdbb5f3ea253ba",
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
RANK_THRESHOLD = 200


@dataclass
class SportHScoreConfig:
    """Runtime configuration for the Sport H-Score model."""

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
