-- deploy_formula.sql
--
-- Optimized H-Score for Polymarket traders.
-- Pulls eligible wallets from the most recent available snapshot,
-- normalizes each feature to a percentile rank within the eligible pool,
-- and computes a weighted sum to produce h_score (0–100).
--
-- Weights sourced from optimal_weights.json (scipy SLSQP, 50 random inits).
-- Sum of weights = 100 by construction; PERCENT_RANK ∈ [0,1] so h_score ∈ [0,100].
--
-- Inverted features (higher raw value = worse outcome):
--   worst_trade                — ORDER BY DESC  (less-negative = better)
--   market_concentration_ratio — ORDER BY DESC  (lower concentration = better)
--
-- Eligibility filters (wallet_profile_metrics, calculation_window_days = 15):
--   roi > 0
--   win_rate BETWEEN 0.45 AND 0.95
--   total_trades BETWEEN 50 AND 100000
--   total_pnl > 5000
--   combined_risk_score <= 50
--
-- Tier thresholds (calibrated against score distribution):
--   Elite    >= 70
--   Sharp    >= 50
--   Solid    >= 35
--   Emerging  < 35

WITH

-- ── 1. most recent snapshot date with window=15 data ─────────────────────────
latest AS (
    SELECT MAX(date) AS snapshot_date
    FROM polymarket.wallet_profile_metrics
    WHERE calculation_window_days = 15
),

-- ── 2. eligible wallets + raw features ───────────────────────────────────────
-- JSONB extraction for pnl_cat_sports happens here via a correlated subquery.
-- COALESCE ensures wallets with no Sports trades still get 0 (not NULL).
-- Nullable ratio columns (calmar_ratio, sortino_ratio) are coalesced to 0
-- to match the fillna(0) applied during training — keeps ranks consistent.
eligible AS (
    SELECT
        m.proxy_wallet,
        -- output columns
        m.total_invested,
        m.total_pnl,
        m.worst_trade,
        m.win_rate,
        m.total_trades,
        m.markets_traded,
        m.trade_size_stdev,
        COALESCE(m.calmar_ratio, 0)          AS calmar_ratio,
        -- scoring-only columns
        m.best_trade,
        COALESCE(m.avg_position_size, 0)     AS avg_position_size,
        COALESCE(m.sortino_ratio, 0)         AS sortino_ratio,
        m.market_concentration_ratio,
        COALESCE(m.statistical_confidence, 0) AS statistical_confidence,
        COALESCE(m.days_active, 0)           AS days_active,
        -- extract pnl for Sports category from JSONB array
        COALESCE(
            (
                SELECT (elem ->> 'pnl')::FLOAT
                FROM   jsonb_array_elements(m.performance_by_category) AS elem
                WHERE  elem ->> 'category' = 'Sports'
                LIMIT  1
            ),
            0.0
        ) AS pnl_cat_sports
    FROM polymarket.wallet_profile_metrics m
    CROSS JOIN latest
    WHERE m.date                    = latest.snapshot_date
      AND m.calculation_window_days = 15
      AND m.roi                     > 0
      AND m.win_rate                BETWEEN 0.45 AND 0.95
      AND m.total_trades            BETWEEN 50 AND 100000
      AND m.total_pnl               > 5000
      AND m.combined_risk_score     <= 50
),

-- ── 3. percentile-rank each feature within the eligible pool ─────────────────
-- Regular features : ORDER BY ASC  → low value = low rank = low score
-- Inverted features: ORDER BY DESC → high raw value = low rank = low score
-- COALESCE in ORDER BY clause guards against residual NULLs shifting ranks.
ranked AS (
    SELECT
        proxy_wallet,
        -- pass-through raw values for output
        total_invested,
        total_pnl,
        worst_trade,
        pnl_cat_sports,
        calmar_ratio,
        win_rate,
        total_trades,
        markets_traded,
        trade_size_stdev,

        -- regular features (ASC)
        PERCENT_RANK() OVER (ORDER BY COALESCE(trade_size_stdev,      0) ASC)  AS pr_trade_size_stdev,
        PERCENT_RANK() OVER (ORDER BY COALESCE(total_invested,        0) ASC)  AS pr_total_invested,
        PERCENT_RANK() OVER (ORDER BY COALESCE(total_trades,          0) ASC)  AS pr_total_trades,
        PERCENT_RANK() OVER (ORDER BY COALESCE(best_trade,            0) ASC)  AS pr_best_trade,
        PERCENT_RANK() OVER (ORDER BY COALESCE(calmar_ratio,          0) ASC)  AS pr_calmar_ratio,
        PERCENT_RANK() OVER (ORDER BY COALESCE(markets_traded,        0) ASC)  AS pr_markets_traded,
        PERCENT_RANK() OVER (ORDER BY COALESCE(total_pnl,             0) ASC)  AS pr_total_pnl,
        PERCENT_RANK() OVER (ORDER BY COALESCE(pnl_cat_sports,        0) ASC)  AS pr_pnl_cat_sports,
        PERCENT_RANK() OVER (ORDER BY COALESCE(statistical_confidence,0) ASC)  AS pr_statistical_confidence,
        PERCENT_RANK() OVER (ORDER BY COALESCE(avg_position_size,     0) ASC)  AS pr_avg_position_size,
        PERCENT_RANK() OVER (ORDER BY COALESCE(sortino_ratio,         0) ASC)  AS pr_sortino_ratio,
        PERCENT_RANK() OVER (ORDER BY COALESCE(days_active,           0) ASC)  AS pr_days_active,

        -- inverted features (DESC): high raw value → low normalized score
        PERCENT_RANK() OVER (ORDER BY COALESCE(worst_trade,               0) DESC) AS pr_worst_trade_inv,
        PERCENT_RANK() OVER (ORDER BY COALESCE(market_concentration_ratio,0) DESC) AS pr_market_conc_inv
    FROM eligible
),

-- ── 4. weighted sum → h_score ─────────────────────────────────────────────────
-- Weights sum to exactly 100; PERCENT_RANK ∈ [0,1] → h_score ∈ [0,100].
scored AS (
    SELECT
        proxy_wallet,
        total_invested,
        total_pnl,
        worst_trade,
        pnl_cat_sports,
        calmar_ratio,
        win_rate,
        total_trades,
        markets_traded,
        trade_size_stdev,

        ROUND((
            pr_trade_size_stdev         * 21.441 +
            pr_worst_trade_inv          * 17.004 +
            pr_total_invested           * 14.808 +
            pr_total_trades             *  9.163 +
            pr_best_trade               *  8.317 +
            pr_calmar_ratio             *  7.688 +
            pr_markets_traded           *  7.040 +
            pr_total_pnl                *  6.718 +
            pr_pnl_cat_sports           *  3.226 +
            pr_statistical_confidence   *  1.749 +
            pr_avg_position_size        *  1.140 +
            pr_sortino_ratio            *  0.746 +
            pr_market_conc_inv          *  0.625 +
            pr_days_active              *  0.335
        )::NUMERIC, 3) AS h_score
    FROM ranked
)

-- ── 5. final output: rank, tier, key metrics ──────────────────────────────────
SELECT
    RANK() OVER (ORDER BY h_score DESC)::INT AS rank,
    proxy_wallet,
    h_score,
    CASE
        WHEN h_score >= 70 THEN 'Elite'
        WHEN h_score >= 50 THEN 'Sharp'
        WHEN h_score >= 35 THEN 'Solid'
        ELSE                    'Emerging'
    END                             AS tier,
    total_invested,
    total_pnl,
    trade_size_stdev,
    worst_trade,
    pnl_cat_sports,
    calmar_ratio,
    win_rate,
    total_trades,
    markets_traded
FROM  scored
ORDER BY h_score DESC
LIMIT 200;
