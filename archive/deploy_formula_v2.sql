-- deploy_formula_v2.sql
--
-- Optimized H-Score for Polymarket traders — feature set v2.
-- Weights from optimal_weights_v2.json (scipy SLSQP, 50 random inits,
-- Spearman rho = 0.2575 on full dataset).
--
-- Key changes from v1:
--   REMOVED  constants (0% variance): trade_size_stdev, max_position_pct,
--            position_size_consistency, trade_timing_correlation_max, edge_decay
--   ADDED    : stddev_position_size, coefficient_of_variation, dominant_market_pnl,
--              curve_smoothness, perfect_timing_score, perfect_entry_count,
--              perfect_exit_count
--   JSONB    : expanded from 1 to 7 categories (Sports, Crypto, World Events,
--              Economics, Politics, Entertainment, Other); Weather extracted
--              but not scored (weight = 0 in this version)
--   IMPUTATION: calmar_ratio, sortino_ratio, gain_to_pain_ratio, annualized_return
--              → COALESCE with within-pool median (not 0)
--
-- Sum of weights = 100 by construction; PERCENT_RANK ∈ [0,1] → h_score ∈ [0,100].
--
-- Inverted features (higher raw value = worse outcome):
--   worst_trade                — ORDER BY DESC
--   market_concentration_ratio — ORDER BY DESC
--
-- Eligibility filters (calculation_window_days = 15):
--   roi > 0
--   win_rate BETWEEN 0.45 AND 0.95
--   total_trades BETWEEN 50 AND 100000
--   total_pnl > 5000
--   combined_risk_score <= 50
--
-- Tier thresholds:
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

-- ── 2. median values for sparse ratio columns ────────────────────────────────
-- Computed within the eligible pool so imputation reflects the peer group,
-- not the full wallet universe.
medians AS (
    SELECT
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY calmar_ratio)       AS med_calmar_ratio,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY sortino_ratio)      AS med_sortino_ratio,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY gain_to_pain_ratio) AS med_gain_to_pain_ratio,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY annualized_return)  AS med_annualized_return
    FROM polymarket.wallet_profile_metrics
    CROSS JOIN latest
    WHERE date                    = latest.snapshot_date
      AND calculation_window_days = 15
      AND roi                     > 0
      AND win_rate                BETWEEN 0.45 AND 0.95
      AND total_trades            BETWEEN 50 AND 100000
      AND total_pnl               > 5000
      AND combined_risk_score     <= 50
),

-- ── 3. eligible wallets + raw features ───────────────────────────────────────
-- Sparse ratios are COALESCED with the within-pool median computed above.
-- JSONB extraction via correlated subqueries with LIMIT 1; missing categories
-- default to 0. Weather extracted for completeness but not used in scoring.
eligible AS (
    SELECT
        m.proxy_wallet,
        -- output / pass-through columns
        m.total_invested,
        m.total_pnl,
        m.best_trade,
        m.worst_trade,
        m.win_rate,
        m.total_trades,
        m.markets_traded,
        -- position sizing
        COALESCE(m.avg_position_size,       0)                        AS avg_position_size,
        COALESCE(m.stddev_position_size,    0)                        AS stddev_position_size,
        COALESCE(m.coefficient_of_variation, 0)                       AS coefficient_of_variation,
        COALESCE(m.dominant_market_pnl,     0)                        AS dominant_market_pnl,
        -- sparse ratios → median imputed
        COALESCE(m.calmar_ratio,       med.med_calmar_ratio)          AS calmar_ratio,
        COALESCE(m.sortino_ratio,      med.med_sortino_ratio)         AS sortino_ratio,
        COALESCE(m.gain_to_pain_ratio, med.med_gain_to_pain_ratio)    AS gain_to_pain_ratio,
        COALESCE(m.annualized_return,  med.med_annualized_return)     AS annualized_return,
        -- activity / quality
        COALESCE(m.days_active,            0)                         AS days_active,
        m.market_concentration_ratio,
        COALESCE(m.statistical_confidence, 0)                         AS statistical_confidence,
        COALESCE(m.curve_smoothness,       0)                         AS curve_smoothness,
        COALESCE(m.perfect_timing_score,   0)                         AS perfect_timing_score,
        COALESCE(m.perfect_entry_count,    0)                         AS perfect_entry_count,
        COALESCE(m.perfect_exit_count,     0)                         AS perfect_exit_count,
        -- JSONB: per-category pnl
        COALESCE(
            (SELECT (elem->>'pnl')::FLOAT
             FROM   jsonb_array_elements(m.performance_by_category) AS elem
             WHERE  elem->>'category' = 'Sports' LIMIT 1), 0.0)      AS pnl_cat_sports,
        COALESCE(
            (SELECT (elem->>'pnl')::FLOAT
             FROM   jsonb_array_elements(m.performance_by_category) AS elem
             WHERE  elem->>'category' = 'Crypto' LIMIT 1), 0.0)      AS pnl_cat_crypto,
        COALESCE(
            (SELECT (elem->>'pnl')::FLOAT
             FROM   jsonb_array_elements(m.performance_by_category) AS elem
             WHERE  elem->>'category' = 'World Events' LIMIT 1), 0.0) AS pnl_cat_world_events,
        COALESCE(
            (SELECT (elem->>'pnl')::FLOAT
             FROM   jsonb_array_elements(m.performance_by_category) AS elem
             WHERE  elem->>'category' = 'Economics' LIMIT 1), 0.0)   AS pnl_cat_economics,
        COALESCE(
            (SELECT (elem->>'pnl')::FLOAT
             FROM   jsonb_array_elements(m.performance_by_category) AS elem
             WHERE  elem->>'category' = 'Politics' LIMIT 1), 0.0)    AS pnl_cat_politics,
        COALESCE(
            (SELECT (elem->>'pnl')::FLOAT
             FROM   jsonb_array_elements(m.performance_by_category) AS elem
             WHERE  elem->>'category' = 'Entertainment' LIMIT 1), 0.0) AS pnl_cat_entertainment,
        COALESCE(
            (SELECT (elem->>'pnl')::FLOAT
             FROM   jsonb_array_elements(m.performance_by_category) AS elem
             WHERE  elem->>'category' = 'Other' LIMIT 1), 0.0)       AS pnl_cat_other,
        COALESCE(
            (SELECT (elem->>'pnl')::FLOAT
             FROM   jsonb_array_elements(m.performance_by_category) AS elem
             WHERE  elem->>'category' = 'Weather' LIMIT 1), 0.0)     AS pnl_cat_weather
    FROM polymarket.wallet_profile_metrics m
    CROSS JOIN latest
    CROSS JOIN medians med
    WHERE m.date                    = latest.snapshot_date
      AND m.calculation_window_days = 15
      AND m.roi                     > 0
      AND m.win_rate                BETWEEN 0.45 AND 0.95
      AND m.total_trades            BETWEEN 50 AND 100000
      AND m.total_pnl               > 5000
      AND m.combined_risk_score     <= 50
),

-- ── 4. percentile-rank each feature within the eligible pool ─────────────────
-- Regular features : ORDER BY ASC  → low value = low rank = low score
-- Inverted features: ORDER BY DESC → high raw value = low rank = low score
-- COALESCE in ORDER BY guards against any residual NULLs shifting ranks.
ranked AS (
    SELECT
        proxy_wallet,
        -- pass-through raw values for output
        total_invested,
        total_pnl,
        best_trade,
        worst_trade,
        win_rate,
        total_trades,
        markets_traded,

        -- regular features (ASC)
        PERCENT_RANK() OVER (ORDER BY COALESCE(total_invested,          0) ASC) AS pr_total_invested,
        PERCENT_RANK() OVER (ORDER BY COALESCE(total_pnl,               0) ASC) AS pr_total_pnl,
        PERCENT_RANK() OVER (ORDER BY COALESCE(best_trade,              0) ASC) AS pr_best_trade,
        PERCENT_RANK() OVER (ORDER BY COALESCE(avg_position_size,       0) ASC) AS pr_avg_position_size,
        PERCENT_RANK() OVER (ORDER BY COALESCE(stddev_position_size,    0) ASC) AS pr_stddev_position_size,
        PERCENT_RANK() OVER (ORDER BY COALESCE(coefficient_of_variation, 0) ASC) AS pr_coefficient_of_variation,
        PERCENT_RANK() OVER (ORDER BY COALESCE(dominant_market_pnl,     0) ASC) AS pr_dominant_market_pnl,
        PERCENT_RANK() OVER (ORDER BY COALESCE(calmar_ratio,            0) ASC) AS pr_calmar_ratio,
        PERCENT_RANK() OVER (ORDER BY COALESCE(sortino_ratio,           0) ASC) AS pr_sortino_ratio,
        PERCENT_RANK() OVER (ORDER BY COALESCE(gain_to_pain_ratio,      0) ASC) AS pr_gain_to_pain_ratio,
        PERCENT_RANK() OVER (ORDER BY COALESCE(annualized_return,       0) ASC) AS pr_annualized_return,
        PERCENT_RANK() OVER (ORDER BY COALESCE(total_trades,            0) ASC) AS pr_total_trades,
        PERCENT_RANK() OVER (ORDER BY COALESCE(markets_traded,          0) ASC) AS pr_markets_traded,
        PERCENT_RANK() OVER (ORDER BY COALESCE(days_active,             0) ASC) AS pr_days_active,
        PERCENT_RANK() OVER (ORDER BY COALESCE(statistical_confidence,  0) ASC) AS pr_statistical_confidence,
        PERCENT_RANK() OVER (ORDER BY COALESCE(curve_smoothness,        0) ASC) AS pr_curve_smoothness,
        PERCENT_RANK() OVER (ORDER BY COALESCE(perfect_timing_score,    0) ASC) AS pr_perfect_timing_score,
        PERCENT_RANK() OVER (ORDER BY COALESCE(perfect_entry_count,     0) ASC) AS pr_perfect_entry_count,
        PERCENT_RANK() OVER (ORDER BY COALESCE(perfect_exit_count,      0) ASC) AS pr_perfect_exit_count,
        PERCENT_RANK() OVER (ORDER BY COALESCE(pnl_cat_sports,          0) ASC) AS pr_pnl_cat_sports,
        PERCENT_RANK() OVER (ORDER BY COALESCE(pnl_cat_crypto,          0) ASC) AS pr_pnl_cat_crypto,
        PERCENT_RANK() OVER (ORDER BY COALESCE(pnl_cat_world_events,    0) ASC) AS pr_pnl_cat_world_events,
        PERCENT_RANK() OVER (ORDER BY COALESCE(pnl_cat_economics,       0) ASC) AS pr_pnl_cat_economics,
        PERCENT_RANK() OVER (ORDER BY COALESCE(pnl_cat_politics,        0) ASC) AS pr_pnl_cat_politics,
        PERCENT_RANK() OVER (ORDER BY COALESCE(pnl_cat_entertainment,   0) ASC) AS pr_pnl_cat_entertainment,
        PERCENT_RANK() OVER (ORDER BY COALESCE(pnl_cat_other,           0) ASC) AS pr_pnl_cat_other,

        -- inverted features (DESC): high raw value → low normalized score
        PERCENT_RANK() OVER (ORDER BY COALESCE(worst_trade,                0) DESC) AS pr_worst_trade_inv,
        PERCENT_RANK() OVER (ORDER BY COALESCE(market_concentration_ratio, 0) DESC) AS pr_market_conc_inv
    FROM eligible
),

-- ── 5. weighted sum → h_score ─────────────────────────────────────────────────
-- Weights sum to exactly 100; PERCENT_RANK ∈ [0,1] → h_score ∈ [0,100].
-- Weights from optimal_weights_v2.json (rounded to 3 d.p.; sum = 100.000).
scored AS (
    SELECT
        proxy_wallet,
        total_invested,
        total_pnl,
        best_trade,
        worst_trade,
        win_rate,
        total_trades,
        markets_traded,

        ROUND((
            pr_total_invested           * 11.023 +
            pr_total_pnl                *  8.997 +
            pr_best_trade               *  1.796 +
            pr_worst_trade_inv          *  5.515 +
            pr_avg_position_size        *  6.073 +
            pr_stddev_position_size     *  2.090 +
            pr_coefficient_of_variation *  1.509 +
            pr_dominant_market_pnl      *  5.485 +
            pr_calmar_ratio             *  1.815 +
            pr_sortino_ratio            *  6.656 +
            pr_gain_to_pain_ratio       *  2.730 +
            pr_annualized_return        *  1.915 +
            pr_total_trades             *  1.406 +
            pr_markets_traded           *  0.938 +
            pr_days_active              *  0.830 +
            pr_market_conc_inv          *  0.151 +
            pr_statistical_confidence   *  0.944 +
            pr_curve_smoothness         *  0.155 +
            pr_perfect_timing_score     *  0.888 +
            pr_perfect_entry_count      *  8.886 +
            pr_perfect_exit_count       *  6.860 +
            pr_pnl_cat_sports           *  6.493 +
            pr_pnl_cat_crypto           *  1.513 +
            pr_pnl_cat_world_events     *  8.245 +
            pr_pnl_cat_economics        *  3.015 +
            pr_pnl_cat_politics         *  0.236 +
            pr_pnl_cat_entertainment    *  0.834 +
            pr_pnl_cat_other            *  3.002
        )::NUMERIC, 3) AS h_score
    FROM ranked
)

-- ── 6. final output: rank, tier, key metrics ──────────────────────────────────
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
    win_rate,
    worst_trade,
    total_trades,
    markets_traded
FROM  scored
ORDER BY h_score DESC
LIMIT 200;
