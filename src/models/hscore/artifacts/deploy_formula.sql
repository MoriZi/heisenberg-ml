-- deploy_formula_v8.sql
--
-- ⚠ STALE — pending regeneration after v2 retrain.
-- This formula references polymarket.wallet_profile_metrics (v1) and
-- the v1 column names (best_trade, worst_trade, avg_position_size,
-- perfect_entry_count, …). The Python pipeline/training stack now
-- targets wallet_profile_metrics_v2, so the weights baked in below
-- no longer match the feature set produced by train.py.
-- Rebuild: rerun pipeline.py → train.py → regenerate this SQL.
--
-- H-Score formula — feature set v8 (Precision@25 objective).
-- Weights from optimal_weights_v8.json (scipy SLSQP, 50 random inits,
-- objective: mean Precision@25 across snapshot dates).
--
-- Key changes from v2:
--   OBJECTIVE  : P@25 (not Spearman rho)
--   WINDOWS    : 15d base + LEFT JOIN 1d, 3d, 7d for 16 short-window features
--   FEATURES   : 37 total (21 from w=15, 5 from w=1, 4 from w=3, 7 from w=7)
--   MEDIANS    : sortino_ratio, calmar_ratio, gain_to_pain_ratio,
--                annualized_return → COALESCE with within-pool median
--   JSONB      : Sports, Other, Economics, Crypto from w=15;
--                Other from w=7 (pnl_cat_other_7d)
--
-- Inverted features (higher raw value → lower score, ORDER BY DESC):
--   worst_trade, worst_trade_1d
--   roi_1d, roi_3d
--   profit_factor, profit_factor_7d
--   win_rate, win_rate_1d, win_rate_3d
--   market_concentration_ratio
--   pnl_cat_crypto
--
-- Eligibility filters (on calculation_window_days = 15):
--   roi > 0
--   win_rate BETWEEN 0.45 AND 0.95
--   total_trades BETWEEN 50 AND 20000
--   total_pnl > 5000
--   combined_risk_score <= 50
--   (total_pnl / total_trades) > 10  — excludes market makers
--
-- Tier thresholds:
--   Elite    >= 70
--   Sharp    >= 50
--   Solid    >= 35
--   Emerging  < 35
--
-- Sum of weights = 100 by construction; PERCENT_RANK ∈ [0,1] → h_score ∈ [0,100].

WITH

-- ── 1. most recent snapshot date (window=15) ──────────────────────────────────
latest AS (
    SELECT MAX(date) AS snapshot_date
    FROM polymarket.wallet_profile_metrics
    WHERE calculation_window_days = 15
),

-- ── 2. within-pool medians for sparse ratio columns ───────────────────────────
-- Computed within the eligible pool so imputation reflects the peer group.
medians AS (
    SELECT
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY sortino_ratio)      AS med_sortino_ratio,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY calmar_ratio)       AS med_calmar_ratio,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY gain_to_pain_ratio) AS med_gain_to_pain_ratio,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY annualized_return)  AS med_annualized_return
    FROM polymarket.wallet_profile_metrics
    CROSS JOIN latest
    WHERE date                    = latest.snapshot_date
      AND calculation_window_days = 15
      AND roi                     > 0
      AND win_rate                BETWEEN 0.45 AND 0.95
      AND total_trades            BETWEEN 50 AND 20000
      AND total_pnl               > 5000
      AND combined_risk_score     <= 50
),

-- ── 3. eligible wallets (w=15) + short-window features (LEFT JOIN w=1,3,7) ────
-- Eligibility filters applied to m15 only.
-- Wallets absent from a shorter window get 0 via COALESCE.
eligible AS (
    SELECT
        m15.proxy_wallet,

        -- ── pass-through output columns ───────────────────────────────────
        m15.total_pnl,
        m15.total_invested,
        m15.best_trade,
        m15.worst_trade,
        m15.win_rate,
        m15.total_trades,
        m15.markets_traded,

        -- ── window=15 regular features ────────────────────────────────────
        COALESCE(m15.avg_position_size,       0)                              AS avg_position_size,
        COALESCE(m15.dominant_market_pnl,     0)                              AS dominant_market_pnl,
        COALESCE(m15.perfect_entry_count,     0)                              AS perfect_entry_count,
        COALESCE(m15.statistical_confidence,  0)                              AS statistical_confidence,
        COALESCE(m15.market_concentration_ratio, 0)                           AS market_concentration_ratio,
        COALESCE(m15.profit_factor,           0)                              AS profit_factor,

        -- ── window=15 sparse ratios → median imputed ──────────────────────
        COALESCE(m15.sortino_ratio,      med.med_sortino_ratio)               AS sortino_ratio,
        COALESCE(m15.calmar_ratio,       med.med_calmar_ratio)                AS calmar_ratio,
        COALESCE(m15.gain_to_pain_ratio, med.med_gain_to_pain_ratio)          AS gain_to_pain_ratio,
        COALESCE(m15.annualized_return,  med.med_annualized_return)           AS annualized_return,

        -- ── window=15 JSONB: per-category PnL ────────────────────────────
        COALESCE(
            (SELECT (elem->>'pnl')::FLOAT
             FROM   jsonb_array_elements(m15.performance_by_category) AS elem
             WHERE  elem->>'category' = 'Sports' LIMIT 1), 0.0)               AS pnl_cat_sports,
        COALESCE(
            (SELECT (elem->>'pnl')::FLOAT
             FROM   jsonb_array_elements(m15.performance_by_category) AS elem
             WHERE  elem->>'category' = 'Other' LIMIT 1), 0.0)                AS pnl_cat_other,
        COALESCE(
            (SELECT (elem->>'pnl')::FLOAT
             FROM   jsonb_array_elements(m15.performance_by_category) AS elem
             WHERE  elem->>'category' = 'Economics' LIMIT 1), 0.0)            AS pnl_cat_economics,
        COALESCE(
            (SELECT (elem->>'pnl')::FLOAT
             FROM   jsonb_array_elements(m15.performance_by_category) AS elem
             WHERE  elem->>'category' = 'Crypto' LIMIT 1), 0.0)               AS pnl_cat_crypto,

        -- ── window=1 features ─────────────────────────────────────────────
        COALESCE(m1.total_pnl,               0)                               AS total_pnl_1d,
        COALESCE(m1.statistical_confidence,  0)                               AS statistical_confidence_1d,
        COALESCE(m1.worst_trade,             0)                               AS worst_trade_1d,
        COALESCE(m1.roi,                     0)                               AS roi_1d,
        COALESCE(m1.win_rate,                0)                               AS win_rate_1d,

        -- ── window=3 features ─────────────────────────────────────────────
        COALESCE(m3.total_pnl,               0)                               AS total_pnl_3d,
        COALESCE(m3.total_invested,          0)                               AS total_invested_3d,
        COALESCE(m3.roi,                     0)                               AS roi_3d,
        COALESCE(m3.win_rate,                0)                               AS win_rate_3d,

        -- ── window=7 features ─────────────────────────────────────────────
        COALESCE(m7.total_pnl,               0)                               AS total_pnl_7d,
        COALESCE(m7.total_invested,          0)                               AS total_invested_7d,
        COALESCE(m7.best_trade,              0)                               AS best_trade_7d,
        COALESCE(m7.stddev_position_size,    0)                               AS stddev_position_size_7d,
        COALESCE(m7.dominant_market_pnl,     0)                               AS dominant_market_pnl_7d,
        COALESCE(m7.profit_factor,           0)                               AS profit_factor_7d,
        COALESCE(
            (SELECT (elem->>'pnl')::FLOAT
             FROM   jsonb_array_elements(m7.performance_by_category) AS elem
             WHERE  elem->>'category' = 'Other' LIMIT 1), 0.0)                AS pnl_cat_other_7d

    FROM polymarket.wallet_profile_metrics m15
    CROSS JOIN latest
    CROSS JOIN medians med

    -- short-window joins: same wallet, same date, different window
    LEFT JOIN polymarket.wallet_profile_metrics m1
           ON m1.proxy_wallet            = m15.proxy_wallet
          AND m1.date                    = latest.snapshot_date
          AND m1.calculation_window_days = 1

    LEFT JOIN polymarket.wallet_profile_metrics m3
           ON m3.proxy_wallet            = m15.proxy_wallet
          AND m3.date                    = latest.snapshot_date
          AND m3.calculation_window_days = 3

    LEFT JOIN polymarket.wallet_profile_metrics m7
           ON m7.proxy_wallet            = m15.proxy_wallet
          AND m7.date                    = latest.snapshot_date
          AND m7.calculation_window_days = 7

    WHERE m15.date                    = latest.snapshot_date
      AND m15.calculation_window_days = 15
      AND m15.roi                     > 0
      AND m15.win_rate                BETWEEN 0.45 AND 0.95
      AND m15.total_trades            BETWEEN 50 AND 20000
      AND m15.total_pnl               > 5000
      AND m15.combined_risk_score     <= 50
      AND (m15.total_pnl / NULLIF(m15.total_trades, 0)) > 10
),

-- ── 4. percentile-rank each feature within the eligible pool ─────────────────
-- Regular   (ASC):  low value → low rank → low score
-- Inverted  (DESC): high value → low rank → low score
ranked AS (
    SELECT
        proxy_wallet,
        total_pnl,
        total_invested,
        best_trade,
        worst_trade,
        win_rate,
        total_trades,
        markets_traded,

        -- ── window=15 regular (ASC) ───────────────────────────────────────
        PERCENT_RANK() OVER (ORDER BY COALESCE(total_pnl,               0) ASC) AS pr_total_pnl,
        PERCENT_RANK() OVER (ORDER BY COALESCE(total_invested,          0) ASC) AS pr_total_invested,
        PERCENT_RANK() OVER (ORDER BY COALESCE(best_trade,              0) ASC) AS pr_best_trade,
        PERCENT_RANK() OVER (ORDER BY COALESCE(avg_position_size,       0) ASC) AS pr_avg_position_size,
        PERCENT_RANK() OVER (ORDER BY COALESCE(dominant_market_pnl,     0) ASC) AS pr_dominant_market_pnl,
        PERCENT_RANK() OVER (ORDER BY COALESCE(pnl_cat_sports,          0) ASC) AS pr_pnl_cat_sports,
        PERCENT_RANK() OVER (ORDER BY COALESCE(pnl_cat_other,           0) ASC) AS pr_pnl_cat_other,
        PERCENT_RANK() OVER (ORDER BY COALESCE(pnl_cat_economics,       0) ASC) AS pr_pnl_cat_economics,
        PERCENT_RANK() OVER (ORDER BY COALESCE(perfect_entry_count,     0) ASC) AS pr_perfect_entry_count,
        PERCENT_RANK() OVER (ORDER BY COALESCE(statistical_confidence,  0) ASC) AS pr_statistical_confidence,
        PERCENT_RANK() OVER (ORDER BY COALESCE(markets_traded,          0) ASC) AS pr_markets_traded,
        PERCENT_RANK() OVER (ORDER BY COALESCE(sortino_ratio,           0) ASC) AS pr_sortino_ratio,
        PERCENT_RANK() OVER (ORDER BY COALESCE(calmar_ratio,            0) ASC) AS pr_calmar_ratio,
        PERCENT_RANK() OVER (ORDER BY COALESCE(gain_to_pain_ratio,      0) ASC) AS pr_gain_to_pain_ratio,
        PERCENT_RANK() OVER (ORDER BY COALESCE(annualized_return,       0) ASC) AS pr_annualized_return,
        PERCENT_RANK() OVER (ORDER BY COALESCE(total_trades,            0) ASC) AS pr_total_trades,

        -- ── window=15 inverted (DESC) ─────────────────────────────────────
        PERCENT_RANK() OVER (ORDER BY COALESCE(worst_trade,                0) DESC) AS pr_worst_trade_inv,
        PERCENT_RANK() OVER (ORDER BY COALESCE(profit_factor,              0) DESC) AS pr_profit_factor_inv,
        PERCENT_RANK() OVER (ORDER BY COALESCE(win_rate,                   0) DESC) AS pr_win_rate_inv,
        PERCENT_RANK() OVER (ORDER BY COALESCE(market_concentration_ratio, 0) DESC) AS pr_market_conc_inv,
        PERCENT_RANK() OVER (ORDER BY COALESCE(pnl_cat_crypto,             0) DESC) AS pr_pnl_cat_crypto_inv,

        -- ── window=1 regular (ASC) ────────────────────────────────────────
        PERCENT_RANK() OVER (ORDER BY COALESCE(total_pnl_1d,             0) ASC) AS pr_total_pnl_1d,
        PERCENT_RANK() OVER (ORDER BY COALESCE(statistical_confidence_1d, 0) ASC) AS pr_statistical_confidence_1d,

        -- ── window=1 inverted (DESC) ──────────────────────────────────────
        PERCENT_RANK() OVER (ORDER BY COALESCE(worst_trade_1d,            0) DESC) AS pr_worst_trade_1d_inv,
        PERCENT_RANK() OVER (ORDER BY COALESCE(roi_1d,                    0) DESC) AS pr_roi_1d_inv,
        PERCENT_RANK() OVER (ORDER BY COALESCE(win_rate_1d,               0) DESC) AS pr_win_rate_1d_inv,

        -- ── window=3 regular (ASC) ────────────────────────────────────────
        PERCENT_RANK() OVER (ORDER BY COALESCE(total_pnl_3d,             0) ASC) AS pr_total_pnl_3d,
        PERCENT_RANK() OVER (ORDER BY COALESCE(total_invested_3d,        0) ASC) AS pr_total_invested_3d,

        -- ── window=3 inverted (DESC) ──────────────────────────────────────
        PERCENT_RANK() OVER (ORDER BY COALESCE(roi_3d,                    0) DESC) AS pr_roi_3d_inv,
        PERCENT_RANK() OVER (ORDER BY COALESCE(win_rate_3d,               0) DESC) AS pr_win_rate_3d_inv,

        -- ── window=7 regular (ASC) ────────────────────────────────────────
        PERCENT_RANK() OVER (ORDER BY COALESCE(total_pnl_7d,             0) ASC) AS pr_total_pnl_7d,
        PERCENT_RANK() OVER (ORDER BY COALESCE(total_invested_7d,        0) ASC) AS pr_total_invested_7d,
        PERCENT_RANK() OVER (ORDER BY COALESCE(best_trade_7d,            0) ASC) AS pr_best_trade_7d,
        PERCENT_RANK() OVER (ORDER BY COALESCE(stddev_position_size_7d,  0) ASC) AS pr_stddev_position_size_7d,
        PERCENT_RANK() OVER (ORDER BY COALESCE(dominant_market_pnl_7d,   0) ASC) AS pr_dominant_market_pnl_7d,
        PERCENT_RANK() OVER (ORDER BY COALESCE(pnl_cat_other_7d,         0) ASC) AS pr_pnl_cat_other_7d,

        -- ── window=7 inverted (DESC) ──────────────────────────────────────
        PERCENT_RANK() OVER (ORDER BY COALESCE(profit_factor_7d,          0) DESC) AS pr_profit_factor_7d_inv

    FROM eligible
),

-- ── 5. weighted sum → h_score ────────────────────────────────────────────────
-- Weights from optimal_weights_v8.json; sum = 100.000.
-- PERCENT_RANK ∈ [0,1] → h_score ∈ [0,100].
scored AS (
    SELECT
        proxy_wallet,
        total_pnl,
        total_invested,
        best_trade,
        worst_trade,
        win_rate,
        total_trades,
        markets_traded,

        ROUND((
            -- window=15 regular
            pr_total_pnl                  *  2.8681 +
            pr_total_invested             *  8.5946 +
            pr_best_trade                 *  3.2702 +
            pr_avg_position_size          *  1.0466 +
            pr_dominant_market_pnl        *  1.3369 +
            pr_pnl_cat_sports             *  0.9309 +
            pr_pnl_cat_other              *  3.9187 +
            pr_pnl_cat_economics          *  0.1303 +
            pr_perfect_entry_count        *  0.8863 +
            pr_statistical_confidence     *  3.3303 +
            pr_markets_traded             *  3.1074 +
            pr_sortino_ratio              *  0.4581 +
            pr_calmar_ratio               *  2.8014 +
            pr_gain_to_pain_ratio         *  1.6660 +
            pr_annualized_return          *  5.6174 +
            pr_total_trades               *  2.2057 +
            -- window=15 inverted
            pr_worst_trade_inv            *  0.5265 +
            pr_profit_factor_inv          *  1.9074 +
            pr_win_rate_inv               *  0.4673 +
            pr_market_conc_inv            *  2.6875 +
            pr_pnl_cat_crypto_inv         *  6.6407 +
            -- window=1 regular
            pr_total_pnl_1d              *  4.4782 +
            pr_statistical_confidence_1d *  1.8773 +
            -- window=1 inverted
            pr_worst_trade_1d_inv        *  0.7262 +
            pr_roi_1d_inv                *  1.1450 +
            pr_win_rate_1d_inv           *  9.6725 +
            -- window=3 regular
            pr_total_pnl_3d              *  4.0181 +
            pr_total_invested_3d         *  1.9045 +
            -- window=3 inverted
            pr_roi_3d_inv                *  1.8332 +
            pr_win_rate_3d_inv           *  1.4033 +
            -- window=7 regular
            pr_total_pnl_7d              *  1.8269 +
            pr_total_invested_7d         *  0.9005 +
            pr_best_trade_7d             *  6.1411 +
            pr_stddev_position_size_7d   *  2.9950 +
            pr_dominant_market_pnl_7d    *  0.1102 +
            pr_pnl_cat_other_7d          *  4.7032 +
            -- window=7 inverted
            pr_profit_factor_7d_inv      *  1.8667
        )::NUMERIC, 3) AS h_score

    FROM ranked
)

-- ── 6. final output: rank, tier, key metrics ─────────────────────────────────
SELECT
    RANK() OVER (ORDER BY h_score DESC)::INT AS rank,
    proxy_wallet,
    h_score,
    CASE
        WHEN h_score >= 70 THEN 'Elite'
        WHEN h_score >= 50 THEN 'Sharp'
        WHEN h_score >= 35 THEN 'Solid'
        ELSE                    'Emerging'
    END                              AS tier,
    total_invested,
    total_pnl,
    win_rate,
    worst_trade,
    total_trades,
    markets_traded
FROM  scored
ORDER BY h_score DESC
LIMIT 200;
