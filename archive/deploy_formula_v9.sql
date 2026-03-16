-- deploy_formula_v9.sql
--
-- H-Score formula — feature set v9 (Precision@25 objective).
-- Weights from optimal_weights_v9.json (scipy SLSQP, 50 random inits,
-- objective: mean Precision@25 across snapshot dates).
--
-- Key changes from v8:
--   FEATURES   : 41 total (37 from v8 + pnl_per_trade x4 windows)
--   NEW FEATS  : pnl_per_trade    = total_pnl / total_trades  (15d)
--                pnl_per_trade_1d = total_pnl_1d / total_trades_1d
--                pnl_per_trade_3d = total_pnl_3d / total_trades_3d
--                pnl_per_trade_7d = total_pnl_7d / total_trades_7d
--   All pnl_per_trade features are regular (higher = better, ASC rank).
--   0 where trades = 0 in that window (NULLIF safe division → COALESCE 0).
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
-- pnl_per_trade computed via NULLIF safe division; 0 where trades = 0.
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

        -- ── window=15 pnl_per_trade ───────────────────────────────────────
        COALESCE(m15.total_pnl / NULLIF(m15.total_trades, 0), 0.0)           AS pnl_per_trade,

        -- ── window=1 features ─────────────────────────────────────────────
        COALESCE(m1.total_pnl,               0)                               AS total_pnl_1d,
        COALESCE(m1.statistical_confidence,  0)                               AS statistical_confidence_1d,
        COALESCE(m1.worst_trade,             0)                               AS worst_trade_1d,
        COALESCE(m1.roi,                     0)                               AS roi_1d,
        COALESCE(m1.win_rate,                0)                               AS win_rate_1d,
        COALESCE(m1.total_pnl / NULLIF(m1.total_trades, 0), 0.0)             AS pnl_per_trade_1d,

        -- ── window=3 features ─────────────────────────────────────────────
        COALESCE(m3.total_pnl,               0)                               AS total_pnl_3d,
        COALESCE(m3.total_invested,          0)                               AS total_invested_3d,
        COALESCE(m3.roi,                     0)                               AS roi_3d,
        COALESCE(m3.win_rate,                0)                               AS win_rate_3d,
        COALESCE(m3.total_pnl / NULLIF(m3.total_trades, 0), 0.0)             AS pnl_per_trade_3d,

        -- ── window=7 features ─────────────────────────────────────────────
        COALESCE(m7.total_pnl,               0)                               AS total_pnl_7d,
        COALESCE(m7.total_invested,          0)                               AS total_invested_7d,
        COALESCE(m7.best_trade,              0)                               AS best_trade_7d,
        COALESCE(m7.stddev_position_size,    0)                               AS stddev_position_size_7d,
        COALESCE(m7.dominant_market_pnl,     0)                               AS dominant_market_pnl_7d,
        COALESCE(m7.profit_factor,           0)                               AS profit_factor_7d,
        COALESCE(m7.total_pnl / NULLIF(m7.total_trades, 0), 0.0)             AS pnl_per_trade_7d,
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
        PERCENT_RANK() OVER (ORDER BY COALESCE(pnl_per_trade,           0) ASC) AS pr_pnl_per_trade,

        -- ── window=15 inverted (DESC) ─────────────────────────────────────
        PERCENT_RANK() OVER (ORDER BY COALESCE(worst_trade,                0) DESC) AS pr_worst_trade_inv,
        PERCENT_RANK() OVER (ORDER BY COALESCE(profit_factor,              0) DESC) AS pr_profit_factor_inv,
        PERCENT_RANK() OVER (ORDER BY COALESCE(win_rate,                   0) DESC) AS pr_win_rate_inv,
        PERCENT_RANK() OVER (ORDER BY COALESCE(market_concentration_ratio, 0) DESC) AS pr_market_conc_inv,
        PERCENT_RANK() OVER (ORDER BY COALESCE(pnl_cat_crypto,             0) DESC) AS pr_pnl_cat_crypto_inv,

        -- ── window=1 regular (ASC) ────────────────────────────────────────
        PERCENT_RANK() OVER (ORDER BY COALESCE(total_pnl_1d,             0) ASC) AS pr_total_pnl_1d,
        PERCENT_RANK() OVER (ORDER BY COALESCE(statistical_confidence_1d, 0) ASC) AS pr_statistical_confidence_1d,
        PERCENT_RANK() OVER (ORDER BY COALESCE(pnl_per_trade_1d,         0) ASC) AS pr_pnl_per_trade_1d,

        -- ── window=1 inverted (DESC) ──────────────────────────────────────
        PERCENT_RANK() OVER (ORDER BY COALESCE(worst_trade_1d,            0) DESC) AS pr_worst_trade_1d_inv,
        PERCENT_RANK() OVER (ORDER BY COALESCE(roi_1d,                    0) DESC) AS pr_roi_1d_inv,
        PERCENT_RANK() OVER (ORDER BY COALESCE(win_rate_1d,               0) DESC) AS pr_win_rate_1d_inv,

        -- ── window=3 regular (ASC) ────────────────────────────────────────
        PERCENT_RANK() OVER (ORDER BY COALESCE(total_pnl_3d,             0) ASC) AS pr_total_pnl_3d,
        PERCENT_RANK() OVER (ORDER BY COALESCE(total_invested_3d,        0) ASC) AS pr_total_invested_3d,
        PERCENT_RANK() OVER (ORDER BY COALESCE(pnl_per_trade_3d,         0) ASC) AS pr_pnl_per_trade_3d,

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
        PERCENT_RANK() OVER (ORDER BY COALESCE(pnl_per_trade_7d,         0) ASC) AS pr_pnl_per_trade_7d,

        -- ── window=7 inverted (DESC) ──────────────────────────────────────
        PERCENT_RANK() OVER (ORDER BY COALESCE(profit_factor_7d,          0) DESC) AS pr_profit_factor_7d_inv

    FROM eligible
),

-- ── 5. weighted sum → h_score ────────────────────────────────────────────────
-- Weights from optimal_weights_v9.json; sum = 100.000.
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
            pr_total_pnl                  *  1.5648 +
            pr_total_invested             *  1.0247 +
            pr_best_trade                 *  3.1370 +
            pr_avg_position_size          *  2.5445 +
            pr_dominant_market_pnl        *  0.3177 +
            pr_pnl_cat_sports             *  8.6459 +
            pr_pnl_cat_other              *  3.2831 +
            pr_pnl_cat_economics          *  3.6507 +
            pr_perfect_entry_count        *  3.6693 +
            pr_statistical_confidence     *  1.8772 +
            pr_markets_traded             *  2.8417 +
            pr_sortino_ratio              *  2.2605 +
            pr_calmar_ratio               *  1.2903 +
            pr_gain_to_pain_ratio         *  2.3667 +
            pr_annualized_return          *  2.2407 +
            pr_total_trades               *  0.7722 +
            pr_pnl_per_trade              *  0.7541 +
            -- window=15 inverted
            pr_worst_trade_inv            *  0.0149 +
            pr_profit_factor_inv          *  0.3423 +
            pr_win_rate_inv               *  1.3615 +
            pr_market_conc_inv            *  1.2596 +
            pr_pnl_cat_crypto_inv         *  7.7882 +
            -- window=1 regular
            pr_total_pnl_1d              *  1.5125 +
            pr_statistical_confidence_1d *  3.9951 +
            pr_pnl_per_trade_1d          *  5.9374 +
            -- window=1 inverted
            pr_worst_trade_1d_inv        * 11.1479 +
            pr_roi_1d_inv                *  0.1231 +
            pr_win_rate_1d_inv           *  1.1405 +
            -- window=3 regular
            pr_total_pnl_3d              *  2.2963 +
            pr_total_invested_3d         *  0.0508 +
            pr_pnl_per_trade_3d          *  0.2061 +
            -- window=3 inverted
            pr_roi_3d_inv                *  1.2919 +
            pr_win_rate_3d_inv           *  7.4823 +
            -- window=7 regular
            pr_total_pnl_7d              *  1.1290 +
            pr_total_invested_7d         *  4.0549 +
            pr_best_trade_7d             *  1.3137 +
            pr_stddev_position_size_7d   *  1.3161 +
            pr_dominant_market_pnl_7d    *  0.4050 +
            pr_pnl_cat_other_7d          *  1.1572 +
            pr_pnl_per_trade_7d          *  1.8725 +
            -- window=7 inverted
            pr_profit_factor_7d_inv      *  0.5598
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
