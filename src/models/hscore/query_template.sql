WITH params AS (
  SELECT
    COALESCE(NULLIF($$Value_{min_roi_15d}, '')::numeric, 0)            AS min_roi,
    COALESCE(NULLIF($$Value_{min_win_rate_15d}, '')::numeric, 0.45)    AS min_win_rate,
    COALESCE(NULLIF($$Value_{max_win_rate_15d}, '')::numeric, 0.95)    AS max_win_rate,
    COALESCE(NULLIF($$Value_{min_total_trades_15d}, '')::int, 50)      AS min_total_trades,
    COALESCE(NULLIF($$Value_{max_total_trades_15d}, '')::int, 20000)   AS max_total_trades,
    COALESCE(NULLIF($$Value_{min_pnl_15d}, '')::numeric, 5000)         AS min_pnl,
    COALESCE(NULLIF($$Value_{sort_by}, ''), 'h_score')                 AS sort_by
),

-- latest snapshot date
latest AS (
  SELECT MAX(date) AS snapshot_date
  FROM polymarket.wallet_profile_metrics_v2
  WHERE calculation_window_days = 15
),

-- medians for sparse ratio imputation (15d window only)
medians AS (
  SELECT
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY sortino_ratio)      AS med_sortino_ratio,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY calmar_ratio)       AS med_calmar_ratio,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY gain_to_pain_ratio) AS med_gain_to_pain_ratio,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY annualized_return)  AS med_annualized_return
  FROM polymarket.wallet_profile_metrics_v2
  CROSS JOIN latest
  CROSS JOIN params p
  WHERE date = latest.snapshot_date
    AND calculation_window_days = 15
    AND roi > p.min_roi
    AND win_rate BETWEEN p.min_win_rate AND p.max_win_rate
    AND total_trades BETWEEN p.min_total_trades AND p.max_total_trades
    AND total_pnl > p.min_pnl
    AND combined_risk_score <= 50
),

-- 1d window features
snap_1d AS (
  SELECT
    proxy_wallet,
    total_pnl              AS total_pnl_1d,
    worst_market_pnl            AS worst_market_pnl_1d,
    roi                    AS roi_1d,
    win_rate               AS win_rate_1d,
    statistical_confidence AS statistical_confidence_1d
  FROM polymarket.wallet_profile_metrics_v2
  CROSS JOIN latest
  WHERE date = latest.snapshot_date
    AND calculation_window_days = 1
),

-- 3d window features
snap_3d AS (
  SELECT
    proxy_wallet,
    total_pnl      AS total_pnl_3d,
    total_invested AS total_invested_3d,
    roi            AS roi_3d,
    win_rate       AS win_rate_3d
  FROM polymarket.wallet_profile_metrics_v2
  CROSS JOIN latest
  WHERE date = latest.snapshot_date
    AND calculation_window_days = 3
),

-- 7d window features
snap_7d AS (
  SELECT
    m7.proxy_wallet,
    m7.total_pnl            AS total_pnl_7d,
    m7.total_invested       AS total_invested_7d,
    m7.best_market_pnl           AS best_market_pnl_7d,
    m7.stddev_position_size AS stddev_position_size_7d,
    m7.dominant_market_pnl  AS dominant_market_pnl_7d,
    m7.profit_factor        AS profit_factor_7d,
    COALESCE(
      (SELECT SUM((elem->>'total_pnl')::numeric)
       FROM jsonb_array_elements(m7.performance_by_category) AS elem
       WHERE elem->>'category' = 'Other'), 0) AS pnl_cat_other_7d
  FROM polymarket.wallet_profile_metrics_v2 m7
  CROSS JOIN latest
  WHERE m7.date = latest.snapshot_date
    AND m7.calculation_window_days = 7
),

-- eligible wallets (15d base, same filter as original)
eligible AS (
  SELECT
    m15.proxy_wallet,
    m15.total_pnl,
    m15.total_invested,
    m15.best_market_pnl,
    m15.worst_market_pnl,
    m15.win_rate,
    m15.roi,
    m15.sharpe_ratio,
    m15.performance_trend,
    m15.total_trades,
    m15.markets_traded,

    COALESCE(m15.avg_market_exposure, 0)          AS avg_market_exposure,
    COALESCE(m15.dominant_market_pnl, 0)        AS dominant_market_pnl,
    COALESCE(m15.statistical_confidence, 0)     AS statistical_confidence,
    COALESCE(m15.market_concentration_ratio, 0) AS market_concentration_ratio,
    COALESCE(m15.profit_factor, 0)              AS profit_factor,
    COALESCE(m15.stddev_position_size, 0)       AS stddev_position_size,

    COALESCE(m15.sortino_ratio,      med.med_sortino_ratio)      AS sortino_ratio,
    COALESCE(m15.calmar_ratio,       med.med_calmar_ratio)       AS calmar_ratio,
    COALESCE(m15.gain_to_pain_ratio, med.med_gain_to_pain_ratio) AS gain_to_pain_ratio,
    COALESCE(m15.annualized_return,  med.med_annualized_return)  AS annualized_return,

    -- category PnL from 15d JSONB
    COALESCE(
      (SELECT SUM((elem->>'total_pnl')::numeric)
       FROM jsonb_array_elements(m15.performance_by_category) AS elem
       WHERE elem->>'category' = 'Sports'), 0)    AS pnl_cat_sports,
    COALESCE(
      (SELECT SUM((elem->>'total_pnl')::numeric)
       FROM jsonb_array_elements(m15.performance_by_category) AS elem
       WHERE elem->>'category' = 'Economics'), 0) AS pnl_cat_economics,
    COALESCE(
      (SELECT SUM((elem->>'total_pnl')::numeric)
       FROM jsonb_array_elements(m15.performance_by_category) AS elem
       WHERE elem->>'category' = 'Crypto'), 0)    AS pnl_cat_crypto,
    COALESCE(
      (SELECT SUM((elem->>'total_pnl')::numeric)
       FROM jsonb_array_elements(m15.performance_by_category) AS elem
       WHERE elem->>'category' = 'Other'), 0)     AS pnl_cat_other,

    -- multi-window features (LEFT JOIN, 0 if wallet absent in shorter window)
    COALESCE(s1.total_pnl_1d,              0) AS total_pnl_1d,
    COALESCE(s1.worst_market_pnl_1d,            0) AS worst_market_pnl_1d,
    COALESCE(s1.roi_1d,                    0) AS roi_1d,
    COALESCE(s1.win_rate_1d,               0) AS win_rate_1d,
    COALESCE(s1.statistical_confidence_1d, 0) AS statistical_confidence_1d,

    COALESCE(s3.total_pnl_3d,              0) AS total_pnl_3d,
    COALESCE(s3.total_invested_3d,         0) AS total_invested_3d,
    COALESCE(s3.roi_3d,                    0) AS roi_3d,
    COALESCE(s3.win_rate_3d,               0) AS win_rate_3d,

    COALESCE(s7.total_pnl_7d,              0) AS total_pnl_7d,
    COALESCE(s7.total_invested_7d,         0) AS total_invested_7d,
    COALESCE(s7.best_market_pnl_7d,             0) AS best_market_pnl_7d,
    COALESCE(s7.stddev_position_size_7d,   0) AS stddev_position_size_7d,
    COALESCE(s7.dominant_market_pnl_7d,    0) AS dominant_market_pnl_7d,
    COALESCE(s7.profit_factor_7d,          0) AS profit_factor_7d,
    COALESCE(s7.pnl_cat_other_7d,          0) AS pnl_cat_other_7d

  FROM polymarket.wallet_profile_metrics_v2 m15
  CROSS JOIN latest
  CROSS JOIN medians med
  CROSS JOIN params p
  LEFT JOIN snap_1d s1 ON s1.proxy_wallet = m15.proxy_wallet
  LEFT JOIN snap_3d s3 ON s3.proxy_wallet = m15.proxy_wallet
  LEFT JOIN snap_7d s7 ON s7.proxy_wallet = m15.proxy_wallet

  WHERE m15.date = latest.snapshot_date
    AND m15.calculation_window_days = 15
    AND m15.roi > p.min_roi
    AND m15.win_rate BETWEEN p.min_win_rate AND p.max_win_rate
    AND m15.total_trades BETWEEN p.min_total_trades AND p.max_total_trades
    AND m15.total_pnl > p.min_pnl
    AND m15.combined_risk_score <= 50
    AND (m15.total_pnl / NULLIF(m15.total_trades, 0)) > 10
),

-- percentile ranking
ranked AS (
  SELECT
    *,
    -- standard (higher = better)
    PERCENT_RANK() OVER (ORDER BY total_pnl ASC)                  AS pr_total_pnl,
    PERCENT_RANK() OVER (ORDER BY total_pnl_1d ASC)               AS pr_total_pnl_1d,
    PERCENT_RANK() OVER (ORDER BY total_pnl_3d ASC)               AS pr_total_pnl_3d,
    PERCENT_RANK() OVER (ORDER BY total_pnl_7d ASC)               AS pr_total_pnl_7d,
    PERCENT_RANK() OVER (ORDER BY total_invested ASC)             AS pr_total_invested,
    PERCENT_RANK() OVER (ORDER BY total_invested_3d ASC)          AS pr_total_invested_3d,
    PERCENT_RANK() OVER (ORDER BY total_invested_7d ASC)          AS pr_total_invested_7d,
    PERCENT_RANK() OVER (ORDER BY best_market_pnl ASC)                 AS pr_best_trade,
    PERCENT_RANK() OVER (ORDER BY best_market_pnl_7d ASC)              AS pr_best_trade_7d,
    PERCENT_RANK() OVER (ORDER BY avg_market_exposure ASC)          AS pr_avg_position_size,
    PERCENT_RANK() OVER (ORDER BY stddev_position_size_7d ASC)    AS pr_stddev_position_size_7d,
    PERCENT_RANK() OVER (ORDER BY dominant_market_pnl ASC)        AS pr_dominant_market_pnl,
    PERCENT_RANK() OVER (ORDER BY dominant_market_pnl_7d ASC)     AS pr_dominant_market_pnl_7d,
    PERCENT_RANK() OVER (ORDER BY pnl_cat_sports ASC)             AS pr_pnl_cat_sports,
    PERCENT_RANK() OVER (ORDER BY pnl_cat_other ASC)              AS pr_pnl_cat_other,
    PERCENT_RANK() OVER (ORDER BY pnl_cat_other_7d ASC)           AS pr_pnl_cat_other_7d,
    PERCENT_RANK() OVER (ORDER BY statistical_confidence ASC)     AS pr_statistical_confidence,
    PERCENT_RANK() OVER (ORDER BY statistical_confidence_1d ASC)  AS pr_statistical_confidence_1d,
    PERCENT_RANK() OVER (ORDER BY markets_traded ASC)             AS pr_markets_traded,
    PERCENT_RANK() OVER (ORDER BY pnl_cat_economics ASC)          AS pr_pnl_cat_economics,
    PERCENT_RANK() OVER (ORDER BY sortino_ratio ASC)              AS pr_sortino_ratio,
    PERCENT_RANK() OVER (ORDER BY calmar_ratio ASC)               AS pr_calmar_ratio,
    PERCENT_RANK() OVER (ORDER BY gain_to_pain_ratio ASC)         AS pr_gain_to_pain_ratio,
    PERCENT_RANK() OVER (ORDER BY annualized_return ASC)          AS pr_annualized_return,
    PERCENT_RANK() OVER (ORDER BY total_trades ASC)               AS pr_total_trades,
    -- inverted (lower = better, DESC so higher rank = better)
    PERCENT_RANK() OVER (ORDER BY worst_market_pnl DESC)               AS pr_worst_trade_inv,
    PERCENT_RANK() OVER (ORDER BY worst_market_pnl_1d DESC)            AS pr_worst_trade_1d_inv,
    PERCENT_RANK() OVER (ORDER BY roi_1d DESC)                    AS pr_roi_1d_inv,
    PERCENT_RANK() OVER (ORDER BY roi_3d DESC)                    AS pr_roi_3d_inv,
    PERCENT_RANK() OVER (ORDER BY profit_factor DESC)             AS pr_profit_factor_inv,
    PERCENT_RANK() OVER (ORDER BY profit_factor_7d DESC)          AS pr_profit_factor_7d_inv,
    PERCENT_RANK() OVER (ORDER BY win_rate DESC)                  AS pr_win_rate_inv,
    PERCENT_RANK() OVER (ORDER BY win_rate_1d DESC)               AS pr_win_rate_1d_inv,
    PERCENT_RANK() OVER (ORDER BY win_rate_3d DESC)               AS pr_win_rate_3d_inv,
    PERCENT_RANK() OVER (ORDER BY market_concentration_ratio DESC) AS pr_market_conc_inv,
    PERCENT_RANK() OVER (ORDER BY pnl_cat_crypto DESC)            AS pr_pnl_cat_crypto_inv
  FROM eligible
),

-- scoring with new ML weights
scored AS (
  SELECT
    *,
    ROUND((
      pr_total_pnl                *                   8.496810 +
      pr_total_pnl_1d             *                6.545811 +
      pr_total_pnl_3d             *                2.120489 +
      pr_total_pnl_7d             *                3.990123 +
      pr_total_invested           *              3.649548 +
      pr_total_invested_3d        *           1.586478 +
      pr_total_invested_7d        *           0.255536 +
      pr_best_trade               *                  3.057462 +
      pr_best_trade_7d            *               0.247587 +
      pr_avg_position_size        *           4.182631 +
      pr_stddev_position_size_7d  *     1.462315 +
      pr_dominant_market_pnl      *         0.000000 +
      pr_dominant_market_pnl_7d   *      0.000003 +
      pr_pnl_cat_sports           *              1.533078 +
      pr_pnl_cat_other            *               0.870362 +
      pr_pnl_cat_other_7d         *            0.784781 +
      pr_statistical_confidence   *      3.157165 +
      pr_statistical_confidence_1d*   1.846289 +
      pr_markets_traded           *              2.956671 +
      pr_pnl_cat_economics        *           6.851369 +
      pr_worst_trade_inv          *             2.998568 +
      pr_worst_trade_1d_inv       *          4.872509 +
      pr_roi_1d_inv               *                  1.951865 +
      pr_roi_3d_inv               *                  8.339105 +
      pr_profit_factor_inv        *           2.590872 +
      pr_profit_factor_7d_inv     *        0.000000 +
      pr_win_rate_inv             *                2.036701 +
      pr_win_rate_1d_inv          *             1.741325 +
      pr_win_rate_3d_inv          *             0.285943 +
      pr_market_conc_inv          *             1.143423 +
      pr_pnl_cat_crypto_inv       *          1.076091 +
      pr_sortino_ratio            *               2.027843 +
      pr_calmar_ratio             *                1.901358 +
      pr_gain_to_pain_ratio       *          6.715465 +
      pr_annualized_return        *           0.000002 +
      pr_total_trades             *                8.724423
    )::numeric, 3) AS h_score
  FROM ranked
),

-- final output - identical schema to original query
final AS (
  SELECT
    RANK() OVER (ORDER BY h_score DESC) AS leaderboard_rank,
    proxy_wallet                         AS wallet,

    CASE
      WHEN h_score >= 70 THEN 'Elite'
      WHEN h_score >= 50 THEN 'Sharp'
      WHEN h_score >= 35 THEN 'Solid'
      ELSE 'Emerging'
    END AS tier,

    h_score,
    ROUND(roi::numeric, 1)              AS roi_pct_15d,
    ROUND((win_rate * 100)::numeric, 1) AS win_rate_pct_15d,
    ROUND(sharpe_ratio::numeric, 2)     AS sharpe_ratio_15d,
    total_trades                        AS total_trades_15d,
    markets_traded                      AS markets_traded_15d,
    ROUND(total_pnl::numeric, 2)        AS total_pnl_15d,
    ROUND(total_invested::numeric, 2)   AS total_volume_15d,
    performance_trend                   AS trajectory,

    params.sort_by
  FROM scored
  CROSS JOIN params
)

SELECT
  leaderboard_rank,
  wallet,
  tier,
  h_score,
  roi_pct_15d,
  win_rate_pct_15d,
  sharpe_ratio_15d,
  total_trades_15d,
  markets_traded_15d,
  total_pnl_15d,
  total_volume_15d,
  trajectory
FROM final
ORDER BY
  CASE sort_by
    WHEN 'h_score'  THEN h_score
    WHEN 'roi'      THEN roi_pct_15d
    WHEN 'pnl'      THEN total_pnl_15d
    WHEN 'win_rate' THEN win_rate_pct_15d
    WHEN 'trades'   THEN total_trades_15d::numeric
    WHEN 'sharpe'   THEN sharpe_ratio_15d
    ELSE h_score
  END DESC
LIMIT 200;
