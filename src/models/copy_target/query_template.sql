WITH params AS (
  SELECT
    COALESCE(NULLIF($$Value_{wallet_addresses}, ''), '')              AS wallets_csv,
    NULLIF($$Value_{snapshot_date}, '')::date                         AS requested_date,
    COALESCE(NULLIF($$Value_{trajectory_days}, '')::int, 60)          AS trajectory_days,
    COALESCE(NULLIF($$Value_{top_n}, '')::int, 50)                    AS top_n,
    COALESCE(NULLIF($$Value_{min_composite_score}, '')::numeric, 0)   AS min_composite_score
),

-- latest available snapshot, or the user-specified date if provided
latest AS (
  SELECT COALESCE(
    (SELECT requested_date FROM params WHERE requested_date IS NOT NULL),
    (SELECT MAX(date) FROM polymarket.wallet_profile_metrics_v2 WHERE calculation_window_days = 15)
  ) AS snapshot_date
),

-- parsed wallet input list (empty = score the whole universe)
wallet_input AS (
  SELECT DISTINCT TRIM(unnest(string_to_array(wallets_csv, ','))) AS proxy_wallet
  FROM params
  WHERE wallets_csv <> ''
),

-- eligible universe at the 15d snapshot (same filters as H-Score)
scoring_universe AS (
  SELECT
    m15.proxy_wallet,
    m15.total_pnl,
    m15.total_invested,
    m15.total_trades,
    m15.markets_traded,
    m15.days_active,
    m15.roi,
    m15.win_rate,
    m15.win_rate_z_score,
    m15.avg_trade_size,
    m15.coefficient_of_variation,
    m15.position_size_consistency,
    m15.statistical_confidence,
    m15.profitable_markets_count,
    m15.performance_trend,
    m15.combined_risk_score,
    m15.sybil_risk_score,
    m15.similar_wallets_count,
    m15.single_market_dependence_flag,
    m15.max_drawdown,
    m15.ulcer_index,
    m15.recovery_time_avg,
    m15.sortino_ratio,
    m15.calmar_ratio,
    m15.gain_to_pain_ratio,
    m15.annualized_return,
    m15.performance_by_category
  FROM polymarket.wallet_profile_metrics_v2 m15
  CROSS JOIN latest
  WHERE m15.date = latest.snapshot_date
    AND m15.calculation_window_days = 15
    AND m15.roi > 0
    AND m15.win_rate BETWEEN 0.45 AND 0.95
    AND m15.total_trades BETWEEN 50 AND 100000
    AND m15.total_pnl > 5000
    AND m15.combined_risk_score <= 50
),

-- specialization: top-level category Herfindahl + dominant subcategory
specialization AS (
  SELECT
    u.proxy_wallet,
    COALESCE((
      SELECT SUM(POWER((e->>'total_pnl')::numeric, 2))
             / NULLIF(POWER(SUM((e->>'total_pnl')::numeric), 2), 0)
      FROM jsonb_array_elements(u.performance_by_category) e
      WHERE position(' / ' in (e->>'category')) = 0
        AND (e->>'total_pnl')::numeric > 0
    ), 0) AS category_hhi,
    (SELECT e->>'category'
     FROM jsonb_array_elements(u.performance_by_category) e
     WHERE position(' / ' in (e->>'category')) = 0
     ORDER BY (e->>'total_pnl')::numeric DESC NULLS LAST
     LIMIT 1) AS dominant_category,
    (SELECT e->>'category'
     FROM jsonb_array_elements(u.performance_by_category) e
     WHERE position(' / ' in (e->>'category')) > 0
     ORDER BY (e->>'total_pnl')::numeric DESC NULLS LAST
     LIMIT 1) AS dominant_subcategory,
    COALESCE((
      SELECT MAX((e->>'total_pnl')::numeric) / NULLIF(u.total_pnl, 0)
      FROM jsonb_array_elements(u.performance_by_category) e
      WHERE position(' / ' in (e->>'category')) = 0
        AND (e->>'total_pnl')::numeric > 0
    ), 0) AS top1_category_share
  FROM scoring_universe u
),

-- within-universe medians for sparse ratio imputation
medians AS (
  SELECT
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY sortino_ratio)      AS med_sortino,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY calmar_ratio)       AS med_calmar,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY gain_to_pain_ratio) AS med_g2p,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY annualized_return)  AS med_ar
  FROM scoring_universe
),

-- impute + derive features (all from v2 only, no daily_pnl scan)
enriched AS (
  SELECT
    u.*,
    s.category_hhi,
    s.dominant_category,
    s.dominant_subcategory,
    s.top1_category_share,
    COALESCE(u.profitable_markets_count::numeric / NULLIF(u.markets_traded, 0), 0) AS profitable_market_rate,
    u.total_trades::numeric / 15.0 * 7                                              AS trades_per_week,
    COALESCE(u.sortino_ratio, m.med_sortino)                                        AS sortino_ratio_imp,
    COALESCE(u.calmar_ratio, m.med_calmar)                                          AS calmar_ratio_imp,
    COALESCE(u.gain_to_pain_ratio, m.med_g2p)                                       AS g2p_imp,
    COALESCE(u.annualized_return, m.med_ar)                                         AS ar_imp,
    CASE WHEN u.performance_trend = 'improving' THEN 1.0
         WHEN u.performance_trend = 'stable'    THEN 0.5
         ELSE 0.0 END                                                               AS perf_trend_norm
  FROM scoring_universe u
  LEFT JOIN specialization s ON s.proxy_wallet = u.proxy_wallet
  CROSS JOIN medians m
),

-- normalize each component via PERCENT_RANK within the universe
normalized AS (
  SELECT
    e.proxy_wallet,
    -- skill components
    PERCENT_RANK() OVER (ORDER BY profitable_market_rate ASC)   AS pr_pmr,
    PERCENT_RANK() OVER (ORDER BY statistical_confidence ASC)   AS pr_statconf,
    PERCENT_RANK() OVER (ORDER BY g2p_imp ASC)                  AS pr_g2p,
    PERCENT_RANK() OVER (ORDER BY sortino_ratio_imp ASC)        AS pr_sortino,
    PERCENT_RANK() OVER (ORDER BY ar_imp ASC)                   AS pr_ar,
    PERCENT_RANK() OVER (ORDER BY win_rate_z_score ASC)         AS pr_wr_z,
    perf_trend_norm                                             AS pr_trend,
    -- specialization
    PERCENT_RANK() OVER (ORDER BY category_hhi ASC)             AS pr_hhi,
    PERCENT_RANK() OVER (ORDER BY top1_category_share ASC)      AS pr_top1share,
    -- copyability (smaller raw values mean more retail-friendly, so DESC)
    PERCENT_RANK() OVER (ORDER BY avg_trade_size DESC)          AS pr_size_retail,
    PERCENT_RANK() OVER (ORDER BY trades_per_week DESC)         AS pr_freq_inv,
    PERCENT_RANK() OVER (ORDER BY position_size_consistency ASC) AS pr_psc,
    PERCENT_RANK() OVER (ORDER BY coefficient_of_variation DESC) AS pr_cv_inv,
    CASE WHEN single_market_dependence_flag THEN 0 ELSE 1 END   AS smd_ok,
    -- risk (raw "higher = worse" needs DESC inversion to make "higher = safer")
    PERCENT_RANK() OVER (ORDER BY combined_risk_score DESC)     AS pr_risk_inv,
    PERCENT_RANK() OVER (ORDER BY sybil_risk_score DESC)        AS pr_sybil_inv,
    PERCENT_RANK() OVER (ORDER BY similar_wallets_count DESC)   AS pr_simwall_inv,
    -- max_drawdown is negative, ASC means "less negative" -> higher rank -> safer
    PERCENT_RANK() OVER (ORDER BY max_drawdown ASC)             AS pr_dd,
    PERCENT_RANK() OVER (ORDER BY ulcer_index DESC)             AS pr_ulcer_inv,
    PERCENT_RANK() OVER (ORDER BY recovery_time_avg DESC)       AS pr_recovery_inv
  FROM enriched e
),

-- sub-scores 0..100 (no trajectory dependency)
sub_scores AS (
  SELECT
    n.proxy_wallet,
    ROUND((
      n.pr_pmr      * 25 +
      n.pr_statconf * 20 +
      n.pr_g2p      * 15 +
      n.pr_sortino  * 10 +
      n.pr_ar       * 10 +
      n.pr_wr_z     * 10 +
      n.pr_trend    * 10
    )::numeric, 2) AS skill_score,

    ROUND((
      n.pr_hhi       * 70 +
      n.pr_top1share * 30
    )::numeric, 2) AS specialization_score,

    ROUND((
      n.pr_size_retail * 30 +
      n.pr_freq_inv    * 20 +
      n.pr_psc         * 20 +
      n.pr_cv_inv      * 15 +
      n.smd_ok         * 15
    )::numeric, 2) AS copyability_score,

    ROUND((
      n.pr_risk_inv     * 30 +
      n.pr_sybil_inv    * 25 +
      n.pr_simwall_inv  * 10 +
      n.pr_dd           * 15 +
      n.pr_ulcer_inv    * 10 +
      n.pr_recovery_inv * 10
    )::numeric, 2) AS risk_score
  FROM normalized n
),

-- composite score + tier label (universe-wide)
composite AS (
  SELECT
    e.proxy_wallet,
    ss.skill_score,
    ss.specialization_score,
    ss.copyability_score,
    ss.risk_score,
    ROUND((
      ss.skill_score          * 0.35 +
      ss.specialization_score * 0.15 +
      ss.copyability_score    * 0.20 +
      ss.risk_score           * 0.30
    )::numeric, 2) AS composite_score,
    e.dominant_category,
    e.dominant_subcategory,
    e.category_hhi,
    e.profitable_market_rate,
    e.avg_trade_size,
    e.trades_per_week,
    e.max_drawdown,
    e.sybil_risk_score,
    e.similar_wallets_count,
    e.total_pnl,
    e.total_trades,
    e.win_rate,
    e.roi
  FROM sub_scores ss
  JOIN enriched e USING (proxy_wallet)
),

-- narrow down to the output set BEFORE pulling trajectory
output_wallets AS (
  SELECT c.*
  FROM composite c
  LEFT JOIN wallet_input wi ON wi.proxy_wallet = c.proxy_wallet
  CROSS JOIN params p
  WHERE (p.wallets_csv = '' OR wi.proxy_wallet IS NOT NULL)
    AND c.composite_score >= p.min_composite_score
  ORDER BY c.composite_score DESC
  LIMIT (SELECT CASE WHEN wallets_csv = '' THEN top_n ELSE 10000 END FROM params)
),

-- trajectory ONLY for the output wallets: bounded scan on wallet_daily_pnl.
-- First aggregate to one row per (wallet, date), then compute trajectory stats.
trajectory_daily AS (
  SELECT
    d.proxy_wallet,
    d.date,
    SUM(d.pnl) AS day_pnl
  FROM polymarket.wallet_daily_pnl d
  CROSS JOIN latest
  CROSS JOIN params p
  WHERE d.date BETWEEN (latest.snapshot_date - p.trajectory_days * INTERVAL '1 day')::date
                   AND latest.snapshot_date
    AND d.proxy_wallet = ANY(ARRAY(SELECT proxy_wallet FROM output_wallets))
  GROUP BY d.proxy_wallet, d.date
),
trajectory AS (
  SELECT
    proxy_wallet,
    SUM(day_pnl)                                          AS trailing_pnl,
    COUNT(*)                                              AS active_days,
    SUM(CASE WHEN day_pnl > 0 THEN 1 ELSE 0 END)::numeric
      / NULLIF(COUNT(*), 0)                               AS positive_days_pct
  FROM trajectory_daily
  GROUP BY proxy_wallet
)

SELECT
  o.proxy_wallet AS wallet,
  CASE
    WHEN o.composite_score >= 75 THEN 'Strong Copy'
    WHEN o.composite_score >= 50 THEN 'Watch'
    ELSE 'Skip'
  END AS tier,
  o.composite_score,
  o.skill_score,
  o.specialization_score,
  o.copyability_score,
  o.risk_score,
  o.dominant_category,
  o.dominant_subcategory,
  ROUND(o.category_hhi::numeric, 3)                  AS category_hhi,
  ROUND(o.profitable_market_rate::numeric, 3)        AS profitable_market_rate,
  ROUND(o.avg_trade_size::numeric, 2)                AS avg_trade_size,
  ROUND(o.trades_per_week::numeric, 1)               AS trades_per_week,
  ROUND(o.max_drawdown::numeric, 2)                  AS max_drawdown,
  ROUND(o.sybil_risk_score::numeric, 1)              AS sybil_risk_score,
  o.similar_wallets_count,
  ROUND(COALESCE(t.trailing_pnl, 0)::numeric, 2)     AS trailing_pnl,
  COALESCE(t.active_days, 0)                         AS active_days,
  ROUND((COALESCE(t.positive_days_pct, 0) * 100)::numeric, 1) AS positive_days_pct,
  ROUND(o.total_pnl::numeric, 2)                     AS total_pnl_15d,
  o.total_trades                                     AS total_trades_15d,
  ROUND((o.win_rate * 100)::numeric, 1)              AS win_rate_pct_15d,
  ROUND(o.roi::numeric, 2)                           AS roi_15d,
  CONCAT_WS(' | ',
    CASE WHEN o.skill_score >= 70                THEN CONCAT('skill=', o.skill_score) END,
    CASE WHEN o.specialization_score >= 70       THEN CONCAT('specialist(', o.dominant_category, ')') END,
    CASE WHEN o.specialization_score < 30        THEN 'generalist' END,
    CASE WHEN o.copyability_score < 30           THEN 'hard-to-copy size or frequency' END,
    CASE WHEN o.risk_score < 30                  THEN 'risk-flagged' END,
    CASE WHEN o.sybil_risk_score > 30            THEN 'sybil concern' END,
    CASE WHEN o.similar_wallets_count > 5        THEN 'wallet cluster' END,
    CASE WHEN COALESCE(t.positive_days_pct, 1) < 0.45  THEN 'spiky pnl' END
  ) AS rationale
FROM output_wallets o
LEFT JOIN trajectory t ON t.proxy_wallet = o.proxy_wallet
ORDER BY o.composite_score DESC;
