-- 598 - Sport sharp universe (unified definition)
--
-- Returns the wallet list considered "sharp on sports" at the latest snapshot,
-- with their tier from the trained Sport H-Score leaderboard (595).
--
-- This is the SINGLE source of truth for "is this wallet sharp." Position-finder
-- agents (591/593/594) consume this instead of redefining sharp filters locally.
--
-- Inputs
--   min_tier  ('Elite' | 'Sharp' | 'Solid' | 'Emerging')  default 'Sharp'
--   sport     (substring matched against Sports / X / Y categories)  default ''
--             '' means any sport-active wallet.
--
-- Output cols: wallet, tier, sport_h_score, sports_pnl_15d, sports_trades_15d,
--              dominant_sport, win_rate_pct_15d, combined_risk_score
--
-- Tables: wallet_profile_metrics_v2 (window=15), wallet_profile_metrics_category_v2.
-- No polymarket_trade access. Bounded to the latest snapshot date only.

WITH params AS (
    SELECT
        COALESCE(NULLIF($$Value_{min_tier}, ''), 'Sharp')  AS min_tier,
        COALESCE(NULLIF($$Value_{sport}, ''), '')          AS sport
),

tier_floor AS (
    SELECT CASE (SELECT min_tier FROM params)
        WHEN 'Elite'    THEN 70.0
        WHEN 'Sharp'    THEN 50.0
        WHEN 'Solid'    THEN 35.0
        WHEN 'Emerging' THEN 0.0
        ELSE 50.0
    END AS floor_score
),

latest_v2 AS (
    SELECT MAX(date) AS d FROM polymarket.wallet_profile_metrics_v2 WHERE calculation_window_days = 15
),

latest_cat AS (
    SELECT MAX(date) AS d FROM polymarket.wallet_profile_metrics_category_v2 WHERE calculation_window_days = 15
),

base_raw AS (
    SELECT
        g.proxy_wallet,
        g.total_pnl,
        g.best_market_pnl,
        g.worst_market_pnl,
        g.avg_market_exposure,
        g.profitable_markets_count,
        g.sortino_ratio,
        g.annualized_return,
        g.win_rate,
        g.combined_risk_score,
        g.total_trades,
        g.markets_traded,
        g.total_invested
    FROM polymarket.wallet_profile_metrics_v2 g, latest_v2
    WHERE g.date = latest_v2.d
      AND g.calculation_window_days = 15
      AND g.sybil_risk_flag = false
      AND g.suspicious_win_rate_flag = false
      AND g.combined_risk_score <= 50
      AND g.total_pnl > 1000
      AND g.total_trades BETWEEN 10 AND 200000
      AND g.win_rate BETWEEN 0 AND 0.95
      AND g.roi > 0
),

medians AS (
    SELECT
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY sortino_ratio)     AS med_sortino,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY annualized_return) AS med_annret
    FROM base_raw
),

base AS (
    SELECT
        b.proxy_wallet,
        b.total_pnl,
        b.best_market_pnl,
        b.worst_market_pnl,
        b.avg_market_exposure,
        b.profitable_markets_count,
        COALESCE(b.sortino_ratio,     m.med_sortino) AS sortino_ratio,
        COALESCE(b.annualized_return, m.med_annret)  AS annualized_return,
        b.win_rate,
        b.combined_risk_score,
        b.total_trades,
        b.markets_traded,
        b.total_invested
    FROM base_raw b
    CROSS JOIN medians m
),

sport_15d AS (
    SELECT
        c.proxy_wallet,
        COALESCE(MAX(CASE WHEN c.category = 'Sports' THEN c.total_pnl END),
                 SUM(CASE WHEN c.category LIKE 'Sports / %' THEN c.total_pnl ELSE 0 END))  AS sports_pnl,
        COALESCE(MAX(CASE WHEN c.category = 'Sports' THEN c.total_trades END),
                 SUM(CASE WHEN c.category LIKE 'Sports / %' THEN c.total_trades ELSE 0 END))::int AS sports_trades,
        COALESCE(MAX(CASE WHEN c.category = 'Sports' THEN c.total_invested END),
                 SUM(CASE WHEN c.category LIKE 'Sports / %' THEN c.total_invested ELSE 0 END)) AS sports_invested,
        COALESCE(MAX(CASE WHEN c.category = 'Sports' THEN c.win_rate END), 0)              AS sport_win_rate,
        (
            SELECT cc.category
            FROM polymarket.wallet_profile_metrics_category_v2 cc, latest_cat
            WHERE cc.proxy_wallet = c.proxy_wallet
              AND cc.calculation_window_days = 15
              AND cc.date = latest_cat.d
              AND cc.category LIKE 'Sports / % / %'
              AND cc.total_pnl > 0
            ORDER BY cc.total_pnl DESC
            LIMIT 1
        ) AS dominant_sport
    FROM polymarket.wallet_profile_metrics_category_v2 c, latest_cat
    WHERE c.calculation_window_days = 15
      AND c.date = latest_cat.d
      AND c.category LIKE 'Sports%'
      AND c.proxy_wallet IN (SELECT proxy_wallet FROM base)
    GROUP BY c.proxy_wallet
    HAVING COALESCE(MAX(CASE WHEN c.category = 'Sports' THEN c.total_pnl END),
                    SUM(CASE WHEN c.category LIKE 'Sports / %' THEN c.total_pnl ELSE 0 END)) > 0
),

eligible AS (
    SELECT
        b.*,
        s.sports_pnl,
        s.sports_trades,
        s.sports_invested,
        s.sport_win_rate,
        s.dominant_sport
    FROM base b
    JOIN sport_15d s ON s.proxy_wallet = b.proxy_wallet
),

scored AS (
    SELECT
        proxy_wallet,
        sports_pnl,
        sports_trades,
        sport_win_rate,
        dominant_sport,
        win_rate,
        combined_risk_score,
        -- NOTE: keep this in sync with the 595 leaderboard formula and the
        --       optimal_weights.json artifact. After every retrain of
        --       sport_hscore, regenerate this expression with the new weights.
        ROUND((
            PERCENT_RANK() OVER (ORDER BY total_pnl ASC)                * 11.907 +
            PERCENT_RANK() OVER (ORDER BY best_market_pnl ASC)          *  9.646 +
            PERCENT_RANK() OVER (ORDER BY annualized_return ASC)        *  5.949 +
            PERCENT_RANK() OVER (ORDER BY profitable_markets_count ASC) *  5.770 +
            PERCENT_RANK() OVER (ORDER BY avg_market_exposure ASC)      *  5.682 +
            PERCENT_RANK() OVER (ORDER BY sports_trades ASC)            *  7.153 +
            PERCENT_RANK() OVER (ORDER BY sports_invested ASC)          *  3.024 +
            PERCENT_RANK() OVER (ORDER BY sports_pnl ASC)               *  1.809 +
            PERCENT_RANK() OVER (ORDER BY worst_market_pnl DESC)        *  2.038 +
            PERCENT_RANK() OVER (ORDER BY sortino_ratio ASC)            *  1.820
        )::numeric, 1) AS sport_h_score
    FROM eligible
)

SELECT
    s.proxy_wallet                                                     AS wallet,
    CASE
        WHEN s.sport_h_score >= 70 THEN 'Elite'
        WHEN s.sport_h_score >= 50 THEN 'Sharp'
        WHEN s.sport_h_score >= 35 THEN 'Solid'
        ELSE 'Emerging'
    END                                                                AS tier,
    s.sport_h_score,
    ROUND(s.sports_pnl::numeric, 2)                                    AS sports_pnl_15d,
    s.sports_trades                                                    AS sports_trades_15d,
    ROUND((s.sport_win_rate * 100)::numeric, 1)                        AS win_rate_pct_15d,
    s.dominant_sport,
    ROUND(s.combined_risk_score::numeric, 1)                           AS combined_risk_score
FROM scored s
CROSS JOIN tier_floor tf
CROSS JOIN params p
WHERE s.sport_h_score >= tf.floor_score
  AND (p.sport = '' OR s.dominant_sport ILIKE '%' || p.sport || '%')
ORDER BY s.sport_h_score DESC;
