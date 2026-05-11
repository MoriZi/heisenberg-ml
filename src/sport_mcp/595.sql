WITH params AS (
    SELECT
        COALESCE(NULLIF($$Value_{min_pnl_15d}, '')::numeric, 1000)         AS min_pnl,
        COALESCE(NULLIF($$Value_{min_total_trades_15d}, '')::int, 10)      AS min_total_trades,
        COALESCE(NULLIF($$Value_{max_total_trades_15d}, '')::int, 200000)  AS max_total_trades,
        COALESCE(NULLIF($$Value_{min_win_rate_15d}, '')::numeric, 0)       AS min_win_rate,
        COALESCE(NULLIF($$Value_{max_win_rate_15d}, '')::numeric, 0.95)    AS max_win_rate,
        COALESCE(NULLIF($$Value_{min_roi_15d}, '')::numeric, 0)            AS min_roi,
        COALESCE(NULLIF($$Value_{sort_by}, ''), 'sport_h_score')           AS sort_by
),
latest_v2_15d AS (
    SELECT MAX(date) AS d FROM polymarket.wallet_profile_metrics_v2 WHERE calculation_window_days = 15
),
latest_v2_7d AS (
    SELECT MAX(date) AS d FROM polymarket.wallet_profile_metrics_v2 WHERE calculation_window_days = 7
),
latest_cat_15d AS (
    SELECT MAX(date) AS d FROM polymarket.wallet_profile_metrics_category_v2 WHERE calculation_window_days = 15
),
latest_cat_7d AS (
    SELECT MAX(date) AS d FROM polymarket.wallet_profile_metrics_category_v2 WHERE calculation_window_days = 7
),

base AS (
    SELECT
        g.proxy_wallet, g.total_pnl, g.best_market_pnl, g.worst_market_pnl,
        g.avg_market_exposure, g.profitable_markets_count,
        COALESCE(g.sortino_ratio, 0.9566) AS sortino_ratio,
        COALESCE(g.annualized_return, 3.0638) AS annualized_return,
        g.win_rate, g.sharpe_ratio, g.performance_trend,
        g.total_trades, g.markets_traded, g.total_invested
    FROM polymarket.wallet_profile_metrics_v2 g, latest_v2_15d, params p
    WHERE g.date = latest_v2_15d.d AND g.calculation_window_days = 15
        AND g.sybil_risk_flag = false AND g.suspicious_win_rate_flag = false
        AND g.combined_risk_score <= 50
        AND g.total_pnl > p.min_pnl
        AND g.total_trades BETWEEN p.min_total_trades AND p.max_total_trades
        AND g.win_rate BETWEEN p.min_win_rate AND p.max_win_rate
        AND g.roi > p.min_roi
),

sport_15d AS (
    SELECT
        c.proxy_wallet,
        COALESCE(MAX(CASE WHEN c.category = 'Sports' THEN c.total_pnl END),
                 SUM(CASE WHEN c.category LIKE 'Sports / %' THEN c.total_pnl ELSE 0 END)) AS sports_pnl,
        COALESCE(MAX(CASE WHEN c.category = 'Sports' THEN c.total_trades END),
                 SUM(CASE WHEN c.category LIKE 'Sports / %' THEN c.total_trades ELSE 0 END))::int AS sports_trades,
        COALESCE(MAX(CASE WHEN c.category = 'Sports' THEN c.total_invested END),
                 SUM(CASE WHEN c.category LIKE 'Sports / %' THEN c.total_invested ELSE 0 END)) AS sports_invested,
        MAX(CASE WHEN c.category = 'Sports / Baseball / MLB' THEN c.total_pnl ELSE 0 END) AS sports_pnl_mlb,
        MAX(CASE WHEN c.category = 'Sports / Soccer / La Liga' THEN c.total_pnl ELSE 0 END) AS sports_pnl_la_liga,
        MAX(CASE WHEN c.category = 'Sports / Soccer / Ligue 1' THEN c.total_pnl ELSE 0 END) AS sports_pnl_ligue_1,
        STRING_AGG(
            CASE WHEN c.category LIKE 'Sports / % / %' AND c.total_pnl > 0
            THEN c.category ELSE NULL END,
            ', ' ORDER BY c.total_pnl DESC
        ) AS sport_tags
    FROM polymarket.wallet_profile_metrics_category_v2 c, latest_cat_15d
    WHERE c.calculation_window_days = 15 AND c.date = latest_cat_15d.d AND c.category LIKE 'Sports%'
        AND c.proxy_wallet IN (SELECT proxy_wallet FROM base)
    GROUP BY c.proxy_wallet
    HAVING COALESCE(MAX(CASE WHEN c.category = 'Sports' THEN c.total_pnl END),
                    SUM(CASE WHEN c.category LIKE 'Sports / %' THEN c.total_pnl ELSE 0 END)) > 0
),

snap_7d AS (
    SELECT proxy_wallet, total_invested AS total_invested_7d,
        best_market_pnl AS best_market_pnl_7d, worst_market_pnl AS worst_market_pnl_7d,
        profit_factor AS profit_factor_7d, win_rate AS win_rate_7d
    FROM polymarket.wallet_profile_metrics_v2, latest_v2_7d
    WHERE date = latest_v2_7d.d AND calculation_window_days = 7
        AND proxy_wallet IN (SELECT proxy_wallet FROM sport_15d)
),

sport_7d AS (
    SELECT proxy_wallet,
        COALESCE(MAX(CASE WHEN category = 'Sports' THEN total_pnl END),
                 SUM(CASE WHEN category LIKE 'Sports / %' THEN total_pnl ELSE 0 END)) AS pnl_cat_sports_7d
    FROM polymarket.wallet_profile_metrics_category_v2, latest_cat_7d
    WHERE calculation_window_days = 7 AND date = latest_cat_7d.d AND category LIKE 'Sports%'
        AND proxy_wallet IN (SELECT proxy_wallet FROM sport_15d)
    GROUP BY proxy_wallet
),

eligible AS (
    SELECT b.*, s15.sports_pnl, s15.sports_trades, s15.sports_invested,
        s15.sports_pnl_mlb, s15.sports_pnl_la_liga, s15.sports_pnl_ligue_1,
        s15.sport_tags,
        COALESCE(s7g.total_invested_7d, 0) AS total_invested_7d,
        COALESCE(s7g.best_market_pnl_7d, 0) AS best_market_pnl_7d,
        COALESCE(s7g.worst_market_pnl_7d, 0) AS worst_market_pnl_7d,
        COALESCE(s7g.profit_factor_7d, 0) AS profit_factor_7d,
        COALESCE(s7g.win_rate_7d, 0) AS win_rate_7d,
        COALESCE(s7s.pnl_cat_sports_7d, 0) AS pnl_cat_sports_7d
    FROM base b
    JOIN sport_15d s15 ON s15.proxy_wallet = b.proxy_wallet
    LEFT JOIN snap_7d s7g ON s7g.proxy_wallet = b.proxy_wallet
    LEFT JOIN sport_7d s7s ON s7s.proxy_wallet = b.proxy_wallet
),

scored AS (
    SELECT proxy_wallet, total_pnl, sports_pnl, sports_trades, win_rate,
        sharpe_ratio, sortino_ratio, total_invested_7d, win_rate_7d,
        performance_trend, total_trades, markets_traded, total_invested,
        sport_tags,
        ROUND((
            PERCENT_RANK() OVER (ORDER BY total_pnl ASC)                  * 12.429 +
            PERCENT_RANK() OVER (ORDER BY best_market_pnl ASC)            * 10.106 +
            PERCENT_RANK() OVER (ORDER BY total_invested_7d ASC)          * 10.050 +
            PERCENT_RANK() OVER (ORDER BY win_rate_7d DESC)               *  8.752 +
            PERCENT_RANK() OVER (ORDER BY sports_trades ASC)              *  7.473 +
            PERCENT_RANK() OVER (ORDER BY best_market_pnl_7d ASC)         *  7.559 +
            PERCENT_RANK() OVER (ORDER BY worst_market_pnl_7d DESC)       *  7.336 +
            PERCENT_RANK() OVER (ORDER BY annualized_return ASC)          *  5.034 +
            PERCENT_RANK() OVER (ORDER BY profitable_markets_count ASC)   *  6.269 +
            PERCENT_RANK() OVER (ORDER BY avg_market_exposure ASC)        *  6.344 +
            PERCENT_RANK() OVER (ORDER BY pnl_cat_sports_7d ASC)          *  2.864 +
            PERCENT_RANK() OVER (ORDER BY sports_invested ASC)            *  3.201 +
            PERCENT_RANK() OVER (ORDER BY worst_market_pnl DESC)          *  2.173 +
            PERCENT_RANK() OVER (ORDER BY sports_pnl_mlb ASC)             *  2.760 +
            PERCENT_RANK() OVER (ORDER BY sortino_ratio ASC)              *  1.644 +
            PERCENT_RANK() OVER (ORDER BY sports_pnl ASC)                 *  1.949 +
            PERCENT_RANK() OVER (ORDER BY sports_pnl_la_liga ASC)         *  1.154 +
            PERCENT_RANK() OVER (ORDER BY sports_pnl_ligue_1 ASC)         *  1.033 +
            PERCENT_RANK() OVER (ORDER BY profit_factor_7d DESC)          *  1.872
        )::numeric, 1) AS sport_h_score
    FROM eligible
)

SELECT
    RANK() OVER (ORDER BY sport_h_score DESC) AS leaderboard_rank,
    proxy_wallet AS wallet,
    CASE WHEN sport_h_score >= 70 THEN 'Elite' WHEN sport_h_score >= 50 THEN 'Sharp'
         WHEN sport_h_score >= 35 THEN 'Solid' ELSE 'Emerging' END AS tier,
    sport_h_score,
    ROUND(sports_pnl::numeric, 2) AS sports_pnl_15d,
    sports_trades AS sports_trades_15d,
    ROUND((win_rate * 100)::numeric, 1) AS win_rate_pct_15d,
    ROUND(COALESCE(sharpe_ratio, 0)::numeric, 2) AS sharpe_ratio_15d,
    ROUND(sortino_ratio::numeric, 2) AS sortino_ratio_15d,
    total_trades AS total_trades_15d,
    markets_traded AS markets_traded_15d,
    ROUND(total_pnl::numeric, 2) AS total_pnl_15d,
    ROUND(total_invested::numeric, 2) AS total_volume_15d,
    performance_trend AS trajectory,
    sport_tags
FROM scored
CROSS JOIN params
ORDER BY
    CASE (SELECT sort_by FROM params)
        WHEN 'sport_h_score' THEN sport_h_score
        WHEN 'pnl'           THEN sports_pnl::numeric
        WHEN 'roi'           THEN (total_pnl / NULLIF(total_invested, 0))::numeric
        WHEN 'win_rate'      THEN win_rate::numeric
        WHEN 'trades'        THEN sports_trades::numeric
        ELSE sport_h_score
    END DESC
LIMIT 200;