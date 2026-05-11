WITH latest_v2 AS (
    SELECT MAX(date) AS d FROM polymarket.wallet_profile_metrics_v2 WHERE calculation_window_days = 15
),
latest_cat AS (
    SELECT MAX(date) AS d FROM polymarket.wallet_profile_metrics_category_v2 WHERE calculation_window_days = 15
),

sport_wallets AS (
    SELECT c.proxy_wallet,
        COALESCE(MAX(CASE WHEN c.category = 'Sports' THEN c.total_pnl END),
                 SUM(CASE WHEN c.category LIKE 'Sports / %' THEN c.total_pnl ELSE 0 END)) AS sport_pnl_15d,
        COALESCE(MAX(CASE WHEN c.category = 'Sports' THEN c.roi END), 0) AS sport_roi_15d,
        COALESCE(MAX(CASE WHEN c.category = 'Sports' THEN c.win_rate END), 0) AS sport_win_rate_15d,
        COALESCE(MAX(CASE WHEN c.category = 'Sports' THEN c.total_trades END),
                 SUM(CASE WHEN c.category LIKE 'Sports / %' THEN c.total_trades ELSE 0 END))::int AS sport_trades_15d
    FROM polymarket.wallet_profile_metrics_category_v2 c, latest_cat
    WHERE c.calculation_window_days = 15 AND c.date = latest_cat.d
        AND c.category LIKE 'Sports%'
        AND c.proxy_wallet IN (
            SELECT proxy_wallet FROM polymarket.wallet_profile_metrics_v2, latest_v2
            WHERE date = latest_v2.d AND calculation_window_days = 15
                AND sybil_risk_flag = false AND suspicious_win_rate_flag = false
                AND combined_risk_score <= 50
                AND total_pnl >= COALESCE(NULLIF('$$Value_{min_sport_pnl}', '')::float, 10000)
        )
    GROUP BY c.proxy_wallet
    HAVING COALESCE(MAX(CASE WHEN c.category = 'Sports' THEN c.total_pnl END),
                    SUM(CASE WHEN c.category LIKE 'Sports / %' THEN c.total_pnl ELSE 0 END))
           >= COALESCE(NULLIF('$$Value_{min_sport_pnl}', '')::float, 10000)
       AND COALESCE(MAX(CASE WHEN c.category = 'Sports' THEN c.total_trades END),
                    SUM(CASE WHEN c.category LIKE 'Sports / %' THEN c.total_trades ELSE 0 END))
           >= COALESCE(NULLIF('$$Value_{min_trades}', '')::int, 5)
),

live_conditions AS (
    SELECT DISTINCT m.condition_id, m.question, m.last_trade_price,
        m.volume, m.liquidity, m.end_date, m.slug
    FROM polymarket.polymarket_market m
    JOIN polymarket.polymarket_market_tag mt ON mt.market_id = m.id
    JOIN polymarket.tag_category tc ON tc.tag_id = mt.tag_id
    WHERE tc.category LIKE 'Sports%'
        AND tc.category LIKE '%' || COALESCE(NULLIF('$$Value_{sport}', ''), 'Sports') || '%'
        AND m.active = true AND m.closed = false
        AND m.end_date > NOW()
        AND m.last_trade_price BETWEEN 0.02 AND 0.98
),

positions AS (
    SELECT wp.proxy_wallet, wp.condition_id, wp.outcome, wp.shares, wp.avg_cost
    FROM polymarket.wallet_positions wp
    WHERE wp.proxy_wallet IN (SELECT proxy_wallet FROM sport_wallets)
        AND wp.condition_id IN (SELECT condition_id FROM live_conditions)
        AND wp.shares >= COALESCE(NULLIF('$$Value_{min_shares}', '')::float, 100)
),

hedgers AS (
    SELECT proxy_wallet, condition_id
    FROM positions
    GROUP BY proxy_wallet, condition_id
    HAVING COUNT(DISTINCT outcome) > 1
)

SELECT
    sw.proxy_wallet,
    lc.question,
    p.outcome,
    ROUND(p.shares::numeric, 2) AS shares,
    ROUND(p.avg_cost::numeric, 4) AS avg_cost,
    ROUND(lc.last_trade_price::numeric, 4) AS current_price,
    ROUND(((lc.last_trade_price - p.avg_cost) * p.shares)::numeric, 2) AS unrealized_pnl,
    ROUND(lc.volume::numeric, 2) AS market_volume,
    ROUND(lc.liquidity::numeric, 2) AS liquidity,
    lc.end_date,
    lc.slug,
    ROUND(sw.sport_pnl_15d::numeric, 2) AS sport_pnl_15d,
    ROUND(sw.sport_roi_15d::numeric, 2) AS sport_roi_15d,
    ROUND((sw.sport_win_rate_15d * 100)::numeric, 1) AS sport_win_rate_pct_15d,
    sw.sport_trades_15d
FROM positions p
JOIN sport_wallets sw ON sw.proxy_wallet = p.proxy_wallet
JOIN live_conditions lc ON lc.condition_id = p.condition_id
LEFT JOIN hedgers h ON h.proxy_wallet = p.proxy_wallet AND h.condition_id = p.condition_id
WHERE h.proxy_wallet IS NULL
ORDER BY sw.sport_pnl_15d DESC, p.shares DESC
LIMIT COALESCE(NULLIF('$$Value_{limit}', '')::int, 50);