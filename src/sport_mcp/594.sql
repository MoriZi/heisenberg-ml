WITH latest AS (
    SELECT MAX(date) AS snapshot_date
    FROM polymarket.wallet_profile_metrics_v2
    WHERE calculation_window_days = 15
),

target AS (
    SELECT
        m.condition_id,
        m.question,
        m.last_trade_price,
        m.volume,
        m.liquidity,
        m.end_date,
        m.slug,
        COALESCE(
            NULLIF($$Value_{sport}, ''),
            (
                SELECT tc.category
                FROM polymarket.polymarket_market_tag mt
                JOIN polymarket.tag_category tc ON tc.tag_id = mt.tag_id
                WHERE mt.market_id = m.id
                  AND tc.category ILIKE 'Sports /%'
                ORDER BY LENGTH(tc.category) DESC
                LIMIT 1
            )
        ) AS sport
    FROM polymarket.polymarket_market m
    WHERE m.condition_id = $$Value_{condition_id}
    LIMIT 1
),

market_positions AS (
    SELECT
        wp.proxy_wallet,
        wp.outcome,
        wp.shares,
        wp.avg_cost
    FROM polymarket.wallet_positions wp
    JOIN target t ON t.condition_id = wp.condition_id
    WHERE wp.shares >= COALESCE(NULLIF($$Value_{min_shares}, '')::float, 100)
),

hedgers AS (
    SELECT proxy_wallet
    FROM market_positions
    GROUP BY proxy_wallet
    HAVING COUNT(DISTINCT outcome) > 1
),

eligible_metrics AS (
    SELECT DISTINCT ON (w.proxy_wallet)
        w.proxy_wallet,
        (cat->>'total_pnl')::float   AS sport_pnl_15d,
        (cat->>'roi')::float         AS sport_roi_15d,
        (cat->>'win_rate')::float    AS sport_win_rate_15d,
        (cat->>'total_trades')::int  AS sport_trades_15d
    FROM polymarket.wallet_profile_metrics_v2 w,
        jsonb_array_elements(w.performance_by_category) AS cat
    CROSS JOIN latest
    CROSS JOIN target t
    WHERE w.proxy_wallet IN (SELECT proxy_wallet FROM market_positions)
        AND w.calculation_window_days = 15
        AND w.date = latest.snapshot_date
        AND (cat->>'category') = t.sport
        AND (cat->>'total_pnl')::float >= COALESCE(NULLIF($$Value_{min_sport_pnl}, '')::float, 1000)
        AND w.sybil_risk_flag = false
        AND w.suspicious_win_rate_flag = false
        AND w.combined_risk_score <= 50
        AND w.total_pnl > 1000
        AND w.total_trades BETWEEN 10 AND 200000
        AND w.win_rate BETWEEN 0 AND 0.95
        AND w.roi > 0
    ORDER BY w.proxy_wallet, (cat->>'total_pnl')::float DESC
),

sharp_positions AS (
    SELECT
        mp.proxy_wallet,
        mp.outcome,
        mp.shares,
        mp.avg_cost,
        em.sport_pnl_15d,
        em.sport_roi_15d,
        em.sport_win_rate_15d,
        em.sport_trades_15d
    FROM market_positions mp
    JOIN eligible_metrics em ON em.proxy_wallet = mp.proxy_wallet
    LEFT JOIN hedgers h ON h.proxy_wallet = mp.proxy_wallet
    WHERE h.proxy_wallet IS NULL
)

SELECT
    t.condition_id,
    t.question,
    t.slug,
    t.sport                                                              AS detected_sport,
    ROUND(t.last_trade_price::numeric, 4)                               AS current_price,
    ROUND(t.volume::numeric, 2)                                         AS market_volume,
    ROUND(t.liquidity::numeric, 2)                                      AS liquidity,
    t.end_date,
    sp.proxy_wallet,
    sp.outcome,
    ROUND(sp.shares::numeric, 2)                                        AS shares,
    ROUND(sp.avg_cost::numeric, 4)                                      AS avg_cost,
    ROUND(((t.last_trade_price - sp.avg_cost) * sp.shares)::numeric, 2) AS unrealized_pnl,
    ROUND(sp.sport_pnl_15d::numeric, 2)                                 AS sport_pnl_15d,
    ROUND(sp.sport_roi_15d::numeric, 2)                                 AS sport_roi_15d,
    ROUND((sp.sport_win_rate_15d * 100)::numeric, 1)                    AS sport_win_rate_pct_15d,
    sp.sport_trades_15d
FROM sharp_positions sp
CROSS JOIN target t
ORDER BY sp.outcome, sp.shares DESC
LIMIT COALESCE(NULLIF($$Value_{limit}, '')::int, 100);