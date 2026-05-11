WITH params AS (
    SELECT 
        CASE $$Value_{granularity}
            WHEN '1d'  THEN 86400
            WHEN '3d'  THEN 259200
            WHEN '1w'  THEN 604800
            WHEN '1m'  THEN 2592000
            WHEN 'all' THEN 3153600000
        END AS granularity_seconds
),
wallet_conditions AS (
    SELECT DISTINCT condition_id
    FROM polymarket.wallet_daily_pnl_v2
    WHERE proxy_wallet = LOWER($$Value_{wallet})
      AND date >= $$Value_{start_time}::timestamptz
      AND date <= $$Value_{end_time}::timestamptz
      AND ('ALL' = '$$Value_{condition_id}' OR condition_id = LOWER('$$Value_{condition_id}'))
),
market_sport AS (
    SELECT DISTINCT ON (wc.condition_id)
        wc.condition_id,
        tc.category AS sport
    FROM wallet_conditions wc
    JOIN polymarket.polymarket_market m ON m.condition_id = wc.condition_id
    JOIN polymarket.polymarket_market_tag mt ON mt.market_id = m.id
    JOIN polymarket.tag_category tc ON tc.tag_id = mt.tag_id
    WHERE tc.category LIKE 'Sports%'
    AND tc.category LIKE '%' || '$$Value_{sport}' || '%'
    ORDER BY wc.condition_id, LENGTH(tc.category) DESC
)
SELECT 
    to_timestamp(FLOOR(EXTRACT(EPOCH FROM d.date) / p.granularity_seconds) * p.granularity_seconds) AS date,
    d.proxy_wallet,
    ms.sport,
    ROUND(SUM(d.pnl)::numeric, 2) AS pnl,
    ROUND(SUM(d.invested)::numeric, 2) AS invested,
    SUM(d.trades) AS trades,
    SUM(d.wins) AS wins,
    SUM(d.losses) AS losses,
    ROUND((SUM(d.wins)::numeric / NULLIF(SUM(d.wins) + SUM(d.losses), 0) * 100), 2) AS win_rate
FROM polymarket.wallet_daily_pnl_v2 d
JOIN market_sport ms ON ms.condition_id = d.condition_id
CROSS JOIN params p
WHERE d.proxy_wallet = LOWER($$Value_{wallet})
  AND d.date >= $$Value_{start_time}::timestamptz
  AND d.date <= $$Value_{end_time}::timestamptz
  AND p.granularity_seconds IS NOT NULL
GROUP BY 1, d.proxy_wallet, ms.sport
ORDER BY 1 ASC, pnl DESC;