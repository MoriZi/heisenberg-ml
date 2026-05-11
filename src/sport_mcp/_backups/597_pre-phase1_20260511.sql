WITH sport_markets AS (
    SELECT condition_id
    FROM polymarket.polymarket_market
    WHERE slug LIKE '$$Value_{market_slug}' || '%'
)
SELECT 
    t.id,
    t.side,
    t.size,
    t.price,
    t.timestamp,
    t.transaction_hash,
    t.slug,
    t.outcome,
    t.asset AS token_id,
    t.proxy_wallet,
    t.condition_id
FROM polymarket.polymarket_trade t
JOIN sport_markets sm ON sm.condition_id = t.condition_id
WHERE ('ALL' = '$$Value_{condition_id}' OR t.condition_id = LOWER('$$Value_{condition_id}'))
AND ('ALL' = '$$Value_{side}' OR t.side = '$$Value_{side}')
AND (
    '$$Value_{proxy_wallet}' = 'ALL'
    OR t.proxy_wallet = LOWER(('$$Value_{proxy_wallet}')::text)
)
AND (
    1600000000 = $$Value_{start_time} OR
    t.timestamp >= TO_TIMESTAMP($$Value_{start_time}::bigint)
)
AND (
    2200000000 = $$Value_{end_time} OR
    t.timestamp <= TO_TIMESTAMP($$Value_{end_time}::bigint)
)
ORDER BY t.timestamp DESC;