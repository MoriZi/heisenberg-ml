WITH requested_window AS (
    SELECT
        CASE WHEN $$Value_{end_time}::bigint = 2200000000
             THEN NOW()
             ELSE TO_TIMESTAMP($$Value_{end_time}::bigint)
        END AS end_ts,
        CASE WHEN $$Value_{start_time}::bigint = 1600000000
             THEN (CASE WHEN $$Value_{end_time}::bigint = 2200000000
                        THEN NOW() ELSE TO_TIMESTAMP($$Value_{end_time}::bigint) END)
                  - INTERVAL '7 days'
             ELSE TO_TIMESTAMP($$Value_{start_time}::bigint)
        END AS requested_start_ts
),
bounded_window AS (
    SELECT
        end_ts,
        -- HARD CAP: never let the window exceed 30 days
        GREATEST(requested_start_ts, end_ts - INTERVAL '30 days') AS start_ts
    FROM requested_window
),
sport_markets AS (
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
CROSS JOIN bounded_window bw
WHERE t.timestamp >= bw.start_ts
  AND t.timestamp <= bw.end_ts
  AND ('ALL' = '$$Value_{condition_id}' OR t.condition_id = LOWER('$$Value_{condition_id}'))
  AND ('ALL' = '$$Value_{side}' OR t.side = '$$Value_{side}')
  AND (
      '$$Value_{proxy_wallet}' = 'ALL'
      OR t.proxy_wallet = LOWER(('$$Value_{proxy_wallet}')::text)
  )
ORDER BY t.timestamp DESC;
