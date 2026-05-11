
WITH input AS (
    SELECT
        CASE
            WHEN ($$Value_{lookback_hours}::numeric) > 168
            THEN 168                      
            ELSE $$Value_{lookback_hours}::numeric
        END AS effective_lookback_hours,
        CASE
            WHEN ($$Value_{min_change_pct}::numeric) < 1
            THEN 1
            ELSE $$Value_{min_change_pct}::numeric
        END AS effective_min_change_pct
),
params AS (
    SELECT 
        CASE $$Value_{resolution}
            WHEN '1m'  THEN 60
            WHEN '5m'  THEN 300
            WHEN '15m' THEN 900
            WHEN '1h'  THEN 3600
            WHEN '4h'  THEN 14400
            WHEN '1d'  THEN 86400
        END AS interval_seconds
),
candles AS (
    SELECT
        to_timestamp(
            FLOOR(EXTRACT(EPOCH FROM c.candle_time) / p.interval_seconds) * p.interval_seconds
        ) AS bucket,
        c.condition_id,
        c.outcome,
        c.token_id,
        ROUND((ARRAY_AGG(c.open  ORDER BY c.candle_time ASC))[1]::numeric, 4)  AS open,
        ROUND((ARRAY_AGG(c.close ORDER BY c.candle_time DESC))[1]::numeric, 4) AS close,
        ROUND(SUM(c.volume)::numeric, 2)   AS volume,
        SUM(c.trade_count)                 AS trade_count
    FROM polymarket.candlestick c, params p, input i
    WHERE c.token_id = $$Value_{token_id}
      AND c.candle_time >= NOW() - (i.effective_lookback_hours * INTERVAL '1 hour')
      AND p.interval_seconds IS NOT NULL
    GROUP BY 1, c.condition_id, c.outcome, c.token_id
),
jumps AS (
    SELECT
        bucket                                                          AS jump_time,
        condition_id,
        outcome,
        token_id,
        LAG(close) OVER (PARTITION BY token_id ORDER BY bucket)        AS price_before,
        close                                                           AS price_after,
        ROUND(((close - LAG(close) OVER (PARTITION BY token_id ORDER BY bucket))
               / NULLIF(LAG(close) OVER (PARTITION BY token_id ORDER BY bucket), 0) * 100)::numeric, 2) AS change_pct,
        CASE WHEN close > LAG(close) OVER (PARTITION BY token_id ORDER BY bucket)
             THEN 'up' ELSE 'down' END                                 AS direction,
        volume,
        trade_count
    FROM candles
)
SELECT
    jump_time,
    condition_id,
    outcome,
    token_id,
    price_before,
    price_after,
    change_pct,
    direction,
    volume,
    trade_count
FROM jumps, input i
WHERE ABS(change_pct) >= i.effective_min_change_pct
  AND price_before IS NOT NULL
ORDER BY ABS(change_pct) DESC;