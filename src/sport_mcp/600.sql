SELECT
    mqi.snapshot_time,
    mqi.condition_id,
    mqi.slug,
    mqi.question,
    mqi.end_date,
    mqi.market_active,
    mqi.market_closed,
    mqi.current_volume_24h,
    mqi.current_volume_7d,
    mqi.liquidity_percentile,
    mqi.liquidity_tier,
    mqi.liquidity_risk_flag,
    mqi.volume_trend,
    mqi.volume_collapse_risk_flag,
    mqi.volume_ratio_24h_to_7d,
    mqi.top1_wallet_pct,
    mqi.top3_wallet_pct,
    mqi.top10_wallet_pct,
    mqi.whale_control_flag,
    mqi.unique_traders_7d,
    mqi.trades_per_hour_avg,
    mqi.peak_hour_trades,
    mqi.trade_concentration_flag,
    mqi.squeeze_risk_flag,
    mqi.yes_avg_pnl,
    mqi.no_avg_pnl,
    CASE WHEN mqi.yes_avg_pnl = 0 AND mqi.no_avg_pnl = 0 THEN NULL ELSE mqi.winning_side END AS winning_side,
    mqi.net_pnl,
    mqi.profit_loss_ratio,
    mqi.created_at,
    mqi.updated_at
FROM polymarket.market_quality_insights mqi
WHERE 1=1
AND ('ALL' = '$$Value_{condition_id}' OR mqi.condition_id = LOWER('$$Value_{condition_id}'))
AND (
    '0' = '$$Value_{min_volume_24h}'
    OR mqi.current_volume_24h >= $$Value_{min_volume_24h}::numeric
)
AND (
    '0' = '$$Value_{min_liquidity_percentile}'
    OR mqi.liquidity_percentile >= $$Value_{min_liquidity_percentile}::numeric
)
AND (
    'ALL' = '$$Value_{volume_trend}'
    OR mqi.volume_trend = '$$Value_{volume_trend}'
)
AND (
    '0' = '$$Value_{min_top1_wallet_pct}'
    OR mqi.top1_wallet_pct >= $$Value_{min_top1_wallet_pct}::numeric
)
AND (
    '0' = '$$Value_{max_unique_traders_7d}'
    OR mqi.unique_traders_7d <= $$Value_{max_unique_traders_7d}::integer
)
AND EXISTS (
    SELECT 1
    FROM polymarket.polymarket_market m
    JOIN polymarket.polymarket_market_tag mt ON mt.market_id = m.id
    JOIN polymarket.tag_category tc ON tc.tag_id = mt.tag_id
    WHERE m.condition_id = mqi.condition_id
    AND tc.category LIKE 'Sports%'
    AND tc.category LIKE '%' || '$$Value_{sport}' || '%'
)
ORDER BY mqi.current_volume_24h DESC NULLS LAST;