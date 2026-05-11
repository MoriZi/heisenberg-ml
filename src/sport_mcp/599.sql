SELECT 
    -- general profile (risk, behavioral, confidence)
    g.date,
    g.proxy_wallet,
    g.calculation_window_days,
    g.date_range_start,
    g.date_range_end,
    g.total_trades        AS overall_trades,
    g.win_rate            AS overall_win_rate,
    g.total_pnl           AS overall_pnl,
    g.roi                 AS overall_roi,
    g.sharpe_ratio,
    g.sortino_ratio,
    g.calmar_ratio,
    g.max_drawdown,
    g.profit_factor,
    g.ulcer_index,
    g.gain_to_pain_ratio,
    g.annualized_return,
    g.performance_trend,
    g.equity_curve_pattern,
    g.statistical_confidence,
    g.days_active,
    g.markets_traded,
    g.last_active,
    -- risk flags
    g.sybil_risk_flag,
    g.sybil_risk_score,
    g.suspicious_win_rate_flag,
    g.timing_anomaly_flag,
    g.position_size_volatility_flag,
    g.perfect_timing_flag,
    g.combined_risk_score,
    g.risk_level,
    g.flagged_metrics,
    -- sport-specific performance
    c.category            AS sport,
    c.total_pnl           AS sport_pnl,
    c.total_trades        AS sport_trades,
    c.total_buy_trades    AS sport_buy_trades,
    c.total_sell_trades   AS sport_sell_trades,
    c.winning_trades      AS sport_wins,
    c.losing_trades       AS sport_losses,
    c.win_rate            AS sport_win_rate,
    c.roi                 AS sport_roi,
    c.total_invested      AS sport_invested
FROM polymarket.wallet_profile_metrics_v2 g
JOIN polymarket.wallet_profile_metrics_category_v2 c
    ON c.proxy_wallet = g.proxy_wallet
    AND c.calculation_window_days = g.calculation_window_days
    AND c.date = g.date
WHERE g.proxy_wallet = LOWER('$$Value_{proxy_wallet}')
    AND g.calculation_window_days = $$Value_{window_days}
    AND c.category LIKE 'Sports%'
    AND c.category LIKE '%' || '$$Value_{sport}' || '%'
ORDER BY g.date DESC, c.total_pnl DESC;