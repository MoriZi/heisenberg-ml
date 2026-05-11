WITH market AS (                                                                                
      SELECT condition_id, question, slug, end_date, closed,                                      
             outcomes::text AS outcomes_text, outcome_prices::text AS prices_text                 
      FROM polymarket.polymarket_market                                                           
      WHERE condition_id = LOWER('$$Value_{condition_id}')                                        
  ),                                                                                              
  closing_trades AS (
      SELECT                                                                                      
          t.outcome,                                                                              
          t.price,                                                                                
          t.size,                         
          t.timestamp
      FROM polymarket.polymarket_trade t                                                          
      JOIN market m ON m.condition_id = t.condition_id
      WHERE t.timestamp BETWEEN (m.end_date - INTERVAL '10 minutes') AND m.end_date               
  )                                       
  SELECT
      m.condition_id,                                                                             
      m.question,                         
      m.slug,                                                                                     
      ct.outcome,                                                                                 
      ROUND((SUM(ct.price * ct.size) / NULLIF(SUM(ct.size), 0))::numeric, 4) AS closing_vwap,
      ROUND(MIN(ct.price)::numeric, 4) AS closing_low,                                            
      ROUND(MAX(ct.price)::numeric, 4) AS closing_high,
      ROUND(SUM(ct.size)::numeric, 2) AS closing_volume,
      COUNT(*) AS closing_trades_count,                                                           
      MAX(ct.timestamp) AS last_trade_time,                                                       
      m.end_date AS game_start,                                                                   
      m.closed AS resolved,                                                                       
      CASE        
          WHEN m.closed AND m.prices_text LIKE '[1%' THEN (m.outcomes_text::jsonb)->>0            
          WHEN m.closed AND m.prices_text LIKE '[0%' THEN (m.outcomes_text::jsonb)->>1
          ELSE NULL                                                                               
      END AS winning_outcome              
  FROM closing_trades ct                                                                          
  CROSS JOIN market m                                                                             
  GROUP BY m.condition_id, m.question, m.slug, ct.outcome,                                        
           m.end_date, m.closed, m.outcomes_text, m.prices_text                                   
  ORDER BY ct.outcome;