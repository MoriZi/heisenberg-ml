"""
BTC Up/Down signal pipeline.

Multi-feature signal card system for Polymarket BTC 5-minute binary markets.
Each feature independently evaluates a different angle (historical patterns,
current market state, external BTC data) and produces a typed result.
The signal card aggregates all features for human or agent decision-making.
"""
