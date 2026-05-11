"""
Feature: External BTC spot data (placeholder).

This is the highest-potential feature — these markets settle on BTC
price direction, so BTC spot momentum is the most direct predictor.

TODO: Integrate Binance or similar BTC price feed.
Planned signals:
- BTC 5m/15m/1h momentum (price change %)
- Volatility regime (trending vs choppy)
- Volume-weighted momentum

For now, always returns no-opinion so the card handles it gracefully.
"""

import pandas as pd

from src.models.btc_signal.config import BTCSignalConfig
from src.models.btc_signal.features import FeatureResult, register


class BTCExternalFeature:
    name = "btc_external"

    def compute(
        self,
        target: pd.Series,
        history: pd.DataFrame,
        config: BTCSignalConfig,
        conn=None,
    ) -> FeatureResult:
        return FeatureResult(
            name=self.name,
            signal=None,
            confidence=0.0,
            detail={"reason": "not_implemented"},
        )


register(BTCExternalFeature())
