"""
Feature: Early window price movement.

The first minute of the BTC price window sees explosive volume
and large price swings. This feature captures the direction and
magnitude of movement in the first N seconds.

From 500-market analysis:
- Price >= 0.65 after minute 0: 75.2% Up wins
- Price <  0.35 after minute 0: 77.9% Down wins
- But: market price already reflects this, so edge depends on
  getting in at a better price than the minute-0 close.

Data needed: candlestick data for the target market's first minute.
Note: only applicable in live monitoring or post-hoc backtesting.
"""

import pandas as pd

from src.models.btc_signal.config import BTCSignalConfig
from src.models.btc_signal.features import FeatureResult, register
from src.models.btc_signal.utils import slug_to_window_start


class EarlyMovementFeature:
    name = "early_movement"

    def compute(
        self,
        target: pd.Series,
        history: pd.DataFrame,
        config: BTCSignalConfig,
        conn=None,
    ) -> FeatureResult:
        if conn is None:
            return FeatureResult(
                name=self.name, signal=None, confidence=0.0,
                detail={"reason": "no_conn"},
            )

        window_start = slug_to_window_start(target["slug"])
        if window_start is None:
            return FeatureResult(
                name=self.name, signal=None, confidence=0.0,
                detail={"reason": "no_slug_timestamp"},
            )

        # Get the first candle at window start
        sql = """
            SELECT open, close, high, low, volume, trade_count
            FROM   polymarket.candlestick
            WHERE  condition_id = %(cid)s
              AND  outcome = 'Up'
              AND  candle_time = %(candle_time)s
        """
        df = pd.read_sql(sql, conn, params={
            "cid": target["condition_id"],
            "candle_time": window_start,
        })

        if df.empty:
            return FeatureResult(
                name=self.name, signal=None, confidence=0.0,
                detail={"reason": "no_candle_at_window_start"},
            )

        row = df.iloc[0]
        open_price = float(row["open"])
        close_price = float(row["close"])
        volume = float(row["volume"]) if row["volume"] else 0
        trade_count = int(row["trade_count"]) if row["trade_count"] else 0

        displacement = close_price - 0.50
        abs_disp = abs(displacement)

        if abs_disp < 0.05:
            return FeatureResult(
                name=self.name, signal=None, confidence=0.0,
                detail={
                    "open": open_price,
                    "close": close_price,
                    "displacement": round(displacement, 3),
                    "volume": volume,
                    "trade_count": trade_count,
                    "reason": "movement_too_small",
                },
            )

        signal = "Up" if displacement > 0 else "Down"

        # Strong moves (>0.15 from 0.50) are highly predictive
        if abs_disp >= 0.20:
            confidence = 0.8
        elif abs_disp >= 0.15:
            confidence = 0.7
        elif abs_disp >= 0.10:
            confidence = 0.6
        else:
            confidence = 0.5

        return FeatureResult(
            name=self.name,
            signal=signal,
            confidence=confidence,
            detail={
                "open": open_price,
                "close": close_price,
                "displacement": round(displacement, 3),
                "volume": round(volume, 0),
                "trade_count": trade_count,
            },
        )


register(EarlyMovementFeature())
