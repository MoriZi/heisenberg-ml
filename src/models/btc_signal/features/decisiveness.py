"""
Feature: Prior market decisiveness.

Measures how quickly/strongly the prior market moved away from 0.50
in its first minute. Indecisive prior markets (price stayed near 0.50)
predict reversal; decisive markets predict continuation.

From 300-market analysis:
- Indecisive (|move| < 0.10): 58.5% reversal
- Decisive (|move| >= 0.20): 53.8% continuation

Data needed: candlestick data for the prior market's first minute.
"""

from datetime import timedelta, timezone, datetime

import pandas as pd

from src.models.btc_signal.config import BTCSignalConfig
from src.models.btc_signal.features import FeatureResult, register
from src.models.btc_signal.utils import slug_to_window_start


class DecisivenessFeature:
    name = "decisiveness"

    def compute(
        self,
        target: pd.Series,
        history: pd.DataFrame,
        config: BTCSignalConfig,
        conn=None,
    ) -> FeatureResult:
        if len(history) < 1 or conn is None:
            return FeatureResult(
                name=self.name, signal=None, confidence=0.0,
                detail={"reason": "no_history_or_conn"},
            )

        prev = history.iloc[-1]
        prev_winner = prev["winner"]
        window_start = slug_to_window_start(prev["slug"])

        if window_start is None:
            return FeatureResult(
                name=self.name, signal=None, confidence=0.0,
                detail={"reason": "no_slug_timestamp"},
            )

        # Get the Up outcome close price at minute 0 of the prior market
        sql = """
            SELECT close FROM polymarket.candlestick
            WHERE condition_id = %(cid)s
              AND outcome = 'Up'
              AND candle_time = %(candle_time)s
        """
        df = pd.read_sql(sql, conn, params={
            "cid": prev["condition_id"],
            "candle_time": window_start,
        })

        if df.empty:
            return FeatureResult(
                name=self.name, signal=None, confidence=0.0,
                detail={"reason": "no_candle_data", "cid": prev["condition_id"]},
            )

        min0_close = float(df.iloc[0]["close"])
        displacement = abs(min0_close - 0.50)

        if displacement < config.indecisive_max_displacement:
            # Indecisive → reversal signal
            signal = "Down" if prev_winner == "Up" else "Up"
            confidence = 0.6 + (config.indecisive_max_displacement - displacement) * 2
            regime = "indecisive"
        elif displacement >= config.decisive_min_displacement:
            # Decisive → continuation signal
            signal = prev_winner
            confidence = 0.5 + (displacement - config.decisive_min_displacement)
            regime = "decisive"
        else:
            # Middle zone → no opinion
            return FeatureResult(
                name=self.name, signal=None, confidence=0.0,
                detail={
                    "min0_close": min0_close,
                    "displacement": round(displacement, 3),
                    "prev_winner": prev_winner,
                    "regime": "neutral",
                },
            )

        confidence = min(max(confidence, 0.0), 1.0)

        return FeatureResult(
            name=self.name,
            signal=signal,
            confidence=round(confidence, 3),
            detail={
                "min0_close": min0_close,
                "displacement": round(displacement, 3),
                "prev_winner": prev_winner,
                "regime": regime,
            },
        )


register(DecisivenessFeature())
