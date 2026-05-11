"""
Feature: Pre-window price lean.

Before the 5-minute BTC price window opens, there is thin trading
on the market near $0.50. Occasionally the price leans directionally
(range observed: 0.30–0.69). This feature captures that lean.

Data needed: candlestick data for the target market's pre-window period.
"""

from datetime import timedelta

import pandas as pd

from src.models.btc_signal.config import BTCSignalConfig
from src.models.btc_signal.features import FeatureResult, register
from src.models.btc_signal.utils import slug_to_window_start


class PriceLeanFeature:
    name = "price_lean"

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

        # Look at candles in the pre-window period
        pre_start = window_start - timedelta(
            minutes=config.price_lean_pre_window_minutes
        )

        sql = """
            SELECT candle_time, close, volume
            FROM   polymarket.candlestick
            WHERE  condition_id = %(cid)s
              AND  outcome = 'Up'
              AND  candle_time >= %(start)s
              AND  candle_time < %(end)s
              AND  volume > 0
            ORDER BY candle_time DESC
            LIMIT 1
        """
        df = pd.read_sql(sql, conn, params={
            "cid": target["condition_id"],
            "start": pre_start,
            "end": window_start,
        })

        if df.empty:
            return FeatureResult(
                name=self.name, signal=None, confidence=0.0,
                detail={"reason": "no_pre_window_candles"},
            )

        last_pre_close = float(df.iloc[0]["close"])
        lean = last_pre_close - 0.50

        if abs(lean) < 0.03:
            return FeatureResult(
                name=self.name, signal=None, confidence=0.0,
                detail={
                    "pre_window_price": last_pre_close,
                    "lean": round(lean, 3),
                    "reason": "lean_too_small",
                },
            )

        signal = "Up" if lean > 0 else "Down"
        confidence = min(abs(lean) * 3, 0.9)  # 0.03→0.09, 0.10→0.30, 0.20→0.60

        return FeatureResult(
            name=self.name,
            signal=signal,
            confidence=round(confidence, 3),
            detail={
                "pre_window_price": last_pre_close,
                "lean": round(lean, 3),
            },
        )


register(PriceLeanFeature())
