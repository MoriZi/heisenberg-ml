"""
Feature: Volume trend across recent markets.

Compares volume levels in the most recent markets to detect
acceleration or deceleration. From 1000-market analysis:
- Low prior volume → 53.2% continuation
- High prior volume (Q3) → 44.4% continuation (reversal tendency)

Data needed: only 'volume' column from history (no DB queries).
"""

import pandas as pd

from src.models.btc_signal.config import BTCSignalConfig
from src.models.btc_signal.features import FeatureResult, register


class VolumeTrendFeature:
    name = "volume_trend"

    def compute(
        self,
        target: pd.Series,
        history: pd.DataFrame,
        config: BTCSignalConfig,
        conn=None,
    ) -> FeatureResult:
        lookback = history.tail(config.volume_lookback)

        if len(lookback) < 3:
            return FeatureResult(
                name=self.name, signal=None, confidence=0.0,
                detail={"reason": "insufficient_history"},
            )

        volumes = lookback["volume"].tolist()
        prev_winner = lookback.iloc[-1]["winner"]

        # Volume trend: compare recent half vs older half
        mid = len(volumes) // 2
        older_avg = sum(volumes[:mid]) / mid if mid > 0 else 0
        recent_avg = sum(volumes[mid:]) / (len(volumes) - mid)

        if older_avg == 0:
            return FeatureResult(
                name=self.name, signal=None, confidence=0.0,
                detail={"reason": "zero_older_volume"},
            )

        vol_ratio = recent_avg / older_avg

        if vol_ratio > 1.3:
            # Volume accelerating → reversal tendency (high volume exhaustion)
            signal = "Down" if prev_winner == "Up" else "Up"
            trend = "accelerating"
            confidence = min(0.4 + (vol_ratio - 1.3) * 0.5, 0.8)
        elif vol_ratio < 0.7:
            # Volume decelerating → continuation
            signal = prev_winner
            trend = "decelerating"
            confidence = min(0.4 + (0.7 - vol_ratio) * 0.5, 0.8)
        else:
            # Stable volume → no opinion
            return FeatureResult(
                name=self.name, signal=None, confidence=0.0,
                detail={
                    "vol_ratio": round(vol_ratio, 3),
                    "trend": "stable",
                    "prev_winner": prev_winner,
                    "recent_avg": round(recent_avg, 0),
                    "older_avg": round(older_avg, 0),
                },
            )

        return FeatureResult(
            name=self.name,
            signal=signal,
            confidence=round(confidence, 3),
            detail={
                "vol_ratio": round(vol_ratio, 3),
                "trend": trend,
                "prev_winner": prev_winner,
                "recent_avg": round(recent_avg, 0),
                "older_avg": round(older_avg, 0),
            },
        )


register(VolumeTrendFeature())
