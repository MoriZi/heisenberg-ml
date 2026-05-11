"""
Feature: Streak reversal.

After N consecutive same-direction outcomes, signal the opposite direction.
From 2000-market analysis: after 3+ same → 54.4% reversal rate.

Data needed: only resolved winners from history (no DB queries).
"""

import pandas as pd

from src.models.btc_signal.config import BTCSignalConfig
from src.models.btc_signal.features import FeatureResult, register


class StreakFeature:
    name = "streak"

    def compute(
        self,
        target: pd.Series,
        history: pd.DataFrame,
        config: BTCSignalConfig,
        conn=None,
    ) -> FeatureResult:
        lookback = history.tail(config.streak_lookback)

        if len(lookback) < config.streak_min_length:
            return FeatureResult(
                name=self.name, signal=None, confidence=0.0,
                detail={"streak_length": 0, "reason": "insufficient_history"},
            )

        # Count consecutive same outcomes from the tail
        winners = lookback["winner"].tolist()
        streak_dir = winners[-1]
        streak_len = 0
        for w in reversed(winners):
            if w == streak_dir:
                streak_len += 1
            else:
                break

        if streak_len < config.streak_min_length:
            return FeatureResult(
                name=self.name, signal=None, confidence=0.0,
                detail={
                    "streak_length": streak_len,
                    "streak_direction": streak_dir,
                    "reason": "streak_too_short",
                },
            )

        # Signal reversal — confidence scales with streak length
        reversal_dir = "Down" if streak_dir == "Up" else "Up"
        # 3 → 0.5, 4 → 0.6, 5 → 0.7, capped at 0.9
        confidence = min(0.5 + (streak_len - config.streak_min_length) * 0.1, 0.9)

        return FeatureResult(
            name=self.name,
            signal=reversal_dir,
            confidence=confidence,
            detail={
                "streak_length": streak_len,
                "streak_direction": streak_dir,
                "reversal_direction": reversal_dir,
            },
        )


register(StreakFeature())
