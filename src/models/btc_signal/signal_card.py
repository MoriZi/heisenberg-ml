"""
Signal card orchestrator.

Runs all registered features for a target market and produces a
SignalCard — the unit of output that a human or agent evaluates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.models.btc_signal.config import BTCSignalConfig
from src.models.btc_signal.features import (
    Feature,
    FeatureResult,
    get_all_features,
    get_feature,
)
from src.models.btc_signal.utils import resolve_winner

# Import feature modules so they auto-register
import src.models.btc_signal.features.streak          # noqa: F401
import src.models.btc_signal.features.decisiveness     # noqa: F401
import src.models.btc_signal.features.volume_trend     # noqa: F401
import src.models.btc_signal.features.price_lean       # noqa: F401
import src.models.btc_signal.features.early_movement   # noqa: F401
import src.models.btc_signal.features.btc_external     # noqa: F401


@dataclass
class SignalCard:
    """Aggregated signal for one market."""

    condition_id: str
    slug: str
    end_date: str
    winner: str | None                          # actual outcome (None if unsettled)
    features: dict[str, FeatureResult] = field(default_factory=dict)
    combined_signal: str | None = None          # "Up", "Down", or None
    combined_confidence: float = 0.0

    @property
    def fires(self) -> bool:
        return self.combined_signal is not None

    @property
    def correct(self) -> bool | None:
        if not self.fires or self.winner is None:
            return None
        return self.combined_signal == self.winner

    def to_dict(self) -> dict:
        """Flatten to a dict for DataFrame/CSV export."""
        row = {
            "condition_id": self.condition_id,
            "slug": self.slug,
            "end_date": self.end_date,
            "winner": self.winner,
            "combined_signal": self.combined_signal,
            "combined_confidence": round(self.combined_confidence, 3),
            "fires": self.fires,
            "correct": self.correct,
        }
        for name, result in self.features.items():
            row[f"f_{name}_signal"] = result.signal
            row[f"f_{name}_conf"] = result.confidence
            # Include key detail fields
            for k, v in result.detail.items():
                row[f"f_{name}_{k}"] = v
        return row


# ── Combiner ────────────────────────────────────────────────────────────────


def _combine_features(features: dict[str, FeatureResult]) -> tuple[str | None, float]:
    """Combine feature signals via confidence-weighted voting.

    Returns (direction, confidence). Direction is None if net signal
    is too weak or no features have an opinion.
    """
    up_weight = 0.0
    down_weight = 0.0

    for result in features.values():
        if result.signal is None or result.confidence <= 0:
            continue
        if result.signal == "Up":
            up_weight += result.confidence
        else:
            down_weight += result.confidence

    total = up_weight + down_weight
    if total == 0:
        return None, 0.0

    net = abs(up_weight - down_weight)
    confidence = net / total

    # Require minimum net confidence to fire
    if confidence < 0.1:
        return None, confidence

    direction = "Up" if up_weight > down_weight else "Down"
    return direction, round(confidence, 3)


# ── Card computation ────────────────────────────────────────────────────────


def compute_signal_card(
    target: pd.Series,
    history: pd.DataFrame,
    config: BTCSignalConfig | None = None,
    feature_names: list[str] | None = None,
    conn=None,
) -> SignalCard:
    """Compute a signal card for a single target market."""
    cfg = config or BTCSignalConfig()

    if feature_names:
        features = [get_feature(n) for n in feature_names]
    else:
        features = get_all_features()

    results: dict[str, FeatureResult] = {}
    for feat in features:
        results[feat.name] = feat.compute(
            target=target, history=history, config=cfg, conn=conn,
        )

    combined_signal, combined_confidence = _combine_features(results)
    winner = resolve_winner(target)

    return SignalCard(
        condition_id=target["condition_id"],
        slug=target["slug"],
        end_date=str(target["end_date"]),
        winner=winner,
        features=results,
        combined_signal=combined_signal,
        combined_confidence=combined_confidence,
    )


def compute_signal_cards(
    markets: pd.DataFrame,
    config: BTCSignalConfig | None = None,
    feature_names: list[str] | None = None,
    conn=None,
) -> list[SignalCard]:
    """Compute signal cards for a sequence of markets.

    The first `history_depth` markets are used as context only;
    signal cards are generated starting from index history_depth onward.
    """
    cfg = config or BTCSignalConfig()
    depth = cfg.history_depth

    # Pre-resolve winners for history
    if "winner" not in markets.columns:
        markets = markets.copy()
        markets["winner"] = markets.apply(resolve_winner, axis=1)

    cards: list[SignalCard] = []
    for i in range(depth, len(markets)):
        target = markets.iloc[i]
        history = markets.iloc[max(0, i - depth):i]
        # Only include settled history
        history = history[history["winner"].notna()]

        card = compute_signal_card(
            target=target,
            history=history,
            config=cfg,
            feature_names=feature_names,
            conn=conn,
        )
        cards.append(card)

    return cards
