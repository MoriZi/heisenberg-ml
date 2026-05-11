"""
Feature modules for the BTC signal pipeline.

Each feature independently evaluates one angle of the market and returns
a FeatureResult. Features are registered here and discovered by the
signal card orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

from src.models.btc_signal.config import BTCSignalConfig


@dataclass
class FeatureResult:
    """Output of a single feature computation."""

    name: str                       # e.g. "streak"
    signal: str | None              # "Up", "Down", or None (no opinion)
    confidence: float               # 0.0 to 1.0 — strength of this reading
    detail: dict = field(default_factory=dict)  # feature-specific metadata


class Feature(Protocol):
    """Contract for all feature implementations."""

    name: str

    def compute(
        self,
        target: pd.Series,
        history: pd.DataFrame,
        config: BTCSignalConfig,
        conn=None,
    ) -> FeatureResult:
        """Compute this feature for a target market given history.

        Parameters
        ----------
        target : pd.Series — the market row to predict
        history : pd.DataFrame — preceding closed markets (chronological,
                  with 'winner' column), most recent last
        config : BTCSignalConfig
        conn : optional DB connection for features that need trade/candle data
        """
        ...


# ── Registry ────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, Feature] = {}


def register(feature: Feature) -> Feature:
    """Register a feature instance."""
    _REGISTRY[feature.name] = feature
    return feature


def get_feature(name: str) -> Feature:
    return _REGISTRY[name]


def get_all_features() -> list[Feature]:
    return list(_REGISTRY.values())


def get_feature_names() -> list[str]:
    return list(_REGISTRY.keys())
