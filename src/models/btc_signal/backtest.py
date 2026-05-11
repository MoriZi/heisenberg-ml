"""
Backtest for the BTC signal card pipeline.

Tests features individually and in combination over historical markets.
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.common.db import get_engine
from src.models.btc_signal.config import BTCSignalConfig
from src.models.btc_signal.features import get_feature_names
from src.models.btc_signal.signal_card import SignalCard, compute_signal_cards
from src.models.btc_signal.utils import fetch_markets_with_winners


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


# ── Backtest runner ─────────────────────────────────────────────────────────


def run_backtest(
    days: int | None = None,
    last: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    features: list[str] | None = None,
    slug_pattern: str | None = None,
    config: BTCSignalConfig | None = None,
    save_csv: bool = True,
) -> pd.DataFrame:
    """Run a backtest over BTC Up/Down markets."""
    cfg = config or BTCSignalConfig()
    pattern = slug_pattern or cfg.slug_pattern
    extra = cfg.history_depth + 2

    if end_date is None:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # ── Header ────────────────────────────────────────────────────────
    feature_list = features or get_feature_names()
    print(f"\n{'=' * 70}")
    print(f"  BTC Signal Card Backtest")
    print(f"  Pattern:  {pattern}")
    if last:
        print(f"  Scope:    last {last} markets")
    else:
        lookback = days or cfg.backtest_days
        print(f"  Scope:    {lookback} days")
    print(f"  Features: {', '.join(feature_list)}")
    print(f"{'=' * 70}")

    # ── Fetch markets ─────────────────────────────────────────────────
    engine = get_engine()
    conn = engine.connect()
    print("\nFetching markets...")

    if last is not None:
        markets = fetch_markets_with_winners(
            conn, slug_pattern=pattern, end_date=end_date,
            limit=last + extra,
        )
    else:
        lookback = days or cfg.backtest_days
        start_dt = datetime.now(timezone.utc) - timedelta(days=lookback)
        ctx_start = (start_dt - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        markets = fetch_markets_with_winners(
            conn, slug_pattern=pattern,
            start_date=ctx_start, end_date=end_date,
        )

    print(f"  Settled markets: {len(markets):,}")

    if len(markets) < cfg.history_depth + 1:
        print("  Not enough markets.")
        conn.close()
        return pd.DataFrame()

    # ── Compute signal cards ──────────────────────────────────────────
    print(f"\nComputing signal cards (history depth={cfg.history_depth})...")
    cards = compute_signal_cards(
        markets, config=cfg, feature_names=features, conn=conn,
    )
    conn.close()
    engine.dispose()

    # Trim to requested scope
    if last is not None and len(cards) > last:
        cards = cards[-last:]

    print(f"  Cards generated: {len(cards):,}")

    if not cards:
        return pd.DataFrame()

    # ── Print results ─────────────────────────────────────────────────
    print_card_detail(cards, feature_list)
    print_feature_accuracy(cards, feature_list)
    print_combined_summary(cards)

    # ── Export ─────────────────────────────────────────────────────────
    df = pd.DataFrame([c.to_dict() for c in cards])

    if save_csv and not df.empty:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        label = "5m" if "5m" in pattern else "15m"
        out_path = OUTPUT_DIR / f"backtest_{label}_{ts}.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")

    return df


# ── Detail printing ─────────────────────────────────────────────────────────


def print_card_detail(cards: list[SignalCard], feature_names: list[str]) -> None:
    """Print per-market detail table."""
    # Build header
    feat_cols = "  ".join(f"{n[:8]:>8}" for n in feature_names)
    w = 6 + 34 + 5 + len(feat_cols) + 12 + 5 + 4
    print(f"\n{'─' * w}")
    header = (
        f"  {'#':>3}  {'Slug':32}  {'Win':4}  "
        f"{feat_cols}  "
        f"{'Combined':>8}  {'Conf':>5}  {'OK':2}"
    )
    print(header)
    print(f"{'─' * w}")

    for i, card in enumerate(cards):
        winner = card.winner or "—"

        # Feature signals: show signal + confidence
        feat_cells = []
        for name in feature_names:
            r = card.features.get(name)
            if r and r.signal:
                cell = f"{r.signal[0]}{r.confidence:.1f}"
            else:
                cell = "—"
            feat_cells.append(f"{cell:>8}")
        feat_str = "  ".join(feat_cells)

        combined = card.combined_signal or "—"
        conf = f"{card.combined_confidence:.3f}" if card.fires else "  —"

        if card.correct is True:
            ok = "+"
        elif card.correct is False:
            ok = "x"
        else:
            ok = "—"

        slug_short = card.slug.replace("btc-updown-5m-", "5m:")
        print(
            f"  {i + 1:>3}  {slug_short:32}  {winner:4}  "
            f"{feat_str}  "
            f"{combined:>8}  {conf:>5}  {ok:2}"
        )

    print(f"{'─' * w}")


# ── Per-feature accuracy ────────────────────────────────────────────────────


def print_feature_accuracy(
    cards: list[SignalCard], feature_names: list[str],
) -> None:
    """Print accuracy breakdown for each feature independently."""
    print(f"\n{'=' * 70}")
    print(f"  Per-Feature Accuracy")
    print(f"{'=' * 70}")

    for name in feature_names:
        fired = []
        for card in cards:
            r = card.features.get(name)
            if r and r.signal and card.winner:
                fired.append(r.signal == card.winner)

        n = len(fired)
        if n == 0:
            print(f"  {name:20}  fired=  0")
            continue

        correct = sum(fired)
        acc = correct / n
        print(f"  {name:20}  fired={n:>4}  correct={correct:>4}  acc={acc:.1%}")

    print(f"{'=' * 70}")


# ── Combined summary ────────────────────────────────────────────────────────


def print_combined_summary(cards: list[SignalCard]) -> None:
    """Print summary for the combined signal."""
    total = len(cards)
    fired = [c for c in cards if c.fires]
    evaluable = [c for c in fired if c.correct is not None]
    correct = sum(1 for c in evaluable if c.correct)

    print(f"\n{'=' * 70}")
    print(f"  Combined Signal Summary")
    print(f"{'=' * 70}")
    print(f"  Total markets:    {total:,}")
    print(f"  Signals fired:    {len(fired):,}")
    print(f"  Abstained:        {total - len(fired):,}")

    if total > 0:
        print(f"  Abstention rate:  {(total - len(fired)) / total:.1%}")

    if evaluable:
        n = len(evaluable)
        acc = correct / n
        print(f"\n  Evaluable:        {n:,}")
        print(f"  Correct:          {correct:,}")
        print(f"  Accuracy:         {acc:.1%}")

        # Accuracy by confidence band
        print(f"\n  {'Confidence Band':20}  {'N':>4}  {'Correct':>7}  {'Acc':>6}")
        bands = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
        for lo, hi in bands:
            band_cards = [c for c in evaluable
                          if lo <= c.combined_confidence < hi]
            if band_cards:
                bc = sum(1 for c in band_cards if c.correct)
                bn = len(band_cards)
                label = f"  [{lo:.1f}, {hi:.1f})"
                print(f"  {label:20}  {bn:>4}  {bc:>7}  {bc / bn:.1%}")
    else:
        print(f"\n  No evaluable signals.")

    print(f"{'=' * 70}")
