"""
Market Matcher — find matching markets between Polymarket and Kalshi.

Two-stage approach:
  1. Check cache (market_matches table) for confirmed/previous matches
  2. Tight date-window candidates + MiniLM cosine similarity on normalised
     participant names
"""

import re
import time
from datetime import datetime

import numpy as np
from sqlalchemy import text

from src.common.db import get_engine

# ── Lazy-loaded embedding model ─────────────────────────────────────────────

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


# ── Normalise titles into embed strings ─────────────────────────────────────


def build_embed_text(title: str, date: str) -> str:
    """
    Extract participant names from a market title/question and combine with
    date.  *date* should be a ``YYYY-MM-DD`` string.
    """
    text_ = title

    # Strip league prefixes like "AHL: ", "NBA: ", "LoL: "
    # Only strip if what follows looks like a matchup (contains "vs" or team names),
    # not if the colon is mid-sentence (e.g. "1H Spread: ...")
    text_ = re.sub(
        r"^[A-Za-z0-9]{2,10}:\s*(?=[A-Z])", "", text_, count=1
    )

    # Strip prop suffixes after colon  e.g. ": O/U 138.5", ": Total Games"
    # But only if there's a "vs" before the colon (means the matchup is on the left)
    colon_idx = text_.find(":")
    if colon_idx > 0 and re.search(r"\bvs\.?\b", text_[:colon_idx], re.IGNORECASE):
        text_ = text_[:colon_idx].strip()

    # Strip womens suffix
    text_ = text_.replace("(W)", "").strip()

    # Find matchup via "vs" token
    vs_pattern = re.compile(r"\bvs\.?\s*", re.IGNORECASE)
    matches = list(vs_pattern.finditer(text_))

    if matches:
        last_vs = matches[-1]

        # Team A — walk left, strip leading noise
        left = text_[: last_vs.start()].strip().rstrip(".")
        left = re.sub(
            r"^.*?\b(in the|during the|during)\b\s*",
            "",
            left,
            flags=re.IGNORECASE,
        ).strip()
        left = re.sub(
            r"^Will\s+\S+\s+win\s+(set\s+\d+\s+in\s+the\s+)?",
            "",
            left,
            flags=re.IGNORECASE,
        ).strip()

        # Team B — walk right, stop at common boundary words or parens
        right = text_[last_vs.end() :].strip()
        right = re.split(
            r"\s+(match|game|professional|mma|winner|fight|originally)\b"
            r"|\s*\(|\s+-\s+",
            right,
            flags=re.IGNORECASE,
        )[0].strip()
        right = right.rstrip("?").strip()

        if left and right:
            return f"{left} {right} {date}"

    # Fallback: cleaned full text + date
    text_ = text_.rstrip("?").strip()
    return f"{text_} {date}"


# ── SQL ─────────────────────────────────────────────────────────────────────

_POLY_LOOKUP_SQL = text("""
    SELECT DISTINCT ON (event_slug)
        event_slug, question, end_date, last_trade_price
    FROM polymarket.polymarket_market
    WHERE slug = :slug
    ORDER BY event_slug, end_date DESC NULLS LAST
""")

_KALSHI_CANDIDATES_SQL = text("""
    SELECT DISTINCT ON (event_ticker)
        event_ticker, title, close_time,
        expected_expiration_time, implied_probability
    FROM kalshi.kalshi_markets
    WHERE ticker NOT LIKE 'KXMV%'
      AND expected_expiration_time BETWEEN :lo AND :hi
    ORDER BY event_ticker, expected_expiration_time
""")

_CACHE_LOOKUP_SQL = text("""
    SELECT kalshi_event_ticker, score, confidence, method
    FROM market_matches
    WHERE poly_event_slug = :event_slug
      AND confirmed = TRUE
    LIMIT 1
""")

_CACHE_UPSERT_SQL = text("""
    INSERT INTO market_matches
        (poly_event_slug, kalshi_event_ticker,
         poly_embed_text, kalshi_embed_text,
         score, confidence, method)
    VALUES
        (:poly_event_slug, :kalshi_event_ticker,
         :poly_embed_text, :kalshi_embed_text,
         :score, :confidence, :method)
    ON CONFLICT (poly_event_slug, kalshi_event_ticker) DO UPDATE SET
        score      = EXCLUDED.score,
        confidence = EXCLUDED.confidence,
        method     = EXCLUDED.method,
        poly_embed_text  = EXCLUDED.poly_embed_text,
        kalshi_embed_text = EXCLUDED.kalshi_embed_text
""")

_CREATE_CACHE_TABLE_SQL = text("""
    CREATE TABLE IF NOT EXISTS market_matches (
        poly_event_slug     TEXT NOT NULL,
        kalshi_event_ticker TEXT NOT NULL,
        poly_embed_text     TEXT,
        kalshi_embed_text   TEXT,
        score               FLOAT,
        confidence          VARCHAR(10),
        method              VARCHAR(20),
        confirmed           BOOLEAN DEFAULT FALSE,
        created_at          TIMESTAMPTZ DEFAULT now(),
        PRIMARY KEY (poly_event_slug, kalshi_event_ticker)
    )
""")


# ── Internals ───────────────────────────────────────────────────────────────


def _ensure_cache_table(conn):
    conn.execute(_CREATE_CACHE_TABLE_SQL)
    conn.commit()


def _confidence(score: float) -> str:
    if score >= 0.90:
        return "high"
    if score >= 0.75:
        return "medium"
    return "low"


def _fetch_candidates(conn, end_date, hours: int):
    """Return Kalshi candidates within ±hours of end_date."""
    from datetime import timedelta

    lo = end_date - timedelta(hours=hours)
    hi = end_date + timedelta(hours=hours)
    return conn.execute(_KALSHI_CANDIDATES_SQL, {"lo": lo, "hi": hi}).mappings().all()


def _find_poly_to_kalshi(slug: str, engine) -> dict:
    t0 = time.perf_counter()

    with engine.connect() as conn:
        _ensure_cache_table(conn)

        # 1. Look up poly market
        poly = conn.execute(_POLY_LOOKUP_SQL, {"slug": slug}).mappings().first()
        if poly is None:
            return _result(slug, None, [], "poly_not_found", t0)

        event_slug = poly["event_slug"]
        question = poly["question"]
        end_date = poly["end_date"]
        last_trade_price = float(poly["last_trade_price"] or 0)

        if end_date is None:
            return _result(slug, None, [], "no_end_date", t0)

        # 2. Check cache
        cached = conn.execute(
            _CACHE_LOOKUP_SQL, {"event_slug": event_slug}
        ).mappings().first()
        if cached:
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)
            return {
                "input": slug,
                "poly_event_slug": event_slug,
                "direction": "poly_to_kalshi",
                "match": {
                    "event_ticker": cached["kalshi_event_ticker"],
                    "score": float(cached["score"]),
                    "confidence": cached["confidence"],
                },
                "candidates": [],
                "method": "cache",
                "duration_ms": duration_ms,
            }

        # 3. Get Kalshi candidates — ±6h first, widen to ±24h if empty
        candidates = _fetch_candidates(conn, end_date, hours=6)
        if not candidates:
            candidates = _fetch_candidates(conn, end_date, hours=24)

        if not candidates:
            return _result(slug, None, [], "no_kalshi_candidates", t0,
                           poly_event_slug=event_slug)

        # 4. Build embed texts
        date_str = end_date.strftime("%Y-%m-%d")
        poly_embed_text = build_embed_text(question, date_str)
        kalshi_texts = [
            build_embed_text(
                c["title"],
                (c["expected_expiration_time"] or c["close_time"]).strftime(
                    "%Y-%m-%d"
                ),
            )
            for c in candidates
        ]

        # 5. Encode and rank
        model = _get_model()
        poly_vec = model.encode(poly_embed_text, normalize_embeddings=True)
        kalshi_vecs = model.encode(kalshi_texts, normalize_embeddings=True)
        scores = np.dot(kalshi_vecs, poly_vec)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best = candidates[best_idx]
        confidence = _confidence(best_score)

        match_out = {
            "event_ticker": best["event_ticker"],
            "title": best["title"],
            "close_time": str(best["close_time"]),
            "implied_probability": round(
                float(best["implied_probability"] or 0), 4
            ),
            "embed_text": kalshi_texts[best_idx],
            "score": round(best_score, 4),
            "confidence": confidence,
        }

        all_candidates = sorted(
            [
                {
                    "event_ticker": c["event_ticker"],
                    "title": c["title"],
                    "score": round(float(scores[i]), 4),
                }
                for i, c in enumerate(candidates)
            ],
            key=lambda x: x["score"],
            reverse=True,
        )

        # 6. Write to cache
        conn.execute(
            _CACHE_UPSERT_SQL,
            {
                "poly_event_slug": event_slug,
                "kalshi_event_ticker": best["event_ticker"],
                "poly_embed_text": poly_embed_text,
                "kalshi_embed_text": kalshi_texts[best_idx],
                "score": round(best_score, 4),
                "confidence": confidence,
                "method": "minilm_cosine",
            },
        )
        conn.commit()

        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "input": slug,
            "poly_event_slug": event_slug,
            "poly_embed_text": poly_embed_text,
            "direction": "poly_to_kalshi",
            "match": match_out,
            "candidates": all_candidates,
            "method": "minilm_cosine",
            "duration_ms": duration_ms,
        }


def _result(slug, match, candidates, method, t0, **extra):
    duration_ms = round((time.perf_counter() - t0) * 1000, 1)
    out = {
        "input": slug,
        "direction": "poly_to_kalshi",
        "match": match,
        "candidates": candidates,
        "method": method,
        "duration_ms": duration_ms,
    }
    out.update(extra)
    return out


# ── Public API ──────────────────────────────────────────────────────────────


def find_match(slug_or_ticker: str, direction: str = "poly_to_kalshi") -> dict:
    """
    Find the matching market on the other platform.

    Parameters
    ----------
    slug_or_ticker : str
        Polymarket slug (when direction='poly_to_kalshi').
    direction : str
        Only 'poly_to_kalshi' is supported for now.

    Returns
    -------
    dict with keys: input, direction, match, candidates, method, duration_ms.
    """
    if direction != "poly_to_kalshi":
        raise NotImplementedError(f"Direction '{direction}' not yet supported")

    engine = get_engine()
    try:
        return _find_poly_to_kalshi(slug_or_ticker, engine)
    finally:
        engine.dispose()


def match_batch(
    slugs: list[str], direction: str = "poly_to_kalshi"
) -> list[dict]:
    """Run find_match for a list of slugs, reusing a single engine."""
    if direction != "poly_to_kalshi":
        raise NotImplementedError(f"Direction '{direction}' not yet supported")

    engine = get_engine()
    try:
        return [_find_poly_to_kalshi(slug, engine) for slug in slugs]
    finally:
        engine.dispose()
