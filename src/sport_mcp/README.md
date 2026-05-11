# Heisenberg Sport MCP — Data Agent Inventory

Parameterized SQL data agents that power Heisenberg's sport prediction-market MCP.
Each `.sql` file maps 1:1 to a production agent ID (the filename). Edit here, copy
into the production agent interface.

## How this folder works

| Path | Purpose |
| --- | --- |
| `*.sql` (numeric name) | Live agent. Filename = production agent ID. |
| `_backups/` | Timestamped copy of every agent taken before each material edit. **Required** — see `_backups/README` rule below. |
| `_drafts/` | New agents not yet registered in production. Move to top-level with the assigned ID after registration. |
| `README.md` | This file. Single source of truth for "what does agent N do." |

**Before editing any agent**, copy the current version into `_backups/<id>_<reason>_<YYYYMMDD>.sql`.
Untracked files are not recoverable from git.

**Rules for the SQL itself** (learned the hard way):
1. The file must start with `WITH` or `SELECT` on line 1 — **no leading `-- comment` blocks**. The production binder fails with `"could not find name <empty>"` if any `--` line precedes the first SQL statement. Keep provenance in this README instead.
2. **No `;` inside `-- comments`** — the binder treats them as statement terminators. Only the file-terminating `;` is allowed.
3. **Pure ASCII** in the SQL body. Em-dashes, arrows, etc. have tripped the validator before.
4. **Avoid SQL keyword collisions** in CTE names. `WITH window AS (...)` fails; use `WITH requested_window AS (...)`.

---

## Agent inventory (audit-grade)

Each block: purpose · inputs · outputs · tables · subjective thresholds · open issues.

### 591 — Sport sharp active positions (universe scan)

**Purpose:** For every wallet that qualifies as "sharp on sports," list their active
positions in live sport markets (last_trade_price 0.02–0.98). Hedgers (wallets holding
both YES and NO in the same market) are excluded.

**Inputs**
- `min_sport_pnl` (default 10000) — minimum 15d sport PnL to qualify as sharp.
- `min_trades` (default 5) — minimum 15d sport trade count.
- `sport` (default 'Sports') — sport substring filter on tag_category.
- `min_shares` (default 100) — minimum position size in shares.
- `limit` (default 50).

**Output cols**
proxy_wallet, question, outcome, shares, avg_cost, current_price, unrealized_pnl,
market_volume, liquidity, end_date, slug, sport_pnl_15d, sport_roi_15d,
sport_win_rate_pct_15d, sport_trades_15d.

**Tables:** `wallet_profile_metrics_v2`, `wallet_profile_metrics_category_v2`, `wallet_positions`, `polymarket_market`, `polymarket_market_tag`, `tag_category`.

**Subjective thresholds:**
- `combined_risk_score <= 50` — Heisenberg risk-cliff (eligibility filter, not feature).
- `sybil_risk_flag = false`, `suspicious_win_rate_flag = false` — boolean kills.
- `total_pnl >= min_sport_pnl` (default $10,000) **and** sport PnL ≥ same threshold — high bar.
- `last_trade_price BETWEEN 0.02 AND 0.98` — excludes nearly-resolved markets.
- `min_shares >= 100` — count, not notional. $5 floor on a 5¢ market, $95 floor on a 95¢ market.

**Open issues**
- "Sharp" definition is local to this query — diverges from 593/594/595. See *Cross-agent inconsistencies* below.
- `min_shares` should be `min_notional_usd` for fair comparison across price levels.

---

### 593 — Per-market sharp summary

**Purpose:** Given a single market (condition_id), aggregate the sharp-money picture:
how many sharps are on each side, their average win-rate, total shares, average entry price.

**Inputs**
- `condition_id` (required).
- `sport` (optional) — overrides tag-derived sport.
- `min_shares` (default 100).
- `min_sport_pnl` (default **250**).

**Output cols** (one row per outcome side)
question, slug, detected_sport, current_price, market_volume, liquidity, end_date,
outcome, sharp_wallet_count, total_sharp_shares, avg_win_rate_pct, avg_sport_roi,
total_sport_pnl, avg_entry_price.

**Tables:** v2 + category JSONB on v2, `wallet_positions`, `polymarket_market`.

**Subjective thresholds:**
- `min_sport_pnl = 250` default — **40× weaker** than 591's $10,000.
- Same risk-flag set as 591.
- Hedger exclusion via `COUNT(DISTINCT outcome) > 1`.

**Open issues**
- `$250` vs `$10,000` minimum sport-PnL inconsistency — same wallet may be "sharp" here but not in 591.
- Output column inconsistency: `detected_sport` here vs no sport column in 591.

---

### 594 — Per-market sharp positions (individual rows)

**Purpose:** Same as 593 but un-aggregated — one row per (sharp wallet × outcome) in the market.

**Inputs**
- `condition_id` (required), `sport`, `min_shares` (100), `min_sport_pnl` (default **1000**), `limit` (100).

**Output cols** include per-wallet detail: proxy_wallet, outcome, shares, avg_cost, unrealized_pnl, sport PnL/ROI/win_rate/trades.

**Subjective thresholds:**
- `min_sport_pnl = 1000` default — **a third value** vs 591 ($10k) and 593 ($250).

**Open issues**
- Three different "sharp" cutoffs in three position-finder agents.

---

### 595 — Sport H-Score leaderboard

**Purpose:** Ranked list of sport-active wallets by trained Sport H-Score. Inputs
let callers loosen/tighten the eligibility filters; output schema matches the general
H-Score leaderboard.

**Inputs**
- `min_pnl_15d` (1000), `min_total_trades_15d` (10), `max_total_trades_15d` (200000),
  `min_win_rate_15d` (0), `max_win_rate_15d` (0.95), `min_roi_15d` (0).
- `sort_by` (sport_h_score | pnl | roi | win_rate | trades).

**Output cols**
leaderboard_rank, wallet, tier (Elite/Sharp/Solid/Emerging), sport_h_score,
sports_pnl_15d, sports_trades_15d, win_rate_pct_15d, sharpe_ratio_15d, sortino_ratio_15d,
total_trades_15d, markets_traded_15d, total_pnl_15d, total_volume_15d, trajectory, sport_tags.

**Tables:** v2 + category_v2 (15d + 7d windows).

**Subjective thresholds and frozen constants:**
- 19 hand-coded weights summing to 100, embedded in SQL. **Provenance unknown.** Don't match `sport_hscore/artifacts/optimal_weights.json` — likely from an older training run.
- `COALESCE(sortino_ratio, 1.0094)` and `COALESCE(annualized_return, 3.18675)` — frozen medians from a past snapshot. Population shifts; these don't.
- Tier thresholds Elite ≥ 70, Sharp ≥ 50, Solid ≥ 35 — **absolute**, not percentile-based. As the score distribution shifts, the tier mix shifts arbitrarily.

**Open issues**
- Weights need to be reconciled with the live `optimal_weights.json` (resync after every retrain).
- Median fallbacks need to become within-pool `PERCENTILE_CONT`.
- Tier thresholds should be percentile-based for stability.

---

### 596 — Price jumps over candlesticks

**Purpose:** Find buckets where price changed > threshold over a chosen resolution
(1m/5m/15m/1h/4h/1d) for a token, within a lookback window.

**Inputs:** `token_id`, `lookback_hours` (capped at 168), `min_change_pct` (floored at 1), `resolution`.

**Tables:** `polymarket.candlestick`.

**Subjective thresholds:**
- `lookback_hours` capped at 168 (1 week) — sane bound.
- `min_change_pct` floored at 1% — moderate.

**Open issues**
- None major. The agent is well-bounded.

---

### 597 — Sport trades by market/wallet  ⚠ TABLE-SAFETY RISK

**Purpose:** Raw trades on sport markets matching a slug prefix, optionally filtered by
condition_id / side / proxy_wallet / time range.

**Inputs:** `market_slug`, `condition_id` ('ALL' or value), `side`, `proxy_wallet`,
`start_time` (epoch), `end_time` (epoch).

**Tables:** `polymarket.polymarket_trade` (825 GB), `polymarket_market`.

**Subjective thresholds:** none — but **no required time bound**. If a caller passes
`start_time=1600000000, end_time=2200000000` (the sentinels), the time predicate is
disabled and the query scans the whole trade table for any sport market in the slug
prefix. That's a 825-GB risk vector.

**Open issues**
- Time-window cap must be enforced in SQL (e.g., refuse if end-start > 30 days).
- Even with slug filter, the unbounded time variant will be slow for large slug families.

---

### 599 — Wallet 360 (general + sport)

**Purpose:** Pull general + per-sport metrics for a single wallet at a given window.

**Inputs:** `proxy_wallet`, `window_days` (1/3/7/15), `sport` (substring).

**Tables:** `wallet_profile_metrics_v2`, `wallet_profile_metrics_category_v2`.

**Subjective thresholds:** none.

**Open issues**
- Returns multiple rows when wallet has multiple matching sport subcategories — output schema is per-row not per-wallet. Document.

---

### 600 — Market quality insights (sport filter)

**Purpose:** Surface markets matching quality/whale/liquidity criteria for a given sport.
Wraps the `market_quality_insights` materialized view.

**Inputs:** `condition_id`, `min_volume_24h`, `min_liquidity_percentile`, `volume_trend`,
`min_top1_wallet_pct`, `max_unique_traders_7d`, `sport`.

**Tables:** `polymarket.market_quality_insights` (materialized view), `polymarket_market`, tag join.

**Subjective thresholds:** all configurable. Defaults are `'0'`/`'ALL'` which disable the filter — caller must opt in.

**Open issues**
- Underlying materialized view definition not documented here — should be linked or summarized so callers know what `liquidity_percentile`, `whale_control_flag`, `squeeze_risk_flag` actually mean.

---

### 601 — Wallet sport PnL history (granular)

**Purpose:** Aggregated daily PnL for one wallet on sport markets, bucketed by granularity (1d/3d/1w/1m/all).

**Inputs:** `wallet`, `start_time`, `end_time`, `granularity`, `condition_id` (or 'ALL'), `sport`.

**Tables:** `wallet_daily_pnl_v2`, `polymarket_market`, tag join.

**Subjective thresholds:** none.

**Open issues**
- No max-window cap on `(end_time - start_time)`. Sane wallet-bounded query but worth a hard cap.

---

### 602 — Closing-line VWAP  ⚠ TABLE-SAFETY RISK

**Purpose:** Compute volume-weighted-average closing price per outcome in the 10 minutes
before a market's `end_date`. The basis for CLV computation.

**Inputs:** `condition_id` (required).

**Tables:** `polymarket.polymarket_market`, `polymarket.polymarket_trade` (825 GB).

**Subjective thresholds:**
- 10-minute closing window is hard-coded. Defensible default; should be a parameter (`closing_window_minutes`).

**Open issues**
- Trade-table scan bounded by condition_id + 10-min window. With condition_id index this is fast in practice — but the unbounded `start_time` problem in 597 means we should standardize a "trade-table safety policy" across all agents that touch it.

---

## Cross-agent inconsistencies (the audit's #1 finding)

**"Sharp wallet" is defined differently in every position-finder agent:**

| Agent | min combined_risk | min sport_pnl | other |
| --- | --- | --- | --- |
| 591 | ≤ 50 | **$10,000** | + total_pnl ≥ $10k |
| 593 | ≤ 50 | **$250** | — |
| 594 | ≤ 50 | **$1,000** | — |
| 595 (leaderboard, source of truth) | ≤ 50 | implicit via sport tier | win_rate 0–0.95, ROI > 0, trades 10–200k |

Same risk-flag set across all four, but the sport-PnL bar is 40× different between 591 and 593. A wallet that's "sharp" in 593 won't be in 591. This is the **first thing OddsJam/DraftKings will notice when they cross-check outputs.**

**Fix in Phase 1:** add `_drafts/598_sport_sharp_universe.sql` — a one-column wallet-list agent that returns "sharp universe" given a tier filter. 591/593/594 are refactored to consume it. Then "sharp" is defined exactly once, in 595's leaderboard, and changing it retunes everything.

---

## Cross-agent constant table (post-Phase-1 target)

These should all live in ONE place — the leaderboard (595) — and other agents inherit:

| Constant | Where today | Where it should live |
| --- | --- | --- |
| `combined_risk_score <= 50` | 591, 593, 594, 595 | 595 (and inherited via 598) |
| `sybil_risk_flag = false` | 591, 593, 594, 595 | 595 |
| `suspicious_win_rate_flag = false` | 591, 593, 594, 595 | 595 |
| `total_pnl > 1000` | 595 default | 595 |
| min sport_pnl | varies $250/$1k/$10k | from sport_hscore tier |
| Tier thresholds 70/50/35 | 595 | 595 (percentile-based, target Phase 2) |
| Score weights | 595 hard-coded | `sport_hscore/artifacts/optimal_weights.json` |
| Median fallbacks | 595 hard-coded | computed within-pool, mirror `query_template.sql` |

---

## Versioning policy

Every material edit to an agent must:

1. Copy the previous version into `_backups/<id>_<reason>_<YYYYMMDD>.sql`.
2. Update the agent's "Last updated / weights source" line below.
3. Update the audit section above if any threshold, filter, or scoring weight changed.

| Agent | Last updated | Source artifact / change |
| --- | --- | --- |
| 591 | 2026-05-11 | unified sharp eligibility (inlined in `sport_wallets` subquery to keep planner happy); default `min_sport_pnl` lowered $10k→$1k. 4s end-to-end. |
| 593 | 2026-05-11 | unified sharp eligibility (inlined in `eligible_metrics`); default `min_sport_pnl` raised $250→$1k. 0.1s. |
| 594 | 2026-05-11 | unified sharp eligibility (inlined in `eligible_metrics`). 0.03s. |
| 595 | 2026-05-11 | regenerated by `scripts/regenerate_595.py` from `sport_hscore/artifacts/optimal_weights.json` (27 features, retrained 2026-05-11 on 2026-03-05 → 2026-05-04). Within-pool medians, **percentile-based tiers** (top 5/15/35%). 0.4s. |
| 596 | (original) | — |
| 597 | 2026-05-11 | enforced 30-day max trade-window cap; sentinels default to last 7 days. CTE renamed `window`→`requested_window` (SQL keyword collision). |
| 599 | (original) | — |
| 600 | (original) | — |
| 601 | (original) | — |
| 602 | (original) | — |

---

## Drafts queue

Agents in `_drafts/` waiting on a production agent ID:

| File | Status | Purpose |
| --- | --- | --- |
| `598_sport_sharp_universe.sql` | Pending registration | Single source-of-truth for "is this wallet sharp on sports?" Returns wallet list filtered by tier and (optional) sport. Tested against live DB on 2026-05-11. Recommended for OddsJam/DraftKings to see *the canonical sharp definition* in one place. |
