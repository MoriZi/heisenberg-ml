# Heisenberg H-Score Backtesting — Project Context

## What We're Building
A backtesting and weight-optimisation pipeline for the H-Score, Heisenberg's proprietary trader ranking metric for Polymarket. The pipeline produces empirically-grounded component weights to replace the current hand-tuned formula.

---

## Database

- **Engine**: PostgreSQL (AWS)
- **Schema**: `polymarket`
- **Connection**: Use psycopg2 or SQLAlchemy. Ask the project owner for credentials before starting.

### Tables Used in This Project

#### `polymarket.wallet_profile_metrics` — PRIMARY SOURCE
Pre-computed daily snapshots per wallet per window. 67 columns. Covers 2025-07-10 to present.

**Key columns:**
```
date                        DATE       — snapshot date
proxy_wallet                TEXT       — wallet address
calculation_window_days     INT        — 1, 3, 7, or 15. Always filter on 15 for scoring.
total_pnl                   FLOAT
roi                         FLOAT
win_rate                    FLOAT
total_trades                INT
markets_traded              INT
sharpe_ratio                FLOAT      — nullable
sortino_ratio               FLOAT      — nullable
calmar_ratio                FLOAT      — nullable
max_drawdown                FLOAT
profit_factor               FLOAT
edge_decay                  FLOAT      — nullable
performance_trend           TEXT       — 'improving' / 'stable' / 'declining'
market_concentration_ratio  FLOAT
category_diversity_score    FLOAT
curve_smoothness            FLOAT
position_size_consistency   FLOAT
suspicious_win_rate_flag    BOOL
sybil_risk_score            FLOAT
sybil_risk_flag             BOOL
combined_risk_score         FLOAT
risk_level                  TEXT       — 'LOW' / 'MEDIUM' / 'HIGH'
performance_by_category     JSONB      — array of {pnl, trades, category}
```

**JSONB example:**
```json
[{"pnl": 24.0, "trades": 15, "category": "Crypto"},
 {"pnl": 20.7, "trades":  5, "category": "Science & Tech"}]
```
Categories: Sports, Crypto, World Events, Economics, Science & Tech, Other

**Always filter like this — never full-scan:**
```sql
WHERE date = '2026-01-15'
  AND calculation_window_days = 15
```

#### `polymarket.wallet_daily_pnl` — ELIGIBILITY + LABELS ONLY (41GB — filter aggressively)
```
proxy_wallet    TEXT
date            DATE
condition_id    TEXT    — one row per market per day
pnl             FLOAT
invested        FLOAT
trades          INT
wins            INT
losses          INT
```
Note: Rows with trades=0, invested=0 but non-zero pnl are settlement payouts. Include them — intentional and correct.

#### `polymarket.leaderboard` — MATERIALIZED VIEW (for rank reconstruction)
```
period      TEXT    — '1d', '3d', '7d', '30d'
address     TEXT    — wallet address (note: 'address' not 'proxy_wallet')
rank        INT
roi         FLOAT
win_rate    FLOAT
sharpe_ratio FLOAT
total_trades INT
markets_traded INT
total_pnl   FLOAT
total_volume FLOAT
```

---

## Current H-Score Formula

Six components, 100 points total. Currently uses 30d leaderboard data.

```python
consistency_score    = min(25.0 / avg_rank * 50, 25)          # 25 pts
roi_stability_score  = min(roi_30d / roi_7d, 1.0) * 20        # 20 pts  (if roi_7d > 0 else 0)
win_rate_score       = min(max(win_rate - 0.5, 0) / 0.45 * 20, 20)  # 20 pts
sharpe_score         = min(max(sharpe_ratio, 0) * 3, 15)       # 15 pts
diversification_score = step(markets_traded, [5,10,20,50], [3,5,8,10])  # 10 pts
sample_score         = step(total_trades, [100,500,10000,50000], [6,10,4,2])  # 10 pts

h_score = sum of all components
```

avg_rank = (rank_1d + rank_3d + rank_7d + rank_30d) / 4.0

trajectory:
- Improving: rank_1d < rank_7d < rank_30d
- Decaying:  rank_1d > rank_7d > rank_30d
- Stable:    otherwise

---

## Window Strategy — Hybrid

| Purpose | Window | Source |
|---------|--------|--------|
| Eligibility filtering | 30d | wallet_daily_pnl (reconstructed) |
| Scoring inputs | 15d | wallet_profile_metrics (window=15) |
| Forward label | 14d | wallet_daily_pnl (reconstructed) |

---

## Eligibility Filters (30d)

Apply these before scoring. Reconstruct from wallet_daily_pnl:

```python
roi_30d > 0
win_rate_30d BETWEEN 0.45 AND 0.95
total_trades_30d BETWEEN 50 AND 100000
total_volume_30d > 10000
total_pnl_30d > 5000
trajectory != 'Decaying'
combined_risk_score <= 50   # from wallet_profile_metrics window=15
```

---

## Ground Truth Label

Binary label per wallet per snapshot date T:
- **label = 1** if: forward_pnl > 0 AND forward_rank <= 500 (in T+1 to T+14)
- **label = 0** otherwise

Expect ~20-30% positive labels (class imbalance — handle with class_weight='balanced' in LightGBM).

---

## Snapshot Dates

Cadence: every 14 days starting 2025-07-24.
Last usable snapshot: ~2026-02-16 (needs 14 days forward for label).
Expected: ~14 usable snapshots.

Generate with:
```python
import pandas as pd
snapshots = pd.date_range(start='2025-07-24', end='2026-02-16', freq='14D')
```

---

## Scripts to Build (in order)

1. `build_eligibility.py` — 30d eligibility flags per wallet per snapshot date
2. `build_features.py`    — 15d features from wallet_profile_metrics + JSONB parsing
3. `build_labels.py`      — 14d forward labels from wallet_daily_pnl
4. `correlation_analysis.py` — Pearson/Spearman correlation table vs label
5. `weight_optimizer.py`  — scipy SLSQP weight search (Variant A: existing, Variant B: new features)
6. `lgbm_importance.py`   — LightGBM feature importance + walk-forward CV
7. `evaluate.py`          — unified metrics harness for any weight vector

## Output Files

```
features.parquet          # wallet x snapshot_date x all features
labels.parquet            # forward labels aligned to features
correlation_table.csv     # features ranked by Spearman rho vs label
optimal_weights_A.json    # best weights, existing 6 components
optimal_weights_B.json    # best weights, with new candidate features
lgbm_importances.csv      # LightGBM gain-based feature importances
lgbm_importances.png      # bar chart top 20 features
evaluation_report.md      # metrics: current vs optimised A vs optimised B
```

---

## Critical Constraints

1. **Never query polymarket_trade** — 825GB, will time out
2. **Always filter wallet_profile_metrics by date AND calculation_window_days** — never full-scan
3. **wallet_daily_pnl is 41GB** — always filter by date range first
4. **Walk-forward CV only** — never shuffle across time, never use sklearn KFold
5. **Include settlement rows** — trades=0, invested=0, pnl!=0 rows are valid, do not filter out
6. **Null handling** — sortino_ratio, calmar_ratio, edge_decay can be null. Use fillna(0) or median. Document choice.

---

## Validation Checkpoints

After each script, validate against these known wallets before proceeding:

| Alias | Address | Expected domain |
|-------|---------|-----------------|
| cf11 (NBA) | 0xcf119e969f31de9653a58cb3dc213b485cd48399 | Sports / Basketball |
| ccb2 (CS2) | 0xccb290b1c145d1c95695d3756346bba9f1398586 | Esports + Soccer |
| 916f (UCL) | 0x916f7165c2c836aba22edb6453cdbb5f3ea253ba | Sports / Soccer |
| d008 (selective) | 0xd008786fad743d0d5c60f99bff5d90ebc212135d | Sports / Soccer |

cf11 known metrics (approx, 30d as of Feb 2026): ROI ~27.8%, win_rate ~73.6%, total_pnl ~$248k

---

## Tech Stack

```
psycopg2 or sqlalchemy     DB connection
pandas, numpy              data manipulation
scipy.optimize             weight optimisation (SLSQP)
lightgbm                   feature importance + classification
sklearn.metrics            roc_auc_score, precision_score
matplotlib                 feature importance chart
```

Install:
```bash
pip install psycopg2-binary sqlalchemy pandas numpy scipy lightgbm scikit-learn matplotlib pyarrow
```
