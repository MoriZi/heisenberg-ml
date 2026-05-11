"""
Regenerate src/sport_mcp/595.sql from
src/models/sport_hscore/artifacts/optimal_weights.json.

DESIGN PRINCIPLE: do not change the template. The production binder is fragile
about leading comments, extra CTEs, and re-aliased columns. This script only
touches two things:

  1. The two hard-coded median constants in the `base` CTE
     (`COALESCE(g.sortino_ratio, X) AS sortino_ratio` -> new X from artifact).
  2. The 19 PERCENT_RANK weight values inside the `scored` CTE.

The template SLOTS are fixed: 19 features in the scored block. The trained model
has 27 features. The 8 features the template doesn't expose (calmar_ratio,
sports_pnl_bundesliga/wnba/golf/college_football/f1, total_pnl_3d,
total_invested_3d) are dropped and the remaining 19 weights are renormalised
to sum to 100.

Idempotent: running twice with the same artifact produces the same output.

Run: python scripts/regenerate_595.py
"""

import json
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "src/models/sport_hscore/artifacts/optimal_weights.json"
TARGET   = ROOT / "src/sport_mcp/595.sql"
BACKUP   = ROOT / "src/sport_mcp/_backups" / f"595_pre-595-regen_{date.today().strftime('%Y%m%d')}.sql"


# The 19 features the template's `scored` CTE exposes, in the exact ORDER they appear,
# mapped to (json_feature_name, pr_alias_in_sql, sort_direction).
# This list is locked to the template — do NOT add features here without first
# adding the matching PERCENT_RANK line in 595.sql.
TEMPLATE_SLOTS = [
    ("total_pnl",                "pr_total_pnl",                "ASC"),
    ("best_market_pnl",          "pr_best_market_pnl",          "ASC"),
    ("total_invested_7d",        "pr_total_invested_7d",        "ASC"),
    ("win_rate_7d",              "pr_win_rate_7d",              "DESC"),
    ("sports_trades",            "pr_sports_trades",            "ASC"),
    ("best_market_pnl_7d",       "pr_best_market_pnl_7d",       "ASC"),
    ("worst_market_pnl_7d",      "pr_worst_market_pnl_7d",      "DESC"),
    ("annualized_return",        "pr_annualized_return",        "ASC"),
    ("profitable_markets_count", "pr_profitable_markets_count", "ASC"),
    ("avg_market_exposure",      "pr_avg_market_exposure",      "ASC"),
    ("pnl_cat_sports_7d",        "pr_pnl_cat_sports_7d",        "ASC"),
    ("sports_invested",          "pr_sports_invested",          "ASC"),
    ("worst_market_pnl",         "pr_worst_market_pnl",         "DESC"),
    ("sports_pnl_mlb",           "pr_sports_pnl_mlb",           "ASC"),
    ("sortino_ratio",            "pr_sortino_ratio",            "ASC"),
    ("sports_pnl",               "pr_sports_pnl",               "ASC"),
    ("sports_pnl_la_liga",       "pr_sports_pnl_la_liga",       "ASC"),
    ("sports_pnl_ligue_1",       "pr_sports_pnl_ligue_1",       "ASC"),
    ("profit_factor_7d",         "pr_profit_factor_7d",         "DESC"),
]

# Markers that bound the scored-CTE weight block in the SQL. Surgical replace
# happens between these two strings (exclusive of the markers themselves).
SCORED_START = "        ROUND((\n"
SCORED_END   = "        )::numeric, 1) AS sport_h_score"


def main():
    art = json.loads(ARTIFACT.read_text())
    json_feats = art["features"]
    json_weights = art["weights"]
    inverted = set(art.get("inverted", []))
    medians = art.get("medians", {})

    feat_to_w = dict(zip(json_feats, json_weights))

    # Pick the 19 weights for our template slots and check inversion direction.
    picked = []
    missing = []
    for feat, alias, direction in TEMPLATE_SLOTS:
        if feat not in feat_to_w:
            missing.append(feat)
            continue
        # Sanity: SQL direction should match training inversion.
        is_inverted_in_json = feat in inverted
        is_inverted_in_sql = (direction == "DESC")
        if is_inverted_in_json != is_inverted_in_sql:
            print(
                f"WARNING: inversion mismatch for {feat}: "
                f"json inverted={is_inverted_in_json}, sql dir={direction}"
            )
        picked.append((feat, alias, direction, feat_to_w[feat]))

    if missing:
        print(f"ERROR: artifact missing required features: {missing}")
        sys.exit(1)

    # Renormalise the 19 weights to sum to 100.
    raw_sum = sum(w for _, _, _, w in picked)
    if raw_sum <= 0:
        print(f"ERROR: artifact mapped weights sum to {raw_sum}")
        sys.exit(1)
    factor = 100.0 / raw_sum
    print(f"Mapped {len(picked)}/{len(json_feats)} features (raw sum={raw_sum:.4f}); renormalising x{factor:.6f}")

    # Build the scored-block expression matching the ORIGINAL template byte-for-byte:
    # single space between column and ASC/DESC, padding AFTER the closing paren, then
    # `* <weight>` (or `* <weight> +` for non-last lines).
    # Example from original:
    #     PERCENT_RANK() OVER (ORDER BY total_pnl ASC)                * 11.907 +
    #     PERCENT_RANK() OVER (ORDER BY profit_factor_7d DESC)        *  1.107
    lines = []
    n = len(picked)
    for i, (feat, alias, direction, raw_w) in enumerate(picked):
        norm_w = raw_w * factor
        sep = " +" if i < n - 1 else ""
        prefix = f"PERCENT_RANK() OVER (ORDER BY {feat} {direction})"
        # Pad prefix to width that puts `*` consistently aligned (matches original padding ~60).
        padded = prefix.ljust(62)
        line = f"            {padded}* {norm_w:>6.3f}{sep}"
        lines.append(line)
    new_scored_block = "\n".join(lines)

    # Read current 595 (the byte-for-byte original template restored from backup).
    src = TARGET.read_text()

    # Surgical edit 1: replace the two hard-coded medians constants.
    new_sortino  = medians.get("sortino_ratio")
    new_annret   = medians.get("annualized_return")
    if new_sortino is not None:
        src = _replace_coalesce(src, "g.sortino_ratio",     new_sortino, "AS sortino_ratio,")
    if new_annret is not None:
        src = _replace_coalesce(src, "g.annualized_return", new_annret,  "AS annualized_return,")

    # Surgical edit 2: replace the scored-block PERCENT_RANK lines, in place.
    start = src.find(SCORED_START)
    end   = src.find(SCORED_END)
    if start == -1 or end == -1 or end <= start:
        print("ERROR: could not locate scored-CTE markers in 595.sql.")
        sys.exit(1)
    before = src[: start + len(SCORED_START)]
    after  = src[end:]
    src = before + new_scored_block + "\n" + after

    # Backup and write.
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(TARGET, BACKUP)
    TARGET.write_text(src)

    print(f"Wrote {TARGET.relative_to(ROOT)}")
    print(f"Backup: {BACKUP.relative_to(ROOT)}")


def _replace_coalesce(src: str, col_ref: str, new_const: float, alias_suffix: str) -> str:
    """
    Find `COALESCE({col_ref}, <number>) {alias_suffix}` and rewrite the number.
    Idempotent: re-running with the same new_const produces the same string.
    """
    import re
    pattern = re.compile(
        rf"COALESCE\(\s*{re.escape(col_ref)}\s*,\s*[0-9.]+\s*\)\s+{re.escape(alias_suffix)}"
    )
    replacement = f"COALESCE({col_ref}, {new_const:.4f}) {alias_suffix}"
    new_src, n = pattern.subn(replacement, src)
    if n == 0:
        print(f"WARNING: did not find COALESCE for {col_ref}.")
    return new_src


if __name__ == "__main__":
    sys.exit(main() or 0)
