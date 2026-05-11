"""Quick smoke test for the copy_target scorecard SQL template."""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.db import get_engine
from sqlalchemy import text


def fill_defaults(sql: str, **overrides) -> str:
    """Replace $$Value_{name} placeholders. overrides[name] = literal SQL value."""
    for name, val in overrides.items():
        sql = sql.replace(f"$$Value_{{{name}}}", val)
    sql = re.sub(r"\$\$Value_\{[^}]+\}", "''", sql)
    return sql


def main():
    eng = get_engine()

    t = time.time()
    print(f"[+{time.time()-t:5.1f}s] Connecting and pulling top 10 from h_score...", flush=True)
    hscore_sql = open("src/models/hscore/query_template.sql").read()
    hscore_sql = fill_defaults(hscore_sql)
    with eng.connect() as c:
        rows = c.execute(text(hscore_sql)).fetchall()
    top10 = [r[1] for r in rows[:10]]
    print(f"[+{time.time()-t:5.1f}s] Got {len(rows)} h_score rows; top10 captured", flush=True)
    for r in rows[:10]:
        print(f"   #{r[0]:<3} {r[1][:14]}... h={r[3]} pnl=${r[9]:,.0f}", flush=True)

    print(f"\n[+{time.time()-t:5.1f}s] Running copy_target scorecard on those 10...", flush=True)
    ct_sql = open("src/models/copy_target/query_template.sql").read()
    csv = ",".join(top10)
    # Replace wallet_addresses with the actual CSV (quoted string in SQL)
    ct_sql = ct_sql.replace(
        "$$Value_{wallet_addresses}",
        f"'{csv}'",  # becomes COALESCE(NULLIF('0x...,0x...', ''), '')
    )
    # Strip remaining placeholders -> NULL via empty-string trick
    ct_sql = re.sub(r"\$\$Value_\{[^}]+\}", "''", ct_sql)

    with eng.connect() as c:
        result = c.execute(text(ct_sql))
        cols = list(result.keys())
        ct_rows = result.fetchall()
    print(f"[+{time.time()-t:5.1f}s] copy_target returned {len(ct_rows)} rows (expected 10)", flush=True)

    if not ct_rows:
        print("NO ROWS - check that the top 10 wallets are in the eligible universe.", flush=True)
        return

    # Build hscore_rank lookup
    hscore_rank = {w: i+1 for i, w in enumerate(top10)}
    idx = {c: i for i, c in enumerate(cols)}

    # Sort by composite_score desc
    ct_rows = sorted(ct_rows, key=lambda r: -float(r[idx["composite_score"]]))

    print(f"\n=== Copy-target scorecard (sorted by composite) ===\n", flush=True)
    for r in ct_rows:
        w = r[idx["wallet"]]
        print(
            f"hsc#{hscore_rank.get(w, '?'):<2}  {w[:14]}...  tier={r[idx['tier']]:<11} "
            f"comp={r[idx['composite_score']]:>5}  "
            f"sk={r[idx['skill_score']]:>5} sp={r[idx['specialization_score']]:>5} "
            f"cp={r[idx['copyability_score']]:>5} rk={r[idx['risk_score']]:>5}  "
            f"cat={str(r[idx['dominant_category']])[:8]:<8} "
            f"hhi={float(r[idx['category_hhi']]):.2f}  "
            f"pmr={float(r[idx['profitable_market_rate']]):.2f}  "
            f"size=${float(r[idx['avg_trade_size']]):>7,.0f}  "
            f"freq={float(r[idx['trades_per_week']]):>5.1f}/w  "
            f"posd={float(r[idx['positive_days_pct']]):>4.1f}%",
            flush=True,
        )
        rat = r[idx["rationale"]]
        if rat:
            print(f"     -> {rat}", flush=True)
    eng.dispose()
    print(f"\n[+{time.time()-t:5.1f}s] Done.", flush=True)


if __name__ == "__main__":
    main()
