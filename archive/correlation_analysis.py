"""
correlation_analysis.py

Computes Pearson and Spearman correlation of every feature column
against the binary label, sorts by Spearman rho descending,
prints the full table, and saves to correlation_table.csv.

Usage:
    python correlation_analysis.py                    # default 7-day labels
    python correlation_analysis.py --forward-days 14  # 14-day labels

Inputs:  features.parquet, labels.parquet
Output:  correlation_table_{N}d.csv
"""

import argparse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# ── columns that are not features ─────────────────────────────────────────────
NON_FEATURE_COLS = {
    "proxy_wallet",
    "snapshot_date",
    "calculation_window_days",
    "date_range_start",
    "date_range_end",
    # label itself
    "label",
    "forward_pnl",
    "forward_rank",
}

# Text/categorical columns — excluded from correlation (encoding would be
# arbitrary and misleading; analyse separately if needed).
CATEGORICAL_COLS = {
    "performance_trend",
    "risk_level",
    "equity_curve_pattern",
    "dominant_category",
}


def load_data() -> pd.DataFrame:
    features = pd.read_parquet("features.parquet")
    labels   = pd.read_parquet("labels.parquet")[
        ["proxy_wallet", "snapshot_date", "label"]
    ]
    df = features.merge(labels, on=["proxy_wallet", "snapshot_date"], how="inner")
    print(f"Joined dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Label balance : {df['label'].mean()*100:.1f}% positive")
    return df


def identify_feature_cols(df: pd.DataFrame) -> list[str]:
    excluded = NON_FEATURE_COLS | CATEGORICAL_COLS
    feature_cols = []
    skipped_text = []

    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            feature_cols.append(col)          # cast to int below
        elif pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
        else:
            skipped_text.append(col)

    if skipped_text:
        print(f"Skipped {len(skipped_text)} non-numeric columns: {skipped_text}")

    return feature_cols


def compute_correlations(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    label = df["label"].values
    rows  = []

    for col in feature_cols:
        series = df[col]

        # cast boolean to int so correlation is meaningful
        if pd.api.types.is_bool_dtype(series):
            series = series.astype(int)

        values = series.values

        # nan_policy='omit' drops NaN pairs before computing correlation
        try:
            pearson_r, pearson_p   = pearsonr(
                pd.Series(values).fillna(pd.Series(values).median()),
                label,
            )
        except Exception:
            pearson_r, pearson_p = np.nan, np.nan

        try:
            spearman_r, spearman_p = spearmanr(values, label, nan_policy="omit")
        except Exception:
            spearman_r, spearman_p = np.nan, np.nan

        n_valid = int(pd.notna(pd.Series(values)).sum())

        rows.append({
            "feature":     col,
            "spearman_rho": round(float(spearman_r), 4),
            "spearman_p":   round(float(spearman_p), 4),
            "pearson_r":    round(float(pearson_r),  4),
            "pearson_p":    round(float(pearson_p),  4),
            "n_valid":      n_valid,
        })

    result = (
        pd.DataFrame(rows)
        .sort_values("spearman_rho", ascending=False)
        .reset_index(drop=True)
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-days", type=int, default=7,
                        help="Forward window used when building labels (default: 7). "
                             "Used only to name the output file.")
    args = parser.parse_args()

    df = load_data()

    feature_cols = identify_feature_cols(df)
    print(f"Computing correlations for {len(feature_cols)} feature columns...\n")

    corr = compute_correlations(df, feature_cols)

    # ── print full table ──────────────────────────────────────────────────
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(corr.to_string(index=True))

    # ── save ──────────────────────────────────────────────────────────────
    out_path = f"correlation_table_{args.forward_days}d.csv"
    corr.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(corr)} features)")

    # ── quick summary: top 10 and bottom 5 ───────────────────────────────
    print(f"\nForward window : {args.forward_days}d")
    print("\nTop 10 by Spearman rho:")
    print(corr.head(10)[["feature", "spearman_rho", "spearman_p"]].to_string(index=False))

    print("\nBottom 5 (weakest / negative):")
    print(corr.tail(5)[["feature", "spearman_rho", "spearman_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
