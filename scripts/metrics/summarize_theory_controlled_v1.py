from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
FULL = ROOT / "results/theory_predictions_v1_fullzoo"
OUT = FULL / "controlled_summaries"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FULL / "fullzoo_theory_summary_by_config.csv")

def find_keep_col(df):
    for c in ["keep_ratio", "keep", "rank"]:
        if c in df.columns:
            return c
    return None

keep_col = find_keep_col(df)

def subset_mask(name):
    if name == "all":
        return np.ones(len(df), dtype=bool)
    if name == "keep0p8":
        if keep_col is None:
            return np.zeros(len(df), dtype=bool)
        x = df[keep_col]
        if x.dtype == object:
            return x.astype(str).str.contains("0p8|0.8", regex=True, na=False).values
        return np.isclose(pd.to_numeric(x, errors="coerce"), 0.8, atol=1e-6)
    if name.startswith("abs_acc_le_"):
        th = float(name.replace("abs_acc_le_", ""))
        return df["acc_delta_pct"].abs().le(th).values
    if name.startswith("compressed_acc_ge_"):
        th = float(name.replace("compressed_acc_ge_", ""))
        return df["compressed_acc_pct"].ge(th).values
    raise ValueError(name)

metrics = [
    "dense_acc_pct",
    "compressed_acc_pct",
    "acc_delta_pct",
    "pred_flip_pct",
    "hidden_churn_pct",
    "damage_pct",
    "repair_pct",
    "wrong_to_wrong_pct",
    "bar_lt1_flip_pct",
    "bar_ge1_flip_pct",
    "bar_1_2_flip_pct",
    "bar_gt2_flip_pct",
    "proj_task_bias_prob_js_delta_pct",
    "proj_task_bias_hidden_churn_delta_pct",
    "energy_task_bias_explained_pct",
    "energy_global_by_nchoices_explained_pct",
]
metrics = [c for c in metrics if c in df.columns]

subsets = [
    "all",
    "keep0p8",
    "abs_acc_le_10",
    "abs_acc_le_5",
    "abs_acc_le_2",
    "compressed_acc_ge_50",
    "compressed_acc_ge_60",
]

rows = []
for s in subsets:
    m = subset_mask(s)
    g = df[m].copy()
    row = {"subset": s, "n_configs": len(g)}
    for c in metrics:
        row[c + "_mean"] = g[c].mean()
        row[c + "_median"] = g[c].median()
    rows.append(row)

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "controlled_theory_summary.csv", index=False)

show_cols = [
    "subset", "n_configs",
    "acc_delta_pct_mean",
    "pred_flip_pct_mean",
    "hidden_churn_pct_mean",
    "bar_lt1_flip_pct_mean",
    "bar_ge1_flip_pct_mean",
    "bar_1_2_flip_pct_mean",
    "bar_gt2_flip_pct_mean",
    "proj_task_bias_prob_js_delta_pct_mean",
    "proj_task_bias_hidden_churn_delta_pct_mean",
    "energy_task_bias_explained_pct_mean",
]
show_cols = [c for c in show_cols if c in summary.columns]

print("===== CONTROLLED THEORY SUMMARY =====")
print(summary[show_cols].to_string(index=False))
print("WROTE", OUT / "controlled_theory_summary.csv")

# Also produce method/keep table if available.
group_cols = []
for c in ["method_key", "method", "compression_method"]:
    if c in df.columns:
        group_cols.append(c)
        break
if keep_col is not None:
    group_cols.append(keep_col)

if group_cols:
    num_cols = [c for c in metrics if c not in group_cols]
    by = df.groupby(group_cols, dropna=False)[num_cols].mean().reset_index()
    by.to_csv(OUT / "controlled_by_method_keep.csv", index=False)
    print("WROTE", OUT / "controlled_by_method_keep.csv")

# Worst and best cases for discussion.
show = [c for c in ["target_id", "model_key", "method_key", keep_col, "acc_delta_pct", "pred_flip_pct", "hidden_churn_pct", "bar_ge1_flip_pct"] if c and c in df.columns]

df.sort_values("hidden_churn_pct", ascending=False)[show].head(30).to_csv(OUT / "worst_hidden_churn_top30.csv", index=False)
df.sort_values("hidden_churn_pct", ascending=True)[show].head(30).to_csv(OUT / "best_hidden_churn_top30.csv", index=False)

print("WROTE worst/best hidden churn tables")
