import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PARSER = argparse.ArgumentParser(
    description="Generate aggregate analysis tables and Figures 1--6."
)
PARSER.add_argument(
    "--project-root", type=Path, default=Path("."),
    help="Workspace containing results/summary_tables (default: current directory).",
)
ARGS = PARSER.parse_args()
ROOT = ARGS.project_root.expanduser().resolve()
TABLE_DIR = ROOT / "results" / "summary_tables"
OUT_TABLE = ROOT / "results" / "analysis_tables"
OUT_FIG = ROOT / "results" / "figures"

OUT_TABLE.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

CFG = TABLE_DIR / "fullzoo_lowrank_pair_metrics_by_config_normchar.csv"
TASK = TABLE_DIR / "fullzoo_lowrank_pair_metrics_by_task_normchar.csv"
METHOD_KEEP = TABLE_DIR / "fullzoo_lowrank_pair_metrics_by_method_keep_normchar.csv"

cfg = pd.read_csv(CFG)
task = pd.read_csv(TASK)
method_keep = pd.read_csv(METHOD_KEEP)

# Convert to percentage points for plotting/tables.
for df in [cfg, task, method_keep]:
    for c in ["acc_norm_delta", "pred_flip", "prob_js", "gold_prob_shift", "nll_shift"]:
        if c in df.columns:
            df[c + "_pct"] = df[c] * 100.0

# -----------------------------
# Analysis tables
# -----------------------------

summary_rows = []

def add_summary(name, df):
    summary_rows.append({
        "subset": name,
        "n_rows": len(df),
        "acc_norm_delta_mean_pp": df["acc_norm_delta_pct"].mean(),
        "acc_norm_delta_median_pp": df["acc_norm_delta_pct"].median(),
        "pred_flip_mean_pct": df["pred_flip_pct"].mean(),
        "prob_js_mean_x100": df["prob_js_pct"].mean(),
        "gold_prob_shift_mean_pp": df["gold_prob_shift_pct"].mean(),
        "nll_shift_mean": df["nll_shift"].mean(),
    })

add_summary("all_config", cfg)
add_summary("keep0p8_config", cfg[cfg["keep_ratio"] == 0.8])
add_summary("keep0p8_acc_drop_le_10pp_config", cfg[(cfg["keep_ratio"] == 0.8) & (cfg["acc_norm_delta"] >= -0.10)])
add_summary("acc_abs_le_5pp_config", cfg[cfg["acc_norm_delta"].abs() <= 0.05])
add_summary("acc_abs_le_10pp_config", cfg[cfg["acc_norm_delta"].abs() <= 0.10])

add_summary("all_task", task)
add_summary("keep0p8_task", task[task["keep_ratio"] == 0.8])
add_summary("keep0p8_acc_drop_le_10pp_task", task[(task["keep_ratio"] == 0.8) & (task["acc_norm_delta"] >= -0.10)])
add_summary("acc_abs_le_1pp_task", task[task["acc_norm_delta"].abs() <= 0.01])
add_summary("acc_abs_le_3pp_task", task[task["acc_norm_delta"].abs() <= 0.03])
add_summary("acc_abs_le_5pp_task", task[task["acc_norm_delta"].abs() <= 0.05])
add_summary("acc_abs_le_10pp_task", task[task["acc_norm_delta"].abs() <= 0.10])

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_TABLE / "fullzoo_key_subsets_summary.csv", index=False)

severity_cfg = cfg.groupby("keep_ratio", as_index=False).agg(
    n_configs=("target_id", "count"),
    acc_norm_delta_mean=("acc_norm_delta", "mean"),
    pred_flip_mean=("pred_flip", "mean"),
    prob_js_mean=("prob_js", "mean"),
    gold_prob_shift_mean=("gold_prob_shift", "mean"),
    nll_shift_mean=("nll_shift", "mean"),
)
for c in ["acc_norm_delta_mean", "pred_flip_mean", "prob_js_mean", "gold_prob_shift_mean"]:
    severity_cfg[c + "_pct"] = severity_cfg[c] * 100
severity_cfg.to_csv(OUT_TABLE / "severity_by_keep_ratio_config.csv", index=False)

method_summary = cfg.groupby("method_key", as_index=False).agg(
    n_configs=("target_id", "count"),
    acc_norm_delta_mean=("acc_norm_delta", "mean"),
    pred_flip_mean=("pred_flip", "mean"),
    prob_js_mean=("prob_js", "mean"),
    gold_prob_shift_mean=("gold_prob_shift", "mean"),
    nll_shift_mean=("nll_shift", "mean"),
).sort_values("pred_flip_mean")
for c in ["acc_norm_delta_mean", "pred_flip_mean", "prob_js_mean", "gold_prob_shift_mean"]:
    method_summary[c + "_pct"] = method_summary[c] * 100
method_summary.to_csv(OUT_TABLE / "method_ranking_config.csv", index=False)

model_summary = cfg.groupby("model_key", as_index=False).agg(
    n_configs=("target_id", "count"),
    acc_norm_delta_mean=("acc_norm_delta", "mean"),
    pred_flip_mean=("pred_flip", "mean"),
    prob_js_mean=("prob_js", "mean"),
    gold_prob_shift_mean=("gold_prob_shift", "mean"),
    nll_shift_mean=("nll_shift", "mean"),
).sort_values("pred_flip_mean")
for c in ["acc_norm_delta_mean", "pred_flip_mean", "prob_js_mean", "gold_prob_shift_mean"]:
    model_summary[c + "_pct"] = model_summary[c] * 100
model_summary.to_csv(OUT_TABLE / "model_ranking_config.csv", index=False)

task_summary = task.groupby("task", as_index=False).agg(
    n_rows=("target_id", "count"),
    acc_norm_delta_mean=("acc_norm_delta", "mean"),
    pred_flip_mean=("pred_flip", "mean"),
    prob_js_mean=("prob_js", "mean"),
    gold_prob_shift_mean=("gold_prob_shift", "mean"),
    nll_shift_mean=("nll_shift", "mean"),
).sort_values("pred_flip_mean")
for c in ["acc_norm_delta_mean", "pred_flip_mean", "prob_js_mean", "gold_prob_shift_mean"]:
    task_summary[c + "_pct"] = task_summary[c] * 100
task_summary.to_csv(OUT_TABLE / "task_ranking_by_task_rows.csv", index=False)

# Accuracy-controlled buckets at task level.
bins = [-np.inf, -0.30, -0.20, -0.10, -0.05, -0.01, 0.01, 0.05, 0.10, np.inf]
labels = ["drop >30pp", "drop 20-30pp", "drop 10-20pp", "drop 5-10pp", "drop 1-5pp", "within ±1pp", "gain 1-5pp", "gain 5-10pp", "gain >10pp"]
task["acc_bucket"] = pd.cut(task["acc_norm_delta"], bins=bins, labels=labels)

bucket = task.groupby("acc_bucket", observed=False).agg(
    n_rows=("target_id", "count"),
    acc_norm_delta_mean=("acc_norm_delta", "mean"),
    pred_flip_mean=("pred_flip", "mean"),
    prob_js_mean=("prob_js", "mean"),
    gold_prob_shift_mean=("gold_prob_shift", "mean"),
    nll_shift_mean=("nll_shift", "mean"),
).reset_index()
for c in ["acc_norm_delta_mean", "pred_flip_mean", "prob_js_mean", "gold_prob_shift_mean"]:
    bucket[c + "_pct"] = bucket[c] * 100
bucket.to_csv(OUT_TABLE / "accuracy_bucket_task_summary.csv", index=False)

# Accuracy-stable high-drift examples.
stable_examples = (
    task[task["acc_norm_delta"].abs() <= 0.05]
    .sort_values(["pred_flip", "prob_js"], ascending=False)
    .loc[:, [
        "target_id", "model_key", "method_key", "keep_tag", "keep_ratio", "task",
        "acc_norm_delta", "pred_flip", "prob_js", "gold_prob_shift", "nll_shift",
        "dense_acc_norm", "compressed_acc_norm"
    ]]
)
stable_examples.to_csv(OUT_TABLE / "accuracy_stable_high_drift_examples_task.csv", index=False)

# Literature-aligned subset.
lit_methods = {
    "svdllm_v1/alpaca_recover",
    "modegpt",
    "svdllm_v1/update",
    "asvd/gqa_fix",
    "basis_sharing",
}
lit = cfg[(cfg["keep_ratio"] == 0.8) & (cfg["method_key"].isin(lit_methods))].copy()
lit.to_csv(OUT_TABLE / "keep0p8_literature_aligned_subset_config.csv", index=False)

# -----------------------------
# Figures
# -----------------------------

def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT_FIG / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(OUT_FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()

# Fig 1: severity curve.
x = severity_cfg["keep_ratio"].values
plt.figure(figsize=(6.8, 4.6))
plt.plot(x, severity_cfg["pred_flip_mean_pct"], marker="o", label="Prediction flip")
plt.plot(x, -severity_cfg["acc_norm_delta_mean_pct"], marker="o", label="Accuracy drop")
plt.plot(x, severity_cfg["prob_js_mean_pct"], marker="o", label="Prob-JS ×100")
plt.xlabel("Keep ratio")
plt.ylabel("Percent / scaled value")
plt.title("Low-rank compression severity increases behavior drift")
plt.legend()
plt.grid(True, alpha=0.3)
savefig("fig1_severity_curve")

# Fig 2: method ranking by pred flip.
plot_methods = method_summary[method_summary["n_configs"] >= 5].copy()
plot_methods = plot_methods.sort_values("pred_flip_mean_pct", ascending=True)
plt.figure(figsize=(8.2, 5.8))
plt.barh(plot_methods["method_key"], plot_methods["pred_flip_mean_pct"])
plt.xlabel("Mean prediction flip (%)")
plt.ylabel("Method")
plt.title("Behavior drift differs substantially across low-rank methods")
plt.grid(axis="x", alpha=0.3)
savefig("fig2_method_ranking_pred_flip")

# Fig 3: acc delta vs pred flip scatter.
plt.figure(figsize=(6.2, 5.0))
plt.scatter(cfg["acc_norm_delta_pct"], cfg["pred_flip_pct"], alpha=0.65, s=24)
plt.axvline(0, linewidth=1)
plt.axvline(-5, linestyle="--", linewidth=1)
plt.axvline(-10, linestyle="--", linewidth=1)
plt.xlabel("Accuracy-normalized delta (pp)")
plt.ylabel("Prediction flip (%)")
plt.title("Accuracy drop and behavior drift are related but not identical")
plt.grid(True, alpha=0.3)
savefig("fig3_acc_delta_vs_pred_flip_scatter")

# Fig 4: accuracy bucket.
bucket_plot = bucket[bucket["n_rows"] > 0].copy()
plt.figure(figsize=(8.0, 4.8))
plt.bar(bucket_plot["acc_bucket"].astype(str), bucket_plot["pred_flip_mean_pct"])
plt.xticks(rotation=35, ha="right")
plt.xlabel("Task-level accuracy-change bucket")
plt.ylabel("Mean prediction flip (%)")
plt.title("Non-trivial behavior drift remains in accuracy-stable task rows")
plt.grid(axis="y", alpha=0.3)
savefig("fig4_accuracy_controlled_buckets")

# Fig 5: model × task heatmap for pred flip.
pivot = task.pivot_table(index="model_key", columns="task", values="pred_flip", aggfunc="mean") * 100
plt.figure(figsize=(9.0, 4.8))
im = plt.imshow(pivot.values, aspect="auto")
plt.colorbar(im, label="Prediction flip (%)")
plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
plt.yticks(range(len(pivot.index)), pivot.index)
plt.title("Model-task heterogeneity in behavior drift")
savefig("fig5_model_task_pred_flip_heatmap")

# Fig 6: keep0p8 method ranking.
keep08 = cfg[cfg["keep_ratio"] == 0.8].groupby("method_key", as_index=False).agg(
    n_configs=("target_id", "count"),
    acc_norm_delta_mean=("acc_norm_delta", "mean"),
    pred_flip_mean=("pred_flip", "mean"),
    prob_js_mean=("prob_js", "mean"),
)
keep08 = keep08[keep08["n_configs"] >= 2].sort_values("pred_flip_mean")
keep08["pred_flip_mean_pct"] = keep08["pred_flip_mean"] * 100
plt.figure(figsize=(8.0, 5.6))
plt.barh(keep08["method_key"], keep08["pred_flip_mean_pct"])
plt.xlabel("Mean prediction flip at keep0.8 (%)")
plt.ylabel("Method")
plt.title("Even mild low-rank compression induces behavior drift")
plt.grid(axis="x", alpha=0.3)
savefig("fig6_keep0p8_method_ranking")

# Correlation table.
corr_cols = ["keep_ratio", "acc_norm_delta", "pred_flip", "prob_js", "gold_prob_shift", "nll_shift"]
corr = cfg[corr_cols].corr()
corr.to_csv(OUT_TABLE / "config_metric_correlations.csv")

print("===== DATA CHECK =====")
print("cfg rows", len(cfg))
print("task rows", len(task))
print("method_keep rows", len(method_keep))

print("\n===== KEY SUBSETS =====")
print(summary.to_string(index=False))

print("\n===== SEVERITY =====")
print(severity_cfg[[
    "keep_ratio", "n_configs", "acc_norm_delta_mean_pct",
    "pred_flip_mean_pct", "prob_js_mean_pct", "gold_prob_shift_mean_pct", "nll_shift_mean"
]].to_string(index=False))

print("\n===== METHOD RANKING >=5 CONFIGS =====")
print(plot_methods[[
    "method_key", "n_configs", "acc_norm_delta_mean_pct",
    "pred_flip_mean_pct", "prob_js_mean_pct", "gold_prob_shift_mean_pct", "nll_shift_mean"
]].to_string(index=False))

print("\n===== ACCURACY BUCKETS =====")
print(bucket[[
    "acc_bucket", "n_rows", "acc_norm_delta_mean_pct",
    "pred_flip_mean_pct", "prob_js_mean_pct", "gold_prob_shift_mean_pct", "nll_shift_mean"
]].to_string(index=False))

print("\n===== WROTE TABLES =====")
for p in sorted(OUT_TABLE.glob("*.csv")):
    print(p)

print("\n===== WROTE FIGURES =====")
for p in sorted(OUT_FIG.glob("fig*.pdf")):
    print(p)
