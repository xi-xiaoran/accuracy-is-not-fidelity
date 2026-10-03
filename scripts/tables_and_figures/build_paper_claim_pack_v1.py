import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARSER = argparse.ArgumentParser(
    description="Generate claim tables and Figures 10--12 from analysis tables."
)
PARSER.add_argument(
    "--project-root", type=Path, default=Path("."),
    help="Workspace containing the expected results tree (default: current directory).",
)
ARGS = PARSER.parse_args()
ROOT = ARGS.project_root.expanduser().resolve()
SUM = ROOT / "results" / "summary_tables"
ANA = ROOT / "results" / "analysis_tables"
FIG = ROOT / "results" / "figures"
OUT = ROOT / "results" / "paper_claim_pack_v1"

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

pair_task = pd.read_csv(SUM / "fullzoo_lowrank_pair_metrics_by_task_normchar.csv")
pair_cfg = pd.read_csv(SUM / "fullzoo_lowrank_pair_metrics_by_config_normchar.csv")
churn_task = pd.read_csv(ANA / "decision_churn_by_task.csv")
churn_cfg = pd.read_csv(ANA / "decision_churn_by_config.csv")
churn_bucket = pd.read_csv(ANA / "decision_churn_accuracy_buckets.csv")
method_rank = pd.read_csv(ANA / "method_ranking_config.csv")
method_churn = pd.read_csv(ANA / "decision_churn_method_summary.csv")
severity = pd.read_csv(ANA / "severity_by_keep_ratio_config.csv")

def pct(x):
    return 100.0 * x

def weighted_mean(df, col, weight="n_samples"):
    x = df[[col, weight]].dropna()
    if len(x) == 0:
        return np.nan
    return np.average(x[col], weights=x[weight])

# -------------------------
# Sanity checks
# -------------------------
m = pair_task.merge(
    churn_task,
    on=["target_id", "model_key", "method_key", "keep_tag", "keep_ratio", "task"],
    how="inner",
    suffixes=("_pair", "_churn")
)

sanity = {
    "n_pair_task_rows": len(pair_task),
    "n_churn_task_rows": len(churn_task),
    "n_joined_rows": len(m),
    "pred_flip_mean_abs_diff": (m["pred_flip"] - m["pred_flip_score"]).abs().mean(),
    "pred_flip_max_abs_diff": (m["pred_flip"] - m["pred_flip_score"]).abs().max(),
    "acc_delta_corr_official_vs_score": m[["acc_norm_delta", "acc_delta_score"]].corr().iloc[0,1],
    "acc_delta_mean_abs_diff_official_vs_score": (m["acc_norm_delta"] - m["acc_delta_score"]).abs().mean(),
}

pd.DataFrame([sanity]).to_csv(OUT / "sanity_checks.csv", index=False)

# -------------------------
# Core claim numbers
# -------------------------
def subset_row(name, df, level):
    if level == "config":
        return {
            "subset": name,
            "level": level,
            "n_rows": len(df),
            "acc_norm_delta_pp": pct(df["acc_norm_delta"].mean()),
            "pred_flip_pct": pct(df["pred_flip"].mean()),
            "prob_js_x100": pct(df["prob_js"].mean()),
            "gold_prob_shift_pp": pct(df["gold_prob_shift"].mean()),
            "nll_shift": df["nll_shift"].mean(),
        }
    else:
        return {
            "subset": name,
            "level": level,
            "n_rows": len(df),
            "acc_norm_delta_pp": pct(df["acc_norm_delta"].mean()),
            "pred_flip_pct": pct(df["pred_flip"].mean()),
            "prob_js_x100": pct(df["prob_js"].mean()),
            "gold_prob_shift_pp": pct(df["gold_prob_shift"].mean()),
            "nll_shift": df["nll_shift"].mean(),
        }

claim_rows = []
claim_rows.append(subset_row("Full official low-rank zoo", pair_cfg, "config"))
claim_rows.append(subset_row("Mild compression only: keep0.8", pair_cfg[pair_cfg["keep_ratio"] == 0.8], "config"))
claim_rows.append(subset_row("Task rows with |acc_delta| <= 5pp", pair_task[pair_task["acc_norm_delta"].abs() <= 0.05], "task"))
claim_rows.append(subset_row("Task rows with |acc_delta| <= 1pp", pair_task[pair_task["acc_norm_delta"].abs() <= 0.01], "task"))

claim = pd.DataFrame(claim_rows)

# Add churn-specific rows.
churn_claims = []
for name, df, level in [
    ("Full official low-rank zoo", churn_cfg, "config_weighted"),
    ("Mild compression only: keep0.8", churn_cfg[churn_cfg["keep_ratio"] == 0.8], "config_weighted"),
    ("Task rows with |acc_delta| <= 5pp", churn_task[churn_task["acc_delta_score"].abs() <= 0.05], "task_weighted"),
    ("Task rows with |acc_delta| <= 1pp", churn_task[churn_task["acc_delta_score"].abs() <= 0.01], "task_weighted"),
]:
    churn_claims.append({
        "subset": name,
        "level": level,
        "n_rows": len(df),
        "acc_delta_score_pp": pct(weighted_mean(df, "acc_delta_score")),
        "pred_flip_score_pct": pct(weighted_mean(df, "pred_flip_score")),
        "hidden_churn_pct": pct(weighted_mean(df, "hidden_churn")),
        "damage_rate_pct": pct(weighted_mean(df, "damage_rate")),
        "repair_rate_pct": pct(weighted_mean(df, "repair_rate")),
        "wrong_to_wrong_flip_pct": pct(weighted_mean(df, "both_wrong_flip_rate")),
    })

churn_claim = pd.DataFrame(churn_claims)

claim.to_csv(OUT / "core_uncertainty_claims.csv", index=False)
churn_claim.to_csv(OUT / "core_decision_churn_claims.csv", index=False)

# -------------------------
# Paper tables
# -------------------------

# Table 1: severity + hidden churn.
churn_keep = []
for k, g in churn_cfg.groupby("keep_ratio"):
    churn_keep.append({
        "keep_ratio": k,
        "n_configs_churn": len(g),
        "acc_delta_score_pp": pct(weighted_mean(g, "acc_delta_score")),
        "pred_flip_score_pct": pct(weighted_mean(g, "pred_flip_score")),
        "hidden_churn_pct": pct(weighted_mean(g, "hidden_churn")),
        "damage_rate_pct": pct(weighted_mean(g, "damage_rate")),
        "repair_rate_pct": pct(weighted_mean(g, "repair_rate")),
    })
churn_keep = pd.DataFrame(churn_keep)

severity_table = severity.merge(churn_keep, on="keep_ratio", how="left")
severity_table = severity_table[[
    "keep_ratio", "n_configs",
    "acc_norm_delta_mean_pct", "pred_flip_mean_pct", "prob_js_mean_pct",
    "gold_prob_shift_mean_pct", "hidden_churn_pct", "damage_rate_pct", "repair_rate_pct"
]]
severity_table.to_csv(OUT / "paper_table_severity_plus_churn.csv", index=False)

# Table 2: accuracy bucket churn.
bucket_cols = [
    "acc_bucket", "n_rows", "n_samples",
    "acc_delta_score_pct", "pred_flip_score_pct", "hidden_churn_pct",
    "damage_rate_pct", "repair_rate_pct", "both_wrong_flip_rate_pct"
]
bucket_table = churn_bucket[bucket_cols].copy()
bucket_table.to_csv(OUT / "paper_table_accuracy_bucket_churn.csv", index=False)

# Table 3: method summary.
method_table = method_rank.merge(
    method_churn[[
        "method_key", "hidden_churn_pct", "damage_rate_pct", "repair_rate_pct", "both_wrong_flip_rate_pct"
    ]],
    on="method_key",
    how="left",
)
method_table = method_table[method_table["n_configs"] >= 5].copy()
method_table = method_table[[
    "method_key", "n_configs",
    "acc_norm_delta_mean_pct", "pred_flip_mean_pct", "prob_js_mean_pct",
    "gold_prob_shift_mean_pct", "hidden_churn_pct",
    "damage_rate_pct", "repair_rate_pct", "both_wrong_flip_rate_pct"
]].sort_values("hidden_churn_pct")
method_table.to_csv(OUT / "paper_table_method_plus_churn.csv", index=False)

# Accuracy-stable examples.
stable_examples = churn_task[churn_task["acc_delta_score"].abs() <= 0.01].copy()
stable_examples = stable_examples.sort_values("hidden_churn", ascending=False)
stable_examples.to_csv(OUT / "paper_examples_within_1pp_high_churn.csv", index=False)

stable5_examples = churn_task[churn_task["acc_delta_score"].abs() <= 0.05].copy()
stable5_examples = stable5_examples.sort_values("hidden_churn", ascending=False)
stable5_examples.to_csv(OUT / "paper_examples_within_5pp_high_churn.csv", index=False)

# -------------------------
# LaTeX table exports
# -------------------------
def to_latex(df, path, float_format="%.2f"):
    with open(path, "w") as f:
        f.write(df.to_latex(index=False, float_format=lambda x: float_format % x if isinstance(x, float) else str(x)))

to_latex(severity_table, OUT / "paper_table_severity_plus_churn.tex")
to_latex(bucket_table, OUT / "paper_table_accuracy_bucket_churn.tex")
to_latex(method_table, OUT / "paper_table_method_plus_churn.tex")

# -------------------------
# Combined paper figures
# -------------------------

# Fig 10: main story: severity + hidden churn buckets.
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

ax = axes[0]
x = severity_table["keep_ratio"].values
ax.plot(x, -severity_table["acc_norm_delta_mean_pct"], marker="o", label="Accuracy drop")
ax.plot(x, severity_table["pred_flip_mean_pct"], marker="o", label="Prediction flip")
ax.plot(x, severity_table["hidden_churn_pct"], marker="o", label="Hidden churn")
ax.set_xlabel("Keep ratio")
ax.set_ylabel("Percent")
ax.set_title("Compression severity increases drift")
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1]
bp = bucket_table[bucket_table["n_rows"] > 0].copy()
ax.bar(bp["acc_bucket"], bp["hidden_churn_pct"])
ax.set_xticklabels(bp["acc_bucket"], rotation=35, ha="right")
ax.set_ylabel("Hidden churn (%)")
ax.set_xlabel("Accuracy-change bucket")
ax.set_title("Accuracy-stable rows still churn")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(FIG / "fig10_main_story_severity_and_hidden_churn.pdf", bbox_inches="tight")
plt.savefig(FIG / "fig10_main_story_severity_and_hidden_churn.png", dpi=300, bbox_inches="tight")
plt.close()

# Fig 11: churn decomposition for buckets.
fig, ax = plt.subplots(figsize=(9.2, 4.8))
bp = bucket_table[bucket_table["n_rows"] > 0].copy()
x = np.arange(len(bp))
damage = bp["damage_rate_pct"].values
repair = bp["repair_rate_pct"].values
wrongflip = bp["both_wrong_flip_rate_pct"].values

ax.bar(x, damage, label="Correct→wrong")
ax.bar(x, repair, bottom=damage, label="Wrong→correct")
ax.bar(x, wrongflip, bottom=damage + repair, label="Wrong→different wrong")
ax.set_xticks(x)
ax.set_xticklabels(bp["acc_bucket"], rotation=35, ha="right")
ax.set_ylabel("Rate (%)")
ax.set_xlabel("Accuracy-change bucket")
ax.set_title("Net accuracy hides offsetting decision changes")
ax.grid(axis="y", alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(FIG / "fig11_churn_decomposition_accuracy_buckets.pdf", bbox_inches="tight")
plt.savefig(FIG / "fig11_churn_decomposition_accuracy_buckets.png", dpi=300, bbox_inches="tight")
plt.close()

# Fig 12: method hidden churn.
mp = method_table.sort_values("hidden_churn_pct", ascending=True)
fig, ax = plt.subplots(figsize=(8.8, 5.7))
ax.barh(mp["method_key"], mp["hidden_churn_pct"])
ax.set_xlabel("Hidden churn (%)")
ax.set_ylabel("Method")
ax.set_title("Hidden decision churn varies by low-rank method")
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(FIG / "fig12_method_hidden_churn_paper.pdf", bbox_inches="tight")
plt.savefig(FIG / "fig12_method_hidden_churn_paper.png", dpi=300, bbox_inches="tight")
plt.close()

# -------------------------
# Markdown report
# -------------------------
report = []

report.append("# Paper Claim Pack v1\n")
report.append("## Sanity checks\n")
for k, v in sanity.items():
    report.append(f"- {k}: {v}")

report.append("\n## Core uncertainty claims\n")
report.append(claim.to_markdown(index=False))

report.append("\n## Core decision churn claims\n")
report.append(churn_claim.to_markdown(index=False))

report.append("\n## Severity + churn\n")
report.append(severity_table.to_markdown(index=False))

report.append("\n## Accuracy bucket churn\n")
report.append(bucket_table.to_markdown(index=False))

report.append("\n## Method + churn\n")
report.append(method_table.to_markdown(index=False))

(OUT / "paper_claim_pack_v1_report.md").write_text("\n".join(report))

print("===== SANITY =====")
for k, v in sanity.items():
    print(k, v)

print("\n===== CORE DECISION CHURN CLAIMS =====")
print(churn_claim.to_string(index=False))

print("\n===== SEVERITY TABLE =====")
print(severity_table.to_string(index=False))

print("\n===== ACCURACY BUCKET TABLE =====")
print(bucket_table.to_string(index=False))

print("\n===== METHOD TABLE =====")
print(method_table.to_string(index=False))

print("\n===== WROTE CLAIM PACK =====")
for p in sorted(OUT.glob("*")):
    print(p)

print("\n===== WROTE PAPER FIGURES =====")
for p in [
    FIG / "fig10_main_story_severity_and_hidden_churn.pdf",
    FIG / "fig11_churn_decomposition_accuracy_buckets.pdf",
    FIG / "fig12_method_hidden_churn_paper.pdf",
]:
    print(p)
