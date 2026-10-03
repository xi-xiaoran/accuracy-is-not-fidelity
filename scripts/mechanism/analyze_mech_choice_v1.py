from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(".").resolve()
BASE = ROOT / "results/mechanism_choice_v1"
OUT = BASE / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

THEORY = ROOT / "results/theory_predictions_v1_fullzoo/fullzoo_theory_summary_by_config.csv"

sample = pd.read_csv(BASE / "fullzoo_choice_sample_summary.csv")
layer = pd.read_csv(BASE / "fullzoo_choice_layer_summary.csv")
cat = pd.read_csv(BASE / "fullzoo_choice_category_summary.csv")
theory = pd.read_csv(THEORY)

report = []
report.append("=== Mechanism Choice v1 Analysis ===")
report.append(f"sample shape: {sample.shape}")
report.append(f"layer shape: {layer.shape}")
report.append(f"category shape: {cat.shape}")
report.append(f"n targets sample: {sample['target_id'].nunique()}")
report.append(f"n targets layer: {layer['target_id'].nunique()}")
report.append("")

# -------------------------
# Sample category overall
# -------------------------
sample["boundary_ratio_num"] = pd.to_numeric(sample["boundary_ratio"], errors="coerce").replace([np.inf, -np.inf], np.nan)

sample_cat = sample.groupby("category").agg(
    n=("dense_uid", "count"),
    mean_choice_rel_l2=("mean_choice_rel_l2", "mean"),
    max_choice_rel_l2=("max_choice_rel_l2", "mean"),
    std_choice_rel_l2=("std_choice_rel_l2", "mean"),
    mean_choice_cos=("mean_choice_cos", "mean"),
    max_choice_cos=("max_choice_cos", "mean"),
    std_choice_cos=("std_choice_cos", "mean"),
    dense_pred_rel_l2=("dense_pred_rel_l2", "mean"),
    compressed_pred_rel_l2=("compressed_pred_rel_l2", "mean"),
    gold_rel_l2=("gold_rel_l2", "mean"),
    dense_pred_cos=("dense_pred_cos", "mean"),
    compressed_pred_cos=("compressed_pred_cos", "mean"),
    gold_cos=("gold_cos", "mean"),
    winner_delta_rel_l2=("winner_delta_rel_l2", "mean"),
    winner_delta_cos=("winner_delta_cos", "mean"),
    last_mean_choice_rel_l2=("last_mean_choice_rel_l2", "mean"),
    last_std_choice_rel_l2=("last_std_choice_rel_l2", "mean"),
    last_mean_choice_cos=("last_mean_choice_cos", "mean"),
    last_std_choice_cos=("last_std_choice_cos", "mean"),
    mean_boundary_ratio=("boundary_ratio_num", "mean"),
).reset_index().sort_values("mean_choice_cos", ascending=False)

sample_cat.to_csv(OUT / "choice_sample_category_overall.csv", index=False)

report.append("=== Sample category overall ===")
report.append(sample_cat.to_string(index=False))
report.append("")

# -------------------------
# Flip vs nonflip direct comparison
# -------------------------
flip_summary = sample.groupby("flip").agg(
    n=("dense_uid", "count"),
    mean_choice_rel_l2=("mean_choice_rel_l2", "mean"),
    max_choice_rel_l2=("max_choice_rel_l2", "mean"),
    std_choice_rel_l2=("std_choice_rel_l2", "mean"),
    mean_choice_cos=("mean_choice_cos", "mean"),
    max_choice_cos=("max_choice_cos", "mean"),
    std_choice_cos=("std_choice_cos", "mean"),
    dense_pred_rel_l2=("dense_pred_rel_l2", "mean"),
    compressed_pred_rel_l2=("compressed_pred_rel_l2", "mean"),
    gold_rel_l2=("gold_rel_l2", "mean"),
    dense_pred_cos=("dense_pred_cos", "mean"),
    compressed_pred_cos=("compressed_pred_cos", "mean"),
    gold_cos=("gold_cos", "mean"),
    winner_delta_rel_l2=("winner_delta_rel_l2", "mean"),
    winner_delta_cos=("winner_delta_cos", "mean"),
).reset_index()

flip_summary.to_csv(OUT / "choice_flip_nonflip_overall.csv", index=False)

report.append("=== Flip vs nonflip overall ===")
report.append(flip_summary.to_string(index=False))
report.append("")

# -------------------------
# Config-level summary joined with theory
# -------------------------
config = sample.groupby(["target_id", "model_key", "method_key", "keep_tag"]).agg(
    n=("dense_uid", "count"),
    flip_rate=("flip", "mean"),
    mean_choice_rel_l2=("mean_choice_rel_l2", "mean"),
    max_choice_rel_l2=("max_choice_rel_l2", "mean"),
    std_choice_rel_l2=("std_choice_rel_l2", "mean"),
    mean_choice_cos=("mean_choice_cos", "mean"),
    max_choice_cos=("max_choice_cos", "mean"),
    std_choice_cos=("std_choice_cos", "mean"),
    dense_pred_rel_l2=("dense_pred_rel_l2", "mean"),
    compressed_pred_rel_l2=("compressed_pred_rel_l2", "mean"),
    gold_rel_l2=("gold_rel_l2", "mean"),
    dense_pred_cos=("dense_pred_cos", "mean"),
    compressed_pred_cos=("compressed_pred_cos", "mean"),
    gold_cos=("gold_cos", "mean"),
    winner_delta_rel_l2=("winner_delta_rel_l2", "mean"),
    winner_delta_cos=("winner_delta_cos", "mean"),
    last_mean_choice_rel_l2=("last_mean_choice_rel_l2", "mean"),
    last_std_choice_rel_l2=("last_std_choice_rel_l2", "mean"),
    last_mean_choice_cos=("last_mean_choice_cos", "mean"),
    last_std_choice_cos=("last_std_choice_cos", "mean"),
).reset_index()

# flip/nonflip config gap
flip_cfg = sample.groupby(["target_id", "flip"]).agg(
    mean_choice_rel_l2=("mean_choice_rel_l2", "mean"),
    mean_choice_cos=("mean_choice_cos", "mean"),
    std_choice_rel_l2=("std_choice_rel_l2", "mean"),
    std_choice_cos=("std_choice_cos", "mean"),
    winner_delta_rel_l2=("winner_delta_rel_l2", "mean"),
    winner_delta_cos=("winner_delta_cos", "mean"),
).reset_index()

piv = flip_cfg.pivot(index="target_id", columns="flip")
piv.columns = [f"{a}_{int(b)}" for a, b in piv.columns]
piv = piv.reset_index()

for metric in ["mean_choice_rel_l2", "mean_choice_cos", "std_choice_rel_l2", "std_choice_cos", "winner_delta_rel_l2", "winner_delta_cos"]:
    if f"{metric}_1" in piv.columns and f"{metric}_0" in piv.columns:
        piv[f"{metric}_flip_minus_nonflip"] = piv[f"{metric}_1"] - piv[f"{metric}_0"]

config = config.merge(piv, on="target_id", how="left")
config = config.merge(
    theory[["target_id", "acc_delta_pct", "pred_flip_pct", "hidden_churn_pct", "prob_js_pct"]],
    on="target_id",
    how="left"
)
config.to_csv(OUT / "choice_config_join_theory.csv", index=False)

corr_metrics = [
    "mean_choice_rel_l2", "max_choice_rel_l2", "std_choice_rel_l2",
    "mean_choice_cos", "max_choice_cos", "std_choice_cos",
    "dense_pred_rel_l2", "compressed_pred_rel_l2", "gold_rel_l2",
    "dense_pred_cos", "compressed_pred_cos", "gold_cos",
    "winner_delta_rel_l2", "winner_delta_cos",
    "last_mean_choice_rel_l2", "last_std_choice_rel_l2",
    "last_mean_choice_cos", "last_std_choice_cos",
    "mean_choice_rel_l2_flip_minus_nonflip",
    "mean_choice_cos_flip_minus_nonflip",
    "std_choice_rel_l2_flip_minus_nonflip",
    "std_choice_cos_flip_minus_nonflip",
    "winner_delta_rel_l2_flip_minus_nonflip",
    "winner_delta_cos_flip_minus_nonflip",
]
theory_metrics = ["acc_delta_pct", "pred_flip_pct", "hidden_churn_pct", "prob_js_pct"]

corr_rows = []
for a in corr_metrics:
    if a not in config.columns:
        continue
    for b in theory_metrics:
        x = pd.to_numeric(config[a], errors="coerce")
        y = pd.to_numeric(config[b], errors="coerce")
        m = x.notna() & y.notna()
        if m.sum() >= 10:
            corr_rows.append({
                "choice_metric": a,
                "theory_metric": b,
                "n": int(m.sum()),
                "pearson": float(x[m].corr(y[m], method="pearson")),
                "spearman": float(x[m].corr(y[m], method="spearman")),
            })
corr = pd.DataFrame(corr_rows)
corr.to_csv(OUT / "choice_config_theory_correlations.csv", index=False)

report.append("=== Top correlations with pred_flip_pct ===")
tmp = corr[corr["theory_metric"] == "pred_flip_pct"].copy()
tmp["abs_spearman"] = tmp["spearman"].abs()
report.append(tmp.sort_values("abs_spearman", ascending=False).head(20).drop(columns=["abs_spearman"]).to_string(index=False))
report.append("")

report.append("=== Top correlations with hidden_churn_pct ===")
tmp = corr[corr["theory_metric"] == "hidden_churn_pct"].copy()
tmp["abs_spearman"] = tmp["spearman"].abs()
report.append(tmp.sort_values("abs_spearman", ascending=False).head(20).drop(columns=["abs_spearman"]).to_string(index=False))
report.append("")

# -------------------------
# Layerwise normalized depth
# -------------------------
max_layer = layer.groupby("target_id")["layer"].max().rename("max_layer").reset_index()
layer = layer.merge(max_layer, on="target_id", how="left")
layer["depth_frac"] = layer["layer"] / layer["max_layer"].replace(0, np.nan)
layer["depth_bin"] = pd.cut(
    layer["depth_frac"],
    bins=np.linspace(0, 1, 11),
    include_lowest=True,
    labels=[f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
)

depth = layer.groupby(["group", "depth_bin"], observed=True).agg(
    n_rows=("target_id", "size"),
    n_total=("n", "sum"),
    mean_choice_rel_l2=("mean_choice_rel_l2", "mean"),
    std_choice_rel_l2=("std_choice_rel_l2", "mean"),
    mean_choice_cos=("mean_choice_cos", "mean"),
    std_choice_cos=("std_choice_cos", "mean"),
    dense_pred_rel_l2=("dense_pred_rel_l2", "mean"),
    compressed_pred_rel_l2=("compressed_pred_rel_l2", "mean"),
    gold_rel_l2=("gold_rel_l2", "mean"),
    dense_pred_cos=("dense_pred_cos", "mean"),
    compressed_pred_cos=("compressed_pred_cos", "mean"),
    gold_cos=("gold_cos", "mean"),
).reset_index()

depth.to_csv(OUT / "choice_layer_by_depth_bin.csv", index=False)

report.append("=== Choice layer drift by depth bin: mean_choice_cos pivot ===")
piv_depth = depth.pivot(index="depth_bin", columns="group", values="mean_choice_cos")
cols = [c for c in ["all", "nonflip", "flip", "damage", "repair", "wrong_to_wrong", "stable_correct", "stable_wrong", "bar_lt1", "bar_ge1"] if c in piv_depth.columns]
report.append(piv_depth[cols].to_string())
report.append("")

report.append("=== Choice layer drift by depth bin: std_choice_cos pivot ===")
piv_depth_std = depth.pivot(index="depth_bin", columns="group", values="std_choice_cos")
report.append(piv_depth_std[cols].to_string())
report.append("")

# -------------------------
# Compare prompt activation v2 if available
# -------------------------
ACTA = ROOT / "results/mechanism_activation_v2/analysis/config_activation_join_theory.csv"
if ACTA.exists():
    prompt_cfg = pd.read_csv(ACTA)
    merged = config.merge(
        prompt_cfg[["target_id", "mean_rel_l2", "mean_cos_dist", "sample_rel_l2_flip_gap", "sample_cos_flip_gap"]],
        on="target_id",
        how="left",
        suffixes=("_choice", "_prompt")
    )
    compare_rows = []
    for a in ["mean_rel_l2", "mean_cos_dist", "sample_rel_l2_flip_gap", "sample_cos_flip_gap",
              "mean_choice_rel_l2", "mean_choice_cos", "std_choice_rel_l2", "std_choice_cos",
              "dense_pred_cos", "compressed_pred_cos", "gold_cos"]:
        if a not in merged.columns:
            continue
        for b in ["pred_flip_pct", "hidden_churn_pct"]:
            x = pd.to_numeric(merged[a], errors="coerce")
            y = pd.to_numeric(merged[b], errors="coerce")
            m = x.notna() & y.notna()
            compare_rows.append({
                "metric": a,
                "target": b,
                "n": int(m.sum()),
                "pearson": float(x[m].corr(y[m], method="pearson")),
                "spearman": float(x[m].corr(y[m], method="spearman")),
            })
    cmp = pd.DataFrame(compare_rows)
    cmp.to_csv(OUT / "prompt_vs_choice_config_correlations.csv", index=False)
    report.append("=== Prompt vs choice config correlations ===")
    report.append(cmp.sort_values(["target", "spearman"], ascending=[True, False]).to_string(index=False))
    report.append("")

# -------------------------
# Phase-2 candidate based on choice metrics
# -------------------------
rank = config.copy()
rank["phase2_choice_score"] = (
    rank["hidden_churn_pct"].fillna(0)
    + 0.5 * rank["pred_flip_pct"].fillna(0)
    + 50.0 * rank["std_choice_cos"].fillna(0)
    + 20.0 * rank["mean_choice_cos"].fillna(0)
)
rank = rank.sort_values("phase2_choice_score", ascending=False)
rank_cols = [
    "target_id", "model_key", "method_key", "keep_tag", "n",
    "acc_delta_pct", "pred_flip_pct", "hidden_churn_pct",
    "mean_choice_rel_l2", "std_choice_rel_l2", "mean_choice_cos", "std_choice_cos",
    "dense_pred_cos", "compressed_pred_cos", "gold_cos",
    "winner_delta_cos", "phase2_choice_score"
]
rank[[c for c in rank_cols if c in rank.columns]].head(50).to_csv(OUT / "choice_phase2_candidate_top50.csv", index=False)

report.append("=== Choice phase-2 candidate top 20 ===")
report.append(rank[[c for c in rank_cols if c in rank.columns]].head(20).to_string(index=False))
report.append("")

# -------------------------
# Figures
# -------------------------
plt.figure(figsize=(6.4, 4.2))
for group in ["nonflip", "flip", "damage", "repair", "wrong_to_wrong"]:
    sub = depth[depth["group"] == group]
    if len(sub):
        x = np.arange(len(sub))
        plt.plot(x, sub["mean_choice_cos"], marker="o", label=group)
plt.xticks(np.arange(10), [str(x) for x in depth["depth_bin"].drop_duplicates()], rotation=35)
plt.xlabel("Normalized depth bin")
plt.ylabel("Choice-conditioned cosine drift")
plt.title("Answer-path drift by churn category")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "fig_choice_cos_by_depth_category.pdf")
plt.savefig(OUT / "fig_choice_cos_by_depth_category.png", dpi=220)
plt.close()

plt.figure(figsize=(6.4, 4.2))
for group in ["nonflip", "flip", "damage", "repair", "wrong_to_wrong"]:
    sub = depth[depth["group"] == group]
    if len(sub):
        x = np.arange(len(sub))
        plt.plot(x, sub["std_choice_cos"], marker="o", label=group)
plt.xticks(np.arange(10), [str(x) for x in depth["depth_bin"].drop_duplicates()], rotation=35)
plt.xlabel("Normalized depth bin")
plt.ylabel("Across-choice cosine drift std")
plt.title("Choice-differential drift by churn category")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "fig_choice_std_cos_by_depth_category.pdf")
plt.savefig(OUT / "fig_choice_std_cos_by_depth_category.png", dpi=220)
plt.close()

plt.figure(figsize=(6.2, 4.2))
plt.scatter(config["hidden_churn_pct"], config["std_choice_cos"], s=18, alpha=0.65)
plt.xlabel("Hidden churn (%)")
plt.ylabel("Across-choice cosine drift std")
plt.title("Choice-differential drift vs hidden churn")
plt.tight_layout()
plt.savefig(OUT / "fig_choice_std_cos_vs_hidden_churn.pdf")
plt.savefig(OUT / "fig_choice_std_cos_vs_hidden_churn.png", dpi=220)
plt.close()

plt.figure(figsize=(6.2, 4.2))
plt.scatter(config["pred_flip_pct"], config["mean_choice_cos"], s=18, alpha=0.65)
plt.xlabel("Prediction flip (%)")
plt.ylabel("Mean choice-conditioned cosine drift")
plt.title("Answer-path drift vs prediction flip")
plt.tight_layout()
plt.savefig(OUT / "fig_choice_mean_cos_vs_pred_flip.pdf")
plt.savefig(OUT / "fig_choice_mean_cos_vs_pred_flip.png", dpi=220)
plt.close()

# Write report
report_path = OUT / "CHOICE_MECHANISM_REPORT.txt"
report_path.write_text("\n".join(report))
print("WROTE", report_path)
print("\n".join(report[:80]))
print("\nOUTPUT_DIR", OUT)
