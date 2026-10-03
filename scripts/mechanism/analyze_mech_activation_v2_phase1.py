from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(".").resolve()
BASE = ROOT / "results/mechanism_activation_v2"
OUT = BASE / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

THEORY = ROOT / "results/theory_predictions_v1_fullzoo/fullzoo_theory_summary_by_config.csv"

layer = pd.read_csv(BASE / "fullzoo_layer_summary.csv")
gap = pd.read_csv(BASE / "fullzoo_flip_nonflip_layer_gap.csv")
sample = pd.read_csv(BASE / "fullzoo_sample_summary.csv")
cat = pd.read_csv(BASE / "fullzoo_category_summary.csv")
theory = pd.read_csv(THEORY)

# -------------------------
# Basic integrity
# -------------------------
report = []
report.append("=== Mechanism Activation v2 Phase-1 Analysis ===")
report.append(f"layer shape: {layer.shape}")
report.append(f"gap shape: {gap.shape}")
report.append(f"sample shape: {sample.shape}")
report.append(f"category shape: {cat.shape}")
report.append(f"n targets layer: {layer['target_id'].nunique()}")
report.append(f"n targets sample: {sample['target_id'].nunique()}")
report.append("")

# Add normalized depth for cross-model aggregation.
max_layer = layer.groupby("target_id")["layer"].max().rename("max_layer").reset_index()
layer = layer.merge(max_layer, on="target_id", how="left")
layer["depth_frac"] = layer["layer"] / layer["max_layer"].replace(0, np.nan)

gap = gap.merge(max_layer, on="target_id", how="left")
gap["depth_frac"] = gap["layer"] / gap["max_layer"].replace(0, np.nan)

# -------------------------
# Overall flip-vs-nonflip gap by raw layer and normalized layer bins
# -------------------------
gap["rel_gap"] = gap["rel_l2_flip_minus_nonflip"]
gap["cos_gap"] = gap["cos_dist_flip_minus_nonflip"]

overall_gap = gap.groupby("layer").agg(
    n_configs=("target_id", "nunique"),
    rel_gap_mean=("rel_gap", "mean"),
    rel_gap_median=("rel_gap", "median"),
    rel_gap_std=("rel_gap", "std"),
    cos_gap_mean=("cos_gap", "mean"),
    cos_gap_median=("cos_gap", "median"),
    rel_flip_mean=("rel_l2_mean_flip", "mean"),
    rel_nonflip_mean=("rel_l2_mean_nonflip", "mean"),
    cos_flip_mean=("cos_dist_mean_flip", "mean"),
    cos_nonflip_mean=("cos_dist_mean_nonflip", "mean"),
).reset_index()
overall_gap.to_csv(OUT / "overall_flip_nonflip_gap_by_layer.csv", index=False)

gap["depth_bin"] = pd.cut(
    gap["depth_frac"],
    bins=np.linspace(0, 1, 11),
    include_lowest=True,
    labels=[f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
)
depth_gap = gap.groupby("depth_bin", observed=True).agg(
    n_rows=("target_id", "size"),
    n_configs=("target_id", "nunique"),
    rel_gap_mean=("rel_gap", "mean"),
    rel_gap_median=("rel_gap", "median"),
    cos_gap_mean=("cos_gap", "mean"),
    cos_gap_median=("cos_gap", "median"),
    rel_flip_mean=("rel_l2_mean_flip", "mean"),
    rel_nonflip_mean=("rel_l2_mean_nonflip", "mean"),
).reset_index()
depth_gap.to_csv(OUT / "overall_flip_nonflip_gap_by_depth_bin.csv", index=False)

report.append("=== Overall flip-nonflip rel-L2 gap by depth bin ===")
report.append(depth_gap.to_string(index=False))
report.append("")

# -------------------------
# Category curves: damage/repair/w2w/stable
# -------------------------
cat_layer = layer[layer["group"].isin([
    "damage", "repair", "wrong_to_wrong", "stable_correct", "stable_wrong",
    "flip", "nonflip", "bar_lt1", "bar_ge1", "all"
])].copy()

cat_depth = cat_layer.copy()
cat_depth["depth_bin"] = pd.cut(
    cat_depth["depth_frac"],
    bins=np.linspace(0, 1, 11),
    include_lowest=True,
    labels=[f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
)
cat_depth_summary = cat_depth.groupby(["group", "depth_bin"], observed=True).agg(
    n_rows=("target_id", "size"),
    n_total=("n", "sum"),
    rel_l2_mean=("rel_l2_mean", "mean"),
    rel_l2_median=("rel_l2_median", "mean"),
    cos_dist_mean=("cos_dist_mean", "mean"),
).reset_index()
cat_depth_summary.to_csv(OUT / "category_drift_by_depth_bin.csv", index=False)

report.append("=== Category drift by depth bin: rel_l2_mean pivot ===")
piv = cat_depth_summary.pivot(index="depth_bin", columns="group", values="rel_l2_mean")
cols = [c for c in ["all","nonflip","flip","damage","repair","wrong_to_wrong","stable_correct","stable_wrong","bar_lt1","bar_ge1"] if c in piv.columns]
report.append(piv[cols].to_string())
report.append("")

# -------------------------
# Sample-level relation with trusted labels
# -------------------------
sample["flip"] = pd.to_numeric(sample["flip"], errors="coerce")
sample["damage"] = pd.to_numeric(sample["damage"], errors="coerce")
sample["repair"] = pd.to_numeric(sample["repair"], errors="coerce")
sample["wrong_to_wrong"] = pd.to_numeric(sample["wrong_to_wrong"], errors="coerce")
sample["boundary_ratio_num"] = pd.to_numeric(sample["boundary_ratio"], errors="coerce").replace([np.inf, -np.inf], np.nan)

sample_summary = sample.groupby("category").agg(
    n=("target_id", "size"),
    mean_rel_l2=("mean_rel_l2", "mean"),
    median_rel_l2=("mean_rel_l2", "median"),
    mean_cos_dist=("mean_cos_dist", "mean"),
    median_cos_dist=("mean_cos_dist", "median"),
    mean_boundary_ratio=("boundary_ratio_num", "mean"),
).reset_index().sort_values("mean_rel_l2", ascending=False)
sample_summary.to_csv(OUT / "sample_category_overall.csv", index=False)

report.append("=== Sample category overall ===")
report.append(sample_summary.to_string(index=False))
report.append("")

# -------------------------
# Config-level activation summary joined with theory.
# -------------------------
config_act = sample.groupby(["target_id","model_key","method_key","keep_tag"]).agg(
    n=("target_id", "size"),
    mean_rel_l2=("mean_rel_l2", "mean"),
    mean_cos_dist=("mean_cos_dist", "mean"),
    flip_mean_rel_l2=("mean_rel_l2", lambda x: np.nan),
).reset_index()

# More explicit config groups.
tmp = sample.copy()
cfg_groups = tmp.pivot_table(
    index=["target_id","model_key","method_key","keep_tag"],
    columns="category",
    values="mean_rel_l2",
    aggfunc="mean"
).reset_index()
cfg_groups.columns = [str(c) for c in cfg_groups.columns]

flip_cfg = sample.groupby(["target_id","model_key","method_key","keep_tag","flip"]).agg(
    mean_rel_l2=("mean_rel_l2","mean"),
    mean_cos_dist=("mean_cos_dist","mean"),
    n=("target_id","size")
).reset_index()
flip_piv = flip_cfg.pivot_table(
    index=["target_id","model_key","method_key","keep_tag"],
    columns="flip",
    values=["mean_rel_l2","mean_cos_dist","n"],
    aggfunc="first"
).reset_index()
flip_piv.columns = [f"{a}_{b}" if b != "" else str(a) for a,b in flip_piv.columns]
flip_piv = flip_piv.rename(columns={
    "mean_rel_l2_0": "nonflip_mean_rel_l2",
    "mean_rel_l2_1": "flip_mean_rel_l2",
    "mean_cos_dist_0": "nonflip_mean_cos_dist",
    "mean_cos_dist_1": "flip_mean_cos_dist",
    "n_0": "n_nonflip",
    "n_1": "n_flip",
})
flip_piv["sample_rel_l2_flip_gap"] = flip_piv["flip_mean_rel_l2"] - flip_piv["nonflip_mean_rel_l2"]
flip_piv["sample_cos_flip_gap"] = flip_piv["flip_mean_cos_dist"] - flip_piv["nonflip_mean_cos_dist"]

config_act = config_act.merge(flip_piv, on=["target_id","model_key","method_key","keep_tag"], how="left")
config_act = config_act.merge(cfg_groups, on=["target_id","model_key","method_key","keep_tag"], how="left")

config = config_act.merge(
    theory[["target_id","acc_delta_pct","pred_flip_pct","hidden_churn_pct","bar_ge1_flip_pct","prob_js_pct"]],
    on="target_id",
    how="left"
)
config.to_csv(OUT / "config_activation_join_theory.csv", index=False)

corr_cols = ["mean_rel_l2","mean_cos_dist","sample_rel_l2_flip_gap","sample_cos_flip_gap"]
target_cols = ["acc_delta_pct","pred_flip_pct","hidden_churn_pct","prob_js_pct"]
corr_rows = []
for a in corr_cols:
    for b in target_cols:
        if a in config and b in config:
            x = pd.to_numeric(config[a], errors="coerce")
            y = pd.to_numeric(config[b], errors="coerce")
            m = x.notna() & y.notna()
            if m.sum() >= 5:
                corr_rows.append({
                    "activation_metric": a,
                    "theory_metric": b,
                    "n": int(m.sum()),
                    "pearson": float(x[m].corr(y[m], method="pearson")),
                    "spearman": float(x[m].corr(y[m], method="spearman")),
                })
corr = pd.DataFrame(corr_rows)
corr.to_csv(OUT / "config_activation_theory_correlations.csv", index=False)

report.append("=== Config activation-theory correlations ===")
report.append(corr.to_string(index=False))
report.append("")

# -------------------------
# Ranking configs for phase-2 patching.
# Criteria: high hidden churn, enough flips, large flip-vs-nonflip activation gap.
# -------------------------
rank = config.copy()
rank["phase2_score"] = (
    rank["hidden_churn_pct"].fillna(0)
    + 0.5 * rank["pred_flip_pct"].fillna(0)
    + 100.0 * rank["sample_rel_l2_flip_gap"].fillna(0)
)
rank = rank.sort_values("phase2_score", ascending=False)
rank_cols = [
    "target_id","model_key","method_key","keep_tag","n","n_flip","n_nonflip",
    "acc_delta_pct","pred_flip_pct","hidden_churn_pct",
    "mean_rel_l2","flip_mean_rel_l2","nonflip_mean_rel_l2","sample_rel_l2_flip_gap",
    "mean_cos_dist","sample_cos_flip_gap","phase2_score"
]
rank[[c for c in rank_cols if c in rank.columns]].head(50).to_csv(OUT / "phase2_candidate_top50.csv", index=False)

report.append("=== Phase-2 candidate top 20 ===")
report.append(rank[[c for c in rank_cols if c in rank.columns]].head(20).to_string(index=False))
report.append("")

# -------------------------
# Model/method/keep summaries
# -------------------------
for keys, name in [
    (["model_key"], "by_model"),
    (["method_key"], "by_method"),
    (["keep_tag"], "by_keep"),
    (["model_key","method_key"], "by_model_method"),
    (["method_key","keep_tag"], "by_method_keep"),
]:
    g = config.groupby(keys, dropna=False).agg(
        n_configs=("target_id","nunique"),
        pred_flip_pct=("pred_flip_pct","mean"),
        hidden_churn_pct=("hidden_churn_pct","mean"),
        mean_rel_l2=("mean_rel_l2","mean"),
        mean_cos_dist=("mean_cos_dist","mean"),
        sample_rel_l2_flip_gap=("sample_rel_l2_flip_gap","mean"),
    ).reset_index()
    g.to_csv(OUT / f"config_summary_{name}.csv", index=False)

# -------------------------
# Figures
# -------------------------
plt.figure(figsize=(6.2, 4.0))
plt.plot(overall_gap["layer"], overall_gap["rel_flip_mean"], marker="o", label="Flip")
plt.plot(overall_gap["layer"], overall_gap["rel_nonflip_mean"], marker="o", label="Non-flip")
plt.xlabel("Layer")
plt.ylabel("Relative L2 drift")
plt.title("Layerwise prompt-state drift: flip vs non-flip")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "fig_flip_vs_nonflip_rel_l2_by_layer.pdf")
plt.savefig(OUT / "fig_flip_vs_nonflip_rel_l2_by_layer.png", dpi=220)
plt.close()

plt.figure(figsize=(6.2, 4.0))
plt.plot(overall_gap["layer"], overall_gap["rel_gap_mean"], marker="o")
plt.axhline(0, linewidth=1)
plt.xlabel("Layer")
plt.ylabel("Flip − non-flip relative L2 drift")
plt.title("Activation drift gap by layer")
plt.tight_layout()
plt.savefig(OUT / "fig_flip_nonflip_gap_by_layer.pdf")
plt.savefig(OUT / "fig_flip_nonflip_gap_by_layer.png", dpi=220)
plt.close()

plt.figure(figsize=(7.2, 4.2))
for gname in ["stable_correct", "stable_wrong", "damage", "repair", "wrong_to_wrong"]:
    sub = cat_depth_summary[cat_depth_summary["group"] == gname]
    if len(sub):
        x = np.arange(len(sub))
        plt.plot(x, sub["rel_l2_mean"], marker="o", label=gname)
plt.xticks(np.arange(10), [str(x) for x in cat_depth_summary["depth_bin"].drop_duplicates()], rotation=35)
plt.xlabel("Normalized depth bin")
plt.ylabel("Relative L2 drift")
plt.title("Layerwise drift by churn category")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "fig_category_rel_l2_by_depth.pdf")
plt.savefig(OUT / "fig_category_rel_l2_by_depth.png", dpi=220)
plt.close()

plt.figure(figsize=(6.4, 4.2))
x = pd.to_numeric(config["hidden_churn_pct"], errors="coerce")
y = pd.to_numeric(config["mean_rel_l2"], errors="coerce")
plt.scatter(x, y, s=18, alpha=0.65)
plt.xlabel("Hidden churn (%)")
plt.ylabel("Mean relative L2 drift")
plt.title("Config-level activation drift vs hidden churn")
plt.tight_layout()
plt.savefig(OUT / "fig_config_hidden_churn_vs_activation_drift.pdf")
plt.savefig(OUT / "fig_config_hidden_churn_vs_activation_drift.png", dpi=220)
plt.close()

# Write report
report_path = OUT / "MECHANISM_PHASE1_REPORT.txt"
report_path.write_text("\n".join(report))
print("WROTE", report_path)
print("\n".join(report[-20:]))
print("\nOUTPUT_DIR", OUT)
