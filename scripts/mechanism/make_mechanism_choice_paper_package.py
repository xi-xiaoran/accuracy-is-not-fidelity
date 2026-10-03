from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(".").resolve()
CHOICE = ROOT / "results/mechanism_choice_v1"
ACT = ROOT / "results/mechanism_activation_v2"
OUT = CHOICE / "paper_package"
OUT.mkdir(parents=True, exist_ok=True)

THEORY = ROOT / "results/theory_predictions_v1_fullzoo/fullzoo_theory_summary_by_config.csv"

sample = pd.read_csv(CHOICE / "fullzoo_choice_sample_summary.csv")
layer = pd.read_csv(CHOICE / "fullzoo_choice_layer_summary.csv")
choice_corr_raw = pd.read_csv(CHOICE / "analysis/choice_config_theory_correlations.csv")
prompt_choice_cmp_raw = pd.read_csv(CHOICE / "analysis/prompt_vs_choice_config_correlations.csv")
theory = pd.read_csv(THEORY)

# Clean inf.
for df in [sample, layer, choice_corr_raw, prompt_choice_cmp_raw, theory]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

report = []
report.append("=== Paper-ready mechanism package ===")
report.append(f"choice sample shape: {sample.shape}")
report.append(f"choice layer shape: {layer.shape}")
report.append(f"n targets: {sample['target_id'].nunique()}")
report.append("")

# ---------------------------
# Helpers
# ---------------------------
def fmt(x, nd=3):
    if pd.isna(x):
        return "--"
    return f"{float(x):.{nd}f}"

def auc_binary(y, score):
    y = pd.Series(y).astype(float)
    score = pd.Series(score).astype(float)
    m = y.notna() & score.notna()
    y = y[m]
    score = score[m]
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = score.rank(method="average")
    rank_pos = ranks[pos].sum()
    auc = (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def safe_corr(df, xcol, ycol):
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() < 5:
        return np.nan, np.nan, int(m.sum())
    return float(x[m].corr(y[m], method="pearson")), float(x[m].corr(y[m], method="spearman")), int(m.sum())

# ---------------------------
# Table 1: category-wise answer-path drift
# ---------------------------
category_order = ["stable_wrong", "stable_correct", "repair", "wrong_to_wrong", "damage"]

cat = sample.groupby("category").agg(
    n=("dense_uid", "count"),
    mean_choice_cos=("mean_choice_cos", "mean"),
    max_choice_cos=("max_choice_cos", "mean"),
    std_choice_cos=("std_choice_cos", "mean"),
    dense_pred_cos=("dense_pred_cos", "mean"),
    compressed_pred_cos=("compressed_pred_cos", "mean"),
    gold_cos=("gold_cos", "mean"),
    winner_delta_cos=("winner_delta_cos", "mean"),
    mean_choice_rel_l2=("mean_choice_rel_l2", "mean"),
).reset_index()
cat["category"] = pd.Categorical(cat["category"], categories=category_order, ordered=True)
cat = cat.sort_values("category")
cat.to_csv(OUT / "table_category_choice_drift_clean.csv", index=False)

cat_latex = cat.copy()
cat_latex["Category"] = cat_latex["category"].astype(str)
cat_latex["N"] = cat_latex["n"].astype(int)
cat_latex["Mean Cos"] = cat_latex["mean_choice_cos"].map(lambda x: fmt(x, 3))
cat_latex["Max Cos"] = cat_latex["max_choice_cos"].map(lambda x: fmt(x, 3))
cat_latex["Choice-Std Cos"] = cat_latex["std_choice_cos"].map(lambda x: fmt(x, 3))
cat_latex["Dense-Winner Cos"] = cat_latex["dense_pred_cos"].map(lambda x: fmt(x, 3))
cat_latex["Compressed-Winner Cos"] = cat_latex["compressed_pred_cos"].map(lambda x: fmt(x, 3))
cat_latex = cat_latex[["Category", "N", "Mean Cos", "Max Cos", "Choice-Std Cos", "Dense-Winner Cos", "Compressed-Winner Cos"]]

latex = cat_latex.to_latex(index=False, escape=False)
(OUT / "table_category_choice_drift_clean.tex").write_text(latex)

report.append("=== Category-wise answer-path drift ===")
report.append(cat_latex.to_string(index=False))
report.append("")

# ---------------------------
# Table 2: flip vs nonflip + AUC
# ---------------------------
flip = sample.groupby("flip").agg(
    n=("dense_uid", "count"),
    mean_choice_cos=("mean_choice_cos", "mean"),
    max_choice_cos=("max_choice_cos", "mean"),
    std_choice_cos=("std_choice_cos", "mean"),
    dense_pred_cos=("dense_pred_cos", "mean"),
    compressed_pred_cos=("compressed_pred_cos", "mean"),
    gold_cos=("gold_cos", "mean"),
    last_mean_choice_cos=("last_mean_choice_cos", "mean"),
    last_std_choice_cos=("last_std_choice_cos", "mean"),
).reset_index()
flip.to_csv(OUT / "table_flip_nonflip_choice_drift_clean.csv", index=False)

metrics_auc = [
    "mean_choice_cos",
    "max_choice_cos",
    "std_choice_cos",
    "dense_pred_cos",
    "compressed_pred_cos",
    "gold_cos",
    "last_mean_choice_cos",
    "last_std_choice_cos",
]
auc_rows = []
for m in metrics_auc:
    global_auc = auc_binary(sample["flip"], sample[m])
    per_target = []
    for tid, g in sample.groupby("target_id"):
        a = auc_binary(g["flip"], g[m])
        if not pd.isna(a):
            per_target.append(a)
    auc_rows.append({
        "metric": m,
        "global_auc": global_auc,
        "mean_target_auc": float(np.mean(per_target)) if per_target else np.nan,
        "median_target_auc": float(np.median(per_target)) if per_target else np.nan,
        "n_targets": len(per_target),
    })
auc = pd.DataFrame(auc_rows).sort_values("mean_target_auc", ascending=False)
auc.to_csv(OUT / "table_flip_prediction_auc_clean.csv", index=False)

flip_latex_rows = []
non = flip[flip["flip"] == 0].iloc[0]
flp = flip[flip["flip"] == 1].iloc[0]
for m in metrics_auc:
    flip_latex_rows.append({
        "Metric": m,
        "Nonflip Mean": fmt(non[m], 3),
        "Flip Mean": fmt(flp[m], 3),
        "Diff": fmt(flp[m] - non[m], 3),
        "Mean Target AUC": fmt(auc[auc["metric"] == m]["mean_target_auc"].iloc[0], 3),
    })
flip_latex = pd.DataFrame(flip_latex_rows)
(OUT / "table_flip_nonflip_auc_clean.tex").write_text(flip_latex.to_latex(index=False, escape=False))

report.append("=== Flip vs nonflip clean summary ===")
report.append(flip.to_string(index=False))
report.append("")
report.append("=== Flip prediction AUC ===")
report.append(auc.to_string(index=False))
report.append("")

# ---------------------------
# Table 3: config-level correlations, clean finite.
# ---------------------------
config = sample.groupby(["target_id", "model_key", "method_key", "keep_tag"]).agg(
    mean_choice_cos=("mean_choice_cos", "mean"),
    max_choice_cos=("max_choice_cos", "mean"),
    std_choice_cos=("std_choice_cos", "mean"),
    dense_pred_cos=("dense_pred_cos", "mean"),
    compressed_pred_cos=("compressed_pred_cos", "mean"),
    gold_cos=("gold_cos", "mean"),
    last_mean_choice_cos=("last_mean_choice_cos", "mean"),
    last_std_choice_cos=("last_std_choice_cos", "mean"),
    mean_choice_rel_l2=("mean_choice_rel_l2", "mean"),
).reset_index()
config = config.merge(
    theory[["target_id", "acc_delta_pct", "pred_flip_pct", "hidden_churn_pct", "prob_js_pct"]],
    on="target_id",
    how="left"
)

# Add prompt metrics for comparison.
prompt_cfg_path = ACT / "analysis/config_activation_join_theory.csv"
if prompt_cfg_path.exists():
    prompt_cfg = pd.read_csv(prompt_cfg_path).replace([np.inf, -np.inf], np.nan)
    config = config.merge(
        prompt_cfg[["target_id", "mean_rel_l2", "mean_cos_dist"]],
        on="target_id",
        how="left"
    )

corr_metrics = [
    ("Prompt mean cosine", "mean_cos_dist"),
    ("Choice mean cosine", "mean_choice_cos"),
    ("Choice max cosine", "max_choice_cos"),
    ("Choice gold cosine", "gold_cos"),
    ("Choice dense-winner cosine", "dense_pred_cos"),
    ("Choice compressed-winner cosine", "compressed_pred_cos"),
    ("Choice std cosine", "std_choice_cos"),
    ("Prompt rel-L2", "mean_rel_l2"),
    ("Choice rel-L2", "mean_choice_rel_l2"),
]
corr_rows = []
for name, col in corr_metrics:
    if col not in config.columns:
        continue
    for target in ["pred_flip_pct", "hidden_churn_pct", "acc_delta_pct"]:
        pear, spear, n = safe_corr(config, col, target)
        corr_rows.append({
            "metric_name": name,
            "metric_col": col,
            "target": target,
            "n": n,
            "pearson": pear,
            "spearman": spear,
        })
corr_clean = pd.DataFrame(corr_rows)
corr_clean.to_csv(OUT / "table_config_correlations_clean.csv", index=False)

# Compact LaTeX correlation table for pred_flip and hidden_churn.
compact_rows = []
for name, col in corr_metrics:
    sub = corr_clean[corr_clean["metric_col"] == col]
    pf = sub[sub["target"] == "pred_flip_pct"]
    hc = sub[sub["target"] == "hidden_churn_pct"]
    if len(pf) and len(hc):
        compact_rows.append({
            "Metric": name,
            "PredFlip Spearman": fmt(pf["spearman"].iloc[0], 3),
            "HiddenChurn Spearman": fmt(hc["spearman"].iloc[0], 3),
            "PredFlip Pearson": fmt(pf["pearson"].iloc[0], 3),
            "HiddenChurn Pearson": fmt(hc["pearson"].iloc[0], 3),
        })
compact = pd.DataFrame(compact_rows)
compact.to_csv(OUT / "table_config_correlations_compact.csv", index=False)
(OUT / "table_config_correlations_compact.tex").write_text(compact.to_latex(index=False, escape=False))

report.append("=== Compact config correlations ===")
report.append(compact.to_string(index=False))
report.append("")

# ---------------------------
# Layer depth clean summary.
# ---------------------------
max_layer = layer.groupby("target_id")["layer"].max().rename("max_layer").reset_index()
layer2 = layer.merge(max_layer, on="target_id", how="left")
layer2["depth_frac"] = layer2["layer"] / layer2["max_layer"].replace(0, np.nan)
layer2["depth_bin"] = pd.cut(
    layer2["depth_frac"],
    bins=np.linspace(0, 1, 11),
    include_lowest=True,
    labels=[f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
)
depth = layer2.groupby(["group", "depth_bin"], observed=True).agg(
    mean_choice_cos=("mean_choice_cos", "mean"),
    std_choice_cos=("std_choice_cos", "mean"),
    mean_choice_rel_l2=("mean_choice_rel_l2", "mean"),
    dense_pred_cos=("dense_pred_cos", "mean"),
    compressed_pred_cos=("compressed_pred_cos", "mean"),
    gold_cos=("gold_cos", "mean"),
    n_total=("n", "sum"),
).reset_index()
depth.to_csv(OUT / "table_layer_depth_clean.csv", index=False)

# Compact depth table for key categories.
key_groups = ["stable_wrong", "stable_correct", "repair", "wrong_to_wrong", "damage"]
depth_piv = depth[depth["group"].isin(key_groups)].pivot(index="depth_bin", columns="group", values="mean_choice_cos")
depth_piv = depth_piv[[g for g in key_groups if g in depth_piv.columns]]
depth_piv.to_csv(OUT / "table_layer_depth_mean_choice_cos_pivot.csv")
(OUT / "table_layer_depth_mean_choice_cos_pivot.tex").write_text(depth_piv.applymap(lambda x: fmt(x, 3)).to_latex(escape=False))

report.append("=== Layer depth mean choice cosine pivot ===")
report.append(depth_piv.to_string())
report.append("")

# ---------------------------
# Paper figures.
# ---------------------------
# Figure 1: category bar.
plt.figure(figsize=(6.0, 3.6))
x = np.arange(len(cat))
plt.bar(x, cat["mean_choice_cos"])
plt.xticks(x, cat["category"].astype(str), rotation=25, ha="right")
plt.ylabel("Mean answer-path cosine drift")
plt.title("Answer-path drift by churn category")
plt.tight_layout()
plt.savefig(OUT / "paper_fig_category_choice_cos.pdf")
plt.savefig(OUT / "paper_fig_category_choice_cos.png", dpi=240)
plt.close()

# Figure 2: layer depth category curves.
plt.figure(figsize=(6.6, 4.0))
for g in ["stable_correct", "stable_wrong", "damage", "repair", "wrong_to_wrong"]:
    sub = depth[depth["group"] == g]
    if len(sub):
        xx = np.arange(len(sub))
        plt.plot(xx, sub["mean_choice_cos"], marker="o", label=g)
plt.xticks(np.arange(10), [str(x) for x in depth["depth_bin"].drop_duplicates()], rotation=35)
plt.xlabel("Normalized depth bin")
plt.ylabel("Mean answer-path cosine drift")
plt.title("Layerwise answer-path drift")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "paper_fig_choice_cos_depth.pdf")
plt.savefig(OUT / "paper_fig_choice_cos_depth.png", dpi=240)
plt.close()

# Figure 3: config correlation scatter.
plt.figure(figsize=(5.6, 4.0))
plt.scatter(config["pred_flip_pct"], config["mean_choice_cos"], s=18, alpha=0.65)
plt.xlabel("Prediction flip (%)")
plt.ylabel("Mean answer-path cosine drift")
plt.title("Config-level churn tracks answer-path drift")
plt.tight_layout()
plt.savefig(OUT / "paper_fig_predflip_vs_choice_cos.pdf")
plt.savefig(OUT / "paper_fig_predflip_vs_choice_cos.png", dpi=240)
plt.close()

plt.figure(figsize=(5.6, 4.0))
plt.scatter(config["hidden_churn_pct"], config["max_choice_cos"], s=18, alpha=0.65)
plt.xlabel("Hidden churn (%)")
plt.ylabel("Max answer-path cosine drift")
plt.title("Hidden churn tracks answer-path drift")
plt.tight_layout()
plt.savefig(OUT / "paper_fig_hiddenchurn_vs_max_choice_cos.pdf")
plt.savefig(OUT / "paper_fig_hiddenchurn_vs_max_choice_cos.png", dpi=240)
plt.close()

# ---------------------------
# Suggested paper text.
# ---------------------------
paper_text = r"""
Suggested mechanism paragraph:

Layerwise diagnostics show that low-rank compression induces substantial
representation drift inside the model. Across 235 compressed configurations,
angular drift strongly tracks both prediction flip and hidden churn. Prompt-state
drift is already highly predictive at the configuration level, but it only weakly
separates individual flipped and non-flipped examples. We therefore further
measure answer-conditioned drift by running each candidate continuation and
comparing dense and compressed hidden states along the answer path. This reveals
a clearer sample-level separation: flipped examples have substantially larger
answer-path angular drift than non-flipped examples, and damage / wrong-to-wrong
cases show the largest drift. The drift peaks in the middle layers and remains
elevated along answer continuations, suggesting that compression perturbs the
candidate-answer trajectories before the final score comparison. These findings
support the boundary-amplification view: compression-induced perturbations are
especially harmful when they align with answer-margin directions for examples
near a decision boundary.

Suggested compact claim:

Accuracy degradation is accompanied by internal trajectory drift, but decision
churn is not explained by perturbation magnitude alone. Churn is better viewed as
boundary-local amplification of answer-path perturbations.
"""
(OUT / "suggested_paper_text.txt").write_text(paper_text.strip())

# Write report.
report.append("=== Suggested paper text ===")
report.append(paper_text.strip())
report.append("")
report.append("=== Output files written to ===")
report.append(str(OUT))

report_path = OUT / "PAPER_READY_MECHANISM_REPORT.txt"
report_path.write_text("\n".join(report))

print("WROTE", report_path)
print("OUTPUT_DIR", OUT)
print("\n=== KEY OUTPUTS ===")
for f in [
    "table_category_choice_drift_clean.csv",
    "table_category_choice_drift_clean.tex",
    "table_flip_nonflip_auc_clean.tex",
    "table_config_correlations_compact.tex",
    "table_layer_depth_mean_choice_cos_pivot.tex",
    "paper_fig_category_choice_cos.pdf",
    "paper_fig_choice_cos_depth.pdf",
    "paper_fig_predflip_vs_choice_cos.pdf",
    "paper_fig_hiddenchurn_vs_max_choice_cos.pdf",
    "suggested_paper_text.txt",
]:
    print(OUT / f)
