from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(".").resolve()
BASE = ROOT / "results/mechanism_choice_v1"
OUT = BASE / "deconfounded"
OUT.mkdir(parents=True, exist_ok=True)

sample = pd.read_csv(BASE / "fullzoo_choice_sample_summary.csv")
sample = sample.replace([np.inf, -np.inf], np.nan)

metrics = [
    "mean_choice_cos",
    "max_choice_cos",
    "std_choice_cos",
    "dense_pred_cos",
    "compressed_pred_cos",
    "gold_cos",
    "last_mean_choice_cos",
    "last_std_choice_cos",
]

report = []
report.append("=== Deconfounded choice-mechanism package ===")
report.append(f"sample shape: {sample.shape}")
report.append(f"n targets: {sample['target_id'].nunique()}")
report.append("")

def fmt(x, nd=4):
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
    return float((rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

# -------------------------
# 1. Within-target flip-vs-nonflip deltas
# -------------------------
within_rows = []
for m in metrics:
    per = sample.groupby(["target_id", "flip"])[m].mean().unstack()
    if 0 not in per.columns or 1 not in per.columns:
        continue
    per["flip_minus_nonflip"] = per[1] - per[0]
    per = per.dropna(subset=["flip_minus_nonflip"])
    within_rows.append({
        "metric": m,
        "n_targets": len(per),
        "mean_delta": per["flip_minus_nonflip"].mean(),
        "median_delta": per["flip_minus_nonflip"].median(),
        "frac_positive": (per["flip_minus_nonflip"] > 0).mean(),
        "q25_delta": per["flip_minus_nonflip"].quantile(0.25),
        "q75_delta": per["flip_minus_nonflip"].quantile(0.75),
    })

within_delta = pd.DataFrame(within_rows)
within_delta.to_csv(OUT / "within_target_flip_nonflip_deltas.csv", index=False)

report.append("=== Within-target flip minus nonflip deltas ===")
report.append(within_delta.to_string(index=False))
report.append("")

# -------------------------
# 2. Within-target AUC
# -------------------------
auc_rows = []
for m in metrics:
    aucs = []
    for tid, g in sample.groupby("target_id"):
        a = auc_binary(g["flip"], g[m])
        if not pd.isna(a):
            aucs.append(a)
    auc_rows.append({
        "metric": m,
        "n_targets": len(aucs),
        "mean_auc": float(np.mean(aucs)) if aucs else np.nan,
        "median_auc": float(np.median(aucs)) if aucs else np.nan,
        "frac_auc_gt_0p5": float(np.mean(np.array(aucs) > 0.5)) if aucs else np.nan,
        "q25_auc": float(np.quantile(aucs, 0.25)) if aucs else np.nan,
        "q75_auc": float(np.quantile(aucs, 0.75)) if aucs else np.nan,
    })

auc_df = pd.DataFrame(auc_rows).sort_values("mean_auc", ascending=False)
auc_df.to_csv(OUT / "within_target_flip_auc.csv", index=False)

report.append("=== Within-target flip AUC ===")
report.append(auc_df.to_string(index=False))
report.append("")

# -------------------------
# 3. Target-centered category residuals
# -------------------------
resid = sample.copy()
for m in metrics:
    resid[f"{m}_target_resid"] = resid[m] - resid.groupby("target_id")[m].transform("mean")
    resid[f"{m}_target_task_resid"] = resid[m] - resid.groupby(["target_id", "task"])[m].transform("mean")

cat_rows = []
for suffix in ["target_resid", "target_task_resid"]:
    for cat, g in resid.groupby("category"):
        row = {"centering": suffix, "category": cat, "n": len(g)}
        for m in metrics:
            row[m] = g[f"{m}_{suffix}"].mean()
        cat_rows.append(row)

cat_resid = pd.DataFrame(cat_rows)
cat_resid.to_csv(OUT / "category_centered_residuals.csv", index=False)

report.append("=== Category residuals after target centering ===")
report.append(cat_resid[cat_resid["centering"] == "target_resid"].to_string(index=False))
report.append("")
report.append("=== Category residuals after target+task centering ===")
report.append(cat_resid[cat_resid["centering"] == "target_task_resid"].to_string(index=False))
report.append("")

# -------------------------
# 4. Paired category differences within target
# -------------------------
category_pairs = [
    ("damage", "stable_correct"),
    ("wrong_to_wrong", "stable_wrong"),
    ("repair", "stable_wrong"),
    ("damage", "repair"),
    ("damage", "stable_wrong"),
]

pair_rows = []
for m in metrics:
    tab = sample.groupby(["target_id", "category"])[m].mean().unstack()
    for a, b in category_pairs:
        if a in tab.columns and b in tab.columns:
            d = (tab[a] - tab[b]).dropna()
            pair_rows.append({
                "metric": m,
                "pair": f"{a} - {b}",
                "n_targets": len(d),
                "mean_delta": d.mean(),
                "median_delta": d.median(),
                "frac_positive": (d > 0).mean(),
                "q25_delta": d.quantile(0.25),
                "q75_delta": d.quantile(0.75),
            })

pair_df = pd.DataFrame(pair_rows)
pair_df.to_csv(OUT / "within_target_category_pair_deltas.csv", index=False)

report.append("=== Within-target category pair deltas ===")
for m in ["mean_choice_cos", "max_choice_cos", "std_choice_cos", "last_mean_choice_cos"]:
    sub = pair_df[pair_df["metric"] == m]
    report.append(f"\n--- {m} ---")
    report.append(sub.to_string(index=False))
report.append("")

# -------------------------
# 5. BAR sanity within choice package if boundary_ratio exists
# -------------------------
bar_rows = []
if "boundary_ratio" in sample.columns:
    br = pd.to_numeric(sample["boundary_ratio"], errors="coerce")
    tmp = sample.copy()
    tmp["boundary_ratio_num"] = br
    tmp["bar_bucket"] = pd.cut(
        br,
        bins=[-np.inf, 0, 1, 2, np.inf],
        labels=["bar_le0", "0_1", "1_2", "ge2"]
    )
    bar = tmp.groupby("bar_bucket", observed=True).agg(
        n=("dense_uid", "count"),
        flip_rate=("flip", "mean"),
        mean_choice_cos=("mean_choice_cos", "mean"),
        max_choice_cos=("max_choice_cos", "mean"),
        std_choice_cos=("std_choice_cos", "mean"),
    ).reset_index()
    bar.to_csv(OUT / "bar_bucket_choice_drift.csv", index=False)

    report.append("=== BAR bucket sanity ===")
    report.append(bar.to_string(index=False))
    report.append("")

# -------------------------
# 6. Figures
# -------------------------
plt.figure(figsize=(6.2, 3.8))
plot_df = within_delta.copy()
x = np.arange(len(plot_df))
plt.axhline(0.0, linewidth=1)
plt.bar(x, plot_df["mean_delta"])
plt.xticks(x, plot_df["metric"], rotation=35, ha="right")
plt.ylabel("Mean within-target flip - nonflip")
plt.title("Within-target drift gap is weak")
plt.tight_layout()
plt.savefig(OUT / "fig_within_target_flip_nonflip_delta.pdf")
plt.savefig(OUT / "fig_within_target_flip_nonflip_delta.png", dpi=240)
plt.close()

plt.figure(figsize=(6.2, 3.8))
plot_df = auc_df.copy()
x = np.arange(len(plot_df))
plt.axhline(0.5, linewidth=1)
plt.bar(x, plot_df["mean_auc"])
plt.xticks(x, plot_df["metric"], rotation=35, ha="right")
plt.ylabel("Mean target-level AUC")
plt.title("Drift magnitude alone weakly predicts individual flips")
plt.tight_layout()
plt.savefig(OUT / "fig_within_target_flip_auc.pdf")
plt.savefig(OUT / "fig_within_target_flip_auc.png", dpi=240)
plt.close()

# Category residual figure for target+task centered mean_choice_cos.
sub = cat_resid[cat_resid["centering"] == "target_task_resid"].copy()
order = ["stable_wrong", "stable_correct", "repair", "wrong_to_wrong", "damage"]
sub["category"] = pd.Categorical(sub["category"], categories=order, ordered=True)
sub = sub.sort_values("category")

plt.figure(figsize=(5.8, 3.6))
x = np.arange(len(sub))
plt.axhline(0.0, linewidth=1)
plt.bar(x, sub["mean_choice_cos"])
plt.xticks(x, sub["category"].astype(str), rotation=25, ha="right")
plt.ylabel("Target+task centered mean choice cosine")
plt.title("Category drift after target/task centering")
plt.tight_layout()
plt.savefig(OUT / "fig_category_target_task_centered_choice_cos.pdf")
plt.savefig(OUT / "fig_category_target_task_centered_choice_cos.png", dpi=240)
plt.close()

# -------------------------
# 7. Paper-safe wording
# -------------------------
paper_text = r"""
Paper-safe interpretation:

Answer-path angular drift is a strong configuration-level severity diagnostic:
compressed configurations with larger answer-path drift also exhibit larger
prediction flip and hidden churn. However, after conditioning on a fixed
compressed configuration, drift magnitude alone only weakly identifies which
individual examples flip. This negative result is important: it shows that
decision churn is not merely a function of raw representation perturbation size.

Together with the BAR analysis, this supports a two-level mechanism. At the
configuration level, low-rank compression induces systematic internal trajectory
drift. At the example level, a flip occurs only when the induced perturbation is
aligned with an answer-margin direction and is large relative to the dense
decision margin. Thus, representation drift explains compression severity, while
BAR explains boundary-local decision crossing.
"""
(OUT / "paper_safe_interpretation.txt").write_text(paper_text.strip())

report.append("=== Paper-safe interpretation ===")
report.append(paper_text.strip())
report.append("")
report.append("=== Outputs ===")
report.append(str(OUT))

report_path = OUT / "DECONFOUNDED_CHOICE_REPORT.txt"
report_path.write_text("\n".join(report))

print("WROTE", report_path)
print("OUTPUT_DIR", OUT)
print("\nKEY FILES")
for f in [
    "within_target_flip_nonflip_deltas.csv",
    "within_target_flip_auc.csv",
    "category_centered_residuals.csv",
    "within_target_category_pair_deltas.csv",
    "bar_bucket_choice_drift.csv",
    "paper_safe_interpretation.txt",
    "fig_within_target_flip_nonflip_delta.pdf",
    "fig_within_target_flip_auc.pdf",
    "fig_category_target_task_centered_choice_cos.pdf",
]:
    p = OUT / f
    print(p, "exists=", p.exists())
