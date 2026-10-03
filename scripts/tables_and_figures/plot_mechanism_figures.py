import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

PARSER = argparse.ArgumentParser(
    description="Generate mechanism figures from user-supplied summary CSV files."
)
PARSER.add_argument("--data-dir", type=Path, required=True)
PARSER.add_argument("--output-dir", type=Path, required=True)
ARGS = PARSER.parse_args()
DATA = ARGS.data_dir.expanduser().resolve()
FIGS = ARGS.output_dir.expanduser().resolve()
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COL_STABLE_1 = "#6E97BF"
COL_STABLE_2 = "#9DBBDA"
COL_REPAIR = "#64A89E"
COL_CHURN_1 = "#D86A61"
COL_CHURN_2 = "#B83D3A"
COL_GRAY = "#6B7280"
COL_DARK = "#111827"
COL_BLUE = "#4C78A8"
COL_RED = "#C44E52"
COL_GREEN = "#59A14F"
COL_PURPLE = "#7B6FD6"

def clean_axis(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, color="#E5E7EB", linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="both", colors="#111827", length=3, width=0.8)

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(FIGS / f"{name}.png", bbox_inches="tight")
    plt.close()

# -------------------------
# Figure 1: category-wise answer-path drift
# -------------------------
cat = pd.read_csv(DATA / "fig1_category_answer_path_drift.csv")
labels = ["Stable\nwrong", "Stable\ncorrect", "Repair", "Wrong-to-\nwrong", "Damage"]
values = cat["mean_choice_cos"].values
colors = [COL_STABLE_1, COL_STABLE_2, COL_REPAIR, COL_CHURN_1, COL_CHURN_2]

fig, ax = plt.subplots(figsize=(5.5, 3.25))
x = np.arange(len(values))
bars = ax.bar(x, values, color=colors, width=0.68, edgecolor="white", linewidth=0.8, zorder=3)

ax.set_ylabel("Mean answer-path cosine drift")
ax.set_title("Answer-path drift by churn category", pad=8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0.30, 0.46)
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
clean_axis(ax)

for b, v in zip(bars, values):
    ax.text(
        b.get_x() + b.get_width() / 2,
        v + 0.004,
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=COL_DARK,
    )

savefig("fig1_category_answer_path_drift")

# -------------------------
# Figure 2: config-level Spearman correlations
# -------------------------
corr = pd.read_csv(DATA / "fig2_config_level_correlations.csv")
target_labels = {
    "pred_flip_pct": "PredFlip",
    "hidden_churn_pct": "HiddenChurn",
}

metric_display = {
    "Prompt mean cosine": "Prompt\ncos",
    "Choice mean cosine": "Choice\nmean cos",
    "Choice max cosine": "Choice\nmax cos",
    "Choice gold cosine": "Gold\npath cos",
    "Choice dense-winner cosine": "Dense-winner\npath cos",
    "Choice compressed-winner cosine": "Compressed-winner\npath cos",
    "Prompt rel-L2": "Prompt\nrel-L2",
    "Choice rel-L2": "Choice\nrel-L2",
}

metrics = list(dict.fromkeys(corr["metric_name"].tolist()))
x = np.arange(len(metrics))
width = 0.36

pred = []
hidden = []
for m in metrics:
    sub = corr[corr["metric_name"] == m]
    pred.append(float(sub[sub["target"] == "pred_flip_pct"]["spearman"].iloc[0]))
    hidden.append(float(sub[sub["target"] == "hidden_churn_pct"]["spearman"].iloc[0]))

fig, ax = plt.subplots(figsize=(7.2, 3.45))
ax.bar(x - width / 2, pred, width=width, color=COL_BLUE, label="PredFlip", zorder=3)
ax.bar(x + width / 2, hidden, width=width, color=COL_RED, label="HiddenChurn", zorder=3)

ax.set_ylabel("Spearman correlation")
ax.set_title("Representation drift tracks configuration-level churn", pad=8)
ax.set_xticks(x)
ax.set_xticklabels([metric_display.get(m, m) for m in metrics], rotation=0)
ax.set_ylim(0.0, 0.88)
ax.legend(frameon=False, ncol=2, loc="upper right")
clean_axis(ax)

savefig("fig2_config_level_correlations")

# -------------------------
# Figure 3: BAR bucket boundary amplification
# -------------------------
bar = pd.read_csv(DATA / "fig3_bar_bucket_boundary_amplification.csv")
bucket_labels = ["BAR ≤ 0", "0 < BAR < 1", "1 ≤ BAR < 2", "BAR ≥ 2"]
x = np.arange(len(bar))

fig, ax1 = plt.subplots(figsize=(5.7, 3.35))
colors_bar = ["#B8C2CC", "#9DBBDA", "#D86A61", "#B83D3A"]
ax1.bar(x, bar["flip_rate"], color=colors_bar, width=0.68, edgecolor="white", linewidth=0.8, zorder=3)
ax1.set_ylabel("Flip rate")
ax1.set_ylim(0, 0.9)
ax1.set_xticks(x)
ax1.set_xticklabels(bucket_labels)
ax1.set_title("Boundary amplification explains individual flips", pad=8)
clean_axis(ax1)

for xi, v in zip(x, bar["flip_rate"]):
    ax1.text(xi, v + 0.025, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=COL_DARK)

ax2 = ax1.twinx()
ax2.plot(x, bar["mean_choice_cos"], color=COL_GREEN, marker="o", linewidth=1.8, markersize=5, label="Answer-path drift", zorder=4)
ax2.set_ylabel("Mean answer-path cosine drift")
ax2.set_ylim(0.30, 0.45)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_color("#333333")
ax2.tick_params(axis="y", colors="#111827", length=3, width=0.8)

ax2.legend(frameon=False, loc="upper left")

savefig("fig3_bar_bucket_boundary_amplification")

# -------------------------
# Figure 4: within-target flip AUC
# -------------------------
auc = pd.read_csv(DATA / "fig4_within_target_flip_auc.csv")
metric_labels = {
    "std_choice_cos": "Choice-std\ncos",
    "last_std_choice_cos": "Last choice-std\ncos",
    "max_choice_cos": "Max choice\ncos",
    "mean_choice_cos": "Mean choice\ncos",
    "compressed_pred_cos": "Compressed-winner\ncos",
    "gold_cos": "Gold\ncos",
    "dense_pred_cos": "Dense-winner\ncos",
}

fig, ax = plt.subplots(figsize=(6.4, 3.25))
x = np.arange(len(auc))
vals = auc["mean_auc"].values
bar_colors = [COL_PURPLE if v > 0.5 else COL_GRAY for v in vals]
ax.bar(x, vals, color=bar_colors, width=0.68, edgecolor="white", linewidth=0.8, zorder=3)
ax.axhline(0.5, color="#111827", linewidth=1.0, linestyle="--", label="Random", zorder=4)

ax.set_ylabel("Mean within-target AUC")
ax.set_title("Raw drift magnitude weakly identifies individual flips", pad=8)
ax.set_xticks(x)
ax.set_xticklabels([metric_labels.get(m, m) for m in auc["metric"]])
ax.set_ylim(0.44, 0.56)
ax.legend(frameon=False, loc="upper right")
clean_axis(ax)

for xi, v in zip(x, vals):
    ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=COL_DARK)

savefig("fig4_within_target_flip_auc")

# -------------------------
# Figure 5: layer-depth answer-path drift
# -------------------------
depth = pd.read_csv(DATA / "fig5_layer_depth_answer_path_drift.csv")
if "depth_bin" in depth.columns:
    depth_bins = depth["depth_bin"].astype(str).tolist()
else:
    depth_bins = depth.iloc[:, 0].astype(str).tolist()
    depth = depth.rename(columns={depth.columns[0]: "depth_bin"})

groups = [
    ("stable_wrong", "Stable wrong", COL_STABLE_1),
    ("stable_correct", "Stable correct", COL_STABLE_2),
    ("repair", "Repair", COL_REPAIR),
    ("wrong_to_wrong", "Wrong-to-wrong", COL_CHURN_1),
    ("damage", "Damage", COL_CHURN_2),
]

fig, ax = plt.subplots(figsize=(6.5, 3.6))
x = np.arange(len(depth))
for col, name, color in groups:
    if col in depth.columns:
        ax.plot(x, depth[col], marker="o", linewidth=1.8, markersize=4, label=name, color=color)

ax.set_ylabel("Mean answer-path cosine drift")
ax.set_xlabel("Normalized layer depth")
ax.set_title("Answer-path drift peaks in middle layers", pad=8)
ax.set_xticks(x)
ax.set_xticklabels(depth_bins, rotation=35, ha="right")
ax.set_ylim(0.12, 0.48)
ax.legend(frameon=False, ncol=2, loc="lower right")
clean_axis(ax)

savefig("fig5_layer_depth_answer_path_drift")

print("WROTE figures to", FIGS)
