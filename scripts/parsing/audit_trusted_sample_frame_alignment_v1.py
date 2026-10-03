from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
THEORY = ROOT / "results/theory_predictions_v1"
FROZEN = ROOT / "results/theory_predictions_v1_fullzoo/fullzoo_theory_summary_by_config.csv"
AUD = ROOT / "results/mechanism_activation_v1/audits"
AUD.mkdir(parents=True, exist_ok=True)

def pct(x):
    return 100.0 * float(x)

def summarize_one(p: Path):
    tid = p.name.replace("_sample_theory_frame.csv", "")
    df = pd.read_csv(p)
    ev = df[df["split"] == "eval"].copy()

    dense_acc = ev["dense_correct"].mean()
    comp_acc = ev["compressed_correct"].mean()
    acc_delta = comp_acc - dense_acc
    pred_flip = ev["flip"].mean()

    out = {
        "target_id": tid,
        "n_eval_rebuilt": len(ev),
        "dense_acc_pct_rebuilt": pct(dense_acc),
        "compressed_acc_pct_rebuilt": pct(comp_acc),
        "acc_delta_pct_rebuilt": pct(acc_delta),
        "pred_flip_pct_rebuilt": pct(pred_flip),
        "hidden_churn_pct_rebuilt": pct(pred_flip - abs(acc_delta)),
        "damage_pct_rebuilt": pct(ev["damage"].mean()),
        "repair_pct_rebuilt": pct(ev["repair"].mean()),
        "wrong_to_wrong_pct_rebuilt": pct(ev["wrong_to_wrong"].mean()),
        "correct_to_correct_pct_rebuilt": pct(ev["correct_to_correct"].mean()),
        "prob_js_pct_rebuilt": pct(ev["prob_js"].mean()),
        "mean_dense_margin_rebuilt": float(ev["dense_margin"].mean()),
        "mean_boundary_ratio_rebuilt": float(ev["boundary_ratio"].replace([np.inf, -np.inf], np.nan).mean()),
    }

    bar_lt1 = ev[ev["boundary_ratio"] < 1]
    bar_ge1 = ev[ev["boundary_ratio"] >= 1]
    bar_1_2 = ev[(ev["boundary_ratio"] >= 1) & (ev["boundary_ratio"] < 2)]
    bar_gt2 = ev[ev["boundary_ratio"] >= 2]

    def flip_rate(g):
        return float(g["flip"].mean()) if len(g) else np.nan

    out.update({
        "bar_lt1_n_rebuilt": len(bar_lt1),
        "bar_ge1_n_rebuilt": len(bar_ge1),
        "bar_1_2_n_rebuilt": len(bar_1_2),
        "bar_gt2_n_rebuilt": len(bar_gt2),
        "bar_lt1_flip_pct_rebuilt": pct(flip_rate(bar_lt1)),
        "bar_ge1_flip_pct_rebuilt": pct(flip_rate(bar_ge1)),
        "bar_1_2_flip_pct_rebuilt": pct(flip_rate(bar_1_2)),
        "bar_gt2_flip_pct_rebuilt": pct(flip_rate(bar_gt2)),
    })
    return out

files = sorted(THEORY.glob("*_sample_theory_frame.csv"))
print("N_SAMPLE_FRAME_FILES", len(files))

rebuilt = pd.DataFrame([summarize_one(p) for p in files])
frozen = pd.read_csv(FROZEN)

merged = rebuilt.merge(frozen, on="target_id", how="left", suffixes=("", "_frozen"))

check_cols = [
    "n_eval",
    "dense_acc_pct",
    "compressed_acc_pct",
    "acc_delta_pct",
    "pred_flip_pct",
    "hidden_churn_pct",
    "damage_pct",
    "repair_pct",
    "wrong_to_wrong_pct",
    "correct_to_correct_pct",
    "prob_js_pct",
    "mean_dense_margin",
    "mean_boundary_ratio",
    "bar_lt1_n",
    "bar_ge1_n",
    "bar_1_2_n",
    "bar_gt2_n",
    "bar_lt1_flip_pct",
    "bar_ge1_flip_pct",
    "bar_1_2_flip_pct",
    "bar_gt2_flip_pct",
]

summary_rows = []
for c in check_cols:
    rb = f"{c}_rebuilt"
    if rb not in merged.columns or c not in merged.columns:
        continue
    diff = (pd.to_numeric(merged[rb], errors="coerce") - pd.to_numeric(merged[c], errors="coerce")).abs()
    merged[f"{c}_absdiff"] = diff
    summary_rows.append({
        "metric": c,
        "n": int(diff.notna().sum()),
        "mean_absdiff": float(diff.mean()),
        "median_absdiff": float(diff.median()),
        "max_absdiff": float(diff.max()),
        "n_le_1e-9": int((diff <= 1e-9).sum()),
        "n_le_1e-6": int((diff <= 1e-6).sum()),
        "n_le_1e-4": int((diff <= 1e-4).sum()),
    })

merged_path = AUD / "trusted_sample_frame_alignment_merged_v1.csv"
summary_path = AUD / "trusted_sample_frame_alignment_summary_v1.csv"
merged.to_csv(merged_path, index=False)
pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

print("WROTE", merged_path, merged.shape)
print("WROTE", summary_path)

print("\nALIGNMENT_SUMMARY")
s = pd.DataFrame(summary_rows)
print(s.to_string(index=False))

print("\nGLOBAL_REBUILT_MEANS")
for c in ["acc_delta_pct_rebuilt", "pred_flip_pct_rebuilt", "hidden_churn_pct_rebuilt"]:
    print(c, rebuilt[c].mean())

print("\nWORST_DIFFS")
for c in ["acc_delta_pct", "pred_flip_pct", "hidden_churn_pct", "bar_lt1_flip_pct", "bar_ge1_flip_pct"]:
    dc = f"{c}_absdiff"
    if dc in merged.columns:
        print("\n", c)
        show = ["target_id", f"{c}_rebuilt", c, dc]
        print(merged.sort_values(dc, ascending=False)[show].head(8).to_string(index=False))

print("\nDONE")
