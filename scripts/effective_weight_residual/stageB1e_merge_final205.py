from pathlib import Path
import hashlib
import pandas as pd
import numpy as np

ROOT = Path(".").resolve()

B1C = ROOT / "results/paper_packages/stageB1c_effective_weight_residual_fullzoo_v1_20260702"
B1D = ROOT / "results/paper_packages/stageB1d_effective_weight_patch_dobi_basis_v1_20260702"
PKG = ROOT / "results/paper_packages/stageB1e_effective_weight_residual_final205_v1_20260702"
DATA = PKG / "data"
LOGS = PKG / "logs"
DATA.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

SUMMARY = ROOT / "results/theory_predictions_v1_fullzoo/fullzoo_theory_summary_by_config.csv"

def find_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    low = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None

def agg_rel(g):
    dense_sq = pd.to_numeric(g["dense_sq"], errors="coerce").sum()
    diff_sq = pd.to_numeric(g["diff_sq"], errors="coerce").sum()
    return pd.Series({
        "n_modules": len(g),
        "n_params": pd.to_numeric(g["n_params"], errors="coerce").sum(),
        "dense_sq": dense_sq,
        "diff_sq": diff_sq,
        "rel_frob": (diff_sq / (dense_sq + 1e-12)) ** 0.5,
        "mean_module_rel_frob": pd.to_numeric(g["rel_frob"], errors="coerce").mean(),
        "median_module_rel_frob": pd.to_numeric(g["rel_frob"], errors="coerce").median(),
        "mean_cosine": pd.to_numeric(g["cosine"], errors="coerce").mean(),
        "n_factorized": int(g["materialization"].isin(["ALinear@BLinear", "BLinear@ALinear", "u_proj@v_proj"]).sum()),
        "n_linear": int((g["materialization"] == "Linear").sum()),
    })

# B1c original effective residual rows.
b1c = pd.read_csv(B1C / "data/stageB1c_effective_residual_all.csv")
b1c = b1c[b1c["ok"] == True].copy()
b1c["source_stage"] = "B1c"

# B1d patch rows.
patch_frames = []
for p in sorted((B1D / "data/per_target").glob("*_effective_residual_patch.csv")):
    df = pd.read_csv(p)
    df = df[df["ok"] == True].copy()
    if len(df):
        df["source_stage"] = "B1d_patch"
        patch_frames.append(df)

b1d = pd.concat(patch_frames, ignore_index=True) if patch_frames else pd.DataFrame()

all_res = pd.concat([b1c, b1d], ignore_index=True)

# Safety check for duplicate module rows.
dup_cols = ["target_id", "module_name"]
dups = all_res.duplicated(dup_cols).sum()
if dups:
    raise RuntimeError(f"Duplicate target-module rows after merge: {dups}")

all_res.to_csv(DATA / "stageB1e_effective_residual_all_final205.csv", index=False)

target_agg = all_res.groupby(["target_id", "model_key", "method_key", "keep_ratio"]).apply(agg_rel).reset_index()
module_agg = all_res.groupby(["target_id", "model_key", "method_key", "keep_ratio", "module_type"]).apply(agg_rel).reset_index()
method_agg = all_res.groupby(["method_key"]).apply(agg_rel).reset_index()
model_method_agg = all_res.groupby(["model_key", "method_key"]).apply(agg_rel).reset_index()

target_agg.to_csv(DATA / "stageB1e_target_effective_residual_summary_final205.csv", index=False)
module_agg.to_csv(DATA / "stageB1e_target_module_effective_residual_summary_final205.csv", index=False)
method_agg.to_csv(DATA / "stageB1e_method_effective_residual_summary_final205.csv", index=False)
model_method_agg.to_csv(DATA / "stageB1e_model_method_effective_residual_summary_final205.csv", index=False)

# Merge behavior metrics.
summary = pd.read_csv(SUMMARY)
metric_map = {
    "acc_delta": ["acc_delta", "acc_delta_pct", "delta_acc"],
    "pred_flip": ["pred_flip", "pred_flip_pct", "prediction_flip"],
    "hidden_churn": ["hidden_churn", "hidden_churn_pct"],
    "prob_js": ["prob_js", "prob_js_pct"],
    "bar_ge1_flip": ["bar_ge1_flip", "bar_ge1_flip_pct"],
    "bar_lt1_flip": ["bar_lt1_flip", "bar_lt1_flip_pct"],
    "compressed_acc": ["compressed_acc", "compressed_acc_pct"],
}
cols = {"target_id": find_col(summary, ["target_id"])}
for k, names in metric_map.items():
    cols[k] = find_col(summary, names)

merge_cols = ["target_id"] + [c for k, c in cols.items() if k != "target_id" and c is not None]
metric_df = summary[merge_cols].copy()
metric_df = metric_df.rename(columns={cols[k]: k for k in metric_map if cols.get(k) is not None})

target_behavior = target_agg.merge(metric_df, on="target_id", how="left")
module_behavior = module_agg.merge(metric_df, on="target_id", how="left")

target_behavior.to_csv(DATA / "stageB1e_target_effective_residual_with_behavior_final205.csv", index=False)
module_behavior.to_csv(DATA / "stageB1e_module_effective_residual_with_behavior_final205.csv", index=False)

corr_rows = []

metric_cols = [m for m in metric_map if m in target_behavior.columns]
for feat in ["rel_frob", "mean_module_rel_frob", "median_module_rel_frob", "mean_cosine", "n_factorized", "n_linear"]:
    for met in metric_cols:
        x = pd.to_numeric(target_behavior[feat], errors="coerce")
        y = pd.to_numeric(target_behavior[met], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() >= 10:
            corr_rows.append({
                "scope": "target_effective_residual",
                "feature": feat,
                "metric": met,
                "n": int(mask.sum()),
                "pearson": float(x[mask].corr(y[mask], method="pearson")),
                "spearman": float(x[mask].corr(y[mask], method="spearman")),
            })

for module, g in module_behavior.groupby("module_type"):
    for feat in ["rel_frob", "mean_module_rel_frob", "median_module_rel_frob", "mean_cosine"]:
        for met in metric_cols:
            x = pd.to_numeric(g[feat], errors="coerce")
            y = pd.to_numeric(g[met], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() >= 10:
                corr_rows.append({
                    "scope": f"module:{module}",
                    "feature": feat,
                    "metric": met,
                    "n": int(mask.sum()),
                    "pearson": float(x[mask].corr(y[mask], method="pearson")),
                    "spearman": float(x[mask].corr(y[mask], method="spearman")),
                })

corr = pd.DataFrame(corr_rows)
if len(corr):
    corr["abs_spearman"] = corr["spearman"].abs()
    corr = corr.sort_values(["abs_spearman", "n"], ascending=[False, False])
corr.to_csv(DATA / "stageB1e_effective_residual_behavior_correlations_final205.csv", index=False)

# Coverage accounting.
targets_all = pd.read_csv(B1C / "data/stageB1c_targets.csv")
coverage = targets_all[["target_id", "model_key", "method_key", "keep_ratio", "residual_mode"]].copy()
coverage["has_effective_residual"] = coverage["target_id"].isin(set(target_agg["target_id"]))
coverage.to_csv(DATA / "stageB1e_coverage_final205.csv", index=False)

# Keep ModeGPT shape profile from B1c.
shape_summary_path = B1C / "data/stageB1c_target_shape_summary.csv"
if shape_summary_path.exists():
    shape = pd.read_csv(shape_summary_path)
    shape.to_csv(DATA / "stageB1e_target_shape_summary_from_B1c.csv", index=False)

report = []
report.append("=== Stage B1e Final Effective-Weight Residual Report ===")
report.append(f"total_configs_in_zoo: {len(targets_all)}")
report.append(f"configs_with_effective_residual: {target_agg['target_id'].nunique()}")
report.append(f"effective_residual_rows: {len(all_res)}")
report.append(f"B1c_rows: {len(b1c)}")
report.append(f"B1d_patch_rows: {len(b1d)}")
report.append("")
report.append("=== Coverage by method ===")
report.append(coverage.groupby("method_key")["has_effective_residual"].agg(["count", "sum", "mean"]).to_string())
report.append("")
report.append("=== Method effective residual summary ===")
report.append(method_agg.to_string(index=False))
report.append("")
report.append("=== Top correlations n>=10 ===")
report.append(corr.head(120).to_string(index=False) if len(corr) else "NO CORR")
report.append("")
report.append("=== Excluded / no effective residual ===")
report.append(coverage[coverage["has_effective_residual"] == False].groupby(["method_key", "residual_mode"]).size().reset_index(name="n").to_string(index=False))

(LOGS / "STAGEB1E_EFFECTIVE_WEIGHT_FINAL_REPORT.txt").write_text("\n".join(report))

manifest = []
for p in sorted(PKG.rglob("*")):
    if p.is_file() and p.name != "MANIFEST.csv":
        manifest.append({
            "path": str(p.relative_to(PKG)),
            "size_bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        })
pd.DataFrame(manifest).to_csv(PKG / "MANIFEST.csv", index=False)

print("\n".join(report))
