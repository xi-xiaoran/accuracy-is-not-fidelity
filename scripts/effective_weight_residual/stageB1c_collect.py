from pathlib import Path
import json
import hashlib
import pandas as pd
import numpy as np

ROOT = Path(".").resolve()
PKG = ROOT / "results/paper_packages/stageB1c_effective_weight_residual_fullzoo_v1_20260702"
DATA = PKG / "data"
LOGS = PKG / "logs"
OUT_DIR = DATA / "per_target"
SUM_DIR = DATA / "per_target_summary"
SHAPE_DIR = DATA / "per_target_shape"
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
        "n_factorized": int(g["materialization"].isin(["ALinear@BLinear", "u_proj@v_proj"]).sum()),
        "n_linear": int((g["materialization"] == "Linear").sum()),
    })

targets = pd.read_csv(DATA / "stageB1c_targets.csv")
summary = pd.read_csv(SUMMARY)

status_rows = []
for p in sorted(SUM_DIR.glob("*_summary.json")):
    try:
        status_rows.append(json.loads(p.read_text()))
    except Exception as e:
        status_rows.append({"summary_path": str(p), "ok": False, "error": repr(e)})
status_df = pd.DataFrame(status_rows)
status_df.to_csv(DATA / "stageB1c_target_status.csv", index=False)

res_frames = []
for p in sorted(OUT_DIR.glob("*_effective_residual.csv")):
    try:
        df = pd.read_csv(p)
        if len(df):
            res_frames.append(df)
    except Exception:
        pass
res_all = pd.concat(res_frames, ignore_index=True) if res_frames else pd.DataFrame()
res_all.to_csv(DATA / "stageB1c_effective_residual_all.csv", index=False)

shape_frames = []
for p in sorted(SHAPE_DIR.glob("*_shape_profile.csv")):
    try:
        df = pd.read_csv(p)
        if len(df):
            shape_frames.append(df)
    except Exception:
        pass
shape_all = pd.concat(shape_frames, ignore_index=True) if shape_frames else pd.DataFrame()
shape_all.to_csv(DATA / "stageB1c_shape_profile_all.csv", index=False)

ok = res_all[res_all["ok"] == True].copy() if len(res_all) else pd.DataFrame()

if len(ok):
    target_agg = ok.groupby(["target_id", "model_key", "method_key", "keep_ratio"]).apply(agg_rel).reset_index()
    module_agg = ok.groupby(["target_id", "model_key", "method_key", "keep_ratio", "module_type"]).apply(agg_rel).reset_index()
    method_agg = ok.groupby(["method_key"]).apply(agg_rel).reset_index()
    model_method_agg = ok.groupby(["model_key", "method_key"]).apply(agg_rel).reset_index()
else:
    target_agg = pd.DataFrame()
    module_agg = pd.DataFrame()
    method_agg = pd.DataFrame()
    model_method_agg = pd.DataFrame()

target_agg.to_csv(DATA / "stageB1c_target_effective_residual_summary.csv", index=False)
module_agg.to_csv(DATA / "stageB1c_target_module_effective_residual_summary.csv", index=False)
method_agg.to_csv(DATA / "stageB1c_method_effective_residual_summary.csv", index=False)
model_method_agg.to_csv(DATA / "stageB1c_model_method_effective_residual_summary.csv", index=False)

# ModeGPT / structural shape profile.
if len(shape_all):
    shape_num = shape_all.copy()
    for c in ["param_retention", "out_dim_retention", "in_dim_retention"]:
        if c in shape_num.columns:
            shape_num[c] = pd.to_numeric(shape_num[c], errors="coerce")
    shape_target = shape_num.groupby(["target_id", "model_key", "method_key", "keep_ratio"]).agg(
        n_shape_modules=("module_name", "count"),
        mean_param_retention=("param_retention", "mean"),
        median_param_retention=("param_retention", "median"),
        mean_out_dim_retention=("out_dim_retention", "mean"),
        mean_in_dim_retention=("in_dim_retention", "mean"),
    ).reset_index()
else:
    shape_target = pd.DataFrame()

shape_target.to_csv(DATA / "stageB1c_target_shape_summary.csv", index=False)

# Merge with behavior.
metric_map = {
    "acc_delta": ["acc_delta", "acc_delta_pct", "delta_acc"],
    "pred_flip": ["pred_flip", "pred_flip_pct", "prediction_flip"],
    "hidden_churn": ["hidden_churn", "hidden_churn_pct"],
    "prob_js": ["prob_js", "prob_js_pct"],
    "bar_ge1_flip": ["bar_ge1_flip", "bar_ge1_flip_pct"],
    "compressed_acc": ["compressed_acc", "compressed_acc_pct"],
}
cols = {"target_id": find_col(summary, ["target_id"])}
for k, names in metric_map.items():
    cols[k] = find_col(summary, names)

merge_cols = ["target_id"] + [c for k, c in cols.items() if k != "target_id" and c is not None]
metric_df = summary[merge_cols].copy()
metric_df = metric_df.rename(columns={cols[k]: k for k in metric_map if cols.get(k) is not None})

target_behavior = target_agg.merge(metric_df, on="target_id", how="left") if len(target_agg) else pd.DataFrame()
target_behavior.to_csv(DATA / "stageB1c_target_effective_residual_with_behavior.csv", index=False)

shape_behavior = shape_target.merge(metric_df, on="target_id", how="left") if len(shape_target) else pd.DataFrame()
shape_behavior.to_csv(DATA / "stageB1c_target_shape_with_behavior.csv", index=False)

corr_rows = []
if len(target_behavior):
    feature_cols = ["rel_frob", "mean_module_rel_frob", "median_module_rel_frob", "mean_cosine", "n_factorized", "n_linear"]
    metric_cols = [m for m in metric_map if m in target_behavior.columns]
    for feat in feature_cols:
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

if len(module_agg):
    module_behavior = module_agg.merge(metric_df, on="target_id", how="left")
    module_behavior.to_csv(DATA / "stageB1c_module_effective_residual_with_behavior.csv", index=False)
    for module, g in module_behavior.groupby("module_type"):
        for feat in ["rel_frob", "mean_module_rel_frob", "mean_cosine"]:
            for met in [m for m in metric_map if m in g.columns]:
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

if len(shape_behavior):
    metric_cols = [m for m in metric_map if m in shape_behavior.columns]
    for feat in ["mean_param_retention", "median_param_retention", "mean_out_dim_retention", "mean_in_dim_retention"]:
        if feat not in shape_behavior.columns:
            continue
        for met in metric_cols:
            x = pd.to_numeric(shape_behavior[feat], errors="coerce")
            y = pd.to_numeric(shape_behavior[met], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() >= 10:
                corr_rows.append({
                    "scope": "target_shape_profile",
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
corr.to_csv(DATA / "stageB1c_effective_residual_behavior_correlations.csv", index=False)

report = []
report.append("=== Stage B1c Effective-Weight Residual FullZoo Report ===")
report.append(f"targets_total: {len(targets)}")
report.append(f"status_jsons: {len(status_df)}")
report.append(f"effective_residual_rows: {len(res_all)}")
report.append(f"effective_residual_ok_rows: {len(ok)}")
report.append(f"shape_profile_rows: {len(shape_all)}")
report.append("")
if len(status_df):
    report.append("=== Target OK ===")
    report.append(status_df["ok"].value_counts(dropna=False).to_string())
    report.append("")
    report.append("=== By method OK ===")
    report.append(status_df.groupby("method_key")["ok"].agg(["count", "sum", "mean"]).to_string())
    report.append("")
    report.append("=== Residual computed by method ===")
    show_cols = [c for c in ["method_key", "residual_mode", "n_residual_ok", "n_shape_rows", "global_rel_frob", "n_factorized_modules", "n_linear_modules", "n_shape_mismatch"] if c in status_df.columns]
    report.append(status_df.groupby(["method_key", "residual_mode"])[["n_residual_ok", "n_shape_rows", "n_factorized_modules", "n_linear_modules", "n_shape_mismatch"]].sum().to_string())
if len(method_agg):
    report.append("")
    report.append("=== Method effective residual summary ===")
    report.append(method_agg.to_string(index=False))
if len(model_method_agg):
    report.append("")
    report.append("=== Model-method effective residual summary sample ===")
    report.append(model_method_agg.head(100).to_string(index=False))
if len(shape_target):
    report.append("")
    report.append("=== Shape profile summary by method ===")
    report.append(shape_target.groupby("method_key")[["n_shape_modules", "mean_param_retention", "mean_out_dim_retention", "mean_in_dim_retention"]].mean().to_string())
if len(corr):
    report.append("")
    report.append("=== Top correlations n>=10 ===")
    report.append(corr.head(100).to_string(index=False))

(LOGS / "STAGEB1C_EFFECTIVE_WEIGHT_RESIDUAL_REPORT.txt").write_text("\n".join(report))

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
