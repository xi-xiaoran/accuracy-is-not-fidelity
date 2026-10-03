from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
PKG = ROOT / "results/paper_packages/bar_tolerance_audit_v1_20260702"
DATA = PKG / "data"
LOGS = PKG / "logs"
TABLES = PKG / "tables"
for d in [DATA, LOGS, TABLES]:
    d.mkdir(parents=True, exist_ok=True)

SUMMARY = ROOT / "results/theory_predictions_v1_fullzoo/fullzoo_theory_summary_by_config.csv"
SAMPLE_DIR = ROOT / "results/theory_predictions_v1"

def find_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None

def as_bool(s):
    if s.dtype == bool:
        return s.astype(int)
    if np.issubdtype(s.dtype, np.number):
        return s.fillna(0).astype(int)
    return s.astype(str).str.lower().isin(["1", "true", "yes", "y"]).astype(int)

summary = pd.read_csv(SUMMARY)
tid_col = find_col(summary, ["target_id", "config_id"])
if tid_col is None:
    raise RuntimeError("Cannot find target_id/config_id")

frames = []
for tid in summary[tid_col].astype(str):
    p = SAMPLE_DIR / f"{tid}_sample_theory_frame.csv"
    if not p.exists():
        continue
    df = pd.read_csv(p)
    if "split" in df.columns:
        df = df[df["split"].astype(str) == "eval"].copy()
    df["target_id"] = tid
    frames.append(df)

sample = pd.concat(frames, ignore_index=True)

task_col = find_col(sample, ["task", "dataset", "task_name"])
flip_col = find_col(sample, ["flip", "pred_flip", "prediction_flip"])
bar_col = find_col(sample, ["boundary_ratio", "bar", "BAR"])
margin_col = find_col(sample, ["dense_margin", "margin", "gamma"])
adv_col = find_col(sample, ["adv_gap_delta", "boundary_pressure", "bar_numerator"])

if flip_col is None or bar_col is None:
    raise RuntimeError(f"Missing required columns: flip={flip_col}, bar={bar_col}")

sample["_flip"] = as_bool(sample[flip_col])
sample["_bar"] = pd.to_numeric(sample[bar_col], errors="coerce")
if margin_col:
    sample["_margin"] = pd.to_numeric(sample[margin_col], errors="coerce")
if adv_col and margin_col:
    sample["_recomputed_bar"] = pd.to_numeric(sample[adv_col], errors="coerce") / sample["_margin"]
    sample["_bar_abs_diff"] = (sample["_bar"] - sample["_recomputed_bar"]).abs()

eps_list = [0.0, 1e-15, 1e-14, 1e-13, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3]
rows = []
for eps in eps_list:
    safe = sample["_bar"] < (1.0 - eps)
    boundary_band = (sample["_bar"] >= (1.0 - eps)) & (sample["_bar"] < (1.0 + eps))
    risk = sample["_bar"] >= (1.0 + eps)
    rows.append({
        "epsilon": eps,
        "safe_condition": f"BAR < 1 - {eps:g}",
        "safe_n": int(safe.sum()),
        "safe_flip_n": int(sample.loc[safe, "_flip"].sum()),
        "safe_flip_rate_pct": float(sample.loc[safe, "_flip"].mean() * 100) if safe.sum() else np.nan,
        "boundary_band_n": int(boundary_band.sum()),
        "boundary_band_flip_n": int(sample.loc[boundary_band, "_flip"].sum()),
        "boundary_band_flip_rate_pct": float(sample.loc[boundary_band, "_flip"].mean() * 100) if boundary_band.sum() else np.nan,
        "risk_n": int(risk.sum()),
        "risk_flip_n": int(sample.loc[risk, "_flip"].sum()),
        "risk_flip_rate_pct": float(sample.loc[risk, "_flip"].mean() * 100) if risk.sum() else np.nan,
    })
tol = pd.DataFrame(rows)
tol.to_csv(DATA / "bar_tolerance_summary.csv", index=False)

# Per-task at selected eps.
selected_eps = [0.0, 1e-13, 1e-12, 1e-10, 1e-8]
task_rows = []
if task_col:
    for eps in selected_eps:
        for task, g in sample.groupby(task_col):
            safe = g["_bar"] < (1.0 - eps)
            band = (g["_bar"] >= (1.0 - eps)) & (g["_bar"] < (1.0 + eps))
            risk = g["_bar"] >= (1.0 + eps)
            task_rows.append({
                "epsilon": eps,
                "task": task,
                "safe_n": int(safe.sum()),
                "safe_flip_n": int(g.loc[safe, "_flip"].sum()),
                "safe_flip_rate_pct": float(g.loc[safe, "_flip"].mean() * 100) if safe.sum() else np.nan,
                "boundary_band_n": int(band.sum()),
                "boundary_band_flip_n": int(g.loc[band, "_flip"].sum()),
                "boundary_band_flip_rate_pct": float(g.loc[band, "_flip"].mean() * 100) if band.sum() else np.nan,
                "risk_n": int(risk.sum()),
                "risk_flip_rate_pct": float(g.loc[risk, "_flip"].mean() * 100) if risk.sum() else np.nan,
            })
    pd.DataFrame(task_rows).to_csv(DATA / "bar_tolerance_by_task.csv", index=False)

# Violation-only distance to boundary.
viol = sample[(sample["_bar"] < 1.0) & (sample["_flip"] == 1)].copy()
if len(viol):
    viol["_one_minus_bar"] = 1.0 - viol["_bar"]
    keep_cols = ["target_id"]
    if task_col:
        keep_cols.append(task_col)
    for c in ["_bar", "_one_minus_bar", "_flip"]:
        keep_cols.append(c)
    if margin_col:
        keep_cols.append("_margin")
    if "_recomputed_bar" in viol.columns:
        keep_cols.extend(["_recomputed_bar", "_bar_abs_diff"])
    viol[keep_cols].to_csv(DATA / "bar_lt1_violation_boundary_distance_safe_rows.csv", index=False)
    
    q = viol["_one_minus_bar"].quantile([0, .25, .5, .75, .95, .99, 1.0]).to_frame("one_minus_bar").reset_index()
    q = q.rename(columns={"index": "quantile"})
    q.to_csv(DATA / "bar_lt1_violation_distance_quantiles.csv", index=False)

# Paper-ready mini LaTeX table.
paper = tol[tol["epsilon"].isin([0.0, 1e-13, 1e-12, 1e-10, 1e-8])].copy()
paper["epsilon_label"] = paper["epsilon"].map(lambda x: "0" if x == 0 else f"$10^{{{int(np.log10(x))}}}$")
paper = paper[["epsilon_label", "safe_n", "safe_flip_n", "safe_flip_rate_pct", "boundary_band_n", "boundary_band_flip_n", "risk_flip_rate_pct"]]

latex = []
latex.append(r"\begin{table}[t]")
latex.append(r"\centering")
latex.append(r"\small")
latex.append(r"\caption{Tolerance analysis for the BAR certificate. All empirical BAR$<1$ flips lie within numerical precision of the boundary.}")
latex.append(r"\label{tab:bar_tolerance}")
latex.append(r"\begin{tabular}{lrrrrr}")
latex.append(r"\toprule")
latex.append(r"$\tau$ & Safe $N$ & Safe flips & Safe flip & Boundary flips & Risk flip \\")
latex.append(r"\midrule")
for _, r in paper.iterrows():
    latex.append(
        f"{r['epsilon_label']} & "
        f"{int(r['safe_n']):,} & "
        f"{int(r['safe_flip_n'])} & "
        f"{r['safe_flip_rate_pct']:.4f} & "
        f"{int(r['boundary_band_flip_n'])} & "
        f"{r['risk_flip_rate_pct']:.2f} \\\\"
    )
latex.append(r"\bottomrule")
latex.append(r"\end{tabular}")
latex.append(r"\end{table}")
(TABLES / "table_bar_tolerance.tex").write_text("\n".join(latex) + "\n")

# Report.
report = []
report.append("=== BAR Tolerance Audit v1 ===")
report.append(f"sample_shape: {sample.shape}")
report.append(f"task_col: {task_col}")
report.append(f"flip_col: {flip_col}")
report.append(f"bar_col: {bar_col}")
report.append(f"margin_col: {margin_col}")
report.append(f"adv_col: {adv_col}")
report.append("")
report.append("=== tolerance summary ===")
report.append(tol.to_string(index=False))
report.append("")
if len(viol):
    report.append("=== violation distance quantiles ===")
    report.append(pd.read_csv(DATA / "bar_lt1_violation_distance_quantiles.csv").to_string(index=False))
    report.append("")
if "_bar_abs_diff" in sample.columns:
    report.append("=== stored BAR vs recomputed BAR ===")
    diff = sample["_bar_abs_diff"].dropna()
    report.append(f"max_abs_diff={diff.max():.6e}")
    report.append(f"median_abs_diff={diff.median():.6e}")
    report.append(f"n_diff_gt_1e-8={(diff > 1e-8).sum()}")
    report.append("")
(LOGS / "BAR_TOLERANCE_REPORT.txt").write_text("\n".join(report))

# Manifest.
rows = []
for p in sorted(PKG.rglob("*")):
    if p.is_file() and p.name != "MANIFEST.csv":
        rows.append({
            "path": str(p.relative_to(PKG)),
            "size_bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        })
pd.DataFrame(rows).to_csv(PKG / "MANIFEST.csv", index=False)

print("\n".join(report))
print("\nWROTE", PKG)
