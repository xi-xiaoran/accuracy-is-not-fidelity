from pathlib import Path
import argparse
import re
import json
import math
import hashlib
import warnings
import numpy as np
import pandas as pd

PARSER = argparse.ArgumentParser(
    description="Regenerate manuscript tables from user-supplied frozen analysis inputs."
)
PARSER.add_argument(
    "--project-root",
    type=Path,
    default=Path("."),
    help="Analysis workspace containing the expected results/ input tree (default: current directory).",
)
PARSER.add_argument(
    "--output-dir",
    type=Path,
    default=Path("generated_tables"),
    help="Directory for generated tables, copied inputs, and logs.",
)
ARGS = PARSER.parse_args()

ROOT = ARGS.project_root.expanduser().resolve()
PKG = ARGS.output_dir.expanduser().resolve()
DATA = PKG / "data"
TABLES = PKG / "tables"
LOGS = PKG / "logs"
for d in [DATA, TABLES, LOGS]:
    d.mkdir(parents=True, exist_ok=True)

REPORT = []
WARNINGS = []

def log(s=""):
    REPORT.append(str(s))
    print(s)

def warn(s):
    WARNINGS.append(str(s))
    print("[WARN]", s)

def read_csv_first(paths, required=True, desc=""):
    for p in paths:
        p = ROOT / p if not Path(p).is_absolute() else Path(p)
        if p.exists():
            log(f"[OK] {desc or p.name}: {p}")
            return pd.read_csv(p), p
    msg = f"Missing {desc}: tried {[str(p) for p in paths]}"
    if required:
        raise FileNotFoundError(msg)
    warn(msg)
    return None, None

def find_files(pattern):
    return sorted(ROOT.glob(pattern))

def find_one_file(names_or_patterns, required=False, desc=""):
    for pat in names_or_patterns:
        hits = sorted(ROOT.glob(pat))
        if hits:
            log(f"[OK] found {desc or pat}: {hits[0]}")
            return hits[0]
    if required:
        raise FileNotFoundError(f"Missing {desc}: {names_or_patterns}")
    warn(f"Missing optional {desc}: {names_or_patterns}")
    return None

def pick_col(df, candidates, required=False, desc=""):
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    # fuzzy: candidate tokens all included
    for cand in candidates:
        toks = [t for t in re.split(r"[^a-zA-Z0-9]+", cand.lower()) if t]
        for c in cols:
            cl = c.lower()
            if all(t in cl for t in toks):
                return c
    if required:
        raise KeyError(f"Missing column {desc or candidates}. Available: {cols}")
    return None

def pct(x, nd=2):
    if pd.isna(x):
        return "--"
    return f"{float(x):.{nd}f}"

def num(x, nd=3):
    if pd.isna(x):
        return "--"
    return f"{float(x):.{nd}f}"

def intfmt(x):
    if pd.isna(x):
        return "--"
    return f"{int(round(float(x))):,}"

def latex_escape(s):
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)

def df_to_latex_tabular(df, caption, label, align=None, table_star=False, small=True, resize=False, note=None):
    env = "table*" if table_star else "table"
    if align is None:
        align = "l" * len(df.columns)
    lines = []
    lines.append(rf"\begin{{{env}}}[t]")
    lines.append(r"\centering")
    if small:
        lines.append(r"\small")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    if resize:
        lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{align}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join([latex_escape(c) for c in df.columns]) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        vals = [latex_escape(row[c]) for c in df.columns]
        lines.append(" & ".join(vals) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if resize:
        lines.append(r"}")
    if note:
        lines.append(rf"\vspace{{1mm}}\footnotesize{{{note}}}")
    lines.append(rf"\end{{{env}}}")
    return "\n".join(lines) + "\n"

def save_table(df, name, caption, label, align=None, table_star=False, resize=False, note=None):
    csv_path = DATA / f"{name}.csv"
    tex_path = TABLES / f"{name}.tex"
    df.to_csv(csv_path, index=False)
    tex_path.write_text(df_to_latex_tabular(df, caption, label, align=align, table_star=table_star, resize=resize, note=note))
    log(f"[WRITE] {csv_path}")
    log(f"[WRITE] {tex_path}")

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def rank_auc(y, score):
    y = pd.Series(y).astype(float)
    score = pd.Series(score).astype(float)
    m = y.notna() & score.notna()
    y = y[m].astype(int)
    score = score[m]
    n_pos = int(y.sum())
    n_neg = int((1-y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = score.rank(method="average")
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def top_decile_rate(y, score):
    y = pd.Series(y).astype(float)
    score = pd.Series(score).astype(float)
    m = y.notna() & score.notna()
    y = y[m]
    score = score[m]
    if len(y) == 0:
        return np.nan, 0
    thr = score.quantile(0.90)
    top = score >= thr
    if top.sum() == 0:
        return np.nan, 0
    return float(y[top].mean() * 100), int(top.sum())

# ---------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------
summary_df, summary_path = read_csv_first([
    "results/theory_predictions_v1_fullzoo/fullzoo_theory_summary_by_config.csv",
    "results/paper_packages/theory_validation_fullzoo_nanfix_20260702_v2/summary/theory_predictions_v1_fullzoo/fullzoo_theory_summary_by_config.csv",
    "results/paper_packages/theory_validation_fullzoo_nanfix_20260702_v2/summary/fullzoo_theory_summary_by_config.csv",
], desc="fullzoo config summary")

controlled_df, controlled_path = read_csv_first([
    "results/theory_predictions_v1_fullzoo/controlled_summaries/controlled_theory_summary.csv",
    "results/paper_packages/theory_validation_fullzoo_nanfix_20260702_v2/summary/theory_predictions_v1_fullzoo/controlled_summaries/controlled_theory_summary.csv",
    "results/paper_packages/theory_validation_fullzoo_nanfix_20260702_v2/summary/controlled_summaries/controlled_theory_summary.csv",
], required=False, desc="controlled summary")

# Required columns from config summary
col_target = pick_col(summary_df, ["target_id"], required=True)
col_keep = pick_col(summary_df, ["keep_ratio"], required=True)
col_dense = pick_col(summary_df, ["dense_acc_pct", "dense_acc"], required=True)
col_comp = pick_col(summary_df, ["compressed_acc_pct", "compressed_acc"], required=True)
col_delta = pick_col(summary_df, ["acc_delta_pct", "acc_delta", "delta_acc_pct"], required=True)
col_flip = pick_col(summary_df, ["pred_flip_pct", "pred_flip", "prediction_flip_pct"], required=True)
col_hidden = pick_col(summary_df, ["hidden_churn_pct", "hidden_churn"], required=True)
col_probjs = pick_col(summary_df, ["prob_js_pct", "prob_js"], required=False)

# ---------------------------------------------------------------------
# Table 1: main controlled subsets
# ---------------------------------------------------------------------
def subset_row(name, mask):
    d = summary_df.loc[mask].copy()
    return {
        "Subset": name,
        "#Cfg": intfmt(len(d)),
        "Dense Acc.": pct(d[col_dense].mean(), 2),
        "Compressed Acc.": pct(d[col_comp].mean(), 2),
        "$\\Delta$Acc.": pct(d[col_delta].mean(), 2),
        "PredFlip": pct(d[col_flip].mean(), 2),
        "HiddenChurn": pct(d[col_hidden].mean(), 2),
    }

keep_series = summary_df[col_keep].astype(str)
keep_num = pd.to_numeric(summary_df[col_keep], errors="coerce")
table1_rows = [
    subset_row("All", pd.Series(True, index=summary_df.index)),
    subset_row("Keep=0.8", (keep_series.str.contains("0p8|0.8", regex=True)) | (keep_num.round(3) == 0.8)),
    subset_row(r"$|\Delta\mathrm{Acc.}|\le10$", summary_df[col_delta].abs() <= 10),
    subset_row(r"$|\Delta\mathrm{Acc.}|\le5$", summary_df[col_delta].abs() <= 5),
    subset_row(r"Comp. Acc.$\ge50$", summary_df[col_comp] >= 50),
    subset_row(r"Comp. Acc.$\ge60$", summary_df[col_comp] >= 60),
]
table1 = pd.DataFrame(table1_rows)
save_table(
    table1,
    "table1_main_controlled_subsets",
    "Accuracy-controlled decision churn. Metrics are averaged over compressed configurations.",
    "tab:main_controlled_subsets",
    align="lrrrrrr",
    table_star=False,
)

# ---------------------------------------------------------------------
# Appendix A2: bootstrap CI over config-level means
# ---------------------------------------------------------------------
rng = np.random.default_rng(20260702)
def bootstrap_ci(d, col, n_boot=20000):
    vals = pd.to_numeric(d[col], errors="coerce").dropna().to_numpy()
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))

def ci_fmt(mean, lo, hi):
    return f"{mean:.2f} [{lo:.2f},{hi:.2f}]" if not pd.isna(mean) else "--"

def bootstrap_subset_row(name, mask):
    d = summary_df.loc[mask].copy()
    md, ld, hd = bootstrap_ci(d, col_delta)
    mf, lf, hf = bootstrap_ci(d, col_flip)
    mh, lh, hh = bootstrap_ci(d, col_hidden)
    return {
        "Subset": name,
        "#Cfg": intfmt(len(d)),
        "$\\Delta$Acc.": ci_fmt(md, ld, hd),
        "PredFlip": ci_fmt(mf, lf, hf),
        "HiddenChurn": ci_fmt(mh, lh, hh),
    }

tableA2 = pd.DataFrame([
    bootstrap_subset_row("All", pd.Series(True, index=summary_df.index)),
    bootstrap_subset_row("Keep=0.8", (keep_series.str.contains("0p8|0.8", regex=True)) | (keep_num.round(3) == 0.8)),
    bootstrap_subset_row(r"$|\Delta\mathrm{Acc.}|\le10$", summary_df[col_delta].abs() <= 10),
    bootstrap_subset_row(r"$|\Delta\mathrm{Acc.}|\le5$", summary_df[col_delta].abs() <= 5),
    bootstrap_subset_row(r"Comp. Acc.$\ge50$", summary_df[col_comp] >= 50),
    bootstrap_subset_row(r"Comp. Acc.$\ge60$", summary_df[col_comp] >= 60),
])
save_table(
    tableA2,
    "appendix_tableA2_bootstrap_ci",
    "Bootstrap confidence intervals for accuracy-controlled churn summaries. We report mean and 95\\% bootstrap intervals over compressed configurations.",
    "tab:appendix_bootstrap_ci",
    align="lrrrr",
    table_star=False,
    resize=True,
)

# ---------------------------------------------------------------------
# Load sample-level raw theory frames for BAR / task breakdown
# ---------------------------------------------------------------------
raw_dirs = [
    ROOT / "results/paper_packages/theory_validation_fullzoo_nanfix_20260702_v2/raw/theory_predictions_v1",
    ROOT / "results/theory_predictions_v1",
]
raw_files = []
for d in raw_dirs:
    if d.exists():
        raw_files = sorted(d.glob("*_sample_theory_frame.csv"))
        if raw_files:
            break

sample_df = None
if not raw_files:
    warn("No sample-level theory frames found; Table2/A1/A3 will be partial.")
else:
    log(f"[OK] sample-level frames: {len(raw_files)} files under {raw_files[0].parent}")
    header_cols = pd.read_csv(raw_files[0], nrows=0).columns.tolist()
    wanted = [
        "task", "flip", "dense_margin", "boundary_ratio", "adv_gap_delta",
        "prob_js", "score_l2_delta", "logit_l2_delta", "delta_l2", "score_l2",
        "dense_pred", "compressed_pred", "gold", "dense_correct", "compressed_correct",
        "n_choices", "num_choices"
    ]
    usecols = [c for c in wanted if c in header_cols]
    frames = []
    for p in raw_files:
        try:
            h = pd.read_csv(p, nrows=0).columns.tolist()
            uc = [c for c in wanted if c in h]
            df = pd.read_csv(p, usecols=uc)
            df["target_id"] = p.name.replace("_sample_theory_frame.csv", "")
            frames.append(df)
        except Exception as e:
            warn(f"failed reading {p}: {repr(e)}")
    sample_df = pd.concat(frames, ignore_index=True)
    log(f"[OK] loaded sample rows: {len(sample_df):,}")

# ---------------------------------------------------------------------
# Table 2: BAR diagnostic / baseline comparison
# ---------------------------------------------------------------------
table2_rows = []
if sample_df is not None and "flip" in sample_df.columns:
    y = pd.to_numeric(sample_df["flip"], errors="coerce").fillna(0).astype(int)
    metric_specs = []
    if "boundary_ratio" in sample_df.columns:
        metric_specs.append(("BAR", "boundary_ratio", sample_df["boundary_ratio"], "Margin-normalized boundary pressure"))
    if "adv_gap_delta" in sample_df.columns:
        metric_specs.append(("Boundary pressure $B$", "adv_gap_delta", sample_df["adv_gap_delta"], "Raw adversarial gap perturbation"))
    if "dense_margin" in sample_df.columns:
        metric_specs.append(("Dense margin only", "-dense_margin", -pd.to_numeric(sample_df["dense_margin"], errors="coerce"), "Lower dense margin means higher risk"))
    if "prob_js" in sample_df.columns:
        metric_specs.append(("Prob-JS", "prob_js", sample_df["prob_js"], "Distributional shift baseline"))
    l2_col = pick_col(sample_df, ["score_l2_delta", "logit_l2_delta", "delta_l2", "score_l2"], required=False)
    if l2_col:
        metric_specs.append(("Score $\\ell_2$ drift", l2_col, sample_df[l2_col], "Raw score-space magnitude baseline"))

    for name, col, score, interp in metric_specs:
        auc = rank_auc(y, score)
        top_rate, top_n = top_decile_rate(y, score)
        table2_rows.append({
            "Metric": name,
            "AUC": num(auc, 3),
            "Top-decile Flip": pct(top_rate, 2),
            "Top-decile #Ex.": intfmt(top_n),
            "Interpretation": interp,
        })

table2 = pd.DataFrame(table2_rows)
if len(table2) == 0:
    table2 = pd.DataFrame([{"Metric": "MISSING", "AUC": "--", "Top-decile Flip": "--", "Top-decile #Ex.": "--", "Interpretation": "No sample-level flip/risk data found."}])
save_table(
    table2,
    "table2_bar_diagnostics",
    "BAR and baseline diagnostics for flip risk. Higher AUC and top-decile flip rate indicate better concentration of boundary-risk examples.",
    "tab:bar_diagnostics",
    align="lrrrl",
    table_star=True,
    resize=True,
)

# ---------------------------------------------------------------------
# Appendix A3: BAR exactness / tolerance audit from sample frames
# ---------------------------------------------------------------------
if sample_df is not None and {"boundary_ratio", "flip"}.issubset(sample_df.columns):
    bar = pd.to_numeric(sample_df["boundary_ratio"], errors="coerce")
    flip = pd.to_numeric(sample_df["flip"], errors="coerce").fillna(0).astype(int)
    tableA3_rows = []
    for eps in [0.0, 1e-13, 1e-12, 1e-10, 1e-8]:
        safe = bar < (1.0 - eps)
        boundary = (bar >= (1.0 - eps)) & (bar < 1.0)
        risk = bar >= 1.0
        tableA3_rows.append({
            "Condition": f"BAR < 1 - {eps:g}" if eps > 0 else "BAR < 1",
            "Safe #Ex.": intfmt(safe.sum()),
            "Safe Flip Rate": pct(flip[safe].mean() * 100 if safe.sum() else np.nan, 6 if eps <= 1e-12 else 4),
            "Boundary #Ex.": intfmt(boundary.sum()),
            "Boundary Flip Rate": pct(flip[boundary].mean() * 100 if boundary.sum() else np.nan, 4),
            "Risk #Ex.": intfmt(risk.sum()),
            "Risk Flip Rate": pct(flip[risk].mean() * 100 if risk.sum() else np.nan, 4),
        })
    tableA3 = pd.DataFrame(tableA3_rows)
else:
    tableA3 = pd.DataFrame([{"Condition": "MISSING", "Safe #Ex.": "--", "Safe Flip Rate": "--", "Boundary #Ex.": "--", "Boundary Flip Rate": "--", "Risk #Ex.": "--", "Risk Flip Rate": "--"}])

save_table(
    tableA3,
    "appendix_tableA3_bar_tolerance",
    "BAR tolerance audit. Strictly safe examples under BAR$<1-10^{-12}$ have zero observed flips.",
    "tab:appendix_bar_tolerance",
    align="lrrrrrr",
    table_star=True,
    resize=True,
)

# ---------------------------------------------------------------------
# Appendix A1: Per-task and binary/multi breakdown
# ---------------------------------------------------------------------
if sample_df is not None and "task" in sample_df.columns:
    df = sample_df.copy()
    if "dense_correct" in df.columns:
        df["_dense_correct"] = pd.to_numeric(df["dense_correct"], errors="coerce")
    elif {"dense_pred", "gold"}.issubset(df.columns):
        df["_dense_correct"] = (df["dense_pred"].astype(str) == df["gold"].astype(str)).astype(float)
    else:
        df["_dense_correct"] = np.nan

    if "compressed_correct" in df.columns:
        df["_compressed_correct"] = pd.to_numeric(df["compressed_correct"], errors="coerce")
    elif {"compressed_pred", "gold"}.issubset(df.columns):
        df["_compressed_correct"] = (df["compressed_pred"].astype(str) == df["gold"].astype(str)).astype(float)
    else:
        df["_compressed_correct"] = np.nan

    df["_flip"] = pd.to_numeric(df["flip"], errors="coerce") if "flip" in df.columns else np.nan

    def task_row(name, g):
        dense_acc = g["_dense_correct"].mean() * 100
        comp_acc = g["_compressed_correct"].mean() * 100
        return {
            "Task / Group": name,
            "#Rows": intfmt(len(g)),
            "Dense Acc.": pct(dense_acc, 2),
            "Compressed Acc.": pct(comp_acc, 2),
            "$\\Delta$Acc.": pct(comp_acc - dense_acc, 2),
            "PredFlip": pct(g["_flip"].mean() * 100, 2),
            "HiddenChurn": pct(g["_flip"].mean() * 100 - abs(comp_acc - dense_acc), 2),
        }

    rows = []
    for task, g in df.groupby("task"):
        rows.append(task_row(str(task), g))

    binary_tasks = {"boolq", "winogrande"}
    df["_task_lower"] = df["task"].astype(str).str.lower()
    if "n_choices" in df.columns or "num_choices" in df.columns:
        nc = pd.to_numeric(df[pick_col(df, ["n_choices", "num_choices"])], errors="coerce")
        binary_mask = nc <= 2
    else:
        binary_mask = df["_task_lower"].isin(binary_tasks)
    rows.append(task_row("binary", df[binary_mask]))
    rows.append(task_row("multi-choice", df[~binary_mask]))

    tableA1 = pd.DataFrame(rows)
else:
    tableA1 = pd.DataFrame([{"Task / Group": "MISSING", "#Rows": "--", "Dense Acc.": "--", "Compressed Acc.": "--", "$\\Delta$Acc.": "--", "PredFlip": "--", "HiddenChurn": "--"}])

save_table(
    tableA1,
    "appendix_tableA1_task_breakdown",
    "Per-task and binary/multi-choice breakdown of accuracy and decision churn over sample-level rows.",
    "tab:appendix_task_breakdown",
    align="lrrrrrr",
    table_star=True,
    resize=True,
)

# ---------------------------------------------------------------------
# Appendix A4: Global / group correction summary
# ---------------------------------------------------------------------
proj_eval_file = find_one_file([
    "results/theory_predictions_v1_fullzoo/fullzoo_projection_eval_long.csv",
    "results/paper_packages/theory_validation_fullzoo_nanfix_20260702_v2/summary/theory_predictions_v1_fullzoo/fullzoo_projection_eval_long.csv",
    "results/paper_packages/theory_validation_fullzoo_nanfix_20260702_v2/summary/fullzoo_projection_eval_long.csv",
    "results/paper_packages/theory_validation_fullzoo_nanfix_20260702_v2/**/fullzoo_projection_eval_long.csv",
], required=False, desc="projection eval long")

proj_energy_file = find_one_file([
    "results/theory_predictions_v1_fullzoo/fullzoo_projection_energy_long.csv",
    "results/paper_packages/theory_validation_fullzoo_nanfix_20260702_v2/summary/theory_predictions_v1_fullzoo/fullzoo_projection_energy_long.csv",
    "results/paper_packages/theory_validation_fullzoo_nanfix_20260702_v2/summary/fullzoo_projection_energy_long.csv",
    "results/paper_packages/theory_validation_fullzoo_nanfix_20260702_v2/**/fullzoo_projection_energy_long.csv",
], required=False, desc="projection energy long")

tableA4_rows = []
tableA4_rows.append({
    "Correction": "Compressed",
    "Prob-JS": pct(summary_df[col_probjs].mean(), 3) if col_probjs else "--",
    "HiddenChurn": pct(summary_df[col_hidden].mean(), 2),
    "PredFlip": pct(summary_df[col_flip].mean(), 2),
    "Energy Explained": "--",
})

if proj_eval_file and proj_eval_file.exists():
    pe = pd.read_csv(proj_eval_file)
    # Guess grouping column
    group_col = None
    for c in pe.columns:
        cl = c.lower()
        if any(k in cl for k in ["correction", "projection", "strategy", "mode", "repair", "method"]):
            if pe[c].nunique() < max(50, len(pe)//5):
                group_col = c
                break
    if group_col is None:
        obj_cols = [c for c in pe.columns if pe[c].dtype == "object"]
        group_col = obj_cols[0] if obj_cols else None
    prob_col = pick_col(pe, ["prob_js_pct", "prob_js", "new_prob_js_pct", "corrected_prob_js_pct"], required=False)
    hidden_col = pick_col(pe, ["hidden_churn_pct", "hidden_churn", "new_hidden_churn_pct", "corrected_hidden_churn_pct"], required=False)
    flip_col_p = pick_col(pe, ["pred_flip_pct", "pred_flip", "new_pred_flip_pct", "corrected_pred_flip_pct"], required=False)
    if group_col:
        for name, g in pe.groupby(group_col):
            tableA4_rows.append({
                "Correction": str(name),
                "Prob-JS": pct(pd.to_numeric(g[prob_col], errors="coerce").mean(), 3) if prob_col else "--",
                "HiddenChurn": pct(pd.to_numeric(g[hidden_col], errors="coerce").mean(), 2) if hidden_col else "--",
                "PredFlip": pct(pd.to_numeric(g[flip_col_p], errors="coerce").mean(), 2) if flip_col_p else "--",
                "Energy Explained": "--",
            })
    else:
        warn(f"Could not infer projection eval grouping column in {proj_eval_file}")
else:
    # Fallback: scan summary columns for projection/correction metrics
    for c in summary_df.columns:
        if "hidden_churn" in c.lower() and ("proj" in c.lower() or "correction" in c.lower()):
            prefix = re.sub(r"hidden_churn.*", "", c, flags=re.I).strip("_")
            prob_candidates = [x for x in summary_df.columns if x.startswith(prefix) and "prob_js" in x.lower()]
            flip_candidates = [x for x in summary_df.columns if x.startswith(prefix) and "pred_flip" in x.lower()]
            tableA4_rows.append({
                "Correction": prefix,
                "Prob-JS": pct(pd.to_numeric(summary_df[prob_candidates[0]], errors="coerce").mean(), 3) if prob_candidates else "--",
                "HiddenChurn": pct(pd.to_numeric(summary_df[c], errors="coerce").mean(), 2),
                "PredFlip": pct(pd.to_numeric(summary_df[flip_candidates[0]], errors="coerce").mean(), 2) if flip_candidates else "--",
                "Energy Explained": "--",
            })

# Attach energy explained if possible by matching names loosely.
if proj_energy_file and proj_energy_file.exists():
    en = pd.read_csv(proj_energy_file)
    e_group = None
    for c in en.columns:
        cl = c.lower()
        if any(k in cl for k in ["correction", "projection", "strategy", "mode", "repair", "method"]):
            if en[c].nunique() < max(50, len(en)//5):
                e_group = c
                break
    e_col = None
    for c in en.columns:
        if "explained" in c.lower() or "energy" in c.lower():
            if pd.api.types.is_numeric_dtype(en[c]):
                e_col = c
                break
    if e_group and e_col:
        energy_map = en.groupby(e_group)[e_col].mean().to_dict()
        for r in tableA4_rows:
            key = r["Correction"]
            if key in energy_map:
                r["Energy Explained"] = pct(energy_map[key], 2)

tableA4 = pd.DataFrame(tableA4_rows).drop_duplicates()
save_table(
    tableA4,
    "appendix_tableA4_global_correction",
    "Global and group-level correction summary. Values are averaged over compressed configurations when available.",
    "tab:appendix_global_correction",
    align="lrrrr",
    table_star=True,
    resize=True,
)

# ---------------------------------------------------------------------
# Table 3 and Appendix A5: Effective residual and mechanism evidence
# ---------------------------------------------------------------------
eff_corr_df, eff_corr_path = read_csv_first([
    "results/paper_packages/effective_weight_residual_final205_20260702_v1/summary/stageB1e_effective_residual_behavior_correlations_final205.csv",
    "results/paper_packages/stageB1e_effective_weight_residual_final205_v1_20260702/data/stageB1e_effective_residual_behavior_correlations_final205.csv",
], desc="effective residual correlations")

eff_cov_df, eff_cov_path = read_csv_first([
    "results/paper_packages/effective_weight_residual_final205_20260702_v1/summary/stageB1e_coverage_final205.csv",
    "results/paper_packages/stageB1e_effective_weight_residual_final205_v1_20260702/data/stageB1e_coverage_final205.csv",
], desc="effective residual coverage")

def get_eff(scope, feature, metric):
    d = eff_corr_df.copy()
    for c in ["scope", "feature", "metric"]:
        if c not in d.columns:
            return np.nan
    m = (d["scope"].astype(str) == scope) & (d["feature"].astype(str) == feature) & (d["metric"].astype(str) == metric)
    if m.any():
        return float(d.loc[m, "spearman"].iloc[0])
    # fuzzy
    m = d["scope"].astype(str).str.contains(re.escape(scope), regex=True) & d["feature"].astype(str).str.contains(re.escape(feature), regex=True) & d["metric"].astype(str).str.contains(metric, regex=False)
    if m.any():
        return float(d.loc[m, "spearman"].iloc[0])
    return np.nan

# Use exact keys from the canonical mechanism correlation table.  This avoids
# the earlier fuzzy-token/max-absolute selection, which could select the
# answer-path maximum statistic for a row labeled as the answer-path mean.
mech_corr, mech_corr_path = read_csv_first([
    "results/paper_packages/mechanism_diagnostics_v1_coauthor_20260702/tables/table_config_correlations_clean.csv",
    "results/coauthor_complete_aggregate_data_20260714_v3/05_activation_and_choice_mechanism/mechanism_diagnostics/tables/table_config_correlations_clean.csv",
    "results/mechanism_tables/table_config_correlations_clean.csv",
], desc="canonical mechanism correlations")

MECH_REQUIRED_COLUMNS = {"metric_col", "target", "spearman"}
missing_mech_columns = MECH_REQUIRED_COLUMNS.difference(mech_corr.columns)
if missing_mech_columns:
    raise KeyError(
        "Canonical mechanism table is missing required columns: "
        + ", ".join(sorted(missing_mech_columns))
    )

def exact_mech_value(metric_col, target):
    matches = mech_corr.loc[
        (mech_corr["metric_col"].astype(str) == metric_col)
        & (mech_corr["target"].astype(str) == target)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one mechanism row for metric_col={metric_col!r}, "
            f"target={target!r}; found {len(matches)}"
        )
    return float(matches["spearman"].iloc[0])

rep_values = {
    ("Prompt cosine drift", "pred_flip"): exact_mech_value("mean_cos_dist", "pred_flip_pct"),
    ("Prompt cosine drift", "hidden_churn"): exact_mech_value("mean_cos_dist", "hidden_churn_pct"),
    ("Answer-path mean cosine drift", "pred_flip"): exact_mech_value("mean_choice_cos", "pred_flip_pct"),
    ("Answer-path mean cosine drift", "hidden_churn"): exact_mech_value("mean_choice_cos", "hidden_churn_pct"),
}
rep_sources = {key: str(mech_corr_path) for key in rep_values}

table3_rows = [
    {
        "Evidence": "Representation",
        "Signal": "Prompt cosine drift",
        "BAR-risk Flip": "--",
        "HiddenChurn": num(rep_values[("Prompt cosine drift", "hidden_churn")], 3),
        "PredFlip": num(rep_values[("Prompt cosine drift", "pred_flip")], 3),
    },
    {
        "Evidence": "Representation",
        "Signal": "Answer-path mean cosine drift",
        "BAR-risk Flip": "--",
        "HiddenChurn": num(rep_values[("Answer-path mean cosine drift", "hidden_churn")], 3),
        "PredFlip": num(rep_values[("Answer-path mean cosine drift", "pred_flip")], 3),
    },
    {
        "Evidence": "Effective weight",
        "Signal": "$o_{\\mathrm{proj}}$ mean relative Frobenius",
        "BAR-risk Flip": num(get_eff("module:o_proj", "mean_module_rel_frob", "bar_ge1_flip"), 3),
        "HiddenChurn": num(get_eff("module:o_proj", "mean_module_rel_frob", "hidden_churn"), 3),
        "PredFlip": num(get_eff("module:o_proj", "mean_module_rel_frob", "pred_flip"), 3),
    },
    {
        "Evidence": "Effective weight",
        "Signal": "$v_{\\mathrm{proj}}$ cosine similarity",
        "BAR-risk Flip": num(get_eff("module:v_proj", "mean_cosine", "bar_ge1_flip"), 3),
        "HiddenChurn": num(get_eff("module:v_proj", "mean_cosine", "hidden_churn"), 3),
        "PredFlip": num(get_eff("module:v_proj", "mean_cosine", "pred_flip"), 3),
    },
    {
        "Evidence": "Effective weight",
        "Signal": "Target-level mean cosine similarity",
        "BAR-risk Flip": num(get_eff("target_effective_residual", "mean_cosine", "bar_ge1_flip"), 3),
        "HiddenChurn": num(get_eff("target_effective_residual", "mean_cosine", "hidden_churn"), 3),
        "PredFlip": num(get_eff("target_effective_residual", "mean_cosine", "pred_flip"), 3),
    },
]
table3 = pd.DataFrame(table3_rows)
save_table(
    table3,
    "table3_mechanism_evidence",
    "Mechanistic evidence for residual-to-boundary churn. We report Spearman correlations across compressed configurations.",
    "tab:mechanism_evidence",
    align="llrrr",
    table_star=True,
    resize=True,
)

pd.DataFrame([
    {"signal": k[0], "metric": k[1], "spearman": v, "source": rep_sources.get(k)}
    for k, v in rep_values.items()
]).to_csv(DATA / "table3_representation_sources.csv", index=False)

# Appendix A5 coverage / materialization
rule_map = {
    "asvd": "ALinear $\\times$ BLinear",
    "asvd/default_alpha": "ALinear $\\times$ BLinear",
    "asvd/gqa_fix": "ALinear $\\times$ BLinear",
    "basis_sharing": "LowRank factors or reconstructed Linear",
    "dobisvd": "BLinear $\\times$ ALinear",
    "modegpt": "Shape/width pruning; excluded",
    "svdllm_v1/alpaca_recover": "u\\_proj $\\times$ v\\_proj",
    "svdllm_v1/update": "u\\_proj $\\times$ v\\_proj",
    "svdllm_v1/update_c4": "u\\_proj $\\times$ v\\_proj",
    "svdllm_v1/whitening_only": "u\\_proj $\\times$ v\\_proj",
    "svdllm_v1/whitening_only_structured": "u\\_proj $\\times$ v\\_proj",
    "svdllm_v1/whitening_then_update_u256": "u\\_proj $\\times$ v\\_proj",
    "svdllm_v2": "u\\_proj $\\times$ v\\_proj",
    "svdllm_v2/hetero": "u\\_proj $\\times$ v\\_proj",
}
if "has_effective_residual" in eff_cov_df.columns:
    cov = eff_cov_df.copy()
    cov["has_effective_residual"] = cov["has_effective_residual"].astype(bool)
    tableA5 = cov.groupby("method_key")["has_effective_residual"].agg(["count", "sum"]).reset_index()
    tableA5["Excluded"] = tableA5["count"] - tableA5["sum"]
    tableA5["Materialization / Note"] = tableA5["method_key"].map(rule_map).fillna("effective-weight extraction")
    tableA5 = tableA5.rename(columns={"method_key": "Method", "count": "#Cfg", "sum": "#Covered"})
    tableA5["#Cfg"] = tableA5["#Cfg"].map(intfmt)
    tableA5["#Covered"] = tableA5["#Covered"].map(intfmt)
    tableA5["Excluded"] = tableA5["Excluded"].map(intfmt)
else:
    tableA5 = pd.DataFrame([{"Method": "MISSING", "#Cfg": "--", "#Covered": "--", "Excluded": "--", "Materialization / Note": "--"}])

save_table(
    tableA5,
    "appendix_tableA5_effective_residual_coverage",
    "Effective-weight residual coverage and materialization rules.",
    "tab:appendix_effective_residual_coverage",
    align="lrrrl",
    table_star=True,
    resize=True,
)

# ---------------------------------------------------------------------
# README / report / manifest
# ---------------------------------------------------------------------
source_rows = [
    {"table": "Table 1", "source": str(summary_path.relative_to(ROOT)), "note": "config-level fullzoo metrics"},
    {"table": "Table 2", "source": str(raw_files[0].parent.relative_to(ROOT)) if raw_files else "MISSING", "note": "sample-level BAR/baseline AUC"},
    {"table": "Table 3", "source": str(eff_corr_path.relative_to(ROOT)), "note": "effective-weight residual correlations; representation rows from the canonical mechanism correlation table"},
    {"table": "Appendix A1", "source": str(raw_files[0].parent.relative_to(ROOT)) if raw_files else "MISSING", "note": "per-task sample-level breakdown"},
    {"table": "Appendix A2", "source": str(summary_path.relative_to(ROOT)), "note": "bootstrap over config-level summary"},
    {"table": "Appendix A3", "source": str(raw_files[0].parent.relative_to(ROOT)) if raw_files else "MISSING", "note": "BAR tolerance from sample-level frames"},
    {"table": "Appendix A4", "source": str(proj_eval_file.relative_to(ROOT)) if proj_eval_file else str(summary_path.relative_to(ROOT)), "note": "projection/global correction"},
    {"table": "Appendix A5", "source": str(eff_cov_path.relative_to(ROOT)), "note": "effective residual coverage"},
]
pd.DataFrame(source_rows).to_csv(DATA / "table_source_registry.csv", index=False)

readme = """# Generated manuscript tables

This directory contains manuscript-ready CSV and LaTeX tables generated from user-supplied analysis artifacts.

## Tables

Main paper:
- `table1_main_controlled_subsets`
- `table2_bar_diagnostics`
- `table3_mechanism_evidence`

Appendix:
- `appendix_tableA1_task_breakdown`
- `appendix_tableA2_bootstrap_ci`
- `appendix_tableA3_bar_tolerance`
- `appendix_tableA4_global_correction`
- `appendix_tableA5_effective_residual_coverage`

## Notes

No model evaluation or GPU computation is performed. Tables are generated from existing CSV files under the user-supplied analysis workspace.

Table 3 representation cells require unique exact-key rows in the canonical mechanism correlation input; missing or duplicate keys stop the build.
"""
(PKG / "README.md").write_text(readme)

summary_lines = []
summary_lines.append("=== Manuscript Tables Build Report ===")
summary_lines.append(f"summary_path: {summary_path}")
summary_lines.append(f"n_configs: {len(summary_df)}")
summary_lines.append(f"sample_frames: {len(raw_files)}")
summary_lines.append(f"sample_rows: {len(sample_df) if sample_df is not None else 0}")
summary_lines.append(f"controlled_path: {controlled_path}")
summary_lines.append(f"proj_eval_file: {proj_eval_file}")
summary_lines.append(f"proj_energy_file: {proj_energy_file}")
summary_lines.append(f"effective_corr_path: {eff_corr_path}")
summary_lines.append(f"effective_cov_path: {eff_cov_path}")
summary_lines.append("")
summary_lines.append("Generated tables:")
for p in sorted(TABLES.glob("*.tex")):
    summary_lines.append(f"- {p.relative_to(PKG)}")
summary_lines.append("")
summary_lines.append("Warnings:")
summary_lines.extend([f"- {w}" for w in WARNINGS] if WARNINGS else ["NONE"])

(LOGS / "MANUSCRIPT_TABLES_BUILD_REPORT.txt").write_text("\n".join(summary_lines))
(LOGS / "warnings.txt").write_text("\n".join(WARNINGS) if WARNINGS else "NONE\n")

# Manifest and SHA
manifest = []
for p in sorted(PKG.rglob("*")):
    if p.is_file() and p.name not in {"MANIFEST.txt", "SHA256SUMS.txt"}:
        manifest.append(str(p.relative_to(PKG)))
(PKG / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")

with open(PKG / "SHA256SUMS.txt", "w") as f:
    for rel in manifest:
        p = PKG / rel
        f.write(f"{sha256_file(p)}  {rel}\n")

print("\n".join(summary_lines))
