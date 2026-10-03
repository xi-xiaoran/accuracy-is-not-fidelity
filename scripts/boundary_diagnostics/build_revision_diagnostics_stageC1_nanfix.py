from pathlib import Path
import re
import math
import hashlib
import subprocess
import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
PKG = ROOT / "results/paper_packages/revision_diagnostics_stageC1_nanfix_20260702_v2"
DATA = PKG / "data"
TABLES = PKG / "tables"
LOGS = PKG / "logs"
REPRO = PKG / "repro_grep"
for d in [DATA, TABLES, LOGS, REPRO]:
    d.mkdir(parents=True, exist_ok=True)

REPORT = []
WARN = []

def log(x=""):
    print(x)
    REPORT.append(str(x))

def warn(x):
    print("[WARN]", x)
    WARN.append(str(x))

def read_csv(path, required=True, desc=""):
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    if p.exists():
        log(f"[OK] {desc or p.name}: {p}")
        return pd.read_csv(p), p
    if required:
        raise FileNotFoundError(f"Missing {desc}: {p}")
    warn(f"Missing optional {desc}: {p}")
    return None, p

def find_first(patterns, required=True, desc=""):
    for pat in patterns:
        hits = sorted(ROOT.glob(pat))
        if hits:
            log(f"[OK] found {desc or pat}: {hits[0]}")
            return hits[0]
    if required:
        raise FileNotFoundError(f"Missing {desc}: {patterns}")
    warn(f"Missing optional {desc}: {patterns}")
    return None

def pick_col(df, candidates, required=True):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    for cand in candidates:
        toks = [t for t in re.split(r"[^A-Za-z0-9]+", cand.lower()) if t]
        for c in df.columns:
            cl = c.lower()
            if all(t in cl for t in toks):
                return c
    if required:
        raise KeyError(f"Missing columns {candidates}. Available={list(df.columns)}")
    return None

def pct(x, nd=2):
    if pd.isna(x):
        return "--"
    return f"{float(x):.{nd}f}"

def ci_fmt(m, lo, hi, nd=3):
    if pd.isna(m):
        return "--"
    return f"{m:.{nd}f} [{lo:.{nd}f},{hi:.{nd}f}]"

def intfmt(x):
    if pd.isna(x):
        return "--"
    return f"{int(round(float(x))):,}"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

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

def save_latex(df, name, caption, label, align=None, table_star=True, resize=True):
    if align is None:
        align = "l" * len(df.columns)
    env = "table*" if table_star else "table"
    lines = [
        rf"\begin{{{env}}}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
    ]
    if resize:
        lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{align}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(latex_escape(c) for c in df.columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape(row[c]) for c in df.columns) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if resize:
        lines.append(r"}")
    lines.append(rf"\end{{{env}}}")
    out = TABLES / f"{name}.tex"
    out.write_text("\n".join(lines) + "\n")
    log(f"[WRITE] {out}")

def save_df(df, name, caption=None, label=None, align=None):
    out = DATA / f"{name}.csv"
    df.to_csv(out, index=False)
    log(f"[WRITE] {out}")
    if caption and label:
        save_latex(df, name, caption, label, align=align)

def auc_rank(y, score):
    y = pd.Series(y).astype(float)
    score = pd.Series(score).astype(float)
    m = y.notna() & score.notna()
    y = y[m].astype(int)
    score = score[m].astype(float)
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = score.rank(method="average")
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def spearman(x, y):
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    m = x.notna() & y.notna()
    if m.sum() < 4:
        return np.nan
    return float(x[m].rank().corr(y[m].rank()))

def bh_fdr(pvals):
    p = np.array([np.nan if pd.isna(x) else float(x) for x in pvals], dtype=float)
    out = np.full(len(p), np.nan)
    m = np.isfinite(p)
    idx = np.where(m)[0]
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]
    n = len(order)
    prev = 1.0
    for rank, i in enumerate(order[::-1], start=1):
        k = n - rank + 1
        val = min(prev, p[i] * n / k)
        out[i] = val
        prev = val
    return out

def normal_approx_p_from_r(r, n):
    if pd.isna(r) or n <= 3 or abs(r) >= 1:
        return np.nan
    # t approximation
    t = abs(r) * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    # normal tail approximation via erf; good enough for ranking/FDR report
    # p = 2 * (1 - Phi(t))
    p = math.erfc(t / math.sqrt(2))
    return p

def fixed_effect_residual(v, cov_df):
    y = pd.Series(v).astype(float)
    m = y.notna()
    X_parts = []
    for c in cov_df.columns:
        s = cov_df[c].astype(str).fillna("NA")
        d = pd.get_dummies(s, prefix=c, drop_first=True, dtype=float)
        X_parts.append(d)
    if X_parts:
        X = pd.concat(X_parts, axis=1)
    else:
        X = pd.DataFrame(index=cov_df.index)
    X = X.loc[m]
    yv = y.loc[m].to_numpy(float)
    Xmat = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
    try:
        beta, *_ = np.linalg.lstsq(Xmat, yv, rcond=None)
        resid = yv - Xmat @ beta
    except Exception:
        resid = yv - np.nanmean(yv)
    out = pd.Series(np.nan, index=y.index, dtype=float)
    out.loc[m] = resid
    return out

def partial_spearman(x, y, cov_df):
    xr = pd.Series(x).rank()
    yr = pd.Series(y).rank()
    rx = fixed_effect_residual(xr, cov_df)
    ry = fixed_effect_residual(yr, cov_df)
    return spearman(rx, ry)

# ---------------------------------------------------------------------
# Load config summary
# ---------------------------------------------------------------------
summary_path = find_first([
    "results/theory_predictions_v1_fullzoo/fullzoo_theory_summary_by_config.csv",
    "results/paper_packages/theory_validation_fullzoo_nanfix_20260702_v2/**/fullzoo_theory_summary_by_config.csv",
], desc="fullzoo_theory_summary_by_config.csv")
summary = pd.read_csv(summary_path)

target_col = pick_col(summary, ["target_id"])
model_col = pick_col(summary, ["model_key"])
method_col = pick_col(summary, ["method_key"])
keep_col = pick_col(summary, ["keep_ratio"])
dense_col = pick_col(summary, ["dense_acc_pct", "dense_acc"])
comp_col = pick_col(summary, ["compressed_acc_pct", "compressed_acc"])
delta_col = pick_col(summary, ["acc_delta_pct", "delta_acc_pct", "acc_delta"])
flip_col = pick_col(summary, ["pred_flip_pct", "pred_flip"])
hidden_col = pick_col(summary, ["hidden_churn_pct", "hidden_churn"])
bar_risk_col = pick_col(summary, ["bar_ge1_flip", "bar_ge1_flip_pct", "bar_risk_flip"], required=False)

summary["_keep_str"] = summary[keep_col].astype(str)
summary["_keep_norm"] = summary["_keep_str"].str.replace(".", "p", regex=False)

log(f"n_configs={len(summary)}")

# ---------------------------------------------------------------------
# 1. Config coverage / imbalance
# ---------------------------------------------------------------------
coverage_long = (
    summary.groupby([model_col, method_col, keep_col])
    .size()
    .reset_index(name="n_configs")
    .sort_values([model_col, method_col, keep_col])
)
save_df(
    coverage_long,
    "config_matrix_long",
    "Observed compressed configuration matrix. The suite is intentionally not a complete model--method--keep-ratio grid.",
    "tab:config_matrix_long",
    align="lll r"
)

method_cov = (
    summary.groupby(method_col)
    .agg(
        n_configs=(target_col, "count"),
        n_models=(model_col, "nunique"),
        n_keep_ratios=(keep_col, "nunique"),
        mean_delta_acc=(delta_col, "mean"),
        mean_predflip=(flip_col, "mean"),
        mean_hiddenchurn=(hidden_col, "mean"),
    )
    .reset_index()
)
method_cov["low_count_flag"] = np.where(method_cov["n_configs"] <= 5, "low-count", "")
method_cov = method_cov.sort_values(["n_configs", method_col])
method_cov_fmt = method_cov.copy()
for c in ["mean_delta_acc", "mean_predflip", "mean_hiddenchurn"]:
    method_cov_fmt[c] = method_cov_fmt[c].map(lambda x: f"{x:.2f}")
save_df(
    method_cov,
    "method_coverage_summary_raw"
)
save_df(
    method_cov_fmt,
    "method_coverage_summary",
    "Method-family coverage summary. Low-count variants should not be interpreted as stable controlled method rankings.",
    "tab:method_coverage_summary",
    align="lrrrrrl"
)

# Pair counts by model/method as wide matrix
model_method = summary.groupby([model_col, method_col]).size().unstack(fill_value=0)
model_method.to_csv(DATA / "config_matrix_model_by_method.csv")
log(f"[WRITE] {DATA / 'config_matrix_model_by_method.csv'}")
keep_method = summary.groupby([keep_col, method_col]).size().unstack(fill_value=0)
keep_method.to_csv(DATA / "config_matrix_keep_by_method.csv")
log(f"[WRITE] {DATA / 'config_matrix_keep_by_method.csv'}")

# ---------------------------------------------------------------------
# 2. Controlled subset membership and leave-one-out
# ---------------------------------------------------------------------
controlled_masks = {
    "abs_delta_le5": summary[delta_col].abs() <= 5,
    "compressed_acc_ge60": summary[comp_col] >= 60,
    "abs_delta_le10": summary[delta_col].abs() <= 10,
    "compressed_acc_ge50": summary[comp_col] >= 50,
}

member_rows = []
for subset, mask in controlled_masks.items():
    d = summary.loc[mask].copy()
    for _, r in d.iterrows():
        member_rows.append({
            "subset": subset,
            "target_id": r[target_col],
            "model_key": r[model_col],
            "method_key": r[method_col],
            "keep_ratio": r[keep_col],
            "dense_acc": r[dense_col],
            "compressed_acc": r[comp_col],
            "delta_acc": r[delta_col],
            "pred_flip": r[flip_col],
            "hidden_churn": r[hidden_col],
            "bar_risk_flip": r[bar_risk_col] if bar_risk_col else np.nan,
        })
members = pd.DataFrame(member_rows)
save_df(
    members,
    "controlled_subset_membership",
    "Membership of accuracy-controlled subsets.",
    "tab:controlled_subset_membership",
    align="llllrrrrrr"
)

loo_rows = []
for subset, mask in controlled_masks.items():
    d = summary.loc[mask].copy()
    base = {
        "subset": subset,
        "n_configs": len(d),
        "full_delta_acc": d[delta_col].mean(),
        "full_pred_flip": d[flip_col].mean(),
        "full_hidden_churn": d[hidden_col].mean(),
    }
    for _, r in d.iterrows():
        dd = d[d[target_col] != r[target_col]]
        if len(dd) == 0:
            continue
        row = dict(base)
        row.update({
            "left_out_target": r[target_col],
            "left_out_model": r[model_col],
            "left_out_method": r[method_col],
            "left_out_keep": r[keep_col],
            "loo_delta_acc": dd[delta_col].mean(),
            "loo_pred_flip": dd[flip_col].mean(),
            "loo_hidden_churn": dd[hidden_col].mean(),
            "change_hidden_churn": dd[hidden_col].mean() - d[hidden_col].mean(),
            "change_pred_flip": dd[flip_col].mean() - d[flip_col].mean(),
        })
        loo_rows.append(row)
loo = pd.DataFrame(loo_rows)
save_df(loo, "controlled_subset_leave_one_out_raw")

loo_summary = []
for subset, g in loo.groupby("subset"):
    loo_summary.append({
        "Subset": subset,
        "#Cfg": intfmt(g["n_configs"].iloc[0]),
        "Full HiddenChurn": pct(g["full_hidden_churn"].iloc[0], 2),
        "LOO HiddenChurn Min": pct(g["loo_hidden_churn"].min(), 2),
        "LOO HiddenChurn Max": pct(g["loo_hidden_churn"].max(), 2),
        "Max |Change|": pct(g["change_hidden_churn"].abs().max(), 2),
        "Most Influential Config": str(g.loc[g["change_hidden_churn"].abs().idxmax(), "left_out_target"]) if len(g) else "--",
    })
loo_summary = pd.DataFrame(loo_summary)
save_df(
    loo_summary,
    "controlled_subset_leave_one_out_summary",
    "Leave-one-out stability of accuracy-controlled subset HiddenChurn.",
    "tab:controlled_subset_loo",
    align="lrrrrrl"
)

# ---------------------------------------------------------------------
# 3. Load sample frames for row-count audit and BAR AUC cluster CI
# ---------------------------------------------------------------------
raw_dir = find_first([
    "results/paper_packages/theory_validation_fullzoo_nanfix_20260702_v2/raw/theory_predictions_v1",
    "results/theory_predictions_v1",
], desc="sample theory frames directory")
raw_files = sorted(raw_dir.glob("*_sample_theory_frame.csv"))
if len(raw_files) == 0:
    raise FileNotFoundError(f"No sample theory frames under {raw_dir}")
log(f"sample_frames={len(raw_files)}")

wanted = [
    "task", "flip", "boundary_ratio", "adv_gap_delta", "dense_margin",
    "prob_js", "score_l2_delta", "logit_l2_delta", "delta_l2", "score_l2",
    "dense_correct", "compressed_correct", "dense_pred", "compressed_pred",
    "gold", "n_choices", "num_choices"
]
frames = []
count_rows = []
for p in raw_files:
    header = pd.read_csv(p, nrows=0).columns.tolist()
    usecols = [c for c in wanted if c in header]
    df = pd.read_csv(p, usecols=usecols)
    tid = p.name.replace("_sample_theory_frame.csv", "")
    df["target_id"] = tid
    frames.append(df)
    if "task" in df.columns:
        for task, g in df.groupby("task"):
            count_rows.append({"target_id": tid, "task": task, "n_rows": len(g)})
sample = pd.concat(frames, ignore_index=True)
log(f"sample_rows={len(sample):,}")

counts = pd.DataFrame(count_rows)
counts = counts.merge(summary[[target_col, model_col, method_col, keep_col]], left_on="target_id", right_on=target_col, how="left")
save_df(counts, "task_row_counts_by_config")

task_count_summary = (
    counts.groupby("task")
    .agg(
        n_configs=("target_id", "nunique"),
        total_rows=("n_rows", "sum"),
        min_rows=("n_rows", "min"),
        max_rows=("n_rows", "max"),
        mean_rows=("n_rows", "mean"),
        n_nonstandard=("n_rows", lambda s: int((s != s.max()).sum())),
    )
    .reset_index()
)
save_df(
    task_count_summary,
    "task_row_count_audit",
    "Task row-count audit across compressed configurations.",
    "tab:task_row_count_audit",
    align="lrrrrrr"
)

# OpenBookQA missing row trace
open_mask = counts["task"].astype(str).str.lower().eq("openbookqa")
open_counts = counts.loc[open_mask].copy()
open_missing = pd.DataFrame()
if len(open_counts):
    expected = int(open_counts["n_rows"].max())
    open_missing = open_counts[open_counts["n_rows"] != expected].copy()
    open_missing["expected_rows"] = expected
    open_missing["missing_rows"] = expected - open_missing["n_rows"]
save_df(
    open_missing,
    "openbookqa_missing_row_trace",
    "OpenBookQA row-count deviations by compressed configuration.",
    "tab:openbookqa_missing_trace",
    align="llllrrrr"
)

# ---------------------------------------------------------------------
# 4. Cluster-level AUC CI for BAR baseline table
# ---------------------------------------------------------------------
sample["flip_numeric"] = pd.to_numeric(sample["flip"], errors="coerce").fillna(0).astype(int)
metric_defs = []
if "boundary_ratio" in sample.columns:
    metric_defs.append(("BAR", "boundary_ratio", sample["boundary_ratio"]))
if "adv_gap_delta" in sample.columns:
    metric_defs.append(("Raw competitor-over-winner perturbation B", "adv_gap_delta", sample["adv_gap_delta"]))
if "dense_margin" in sample.columns:
    metric_defs.append(("Negative dense margin", "neg_dense_margin", -pd.to_numeric(sample["dense_margin"], errors="coerce")))
if "prob_js" in sample.columns:
    metric_defs.append(("Prob-JS", "prob_js", sample["prob_js"]))
l2_col = None
for c in ["score_l2_delta", "logit_l2_delta", "delta_l2", "score_l2"]:
    if c in sample.columns:
        l2_col = c
        break
if l2_col:
    metric_defs.append(("Score L2 drift", l2_col, sample[l2_col]))

# global AUC + per-target AUC
auc_rows = []
target_auc_rows = []
for name, key, score in metric_defs:
    global_auc = auc_rank(sample["flip_numeric"], score)
    tmp = pd.DataFrame({"target_id": sample["target_id"], "flip": sample["flip_numeric"], "score": score})
    vals = []
    for tid, g in tmp.groupby("target_id"):
        a = auc_rank(g["flip"], g["score"])
        if not pd.isna(a):
            vals.append((tid, a, len(g)))
            target_auc_rows.append({"metric": name, "target_id": tid, "target_auc": a, "n_rows": len(g)})
    arr = np.array([v[1] for v in vals], dtype=float)
    rng = np.random.default_rng(20260702)
    boots = []
    if len(arr) > 0:
        for _ in range(10000):
            idx = rng.integers(0, len(arr), len(arr))
            boots.append(np.nanmean(arr[idx]))
        lo, hi = np.quantile(boots, [0.025, 0.975])
        mean_target_auc = float(np.nanmean(arr))
    else:
        lo = hi = mean_target_auc = np.nan
    auc_rows.append({
        "Metric": name,
        "Global row-level AUC": global_auc,
        "Mean target-level AUC": mean_target_auc,
        "Target-cluster 95% CI low": lo,
        "Target-cluster 95% CI high": hi,
        "#Targets": len(arr),
    })
auc_df = pd.DataFrame(auc_rows)
auc_fmt = auc_df.copy()
auc_fmt["Global row-level AUC"] = auc_fmt["Global row-level AUC"].map(lambda x: f"{x:.3f}")
auc_fmt["Mean target-level AUC"] = auc_fmt.apply(lambda r: ci_fmt(r["Mean target-level AUC"], r["Target-cluster 95% CI low"], r["Target-cluster 95% CI high"], 3), axis=1)
auc_fmt = auc_fmt[["Metric", "Global row-level AUC", "Mean target-level AUC", "#Targets"]]
save_df(pd.DataFrame(target_auc_rows), "table2_target_level_auc_raw")
save_df(
    auc_df,
    "table2_cluster_bootstrap_auc_ci_raw"
)
save_df(
    auc_fmt,
    "table2_cluster_bootstrap_auc_ci",
    "BAR and baseline AUC with target-cluster bootstrap intervals. Dense margin uses negative margin because lower margin means higher risk.",
    "tab:bar_auc_cluster_ci",
    align="lrrr"
)

# ---------------------------------------------------------------------
# 5. Effective residual controlled correlations
# ---------------------------------------------------------------------
eff_pkg = ROOT / "results/paper_packages/effective_weight_residual_final205_20260702_v1"
module_file = find_first([
    "results/paper_packages/effective_weight_residual_final205_20260702_v1/raw/stageB1e_target_module_effective_residual_summary_final205.csv",
    "results/paper_packages/effective_weight_residual_final205_20260702_v1/**/stageB1e_target_module_effective_residual_summary_final205.csv",
], desc="target-module effective residual summary")
target_file = find_first([
    "results/paper_packages/effective_weight_residual_final205_20260702_v1/raw/stageB1e_target_effective_residual_summary_final205.csv",
    "results/paper_packages/effective_weight_residual_final205_20260702_v1/**/stageB1e_target_effective_residual_summary_final205.csv",
], desc="target effective residual summary")
module_df = pd.read_csv(module_file)
target_eff = pd.read_csv(target_file)

# identify columns
tid_m = pick_col(module_df, ["target_id"])
module_name_col = None
for c in module_df.columns:
    if module_df[c].dtype == "object":
        vals = module_df[c].astype(str).head(5000).str.lower()
        if vals.str.contains("o_proj|v_proj|q_proj|k_proj|gate_proj|up_proj|down_proj", regex=True).any():
            module_name_col = c
            break
if module_name_col is None:
    raise KeyError(f"Cannot identify module column in {module_df.columns}")

numeric_cols = [c for c in module_df.columns if c not in [tid_m, module_name_col] and pd.api.types.is_numeric_dtype(module_df[c])]
metric_candidates = [c for c in numeric_cols if any(k in c.lower() for k in ["rel_frob", "cosine", "mean_module", "median_module"])]
if not metric_candidates:
    metric_candidates = numeric_cols

# aggregate by target and module short name
def module_short(x):
    x = str(x)
    for key in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        if key in x:
            return key
    return x.split(".")[-1]

module_df["_module_short"] = module_df[module_name_col].map(module_short)
mod_agg = module_df.groupby([tid_m, "_module_short"])[metric_candidates].mean().reset_index()

# make wide features: module:metric
wide_parts = []
for metric in metric_candidates:
    piv = mod_agg.pivot(index=tid_m, columns="_module_short", values=metric)
    piv.columns = [f"module:{c}::{metric}" for c in piv.columns]
    wide_parts.append(piv)
module_wide = pd.concat(wide_parts, axis=1).reset_index().rename(columns={tid_m: "target_id"})

tid_t = pick_col(target_eff, ["target_id"])
target_eff2 = target_eff.rename(columns={tid_t: "target_id"})
target_numeric = [c for c in target_eff2.columns if c != "target_id" and pd.api.types.is_numeric_dtype(target_eff2[c])]
target_eff2 = target_eff2[["target_id"] + target_numeric]
target_eff2 = target_eff2.rename(columns={c: f"target::{c}" for c in target_numeric})

beh = summary.rename(columns={target_col: "target_id"}).copy()
merged = beh.merge(module_wide, on="target_id", how="inner").merge(target_eff2, on="target_id", how="left")
save_df(merged, "effective_residual_behavior_merged_for_controls")

behavior_metrics = {
    "PredFlip": flip_col,
    "HiddenChurn": hidden_col,
}
if bar_risk_col:
    behavior_metrics["Flip Rate | BAR>=1"] = bar_risk_col

feature_cols = [c for c in merged.columns if c.startswith("module:") or c.startswith("target::")]
# Prefer paper features, then all features
preferred_patterns = [
    "module:o_proj::mean_module_rel_frob",
    "module:o_proj::rel_frob",
    "module:o_proj::mean_cosine",
    "module:v_proj::mean_cosine",
    "module:v_proj::rel_frob",
    "module:v_proj::mean_module_rel_frob",
    "target::mean_cosine",
    "target::median_module_rel_frob",
]
preferred = []
for pat in preferred_patterns:
    hits = [c for c in feature_cols if c == pat or pat in c]
    preferred.extend(hits)
preferred = list(dict.fromkeys(preferred))
if not preferred:
    preferred = feature_cols[:20]

def bootstrap_spearman(df, xcol, ycol, n_boot=10000, seed=20260702):
    d = df[[xcol, ycol]].dropna()
    n = len(d)
    if n < 4:
        return np.nan, np.nan, np.nan
    base = spearman(d[xcol], d[ycol])
    rng = np.random.default_rng(seed)
    arr = d.to_numpy(float)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(spearman(arr[idx,0], arr[idx,1]))
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return base, float(lo), float(hi)

def bootstrap_partial(df, xcol, ycol, covcols, n_boot=5000, seed=20260702):
    d = df[[xcol, ycol] + covcols].dropna()
    n = len(d)
    if n < 10:
        return np.nan, np.nan, np.nan
    base = partial_spearman(d[xcol], d[ycol], d[covcols])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        dd = d.iloc[idx].reset_index(drop=True)
        vals.append(partial_spearman(dd[xcol], dd[ycol], dd[covcols]))
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return base, float(lo), float(hi)

covcols = [model_col, method_col, keep_col]
controlled_rows = []
all_corr_rows = []

for feat in feature_cols:
    for bname, bcol in behavior_metrics.items():
        d = merged[[feat, bcol] + covcols].copy()
        raw, raw_lo, raw_hi = bootstrap_spearman(d, feat, bcol, n_boot=5000)
        part, part_lo, part_hi = bootstrap_partial(d, feat, bcol, covcols, n_boot=2000)
        n = d[[feat, bcol] + covcols].dropna().shape[0]
        p_raw = normal_approx_p_from_r(raw, n)
        p_part = normal_approx_p_from_r(part, n)
        row = {
            "feature": feat,
            "behavior_metric": bname,
            "n_configs": n,
            "raw_spearman": raw,
            "raw_ci_low": raw_lo,
            "raw_ci_high": raw_hi,
            "partial_spearman_model_method_keep": part,
            "partial_ci_low": part_lo,
            "partial_ci_high": part_hi,
            "p_raw_approx": p_raw,
            "p_partial_approx": p_part,
        }
        all_corr_rows.append(row)
        if feat in preferred:
            controlled_rows.append(row)

all_corr = pd.DataFrame(all_corr_rows)
all_corr["q_raw_bh"] = bh_fdr(all_corr["p_raw_approx"])
all_corr["q_partial_bh"] = bh_fdr(all_corr["p_partial_approx"])
all_corr = all_corr.sort_values("raw_spearman", key=lambda s: s.abs(), ascending=False)
save_df(all_corr, "effective_residual_all_module_correlations_controlled_raw")

controlled_corr = pd.DataFrame(controlled_rows)
controlled_corr["raw_spearman_CI"] = controlled_corr.apply(lambda r: ci_fmt(r["raw_spearman"], r["raw_ci_low"], r["raw_ci_high"], 3), axis=1)
controlled_corr["partial_spearman_CI"] = controlled_corr.apply(lambda r: ci_fmt(r["partial_spearman_model_method_keep"], r["partial_ci_low"], r["partial_ci_high"], 3), axis=1)
controlled_fmt = controlled_corr[[
    "feature", "behavior_metric", "n_configs", "raw_spearman_CI", "partial_spearman_CI"
]].copy()
save_df(
    controlled_corr,
    "effective_residual_preferred_correlations_controlled_raw"
)
save_df(
    controlled_fmt,
    "effective_residual_preferred_correlations_controlled",
    "Effective-weight residual correlations with raw and fixed-effect residualized Spearman estimates. Partial correlations residualize model, method, and keep-ratio indicators.",
    "tab:effective_residual_controlled_corr",
    align="llrrr"
)

# Top all-module table
top_all = all_corr.head(40).copy()
top_all["raw_spearman"] = top_all.apply(lambda r: ci_fmt(r["raw_spearman"], r["raw_ci_low"], r["raw_ci_high"], 3), axis=1)
top_all["partial_spearman_model_method_keep"] = top_all.apply(lambda r: ci_fmt(r["partial_spearman_model_method_keep"], r["partial_ci_low"], r["partial_ci_high"], 3), axis=1)
top_all_fmt = top_all[["feature", "behavior_metric", "n_configs", "raw_spearman", "partial_spearman_model_method_keep", "q_raw_bh", "q_partial_bh"]].copy()
top_all_fmt["q_raw_bh"] = top_all_fmt["q_raw_bh"].map(lambda x: f"{x:.3g}" if not pd.isna(x) else "--")
top_all_fmt["q_partial_bh"] = top_all_fmt["q_partial_bh"].map(lambda x: f"{x:.3g}" if not pd.isna(x) else "--")
save_df(
    top_all_fmt,
    "effective_residual_all_module_top40",
    "Top module-level effective-residual associations before and after fixed-effect residualization.",
    "tab:effective_residual_all_module_top40",
    align="llrrrrr"
)

# ---------------------------------------------------------------------
# 6. Method fixed-effect model summaries for behavior metrics
# ---------------------------------------------------------------------
def ols_fixed_effect(df, ycol, covcols):
    d = df[[ycol] + covcols].dropna().copy()
    y = d[ycol].astype(float).to_numpy()
    X_parts = []
    names = ["Intercept"]
    for c in covcols:
        dum = pd.get_dummies(d[c].astype(str), prefix=c, drop_first=True, dtype=float)
        names.extend(list(dum.columns))
        X_parts.append(dum)
    X = pd.concat(X_parts, axis=1) if X_parts else pd.DataFrame(index=d.index)
    Xmat = np.column_stack([np.ones(len(d)), X.to_numpy(float)])
    beta, *_ = np.linalg.lstsq(Xmat, y, rcond=None)
    yhat = Xmat @ beta
    resid = y - yhat
    sse = float((resid ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst if sst > 0 else np.nan
    return pd.DataFrame({"term": names, "coef": beta, "metric": ycol, "n": len(d), "r2": r2})

ols_rows = []
for ycol in [delta_col, flip_col, hidden_col]:
    ols_rows.append(ols_fixed_effect(summary, ycol, [model_col, keep_col, method_col]))
ols = pd.concat(ols_rows, ignore_index=True)
save_df(ols, "method_model_keep_fixed_effect_ols_raw")

# ---------------------------------------------------------------------
# 7. Reproducibility grep bundle
# ---------------------------------------------------------------------
grep_patterns = [
    "lm_eval|lm-eval|lm_eval.__version__|version",
    "num_fewshot|fewshot|zero-shot|zero_shot",
    "model_args|pretrained=|trust_remote_code|chat_template|template",
    "acc_norm|loglikelihood|length|normalize|normalization",
    "dtype|float16|bfloat16|bf16|fp16|fp32|tie|argmax",
    "qwen|llama|instruct|tokenizer|bos|eos",
]
search_roots = [
    ROOT / "scripts",
    ROOT / "results/paper_packages/theory_validation_fullzoo_20260630_v1/logs",
    ROOT / "results/paper_packages/server_handoff_20260701_v1",
]
for i, pat in enumerate(grep_patterns):
    out = REPRO / f"grep_{i+1}.txt"
    cmd = ["bash", "-lc", " ".join([
        "grep -RInE",
        repr(pat),
        " ".join(str(p) for p in search_roots if p.exists()),
        "2>/dev/null | head -300"
    ])]
    try:
        res = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        out.write_text(res.stdout + ("\nSTDERR:\n" + res.stderr if res.stderr else ""))
    except Exception as e:
        out.write_text(f"grep failed: {repr(e)}\n")
    log(f"[WRITE] {out}")

# ---------------------------------------------------------------------
# 8. Paper-ready notes / action checklist
# ---------------------------------------------------------------------
notes = []
notes.append("# Stage C1 Revision Diagnostics Notes")
notes.append("")
notes.append("## Immediate manuscript implications")
notes.append("")
notes.append("1. BAR must be described as a post-hoc score-space diagnostic, not a pre-compression predictor.")
notes.append("2. BAR<1 tolerance results should be described as a deterministic certificate / numerical consistency check.")
notes.append("3. Method-family averages are unbalanced; do not present them as controlled method rankings.")
notes.append("4. Effective-weight residual results should be described as operator-level associations. Use the controlled partial-correlation table to avoid overclaiming.")
notes.append("5. Dense-margin AUC uses negative margin because lower dense margin indicates higher flip risk.")
notes.append("6. The OpenBookQA row-count trace identifies whether the 117,499 row count is due to a single target-level missing/filtered row.")
notes.append("")
notes.append("## Files to inspect before manuscript patch")
notes.append("")
notes.append("- data/openbookqa_missing_row_trace.csv")
notes.append("- data/controlled_subset_membership.csv")
notes.append("- data/controlled_subset_leave_one_out_summary.csv")
notes.append("- data/table2_cluster_bootstrap_auc_ci.csv")
notes.append("- data/effective_residual_preferred_correlations_controlled.csv")
notes.append("- data/effective_residual_all_module_top40.csv")
notes.append("- repro_grep/*.txt")
(PKG / "REVISION_ACTION_NOTES.md").write_text("\n".join(notes) + "\n")

# ---------------------------------------------------------------------
# Manifest / checksum / report
# ---------------------------------------------------------------------
source_registry = pd.DataFrame([
    {"analysis": "config coverage", "source": str(summary_path.relative_to(ROOT))},
    {"analysis": "controlled subsets", "source": str(summary_path.relative_to(ROOT))},
    {"analysis": "task row counts / BAR AUC", "source": str(raw_dir.relative_to(ROOT))},
    {"analysis": "effective residual control", "source": str(module_file.relative_to(ROOT))},
    {"analysis": "target effective residual", "source": str(target_file.relative_to(ROOT))},
])
save_df(source_registry, "stageC1_source_registry")

report = []
report.append("=== Stage C1 Revision Diagnostics Build Report ===")
report.append(f"summary_path: {summary_path}")
report.append(f"n_configs: {len(summary)}")
report.append(f"raw_dir: {raw_dir}")
report.append(f"sample_frames: {len(raw_files)}")
report.append(f"sample_rows: {len(sample):,}")
report.append(f"openbookqa_missing_configs: {len(open_missing)}")
report.append(f"effective_module_file: {module_file}")
report.append(f"effective_target_file: {target_file}")
report.append(f"merged_effective_configs: {len(merged)}")
report.append("")
report.append("Key output files:")
for p in [
    DATA / "config_matrix_long.csv",
    DATA / "method_coverage_summary.csv",
    DATA / "controlled_subset_membership.csv",
    DATA / "controlled_subset_leave_one_out_summary.csv",
    DATA / "task_row_count_audit.csv",
    DATA / "openbookqa_missing_row_trace.csv",
    DATA / "table2_cluster_bootstrap_auc_ci.csv",
    DATA / "effective_residual_preferred_correlations_controlled.csv",
    DATA / "effective_residual_all_module_top40.csv",
]:
    report.append(f"- {p.relative_to(PKG)}")
report.append("")
report.append("Warnings:")
report.extend([f"- {w}" for w in WARN] if WARN else ["NONE"])

(LOGS / "STAGEC1_BUILD_REPORT.txt").write_text("\n".join(report) + "\n")
(LOGS / "warnings.txt").write_text("\n".join(WARN) + "\n" if WARN else "NONE\n")

manifest = []
for p in sorted(PKG.rglob("*")):
    if p.is_file() and p.name not in {"MANIFEST.txt", "SHA256SUMS.txt"}:
        manifest.append(str(p.relative_to(PKG)))
(PKG / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")
with open(PKG / "SHA256SUMS.txt", "w") as f:
    for rel in manifest:
        p = PKG / rel
        f.write(f"{sha256_file(p)}  {rel}\n")

print("\n".join(report))
