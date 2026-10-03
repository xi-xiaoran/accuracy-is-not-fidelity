from pathlib import Path
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = Path(".").resolve()

BASE = (
    ROOT
    / "results/stage18_official_asvd_hifi_v1"
)

AUTH_PATH = (
    BASE
    / "theory_eval_v1"
    / "stage18_authoritative_theory_eval_config_metrics.csv"
)
CI_PATH = (
    BASE
    / "theory_eval_bootstrap_v1"
    / "stage18_authoritative_sample_bootstrap_ci.csv"
)
BUDGET_PATH = (
    BASE
    / "budget_audit_v1"
    / "stage18_asvd_actual_budget.csv"
)
MAC_PATH = (
    BASE
    / "mac_budget_v1"
    / "stage18_asvd_mac_budget.csv"
)
FULL_PATH = (
    BASE
    / "aggregate_v1"
    / "stage18_asvd_hifi_by_config.csv"
)
COUNTS_PATH = (
    BASE
    / "stage17_extended_v1"
    / "old_vs_extended_threshold_counts.csv"
)
THRESHOLD_PATH = (
    BASE
    / "stage17_extended_v1"
    / "extended_threshold_summary.csv"
)

SOURCE_PATHS = [
    AUTH_PATH,
    CI_PATH,
    BUDGET_PATH,
    MAC_PATH,
    FULL_PATH,
    COUNTS_PATH,
    THRESHOLD_PATH,
]

FINAL_OUT = BASE / "final_evidence_v1"
TMP_OUT = FINAL_OUT.with_name(
    f".final_evidence_v1.tmp.{os.getpid()}"
)

assert not FINAL_OUT.exists()
assert not TMP_OUT.exists()

for path in SOURCE_PATHS:
    assert path.is_file(), path

TMP_OUT.mkdir(parents=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def choose_column(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(
        f"None of {candidates} found in "
        f"{list(df.columns)}"
    )


def model_to_keep(value):
    match = re.search(r"(\d{2})$", str(value))
    if not match:
        raise ValueError(value)
    return int(match.group(1)) / 100.0


auth = pd.read_csv(AUTH_PATH)
ci = pd.read_csv(CI_PATH)
budget = pd.read_csv(BUDGET_PATH)
mac = pd.read_csv(MAC_PATH)
full = pd.read_csv(FULL_PATH)
counts = pd.read_csv(COUNTS_PATH)
threshold = pd.read_csv(THRESHOLD_PATH)

assert len(auth) == 3
assert auth["target_id"].is_unique
assert set(np.round(auth["keep_ratio"], 2)) == {
    0.85, 0.90, 0.95
}

auth["keep_key"] = auth["keep_ratio"].round(4)

# Bootstrap intervals.
ci = ci[
    ci["metric"].isin([
        "acc_delta_pct",
        "pred_flip_pct",
        "hidden_churn_pct",
    ])
].copy()

ci["keep_key"] = ci["keep_ratio"].round(4)

for metric, prefix in [
    ("acc_delta_pct", "acc_delta"),
    ("pred_flip_pct", "pred_flip"),
    ("hidden_churn_pct", "hidden_churn"),
]:
    part = ci[ci["metric"] == metric][[
        "keep_key",
        "n_eval",
        "point_estimate",
        "bootstrap_se",
        "ci_2p5",
        "ci_97p5",
        "p_le_zero",
    ]].rename(columns={
        "point_estimate": f"{prefix}_point",
        "bootstrap_se": f"{prefix}_bootstrap_se",
        "ci_2p5": f"{prefix}_ci_low",
        "ci_97p5": f"{prefix}_ci_high",
        "p_le_zero": f"{prefix}_p_le_zero",
    })

    if "n_eval" in auth.columns:
        part = part.drop(columns=["n_eval"])

    auth = auth.merge(
        part,
        on="keep_key",
        how="left",
        validate="one_to_one",
    )

# Actual parameter/storage budgets.
budget_model_col = choose_column(
    budget,
    ["model", "model_name", "name", "label"],
)
budget_parameter_keep_col = choose_column(
    budget,
    [
        "actual_parameter_keep",
        "actual_parameter_keep_ratio",
        "actual_parameter_ratio",
        "parameter_keep_ratio",
    ],
)
budget_storage_keep_col = choose_column(
    budget,
    [
        "actual_storage_keep",
        "actual_storage_keep_ratio",
        "actual_storage_ratio",
        "storage_keep_ratio",
    ],
)

budget = budget.copy()
budget["keep_key"] = budget[
    budget_model_col
].map(model_to_keep)

budget_small = budget[[
    "keep_key",
    budget_parameter_keep_col,
    budget_storage_keep_col,
]].rename(columns={
    budget_parameter_keep_col:
        "actual_parameter_keep_ratio",
    budget_storage_keep_col:
        "actual_storage_keep_ratio",
})

budget_small["parameter_reduction"] = (
    1.0
    - budget_small["actual_parameter_keep_ratio"]
)
budget_small["storage_reduction"] = (
    1.0
    - budget_small["actual_storage_keep_ratio"]
)

auth = auth.merge(
    budget_small,
    on="keep_key",
    how="left",
    validate="one_to_one",
)

# Linear and standard MAC budgets.
mac = mac[mac["model"] != "dense"].copy()
mac["keep_key"] = mac["model"].map(model_to_keep)

mac_small = mac[[
    "keep_key",
    "factorized_projection_count",
    "block_linear_macs_per_token_reduction",
    "total_linear_macs_per_token_reduction",
    "standard_mac_proxy_s512_reduction",
    "standard_mac_proxy_s2048_reduction",
    "standard_mac_proxy_s4096_reduction",
]]

auth = auth.merge(
    mac_small,
    on="keep_key",
    how="left",
    validate="one_to_one",
)

# Full 20,465-sample robustness results.
full = full.copy()
full["keep_key"] = full["keep_ratio"].round(4)

full_small = full[[
    "keep_key",
    "n_samples",
    "dense_acc_score",
    "compressed_acc_score",
    "acc_delta_score",
    "pred_flip_score",
    "hidden_churn",
]].rename(columns={
    "n_samples": "full_sample_n",
    "dense_acc_score": "full_sample_dense_acc",
    "compressed_acc_score":
        "full_sample_compressed_acc",
    "acc_delta_score": "full_sample_acc_delta",
    "pred_flip_score": "full_sample_pred_flip",
    "hidden_churn": "full_sample_hidden_churn",
})

for column in [
    "full_sample_dense_acc",
    "full_sample_compressed_acc",
    "full_sample_acc_delta",
    "full_sample_pred_flip",
    "full_sample_hidden_churn",
]:
    full_small[column] *= 100.0

auth = auth.merge(
    full_small,
    on="keep_key",
    how="left",
    validate="one_to_one",
)

auth = auth.sort_values("keep_ratio").reset_index(
    drop=True
)

required_non_null = [
    "acc_delta_ci_low",
    "acc_delta_ci_high",
    "pred_flip_ci_low",
    "pred_flip_ci_high",
    "hidden_churn_ci_low",
    "hidden_churn_ci_high",
    "actual_parameter_keep_ratio",
    "actual_storage_keep_ratio",
    "block_linear_macs_per_token_reduction",
    "standard_mac_proxy_s2048_reduction",
    "full_sample_pred_flip",
    "full_sample_hidden_churn",
]

assert not auth[required_non_null].isna().any().any()

# Metric identities.
flip_error = (
    auth["pred_flip_pct"]
    - auth["damage_pct"]
    - auth["repair_pct"]
    - auth["wrong_to_wrong_pct"]
).abs()

hidden_error = (
    auth["hidden_churn_pct"]
    - (
        auth["pred_flip_pct"]
        - auth["acc_delta_pct"].abs()
    )
).abs()

assert float(flip_error.max()) < 1e-10
assert float(hidden_error.max()) < 1e-10

# Dose response.
assert auth["pred_flip_pct"].is_monotonic_decreasing
assert auth["hidden_churn_pct"].is_monotonic_decreasing
assert auth["prob_js_pct"].is_monotonic_decreasing
assert auth["actual_parameter_keep_ratio"].is_monotonic_increasing

# All bootstrap intervals for churn remain positive.
assert (auth["pred_flip_ci_low"] > 0).all()
assert (auth["hidden_churn_ci_low"] > 0).all()

final_columns = [
    "target_id",
    "model_key",
    "method_key",
    "keep_ratio",
    "actual_parameter_keep_ratio",
    "parameter_reduction",
    "actual_storage_keep_ratio",
    "storage_reduction",
    "factorized_projection_count",
    "block_linear_macs_per_token_reduction",
    "total_linear_macs_per_token_reduction",
    "standard_mac_proxy_s512_reduction",
    "standard_mac_proxy_s2048_reduction",
    "standard_mac_proxy_s4096_reduction",
    "dense_acc_pct",
    "compressed_acc_pct",
    "acc_delta_pct",
    "acc_delta_ci_low",
    "acc_delta_ci_high",
    "pred_flip_pct",
    "pred_flip_ci_low",
    "pred_flip_ci_high",
    "hidden_churn_pct",
    "hidden_churn_ci_low",
    "hidden_churn_ci_high",
    "damage_pct",
    "repair_pct",
    "wrong_to_wrong_pct",
    "prob_js_pct",
    "n_eval",
    "full_sample_n",
    "full_sample_acc_delta",
    "full_sample_pred_flip",
    "full_sample_hidden_churn",
]

# n_eval came from the bootstrap table.
if "n_eval" not in auth.columns:
    raise RuntimeError("n_eval missing after CI merge")

final = auth[final_columns].copy()

final.to_csv(
    TMP_OUT / "stage18_final_evidence_table.csv",
    index=False,
)

counts.to_csv(
    TMP_OUT / "stage18_threshold_count_extension.csv",
    index=False,
)
threshold.to_csv(
    TMP_OUT / "stage18_extended_threshold_summary.csv",
    index=False,
)

keep95 = final[
    np.isclose(final["keep_ratio"], 0.95)
].iloc[0]

claims = f"""STAGE 18 FINAL SCIENTIFIC CLAIMS

Primary high-fidelity result
----------------------------
Official ASVD on Llama-2-7B at the nominal 0.95 setting retains
{100 * keep95['actual_parameter_keep_ratio']:.4f}% of parameters and
weights, corresponding to a
{100 * keep95['parameter_reduction']:.4f}% reduction.

On the authoritative held-out theory evaluation frame
(n={int(keep95['n_eval'])}), accuracy changes by
{keep95['acc_delta_pct']:+.2f} percentage points
(95% stratified bootstrap CI:
[{keep95['acc_delta_ci_low']:+.2f},
 {keep95['acc_delta_ci_high']:+.2f}]).

Despite this near-accuracy-neutral aggregate result, prediction
flips remain at {keep95['pred_flip_pct']:.2f}%
(95% CI:
[{keep95['pred_flip_ci_low']:.2f},
 {keep95['pred_flip_ci_high']:.2f}]),
and HiddenChurn remains at
{keep95['hidden_churn_pct']:.2f}%
(95% CI:
[{keep95['hidden_churn_ci_low']:.2f},
 {keep95['hidden_churn_ci_high']:.2f}]).

Compute budget
--------------
The same configuration reduces transformer-block projection/MLP
linear MACs by
{100 * keep95['block_linear_macs_per_token_reduction']:.2f}%,
and all Linear MACs including the uncompressed LM head by
{100 * keep95['total_linear_macs_per_token_reduction']:.2f}%.

Under the stated standard MAC proxy, estimated reductions are
{100 * keep95['standard_mac_proxy_s512_reduction']:.2f}% at S=512,
{100 * keep95['standard_mac_proxy_s2048_reduction']:.2f}% at S=2048,
and {100 * keep95['standard_mac_proxy_s4096_reduction']:.2f}% at S=4096.

Robustness
----------
On the complete 20,465-sample frame, the corresponding values are
DeltaAcc={keep95['full_sample_acc_delta']:+.2f}pp,
PredFlip={keep95['full_sample_pred_flip']:.2f}%,
and HiddenChurn={keep95['full_sample_hidden_churn']:.2f}%.

Scope
-----
These high-fidelity checkpoints establish the existence of
substantial decision churn under official high-retention ASVD
compression. The tightest <=1pp and <=2pp evidence currently uses
one model family and one compression method, and should not be
presented as a universal cross-model or cross-method theorem.
"""

(TMP_OUT / "stage18_final_claims.txt").write_text(
    claims
)

latex = rf"""
% Stage 18 paper-ready paragraph.
At the highest-retention operating point, the official ASVD
checkpoint retains {100 * keep95['actual_parameter_keep_ratio']:.2f}\%
of the parameters. Accuracy changes by only
${keep95['acc_delta_pct']:+.2f}$ percentage points
(95\% stratified bootstrap CI:
$[{keep95['acc_delta_ci_low']:+.2f},
{keep95['acc_delta_ci_high']:+.2f}]$), while
\textsc{{PredFlip}} remains
${keep95['pred_flip_pct']:.2f}\%$
($[{keep95['pred_flip_ci_low']:.2f},
{keep95['pred_flip_ci_high']:.2f}]$)
and \textsc{{HiddenChurn}} remains
${keep95['hidden_churn_pct']:.2f}\%$
($[{keep95['hidden_churn_ci_low']:.2f},
{keep95['hidden_churn_ci_high']:.2f}]$).
This operating point reduces transformer-block linear MACs by
{100 * keep95['block_linear_macs_per_token_reduction']:.2f}\%;
the standard MAC proxy gives a
{100 * keep95['standard_mac_proxy_s2048_reduction']:.2f}\%
reduction at sequence length 2048.
"""

(TMP_OUT / "stage18_paper_ready_snippet.tex").write_text(
    latex.strip() + "\n"
)

provenance = {
    "sources": {
        str(path.relative_to(ROOT)): {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in SOURCE_PATHS
    },
    "authoritative_eval_rows_per_config": 3444,
    "complete_eval_rows_per_config": 20465,
    "bootstrap_replicates": 5000,
    "bootstrap_seed": 20260716,
    "status": "PASS",
}

(TMP_OUT / "stage18_final_provenance.json").write_text(
    json.dumps(provenance, indent=2) + "\n"
)

validation = f"""STAGE18_FINAL_EVIDENCE_PACKAGE_PASS
N_CONFIGS={len(final)}
MAX_FLIP_IDENTITY_ERROR={float(flip_error.max())}
MAX_HIDDEN_IDENTITY_ERROR={float(hidden_error.max())}
DOSE_RESPONSE_PREDFLIP_PASS
DOSE_RESPONSE_HIDDEN_CHURN_PASS
DOSE_RESPONSE_PROB_JS_PASS
POSITIVE_PREDFLIP_CI_PASS
POSITIVE_HIDDEN_CHURN_CI_PASS
SOURCE_HASH_COUNT={len(SOURCE_PATHS)}
NO_GPU_USED
"""

(TMP_OUT / "stage18_final_validation.txt").write_text(
    validation
)

TMP_OUT.rename(FINAL_OUT)

print("===== STAGE18 FINAL EVIDENCE TABLE =====")
print(
    final[[
        "keep_ratio",
        "actual_parameter_keep_ratio",
        "parameter_reduction",
        "block_linear_macs_per_token_reduction",
        "standard_mac_proxy_s2048_reduction",
        "acc_delta_pct",
        "acc_delta_ci_low",
        "acc_delta_ci_high",
        "pred_flip_pct",
        "pred_flip_ci_low",
        "pred_flip_ci_high",
        "hidden_churn_pct",
        "hidden_churn_ci_low",
        "hidden_churn_ci_high",
    ]].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print()
print("WROTE:", FINAL_OUT)
print("FINAL_EVIDENCE_TABLE_PASS")
print("FINAL_CLAIMS_PASS")
print("FINAL_PROVENANCE_PASS")
print("STAGE18_STEP32_FINAL_EVIDENCE_PACKAGE_COMPLETE")
