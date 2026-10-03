from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

ROOT = Path(".").resolve()

OLD = (
    ROOT
    / "results/coauthor_complete_aggregate_data_20260714_v3"
    / "07_additional_robustness/p3_ci_and_coverage"
    / "config_manifest_metrics.csv"
)

NEW = (
    ROOT
    / "results/stage18_official_asvd_hifi_v1"
    / "theory_eval_v1"
    / "stage18_authoritative_theory_eval_config_metrics.csv"
)

FINAL_OUT = (
    ROOT
    / "results/stage18_official_asvd_hifi_v1"
    / "stage17_extended_v1"
)
TMP_OUT = FINAL_OUT.with_name(
    f".stage17_extended_v1.tmp.{os.getpid()}"
)

assert not FINAL_OUT.exists()
assert not TMP_OUT.exists()
TMP_OUT.mkdir(parents=True)

old = pd.read_csv(OLD)
new = pd.read_csv(NEW)

assert list(old.columns) == list(new.columns)
assert len(old) == 235
assert len(new) == 3
assert old["target_id"].is_unique
assert new["target_id"].is_unique
assert not set(old["target_id"]) & set(new["target_id"])

combined = pd.concat(
    [old, new],
    ignore_index=True,
)

assert len(combined) == 238
assert combined["target_id"].is_unique

combined["abs_acc_delta_pct"] = (
    combined["acc_delta_pct"].abs()
)

flip_error = (
    combined["pred_flip_pct"]
    - combined["damage_pct"]
    - combined["repair_pct"]
    - combined["wrong_to_wrong_pct"]
).abs()

hidden_error = (
    combined["hidden_churn_pct"]
    - (
        combined["pred_flip_pct"]
        - combined["abs_acc_delta_pct"]
    )
).abs()

assert float(flip_error.max()) < 1e-10
assert float(hidden_error.max()) < 1e-10

thresholds = [1, 2, 3, 5, 10]
metric_columns = [
    "dense_acc_pct",
    "compressed_acc_pct",
    "acc_delta_pct",
    "pred_flip_pct",
    "hidden_churn_pct",
    "damage_pct",
    "repair_pct",
    "wrong_to_wrong_pct",
    "prob_js_pct",
]

summary_rows = []

for threshold in thresholds:
    subset = combined[
        combined["abs_acc_delta_pct"] <= threshold
    ].copy()

    row = {
        "threshold_pp": threshold,
        "n_configs": len(subset),
        "n_models": subset["model_key"].nunique(),
        "n_methods": subset["method_key"].nunique(),
        "n_method_categories":
            subset["method_category"].nunique(),
        "n_keep_ratios": subset["keep_ratio"].nunique(),
    }

    for column in metric_columns:
        row[f"mean_{column}"] = subset[column].mean()

    summary_rows.append(row)

    subset.sort_values(
        ["abs_acc_delta_pct", "target_id"]
    ).to_csv(
        TMP_OUT
        / f"hifi_le{threshold}_complete_config_list.csv",
        index=False,
    )

summary = pd.DataFrame(summary_rows)

# Configuration-level percentile bootstrap,
# matching the original Stage 17 convention.
rng = np.random.default_rng(20260714)
n_bootstrap = 5000
ci_rows = []

for threshold in thresholds:
    subset = combined[
        combined["abs_acc_delta_pct"] <= threshold
    ].reset_index(drop=True)

    for metric in [
        "acc_delta_pct",
        "pred_flip_pct",
        "hidden_churn_pct",
    ]:
        point = subset[metric].mean()

        if len(subset) == 1:
            low = high = point
            se = 0.0
        else:
            indexes = rng.integers(
                0,
                len(subset),
                size=(n_bootstrap, len(subset)),
            )
            values = subset[metric].to_numpy()[indexes].mean(
                axis=1
            )
            low, high = np.quantile(
                values,
                [0.025, 0.975],
            )
            se = values.std(ddof=1)

        ci_rows.append({
            "threshold_pp": threshold,
            "n_configs": len(subset),
            "metric": metric,
            "point_estimate": point,
            "bootstrap_se": se,
            "ci_2p5": low,
            "ci_97p5": high,
            "n_bootstrap": n_bootstrap,
            "seed": 20260714,
        })

ci = pd.DataFrame(ci_rows)

old_counts = {
    threshold: int(
        (old["acc_delta_pct"].abs() <= threshold).sum()
    )
    for threshold in thresholds
}
new_counts = {
    threshold: int(
        (
            combined["acc_delta_pct"].abs()
            <= threshold
        ).sum()
    )
    for threshold in thresholds
}

comparison = pd.DataFrame([
    {
        "threshold_pp": threshold,
        "old_n_configs": old_counts[threshold],
        "extended_n_configs": new_counts[threshold],
        "added_configs":
            new_counts[threshold] - old_counts[threshold],
    }
    for threshold in thresholds
])

combined.to_csv(
    TMP_OUT / "extended_config_manifest_metrics.csv",
    index=False,
)
summary.to_csv(
    TMP_OUT / "extended_threshold_summary.csv",
    index=False,
)
ci.to_csv(
    TMP_OUT / "extended_threshold_bootstrap_ci.csv",
    index=False,
)
comparison.to_csv(
    TMP_OUT / "old_vs_extended_threshold_counts.csv",
    index=False,
)

focus = new.loc[np.isclose(new["keep_ratio"], 0.95)]
if len(focus) != 1:
    raise RuntimeError(
        "Expected exactly one newly added keep_ratio=0.95 configuration, "
        f"found {len(focus)}"
    )
focus = focus.iloc[0]

verdict = f"""STAGE18 HIGH-FIDELITY EXTENSION VERDICT

The previous evidence gap at |DeltaAcc| <= 1pp and <= 2pp is closed
by official ASVD checkpoints on Llama-2-7B.

The extension establishes substantial decision churn under genuinely
high-fidelity compression:
- keep {float(focus['keep_ratio']):.2f}: DeltaAcc about {float(focus['acc_delta_pct']):.2f}pp
- PredFlip about {float(focus['pred_flip_pct']):.2f}%
- HiddenChurn about {float(focus['hidden_churn_pct']):.2f}%

Scope limitation:
The newly added tight-threshold configurations share one model and one
compression method. This supports the existence claim under official
high-fidelity ASVD compression, but not a universal cross-method or
cross-model claim.
"""

(TMP_OUT / "scientific_verdict.txt").write_text(
    verdict
)

validation = {
    "old_configs": len(old),
    "new_configs": len(new),
    "extended_configs": len(combined),
    "old_counts": old_counts,
    "extended_counts": new_counts,
    "max_flip_identity_error": float(
        flip_error.max()
    ),
    "max_hidden_identity_error": float(
        hidden_error.max()
    ),
    "status": "PASS",
}

(TMP_OUT / "validation.json").write_text(
    json.dumps(validation, indent=2) + "\n"
)

TMP_OUT.rename(FINAL_OUT)

print("===== OLD VS EXTENDED COUNTS =====")
print(comparison.to_string(index=False))

print()
print("===== EXTENDED THRESHOLD SUMMARY =====")
print(
    summary[[
        "threshold_pp",
        "n_configs",
        "n_models",
        "n_methods",
        "n_keep_ratios",
        "mean_acc_delta_pct",
        "mean_pred_flip_pct",
        "mean_hidden_churn_pct",
    ]].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print()
print("WROTE:", FINAL_OUT)
print("EXTENDED_CONFIGS:", len(combined))
print("METRIC_IDENTITIES_PASS")
print("STAGE18_STEP28_STAGE17_INTEGRATION_COMPLETE")
