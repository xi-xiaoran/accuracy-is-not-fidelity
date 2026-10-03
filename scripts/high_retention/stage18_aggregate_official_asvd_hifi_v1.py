from pathlib import Path
import json

import numpy as np
import pandas as pd

from scripts.parsing.build_fullzoo_lowrank_pair_metrics_tables import (
    TASKS,
    group_mean,
    load_task_samples,
    summarize_pair,
)
from scripts.tables_and_figures.analyze_fullzoo_decision_churn_v2 import (
    summarize_churn,
    weighted_mean,
)

ROOT = Path(".").resolve()

DENSE_ROOT = (
    ROOT
    / "results/official_lmeval/fullzoo_keep0p8_full"
    / "dense/llama2_7b"
)

COMP_ROOT = (
    ROOT
    / "results/official_lmeval"
    / "stage18_official_asvd_hifi_v1/full"
)

OUT = (
    ROOT
    / "results/stage18_official_asvd_hifi_v1"
    / "aggregate_v1"
)

OUT.mkdir(parents=True, exist_ok=False)

models = [
    {
        "target_id": "llama2_7b_asvd_official_keep0p85",
        "model_key": "llama2_7b",
        "dense_model_hf": "meta-llama/Llama-2-7b-hf",
        "method_key": "asvd",
        "method_family": "ASVD",
        "keep_tag": "keep0p85",
        "keep_ratio": 0.85,
        "model_name": "asvd85",
        "repo_id": "hahnyuan/Llama-2-7b-hf-asvd85",
    },
    {
        "target_id": "llama2_7b_asvd_official_keep0p90",
        "model_key": "llama2_7b",
        "dense_model_hf": "meta-llama/Llama-2-7b-hf",
        "method_key": "asvd",
        "method_family": "ASVD",
        "keep_tag": "keep0p90",
        "keep_ratio": 0.90,
        "model_name": "asvd90",
        "repo_id": "hahnyuan/Llama-2-7b-hf-asvd90",
    },
    {
        "target_id": "llama2_7b_asvd_official_keep0p95",
        "model_key": "llama2_7b",
        "dense_model_hf": "meta-llama/Llama-2-7b-hf",
        "method_key": "asvd",
        "method_family": "ASVD",
        "keep_tag": "keep0p95",
        "keep_ratio": 0.95,
        "model_name": "asvd95",
        "repo_id": "hahnyuan/Llama-2-7b-hf-asvd95",
    },
]

group_cols = [
    "target_id",
    "model_key",
    "dense_model_hf",
    "method_key",
    "method_family",
    "keep_tag",
    "keep_ratio",
]

churn_cols = [
    "dense_acc_score",
    "compressed_acc_score",
    "acc_delta_score",
    "pred_flip_score",
    "hidden_churn",
    "offsetting_churn",
    "both_correct_same_rate",
    "both_correct_flip_rate",
    "damage_rate",
    "repair_rate",
    "both_wrong_same_rate",
    "both_wrong_flip_rate",
]

task_rows = []

for model in models:
    comp_model_root = COMP_ROOT / model["model_name"]

    for task in TASKS:
        dense_file, dense_rows, _ = load_task_samples(
            DENSE_ROOT / task,
            task,
        )
        comp_file, comp_rows, _ = load_task_samples(
            comp_model_root / task,
            task,
        )

        pair = summarize_pair(dense_rows, comp_rows)
        churn = summarize_churn(dense_rows, comp_rows)

        row = {
            key: model[key]
            for key in group_cols
        }
        row.update({
            "model_name": model["model_name"],
            "repo_id": model["repo_id"],
            "task": task,
            **pair,
            **churn,
            "dense_sample_file": str(dense_file),
            "compressed_sample_file": str(comp_file),
        })

        task_rows.append(row)

task_df = pd.DataFrame(task_rows).sort_values(
    ["keep_ratio", "task"]
)

assert len(task_df) == 21
assert set(task_df["task"]) == set(TASKS)

# Preserve the existing full-zoo convention:
# probability metrics are macro-averaged over seven tasks.
pair_config = pd.DataFrame(
    group_mean(task_rows, group_cols)
)

# Preserve the existing churn convention:
# decision metrics are weighted by task sample count.
churn_config_rows = []

for key, group in task_df.groupby(group_cols, sort=True):
    row = dict(zip(group_cols, key))
    row["n_task_rows"] = len(group)
    row["n_samples"] = int(group["n_samples"].sum())

    for column in churn_cols:
        row[column] = weighted_mean(group, column)

    churn_config_rows.append(row)

churn_config = pd.DataFrame(churn_config_rows)

config_df = pair_config.merge(
    churn_config,
    on=group_cols,
    validate="one_to_one",
)

repo_map = {
    model["target_id"]: model["repo_id"]
    for model in models
}
config_df["repo_id"] = config_df["target_id"].map(repo_map)

config_df = config_df.sort_values("keep_ratio")

# Algebraic consistency checks.
task_flip_residual = (
    task_df["pred_flip_score"]
    - (
        task_df["both_correct_flip_rate"]
        + task_df["damage_rate"]
        + task_df["repair_rate"]
        + task_df["both_wrong_flip_rate"]
    )
).abs()

task_acc_residual = (
    task_df["acc_delta_score"]
    - (
        task_df["repair_rate"]
        - task_df["damage_rate"]
    )
).abs()

task_hidden_residual = (
    task_df["hidden_churn"]
    - (
        task_df["pred_flip_score"]
        - task_df["acc_delta_score"].abs()
    )
).abs()

task_pair_churn_flip_residual = (
    task_df["pred_flip"]
    - task_df["pred_flip_score"]
).abs()

config_flip_residual = (
    config_df["pred_flip_score"]
    - (
        config_df["both_correct_flip_rate"]
        + config_df["damage_rate"]
        + config_df["repair_rate"]
        + config_df["both_wrong_flip_rate"]
    )
).abs()

config_acc_residual = (
    config_df["acc_delta_score"]
    - (
        config_df["repair_rate"]
        - config_df["damage_rate"]
    )
).abs()

config_hidden_residual = (
    config_df["hidden_churn"]
    - (
        config_df["pred_flip_score"]
        - config_df["acc_delta_score"].abs()
    )
).abs()

residuals = {
    "task_flip_identity_max_abs": float(task_flip_residual.max()),
    "task_acc_identity_max_abs": float(task_acc_residual.max()),
    "task_hidden_identity_max_abs": float(task_hidden_residual.max()),
    "task_pair_vs_churn_flip_max_abs": float(
        task_pair_churn_flip_residual.max()
    ),
    "config_flip_identity_max_abs": float(
        config_flip_residual.max()
    ),
    "config_acc_identity_max_abs": float(
        config_acc_residual.max()
    ),
    "config_hidden_identity_max_abs": float(
        config_hidden_residual.max()
    ),
}

# Task-level identities are exact. At config level, hidden_churn is
# aggregated as a sample-weighted mean of task-level hidden_churn.
# Therefore E[flip - abs(delta)] need not equal E[flip] - abs(E[delta]).
strict_residual_keys = [
    "task_flip_identity_max_abs",
    "task_acc_identity_max_abs",
    "task_hidden_identity_max_abs",
    "task_pair_vs_churn_flip_max_abs",
    "config_flip_identity_max_abs",
    "config_acc_identity_max_abs",
]

assert all(
    np.isfinite(residuals[key]) and residuals[key] < 1e-10
    for key in strict_residual_keys
), residuals

assert np.isfinite(
    residuals["config_hidden_identity_max_abs"]
)

task_path = OUT / "stage18_asvd_hifi_by_task.csv"
config_path = OUT / "stage18_asvd_hifi_by_config.csv"
validation_path = OUT / "stage18_asvd_hifi_validation.txt"

task_df.to_csv(task_path, index=False)
config_df.to_csv(config_path, index=False)

validation_lines = [
    "STAGE18_OFFICIAL_ASVD_HIFI_VALIDATION",
    "N_CONFIGS=3",
    "N_TASK_ROWS=21",
    "N_TASKS_PER_CONFIG=7",
    f"N_TOTAL_SAMPLES_PER_CONFIG={int(config_df['n_samples'].iloc[0])}",
]

for key, value in residuals.items():
    validation_lines.append(f"{key}={value:.17g}")

validation_lines.extend([
    "PAIR_CONFIG_AGGREGATION=UNWEIGHTED_TASK_MEAN",
    "CHURN_CONFIG_AGGREGATION=SAMPLE_WEIGHTED_MEAN",
    "CONFIG_HIDDEN_IDENTITY_NOTE=EXPECTED_GAP_DUE_TO_TASKWISE_ABS_NONLINEARITY",
    "VALIDATION=PASS",
])

validation_path.write_text(
    "\n".join(validation_lines) + "\n"
)

display = config_df[[
    "keep_ratio",
    "dense_acc_score",
    "compressed_acc_score",
    "acc_delta_score",
    "pred_flip_score",
    "hidden_churn",
    "damage_rate",
    "repair_rate",
    "both_wrong_flip_rate",
    "prob_js",
]].copy()

for column in [
    "dense_acc_score",
    "compressed_acc_score",
    "acc_delta_score",
    "pred_flip_score",
    "hidden_churn",
    "damage_rate",
    "repair_rate",
    "both_wrong_flip_rate",
]:
    display[column] *= 100.0

print("===== STAGE18 CONFIG SUMMARY =====")
print(display.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print()
print("WROTE:", task_path)
print("WROTE:", config_path)
print("WROTE:", validation_path)
print("N_TASK_ROWS:", len(task_df))
print("N_CONFIG_ROWS:", len(config_df))
print("STAGE18_STEP16_AGGREGATION_PASS")
