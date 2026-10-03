from pathlib import Path
import json
import math
import os

import numpy as np
import pandas as pd

from scripts.parsing.build_fullzoo_lowrank_pair_metrics_tables import (
    TASKS,
    COMP_CSV,
    DENSE_CSV,
    load_task_samples,
    sample_key,
    norm_char_scores,
    extract_gold,
    softmax,
    argmax,
)
from scripts.metrics.analyze_theory_predictions_v1 import sample_frame
from scripts.metrics.aggregate_theory_predictions_fullzoo_v1 import base_from_sample

ROOT = Path(".").resolve()

FINAL_OUT = (
    ROOT
    / "results/stage18_official_asvd_hifi_v1"
    / "theory_eval_v1"
)
TMP_OUT = FINAL_OUT.with_name(
    f".theory_eval_v1.tmp.{os.getpid()}"
)

AUTHORITATIVE = (
    ROOT
    / "results/coauthor_complete_aggregate_data_20260714_v3"
    / "07_additional_robustness/p3_ci_and_coverage"
    / "config_manifest_metrics.csv"
)

OLD_FRAME = (
    ROOT
    / "results/theory_predictions_v1"
    / "llama2_7b_asvd_keep0p8_sample_theory_frame.csv"
)

CAL_PER_TASK = 128
EVAL_PER_TASK = 512
EPS = 1e-12

assert not FINAL_OUT.exists()
assert not TMP_OUT.exists()
TMP_OUT.mkdir(parents=True)

dense_targets = {
    row["model_key"]: row
    for row in pd.read_csv(DENSE_CSV).to_dict("records")
}

comp_targets = pd.read_csv(COMP_CSV).to_dict("records")

dense_root = Path(
    dense_targets["llama2_7b"]["output_dir"]
)

old_comp = [
    row for row in comp_targets
    if row["target_id"] == "llama2_7b_asvd_keep0p8"
]

assert len(old_comp) == 1
old_comp_root = Path(old_comp[0]["output_dir"])

models = [
    {
        "target_id": "llama2_7b_asvd_official_keep0p85",
        "keep_ratio": 0.85,
        "compressed_root": (
            ROOT
            / "results/official_lmeval"
            / "stage18_official_asvd_hifi_v1/full/asvd85"
        ),
    },
    {
        "target_id": "llama2_7b_asvd_official_keep0p90",
        "keep_ratio": 0.90,
        "compressed_root": (
            ROOT
            / "results/official_lmeval"
            / "stage18_official_asvd_hifi_v1/full/asvd90"
        ),
    },
    {
        "target_id": "llama2_7b_asvd_official_keep0p95",
        "keep_ratio": 0.95,
        "compressed_root": (
            ROOT
            / "results/official_lmeval"
            / "stage18_official_asvd_hifi_v1/full/asvd95"
        ),
    },
]


def make_rows_from_dirs(compressed_root):
    task_to_id = {
        task: index
        for index, task in enumerate(TASKS)
    }

    rows = []

    for task in TASKS:
        _, dense_rows, _ = load_task_samples(
            dense_root,
            task,
        )

        _, compressed_rows, _ = load_task_samples(
            compressed_root,
            task,
        )

        dense_map = {
            sample_key(sample, index): sample
            for index, sample in enumerate(dense_rows)
        }

        compressed_map = {
            sample_key(sample, index): sample
            for index, sample in enumerate(compressed_rows)
        }

        keys = sorted(
            set(dense_map) & set(compressed_map)
        )

        task_rows = []

        for local_index, key in enumerate(keys):
            dense_sample = dense_map[key]
            compressed_sample = compressed_map[key]

            try:
                dense_scores = norm_char_scores(
                    dense_sample
                )
                compressed_scores = norm_char_scores(
                    compressed_sample
                )
            except Exception:
                continue

            if (
                len(dense_scores)
                != len(compressed_scores)
                or len(dense_scores) < 2
            ):
                continue

            gold = extract_gold(
                dense_sample,
                len(dense_scores),
            )

            if (
                gold is None
                or not 0 <= gold < len(dense_scores)
            ):
                continue

            compressed_probs = softmax(
                compressed_scores
            )

            sorted_probs = sorted(
                compressed_probs,
                reverse=True,
            )

            margin = (
                sorted_probs[0]
                - sorted_probs[1]
            )

            entropy = -sum(
                float(value)
                * math.log(max(float(value), EPS))
                for value in compressed_probs
            )

            normalized_entropy = (
                entropy
                / math.log(len(compressed_probs))
            )

            risk = (
                0.5 * (1.0 - margin)
                + 0.5 * normalized_entropy
            )
            risk = float(
                max(0.0, min(1.0, risk))
            )

            task_rows.append({
                "task": task,
                "task_id": task_to_id[task],
                "sample_key": key,
                "doc_id": dense_sample.get(
                    "doc_id",
                    local_index,
                ),
                "gold": gold,
                "dense_scores": dense_scores,
                "compressed_scores": compressed_scores,
                "dense_pred": argmax(dense_scores),
                "compressed_pred": argmax(
                    compressed_scores
                ),
                "compressed_probs": compressed_probs,
                "risk": risk,
                "n_choices": len(compressed_scores),
            })

        task_rows = sorted(
            task_rows,
            key=lambda row: str(row["doc_id"]),
        )

        calibration = task_rows[:CAL_PER_TASK]
        evaluation = task_rows[
            CAL_PER_TASK:
            CAL_PER_TASK + EVAL_PER_TASK
        ]

        for row in calibration:
            row["split"] = "cal"

        for row in evaluation:
            row["split"] = "eval"

        rows.extend(calibration)
        rows.extend(evaluation)

        print(
            f"TASK {task:13s} "
            f"total={len(task_rows):5d} "
            f"cal={len(calibration):3d} "
            f"eval={len(evaluation):3d}"
        )

    return rows


def validate_frame(frame):
    assert len(frame) == 4340
    assert int((frame["split"] == "cal").sum()) == 896
    assert int((frame["split"] == "eval").sum()) == 3444

    expected = {
        "arc_challenge": (128, 512),
        "arc_easy": (128, 512),
        "boolq": (128, 512),
        "hellaswag": (128, 512),
        "openbookqa": (128, 372),
        "piqa": (128, 512),
        "winogrande": (128, 512),
    }

    for task, (n_cal, n_eval) in expected.items():
        group = frame[frame["task"] == task]
        assert int((group["split"] == "cal").sum()) == n_cal
        assert int((group["split"] == "eval").sum()) == n_eval


print("===== OLD KEEP0P8 EXACT REPRODUCTION =====")

old_generated = sample_frame(
    make_rows_from_dirs(old_comp_root)
)
old_reference = pd.read_csv(OLD_FRAME)

validate_frame(old_generated)

assert list(old_generated.columns) == list(
    old_reference.columns
)
assert old_generated.shape == old_reference.shape

exact_columns = [
    "task",
    "split",
    "sample_key",
    "gold",
    "n_choices",
    "dense_pred",
    "compressed_pred",
    "dense_correct",
    "compressed_correct",
    "flip",
    "damage",
    "repair",
    "wrong_to_wrong",
    "correct_to_correct",
]

for column in exact_columns:
    left = old_generated[column].astype(str)
    right = old_reference[column].astype(str)
    assert left.tolist() == right.tolist(), column

numeric_columns = [
    column for column in old_generated.columns
    if column not in exact_columns
]

max_frame_residual = 0.0

for column in numeric_columns:
    left = pd.to_numeric(
        old_generated[column],
        errors="coerce",
    ).to_numpy(dtype=float)

    right = pd.to_numeric(
        old_reference[column],
        errors="coerce",
    ).to_numpy(dtype=float)

    assert np.allclose(
        left,
        right,
        rtol=1e-11,
        atol=1e-12,
        equal_nan=True,
    ), column

    finite = np.isfinite(left) & np.isfinite(right)

    if finite.any():
        max_frame_residual = max(
            max_frame_residual,
            float(
                np.max(
                    np.abs(left[finite] - right[finite])
                )
            ),
        )

old_summary = base_from_sample(old_generated)

source = pd.read_csv(AUTHORITATIVE)
source_old = source[
    source["target_id"]
    == "llama2_7b_asvd_keep0p8"
]

assert len(source_old) == 1
source_old = source_old.iloc[0]

assert int(old_summary["n_eval"]) == 3444

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

summary_residuals = {}

for column in metric_columns:
    residual = abs(
        float(old_summary[column])
        - float(source_old[column])
    )
    summary_residuals[column] = residual

assert max(summary_residuals.values()) < 1e-10, (
    summary_residuals
)

print(
    "MAX_FRAME_NUMERIC_RESIDUAL:",
    max_frame_residual,
)
print(
    "MAX_AUTHORITATIVE_METRIC_RESIDUAL:",
    max(summary_residuals.values()),
)
print("OLD_KEEP0P8_EXACT_REPRODUCTION_PASS")

print()
print("===== BUILDING NEW AUTHORITATIVE FRAMES =====")

new_rows = []

method_category = source_old["method_category"]

for model in models:
    print()
    print("TARGET:", model["target_id"])

    frame = sample_frame(
        make_rows_from_dirs(
            model["compressed_root"]
        )
    )

    validate_frame(frame)

    frame_path = (
        TMP_OUT
        / f"{model['target_id']}_sample_theory_frame.csv"
    )
    frame.to_csv(frame_path, index=False)

    summary = base_from_sample(frame)
    assert int(summary["n_eval"]) == 3444

    metadata = {
        "target_id": model["target_id"],
        "model_key": "llama2_7b",
        "method_key": "asvd",
        "keep_ratio": model["keep_ratio"],
        "method_category": method_category,
    }

    row = {}

    for column in source.columns:
        if column in metadata:
            row[column] = metadata[column]
        elif column in summary:
            row[column] = summary[column]
        else:
            raise RuntimeError(
                f"Unmapped authoritative column: {column}"
            )

    new_rows.append(row)

    print(
        f"SUMMARY keep={model['keep_ratio']:.2f} "
        f"n_eval={summary['n_eval']} "
        f"delta={summary['acc_delta_pct']:+.6f}pp "
        f"flip={summary['pred_flip_pct']:.6f}% "
        f"hidden={summary['hidden_churn_pct']:.6f}%"
    )

new_df = pd.DataFrame(
    new_rows,
    columns=source.columns,
).sort_values("keep_ratio")

assert len(new_df) == 3
assert new_df["target_id"].nunique() == 3

flip_identity = (
    new_df["pred_flip_pct"]
    - new_df["damage_pct"]
    - new_df["repair_pct"]
    - new_df["wrong_to_wrong_pct"]
).abs()

hidden_identity = (
    new_df["hidden_churn_pct"]
    - (
        new_df["pred_flip_pct"]
        - new_df["acc_delta_pct"].abs()
    )
).abs()

assert float(flip_identity.max()) < 1e-10
assert float(hidden_identity.max()) < 1e-10

metrics_path = (
    TMP_OUT
    / "stage18_authoritative_theory_eval_config_metrics.csv"
)
new_df.to_csv(metrics_path, index=False)

validation = {
    "old_target_reproduced": (
        "llama2_7b_asvd_keep0p8"
    ),
    "old_frame_rows": len(old_generated),
    "old_eval_rows": int(
        (old_generated["split"] == "eval").sum()
    ),
    "max_frame_numeric_residual": (
        max_frame_residual
    ),
    "max_authoritative_metric_residual": (
        max(summary_residuals.values())
    ),
    "new_config_count": len(new_df),
    "new_frame_rows_per_config": 4340,
    "new_eval_rows_per_config": 3444,
    "cal_per_task": CAL_PER_TASK,
    "eval_per_task": EVAL_PER_TASK,
    "max_flip_identity_error": float(
        flip_identity.max()
    ),
    "max_hidden_identity_error": float(
        hidden_identity.max()
    ),
    "status": "PASS",
}

(TMP_OUT / "validation.json").write_text(
    json.dumps(validation, indent=2) + "\n"
)

TMP_OUT.rename(FINAL_OUT)

print()
print("===== AUTHORITATIVE THEORY-EVAL SUMMARY =====")
print(
    new_df[[
        "keep_ratio",
        "dense_acc_pct",
        "compressed_acc_pct",
        "acc_delta_pct",
        "pred_flip_pct",
        "hidden_churn_pct",
        "damage_pct",
        "repair_pct",
        "wrong_to_wrong_pct",
        "prob_js_pct",
    ]].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print()
print("WROTE:", FINAL_OUT)
print("N_CONFIGS:", len(new_df))
print("N_EVAL_PER_CONFIG: 3444")
print("OLD_AUTHORITATIVE_REPRODUCTION_PASS")
print("NEW_AUTHORITATIVE_THEORY_FRAMES_PASS")
print("STAGE18_STEP27_AUTHORITATIVE_REBUILD_COMPLETE")
