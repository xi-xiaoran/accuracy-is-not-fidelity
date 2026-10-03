import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
PARSER = argparse.ArgumentParser(
    description="Run representative sample-selection robustness analysis."
)
PARSER.add_argument("output_dir", type=Path, help="New output directory.")
ARGS = PARSER.parse_args()
OUT = ARGS.output_dir.resolve()


from scripts.parsing.build_fullzoo_lowrank_pair_metrics_tables import (
    TASKS,
    load_task_samples,
)
from scripts.tables_and_figures.analyze_fullzoo_decision_churn_v2 import (
    summarize_churn,
)

SEED = 20260720

MANIFEST = (
    ROOT
    / "results/coauthor_complete_aggregate_data_20260714_v3"
    / "07_additional_robustness/p3_ci_and_coverage"
    / "config_manifest_metrics.csv"
)

DENSE_ROOT = (
    ROOT
    / "results/official_lmeval"
    / "fullzoo_lowrank_severity_dense_full"
)

COMP_ROOT = (
    ROOT
    / "results/official_lmeval"
    / "fullzoo_lowrank_severity_compressed_full"
)

TARGET_MODELS = [
    "llama2_7b",
    "qwen3_8b_base",
]

TARGET_KEEPS = [
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
]

OUT.mkdir(parents=True, exist_ok=False)


def task_seed(task):
    digest = hashlib.sha256(
        f"{SEED}:{task}".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16)


def doc_id(row):
    value = row.get("doc_id")

    if value is None:
        raise ValueError("sample row has no doc_id")

    return str(value)


def finite_tree(value):
    if value is None:
        return True

    if isinstance(value, (bool, int, np.integer)):
        return True

    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))

    if isinstance(value, str):
        token = value.strip().lower()

        return token not in {
            "nan",
            "+nan",
            "-nan",
            "inf",
            "+inf",
            "-inf",
            "infinity",
            "+infinity",
            "-infinity",
        }

    if isinstance(value, dict):
        return all(
            finite_tree(item)
            for item in value.values()
        )

    if isinstance(value, (list, tuple)):
        return all(
            finite_tree(item)
            for item in value
        )

    return True


def row_is_valid(row):
    return (
        finite_tree(row.get("resps"))
        and finite_tree(row.get("filtered_resps"))
    )


def make_map(rows):
    mapping = {}

    for row in rows:
        key = doc_id(row)

        if key in mapping:
            raise AssertionError(
                f"duplicate doc_id: {key}"
            )

        mapping[key] = row

    return mapping


def load_source(directory, task):
    sample_file, rows, _ = load_task_samples(
        directory,
        task,
    )

    mapping = make_map(rows)

    valid_ids = {
        key
        for key, row in mapping.items()
        if row_is_valid(row)
    }

    return sample_file, mapping, valid_ids


def summarize_aligned(
    dense_map,
    compressed_map,
    selected_ids,
):
    dense_rows = [
        dense_map[key]
        for key in selected_ids
    ]

    compressed_rows = [
        compressed_map[key]
        for key in selected_ids
    ]

    result = summarize_churn(
        dense_rows,
        compressed_rows,
    )

    required = [
        "n_samples",
        "dense_acc_score",
        "compressed_acc_score",
        "acc_delta_score",
        "pred_flip_score",
        "hidden_churn",
        "damage_rate",
        "repair_rate",
        "both_wrong_flip_rate",
    ]

    missing = [
        key
        for key in required
        if key not in result
    ]

    if missing:
        raise KeyError(
            f"missing summarize_churn keys: {missing}"
        )

    for key in required[1:]:
        if not np.isfinite(float(result[key])):
            raise ValueError(
                f"non-finite metric {key}"
            )

    return result


def weighted(group, column):
    return float(
        np.average(
            group[column],
            weights=group["n_samples"],
        )
    )


manifest = pd.read_csv(MANIFEST)

selection = manifest.loc[
    manifest["keep_ratio"].isin(TARGET_KEEPS)
    & (
        (
            (manifest["model_key"] == "llama2_7b")
            & manifest["method_key"].isin(
                ["asvd", "svdllm_v2"]
            )
        )
        |
        (
            (
                manifest["model_key"]
                == "qwen3_8b_base"
            )
            & manifest["method_key"].isin(
                ["asvd/gqa_fix", "svdllm_v2"]
            )
        )
    )
].copy()

selection = selection.sort_values(
    [
        "model_key",
        "method_key",
        "keep_ratio",
    ]
)

assert len(selection) == 20
assert selection["target_id"].is_unique

for _, group in selection.groupby(
    ["model_key", "method_key"]
):
    assert sorted(
        group["keep_ratio"].tolist()
    ) == TARGET_KEEPS


source_specs = []

for model_key in TARGET_MODELS:
    source_specs.append({
        "role": "dense",
        "model_key": model_key,
        "target_id": "DENSE",
        "directory": DENSE_ROOT / model_key,
    })

for _, row in selection.iterrows():
    source_specs.append({
        "role": "compressed",
        "model_key": row["model_key"],
        "target_id": row["target_id"],
        "directory": COMP_ROOT / row["target_id"],
    })


pool_rows = []
selection_rows = []
task_metric_rows = []

for task in TASKS:
    common_valid_ids = None

    # Pass 1: determine the global common-valid pool across
    # both dense sources and all 20 compressed checkpoints.
    for spec in source_specs:
        sample_file, mapping, valid_ids = load_source(
            spec["directory"],
            task,
        )

        if common_valid_ids is None:
            common_valid_ids = set(valid_ids)
        else:
            common_valid_ids &= valid_ids

        pool_rows.append({
            "task": task,
            "role": spec["role"],
            "model_key": spec["model_key"],
            "target_id": spec["target_id"],
            "sample_file": str(sample_file),
            "raw_rows": len(mapping),
            "finite_rows": len(valid_ids),
        })

    common_ids = sorted(
        common_valid_ids,
        key=str,
    )

    if task == "openbookqa":
        assert len(common_ids) >= 490, (
            task,
            len(common_ids),
        )
    else:
        assert len(common_ids) >= 640, (
            task,
            len(common_ids),
        )

    n_selected = min(640, len(common_ids))

    first_selected = common_ids[:n_selected]

    rng = np.random.default_rng(
        task_seed(task)
    )

    random_positions = rng.choice(
        len(common_ids),
        size=n_selected,
        replace=False,
    )

    random_selected = [
        common_ids[int(position)]
        for position in random_positions
    ]

    protocol_ids = {
        "common_first640": first_selected,
        "common_random640": random_selected,
    }

    # The first 128 selected examples remain the calibration
    # partition; metrics are computed on the held-out remainder.
    protocol_eval_ids = {}

    for protocol, chosen_ids in protocol_ids.items():
        calibration_ids = chosen_ids[:128]
        evaluation_ids = chosen_ids[128:]

        protocol_eval_ids[protocol] = evaluation_ids

        for order, key in enumerate(calibration_ids):
            selection_rows.append({
                "task": task,
                "protocol": protocol,
                "split": "calibration",
                "selection_order": order,
                "doc_id": key,
                "common_pool_size": len(common_ids),
                "task_seed": (
                    task_seed(task)
                    if protocol == "common_random640"
                    else np.nan
                ),
            })

        for order, key in enumerate(evaluation_ids):
            selection_rows.append({
                "task": task,
                "protocol": protocol,
                "split": "evaluation",
                "selection_order": order,
                "doc_id": key,
                "common_pool_size": len(common_ids),
                "task_seed": (
                    task_seed(task)
                    if protocol == "common_random640"
                    else np.nan
                ),
            })

    # Pass 2: reload only for metric computation.
    dense_maps = {}

    for model_key in TARGET_MODELS:
        _, dense_map, dense_valid = load_source(
            DENSE_ROOT / model_key,
            task,
        )

        assert set(common_ids).issubset(
            dense_valid
        )

        dense_maps[model_key] = dense_map

    for _, target in selection.iterrows():
        target_id = target["target_id"]
        model_key = target["model_key"]

        compressed_file, compressed_map, compressed_valid = (
            load_source(
                COMP_ROOT / target_id,
                task,
            )
        )

        assert set(common_ids).issubset(
            compressed_valid
        )

        dense_map = dense_maps[model_key]

        for protocol, evaluation_ids in (
            protocol_eval_ids.items()
        ):
            result = summarize_aligned(
                dense_map,
                compressed_map,
                evaluation_ids,
            )

            task_metric_rows.append({
                "target_id": target_id,
                "model_key": model_key,
                "method_key": target["method_key"],
                "method_category": target[
                    "method_category"
                ],
                "keep_ratio": float(
                    target["keep_ratio"]
                ),
                "task": task,
                "protocol": protocol,
                "common_pool_size": len(
                    common_ids
                ),
                "n_selected_total": n_selected,
                "n_samples": int(
                    result["n_samples"]
                ),
                "dense_acc_pct": (
                    100.0
                    * float(
                        result["dense_acc_score"]
                    )
                ),
                "compressed_acc_pct": (
                    100.0
                    * float(
                        result[
                            "compressed_acc_score"
                        ]
                    )
                ),
                "acc_delta_pct": (
                    100.0
                    * float(
                        result["acc_delta_score"]
                    )
                ),
                "pred_flip_pct": (
                    100.0
                    * float(
                        result["pred_flip_score"]
                    )
                ),
                "hidden_churn_pct": (
                    100.0
                    * float(
                        result["hidden_churn"]
                    )
                ),
                "damage_pct": (
                    100.0
                    * float(
                        result["damage_rate"]
                    )
                ),
                "repair_pct": (
                    100.0
                    * float(
                        result["repair_rate"]
                    )
                ),
                "wrong_to_wrong_pct": (
                    100.0
                    * float(
                        result[
                            "both_wrong_flip_rate"
                        ]
                    )
                ),
                "compressed_sample_file": str(
                    compressed_file
                ),
            })


pool_df = pd.DataFrame(pool_rows)

common_pool_summary = (
    pool_df.groupby("task")
    .agg(
        min_raw_rows=("raw_rows", "min"),
        max_raw_rows=("raw_rows", "max"),
        min_finite_rows=("finite_rows", "min"),
        max_finite_rows=("finite_rows", "max"),
    )
    .reset_index()
)

selected_df = pd.DataFrame(selection_rows)
task_df = pd.DataFrame(task_metric_rows)

config_rows = []

for keys, group in task_df.groupby(
    [
        "target_id",
        "model_key",
        "method_key",
        "method_category",
        "keep_ratio",
        "protocol",
    ],
    sort=True,
):
    (
        target_id,
        model_key,
        method_key,
        method_category,
        keep_ratio,
        protocol,
    ) = keys

    dense_acc = weighted(
        group,
        "dense_acc_pct",
    )

    compressed_acc = weighted(
        group,
        "compressed_acc_pct",
    )

    acc_delta = (
        compressed_acc - dense_acc
    )

    pred_flip = weighted(
        group,
        "pred_flip_pct",
    )

    config_rows.append({
        "target_id": target_id,
        "model_key": model_key,
        "method_key": method_key,
        "method_category": method_category,
        "keep_ratio": keep_ratio,
        "protocol": protocol,
        "n_samples": int(
            group["n_samples"].sum()
        ),
        "dense_acc_pct": dense_acc,
        "compressed_acc_pct": compressed_acc,
        "acc_delta_pct": acc_delta,
        "pred_flip_pct": pred_flip,
        "hidden_churn_pct": (
            pred_flip - abs(acc_delta)
        ),
        "damage_pct": weighted(
            group,
            "damage_pct",
        ),
        "repair_pct": weighted(
            group,
            "repair_pct",
        ),
        "wrong_to_wrong_pct": weighted(
            group,
            "wrong_to_wrong_pct",
        ),
    })

config_df = pd.DataFrame(config_rows)

assert len(config_df) == 40

for column in [
    "acc_delta_pct",
    "pred_flip_pct",
    "hidden_churn_pct",
]:
    assert np.isfinite(
        config_df[column]
    ).all()


# ------------------------------------------------------------
# Common-first versus common-random comparison
# ------------------------------------------------------------

metric_columns = [
    "acc_delta_pct",
    "pred_flip_pct",
    "hidden_churn_pct",
]

wide = config_df.pivot(
    index=[
        "target_id",
        "model_key",
        "method_key",
        "method_category",
        "keep_ratio",
    ],
    columns="protocol",
    values=metric_columns,
)

wide.columns = [
    f"{metric}_{protocol}"
    for metric, protocol in wide.columns
]

wide = wide.reset_index()

for metric in [
    "pred_flip_pct",
    "hidden_churn_pct",
]:
    wide[f"{metric}_absolute_difference_pp"] = (
        wide[f"{metric}_common_random640"]
        - wide[f"{metric}_common_first640"]
    ).abs()


# ------------------------------------------------------------
# Historical authority comparison
# ------------------------------------------------------------

authority_columns = [
    "target_id",
    "acc_delta_pct",
    "pred_flip_pct",
    "hidden_churn_pct",
]

authority = manifest[
    authority_columns
].copy()

authority = authority.rename(
    columns={
        column: f"{column}_historical"
        for column in authority_columns
        if column != "target_id"
    }
)

wide = wide.merge(
    authority,
    on="target_id",
    validate="one_to_one",
)


def comparison_row(
    metric,
    left_name,
    right_name,
    left_column,
    right_column,
):
    left = wide[left_column]
    right = wide[right_column]
    difference = right - left

    return {
        "metric": metric,
        "comparison": (
            f"{left_name}_vs_{right_name}"
        ),
        "n_configurations": len(wide),
        "spearman": float(
            left.corr(
                right,
                method="spearman",
            )
        ),
        "pearson": float(
            left.corr(
                right,
                method="pearson",
            )
        ),
        "mean_absolute_difference_pp": float(
            difference.abs().mean()
        ),
        "median_absolute_difference_pp": float(
            difference.abs().median()
        ),
        "maximum_absolute_difference_pp": float(
            difference.abs().max()
        ),
        "mean_signed_difference_pp": float(
            difference.mean()
        ),
    }


comparison_rows = []

for metric in [
    "pred_flip_pct",
    "hidden_churn_pct",
]:
    comparison_rows.append(
        comparison_row(
            metric,
            "common_first640",
            "common_random640",
            f"{metric}_common_first640",
            f"{metric}_common_random640",
        )
    )

    comparison_rows.append(
        comparison_row(
            metric,
            "historical",
            "common_first640",
            f"{metric}_historical",
            f"{metric}_common_first640",
        )
    )

    comparison_rows.append(
        comparison_row(
            metric,
            "historical",
            "common_random640",
            f"{metric}_historical",
            f"{metric}_common_random640",
        )
    )

comparison_df = pd.DataFrame(
    comparison_rows
)


# ------------------------------------------------------------
# Macro keep-label trends
# ------------------------------------------------------------

macro = (
    config_df.groupby(
        [
            "protocol",
            "keep_ratio",
        ],
        as_index=False,
    )[
        [
            "pred_flip_pct",
            "hidden_churn_pct",
        ]
    ]
    .mean()
    .sort_values(
        [
            "protocol",
            "keep_ratio",
        ]
    )
)

macro_rows = []

for protocol, group in macro.groupby(
    "protocol",
    sort=True,
):
    group = group.sort_values(
        "keep_ratio"
    )

    pred = group[
        "pred_flip_pct"
    ].to_numpy()

    hidden = group[
        "hidden_churn_pct"
    ].to_numpy()

    pred_monotone = bool(
        np.all(np.diff(pred) <= 1e-12)
    )

    hidden_monotone = bool(
        np.all(np.diff(hidden) <= 1e-12)
    )

    for _, row in group.iterrows():
        macro_rows.append({
            **row.to_dict(),
            "predflip_monotone_decreasing":
                pred_monotone,
            "hidden_monotone_decreasing":
                hidden_monotone,
        })

macro_df = pd.DataFrame(
    macro_rows
)


# ------------------------------------------------------------
# Per-model-method series trends
# ------------------------------------------------------------

series_rows = []

for keys, group in config_df.groupby(
    [
        "protocol",
        "model_key",
        "method_key",
    ],
    sort=True,
):
    protocol, model_key, method_key = keys

    group = group.sort_values(
        "keep_ratio"
    )

    pred = group[
        "pred_flip_pct"
    ].to_numpy()

    hidden = group[
        "hidden_churn_pct"
    ].to_numpy()

    series_rows.append({
        "protocol": protocol,
        "model_key": model_key,
        "method_key": method_key,
        "n_checkpoints": len(group),
        "keep_predflip_spearman": float(
            group["keep_ratio"].corr(
                group["pred_flip_pct"],
                method="spearman",
            )
        ),
        "keep_hidden_spearman": float(
            group["keep_ratio"].corr(
                group["hidden_churn_pct"],
                method="spearman",
            )
        ),
        "predflip_monotone_decreasing": bool(
            np.all(
                np.diff(pred) <= 1e-12
            )
        ),
        "hidden_monotone_decreasing": bool(
            np.all(
                np.diff(hidden) <= 1e-12
            )
        ),
    })

series_df = pd.DataFrame(
    series_rows
)


# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

pool_df.to_csv(
    OUT / "source_validity_audit.csv",
    index=False,
)

common_pool_summary.to_csv(
    OUT / "common_pool_summary.csv",
    index=False,
)

selected_df.to_csv(
    OUT / "selected_doc_ids.csv",
    index=False,
)

task_df.to_csv(
    OUT / "metrics_by_task.csv",
    index=False,
)

config_df.to_csv(
    OUT / "metrics_by_configuration.csv",
    index=False,
)

wide.to_csv(
    OUT / "historical_first_random_comparison.csv",
    index=False,
)

comparison_df.to_csv(
    OUT / "protocol_correlation_summary.csv",
    index=False,
)

macro_df.to_csv(
    OUT / "macro_keep_trends.csv",
    index=False,
)

series_df.to_csv(
    OUT / "model_method_series_trends.csv",
    index=False,
)

provenance = {
    "seed": SEED,
    "target_models": TARGET_MODELS,
    "target_keeps": TARGET_KEEPS,
    "n_compressed_checkpoints": 20,
    "selection_protocol": (
        "For each task, intersect finite doc_ids across "
        "both dense sources and all 20 compressed checkpoints. "
        "Compare lexicographic first-640 with fixed-seed "
        "random-640 on this identical common-valid pool. "
        "The first 128 selected examples form calibration; "
        "the remainder form held-out evaluation."
    ),
    "historical_authority": str(
        MANIFEST
    ),
}

with (
    OUT / "provenance.json"
).open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        provenance,
        handle,
        indent=2,
        ensure_ascii=False,
    )

random_macro = macro_df.loc[
    macro_df["protocol"]
    == "common_random640"
]

random_series = series_df.loc[
    series_df["protocol"]
    == "common_random640"
]

with (
    OUT / "summary.txt"
).open(
    "w",
    encoding="utf-8",
) as handle:
    handle.write(
        "STAGE 19B REPRESENTATIVE SAMPLE-SELECTION ROBUSTNESS\n"
    )
    handle.write(
        "=" * 78 + "\n\n"
    )

    handle.write(
        "COMMON VALID POOL SIZES\n"
    )
    handle.write(
        common_pool_summary.to_string(
            index=False
        )
    )

    handle.write(
        "\n\nPROTOCOL CORRELATIONS\n"
    )
    handle.write(
        comparison_df.to_string(
            index=False
        )
    )

    handle.write(
        "\n\nRANDOM-640 MACRO KEEP TRENDS\n"
    )
    handle.write(
        random_macro.to_string(
            index=False
        )
    )

    handle.write(
        "\n\nRANDOM-640 MODEL-METHOD SERIES\n"
    )
    handle.write(
        random_series.to_string(
            index=False
        )
    )

    handle.write("\n")

(OUT / "STAGE19B_COMPLETE").write_text(
    "STAGE19B_REPRESENTATIVE_SAMPLE_SELECTION_COMPLETE\n",
    encoding="utf-8",
)

print("=" * 78)
print(
    "STAGE19B_REPRESENTATIVE_SAMPLE_SELECTION_COMPLETE"
)
print(f"output={OUT}")
print()

print("COMMON VALID POOLS")
print(
    common_pool_summary.to_string(
        index=False
    )
)
print()

print("PROTOCOL CORRELATIONS")
print(
    comparison_df.to_string(
        index=False
    )
)
print()

print("RANDOM-640 MACRO KEEP TRENDS")
print(
    random_macro.to_string(
        index=False
    )
)
print()

print("RANDOM-640 SERIES TRENDS")
print(
    random_series.to_string(
        index=False
    )
)
print("=" * 78)
