import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
PARSER = argparse.ArgumentParser(
    description="Run checkpoint bootstrap and sample-selection robustness analyses."
)
PARSER.add_argument("output_dir", type=Path, help="New output directory.")
ARGS = PARSER.parse_args()
OUT = ARGS.output_dir.resolve()


from scripts.parsing.build_fullzoo_lowrank_pair_metrics_tables import (  # noqa: E402
    TASKS,
    load_task_samples,
)
from scripts.tables_and_figures.analyze_fullzoo_decision_churn_v2 import (  # noqa: E402
    summarize_churn,
)

BOOTSTRAP_SEED = 20260720
SAMPLE_SEED = 20260720
N_BOOTSTRAP = 5000
CALIPERS = [0.25, 0.50, 1.00]
MAIN_CALIPER = 0.50

MANIFEST = (
    ROOT
    / "results/coauthor_complete_aggregate_data_20260714_v3"
    / "07_additional_robustness/p3_ci_and_coverage"
    / "config_manifest_metrics.csv"
)

EXP1_METRICS = (
    ROOT
    / "results/exp1_accuracy_matched_v1_20260718_120738"
    / "checkpoint_metrics_235.csv"
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

TARGET_MODELS = ["llama2_7b", "qwen3_8b_base"]
TARGET_KEEPS = [0.4, 0.5, 0.6, 0.7, 0.8]

OUT.mkdir(parents=True, exist_ok=False)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# ============================================================
# Experiment A: accuracy-matched checkpoint bootstrap
# ============================================================

def build_pairs(frame, caliper):
    rows = []

    for model_key, group in frame.groupby(
        "model_key", sort=True
    ):
        records = group.to_dict("records")

        for a, b in itertools.combinations(records, 2):
            if a["method_category"] == b["method_category"]:
                continue

            gap = abs(
                abs(float(a["acc_delta_pct"]))
                - abs(float(b["acc_delta_pct"]))
            )

            if gap > caliper + 1e-12:
                continue

            ta = str(a["target_id"])
            tb = str(b["target_id"])

            rows.append({
                "model_key": model_key,
                "boot_id_a": str(a["_boot_id"]),
                "boot_id_b": str(b["_boot_id"]),
                "target_a": ta,
                "target_b": tb,
                "accuracy_gap_pp": gap,
                "predflip_gap_pp": abs(
                    float(a["pred_flip_pct"])
                    - float(b["pred_flip_pct"])
                ),
                "hidden_gap_pp": abs(
                    float(a["hidden_churn_pct"])
                    - float(b["hidden_churn_pct"])
                ),
            })

    return pd.DataFrame(rows)


def outcome_blind_one_to_one(pairs):
    selected = []

    if pairs.empty:
        return pairs.copy()

    for model_key, group in pairs.groupby(
        "model_key", sort=True
    ):
        group = group.copy()

        group["_target_lo"] = group[
            ["target_a", "target_b"]
        ].min(axis=1)
        group["_target_hi"] = group[
            ["target_a", "target_b"]
        ].max(axis=1)
        group["_boot_lo"] = group[
            ["boot_id_a", "boot_id_b"]
        ].min(axis=1)
        group["_boot_hi"] = group[
            ["boot_id_a", "boot_id_b"]
        ].max(axis=1)

        # Outcome-blind: no PredFlip or HiddenChurn is used.
        group = group.sort_values([
            "accuracy_gap_pp",
            "_target_lo",
            "_target_hi",
            "_boot_lo",
            "_boot_hi",
        ])

        used = set()

        for _, row in group.iterrows():
            a = row["boot_id_a"]
            b = row["boot_id_b"]

            if a in used or b in used:
                continue

            selected.append(row)
            used.add(a)
            used.add(b)

    if not selected:
        return pd.DataFrame(columns=pairs.columns)

    return pd.DataFrame(selected)[pairs.columns]


def pair_result(pairs):
    if pairs.empty:
        return {
            "n_pairs": 0,
            "predflip_mean_gap_pp": np.nan,
            "hidden_mean_gap_pp": np.nan,
        }

    return {
        "n_pairs": int(len(pairs)),
        "predflip_mean_gap_pp": float(
            pairs["predflip_gap_pp"].mean()
        ),
        "hidden_mean_gap_pp": float(
            pairs["hidden_gap_pp"].mean()
        ),
    }


def bootstrap_accuracy_matching(metrics):
    base = metrics.copy()
    base["_boot_id"] = base["target_id"].astype(str)

    point_rows = []

    for caliper in CALIPERS:
        eligible = build_pairs(base, caliper)
        one = outcome_blind_one_to_one(eligible)

        for name, pairs in [
            ("all_eligible", eligible),
            ("one_to_one", one),
        ]:
            result = pair_result(pairs)
            point_rows.append({
                "analysis": name,
                "caliper_pp": caliper,
                **result,
            })

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    replicate_rows = []

    grouped = [
        (model, group.reset_index(drop=True))
        for model, group in metrics.groupby(
            "model_key", sort=True
        )
    ]

    for replicate in range(N_BOOTSTRAP):
        sampled_groups = []

        for model_key, group in grouped:
            draw = rng.integers(
                0,
                len(group),
                size=len(group),
            )

            sampled = group.iloc[draw].copy()
            sampled["_boot_id"] = [
                (
                    f"{model_key}::rep{replicate}"
                    f"::draw{position}"
                    f"::{target}"
                )
                for position, target in enumerate(
                    sampled["target_id"].astype(str)
                )
            ]

            sampled_groups.append(sampled)

        boot = pd.concat(
            sampled_groups,
            ignore_index=True,
        )

        for caliper in CALIPERS:
            eligible = build_pairs(boot, caliper)
            one = outcome_blind_one_to_one(eligible)

            for name, pairs in [
                ("all_eligible", eligible),
                ("one_to_one", one),
            ]:
                result = pair_result(pairs)

                replicate_rows.append({
                    "replicate": replicate,
                    "analysis": name,
                    "caliper_pp": caliper,
                    **result,
                })

    point = pd.DataFrame(point_rows)
    reps = pd.DataFrame(replicate_rows)

    summary_rows = []

    for _, point_row in point.iterrows():
        subset = reps.loc[
            (reps["analysis"] == point_row["analysis"])
            & np.isclose(
                reps["caliper_pp"],
                point_row["caliper_pp"],
            )
        ]

        row = {
            "analysis": point_row["analysis"],
            "caliper_pp": point_row["caliper_pp"],
            "point_n_pairs": int(
                point_row["n_pairs"]
            ),
            "point_predflip_mean_gap_pp": float(
                point_row["predflip_mean_gap_pp"]
            ),
            "point_hidden_mean_gap_pp": float(
                point_row["hidden_mean_gap_pp"]
            ),
            "n_bootstrap": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
        }

        for metric in [
            "predflip_mean_gap_pp",
            "hidden_mean_gap_pp",
        ]:
            values = subset[metric].dropna()

            row[f"{metric}_boot_mean"] = float(
                values.mean()
            )
            row[f"{metric}_ci_low"] = float(
                values.quantile(0.025)
            )
            row[f"{metric}_ci_high"] = float(
                values.quantile(0.975)
            )
            row[f"{metric}_valid_replicates"] = int(
                len(values)
            )

        row["bootstrap_pair_count_min"] = int(
            subset["n_pairs"].min()
        )
        row["bootstrap_pair_count_median"] = float(
            subset["n_pairs"].median()
        )
        row["bootstrap_pair_count_max"] = int(
            subset["n_pairs"].max()
        )

        summary_rows.append(row)

    return (
        point,
        reps,
        pd.DataFrame(summary_rows),
    )


metrics = pd.read_csv(EXP1_METRICS)

assert len(metrics) == 235
assert metrics["target_id"].is_unique

boot_point, boot_reps, boot_summary = (
    bootstrap_accuracy_matching(metrics)
)

boot_point.to_csv(
    OUT / "accuracy_matched_point_estimates.csv",
    index=False,
)
boot_reps.to_csv(
    OUT / "accuracy_matched_bootstrap_replicates.csv",
    index=False,
)
boot_summary.to_csv(
    OUT / "accuracy_matched_bootstrap_summary.csv",
    index=False,
)


# ============================================================
# Experiment B: random-640 sample-selection robustness
# ============================================================

manifest = pd.read_csv(MANIFEST)

selection = manifest.loc[
    manifest["model_key"].isin(TARGET_MODELS)
    & manifest["keep_ratio"].isin(TARGET_KEEPS)
    & (
        (
            (manifest["model_key"] == "llama2_7b")
            & manifest["method_key"].isin([
                "asvd",
                "svdllm_v2",
            ])
        )
        |
        (
            (
                manifest["model_key"]
                == "qwen3_8b_base"
            )
            & manifest["method_key"].isin([
                "asvd/gqa_fix",
                "svdllm_v2",
            ])
        )
    )
].copy()

selection = selection.sort_values([
    "model_key",
    "method_key",
    "keep_ratio",
])

assert len(selection) == 20, selection[
    ["target_id", "model_key", "method_key", "keep_ratio"]
]
assert selection["target_id"].is_unique

grid = (
    selection.groupby(
        ["model_key", "method_key"]
    )["keep_ratio"]
    .apply(lambda x: sorted(x.tolist()))
)

for keeps in grid:
    assert keeps == TARGET_KEEPS, grid


def task_seed(task):
    text = f"{SAMPLE_SEED}:{task}".encode("utf-8")
    digest = hashlib.sha256(text).hexdigest()
    return int(digest[:8], 16)


def row_identity(row):
    return (
        str(row.get("doc_id", "")),
        str(row.get("target_hash", "")),
    )


def select_positions(n_rows, task, protocol):
    n_select = min(640, n_rows)

    if protocol == "first640":
        chosen = np.arange(n_select, dtype=int)
    elif protocol == "random640":
        rng = np.random.default_rng(task_seed(task))
        chosen = rng.permutation(n_rows)[:n_select]
    else:
        raise ValueError(protocol)

    n_cal = min(128, len(chosen))
    calibration = chosen[:n_cal]
    evaluation = chosen[n_cal:]

    return calibration, evaluation


def subset_rows(rows, positions):
    return [rows[int(i)] for i in positions]


def summarize_subset(dense_rows, comp_rows, positions):
    dense_subset = subset_rows(dense_rows, positions)
    comp_subset = subset_rows(comp_rows, positions)

    result = summarize_churn(
        dense_subset,
        comp_subset,
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
        key for key in required
        if key not in result
    ]
    assert not missing, missing

    return result


task_rows = []
selected_rows = []
source_rows = []

for model_key in TARGET_MODELS:
    model_targets = selection.loc[
        selection["model_key"] == model_key
    ].copy()

    for task in TASKS:
        dense_dir = DENSE_ROOT / model_key
        dense_file, dense_rows, _ = load_task_samples(
            dense_dir,
            task,
        )

        assert len(dense_rows) > 0

        positions = {}

        for protocol in ["first640", "random640"]:
            cal_pos, eval_pos = select_positions(
                len(dense_rows),
                task,
                protocol,
            )

            positions[protocol] = {
                "calibration": cal_pos,
                "evaluation": eval_pos,
            }

            for split_name, values in positions[
                protocol
            ].items():
                for order, position in enumerate(values):
                    row = dense_rows[int(position)]

                    selected_rows.append({
                        "model_key": model_key,
                        "task": task,
                        "protocol": protocol,
                        "split": split_name,
                        "selection_order": order,
                        "row_position": int(position),
                        "doc_id": row.get("doc_id"),
                        "prompt_hash": row.get(
                            "prompt_hash"
                        ),
                        "target_hash": row.get(
                            "target_hash"
                        ),
                        "task_seed": (
                            task_seed(task)
                            if protocol == "random640"
                            else np.nan
                        ),
                    })

        dense_identities = [
            row_identity(row)
            for row in dense_rows
        ]

        dense_stat = dense_file.stat()
        source_rows.append({
            "model_key": model_key,
            "target_id": "DENSE",
            "task": task,
            "role": "dense",
            "path": str(dense_file),
            "size_bytes": dense_stat.st_size,
            "mtime_ns": dense_stat.st_mtime_ns,
            "n_rows": len(dense_rows),
        })

        for _, target in model_targets.iterrows():
            target_id = target["target_id"]
            comp_dir = COMP_ROOT / target_id

            comp_file, comp_rows, _ = load_task_samples(
                comp_dir,
                task,
            )

            assert len(comp_rows) == len(dense_rows), (
                target_id,
                task,
                len(dense_rows),
                len(comp_rows),
            )

            comp_identities = [
                row_identity(row)
                for row in comp_rows
            ]

            assert (
                comp_identities == dense_identities
            ), (target_id, task, "alignment failure")

            comp_stat = comp_file.stat()
            source_rows.append({
                "model_key": model_key,
                "target_id": target_id,
                "task": task,
                "role": "compressed",
                "path": str(comp_file),
                "size_bytes": comp_stat.st_size,
                "mtime_ns": comp_stat.st_mtime_ns,
                "n_rows": len(comp_rows),
            })

            for protocol in [
                "first640",
                "random640",
            ]:
                eval_pos = positions[protocol][
                    "evaluation"
                ]

                result = summarize_subset(
                    dense_rows,
                    comp_rows,
                    eval_pos,
                )

                task_rows.append({
                    "target_id": target_id,
                    "model_key": model_key,
                    "method_key": target[
                        "method_key"
                    ],
                    "method_category": target[
                        "method_category"
                    ],
                    "keep_ratio": float(
                        target["keep_ratio"]
                    ),
                    "task": task,
                    "protocol": protocol,
                    "n_samples": int(
                        result["n_samples"]
                    ),
                    "dense_acc_pct": 100.0 * float(
                        result["dense_acc_score"]
                    ),
                    "compressed_acc_pct":
                        100.0 * float(
                            result[
                                "compressed_acc_score"
                            ]
                        ),
                    "acc_delta_pct": 100.0 * float(
                        result["acc_delta_score"]
                    ),
                    "pred_flip_pct": 100.0 * float(
                        result["pred_flip_score"]
                    ),
                    "hidden_churn_task_pct":
                        100.0 * float(
                            result["hidden_churn"]
                        ),
                    "damage_pct": 100.0 * float(
                        result["damage_rate"]
                    ),
                    "repair_pct": 100.0 * float(
                        result["repair_rate"]
                    ),
                    "wrong_to_wrong_pct":
                        100.0 * float(
                            result[
                                "both_wrong_flip_rate"
                            ]
                        ),
                    "dense_sample_file": str(
                        dense_file
                    ),
                    "compressed_sample_file": str(
                        comp_file
                    ),
                })


task_df = pd.DataFrame(task_rows)
selected_df = pd.DataFrame(selected_rows)
sources_df = pd.DataFrame(source_rows)


def weighted(group, column):
    return float(np.average(
        group[column],
        weights=group["n_samples"],
    ))


config_rows = []

for keys, group in task_df.groupby([
    "target_id",
    "model_key",
    "method_key",
    "method_category",
    "keep_ratio",
    "protocol",
], sort=True):
    (
        target_id,
        model_key,
        method_key,
        method_category,
        keep_ratio,
        protocol,
    ) = keys

    dense_acc = weighted(
        group, "dense_acc_pct"
    )
    compressed_acc = weighted(
        group, "compressed_acc_pct"
    )
    acc_delta = compressed_acc - dense_acc
    pred_flip = weighted(
        group, "pred_flip_pct"
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
            group, "damage_pct"
        ),
        "repair_pct": weighted(
            group, "repair_pct"
        ),
        "wrong_to_wrong_pct": weighted(
            group, "wrong_to_wrong_pct"
        ),
    })

config_df = pd.DataFrame(config_rows)

# ------------------------------------------------------------
# First-640 exact reproduction audit
# ------------------------------------------------------------

old = config_df.loc[
    config_df["protocol"] == "first640"
].copy()

authority_columns = [
    "target_id",
    "acc_delta_pct",
    "pred_flip_pct",
    "hidden_churn_pct",
    "damage_pct",
    "repair_pct",
    "wrong_to_wrong_pct",
]

authority = manifest[
    authority_columns
].copy()

audit = old.merge(
    authority,
    on="target_id",
    suffixes=("_recomputed", "_authority"),
    validate="one_to_one",
)

metric_columns = authority_columns[1:]

for column in metric_columns:
    audit[f"{column}_error"] = (
        audit[f"{column}_recomputed"]
        - audit[f"{column}_authority"]
    )

audit["max_abs_error"] = audit[
    [f"{column}_error" for column in metric_columns]
].abs().max(axis=1)

max_reproduction_error = float(
    audit["max_abs_error"].max()
)

task_df.to_csv(
    OUT / "sample_selection_by_task.csv",
    index=False,
)
config_df.to_csv(
    OUT / "sample_selection_by_config.csv",
    index=False,
)
selected_df.to_csv(
    OUT / "sample_selection_indices.csv",
    index=False,
)
sources_df.to_csv(
    OUT / "sample_selection_source_files.csv",
    index=False,
)
audit.to_csv(
    OUT / "first640_reproduction_audit.csv",
    index=False,
)

assert max_reproduction_error < 1e-8, (
    "Original first-640 reproduction failed",
    max_reproduction_error,
    audit.sort_values(
        "max_abs_error",
        ascending=False,
    ).head(10),
)

# ------------------------------------------------------------
# Old versus random summaries
# ------------------------------------------------------------

wide = config_df.pivot(
    index=[
        "target_id",
        "model_key",
        "method_key",
        "method_category",
        "keep_ratio",
    ],
    columns="protocol",
    values=[
        "acc_delta_pct",
        "pred_flip_pct",
        "hidden_churn_pct",
        "damage_pct",
        "repair_pct",
        "wrong_to_wrong_pct",
    ],
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
    wide[f"{metric}_absolute_change"] = (
        wide[f"{metric}_random640"]
        - wide[f"{metric}_first640"]
    ).abs()

wide.to_csv(
    OUT / "sample_selection_old_vs_random.csv",
    index=False,
)

macro = (
    config_df.groupby(
        ["protocol", "keep_ratio"],
        as_index=False,
    )[
        [
            "pred_flip_pct",
            "hidden_churn_pct",
        ]
    ]
    .mean()
    .sort_values(["protocol", "keep_ratio"])
)

macro_rows = []

for protocol, group in macro.groupby(
    "protocol", sort=True
):
    group = group.sort_values("keep_ratio")

    pred_values = group[
        "pred_flip_pct"
    ].to_numpy()
    hidden_values = group[
        "hidden_churn_pct"
    ].to_numpy()

    pred_monotone = bool(
        np.all(np.diff(pred_values) <= 1e-12)
    )
    hidden_monotone = bool(
        np.all(np.diff(hidden_values) <= 1e-12)
    )

    for _, row in group.iterrows():
        macro_rows.append({
            **row.to_dict(),
            "predflip_monotone_decreasing":
                pred_monotone,
            "hidden_monotone_decreasing":
                hidden_monotone,
        })

macro_df = pd.DataFrame(macro_rows)
macro_df.to_csv(
    OUT / "sample_selection_macro_by_keep.csv",
    index=False,
)

series_rows = []

for keys, group in config_df.groupby([
    "protocol",
    "model_key",
    "method_key",
], sort=True):
    protocol, model_key, method_key = keys
    group = group.sort_values("keep_ratio")

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
                np.diff(
                    group[
                        "pred_flip_pct"
                    ].to_numpy()
                )
                <= 1e-12
            )
        ),
        "hidden_monotone_decreasing": bool(
            np.all(
                np.diff(
                    group[
                        "hidden_churn_pct"
                    ].to_numpy()
                )
                <= 1e-12
            )
        ),
    })

series_df = pd.DataFrame(series_rows)
series_df.to_csv(
    OUT / "sample_selection_series_trends.csv",
    index=False,
)

old_new_summary = []

for metric in [
    "pred_flip_pct",
    "hidden_churn_pct",
]:
    old_values = wide[
        f"{metric}_first640"
    ]
    new_values = wide[
        f"{metric}_random640"
    ]

    old_new_summary.append({
        "metric": metric,
        "n_configurations": len(wide),
        "old_new_spearman": float(
            old_values.corr(
                new_values,
                method="spearman",
            )
        ),
        "mean_absolute_difference_pp": float(
            (new_values - old_values).abs().mean()
        ),
        "median_absolute_difference_pp": float(
            (new_values - old_values).abs().median()
        ),
        "maximum_absolute_difference_pp": float(
            (new_values - old_values).abs().max()
        ),
        "mean_signed_difference_pp": float(
            (new_values - old_values).mean()
        ),
    })

old_new_df = pd.DataFrame(old_new_summary)
old_new_df.to_csv(
    OUT / "sample_selection_protocol_summary.csv",
    index=False,
)

# ============================================================
# Provenance and report
# ============================================================

main_boot = boot_summary.loc[
    np.isclose(
        boot_summary["caliper_pp"],
        MAIN_CALIPER,
    )
].copy()

provenance = {
    "bootstrap_seed": BOOTSTRAP_SEED,
    "sample_selection_seed": SAMPLE_SEED,
    "n_bootstrap": N_BOOTSTRAP,
    "bootstrap_strata": "dense model_key",
    "bootstrap_unit": "compressed checkpoint",
    "primary_caliper_pp": MAIN_CALIPER,
    "sample_protocol": (
        "Within each task, deterministically permute all "
        "validation rows from a task-specific seed; use the "
        "first 128 selected rows as calibration and the next "
        "up to 512 rows as held-out evaluation."
    ),
    "target_models": TARGET_MODELS,
    "target_keeps": TARGET_KEEPS,
    "n_target_checkpoints": int(len(selection)),
    "n_tasks": int(len(TASKS)),
    "max_first640_reproduction_error": (
        max_reproduction_error
    ),
    "manifest": str(MANIFEST),
    "manifest_sha256": sha256(MANIFEST),
    "exp1_metrics": str(EXP1_METRICS),
    "exp1_metrics_sha256": sha256(EXP1_METRICS),
}

with (
    OUT / "stage19_provenance.json"
).open("w", encoding="utf-8") as f:
    json.dump(
        provenance,
        f,
        indent=2,
        ensure_ascii=False,
    )

with (
    OUT / "stage19_summary.txt"
).open("w", encoding="utf-8") as f:
    f.write("STAGE 19 RESULTS\n")
    f.write("=" * 72 + "\n\n")

    f.write(
        "ACCURACY-MATCHED CHECKPOINT BOOTSTRAP\n"
    )

    for _, row in main_boot.sort_values(
        "analysis"
    ).iterrows():
        f.write(
            f"{row['analysis']} c=0.5: "
            f"PredFlip mean gap "
            f"{row['point_predflip_mean_gap_pp']:.6f} pp, "
            f"95% CI "
            f"[{row['predflip_mean_gap_pp_ci_low']:.6f}, "
            f"{row['predflip_mean_gap_pp_ci_high']:.6f}]\n"
        )
        f.write(
            f"{row['analysis']} c=0.5: "
            f"HiddenChurn mean gap "
            f"{row['point_hidden_mean_gap_pp']:.6f} pp, "
            f"95% CI "
            f"[{row['hidden_mean_gap_pp_ci_low']:.6f}, "
            f"{row['hidden_mean_gap_pp_ci_high']:.6f}]\n"
        )

    f.write("\nSAMPLE-SELECTION ROBUSTNESS\n")
    f.write(
        f"First-640 reproduction max error: "
        f"{max_reproduction_error:.12g}\n"
    )
    f.write("\nOld-versus-random protocol summary:\n")
    f.write(old_new_df.to_string(index=False))
    f.write("\n\nMacro keep-label summary:\n")
    f.write(macro_df.to_string(index=False))
    f.write("\n\nSeries trends:\n")
    f.write(series_df.to_string(index=False))
    f.write("\n")

(OUT / "STAGE19_COMPLETE").write_text(
    "STAGE19_BOOTSTRAP_AND_SAMPLE_SELECTION_COMPLETE\n",
    encoding="utf-8",
)

print("=" * 78)
print("STAGE19_BOOTSTRAP_AND_SAMPLE_SELECTION_COMPLETE")
print(f"output={OUT}")
print(
    f"max_first640_reproduction_error="
    f"{max_reproduction_error:.12g}"
)
print()
print("MAIN BOOTSTRAP RESULTS")
print(
    main_boot[
        [
            "analysis",
            "point_predflip_mean_gap_pp",
            "predflip_mean_gap_pp_ci_low",
            "predflip_mean_gap_pp_ci_high",
            "point_hidden_mean_gap_pp",
            "hidden_mean_gap_pp_ci_low",
            "hidden_mean_gap_pp_ci_high",
        ]
    ].to_string(index=False)
)
print()
print("SAMPLE-SELECTION PROTOCOL SUMMARY")
print(old_new_df.to_string(index=False))
print()
print("MACRO KEEP SUMMARY")
print(macro_df.to_string(index=False))
print("=" * 78)
