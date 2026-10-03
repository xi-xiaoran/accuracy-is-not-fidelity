from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

from scripts.parsing.build_fullzoo_lowrank_pair_metrics_tables import (
    TASKS,
    argmax,
    extract_gold,
    load_task_samples,
    norm_char_scores,
    sample_key,
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

CONFIG_CSV = (
    ROOT
    / "results/stage18_official_asvd_hifi_v1"
    / "aggregate_v1/stage18_asvd_hifi_by_config.csv"
)

FINAL_OUT = (
    ROOT
    / "results/stage18_official_asvd_hifi_v1/bootstrap_pooled_hidden_v1"
)
TMP_OUT = FINAL_OUT.with_name(
    f".bootstrap_pooled_hidden_v1.tmp.{os.getpid()}"
)

models = [
    ("asvd85", 0.85),
    ("asvd90", 0.90),
    ("asvd95", 0.95),
]

expected_n = {
    "arc_challenge": 1172,
    "arc_easy": 2376,
    "boolq": 3270,
    "hellaswag": 10042,
    "openbookqa": 500,
    "piqa": 1838,
    "winogrande": 1267,
}

metrics = [
    "compressed_acc_score",
    "acc_delta_score",
    "pred_flip_score",
    "hidden_churn",
    "damage_rate",
    "repair_rate",
    "both_wrong_flip_rate",
]

N_BOOT = 5000
CHUNK = 100
SEED = 20260715

assert not FINAL_OUT.exists()
assert not TMP_OUT.exists()
TMP_OUT.mkdir(parents=True)

arrays = {
    model_name: {}
    for model_name, _ in models
}

print("===== BUILDING AUTHORITATIVE SAMPLE INDICATORS =====")

for task in TASKS:
    _, dense_rows, _ = load_task_samples(
        DENSE_ROOT / task,
        task,
    )

    dense_map = {
        sample_key(row, i): row
        for i, row in enumerate(dense_rows)
    }

    for model_name, keep_ratio in models:
        _, comp_rows, _ = load_task_samples(
            COMP_ROOT / model_name / task,
            task,
        )

        comp_map = {
            sample_key(row, i): row
            for i, row in enumerate(comp_rows)
        }

        keys = sorted(
            set(dense_map) & set(comp_map)
        )

        assert len(keys) == expected_n[task], (
            model_name,
            task,
            len(keys),
            expected_n[task],
        )

        dense_correct = []
        comp_correct = []
        pred_flip = []
        damage = []
        repair = []
        wrong_wrong_flip = []

        for key in keys:
            dense_sample = dense_map[key]
            comp_sample = comp_map[key]

            dense_scores = norm_char_scores(dense_sample)
            comp_scores = norm_char_scores(comp_sample)

            assert dense_scores is not None
            assert comp_scores is not None
            assert len(dense_scores) == len(comp_scores)

            gold = extract_gold(
                dense_sample,
                len(dense_scores),
            )
            assert gold is not None
            assert 0 <= gold < len(dense_scores)

            dense_pred = argmax(dense_scores)
            comp_pred = argmax(comp_scores)

            dc = dense_pred == gold
            cc = comp_pred == gold
            flip = dense_pred != comp_pred

            dense_correct.append(dc)
            comp_correct.append(cc)
            pred_flip.append(flip)
            damage.append(dc and not cc)
            repair.append((not dc) and cc)
            wrong_wrong_flip.append(
                (not dc) and (not cc) and flip
            )

        arrays[model_name][task] = {
            "dense_correct": np.asarray(
                dense_correct,
                dtype=np.float64,
            ),
            "comp_correct": np.asarray(
                comp_correct,
                dtype=np.float64,
            ),
            "pred_flip": np.asarray(
                pred_flip,
                dtype=np.float64,
            ),
            "damage": np.asarray(
                damage,
                dtype=np.float64,
            ),
            "repair": np.asarray(
                repair,
                dtype=np.float64,
            ),
            "both_wrong_flip": np.asarray(
                wrong_wrong_flip,
                dtype=np.float64,
            ),
        }

    print(
        f"{task}: n={expected_n[task]} "
        f"models={len(models)}"
    )

total_n = sum(expected_n.values())
assert total_n == 20465

def point_estimates(model_name):
    out = {
        metric: 0.0
        for metric in metrics
    }

    for task in TASKS:
        data = arrays[model_name][task]
        weight = len(data["dense_correct"]) / total_n

        dense_acc = data["dense_correct"].mean()
        comp_acc = data["comp_correct"].mean()
        acc_delta = comp_acc - dense_acc
        flip = data["pred_flip"].mean()

        out["compressed_acc_score"] += (
            weight * comp_acc
        )
        out["acc_delta_score"] += (
            weight * acc_delta
        )
        out["pred_flip_score"] += (
            weight * flip
        )

        # HiddenChurn is recomputed after pooling all tasks so that
        # it matches the manuscript definition exactly.

        out["damage_rate"] += (
            weight * data["damage"].mean()
        )
        out["repair_rate"] += (
            weight * data["repair"].mean()
        )
        out["both_wrong_flip_rate"] += (
            weight
            * data["both_wrong_flip"].mean()
        )

    out["hidden_churn"] = (
        out["pred_flip_score"]
        - abs(out["acc_delta_score"])
    )
    return out

points = {
    model_name: point_estimates(model_name)
    for model_name, _ in models
}

# Confirm exact agreement with the already validated aggregate table.
reference = pd.read_csv(CONFIG_CSV).set_index(
    "keep_ratio"
)

point_residuals = {}

for model_name, keep_ratio in models:
    for metric in [
        value for value in metrics
        if value != "hidden_churn"
    ]:
        residual = abs(
            points[model_name][metric]
            - float(reference.loc[keep_ratio, metric])
        )
        point_residuals[
            f"{model_name}:{metric}"
        ] = residual

assert max(point_residuals.values()) < 1e-12, (
    max(point_residuals.values()),
    point_residuals,
)

print()
print("POINT_ESTIMATE_REPRODUCTION_PASS")
print(
    "MAX_POINT_RESIDUAL:",
    max(point_residuals.values()),
)

rng = np.random.default_rng(SEED)

boot = {
    metric: np.zeros(
        (len(models), N_BOOT),
        dtype=np.float64,
    )
    for metric in metrics
}

print()
print("===== STRATIFIED SAMPLE BOOTSTRAP =====")

for start in range(0, N_BOOT, CHUNK):
    stop = min(start + CHUNK, N_BOOT)
    batch = stop - start

    accum = {
        metric: np.zeros(
            (len(models), batch),
            dtype=np.float64,
        )
        for metric in metrics
    }

    for task in TASKS:
        n = expected_n[task]
        weight = n / total_n

        # Resample within task, preserving benchmark composition.
        indexes = rng.integers(
            0,
            n,
            size=(batch, n),
            dtype=np.int32,
        )

        for model_index, (model_name, _) in enumerate(models):
            data = arrays[model_name][task]

            dense_acc = data["dense_correct"][
                indexes
            ].mean(axis=1)

            comp_acc = data["comp_correct"][
                indexes
            ].mean(axis=1)

            acc_delta = comp_acc - dense_acc

            flip = data["pred_flip"][
                indexes
            ].mean(axis=1)

            damage = data["damage"][
                indexes
            ].mean(axis=1)

            repair = data["repair"][
                indexes
            ].mean(axis=1)

            wrong_wrong = data[
                "both_wrong_flip"
            ][indexes].mean(axis=1)

            accum["compressed_acc_score"][
                model_index
            ] += weight * comp_acc

            accum["acc_delta_score"][
                model_index
            ] += weight * acc_delta

            accum["pred_flip_score"][
                model_index
            ] += weight * flip

            # Recomputed after all task contributions are pooled.

            accum["damage_rate"][
                model_index
            ] += weight * damage

            accum["repair_rate"][
                model_index
            ] += weight * repair

            accum["both_wrong_flip_rate"][
                model_index
            ] += weight * wrong_wrong

    accum["hidden_churn"] = (
        accum["pred_flip_score"]
        - np.abs(accum["acc_delta_score"])
    )

    for metric in metrics:
        boot[metric][:, start:stop] = accum[metric]

    if stop % 500 == 0 or stop == N_BOOT:
        print(f"BOOTSTRAP_PROGRESS={stop}/{N_BOOT}")

rows = []

for model_index, (model_name, keep_ratio) in enumerate(models):
    for metric in metrics:
        values = boot[metric][model_index]

        assert np.isfinite(values).all()

        low, high = np.quantile(
            values,
            [0.025, 0.975],
        )

        rows.append({
            "model_name": model_name,
            "keep_ratio": keep_ratio,
            "metric": metric,
            "point_estimate": points[model_name][metric],
            "bootstrap_mean": float(values.mean()),
            "bootstrap_se": float(
                values.std(ddof=1)
            ),
            "ci_2p5": float(low),
            "ci_97p5": float(high),
            "n_bootstrap": N_BOOT,
            "seed": SEED,
            "bootstrap_scheme": (
                "sample_resampling_within_task_"
                "fixed_task_sizes"
            ),
            "p_bootstrap_le_zero": float(
                np.mean(values <= 0)
            ),
        })

ci_df = pd.DataFrame(rows).sort_values(
    ["keep_ratio", "metric"]
)

ci_path = (
    TMP_OUT
    / "stage18_asvd_hifi_sample_bootstrap_ci.csv"
)
ci_df.to_csv(ci_path, index=False)

validation = [
    "STAGE18_ASVD_HIFI_SAMPLE_BOOTSTRAP",
    f"N_BOOTSTRAP={N_BOOT}",
    f"SEED={SEED}",
    f"N_MODELS={len(models)}",
    f"N_TASKS={len(TASKS)}",
    f"N_SAMPLES_PER_CONFIG={total_n}",
    (
        "SCHEME=STRATIFIED_SAMPLE_RESAMPLING_"
        "WITHIN_TASK_FIXED_TASK_SIZES"
    ),
    (
        "HIDDEN_CHURN_AGGREGATION="
        "POOLED_ALL_SAMPLES_"
        "PRED_FLIP_MINUS_ABS_POOLED_ACC_DELTA"
    ),
    (
        "MAX_POINT_REPRODUCTION_RESIDUAL="
        f"{max(point_residuals.values()):.17g}"
    ),
    "VALIDATION=PASS",
]

(TMP_OUT / "bootstrap_validation.txt").write_text(
    "\n".join(validation) + "\n"
)

TMP_OUT.rename(FINAL_OUT)

print()
print("===== KEY 95% BOOTSTRAP INTERVALS =====")

key_metrics = [
    "acc_delta_score",
    "pred_flip_score",
    "hidden_churn",
]

for model_name, keep_ratio in models:
    print()
    print(
        f"{model_name} keep={keep_ratio:.2f}"
    )

    for metric in key_metrics:
        row = ci_df[
            (ci_df["model_name"] == model_name)
            & (ci_df["metric"] == metric)
        ].iloc[0]

        print(
            f"  {metric:16s} "
            f"{row['point_estimate']*100:+.4f}% "
            f"[{row['ci_2p5']*100:+.4f}%, "
            f"{row['ci_97p5']*100:+.4f}%] "
            f"p<=0={row['p_bootstrap_le_zero']:.4f}"
        )

print()
print("WROTE:", FINAL_OUT)
print("N_CI_ROWS:", len(ci_df))
print("SAMPLE_BOOTSTRAP_VALIDATION_PASS")
print("STAGE18_POOLED_HIDDEN_BOOTSTRAP_COMPLETE")
