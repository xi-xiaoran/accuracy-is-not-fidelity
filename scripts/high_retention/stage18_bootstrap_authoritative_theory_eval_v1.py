from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

ROOT = Path(".").resolve()

FRAME_ROOT = (
    ROOT
    / "results/stage18_official_asvd_hifi_v1"
    / "theory_eval_v1"
)

SUMMARY_CSV = (
    FRAME_ROOT
    / "stage18_authoritative_theory_eval_config_metrics.csv"
)

FINAL_OUT = (
    ROOT
    / "results/stage18_official_asvd_hifi_v1"
    / "theory_eval_bootstrap_v1"
)

TMP_OUT = FINAL_OUT.with_name(
    f".theory_eval_bootstrap_v1.tmp.{os.getpid()}"
)

N_BOOT = 5000
CHUNK = 100
SEED = 20260716

assert not FINAL_OUT.exists()
assert not TMP_OUT.exists()
TMP_OUT.mkdir(parents=True)

summary = pd.read_csv(SUMMARY_CSV)

assert len(summary) == 3
assert summary["target_id"].is_unique

rng = np.random.default_rng(SEED)

metric_names = [
    "dense_acc_pct",
    "compressed_acc_pct",
    "acc_delta_pct",
    "pred_flip_pct",
    "hidden_churn_pct",
    "damage_pct",
    "repair_pct",
    "wrong_to_wrong_pct",
]

all_rows = []
validation_rows = []

for config in summary.to_dict("records"):
    target_id = config["target_id"]

    frame_path = (
        FRAME_ROOT
        / f"{target_id}_sample_theory_frame.csv"
    )

    frame = pd.read_csv(frame_path)
    frame = frame[
        frame["split"].astype(str) == "eval"
    ].copy()

    assert len(frame) == 3444
    assert frame["task"].nunique() == 7

    task_arrays = []

    for task, group in frame.groupby(
        "task",
        sort=True,
    ):
        task_arrays.append({
            "task": task,
            "dense": group[
                "dense_correct"
            ].astype(float).to_numpy(),
            "compressed": group[
                "compressed_correct"
            ].astype(float).to_numpy(),
            "flip": group[
                "flip"
            ].astype(float).to_numpy(),
            "damage": group[
                "damage"
            ].astype(float).to_numpy(),
            "repair": group[
                "repair"
            ].astype(float).to_numpy(),
            "wrong": group[
                "wrong_to_wrong"
            ].astype(float).to_numpy(),
        })

    bootstrap = {
        name: np.empty(N_BOOT, dtype=np.float64)
        for name in metric_names
    }

    position = 0

    while position < N_BOOT:
        batch = min(CHUNK, N_BOOT - position)

        dense_sum = np.zeros(batch)
        compressed_sum = np.zeros(batch)
        flip_sum = np.zeros(batch)
        damage_sum = np.zeros(batch)
        repair_sum = np.zeros(batch)
        wrong_sum = np.zeros(batch)
        total = 0

        for item in task_arrays:
            n = len(item["dense"])
            indexes = rng.integers(
                0,
                n,
                size=(batch, n),
            )

            dense_sum += item["dense"][indexes].sum(
                axis=1
            )
            compressed_sum += item[
                "compressed"
            ][indexes].sum(axis=1)
            flip_sum += item["flip"][indexes].sum(
                axis=1
            )
            damage_sum += item[
                "damage"
            ][indexes].sum(axis=1)
            repair_sum += item[
                "repair"
            ][indexes].sum(axis=1)
            wrong_sum += item["wrong"][indexes].sum(
                axis=1
            )

            total += n

        dense = dense_sum / total
        compressed = compressed_sum / total
        delta = compressed - dense
        flip = flip_sum / total
        hidden = flip - np.abs(delta)

        sl = slice(position, position + batch)

        bootstrap["dense_acc_pct"][sl] = (
            dense * 100
        )
        bootstrap["compressed_acc_pct"][sl] = (
            compressed * 100
        )
        bootstrap["acc_delta_pct"][sl] = (
            delta * 100
        )
        bootstrap["pred_flip_pct"][sl] = (
            flip * 100
        )
        bootstrap["hidden_churn_pct"][sl] = (
            hidden * 100
        )
        bootstrap["damage_pct"][sl] = (
            damage_sum / total * 100
        )
        bootstrap["repair_pct"][sl] = (
            repair_sum / total * 100
        )
        bootstrap["wrong_to_wrong_pct"][sl] = (
            wrong_sum / total * 100
        )

        position += batch

    point = {
        "dense_acc_pct":
            frame["dense_correct"].mean() * 100,
        "compressed_acc_pct":
            frame["compressed_correct"].mean() * 100,
        "acc_delta_pct":
            (
                frame["compressed_correct"].mean()
                - frame["dense_correct"].mean()
            ) * 100,
        "pred_flip_pct":
            frame["flip"].mean() * 100,
        "hidden_churn_pct":
            (
                frame["flip"].mean()
                - abs(
                    frame["compressed_correct"].mean()
                    - frame["dense_correct"].mean()
                )
            ) * 100,
        "damage_pct":
            frame["damage"].mean() * 100,
        "repair_pct":
            frame["repair"].mean() * 100,
        "wrong_to_wrong_pct":
            frame["wrong_to_wrong"].mean() * 100,
    }

    residuals = {
        metric: abs(
            point[metric] - float(config[metric])
        )
        for metric in metric_names
    }

    assert max(residuals.values()) < 1e-10

    for metric in metric_names:
        values = bootstrap[metric]
        low, high = np.quantile(
            values,
            [0.025, 0.975],
        )

        row = {
            "target_id": target_id,
            "keep_ratio": config["keep_ratio"],
            "n_eval": len(frame),
            "metric": metric,
            "point_estimate": point[metric],
            "bootstrap_mean": values.mean(),
            "bootstrap_se": values.std(ddof=1),
            "ci_2p5": low,
            "ci_97p5": high,
            "p_le_zero":
                float(np.mean(values <= 0)),
            "n_bootstrap": N_BOOT,
            "seed": SEED,
        }

        all_rows.append(row)

    validation_rows.append({
        "target_id": target_id,
        "n_eval": len(frame),
        "n_tasks": frame["task"].nunique(),
        "max_point_residual":
            max(residuals.values()),
    })

results = pd.DataFrame(all_rows)
validation = pd.DataFrame(validation_rows)

results.to_csv(
    TMP_OUT
    / "stage18_authoritative_sample_bootstrap_ci.csv",
    index=False,
)

validation.to_csv(
    TMP_OUT / "bootstrap_validation.csv",
    index=False,
)

manifest = {
    "n_configs": 3,
    "n_eval_per_config": 3444,
    "n_bootstrap": N_BOOT,
    "seed": SEED,
    "sampling": (
        "stratified sample bootstrap within each task, "
        "preserving original task sample counts"
    ),
    "max_point_residual": float(
        validation["max_point_residual"].max()
    ),
    "status": "PASS",
}

(TMP_OUT / "bootstrap_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)

TMP_OUT.rename(FINAL_OUT)

show = results[
    results["metric"].isin([
        "acc_delta_pct",
        "pred_flip_pct",
        "hidden_churn_pct",
    ])
].copy()

print("===== AUTHORITATIVE SAMPLE BOOTSTRAP =====")
print(
    show[[
        "keep_ratio",
        "metric",
        "point_estimate",
        "ci_2p5",
        "ci_97p5",
        "p_le_zero",
    ]].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print()
print("WROTE:", FINAL_OUT)
print("N_BOOTSTRAP:", N_BOOT)
print("POINT_ESTIMATE_REPRODUCTION_PASS")
print("STRATIFIED_SAMPLE_BOOTSTRAP_PASS")
print("STAGE18_STEP29_THEORY_EVAL_BOOTSTRAP_COMPLETE")
