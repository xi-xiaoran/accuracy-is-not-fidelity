from pathlib import Path
import hashlib
import os

import numpy as np
import pandas as pd


ROOT = Path(".").resolve()
V3 = ROOT / "results/coauthor_complete_aggregate_data_20260714_v3"

SOURCE = V3 / (
    "07_additional_robustness/p3_ci_and_coverage/"
    "config_manifest_metrics.csv"
)
TABLE1 = V3 / (
    "02_paper_tables_and_controls/manuscript_tables_final/"
    "table1_main_controlled_subsets.csv"
)

OUT = ROOT / "results/stage17_hifi_gap_audit_v1"
TMP = ROOT / (
    f"results/.stage17_hifi_gap_audit_v1_building_{os.getpid()}"
)

if OUT.exists() or OUT.is_symlink():
    raise SystemExit(f"REFUSE: output already exists: {OUT}")

if TMP.exists() or TMP.is_symlink():
    raise SystemExit(f"REFUSE: temporary output already exists: {TMP}")

if not SOURCE.is_file():
    raise SystemExit(f"MISSING SOURCE: {SOURCE}")

if not TABLE1.is_file():
    raise SystemExit(f"MISSING TABLE1: {TABLE1}")


df = pd.read_csv(SOURCE)
table1 = pd.read_csv(TABLE1)

required = [
    "target_id",
    "model_key",
    "method_key",
    "keep_ratio",
    "method_category",
    "dense_acc_pct",
    "compressed_acc_pct",
    "acc_delta_pct",
    "pred_flip_pct",
    "hidden_churn_pct",
    "damage_pct",
    "repair_pct",
    "wrong_to_wrong_pct",
    "prob_js_pct",
    "bar_ge1_flip_pct",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise SystemExit(f"MISSING REQUIRED COLUMNS: {missing}")

if len(df) != 235 or df["target_id"].nunique() != 235:
    raise SystemExit(
        "AUTHORITATIVE SOURCE CARDINALITY FAILURE: "
        f"rows={len(df)}, "
        f"unique_target_id={df['target_id'].nunique()}"
    )

df = df.copy()
df["abs_acc_delta_pct"] = df["acc_delta_pct"].abs()


# ------------------------------------------------------------
# Metric identity validation
# ------------------------------------------------------------

flip_error = (
    df["pred_flip_pct"]
    - df["damage_pct"]
    - df["repair_pct"]
    - df["wrong_to_wrong_pct"]
).abs()

hidden_error = (
    df["hidden_churn_pct"]
    - (
        df["pred_flip_pct"]
        - df["abs_acc_delta_pct"]
    )
).abs()

if float(flip_error.max()) > 1e-8:
    raise SystemExit(
        "FLIP IDENTITY FAILURE: "
        f"max_error={flip_error.max()}"
    )

if float(hidden_error.max()) > 1e-8:
    raise SystemExit(
        "HIDDEN CHURN IDENTITY FAILURE: "
        f"max_error={hidden_error.max()}"
    )


# ------------------------------------------------------------
# Reproduce manuscript Table 1
# ------------------------------------------------------------

def manuscript_summary(frame):
    definitions = [
        (
            "All",
            np.ones(len(frame), dtype=bool),
        ),
        (
            "Keep=0.8",
            np.isclose(frame["keep_ratio"], 0.8),
        ),
        (
            r"$|\Delta\mathrm{Acc.}|\le10$",
            frame["abs_acc_delta_pct"] <= 10,
        ),
        (
            r"$|\Delta\mathrm{Acc.}|\le5$",
            frame["abs_acc_delta_pct"] <= 5,
        ),
        (
            r"Comp. Acc.$\ge50$",
            frame["compressed_acc_pct"] >= 50,
        ),
        (
            r"Comp. Acc.$\ge60$",
            frame["compressed_acc_pct"] >= 60,
        ),
    ]

    rows = []

    for label, mask in definitions:
        sub = frame.loc[mask]

        rows.append(
            {
                "Subset": label,
                "#Cfg": len(sub),
                "Dense Acc.": sub["dense_acc_pct"].mean(),
                "Compressed Acc.":
                    sub["compressed_acc_pct"].mean(),
                r"$\Delta$Acc.":
                    sub["acc_delta_pct"].mean(),
                "PredFlip":
                    sub["pred_flip_pct"].mean(),
                "HiddenChurn":
                    sub["hidden_churn_pct"].mean(),
            }
        )

    return pd.DataFrame(rows)


reproduced = manuscript_summary(df)

if not np.array_equal(
    reproduced["#Cfg"].to_numpy(),
    table1["#Cfg"].to_numpy(),
):
    raise SystemExit(
        "SOURCE REPRODUCTION FAILURE: subset counts differ"
    )

for column in [
    "Dense Acc.",
    "Compressed Acc.",
    r"$\Delta$Acc.",
    "PredFlip",
    "HiddenChurn",
]:
    if not np.allclose(
        reproduced[column].round(2).to_numpy(),
        table1[column].to_numpy(),
        atol=1e-10,
        rtol=0,
    ):
        raise SystemExit(
            f"SOURCE REPRODUCTION FAILURE: {column}"
        )


# ------------------------------------------------------------
# Define controlled subsets
# ------------------------------------------------------------

subset_specs = [
    (
        "all",
        "all",
        np.nan,
        np.ones(len(df), dtype=bool),
    ),
    (
        "keep_eq_0p8",
        "keep_ratio",
        0.8,
        np.isclose(df["keep_ratio"], 0.8),
    ),
]

for threshold in [1, 2, 3, 5, 10]:
    subset_specs.append(
        (
            f"abs_delta_le_{threshold}",
            "abs_acc_delta_pct",
            float(threshold),
            df["abs_acc_delta_pct"] <= threshold,
        )
    )

for threshold in [50, 55, 60, 65]:
    subset_specs.append(
        (
            f"compressed_acc_ge_{threshold}",
            "compressed_acc_pct",
            float(threshold),
            df["compressed_acc_pct"] >= threshold,
        )
    )


metric_columns = [
    "dense_acc_pct",
    "compressed_acc_pct",
    "acc_delta_pct",
    "abs_acc_delta_pct",
    "pred_flip_pct",
    "hidden_churn_pct",
    "damage_pct",
    "repair_pct",
    "wrong_to_wrong_pct",
    "prob_js_pct",
    "bar_ge1_flip_pct",
]


# ------------------------------------------------------------
# Configuration-level bootstrap
# ------------------------------------------------------------

summary_rows = []
ci_rows = []

rng = np.random.default_rng(20260714)
n_bootstrap = 5000

for subset_name, criterion, threshold, mask in subset_specs:
    sub = df.loc[mask].copy()

    base = {
        "subset": subset_name,
        "criterion": criterion,
        "threshold": threshold,
        "n_configs": len(sub),
        "n_models": sub["model_key"].nunique(),
        "n_methods": sub["method_key"].nunique(),
        "n_method_categories":
            sub["method_category"].nunique(),
        "n_keep_ratios":
            sub["keep_ratio"].nunique(),
    }

    row = dict(base)
    ci_row = dict(base)

    for column in metric_columns:
        if len(sub):
            mean_value = sub[column].mean()
        else:
            mean_value = np.nan

        row[f"mean_{column}"] = mean_value
        ci_row[f"mean_{column}"] = mean_value

        if len(sub) >= 2:
            values = sub[column].to_numpy(dtype=float)

            sampled_indices = rng.integers(
                0,
                len(values),
                size=(n_bootstrap, len(values)),
            )

            bootstrap_means = values[
                sampled_indices
            ].mean(axis=1)

            lower, upper = np.quantile(
                bootstrap_means,
                [0.025, 0.975],
            )
        else:
            lower = np.nan
            upper = np.nan

        ci_row[f"ci95_low_{column}"] = lower
        ci_row[f"ci95_high_{column}"] = upper

    summary_rows.append(row)
    ci_rows.append(ci_row)


summary = pd.DataFrame(summary_rows)
summary_with_ci = pd.DataFrame(ci_rows)


# ------------------------------------------------------------
# Complete high-fidelity configuration lists
# ------------------------------------------------------------

listing_columns = [
    "target_id",
    "model_key",
    "method_key",
    "method_category",
    "keep_ratio",
    "dense_acc_pct",
    "compressed_acc_pct",
    "acc_delta_pct",
    "abs_acc_delta_pct",
    "pred_flip_pct",
    "hidden_churn_pct",
    "damage_pct",
    "repair_pct",
    "wrong_to_wrong_pct",
    "prob_js_pct",
    "bar_ge1_flip_pct",
]

le5 = (
    df.loc[
        df["abs_acc_delta_pct"] <= 5,
        listing_columns,
    ]
    .sort_values(
        ["abs_acc_delta_pct", "target_id"]
    )
)

le10 = (
    df.loc[
        df["abs_acc_delta_pct"] <= 10,
        listing_columns,
    ]
    .sort_values(
        ["abs_acc_delta_pct", "target_id"]
    )
)


# ------------------------------------------------------------
# Composition tables
# ------------------------------------------------------------

def composition(frame, column, subset_name):
    result = (
        frame.groupby(column, dropna=False)
        .size()
        .reset_index(name="n_configs")
        .sort_values(
            ["n_configs", column],
            ascending=[False, True],
        )
    )

    result.insert(0, "subset", subset_name)
    result = result.rename(columns={column: "group"})

    return result


model_counts = pd.concat(
    [
        composition(
            le5,
            "model_key",
            "abs_delta_le_5",
        ),
        composition(
            le10,
            "model_key",
            "abs_delta_le_10",
        ),
    ],
    ignore_index=True,
)

method_counts = pd.concat(
    [
        composition(
            le5,
            "method_key",
            "abs_delta_le_5",
        ),
        composition(
            le10,
            "method_key",
            "abs_delta_le_10",
        ),
    ],
    ignore_index=True,
)

keep_counts = pd.concat(
    [
        composition(
            le5,
            "keep_ratio",
            "abs_delta_le_5",
        ),
        composition(
            le10,
            "keep_ratio",
            "abs_delta_le_10",
        ),
    ],
    ignore_index=True,
)


# ------------------------------------------------------------
# Leave-one-group-out sensitivity
# ------------------------------------------------------------

def leave_one_out(frame, group_column, subset_name):
    rows = []
    base_n = len(frame)

    for group, omitted in frame.groupby(
        group_column,
        dropna=False,
    ):
        remaining = frame.loc[
            ~frame.index.isin(omitted.index)
        ]

        remaining_n = len(remaining)

        if remaining_n == 0:
            stability = "NO_REMAINING_CONFIGS"
        elif remaining_n < 5:
            stability = "UNSTABLE_REMAINING_N_LT5"
        else:
            stability = "INTERPRETABLE"

        rows.append(
            {
                "subset": subset_name,
                "group_type": group_column,
                "omitted_group": group,
                "base_n_configs": base_n,
                "omitted_n_configs": len(omitted),
                "remaining_n_configs": remaining_n,
                "remaining_n_models":
                    remaining["model_key"].nunique(),
                "remaining_n_methods":
                    remaining["method_key"].nunique(),
                "remaining_n_keep_ratios":
                    remaining["keep_ratio"].nunique(),
                "mean_pred_flip_pct":
                    remaining["pred_flip_pct"].mean()
                    if remaining_n else np.nan,
                "mean_hidden_churn_pct":
                    remaining["hidden_churn_pct"].mean()
                    if remaining_n else np.nan,
                "stability_flag": stability,
            }
        )

    return pd.DataFrame(rows)


leave_model_out = pd.concat(
    [
        leave_one_out(
            le5,
            "model_key",
            "abs_delta_le_5",
        ),
        leave_one_out(
            le10,
            "model_key",
            "abs_delta_le_10",
        ),
    ],
    ignore_index=True,
)

leave_method_out = pd.concat(
    [
        leave_one_out(
            le5,
            "method_key",
            "abs_delta_le_5",
        ),
        leave_one_out(
            le10,
            "method_key",
            "abs_delta_le_10",
        ),
    ],
    ignore_index=True,
)


# ------------------------------------------------------------
# Write through a temporary directory
# ------------------------------------------------------------

TMP.mkdir(parents=False, exist_ok=False)

summary.to_csv(
    TMP / "config_level_threshold_summary.csv",
    index=False,
)

summary_with_ci.to_csv(
    TMP / "config_level_threshold_summary_with_ci.csv",
    index=False,
)

le5.to_csv(
    TMP / "hifi_le5_complete_config_list.csv",
    index=False,
)

le10.to_csv(
    TMP / "hifi_le10_complete_config_list.csv",
    index=False,
)

model_counts.to_csv(
    TMP / "hifi_subset_model_counts.csv",
    index=False,
)

method_counts.to_csv(
    TMP / "hifi_subset_method_counts.csv",
    index=False,
)

keep_counts.to_csv(
    TMP / "hifi_subset_keep_counts.csv",
    index=False,
)

leave_model_out.to_csv(
    TMP / "leave_one_model_out.csv",
    index=False,
)

leave_method_out.to_csv(
    TMP / "leave_one_method_out.csv",
    index=False,
)


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


provenance = f"""Stage 17 authoritative source provenance

Source CSV:
{SOURCE.relative_to(ROOT)}

Source SHA256:
{sha256(SOURCE)}

Source rows:
{len(df)}

Unique target_id:
{df["target_id"].nunique()}

Manuscript control table:
{TABLE1.relative_to(ROOT)}

Manuscript table SHA256:
{sha256(TABLE1)}

Reproduction:
All six manuscript Table 1 rows reproduce exactly after rounding
to two decimals.

Metric units:
All accuracy, churn, damage, repair, wrong-to-wrong, Prob-JS, and
BAR-conditional flip columns in the source are percentages.

Signed delta:
acc_delta_pct = compressed_acc_pct - dense_acc_pct.

Absolute delta:
abs_acc_delta_pct = abs(acc_delta_pct).

Bootstrap:
95% percentile intervals from {n_bootstrap} configuration-level
bootstrap resamples.

Fixed random seed:
20260714
"""

(TMP / "source_provenance.txt").write_text(
    provenance,
    encoding="utf-8",
)


validation = f"""SOURCE_REPRODUCTION_PASS
METRIC_IDENTITIES_PASS
MAX_FLIP_IDENTITY_ERROR={float(flip_error.max())}
MAX_HIDDEN_CHURN_IDENTITY_ERROR={float(hidden_error.max())}
N_LE1={int((df["abs_acc_delta_pct"] <= 1).sum())}
N_LE2={int((df["abs_acc_delta_pct"] <= 2).sum())}
N_LE3={int((df["abs_acc_delta_pct"] <= 3).sum())}
N_LE5={len(le5)}
N_LE10={len(le10)}
LE5_MODEL_COUNT={le5["model_key"].nunique()}
LE5_METHOD_COUNT={le5["method_key"].nunique()}
LE5_KEEP_COUNT={le5["keep_ratio"].nunique()}
"""

(TMP / "core_validation.txt").write_text(
    validation,
    encoding="utf-8",
)

TMP.rename(OUT)


# ------------------------------------------------------------
# Concise console summary
# ------------------------------------------------------------

print("SOURCE_REPRODUCTION_PASS")
print("METRIC_IDENTITIES_PASS")

for threshold in [1, 2, 3, 5, 10]:
    n = int(
        (
            df["abs_acc_delta_pct"]
            <= threshold
        ).sum()
    )
    print(f"N_LE{threshold}={n}")

print(
    "LE5_MODEL_COUNT="
    f"{le5['model_key'].nunique()}"
)
print(
    "LE5_METHOD_COUNT="
    f"{le5['method_key'].nunique()}"
)
print(
    "LE5_KEEP_COUNT="
    f"{le5['keep_ratio'].nunique()}"
)

print("\n===== LE5 COMPLETE LIST =====")
print(le5.to_string(index=False))

print("\n===== LE5 LEAVE-ONE-MODEL-OUT =====")
print(
    leave_model_out.loc[
        leave_model_out["subset"]
        == "abs_delta_le_5"
    ].to_string(index=False)
)

print("\n===== LE5 LEAVE-ONE-METHOD-OUT =====")
print(
    leave_method_out.loc[
        leave_method_out["subset"]
        == "abs_delta_le_5"
    ].to_string(index=False)
)

print(f"\nOUTPUT_DIR={OUT}")
print("STAGE17_STEP4_COMPLETE")
