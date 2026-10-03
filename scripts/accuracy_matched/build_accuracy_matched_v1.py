import argparse
import hashlib
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PARSER = argparse.ArgumentParser(
    description="Build accuracy-matched eligible-pair and one-to-one analyses."
)
PARSER.add_argument("source_csv", type=Path, help="Configuration-level metrics CSV.")
PARSER.add_argument("output_dir", type=Path, help="New output directory.")
ARGS = PARSER.parse_args()

src = ARGS.source_csv.resolve()
out = ARGS.output_dir.resolve()

if not src.is_file():
    raise FileNotFoundError(f"Missing source CSV: {src}")
if out.exists():
    raise FileExistsError(f"Output already exists: {out}")

out.mkdir(parents=True)

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

df = pd.read_csv(src)

missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

if len(df) != 235:
    raise RuntimeError(f"Expected 235 configurations, found {len(df)}")

if df["target_id"].duplicated().any():
    duplicated = df.loc[
        df["target_id"].duplicated(), "target_id"
    ].tolist()
    raise RuntimeError(
        f"Duplicate target IDs: {duplicated[:10]}"
    )

x = df[required].copy()

numeric_columns = [
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
    "bar_ge1_flip_pct",
]

for column in numeric_columns:
    x[column] = pd.to_numeric(x[column], errors="raise")

# ------------------------------------------------------------
# Metric identities
# ------------------------------------------------------------

x["abs_acc_change_pct"] = x["acc_delta_pct"].abs()

x["identity_hidden_churn_pct"] = (
    x["pred_flip_pct"] - x["abs_acc_change_pct"]
)
x["identity_hidden_error_pct"] = (
    x["hidden_churn_pct"]
    - x["identity_hidden_churn_pct"]
)

x["identity_predflip_from_drw_pct"] = (
    x["damage_pct"]
    + x["repair_pct"]
    + x["wrong_to_wrong_pct"]
)
x["identity_predflip_error_pct"] = (
    x["pred_flip_pct"]
    - x["identity_predflip_from_drw_pct"]
)

x["identity_accdelta_from_dr_pct"] = (
    x["repair_pct"] - x["damage_pct"]
)
x["identity_accdelta_error_pct"] = (
    x["acc_delta_pct"]
    - x["identity_accdelta_from_dr_pct"]
)

audit = {
    "source": str(src),
    "n_configurations": int(len(x)),
    "n_models": int(x["model_key"].nunique()),
    "n_method_keys": int(x["method_key"].nunique()),
    "n_method_categories": int(
        x["method_category"].nunique()
    ),
    "max_abs_hidden_identity_error_pct": float(
        x["identity_hidden_error_pct"].abs().max()
    ),
    "max_abs_predflip_drw_error_pct": float(
        x["identity_predflip_error_pct"].abs().max()
    ),
    "max_abs_accdelta_dr_error_pct": float(
        x["identity_accdelta_error_pct"].abs().max()
    ),
}

for key in [
    "max_abs_hidden_identity_error_pct",
    "max_abs_predflip_drw_error_pct",
    "max_abs_accdelta_dr_error_pct",
]:
    if audit[key] > 1e-8:
        raise RuntimeError(
            f"Metric identity audit failed: "
            f"{key}={audit[key]}"
        )

x.to_csv(
    out / "checkpoint_metrics_235.csv",
    index=False,
)

# ------------------------------------------------------------
# Construct every unordered same-dense-model pair
# ------------------------------------------------------------

records = []

for model_key, group in x.groupby(
    "model_key", sort=True
):
    rows = (
        group.sort_values("target_id")
        .to_dict("records")
    )

    for a, b in itertools.combinations(rows, 2):
        records.append(
            {
                "model_key": model_key,

                "target_a": a["target_id"],
                "target_b": b["target_id"],

                "method_key_a": a["method_key"],
                "method_key_b": b["method_key"],

                "method_category_a":
                    a["method_category"],
                "method_category_b":
                    b["method_category"],

                "keep_ratio_a": a["keep_ratio"],
                "keep_ratio_b": b["keep_ratio"],

                "dense_acc_a_pct":
                    a["dense_acc_pct"],
                "dense_acc_b_pct":
                    b["dense_acc_pct"],

                "compressed_acc_a_pct":
                    a["compressed_acc_pct"],
                "compressed_acc_b_pct":
                    b["compressed_acc_pct"],

                "acc_delta_a_pct":
                    a["acc_delta_pct"],
                "acc_delta_b_pct":
                    b["acc_delta_pct"],

                "abs_acc_change_a_pct":
                    a["abs_acc_change_pct"],
                "abs_acc_change_b_pct":
                    b["abs_acc_change_pct"],

                # Main matching quantity:
                # ||DeltaAcc_i| - |DeltaAcc_j||
                "abs_acc_change_gap_pp": abs(
                    a["abs_acc_change_pct"]
                    - b["abs_acc_change_pct"]
                ),

                # Retained as a signed-change robustness field.
                "signed_acc_delta_gap_pp": abs(
                    a["acc_delta_pct"]
                    - b["acc_delta_pct"]
                ),

                "pred_flip_a_pct":
                    a["pred_flip_pct"],
                "pred_flip_b_pct":
                    b["pred_flip_pct"],
                "pred_flip_gap_pp": abs(
                    a["pred_flip_pct"]
                    - b["pred_flip_pct"]
                ),

                "hidden_churn_a_pct":
                    a["hidden_churn_pct"],
                "hidden_churn_b_pct":
                    b["hidden_churn_pct"],
                "hidden_churn_gap_pp": abs(
                    a["hidden_churn_pct"]
                    - b["hidden_churn_pct"]
                ),

                "damage_a_pct":
                    a["damage_pct"],
                "damage_b_pct":
                    b["damage_pct"],
                "repair_a_pct":
                    a["repair_pct"],
                "repair_b_pct":
                    b["repair_pct"],
                "wrong_to_wrong_a_pct":
                    a["wrong_to_wrong_pct"],
                "wrong_to_wrong_b_pct":
                    b["wrong_to_wrong_pct"],

                "damage_gap_pp": abs(
                    a["damage_pct"]
                    - b["damage_pct"]
                ),
                "repair_gap_pp": abs(
                    a["repair_pct"]
                    - b["repair_pct"]
                ),
                "wrong_to_wrong_gap_pp": abs(
                    a["wrong_to_wrong_pct"]
                    - b["wrong_to_wrong_pct"]
                ),

                "cross_method_key": bool(
                    a["method_key"]
                    != b["method_key"]
                ),

                "cross_method_category": bool(
                    a["method_category"]
                    != b["method_category"]
                ),

                "same_method_key_different_keep": bool(
                    a["method_key"]
                    == b["method_key"]
                    and a["keep_ratio"]
                    != b["keep_ratio"]
                ),
            }
        )

pairs = pd.DataFrame(records)

pairs = pairs.sort_values(
    [
        "model_key",
        "abs_acc_change_gap_pp",
        "target_a",
        "target_b",
    ]
).reset_index(drop=True)

pairs.to_csv(
    out / "same_model_all_pairs.csv",
    index=False,
)

# ------------------------------------------------------------
# Matched analyses
# ------------------------------------------------------------

calipers = [0.25, 0.50, 1.00]

scopes = {
    # Includes different settings within the same method.
    "all_config_pairs":
        pd.Series(True, index=pairs.index),

    # Different exact method/variant keys.
    "cross_method_key":
        pairs["cross_method_key"],

    # Strongest comparison: ASVD vs SVD-LLM, etc.
    "cross_method_category":
        pairs["cross_method_category"],
}


def summarize(
    q: pd.DataFrame,
    caliper: float,
    scope: str,
    analysis_type: str,
) -> dict:
    row = {
        "analysis_type": analysis_type,
        "caliper_pp": caliper,
        "scope": scope,
        "n_pairs": int(len(q)),
        "n_models": (
            int(q["model_key"].nunique())
            if len(q)
            else 0
        ),
        "n_unique_configs": (
            int(
                len(
                    set(q["target_a"]).union(
                        set(q["target_b"])
                    )
                )
            )
            if len(q)
            else 0
        ),
    }

    for metric in [
        "pred_flip_gap_pp",
        "hidden_churn_gap_pp",
    ]:
        values = q[metric]

        statistics = {
            "mean": values.mean(),
            "median": values.median(),
            "p75": values.quantile(0.75),
            "p90": values.quantile(0.90),
            "max": values.max(),
            "gt2_pct":
                100.0 * (values > 2.0).mean(),
            "gt5_pct":
                100.0 * (values > 5.0).mean(),
            "gt10_pct":
                100.0 * (values > 10.0).mean(),
        }

        for name, value in statistics.items():
            row[f"{metric}_{name}"] = (
                float(value)
                if pd.notna(value)
                else np.nan
            )

    return row


summary_rows = []
by_model_rows = []
all_matched_frames = []

for caliper in calipers:
    tag = str(caliper).replace(".", "p")

    for scope, mask in scopes.items():
        matched = pairs.loc[
            mask
            & (
                pairs["abs_acc_change_gap_pp"]
                <= caliper + 1e-12
            )
        ].copy()

        matched.insert(0, "scope", scope)
        matched.insert(0, "caliper_pp", caliper)

        matched.to_csv(
            out
            / f"matched_pairs_{scope}_{tag}pp.csv",
            index=False,
        )

        all_matched_frames.append(matched)

        summary_rows.append(
            summarize(
                matched,
                caliper,
                scope,
                "all_eligible_pairs",
            )
        )

        for model_key, group in matched.groupby(
            "model_key", sort=True
        ):
            by_model_rows.append(
                {
                    "analysis_type":
                        "all_eligible_pairs",
                    "caliper_pp": caliper,
                    "scope": scope,
                    "model_key": model_key,
                    "n_pairs": int(len(group)),
                    "n_unique_configs": int(
                        len(
                            set(group["target_a"])
                            .union(
                                set(group["target_b"])
                            )
                        )
                    ),
                    "mean_pred_flip_gap_pp":
                        float(
                            group[
                                "pred_flip_gap_pp"
                            ].mean()
                        ),
                    "median_pred_flip_gap_pp":
                        float(
                            group[
                                "pred_flip_gap_pp"
                            ].median()
                        ),
                    "max_pred_flip_gap_pp":
                        float(
                            group[
                                "pred_flip_gap_pp"
                            ].max()
                        ),
                    "mean_hidden_churn_gap_pp":
                        float(
                            group[
                                "hidden_churn_gap_pp"
                            ].mean()
                        ),
                    "median_hidden_churn_gap_pp":
                        float(
                            group[
                                "hidden_churn_gap_pp"
                            ].median()
                        ),
                    "max_hidden_churn_gap_pp":
                        float(
                            group[
                                "hidden_churn_gap_pp"
                            ].max()
                        ),
                }
            )

all_matched = pd.concat(
    all_matched_frames,
    ignore_index=True,
)

all_matched.to_csv(
    out / "accuracy_matched_pairs_all_calipers.csv",
    index=False,
)

# ------------------------------------------------------------
# Outcome-blind deterministic one-to-one matching robustness.
# Each checkpoint appears at most once per model/caliper, and pair
# construction/tie-breaking never uses PredFlip or HiddenChurn.
# ------------------------------------------------------------

def greedy_one_to_one(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    selected = []

    for model_key, group in candidates.groupby(
        "model_key", sort=True
    ):
        group = group.sort_values(
            [
                "abs_acc_change_gap_pp",
                "target_a",
                "target_b",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )

        used = set()

        for _, row in group.iterrows():
            target_a = row["target_a"]
            target_b = row["target_b"]

            if (
                target_a in used
                or target_b in used
            ):
                continue

            selected.append(row.to_dict())
            used.add(target_a)
            used.add(target_b)

    return pd.DataFrame(selected)


greedy_frames = []

for caliper in calipers:
    candidates = pairs.loc[
        pairs["cross_method_category"]
        & (
            pairs["abs_acc_change_gap_pp"]
            <= caliper + 1e-12
        )
    ].copy()

    greedy = greedy_one_to_one(candidates)

    if len(greedy):
        greedy.insert(
            0,
            "scope",
            "cross_method_category",
        )
        greedy.insert(
            0,
            "caliper_pp",
            caliper,
        )

        greedy_frames.append(greedy)

        summary_rows.append(
            summarize(
                greedy,
                caliper,
                "cross_method_category",
                "greedy_one_to_one",
            )
        )

greedy_all = pd.concat(
    greedy_frames,
    ignore_index=True,
)

greedy_all.to_csv(
    out
    / "greedy_one_to_one_cross_method_category.csv",
    index=False,
)

# ------------------------------------------------------------
# Final summaries
# ------------------------------------------------------------

summary = pd.DataFrame(summary_rows)

summary = summary.sort_values(
    [
        "analysis_type",
        "caliper_pp",
        "scope",
    ]
).reset_index(drop=True)

summary.to_csv(
    out / "accuracy_matched_pair_summary.csv",
    index=False,
)

by_model = pd.DataFrame(by_model_rows)

by_model.to_csv(
    out
    / "accuracy_matched_pair_summary_by_model.csv",
    index=False,
)

# Main analysis:
# same dense model,
# different method category,
# <= 0.5 pp difference in |DeltaAcc|.
main_pairs = pairs.loc[
    pairs["cross_method_category"]
    & (
        pairs["abs_acc_change_gap_pp"]
        <= 0.50 + 1e-12
    )
].copy()

main_pairs = main_pairs.sort_values(
    [
        "hidden_churn_gap_pp",
        "pred_flip_gap_pp",
    ],
    ascending=False,
)

main_pairs.to_csv(
    out
    / "main_cross_method_category_pairs_0p5pp.csv",
    index=False,
)

main_pairs.head(50).to_csv(
    out / "main_top50_pairs_0p5pp.csv",
    index=False,
)

main_summary = summary.loc[
    (
        summary["analysis_type"]
        == "all_eligible_pairs"
    )
    & (summary["caliper_pp"] == 0.50)
    & (
        summary["scope"]
        == "cross_method_category"
    )
].iloc[0].to_dict()

# ------------------------------------------------------------
# Provenance
# ------------------------------------------------------------

with (
    out / "audit.json"
).open("w", encoding="utf-8") as handle:
    json.dump(
        audit,
        handle,
        indent=2,
        ensure_ascii=False,
    )

with (
    out / "main_result.json"
).open("w", encoding="utf-8") as handle:
    json.dump(
        main_summary,
        handle,
        indent=2,
        ensure_ascii=False,
    )

source_sha256 = hashlib.sha256(
    src.read_bytes()
).hexdigest()

(
    out / "source_sha256.txt"
).write_text(
    f"{source_sha256}  {src}\n",
    encoding="utf-8",
)

shutil.copy2(
    src,
    out / "source_config_manifest_metrics.csv",
)

readme = f"""# Experiment 1: accuracy-matched checkpoint data

This package contains data only. No figures are generated.

## Source

`{src}`

SHA256:

`{source_sha256}`

## Population

- Configurations: {len(x)}
- Dense models: {x['model_key'].nunique()}
- Method keys: {x['method_key'].nunique()}
- Method categories: {x['method_category'].nunique()}

## Main matching rule

An unordered pair is included when:

1. both checkpoints share the same dense `model_key`;
2. they belong to different `method_category` values;
3. `abs(abs(DeltaAcc_i)-abs(DeltaAcc_j)) <= 0.5` percentage points.

Robustness calipers are 0.25 and 1.0 percentage points.

## Main descriptive result

- Eligible pairs: {int(main_summary['n_pairs'])}
- Models represented: {int(main_summary['n_models'])}
- Unique configurations represented: {int(main_summary['n_unique_configs'])}
- Mean PredFlip gap: {main_summary['pred_flip_gap_pp_mean']:.6f} pp
- Median PredFlip gap: {main_summary['pred_flip_gap_pp_median']:.6f} pp
- Maximum PredFlip gap: {main_summary['pred_flip_gap_pp_max']:.6f} pp
- Mean HiddenChurn gap: {main_summary['hidden_churn_gap_pp_mean']:.6f} pp
- Median HiddenChurn gap: {main_summary['hidden_churn_gap_pp_median']:.6f} pp
- Maximum HiddenChurn gap: {main_summary['hidden_churn_gap_pp_max']:.6f} pp
- PredFlip gap greater than 5 pp: {main_summary['pred_flip_gap_pp_gt5_pct']:.6f}%
- PredFlip gap greater than 10 pp: {main_summary['pred_flip_gap_pp_gt10_pct']:.6f}%

These are descriptive all-eligible-pair statistics. A checkpoint can
appear in multiple eligible pairs. The package therefore also includes
an outcome-blind deterministic greedy one-to-one robustness analysis.

## Identity audits

- max |HiddenChurn - (PredFlip - |DeltaAcc|)|:
  {audit['max_abs_hidden_identity_error_pct']:.12g} pp
- max |PredFlip - (Damage + Repair + Wrong-to-wrong)|:
  {audit['max_abs_predflip_drw_error_pct']:.12g} pp
- max |DeltaAcc - (Repair - Damage)|:
  {audit['max_abs_accdelta_dr_error_pct']:.12g} pp
"""

(
    out / "README.md"
).write_text(
    readme,
    encoding="utf-8",
)

print("============================================================")
print("EXP1_DATA_BUILD_COMPLETE")
print(f"out={out}")
print(f"n_configurations={len(x)}")
print(f"n_same_model_pairs={len(pairs)}")
print(f"main_n_pairs={int(main_summary['n_pairs'])}")
print(
    "main_mean_predflip_gap_pp="
    f"{main_summary['pred_flip_gap_pp_mean']:.6f}"
)
print(
    "main_median_predflip_gap_pp="
    f"{main_summary['pred_flip_gap_pp_median']:.6f}"
)
print(
    "main_max_predflip_gap_pp="
    f"{main_summary['pred_flip_gap_pp_max']:.6f}"
)
print(
    "main_mean_hidden_gap_pp="
    f"{main_summary['hidden_churn_gap_pp_mean']:.6f}"
)
print(
    "main_max_hidden_gap_pp="
    f"{main_summary['hidden_churn_gap_pp_max']:.6f}"
)
print("============================================================")
