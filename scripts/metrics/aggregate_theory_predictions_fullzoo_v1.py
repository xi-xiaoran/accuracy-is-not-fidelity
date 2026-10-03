import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.parsing.build_fullzoo_lowrank_pair_metrics_tables import COMP_CSV

ROOT = Path(".").resolve()
OUT = ROOT / "results/theory_predictions_v1"
FULL = ROOT / "results/theory_predictions_v1_fullzoo"
FULL.mkdir(parents=True, exist_ok=True)

META_CANDIDATES = [
    "target_id", "model_key", "method", "method_key", "compression_method",
    "keep", "keep_ratio", "rank", "remote_prefix", "eval_model",
]


def pct(x):
    return 100.0 * float(x)


def safe_read(path):
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return None


def base_from_sample(df):
    ev = df[df["split"] == "eval"].copy()

    dense_acc = ev["dense_correct"].mean()
    comp_acc = ev["compressed_correct"].mean()
    acc_delta = comp_acc - dense_acc
    pred_flip = ev["flip"].mean()

    bar_lt1 = ev[ev["boundary_ratio"] < 1]
    bar_ge1 = ev[ev["boundary_ratio"] >= 1]
    bar_1_2 = ev[(ev["boundary_ratio"] >= 1) & (ev["boundary_ratio"] < 2)]
    bar_gt2 = ev[ev["boundary_ratio"] >= 2]

    def flip_rate(g):
        return float(g["flip"].mean()) if len(g) else np.nan

    def hidden(g):
        if len(g) == 0:
            return np.nan
        da = g["compressed_correct"].mean() - g["dense_correct"].mean()
        pf = g["flip"].mean()
        return float(pf - abs(da))

    return {
        "n_eval": len(ev),
        "dense_acc_pct": pct(dense_acc),
        "compressed_acc_pct": pct(comp_acc),
        "acc_delta_pct": pct(acc_delta),
        "pred_flip_pct": pct(pred_flip),
        "hidden_churn_pct": pct(pred_flip - abs(acc_delta)),
        "damage_pct": pct(ev["damage"].mean()),
        "repair_pct": pct(ev["repair"].mean()),
        "wrong_to_wrong_pct": pct(ev["wrong_to_wrong"].mean()),
        "correct_to_correct_pct": pct(ev["correct_to_correct"].mean()),
        "prob_js_pct": pct(ev["prob_js"].mean()),
        "mean_dense_margin": float(ev["dense_margin"].mean()),
        "mean_boundary_ratio": float(ev["boundary_ratio"].replace([np.inf, -np.inf], np.nan).mean()),
        "bar_lt1_n": len(bar_lt1),
        "bar_ge1_n": len(bar_ge1),
        "bar_1_2_n": len(bar_1_2),
        "bar_gt2_n": len(bar_gt2),
        "bar_lt1_flip_pct": pct(flip_rate(bar_lt1)),
        "bar_ge1_flip_pct": pct(flip_rate(bar_ge1)),
        "bar_1_2_flip_pct": pct(flip_rate(bar_1_2)),
        "bar_gt2_flip_pct": pct(flip_rate(bar_gt2)),
        "bar_lt1_hidden_pct": pct(hidden(bar_lt1)),
        "bar_ge1_hidden_pct": pct(hidden(bar_ge1)),
        "bar_1_2_hidden_pct": pct(hidden(bar_1_2)),
        "bar_gt2_hidden_pct": pct(hidden(bar_gt2)),
    }


def get_method_row(df, method):
    if df is None or "method" not in df.columns:
        return None
    x = df[df["method"] == method]
    if len(x) == 0:
        return None
    return x.iloc[0]


def main():
    comp = pd.read_csv(COMP_CSV)
    meta_cols = [c for c in META_CANDIDATES if c in comp.columns]
    comp_meta = comp[meta_cols].drop_duplicates("target_id")

    summaries = []
    bar_all = []
    proj_all = []
    energy_all = []
    rank_all = []
    failures = []

    ids = comp["target_id"].dropna().astype(str).drop_duplicates().tolist()

    for i, tid in enumerate(ids, 1):
        sample_p = OUT / f"{tid}_sample_theory_frame.csv"
        bar_p = OUT / f"{tid}_boundary_ratio_buckets.csv"
        proj_p = OUT / f"{tid}_projection_eval.csv"
        energy_p = OUT / f"{tid}_projection_energy.csv"
        rank_p = OUT / f"{tid}_correction_direction_rank.csv"

        sample = safe_read(sample_p)
        if sample is None:
            failures.append({"target_id": tid, "reason": "missing_sample"})
            continue

        row = {"target_id": tid}
        try:
            row.update(base_from_sample(sample))
        except Exception as e:
            failures.append({"target_id": tid, "reason": f"base_error:{type(e).__name__}:{e}"})
            continue

        proj = safe_read(proj_p)
        energy = safe_read(energy_p)
        rank = safe_read(rank_p)
        bar = safe_read(bar_p)

        if proj is not None:
            for _, r in proj.iterrows():
                rr = r.to_dict()
                rr["target_id"] = tid
                proj_all.append(rr)

            ident = get_method_row(proj, "identity_compressed")
            task = get_method_row(proj, "task_bias")
            global_n = get_method_row(proj, "global_by_nchoices")
            task_pred = get_method_row(proj, "task_compressed_pred_bias")
            risk80 = get_method_row(proj, "task_risk80_pred_bias")

            for prefix, mr in [
                ("proj_task_bias", task),
                ("proj_global", global_n),
                ("proj_task_pred", task_pred),
                ("proj_risk80_pred", risk80),
            ]:
                if mr is not None:
                    for c in [
                        "repaired_acc_pct", "acc_delta_pct", "pred_flip_pct",
                        "hidden_churn_pct", "prob_js_pct", "nll_shift"
                    ]:
                        if c in mr.index:
                            row[f"{prefix}_{c}"] = mr[c]

                    if ident is not None:
                        row[f"{prefix}_prob_js_delta_pct"] = mr.get("prob_js_pct", np.nan) - ident.get("prob_js_pct", np.nan)
                        row[f"{prefix}_hidden_churn_delta_pct"] = mr.get("hidden_churn_pct", np.nan) - ident.get("hidden_churn_pct", np.nan)
                        row[f"{prefix}_pred_flip_delta_pct"] = mr.get("pred_flip_pct", np.nan) - ident.get("pred_flip_pct", np.nan)

        if energy is not None:
            for _, r in energy.iterrows():
                rr = r.to_dict()
                rr["target_id"] = tid
                energy_all.append(rr)

            for method in ["global_by_nchoices", "task_bias", "task_compressed_pred_bias", "task_risk80_pred_bias"]:
                er = get_method_row(energy, method)
                if er is not None:
                    row[f"energy_{method}_explained_pct"] = er.get("correction_energy_explained_pct", np.nan)

        if rank is not None:
            for _, r in rank.iterrows():
                rr = r.to_dict()
                rr["target_id"] = tid
                rank_all.append(rr)

            def rank_pick(group, n_choices):
                x = rank[(rank["group"] == group) & (rank["n_choices"] == n_choices)]
                return x.iloc[0] if len(x) else None

            for group in ["eval_all", "eval_mismatch", "eval_highrisk_mismatch"]:
                for n_choices in [2, 4]:
                    rr = rank_pick(group, n_choices)
                    if rr is not None:
                        pfx = f"rank_{group}_n{n_choices}"
                        row[f"{pfx}_n"] = rr.get("n", np.nan)
                        row[f"{pfx}_effective_rank"] = rr.get("effective_rank", np.nan)
                        row[f"{pfx}_top1_explained_pct"] = rr.get("top1_explained_pct", np.nan)
                        row[f"{pfx}_top2_explained_pct"] = rr.get("top2_explained_pct", np.nan)

        if bar is not None:
            for _, r in bar.iterrows():
                rr = r.to_dict()
                rr["target_id"] = tid
                bar_all.append(rr)

        summaries.append(row)

    summary = pd.DataFrame(summaries)
    if len(summary):
        summary = summary.merge(comp_meta, on="target_id", how="left")

    summary_csv = FULL / "fullzoo_theory_summary_by_config.csv"
    summary.to_csv(summary_csv, index=False)

    if bar_all:
        pd.DataFrame(bar_all).merge(comp_meta, on="target_id", how="left").to_csv(
            FULL / "fullzoo_boundary_ratio_buckets_long.csv", index=False
        )
    if proj_all:
        pd.DataFrame(proj_all).merge(comp_meta, on="target_id", how="left").to_csv(
            FULL / "fullzoo_projection_eval_long.csv", index=False
        )
    if energy_all:
        pd.DataFrame(energy_all).merge(comp_meta, on="target_id", how="left").to_csv(
            FULL / "fullzoo_projection_energy_long.csv", index=False
        )
    if rank_all:
        pd.DataFrame(rank_all).merge(comp_meta, on="target_id", how="left").to_csv(
            FULL / "fullzoo_correction_direction_rank_long.csv", index=False
        )

    if failures:
        pd.DataFrame(failures).to_csv(FULL / "aggregate_failures.csv", index=False)

    # Aggregate summaries.
    group_candidates = [
        ["method_key", "keep_ratio"],
        ["method", "keep_ratio"],
        ["compression_method", "keep_ratio"],
        ["model_key"],
        ["method_key"],
        ["method"],
    ]

    for gc in group_candidates:
        gc = [c for c in gc if c in summary.columns]
        if not gc:
            continue

        # Exclude group keys from numeric aggregation to avoid reset_index collisions
        # when a group key such as keep_ratio is also numeric.
        numeric_cols = [
            c for c in summary.select_dtypes(include=[np.number]).columns.tolist()
            if c not in gc
        ]

        out = summary.groupby(gc, dropna=False)[numeric_cols].mean().reset_index()
        name = "by_" + "_".join(gc)
        out.to_csv(FULL / f"fullzoo_theory_summary_{name}.csv", index=False)

    report = FULL / "fullzoo_theory_report.txt"
    with open(report, "w") as f:
        f.write(f"N configs summarized: {len(summary)}\n")
        f.write(f"N failures: {len(failures)}\n\n")

        key_cols = [
            "dense_acc_pct", "compressed_acc_pct", "acc_delta_pct",
            "pred_flip_pct", "hidden_churn_pct", "damage_pct", "repair_pct",
            "bar_lt1_flip_pct", "bar_ge1_flip_pct", "bar_1_2_flip_pct", "bar_gt2_flip_pct",
            "proj_task_bias_prob_js_delta_pct",
            "proj_task_bias_hidden_churn_delta_pct",
            "energy_task_bias_explained_pct",
            "energy_global_by_nchoices_explained_pct",
        ]
        key_cols = [c for c in key_cols if c in summary.columns]

        f.write("Overall means:\n")
        f.write(summary[key_cols].mean(numeric_only=True).to_string())
        f.write("\n\nOverall medians:\n")
        f.write(summary[key_cols].median(numeric_only=True).to_string())
        f.write("\n")

    print("WROTE", summary_csv)
    print("WROTE", report)
    print()
    print("===== OVERALL MEANS =====")
    show = [
        "acc_delta_pct", "pred_flip_pct", "hidden_churn_pct",
        "bar_lt1_flip_pct", "bar_ge1_flip_pct", "bar_1_2_flip_pct", "bar_gt2_flip_pct",
        "proj_task_bias_prob_js_delta_pct", "proj_task_bias_hidden_churn_delta_pct",
        "energy_task_bias_explained_pct",
    ]
    show = [c for c in show if c in summary.columns]
    print(summary[show].mean(numeric_only=True).to_string())

    if failures:
        print()
        print("AGGREGATE FAILURES", len(failures))
        print(pd.DataFrame(failures).head(20).to_string(index=False))
    else:
        print("NO AGGREGATE FAILURES")


if __name__ == "__main__":
    main()
