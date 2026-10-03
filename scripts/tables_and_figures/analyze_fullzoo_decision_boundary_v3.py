import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from scripts.parsing.build_fullzoo_lowrank_pair_metrics_tables import (
    TASKS,
    COMP_CSV,
    DENSE_CSV,
    load_task_samples,
    sample_key,
    norm_char_scores,
    extract_gold,
    argmax,
    softmax,
    entropy,
    margin,
    js,
)

OUT_TABLE = Path("results/analysis_tables")
OUT_FIG = Path("results/figures")
OUT_TABLE.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT_FIG / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(OUT_FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def bucket_margin(x):
    bins = [0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.01]
    labels = ["0-0.02", "0.02-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.40", "0.40-0.70", "0.70-1.00"]
    for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
        if lo <= x < hi:
            return lab
    return "other"


def bucket_entropy_quantile(ent, qs):
    if ent <= qs[0]:
        return "Q1 lowest entropy"
    if ent <= qs[1]:
        return "Q2"
    if ent <= qs[2]:
        return "Q3"
    return "Q4 highest entropy"


def summarize_pair_samples(dense_rows, comp_rows):
    dense_map = {sample_key(s, i): s for i, s in enumerate(dense_rows)}
    comp_map = {sample_key(s, i): s for i, s in enumerate(comp_rows)}
    keys = sorted(set(dense_map) & set(comp_map))

    rows = []
    for k in keys:
        ds = dense_map[k]
        cs = comp_map[k]

        try:
            d_scores = norm_char_scores(ds)
            c_scores = norm_char_scores(cs)
        except Exception:
            continue

        if len(d_scores) != len(c_scores) or len(d_scores) < 2:
            continue

        gold = extract_gold(ds, len(d_scores))
        if gold is None or not (0 <= gold < len(d_scores)):
            continue

        dp = softmax(d_scores)
        cp = softmax(c_scores)

        d_pred = argmax(d_scores)
        c_pred = argmax(c_scores)

        dense_correct = int(d_pred == gold)
        comp_correct = int(c_pred == gold)
        flip = int(d_pred != c_pred)

        if dense_correct and not comp_correct:
            churn_type = "damage"
        elif (not dense_correct) and comp_correct:
            churn_type = "repair"
        elif (not dense_correct) and (not comp_correct) and flip:
            churn_type = "wrong_to_wrong_flip"
        elif dense_correct and comp_correct and flip:
            churn_type = "correct_to_correct_flip"
        else:
            churn_type = "stable"

        rows.append({
            "dense_margin": margin(dp),
            "dense_entropy": entropy(dp),
            "dense_gold_prob": dp[gold],
            "comp_margin": margin(cp),
            "comp_entropy": entropy(cp),
            "comp_gold_prob": cp[gold],
            "prob_js": js(dp, cp),
            "dense_correct": dense_correct,
            "compressed_correct": comp_correct,
            "flip": flip,
            "damage": int(churn_type == "damage"),
            "repair": int(churn_type == "repair"),
            "wrong_to_wrong_flip": int(churn_type == "wrong_to_wrong_flip"),
            "churn_type": churn_type,
        })

    return rows


def agg_rates(df, group_col):
    g = df.groupby(group_col, observed=False).agg(
        n_samples=("flip", "count"),
        flip_rate=("flip", "mean"),
        damage_rate=("damage", "mean"),
        repair_rate=("repair", "mean"),
        wrong_to_wrong_flip_rate=("wrong_to_wrong_flip", "mean"),
        prob_js_mean=("prob_js", "mean"),
        dense_margin_mean=("dense_margin", "mean"),
        dense_entropy_mean=("dense_entropy", "mean"),
        dense_gold_prob_mean=("dense_gold_prob", "mean"),
    ).reset_index()

    for c in [
        "flip_rate", "damage_rate", "repair_rate", "wrong_to_wrong_flip_rate",
        "prob_js_mean", "dense_margin_mean", "dense_entropy_mean", "dense_gold_prob_mean"
    ]:
        g[c + "_pct"] = g[c] * 100
    return g


def main():
    dense_targets = {r["model_key"]: r for r in pd.read_csv(DENSE_CSV).to_dict("records")}
    comp_targets = pd.read_csv(COMP_CSV).to_dict("records")

    all_rows = []
    sample_counter = 0

    for i, cr in enumerate(comp_targets, 1):
        dr = dense_targets[cr["model_key"]]

        for task in TASKS:
            dense_file, dense_rows, _ = load_task_samples(Path(dr["output_dir"]), task)
            comp_file, comp_rows, _ = load_task_samples(Path(cr["output_dir"]), task)

            rows = summarize_pair_samples(dense_rows, comp_rows)
            for r in rows:
                r.update({
                    "target_id": cr["target_id"],
                    "model_key": cr["model_key"],
                    "method_key": cr["method_key"],
                    "keep_tag": cr["keep_tag"],
                    "keep_ratio": cr["keep_ratio"],
                    "task": task,
                })
            all_rows.extend(rows)
            sample_counter += len(rows)

        if i % 25 == 0:
            print(f"PROGRESS {i}/{len(comp_targets)} samples={sample_counter}", flush=True)

    df = pd.DataFrame(all_rows)

    # Save sample-level table only as compressed parquet if pyarrow exists, otherwise CSV gz.
    try:
        df.to_parquet(OUT_TABLE / "decision_boundary_sample_level.parquet", index=False)
        print("WROTE", OUT_TABLE / "decision_boundary_sample_level.parquet")
    except Exception as e:
        print("PARQUET_FAILED", repr(e))
        df.to_csv(OUT_TABLE / "decision_boundary_sample_level.csv.gz", index=False, compression="gzip")
        print("WROTE", OUT_TABLE / "decision_boundary_sample_level.csv.gz")

    df["dense_margin_bucket"] = df["dense_margin"].apply(bucket_margin)

    q = df["dense_entropy"].quantile([0.25, 0.50, 0.75]).values
    df["dense_entropy_bucket"] = df["dense_entropy"].apply(lambda x: bucket_entropy_quantile(x, q))

    margin_order = ["0-0.02", "0.02-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.40", "0.40-0.70", "0.70-1.00"]
    entropy_order = ["Q1 lowest entropy", "Q2", "Q3", "Q4 highest entropy"]

    margin_summary = agg_rates(df, "dense_margin_bucket")
    margin_summary["dense_margin_bucket"] = pd.Categorical(margin_summary["dense_margin_bucket"], categories=margin_order, ordered=True)
    margin_summary = margin_summary.sort_values("dense_margin_bucket")
    margin_summary.to_csv(OUT_TABLE / "decision_boundary_by_dense_margin_bucket.csv", index=False)

    entropy_summary = agg_rates(df, "dense_entropy_bucket")
    entropy_summary["dense_entropy_bucket"] = pd.Categorical(entropy_summary["dense_entropy_bucket"], categories=entropy_order, ordered=True)
    entropy_summary = entropy_summary.sort_values("dense_entropy_bucket")
    entropy_summary.to_csv(OUT_TABLE / "decision_boundary_by_dense_entropy_bucket.csv", index=False)

    # Task-level boundary summary.
    task_summary = df.groupby("task").agg(
        n_samples=("flip", "count"),
        flip_rate=("flip", "mean"),
        damage_rate=("damage", "mean"),
        repair_rate=("repair", "mean"),
        wrong_to_wrong_flip_rate=("wrong_to_wrong_flip", "mean"),
        dense_margin_mean=("dense_margin", "mean"),
        dense_entropy_mean=("dense_entropy", "mean"),
        prob_js_mean=("prob_js", "mean"),
    ).reset_index()
    for c in ["flip_rate", "damage_rate", "repair_rate", "wrong_to_wrong_flip_rate", "dense_margin_mean", "dense_entropy_mean", "prob_js_mean"]:
        task_summary[c + "_pct"] = task_summary[c] * 100
    task_summary.to_csv(OUT_TABLE / "decision_boundary_by_task.csv", index=False)

    # Method-level boundary summary.
    method_summary = df.groupby("method_key").agg(
        n_samples=("flip", "count"),
        flip_rate=("flip", "mean"),
        damage_rate=("damage", "mean"),
        repair_rate=("repair", "mean"),
        wrong_to_wrong_flip_rate=("wrong_to_wrong_flip", "mean"),
        dense_margin_mean=("dense_margin", "mean"),
        dense_entropy_mean=("dense_entropy", "mean"),
        prob_js_mean=("prob_js", "mean"),
    ).reset_index()
    for c in ["flip_rate", "damage_rate", "repair_rate", "wrong_to_wrong_flip_rate", "dense_margin_mean", "dense_entropy_mean", "prob_js_mean"]:
        method_summary[c + "_pct"] = method_summary[c] * 100
    method_summary.to_csv(OUT_TABLE / "decision_boundary_by_method.csv", index=False)

    # Figure 13: dense margin bucket vs flip.
    plt.figure(figsize=(7.2, 4.5))
    plt.plot(margin_summary["dense_margin_bucket"].astype(str), margin_summary["flip_rate_pct"], marker="o", label="Prediction flip")
    plt.plot(margin_summary["dense_margin_bucket"].astype(str), margin_summary["damage_rate_pct"], marker="o", label="Correct→wrong")
    plt.plot(margin_summary["dense_margin_bucket"].astype(str), margin_summary["repair_rate_pct"], marker="o", label="Wrong→correct")
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Dense prediction margin bucket")
    plt.ylabel("Rate (%)")
    plt.title("Decision churn concentrates near low-margin dense predictions")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig("fig13_boundary_margin_vs_churn")

    # Figure 14: entropy bucket vs flip.
    plt.figure(figsize=(6.5, 4.4))
    plt.plot(entropy_summary["dense_entropy_bucket"].astype(str), entropy_summary["flip_rate_pct"], marker="o", label="Prediction flip")
    plt.plot(entropy_summary["dense_entropy_bucket"].astype(str), entropy_summary["prob_js_mean_pct"], marker="o", label="Prob-JS ×100")
    plt.xticks(rotation=20, ha="right")
    plt.xlabel("Dense entropy quartile")
    plt.ylabel("Rate / scaled value")
    plt.title("High-entropy dense predictions are more vulnerable to churn")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig("fig14_boundary_entropy_vs_churn")

    # Figure 15: task margin vs flip scatter.
    plt.figure(figsize=(6.0, 4.8))
    plt.scatter(task_summary["dense_margin_mean"], task_summary["flip_rate_pct"], s=70)
    for _, r in task_summary.iterrows():
        plt.text(r["dense_margin_mean"], r["flip_rate_pct"], r["task"], fontsize=8)
    plt.xlabel("Mean dense margin")
    plt.ylabel("Prediction flip (%)")
    plt.title("Tasks with lower dense margin tend to show higher churn")
    plt.grid(True, alpha=0.3)
    savefig("fig15_task_margin_vs_flip")

    print("===== GLOBAL SAMPLE-LEVEL BOUNDARY =====")
    print("n_samples", len(df))
    print("flip_rate_pct", df["flip"].mean() * 100)
    print("damage_rate_pct", df["damage"].mean() * 100)
    print("repair_rate_pct", df["repair"].mean() * 100)
    print("wrong_to_wrong_flip_pct", df["wrong_to_wrong_flip"].mean() * 100)
    print("dense_margin_mean", df["dense_margin"].mean())
    print("dense_entropy_mean", df["dense_entropy"].mean())

    print("\n===== MARGIN BUCKET SUMMARY =====")
    print(margin_summary[[
        "dense_margin_bucket", "n_samples", "flip_rate_pct", "damage_rate_pct",
        "repair_rate_pct", "wrong_to_wrong_flip_rate_pct", "prob_js_mean_pct"
    ]].to_string(index=False))

    print("\n===== ENTROPY BUCKET SUMMARY =====")
    print(entropy_summary[[
        "dense_entropy_bucket", "n_samples", "flip_rate_pct", "damage_rate_pct",
        "repair_rate_pct", "wrong_to_wrong_flip_rate_pct", "prob_js_mean_pct"
    ]].to_string(index=False))

    print("\n===== TASK BOUNDARY SUMMARY =====")
    print(task_summary[[
        "task", "n_samples", "dense_margin_mean", "dense_entropy_mean",
        "flip_rate_pct", "damage_rate_pct", "repair_rate_pct", "prob_js_mean_pct"
    ]].sort_values("flip_rate_pct", ascending=False).to_string(index=False))

    print("\n===== WROTE =====")
    for p in [
        OUT_TABLE / "decision_boundary_by_dense_margin_bucket.csv",
        OUT_TABLE / "decision_boundary_by_dense_entropy_bucket.csv",
        OUT_TABLE / "decision_boundary_by_task.csv",
        OUT_TABLE / "decision_boundary_by_method.csv",
        OUT_FIG / "fig13_boundary_margin_vs_churn.pdf",
        OUT_FIG / "fig14_boundary_entropy_vs_churn.pdf",
        OUT_FIG / "fig15_task_margin_vs_flip.pdf",
    ]:
        print(p)


if __name__ == "__main__":
    main()
