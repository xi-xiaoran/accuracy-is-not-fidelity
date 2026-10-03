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
    mean,
)

OUT_TABLE = Path("results/analysis_tables")
OUT_FIG = Path("results/figures")
OUT_TABLE.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)


def summarize_churn(dense_rows, comp_rows):
    dense_map = {sample_key(s, i): s for i, s in enumerate(dense_rows)}
    comp_map = {sample_key(s, i): s for i, s in enumerate(comp_rows)}
    keys = sorted(set(dense_map) & set(comp_map))

    n = 0
    dense_correct = 0
    comp_correct = 0

    both_correct_same = 0
    both_correct_flip = 0

    damage = 0          # dense correct -> comp wrong
    repair = 0          # dense wrong -> comp correct
    both_wrong_same = 0
    both_wrong_flip = 0

    skipped = 0

    for k in keys:
        ds = dense_map[k]
        cs = comp_map[k]

        try:
            d_scores = norm_char_scores(ds)
            c_scores = norm_char_scores(cs)
        except Exception:
            skipped += 1
            continue

        if len(d_scores) != len(c_scores) or len(d_scores) < 2:
            skipped += 1
            continue

        gold = extract_gold(ds, len(d_scores))
        if gold is None or not (0 <= gold < len(d_scores)):
            skipped += 1
            continue

        d_pred = argmax(d_scores)
        c_pred = argmax(c_scores)

        dc = (d_pred == gold)
        cc = (c_pred == gold)
        flip = (d_pred != c_pred)

        n += 1
        dense_correct += int(dc)
        comp_correct += int(cc)

        if dc and cc:
            if flip:
                both_correct_flip += 1
            else:
                both_correct_same += 1
        elif dc and not cc:
            damage += 1
        elif (not dc) and cc:
            repair += 1
        else:
            if flip:
                both_wrong_flip += 1
            else:
                both_wrong_same += 1

    dense_acc = dense_correct / n if n else np.nan
    comp_acc = comp_correct / n if n else np.nan
    acc_delta = comp_acc - dense_acc if n else np.nan

    pred_flip = (both_correct_flip + damage + repair + both_wrong_flip) / n if n else np.nan
    damage_rate = damage / n if n else np.nan
    repair_rate = repair / n if n else np.nan
    both_wrong_flip_rate = both_wrong_flip / n if n else np.nan

    hidden_churn = pred_flip - abs(acc_delta) if n else np.nan
    offsetting_churn = 2 * min(damage_rate, repair_rate) if n else np.nan

    return {
        "n_samples": n,
        "n_skipped": skipped,

        "dense_acc_score": dense_acc,
        "compressed_acc_score": comp_acc,
        "acc_delta_score": acc_delta,

        "pred_flip_score": pred_flip,
        "hidden_churn": hidden_churn,
        "offsetting_churn": offsetting_churn,

        "both_correct_same_rate": both_correct_same / n if n else np.nan,
        "both_correct_flip_rate": both_correct_flip / n if n else np.nan,
        "damage_rate": damage_rate,
        "repair_rate": repair_rate,
        "both_wrong_same_rate": both_wrong_same / n if n else np.nan,
        "both_wrong_flip_rate": both_wrong_flip_rate,

        "both_correct_same_n": both_correct_same,
        "both_correct_flip_n": both_correct_flip,
        "damage_n": damage,
        "repair_n": repair,
        "both_wrong_same_n": both_wrong_same,
        "both_wrong_flip_n": both_wrong_flip,
    }


def weighted_mean(df, col, weight="n_samples"):
    x = df[[col, weight]].dropna()
    if len(x) == 0:
        return np.nan
    return np.average(x[col], weights=x[weight])


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT_FIG / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(OUT_FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    dense_targets = {r["model_key"]: r for r in pd.read_csv(DENSE_CSV).to_dict("records")}
    comp_targets = pd.read_csv(COMP_CSV).to_dict("records")

    rows = []

    for i, cr in enumerate(comp_targets, 1):
        dr = dense_targets[cr["model_key"]]

        for task in TASKS:
            dense_file, dense_rows, _ = load_task_samples(Path(dr["output_dir"]), task)
            comp_file, comp_rows, _ = load_task_samples(Path(cr["output_dir"]), task)

            m = summarize_churn(dense_rows, comp_rows)

            rows.append({
                "target_id": cr["target_id"],
                "model_key": cr["model_key"],
                "method_key": cr["method_key"],
                "method_family": cr["method_family"],
                "keep_tag": cr["keep_tag"],
                "keep_ratio": cr["keep_ratio"],
                "task": task,
                **m,
                "dense_file": str(dense_file),
                "compressed_file": str(comp_file),
            })

        if i % 25 == 0:
            print(f"PROGRESS {i}/{len(comp_targets)}", flush=True)

    task_churn = pd.DataFrame(rows)
    task_churn.to_csv(OUT_TABLE / "decision_churn_by_task.csv", index=False)

    # Add percentage columns.
    for df in [task_churn]:
        for c in [
            "acc_delta_score", "pred_flip_score", "hidden_churn", "offsetting_churn",
            "damage_rate", "repair_rate", "both_wrong_flip_rate",
            "both_correct_same_rate", "both_wrong_same_rate"
        ]:
            df[c + "_pct"] = df[c] * 100.0

    # Config-level weighted aggregation.
    group_cols = ["target_id", "model_key", "method_key", "method_family", "keep_tag", "keep_ratio"]
    metric_cols = [
        "dense_acc_score", "compressed_acc_score", "acc_delta_score",
        "pred_flip_score", "hidden_churn", "offsetting_churn",
        "both_correct_same_rate", "both_correct_flip_rate",
        "damage_rate", "repair_rate",
        "both_wrong_same_rate", "both_wrong_flip_rate",
    ]

    config_rows = []
    for key, g in task_churn.groupby(group_cols):
        row = {k: v for k, v in zip(group_cols, key)}
        row["n_task_rows"] = len(g)
        row["n_samples"] = g["n_samples"].sum()
        for c in metric_cols:
            row[c] = weighted_mean(g, c)
        config_rows.append(row)

    config_churn = pd.DataFrame(config_rows)
    for c in metric_cols:
        config_churn[c + "_pct"] = config_churn[c] * 100.0
    config_churn.to_csv(OUT_TABLE / "decision_churn_by_config.csv", index=False)

    # Accuracy buckets at task level.
    bins = [-np.inf, -0.30, -0.20, -0.10, -0.05, -0.01, 0.01, 0.05, 0.10, np.inf]
    labels = [
        "drop >30pp", "drop 20-30pp", "drop 10-20pp", "drop 5-10pp",
        "drop 1-5pp", "within ±1pp", "gain 1-5pp", "gain 5-10pp", "gain >10pp"
    ]
    task_churn["acc_bucket"] = pd.cut(task_churn["acc_delta_score"], bins=bins, labels=labels)

    bucket_rows = []
    for b, g in task_churn.groupby("acc_bucket", observed=False):
        row = {"acc_bucket": str(b), "n_rows": len(g), "n_samples": g["n_samples"].sum()}
        for c in metric_cols:
            row[c] = weighted_mean(g, c)
            row[c + "_pct"] = row[c] * 100.0 if pd.notna(row[c]) else np.nan
        bucket_rows.append(row)

    bucket = pd.DataFrame(bucket_rows)
    bucket.to_csv(OUT_TABLE / "decision_churn_accuracy_buckets.csv", index=False)

    # Keep0p8 and literature-ish subsets.
    keep08 = config_churn[config_churn["keep_ratio"] == 0.8].copy()
    keep08.to_csv(OUT_TABLE / "decision_churn_keep0p8_config.csv", index=False)

    stable_task = task_churn[task_churn["acc_delta_score"].abs() <= 0.05].copy()
    stable_task.sort_values(["hidden_churn", "pred_flip_score"], ascending=False).to_csv(
        OUT_TABLE / "decision_churn_accuracy_stable_task_examples.csv",
        index=False,
    )

    # Method summary.
    method_rows = []
    for method, g in config_churn.groupby("method_key"):
        row = {"method_key": method, "n_configs": len(g), "n_samples": g["n_samples"].sum()}
        for c in metric_cols:
            row[c] = weighted_mean(g, c)
            row[c + "_pct"] = row[c] * 100.0 if pd.notna(row[c]) else np.nan
        method_rows.append(row)

    method = pd.DataFrame(method_rows).sort_values("hidden_churn")
    method.to_csv(OUT_TABLE / "decision_churn_method_summary.csv", index=False)

    # Figures.
    # Fig 7: accuracy-hidden churn by bucket.
    bp = bucket[bucket["n_rows"] > 0].copy()
    plt.figure(figsize=(8.2, 4.8))
    plt.bar(bp["acc_bucket"], bp["hidden_churn_pct"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Accuracy-hidden churn (%)")
    plt.xlabel("Task-level accuracy-change bucket")
    plt.title("Net accuracy change hides substantial decision churn")
    plt.grid(axis="y", alpha=0.3)
    savefig("fig7_accuracy_hidden_churn_buckets")

    # Fig 8: churn decomposition by bucket.
    plt.figure(figsize=(8.4, 5.0))
    x = np.arange(len(bp))
    damage = bp["damage_rate_pct"].values
    repair = bp["repair_rate_pct"].values
    wrongflip = bp["both_wrong_flip_rate_pct"].values
    plt.bar(x, damage, label="Correct → wrong")
    plt.bar(x, repair, bottom=damage, label="Wrong → correct")
    plt.bar(x, wrongflip, bottom=damage + repair, label="Wrong → different wrong")
    plt.xticks(x, bp["acc_bucket"], rotation=35, ha="right")
    plt.ylabel("Rate (%)")
    plt.xlabel("Task-level accuracy-change bucket")
    plt.title("Prediction flips decompose into damage, repair, and hidden wrong-answer churn")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    savefig("fig8_churn_decomposition_buckets")

    # Fig 9: method ranking by hidden churn.
    mp = method[method["n_configs"] >= 5].sort_values("hidden_churn_pct")
    plt.figure(figsize=(8.2, 5.8))
    plt.barh(mp["method_key"], mp["hidden_churn_pct"])
    plt.xlabel("Mean accuracy-hidden churn (%)")
    plt.ylabel("Method")
    plt.title("Some methods reduce utility loss but still leave hidden decision churn")
    plt.grid(axis="x", alpha=0.3)
    savefig("fig9_method_hidden_churn_ranking")

    # Main text summaries.
    print("===== DECISION CHURN GLOBAL CONFIG =====")
    for name, df in [
        ("all_config", config_churn),
        ("keep0p8_config", config_churn[config_churn["keep_ratio"] == 0.8]),
        ("acc_abs_le_5pp_task", stable_task),
    ]:
        if len(df) == 0:
            continue
        print()
        print(name, "n_rows", len(df))
        for c in ["acc_delta_score", "pred_flip_score", "hidden_churn", "damage_rate", "repair_rate", "both_wrong_flip_rate"]:
            if "n_samples" in df.columns:
                val = weighted_mean(df, c)
            else:
                val = df[c].mean()
            print(f"{c:24s} {val*100:.3f}")

    print("\n===== ACCURACY BUCKET CHURN =====")
    show_cols = [
        "acc_bucket", "n_rows", "acc_delta_score_pct", "pred_flip_score_pct",
        "hidden_churn_pct", "damage_rate_pct", "repair_rate_pct", "both_wrong_flip_rate_pct"
    ]
    print(bucket[show_cols].to_string(index=False))

    print("\n===== METHOD HIDDEN CHURN >=5 CONFIGS =====")
    show = method[method["n_configs"] >= 5][[
        "method_key", "n_configs", "acc_delta_score_pct", "pred_flip_score_pct",
        "hidden_churn_pct", "damage_rate_pct", "repair_rate_pct", "both_wrong_flip_rate_pct"
    ]]
    print(show.to_string(index=False))

    print("\n===== WROTE =====")
    for p in [
        OUT_TABLE / "decision_churn_by_task.csv",
        OUT_TABLE / "decision_churn_by_config.csv",
        OUT_TABLE / "decision_churn_accuracy_buckets.csv",
        OUT_TABLE / "decision_churn_method_summary.csv",
    ]:
        print(p)

    for p in sorted(OUT_FIG.glob("fig7*.pdf")) + sorted(OUT_FIG.glob("fig8*.pdf")) + sorted(OUT_FIG.glob("fig9*.pdf")):
        print(p)


if __name__ == "__main__":
    main()
