import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
OUT = ROOT / "results/mechanism_activation_v1"
AUD = OUT / "audits"
AUD.mkdir(parents=True, exist_ok=True)

import sys

from scripts.mechanism.mech_common_v1 import (
    TASKS,
    load_targets_csv,
    locate_dense_run_dir,
    locate_compressed_run_dir,
    latest_sample_file,
    read_jsonl,
    pred_from_sample,
    gold_from_sample,
    category_from_preds,
    norm_keep,
)

FROZEN = ROOT / "results/theory_predictions_v1_fullzoo/fullzoo_theory_summary_by_config.csv"

def find_col(df, candidates):
    low = {c.lower(): c for c in df.columns}
    for x in candidates:
        if x in df.columns:
            return x
        if x.lower() in low:
            return low[x.lower()]
    # fuzzy fallback
    for c in df.columns:
        cl = c.lower().replace("_", "").replace("-", "")
        for x in candidates:
            xl = x.lower().replace("_", "").replace("-", "")
            if xl in cl:
                return c
    return None

def to_pct(s):
    s = pd.to_numeric(s, errors="coerce")
    med = s.abs().median()
    if pd.notna(med) and med <= 1.5:
        return s * 100.0
    return s

def config_metrics_for_target(row):
    target_id = str(row["target_id"])
    model_key = str(row["model_key"])

    dense_dir = locate_dense_run_dir(model_key)
    comp_dir = locate_compressed_run_dir(target_id)

    total = 0
    dense_correct = 0
    comp_correct = 0
    pred_flip = 0
    damage = 0
    repair = 0
    wrong_to_wrong = 0
    correct_to_correct_flip = 0
    missing = 0

    task_rows = []

    for task in TASKS:
        dsf = latest_sample_file(dense_dir, task)
        csf = latest_sample_file(comp_dir, task)
        if dsf is None or csf is None:
            task_rows.append({
                "target_id": target_id,
                "task": task,
                "status": "missing_sample_file",
                "n": 0,
            })
            continue

        drows = read_jsonl(dsf)
        crows = read_jsonl(csf)
        n = min(len(drows), len(crows))

        t_total = t_dc = t_cc = t_pf = t_dmg = t_rep = t_w2w = t_c2c = 0
        for i in range(n):
            try:
                dp = pred_from_sample(drows[i])
                cp = pred_from_sample(crows[i])
                gold = gold_from_sample(drows[i])
                if gold is None:
                    gold = gold_from_sample(crows[i])
            except Exception:
                missing += 1
                continue

            d_ok = (gold is not None and dp == gold)
            c_ok = (gold is not None and cp == gold)
            flip = (dp != cp)
            cat = category_from_preds(dp, cp, gold)

            total += 1
            t_total += 1
            dense_correct += int(d_ok)
            comp_correct += int(c_ok)
            t_dc += int(d_ok)
            t_cc += int(c_ok)
            pred_flip += int(flip)
            t_pf += int(flip)

            if cat == "damage":
                damage += 1
                t_dmg += 1
            elif cat == "repair":
                repair += 1
                t_rep += 1
            elif cat == "wrong_to_wrong":
                wrong_to_wrong += 1
                t_w2w += 1
            elif cat == "correct_to_correct_flip":
                correct_to_correct_flip += 1
                t_c2c += 1

        if t_total:
            task_rows.append({
                "target_id": target_id,
                "task": task,
                "status": "ok",
                "n": t_total,
                "dense_acc_pct": 100 * t_dc / t_total,
                "comp_acc_pct": 100 * t_cc / t_total,
                "delta_acc_pp": 100 * (t_cc - t_dc) / t_total,
                "predflip_pct": 100 * t_pf / t_total,
                "damage_pct": 100 * t_dmg / t_total,
                "repair_pct": 100 * t_rep / t_total,
                "w2w_pct": 100 * t_w2w / t_total,
                "c2c_flip_pct": 100 * t_c2c / t_total,
                "hidden_churn_pct": 100 * t_pf / t_total - abs(100 * (t_cc - t_dc) / t_total),
            })

    if total == 0:
        return None, task_rows

    delta = 100 * (comp_correct - dense_correct) / total
    pf = 100 * pred_flip / total
    out = {
        "target_id": target_id,
        "model_key": row.get("model_key", ""),
        "method_key": row.get("method_key", ""),
        "keep_tag": row.get("keep_tag", ""),
        "n": total,
        "missing_parse": missing,
        "dense_acc_pct_parsed": 100 * dense_correct / total,
        "comp_acc_pct_parsed": 100 * comp_correct / total,
        "delta_acc_pp_parsed": delta,
        "predflip_pct_parsed": pf,
        "damage_pct_parsed": 100 * damage / total,
        "repair_pct_parsed": 100 * repair / total,
        "w2w_pct_parsed": 100 * wrong_to_wrong / total,
        "c2c_flip_pct_parsed": 100 * correct_to_correct_flip / total,
        "hidden_churn_pct_parsed": pf - abs(delta),
    }
    return out, task_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_targets", type=int, default=None)
    ap.add_argument("--only_target", default=None)
    args = ap.parse_args()

    if not (ROOT / "scripts/mechanism/mech_common_v1.py").exists():
        raise FileNotFoundError("scripts/mechanism/mech_common_v1.py not found.")

    if not FROZEN.exists():
        raise FileNotFoundError(FROZEN)

    targets_csv = OUT / "targets_fullzoo.csv"
    if targets_csv.exists():
        targets = pd.read_csv(targets_csv)
    else:
        targets = load_targets_csv()
        targets = targets.sort_values(["model_key", "method_key", "keep_tag", "target_id"]).reset_index(drop=True)
        targets["array_idx"] = range(len(targets))
        targets.to_csv(targets_csv, index=False)

    if args.only_target:
        targets = targets[targets["target_id"].astype(str) == args.only_target].copy()
    if args.max_targets:
        targets = targets.head(args.max_targets).copy()

    print("N_TARGETS_TO_PARSE", len(targets))
    print("TARGET_HEAD")
    print(targets[["target_id", "model_key", "method_key", "keep_tag"]].head(10).to_string(index=False))

    rows = []
    task_rows_all = []
    for k, row in targets.iterrows():
        print(f"[{len(rows)+1}/{len(targets)}] parsing {row['target_id']}", flush=True)
        m, tr = config_metrics_for_target(row)
        if m is not None:
            rows.append(m)
        task_rows_all.extend(tr)

    parsed = pd.DataFrame(rows)
    parsed_path = AUD / "mech_label_parsed_by_config_v1.csv"
    parsed.to_csv(parsed_path, index=False)
    print("WROTE", parsed_path, parsed.shape)

    task_df = pd.DataFrame(task_rows_all)
    task_path = AUD / "mech_label_parsed_by_task_v1.csv"
    task_df.to_csv(task_path, index=False)
    print("WROTE", task_path, task_df.shape)

    frozen = pd.read_csv(FROZEN)
    print("\nFROZEN_COLUMNS")
    print(list(frozen.columns))
    print("\nPARSED_COLUMNS")
    print(list(parsed.columns))

    # Normalize join keys.
    f = frozen.copy()
    if "keep_tag" not in f.columns:
        kc = find_col(f, ["keep_tag", "keep_ratio", "keep"])
        if kc:
            f["keep_tag"] = f[kc].map(norm_keep)

    if "method_key" not in f.columns:
        mc = find_col(f, ["method_key", "method"])
        if mc:
            f["method_key"] = f[mc].astype(str)

    if "model_key" not in f.columns:
        mc = find_col(f, ["model_key", "model"])
        if mc:
            f["model_key"] = f[mc].astype(str)

    join_cols = [c for c in ["target_id"] if c in f.columns and c in parsed.columns]
    if not join_cols:
        join_cols = [c for c in ["model_key", "method_key", "keep_tag"] if c in f.columns and c in parsed.columns]

    print("\nJOIN_COLS", join_cols)
    if not join_cols:
        print("NO_JOIN_KEYS_FOUND")
        return

    merged = parsed.merge(f, on=join_cols, how="left", suffixes=("_parsed", "_frozen"))
    merged_path = AUD / "mech_label_alignment_merged_v1.csv"
    merged.to_csv(merged_path, index=False)
    print("WROTE", merged_path, merged.shape)

    # Candidate frozen metric columns.
    frozen_delta_col = find_col(f, ["delta_acc_pp", "delta_acc", "acc_delta", "acc_delta_pp", "mean_delta_acc", "dacc", "DeltaAcc"])
    frozen_pf_col = find_col(f, ["predflip_pct", "pred_flip_pct", "pred_flip", "prediction_flip", "predflip", "flip_rate"])
    frozen_hc_col = find_col(f, ["hidden_churn_pct", "hidden_churn", "HiddenChurn"])

    print("\nMETRIC_COLS")
    print("frozen_delta_col =", frozen_delta_col)
    print("frozen_pf_col    =", frozen_pf_col)
    print("frozen_hc_col    =", frozen_hc_col)

    checks = []
    if frozen_delta_col and frozen_delta_col in merged.columns:
        merged["delta_acc_pp_frozen_norm"] = to_pct(merged[frozen_delta_col])
        merged["delta_acc_absdiff"] = (merged["delta_acc_pp_parsed"] - merged["delta_acc_pp_frozen_norm"]).abs()
        checks.append(("delta_acc_absdiff", "delta_acc_pp_parsed", "delta_acc_pp_frozen_norm"))
    if frozen_pf_col and frozen_pf_col in merged.columns:
        merged["predflip_pct_frozen_norm"] = to_pct(merged[frozen_pf_col])
        merged["predflip_absdiff"] = (merged["predflip_pct_parsed"] - merged["predflip_pct_frozen_norm"]).abs()
        checks.append(("predflip_absdiff", "predflip_pct_parsed", "predflip_pct_frozen_norm"))
    if frozen_hc_col and frozen_hc_col in merged.columns:
        merged["hidden_churn_pct_frozen_norm"] = to_pct(merged[frozen_hc_col])
        merged["hidden_churn_absdiff"] = (merged["hidden_churn_pct_parsed"] - merged["hidden_churn_pct_frozen_norm"]).abs()
        checks.append(("hidden_churn_absdiff", "hidden_churn_pct_parsed", "hidden_churn_pct_frozen_norm"))

    merged.to_csv(merged_path, index=False)

    print("\nALIGNMENT_SUMMARY")
    for diff_col, parsed_col, frozen_col in checks:
        x = merged[diff_col]
        print(diff_col)
        print("  n_matched          =", int(x.notna().sum()))
        print("  mean_absdiff_pp    =", float(x.mean()))
        print("  median_absdiff_pp  =", float(x.median()))
        print("  max_absdiff_pp     =", float(x.max()))
        print("  <=0.05pp           =", int((x <= 0.05).sum()))
        print("  <=0.10pp           =", int((x <= 0.10).sum()))
        print("  <=0.50pp           =", int((x <= 0.50).sum()))

        show_cols = join_cols + ["n", parsed_col, frozen_col, diff_col]
        print("  worst 10:")
        print(merged.sort_values(diff_col, ascending=False)[show_cols].head(10).to_string(index=False))

    print("\nGLOBAL_PARSED_MEANS")
    for c in ["delta_acc_pp_parsed", "predflip_pct_parsed", "hidden_churn_pct_parsed", "damage_pct_parsed", "repair_pct_parsed", "w2w_pct_parsed"]:
        if c in parsed.columns:
            print(c, float(parsed[c].mean()))

    print("\nDONE")

if __name__ == "__main__":
    main()
