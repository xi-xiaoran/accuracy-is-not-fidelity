import sys
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd


from scripts.parsing.build_fullzoo_lowrank_pair_metrics_tables import (
    TASKS,
    COMP_CSV,
    DENSE_CSV,
    load_task_samples,
    sample_key,
    norm_char_scores,
    extract_gold,
    softmax,
    argmax,
    js,
    kl,
)

TARGET_ID = "llama2_7b_modegpt_keep0p8"
CAL_PER_TASK = 256
OUT_DIR = Path("results/method_pilot")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-12


def nll(p, gold):
    return -math.log(max(float(p[gold]), EPS))


def make_rows():
    dense_targets = {r["model_key"]: r for r in pd.read_csv(DENSE_CSV).to_dict("records")}
    comp_targets = pd.read_csv(COMP_CSV).to_dict("records")
    comp = [r for r in comp_targets if r["target_id"] == TARGET_ID]
    assert len(comp) == 1, f"Cannot find target {TARGET_ID}"
    comp = comp[0]
    dense = dense_targets[comp["model_key"]]

    rows = []

    for task in TASKS:
        dense_file, dense_rows, _ = load_task_samples(Path(dense["output_dir"]), task)
        comp_file, comp_rows, _ = load_task_samples(Path(comp["output_dir"]), task)

        dense_map = {sample_key(s, i): s for i, s in enumerate(dense_rows)}
        comp_map = {sample_key(s, i): s for i, s in enumerate(comp_rows)}
        keys = sorted(set(dense_map) & set(comp_map))

        task_rows = []
        for local_i, k in enumerate(keys):
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

            task_rows.append({
                "task": task,
                "sample_key": k,
                "doc_id": ds.get("doc_id", local_i),
                "gold": gold,
                "dense_scores": d_scores,
                "compressed_scores": c_scores,
            })

        # Deterministic split: first CAL_PER_TASK examples for calibration, rest for eval.
        task_rows = sorted(task_rows, key=lambda x: str(x["doc_id"]))
        for i, r in enumerate(task_rows):
            r["split"] = "cal" if i < CAL_PER_TASK else "eval"

        rows.extend(task_rows)

        print(
            "TASK",
            task,
            "n_total", len(task_rows),
            "n_cal", sum(1 for r in task_rows if r["split"] == "cal"),
            "n_eval", sum(1 for r in task_rows if r["split"] == "eval"),
            flush=True,
        )

    return rows


def transform_scores(scores, method, params):
    scores = np.array(scores, dtype=np.float64)

    if method == "identity":
        return scores

    if method in {"temp_ce", "temp_densekl"}:
        T = float(params["T"])
        return scores / max(T, EPS)

    if method == "affine_ce":
        a = float(params["a"])
        b = float(params["b"])
        return a * scores + b

    raise ValueError(method)


def objective_ce(rows, method, params):
    vals = []
    for r in rows:
        p = softmax(transform_scores(r["compressed_scores"], method, params).tolist())
        vals.append(nll(p, r["gold"]))
    return float(np.mean(vals))


def objective_densekl(rows, method, params):
    vals = []
    for r in rows:
        dp = softmax(r["dense_scores"])
        cp = softmax(transform_scores(r["compressed_scores"], method, params).tolist())
        vals.append(kl(dp, cp))
    return float(np.mean(vals))


def fit_temp(rows, objective):
    grid = np.exp(np.linspace(np.log(0.20), np.log(5.00), 161))
    best = None
    for T in grid:
        params = {"T": float(T)}
        val = objective(rows, "temp_ce" if objective == objective_ce else "temp_densekl", params)
        if best is None or val < best[0]:
            best = (val, params)
    return best[1], best[0]


def fit_affine_ce(rows):
    # Global affine a*s+b. b is ranking-invariant but included for completeness.
    # Since softmax is shift-invariant, b should not matter; this sanity-checks that
    # simple monotonic score calibration cannot repair churn.
    a_grid = np.exp(np.linspace(np.log(0.20), np.log(5.00), 121))
    b_grid = [0.0]
    best = None
    for a in a_grid:
        for b in b_grid:
            params = {"a": float(a), "b": float(b)}
            val = objective_ce(rows, "affine_ce", params)
            if best is None or val < best[0]:
                best = (val, params)
    return best[1], best[0]


def eval_method(rows, method, params):
    n = 0
    dense_correct = 0
    repaired_correct = 0

    pred_flip = []
    prob_js = []
    dense_gold_prob = []
    repaired_gold_prob = []
    dense_nll = []
    repaired_nll = []

    damage = 0
    repair = 0
    wrong_to_wrong_flip = 0
    correct_to_correct_flip = 0

    for r in rows:
        dp = softmax(r["dense_scores"])
        rp = softmax(transform_scores(r["compressed_scores"], method, params).tolist())

        d_pred = argmax(r["dense_scores"])
        r_scores = transform_scores(r["compressed_scores"], method, params).tolist()
        rep_pred = argmax(r_scores)

        gold = r["gold"]
        dc = d_pred == gold
        rc = rep_pred == gold
        flip = d_pred != rep_pred

        n += 1
        dense_correct += int(dc)
        repaired_correct += int(rc)
        pred_flip.append(float(flip))
        prob_js.append(js(dp, rp))

        dense_gold_prob.append(dp[gold])
        repaired_gold_prob.append(rp[gold])
        dense_nll.append(nll(dp, gold))
        repaired_nll.append(nll(rp, gold))

        if dc and not rc:
            damage += 1
        elif (not dc) and rc:
            repair += 1
        elif (not dc) and (not rc) and flip:
            wrong_to_wrong_flip += 1
        elif dc and rc and flip:
            correct_to_correct_flip += 1

    dense_acc = dense_correct / n
    repaired_acc = repaired_correct / n
    acc_delta = repaired_acc - dense_acc
    pf = float(np.mean(pred_flip))

    return {
        "n_eval": n,
        "dense_acc": dense_acc,
        "repaired_acc": repaired_acc,
        "acc_delta": acc_delta,
        "pred_flip": pf,
        "hidden_churn": pf - abs(acc_delta),
        "damage_rate": damage / n,
        "repair_rate": repair / n,
        "wrong_to_wrong_flip_rate": wrong_to_wrong_flip / n,
        "correct_to_correct_flip_rate": correct_to_correct_flip / n,
        "prob_js": float(np.mean(prob_js)),
        "gold_prob_shift": float(np.mean(np.array(repaired_gold_prob) - np.array(dense_gold_prob))),
        "nll_shift": float(np.mean(np.array(repaired_nll) - np.array(dense_nll))),
        "method": method,
        **{f"param_{k}": v for k, v in params.items()},
    }


def main():
    rows = make_rows()
    cal = [r for r in rows if r["split"] == "cal"]
    ev = [r for r in rows if r["split"] == "eval"]

    print("TOTAL_ROWS", len(rows), "CAL", len(cal), "EVAL", len(ev), flush=True)

    methods = []

    methods.append(("identity", {}, None))

    p, obj = fit_temp(cal, objective_ce)
    methods.append(("temp_ce", p, obj))

    p, obj = fit_temp(cal, objective_densekl)
    methods.append(("temp_densekl", p, obj))

    p, obj = fit_affine_ce(cal)
    methods.append(("affine_ce", p, obj))

    out_rows = []
    for method, params, obj in methods:
        r = eval_method(ev, method, params)
        r["target_id"] = TARGET_ID
        r["cal_objective"] = obj
        out_rows.append(r)

    df = pd.DataFrame(out_rows)

    # Add percentage columns.
    for c in [
        "dense_acc", "repaired_acc", "acc_delta", "pred_flip", "hidden_churn",
        "damage_rate", "repair_rate", "wrong_to_wrong_flip_rate",
        "correct_to_correct_flip_rate", "prob_js", "gold_prob_shift"
    ]:
        df[c + "_pct"] = df[c] * 100.0

    out = OUT_DIR / f"{TARGET_ID}_score_calibration_baselines.csv"
    df.to_csv(out, index=False)

    print("WROTE", out)
    print()
    print("===== SCORE CALIBRATION PILOT =====")
    show_cols = [
        "method", "n_eval", "dense_acc_pct", "repaired_acc_pct", "acc_delta_pct",
        "pred_flip_pct", "hidden_churn_pct", "damage_rate_pct", "repair_rate_pct",
        "prob_js_pct", "gold_prob_shift_pct", "nll_shift",
        "param_T", "param_a", "cal_objective"
    ]
    for c in show_cols:
        if c not in df.columns:
            df[c] = np.nan
    print(df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
