import sys
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


from scripts.metrics.run_selective_score_repair_pilot_v2 import make_rows
from scripts.parsing.build_fullzoo_lowrank_pair_metrics_tables import softmax, argmax, js

OUT = Path("results/theory_predictions_v1")
OUT.mkdir(parents=True, exist_ok=True)

EPS = 1e-12


def nll(p, gold):
    return -math.log(max(float(p[gold]), EPS))


def center(v):
    x = np.array(v, dtype=np.float64)
    return x - x.mean()


def effective_rank_from_singular_values(s):
    s = np.asarray(s, dtype=np.float64)
    if len(s) == 0 or np.sum(s ** 2) <= 0:
        return 0.0
    p = (s ** 2) / np.sum(s ** 2)
    return float(np.exp(-np.sum(p * np.log(np.clip(p, EPS, 1.0)))))


def explained_by_topk(s, k):
    s = np.asarray(s, dtype=np.float64)
    total = np.sum(s ** 2)
    if total <= 0:
        return 0.0
    return float(np.sum(s[:k] ** 2) / total)


def sample_frame(rows):
    out = []

    for r in rows:
        d = np.array(r["dense_scores"], dtype=np.float64)
        c = np.array(r["compressed_scores"], dtype=np.float64)
        gold = int(r["gold"])

        if len(d) != len(c) or len(d) < 2:
            continue

        d_pred = int(argmax(d.tolist()))
        c_pred = int(argmax(c.tolist()))

        dense_margin = float(d[d_pred] - np.max(np.delete(d, d_pred)))

        # Compression perturbation in score space:
        # compressed = dense + delta.
        delta = c - d

        # For dense winner y*, flip can occur if some competitor j gains over y*
        # by more than dense margin.
        competitors = [j for j in range(len(d)) if j != d_pred]
        adv_gap_delta = max(float(delta[j] - delta[d_pred]) for j in competitors)
        boundary_ratio = adv_gap_delta / max(dense_margin, EPS)

        dp = softmax(d.tolist())
        cp = softmax(c.tolist())

        dense_correct = d_pred == gold
        compressed_correct = c_pred == gold
        flip = d_pred != c_pred

        damage = dense_correct and not compressed_correct
        repair = (not dense_correct) and compressed_correct
        wrong_to_wrong = (not dense_correct) and (not compressed_correct) and flip
        correct_to_correct = dense_correct and compressed_correct and flip

        corr = center(d - c)
        corr_norm = float(np.linalg.norm(corr))

        out.append({
            "task": r["task"],
            "split": r["split"],
            "sample_key": r["sample_key"],
            "gold": gold,
            "n_choices": len(d),
            "risk": float(r["risk"]),
            "dense_pred": d_pred,
            "compressed_pred": c_pred,
            "dense_correct": dense_correct,
            "compressed_correct": compressed_correct,
            "flip": flip,
            "damage": damage,
            "repair": repair,
            "wrong_to_wrong": wrong_to_wrong,
            "correct_to_correct": correct_to_correct,
            "dense_margin": dense_margin,
            "adv_gap_delta": adv_gap_delta,
            "boundary_ratio": boundary_ratio,
            "score_l2_delta": float(np.linalg.norm(delta)),
            "centered_correction_norm": corr_norm,
            "prob_js": js(dp, cp),
            "dense_gold_prob": dp[gold],
            "compressed_gold_prob": cp[gold],
            "gold_prob_shift": cp[gold] - dp[gold],
            "nll_shift": nll(cp, gold) - nll(dp, gold),
        })

    return pd.DataFrame(out)


def aggregate_binary(df, group_cols):
    rows = []

    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        dense_acc = g["dense_correct"].mean()
        comp_acc = g["compressed_correct"].mean()
        acc_delta = comp_acc - dense_acc
        pred_flip = g["flip"].mean()

        row = {c: v for c, v in zip(group_cols, key)}
        row.update({
            "n": len(g),
            "dense_acc": dense_acc,
            "compressed_acc": comp_acc,
            "acc_delta": acc_delta,
            "pred_flip": pred_flip,
            "hidden_churn": pred_flip - abs(acc_delta),
            "damage": g["damage"].mean(),
            "repair": g["repair"].mean(),
            "wrong_to_wrong": g["wrong_to_wrong"].mean(),
            "correct_to_correct": g["correct_to_correct"].mean(),
            "prob_js": g["prob_js"].mean(),
            "gold_prob_shift": g["gold_prob_shift"].mean(),
            "nll_shift": g["nll_shift"].mean(),
            "mean_dense_margin": g["dense_margin"].mean(),
            "mean_boundary_ratio": g["boundary_ratio"].replace([np.inf, -np.inf], np.nan).mean(),
            "mean_risk": g["risk"].mean(),
        })
        rows.append(row)

    return pd.DataFrame(rows)


def add_pct_cols(df):
    for c in [
        "dense_acc", "compressed_acc", "repaired_acc", "acc_delta", "pred_flip", "hidden_churn",
        "damage", "repair", "wrong_to_wrong", "correct_to_correct",
        "prob_js", "gold_prob_shift",
    ]:
        if c in df.columns:
            df[c + "_pct"] = df[c] * 100.0
    return df


def boundary_ratio_analysis(df, target_id):
    ev = df[df["split"] == "eval"].copy()

    bins = [-np.inf, -1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0, np.inf]
    labels = ["<-1", "-1~0", "0~0.25", "0.25~0.5", "0.5~0.75", "0.75~1", "1~2", ">2"]
    ev["boundary_ratio_bucket"] = pd.cut(ev["boundary_ratio"], bins=bins, labels=labels)

    out = aggregate_binary(ev, ["boundary_ratio_bucket"])
    out = add_pct_cols(out)

    csv = OUT / f"{target_id}_boundary_ratio_buckets.csv"
    out.to_csv(csv, index=False)

    print()
    print("===== A. BOUNDARY AMPLIFICATION RATIO =====")
    show = [
        "boundary_ratio_bucket", "n", "mean_boundary_ratio", "mean_dense_margin",
        "pred_flip_pct", "hidden_churn_pct", "damage_pct", "repair_pct", "prob_js_pct"
    ]
    print(out[show].to_string(index=False))
    print("WROTE", csv)

    # Also per task.
    ev["boundary_ratio_qbucket"] = pd.qcut(
        ev["boundary_ratio"].replace([np.inf, -np.inf], np.nan).fillna(ev["boundary_ratio"].replace([np.inf, -np.inf], np.nan).max()),
        q=5,
        duplicates="drop",
    )
    task_out = aggregate_binary(ev, ["task", "boundary_ratio_qbucket"])
    task_out = add_pct_cols(task_out)
    task_csv = OUT / f"{target_id}_boundary_ratio_by_task.csv"
    task_out.to_csv(task_csv, index=False)
    print("WROTE", task_csv)


def correction_rank_analysis(rows, df, target_id):
    # Analyze correction directions R = dense_scores - compressed_scores.
    # Use centered correction because adding a constant to all choice scores is behavior-invariant.
    cal_rows = [r for r in rows if r["split"] == "cal"]
    ev_rows = [r for r in rows if r["split"] == "eval"]

    meta = df.set_index(["split", "sample_key"])

    groups = []

    def add_group(name, selected):
        if len(selected) < 5:
            return

        # Group by n_choices because score vector dimensions differ across tasks.
        by_n = {}
        for r in selected:
            n = len(r["dense_scores"])
            by_n.setdefault(n, []).append(r)

        for n, rs in by_n.items():
            if len(rs) < 5:
                continue

            mat = []
            for r in rs:
                corr = center(np.array(r["dense_scores"], dtype=np.float64) - np.array(r["compressed_scores"], dtype=np.float64))
                mat.append(corr)

            X = np.stack(mat, axis=0)
            X = X - X.mean(axis=0, keepdims=True)

            try:
                s = np.linalg.svd(X, compute_uv=False)
            except np.linalg.LinAlgError:
                continue

            # Direction consistency.
            norms = np.linalg.norm(X, axis=1)
            valid = norms > EPS
            if valid.sum() >= 2:
                Xn = X[valid] / norms[valid, None]
                # Sample pairwise cosine for stability.
                rng = np.random.default_rng(0)
                m = min(len(Xn), 500)
                idx = rng.choice(len(Xn), size=m, replace=False)
                Xs = Xn[idx]
                cos = Xs @ Xs.T
                tri = cos[np.triu_indices_from(cos, k=1)]
                mean_abs_pair_cos = float(np.mean(np.abs(tri)))
                mean_pair_cos = float(np.mean(tri))
            else:
                mean_abs_pair_cos = np.nan
                mean_pair_cos = np.nan

            groups.append({
                "group": name,
                "n_choices": n,
                "n": len(rs),
                "effective_rank": effective_rank_from_singular_values(s),
                "top1_explained": explained_by_topk(s, 1),
                "top2_explained": explained_by_topk(s, 2),
                "top3_explained": explained_by_topk(s, 3),
                "top4_explained": explained_by_topk(s, 4),
                "mean_abs_pair_cos": mean_abs_pair_cos,
                "mean_pair_cos": mean_pair_cos,
                "singular_values": "|".join(f"{x:.6g}" for x in s[:10]),
            })

    add_group("eval_all", ev_rows)

    high_risk_keys = set(
        tuple(x) for x in df[(df["split"] == "eval") & (df["risk"] >= df[df["split"] == "cal"]["risk"].quantile(0.8))][["split", "sample_key"]].values
    )
    mismatch_keys = set(
        tuple(x) for x in df[(df["split"] == "eval") & (df["flip"])][["split", "sample_key"]].values
    )

    add_group("eval_highrisk_q80", [r for r in ev_rows if ("eval", r["sample_key"]) in high_risk_keys])
    add_group("eval_mismatch", [r for r in ev_rows if ("eval", r["sample_key"]) in mismatch_keys])
    add_group("eval_highrisk_mismatch", [r for r in ev_rows if ("eval", r["sample_key"]) in high_risk_keys and ("eval", r["sample_key"]) in mismatch_keys])

    # Per task high-risk mismatch.
    for task in sorted(set(r["task"] for r in ev_rows)):
        add_group(
            f"task={task}:highrisk_mismatch",
            [r for r in ev_rows if r["task"] == task and ("eval", r["sample_key"]) in high_risk_keys and ("eval", r["sample_key"]) in mismatch_keys]
        )

    rank_df = pd.DataFrame(groups)
    for c in ["top1_explained", "top2_explained", "top3_explained", "top4_explained", "mean_abs_pair_cos", "mean_pair_cos"]:
        rank_df[c + "_pct"] = rank_df[c] * 100.0

    csv = OUT / f"{target_id}_correction_direction_rank.csv"
    rank_df.to_csv(csv, index=False)

    print()
    print("===== B. CORRECTION DIRECTION RANK =====")
    show = [
        "group", "n_choices", "n", "effective_rank",
        "top1_explained_pct", "top2_explained_pct", "top3_explained_pct",
        "mean_abs_pair_cos_pct", "mean_pair_cos_pct"
    ]
    print(rank_df[show].sort_values(["group", "n_choices"]).to_string(index=False))
    print("WROTE", csv)


def eval_scores(rows, method_name, score_fn):
    n = 0
    dense_correct = 0
    repaired_correct = 0

    pred_flip = []
    prob_js = []
    dense_gold_prob = []
    repaired_gold_prob = []
    dense_nll = []
    repaired_nll = []

    damage = repair = wrong_to_wrong = correct_to_correct = 0

    for r in rows:
        d = np.array(r["dense_scores"], dtype=np.float64)
        scores = np.array(score_fn(r), dtype=np.float64)

        dp = softmax(d.tolist())
        rp = softmax(scores.tolist())

        d_pred = int(argmax(d.tolist()))
        r_pred = int(argmax(scores.tolist()))
        gold = int(r["gold"])

        dc = d_pred == gold
        rc = r_pred == gold
        flip = d_pred != r_pred

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
            wrong_to_wrong += 1
        elif dc and rc and flip:
            correct_to_correct += 1

    dense_acc = dense_correct / n
    rep_acc = repaired_correct / n
    acc_delta = rep_acc - dense_acc
    pf = float(np.mean(pred_flip))

    return {
        "method": method_name,
        "n_eval": n,
        "dense_acc": dense_acc,
        "repaired_acc": rep_acc,
        "acc_delta": acc_delta,
        "pred_flip": pf,
        "hidden_churn": pf - abs(acc_delta),
        "damage": damage / n,
        "repair": repair / n,
        "wrong_to_wrong": wrong_to_wrong / n,
        "correct_to_correct": correct_to_correct / n,
        "prob_js": float(np.mean(prob_js)),
        "gold_prob_shift": float(np.mean(np.array(repaired_gold_prob) - np.array(dense_gold_prob))),
        "nll_shift": float(np.mean(np.array(repaired_nll) - np.array(dense_nll))),
    }


def fit_group_means(cal_rows, key_fn):
    groups = {}
    for r in cal_rows:
        key = key_fn(r)
        d = np.array(r["dense_scores"], dtype=np.float64)
        c = np.array(r["compressed_scores"], dtype=np.float64)
        corr = center(d - c)
        groups.setdefault(key, []).append(corr)

    means = {}
    counts = {}
    for k, xs in groups.items():
        # Only fit if all vectors in group share dimension.
        dims = {len(x) for x in xs}
        if len(dims) != 1:
            continue
        means[k] = np.mean(np.stack(xs, axis=0), axis=0)
        counts[k] = len(xs)

    return means, counts


def correction_energy_eval(ev_rows, means, key_fn, fallback=None):
    total = 0.0
    residual = 0.0
    used = 0

    for r in ev_rows:
        d = np.array(r["dense_scores"], dtype=np.float64)
        c = np.array(r["compressed_scores"], dtype=np.float64)
        target = center(d - c)

        key = key_fn(r)
        pred = means.get(key, None)
        if pred is None and fallback is not None:
            pred = fallback.get(("n", len(target)), np.zeros_like(target))
        if pred is None or len(pred) != len(target):
            pred = np.zeros_like(target)

        total += float(np.sum(target ** 2))
        residual += float(np.sum((target - pred) ** 2))
        used += 1

    explained = 1.0 - residual / max(total, EPS)
    return {
        "correction_energy_total": total,
        "correction_energy_residual": residual,
        "correction_energy_explained": explained,
        "n_used": used,
    }


def projection_analysis(rows, df, target_id):
    cal = [r for r in rows if r["split"] == "cal"]
    ev = [r for r in rows if r["split"] == "eval"]

    # Fallback by n_choices = global bias per dimension.
    fallback, _ = fit_group_means(cal, lambda r: ("n", len(r["dense_scores"])))

    risk_cal = df[df["split"] == "cal"]["risk"]
    tau80 = float(risk_cal.quantile(0.8))
    tau90 = float(risk_cal.quantile(0.9))

    key_fns = {
        "global_by_nchoices": lambda r: ("n", len(r["dense_scores"])),
        "task_bias": lambda r: ("task", r["task"], len(r["dense_scores"])),
        "task_compressed_pred_bias": lambda r: ("task_pred", r["task"], len(r["dense_scores"]), int(argmax(r["compressed_scores"]))),
        "task_risk80_pred_bias": lambda r: (
            "task_risk80_pred",
            r["task"],
            len(r["dense_scores"]),
            int(r["risk"] >= tau80),
            int(argmax(r["compressed_scores"])),
        ),
        "task_risk90_pred_bias": lambda r: (
            "task_risk90_pred",
            r["task"],
            len(r["dense_scores"]),
            int(r["risk"] >= tau90),
            int(argmax(r["compressed_scores"])),
        ),
    }

    out_rows = []
    out_rows.append(eval_scores(ev, "identity_compressed", lambda r: r["compressed_scores"]))

    energy_rows = []

    for name, key_fn in key_fns.items():
        means, counts = fit_group_means(cal, key_fn)

        energy = correction_energy_eval(ev, means, key_fn, fallback=fallback)
        energy["method"] = name
        energy["n_groups"] = len(means)
        energy["mean_group_count"] = float(np.mean(list(counts.values()))) if counts else 0.0
        energy_rows.append(energy)

        def score_fn(r, m=means, kfn=key_fn):
            c = np.array(r["compressed_scores"], dtype=np.float64)
            key = kfn(r)
            pred = m.get(key, None)
            if pred is None:
                pred = fallback.get(("n", len(c)), np.zeros_like(c))
            if len(pred) != len(c):
                pred = np.zeros_like(c)
            return (c + pred).tolist()

        out_rows.append(eval_scores(ev, name, score_fn))

    proj_df = pd.DataFrame(out_rows)
    proj_df = add_pct_cols(proj_df)

    energy_df = pd.DataFrame(energy_rows)
    energy_df["correction_energy_explained_pct"] = energy_df["correction_energy_explained"] * 100.0

    csv1 = OUT / f"{target_id}_projection_eval.csv"
    csv2 = OUT / f"{target_id}_projection_energy.csv"
    proj_df.to_csv(csv1, index=False)
    energy_df.to_csv(csv2, index=False)

    print()
    print("===== C1. PROJECTION EVAL =====")
    show = [
        "method", "n_eval", "dense_acc_pct", "repaired_acc_pct", "acc_delta_pct",
        "pred_flip_pct", "hidden_churn_pct", "damage_pct", "repair_pct", "prob_js_pct", "nll_shift"
    ]
    print(proj_df[show].to_string(index=False))
    print("WROTE", csv1)

    print()
    print("===== C2. PROJECTION ENERGY =====")
    show2 = [
        "method", "n_groups", "mean_group_count", "correction_energy_explained_pct",
        "correction_energy_total", "correction_energy_residual"
    ]
    print(energy_df[show2].sort_values("correction_energy_explained_pct", ascending=False).to_string(index=False))
    print("WROTE", csv2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_id", default="llama2_7b_modegpt_keep0p8")
    ap.add_argument("--cal_per_task", type=int, default=128)
    ap.add_argument("--eval_per_task", type=int, default=512)
    args = ap.parse_args()

    print("TARGET", args.target_id)
    rows, task_to_id = make_rows(args.target_id, args.cal_per_task, args.eval_per_task)

    df = sample_frame(rows)
    sample_csv = OUT / f"{args.target_id}_sample_theory_frame.csv"
    df.to_csv(sample_csv, index=False)
    print("WROTE", sample_csv)

    # Base decomposition sanity.
    ev = df[df["split"] == "eval"]
    dense_acc = ev["dense_correct"].mean()
    comp_acc = ev["compressed_correct"].mean()
    acc_delta = comp_acc - dense_acc
    pred_flip = ev["flip"].mean()
    hidden = pred_flip - abs(acc_delta)
    identity = {
        "dense_acc_pct": dense_acc * 100,
        "compressed_acc_pct": comp_acc * 100,
        "acc_delta_pct": acc_delta * 100,
        "pred_flip_pct": pred_flip * 100,
        "hidden_churn_pct": hidden * 100,
        "damage_pct": ev["damage"].mean() * 100,
        "repair_pct": ev["repair"].mean() * 100,
        "wrong_to_wrong_pct": ev["wrong_to_wrong"].mean() * 100,
        "correct_to_correct_pct": ev["correct_to_correct"].mean() * 100,
        "identity_check_hidden_pct": (2 * min(ev["damage"].mean(), ev["repair"].mean()) + ev["wrong_to_wrong"].mean() + ev["correct_to_correct"].mean()) * 100,
    }

    print()
    print("===== BASE IDENTITY / HIDDEN CHURN CHECK =====")
    for k, v in identity.items():
        print(f"{k}: {v:.6f}")

    boundary_ratio_analysis(df, args.target_id)
    correction_rank_analysis(rows, df, args.target_id)
    projection_analysis(rows, df, args.target_id)

    print()
    print("DONE theory predictions v1")


if __name__ == "__main__":
    main()
