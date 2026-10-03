import sys
import math
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


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

OUT_DIR = Path("results/method_pilot/score_rank_repair_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-12


def nll(p, gold):
    return -math.log(max(float(p[gold]), EPS))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_rows(target_id, cal_per_task, eval_per_task):
    dense_targets = {r["model_key"]: r for r in pd.read_csv(DENSE_CSV).to_dict("records")}
    comp_targets = pd.read_csv(COMP_CSV).to_dict("records")

    comp = [r for r in comp_targets if r["target_id"] == target_id]
    assert len(comp) == 1, f"Cannot find target_id={target_id}"
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

            cp = softmax(c_scores)
            sorted_p = sorted(cp, reverse=True)
            margin = sorted_p[0] - sorted_p[1]
            ent = -sum(float(x) * math.log(max(float(x), EPS)) for x in cp)
            ent_norm = ent / math.log(len(cp))
            risk = 0.5 * (1.0 - margin) + 0.5 * ent_norm
            risk = float(max(0.0, min(1.0, risk)))

            task_rows.append({
                "task": task,
                "sample_key": k,
                "doc_id": ds.get("doc_id", local_i),
                "gold": gold,
                "dense_scores": d_scores,
                "compressed_scores": c_scores,
                "dense_pred": argmax(d_scores),
                "compressed_pred": argmax(c_scores),
                "compressed_probs": cp,
                "risk": risk,
                "n_choices": len(c_scores),
            })

        task_rows = sorted(task_rows, key=lambda x: str(x["doc_id"]))
        cal = task_rows[:cal_per_task]
        ev = task_rows[cal_per_task: cal_per_task + eval_per_task]

        for r in cal:
            r["split"] = "cal"
        for r in ev:
            r["split"] = "eval"

        rows.extend(cal)
        rows.extend(ev)

        print("TASK", task, "n_total", len(task_rows), "n_cal", len(cal), "n_eval", len(ev), flush=True)

    return rows


class ScoreRepair(torch.nn.Module):
    def __init__(self, max_choices, mode):
        super().__init__()
        self.max_choices = max_choices
        self.mode = mode

        self.log_scale = torch.nn.Parameter(torch.zeros(max_choices))
        self.bias = torch.nn.Parameter(torch.zeros(max_choices))

    def forward(self, scores):
        n = scores.numel()
        if self.mode == "bias":
            return scores + self.bias[:n]
        if self.mode == "affine":
            scale = torch.exp(self.log_scale[:n])
            return scale * scores + self.bias[:n]
        raise ValueError(self.mode)

    def reg(self):
        return torch.sum(self.log_scale ** 2) + torch.sum(self.bias ** 2)


def transform_np(scores, model):
    with torch.no_grad():
        x = torch.tensor(scores, dtype=torch.float32)
        y = model(x).detach().cpu().numpy().tolist()
    return y


def row_loss(row, model, objective, lam_preserve, lam_mismatch_anchor, beta, eta_ce, rho, rank_margin):
    c_scores = torch.tensor(row["compressed_scores"], dtype=torch.float32)
    scores = model(c_scores)
    logp = F.log_softmax(scores, dim=-1)

    gold = torch.tensor([row["gold"]], dtype=torch.long)
    gold_ce = F.cross_entropy(scores.unsqueeze(0), gold)

    comp = torch.tensor(row["compressed_probs"], dtype=torch.float32)
    comp_kl = torch.sum(comp * (torch.log(torch.clamp(comp, min=EPS)) - logp))

    dense = torch.tensor(softmax(row["dense_scores"]), dtype=torch.float32)
    dense_kl = torch.sum(dense * (torch.log(torch.clamp(dense, min=EPS)) - logp))

    dense_pred_int = int(row["dense_pred"])
    comp_pred_int = int(row["compressed_pred"])
    aligned = dense_pred_int == comp_pred_int

    dense_pred = torch.tensor([dense_pred_int], dtype=torch.long)
    dense_top1_ce = F.cross_entropy(scores.unsqueeze(0), dense_pred)

    others = [j for j in range(scores.numel()) if j != dense_pred_int]
    if others:
        other_idx = torch.tensor(others, dtype=torch.long)
        gaps = scores[dense_pred_int] - scores[other_idx]
        rank_hinge = torch.relu(rank_margin - gaps).mean()
    else:
        rank_hinge = torch.zeros(())

    risk = float(row["risk"])

    if objective == "ce":
        loss = gold_ce + rho * model.reg()

    elif objective == "densekl":
        loss = dense_kl + eta_ce * gold_ce + rho * model.reg()

    elif objective == "basc_kl":
        loss = beta * risk * dense_kl + lam_preserve * (1.0 - risk) * comp_kl + eta_ce * gold_ce + rho * model.reg()

    elif objective == "basc_rank":
        if aligned:
            loss = lam_preserve * comp_kl + rho * model.reg()
        else:
            loss = (
                beta * risk * (dense_top1_ce + rank_hinge)
                + lam_mismatch_anchor * comp_kl
                + eta_ce * gold_ce
                + rho * model.reg()
            )
    else:
        raise ValueError(objective)

    return loss


def train_model(cal_rows, max_choices, mode, objective, steps, lr,
                lam_preserve, lam_mismatch_anchor, beta, eta_ce, rho, rank_margin, seed):
    set_seed(seed)
    model = ScoreRepair(max_choices, mode)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    mismatched = [r for r in cal_rows if r["dense_pred"] != r["compressed_pred"]]
    print("TRAIN", mode, objective, "CAL", len(cal_rows), "MISMATCHED", len(mismatched), flush=True)

    losses = []

    for step in range(1, steps + 1):
        if objective == "basc_rank" and mismatched and random.random() < 0.70:
            row = random.choice(mismatched)
        else:
            row = random.choice(cal_rows)

        opt.zero_grad(set_to_none=True)
        loss = row_loss(row, model, objective, lam_preserve, lam_mismatch_anchor, beta, eta_ce, rho, rank_margin)
        loss.backward()
        opt.step()

        losses.append(float(loss.detach()))

        if step % 500 == 0:
            print(
                f"TRAIN {mode}/{objective} step={step}/{steps} loss={np.mean(losses[-500:]):.6f}",
                flush=True,
            )

    return model


def eval_method(rows, method_name, transform_fn):
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
    wrong_to_wrong = 0
    correct_to_correct = 0

    for row in rows:
        rep_scores = transform_fn(row["compressed_scores"])

        dp = softmax(row["dense_scores"])
        rp = softmax(rep_scores)

        d_pred = argmax(row["dense_scores"])
        r_pred = argmax(rep_scores)
        gold = row["gold"]

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
        "damage_rate": damage / n,
        "repair_rate": repair / n,
        "wrong_to_wrong_flip_rate": wrong_to_wrong / n,
        "correct_to_correct_flip_rate": correct_to_correct / n,
        "prob_js": float(np.mean(prob_js)),
        "gold_prob_shift": float(np.mean(np.array(repaired_gold_prob) - np.array(dense_gold_prob))),
        "nll_shift": float(np.mean(np.array(repaired_nll) - np.array(dense_nll))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_id", default="llama2_7b_modegpt_keep0p8")
    ap.add_argument("--cal_per_task", type=int, default=128)
    ap.add_argument("--eval_per_task", type=int, default=512)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--lam_preserve", type=float, default=5.0)
    ap.add_argument("--lam_mismatch_anchor", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--eta_ce", type=float, default=0.0)
    ap.add_argument("--rho", type=float, default=0.01)
    ap.add_argument("--rank_margin", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    rows = make_rows(args.target_id, args.cal_per_task, args.eval_per_task)
    cal = [r for r in rows if r["split"] == "cal"]
    ev = [r for r in rows if r["split"] == "eval"]

    max_choices = max(r["n_choices"] for r in rows)

    print("TOTAL", len(rows), "CAL", len(cal), "EVAL", len(ev), "MAX_CHOICES", max_choices, flush=True)

    out_rows = []

    out_rows.append(eval_method(ev, "identity_compressed", lambda s: s))

    configs = [
        ("bias", "ce"),
        ("bias", "densekl"),
        ("bias", "basc_kl"),
        ("bias", "basc_rank"),
        ("affine", "densekl"),
        ("affine", "basc_kl"),
        ("affine", "basc_rank"),
    ]

    saved_params = {}

    for i, (mode, objective) in enumerate(configs):
        model = train_model(
            cal, max_choices, mode, objective,
            steps=args.steps,
            lr=args.lr,
            lam_preserve=args.lam_preserve,
            lam_mismatch_anchor=args.lam_mismatch_anchor,
            beta=args.beta,
            eta_ce=args.eta_ce,
            rho=args.rho,
            rank_margin=args.rank_margin,
            seed=args.seed + i + 1,
        )

        name = f"{mode}_{objective}"
        out_rows.append(eval_method(ev, name, lambda s, m=model: transform_np(s, m)))

        saved_params[name] = {
            "log_scale": model.log_scale.detach().cpu().numpy().tolist(),
            "bias": model.bias.detach().cpu().numpy().tolist(),
        }

    df = pd.DataFrame(out_rows)

    for c in [
        "dense_acc", "repaired_acc", "acc_delta", "pred_flip", "hidden_churn",
        "damage_rate", "repair_rate", "wrong_to_wrong_flip_rate",
        "correct_to_correct_flip_rate", "prob_js", "gold_prob_shift"
    ]:
        df[c + "_pct"] = df[c] * 100.0

    out_csv = OUT_DIR / f"{args.target_id}_score_rank_repair_v1.csv"
    df.to_csv(out_csv, index=False)

    import json
    with open(OUT_DIR / f"{args.target_id}_score_rank_repair_v1_params.json", "w") as f:
        json.dump(saved_params, f, indent=2)

    print("WROTE", out_csv)
    print()
    print("===== SCORE RANK REPAIR RESULT =====")
    show = [
        "method", "n_eval", "dense_acc_pct", "repaired_acc_pct", "acc_delta_pct",
        "pred_flip_pct", "hidden_churn_pct", "damage_rate_pct", "repair_rate_pct",
        "prob_js_pct", "gold_prob_shift_pct", "nll_shift"
    ]
    print(df[show].to_string(index=False))


if __name__ == "__main__":
    main()
