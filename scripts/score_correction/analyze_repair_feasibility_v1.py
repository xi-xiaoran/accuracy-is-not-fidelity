import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd


from scripts.metrics.run_selective_score_repair_pilot_v2 import make_rows
from scripts.parsing.build_fullzoo_lowrank_pair_metrics_tables import softmax, argmax, js

OUT = Path("results/method_pilot/repair_feasibility_v1")
OUT.mkdir(parents=True, exist_ok=True)

EPS = 1e-12


def nll(p, gold):
    return -math.log(max(float(p[gold]), EPS))


def eval_with_preds(rows, pred_fn, prob_fn, name, tau=None):
    n = 0
    active_n = 0
    dense_correct = 0
    rep_correct = 0

    pred_flip = []
    prob_js = []
    dense_gold_prob = []
    rep_gold_prob = []
    dense_nll = []
    rep_nll = []

    damage = repair = wrong_to_wrong = correct_to_correct = 0

    for r in rows:
        active = tau is not None and r["risk"] >= tau
        active_n += int(active)

        d_scores = r["dense_scores"]
        d_pred = argmax(d_scores)
        d_prob = softmax(d_scores)

        rep_pred = pred_fn(r, active)
        rep_prob = prob_fn(r, active)

        gold = r["gold"]

        dc = d_pred == gold
        rc = rep_pred == gold
        flip = d_pred != rep_pred

        n += 1
        dense_correct += int(dc)
        rep_correct += int(rc)
        pred_flip.append(float(flip))
        prob_js.append(js(d_prob, rep_prob))

        dense_gold_prob.append(d_prob[gold])
        rep_gold_prob.append(rep_prob[gold])
        dense_nll.append(nll(d_prob, gold))
        rep_nll.append(nll(rep_prob, gold))

        if dc and not rc:
            damage += 1
        elif (not dc) and rc:
            repair += 1
        elif (not dc) and (not rc) and flip:
            wrong_to_wrong += 1
        elif dc and rc and flip:
            correct_to_correct += 1

    dense_acc = dense_correct / n
    rep_acc = rep_correct / n
    acc_delta = rep_acc - dense_acc
    pf = float(np.mean(pred_flip))

    return {
        "method": name,
        "tau": tau,
        "n_eval": n,
        "active_rate": active_n / n,
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
        "gold_prob_shift": float(np.mean(np.array(rep_gold_prob) - np.array(dense_gold_prob))),
        "nll_shift": float(np.mean(np.array(rep_nll) - np.array(dense_nll))),
    }


def onehot_prob(n, pred, eps=1e-6):
    p = np.ones(n) * eps
    p[pred] = 1.0 - eps * (n - 1)
    return p.tolist()


def main():
    target_id = "llama2_7b_modegpt_keep0p8"

    rows, task_to_id = make_rows(target_id, cal_per_task=128, eval_per_task=512)
    cal = [r for r in rows if r["split"] == "cal"]
    ev = [r for r in rows if r["split"] == "eval"]

    risks_cal = np.array([r["risk"] for r in cal])
    qs = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95]
    taus = [float(np.quantile(risks_cal, q)) for q in qs]

    print("TARGET", target_id)
    print("CAL", len(cal), "EVAL", len(ev))
    print("TAUS", list(zip(qs, taus)))

    # Gate quality table.
    gate_rows = []
    for q, tau in zip(qs, taus):
        for split_name, split_rows in [("cal", cal), ("eval", ev)]:
            active = [r for r in split_rows if r["risk"] >= tau]
            inactive = [r for r in split_rows if r["risk"] < tau]

            def stats(rs, prefix):
                if not rs:
                    return {
                        f"{prefix}_n": 0,
                        f"{prefix}_mismatch_rate": np.nan,
                        f"{prefix}_damage_rate": np.nan,
                        f"{prefix}_repair_rate": np.nan,
                        f"{prefix}_mean_risk": np.nan,
                    }

                mismatch = [r["dense_pred"] != r["compressed_pred"] for r in rs]

                damage = []
                repair = []
                for r in rs:
                    gold = r["gold"]
                    dc = r["dense_pred"] == gold
                    cc = r["compressed_pred"] == gold
                    damage.append(dc and not cc)
                    repair.append((not dc) and cc)

                return {
                    f"{prefix}_n": len(rs),
                    f"{prefix}_mismatch_rate": float(np.mean(mismatch)),
                    f"{prefix}_damage_rate": float(np.mean(damage)),
                    f"{prefix}_repair_rate": float(np.mean(repair)),
                    f"{prefix}_mean_risk": float(np.mean([r["risk"] for r in rs])),
                }

            total_mismatch = sum(r["dense_pred"] != r["compressed_pred"] for r in split_rows)
            active_mismatch = sum(r["dense_pred"] != r["compressed_pred"] for r in active)

            row = {
                "split": split_name,
                "q": q,
                "tau": tau,
                "n_total": len(split_rows),
                "active_rate": len(active) / len(split_rows),
                "mismatch_recall": active_mismatch / total_mismatch if total_mismatch else np.nan,
            }
            row.update(stats(active, "active"))
            row.update(stats(inactive, "inactive"))
            gate_rows.append(row)

    gate_df = pd.DataFrame(gate_rows)
    for c in gate_df.columns:
        if c.endswith("_rate") or c.endswith("_recall"):
            gate_df[c + "_pct"] = gate_df[c] * 100.0

    gate_out = OUT / f"{target_id}_gate_quality.csv"
    gate_df.to_csv(gate_out, index=False)

    # Oracle variants.
    out_rows = []

    # Identity compressed.
    out_rows.append(eval_with_preds(
        ev,
        pred_fn=lambda r, active: r["compressed_pred"],
        prob_fn=lambda r, active: softmax(r["compressed_scores"]),
        name="identity_compressed",
        tau=None,
    ))

    for q, tau in zip(qs, taus):
        # Oracle decision repair: active mismatches choose dense pred, inactive stays compressed.
        out_rows.append(eval_with_preds(
            ev,
            pred_fn=lambda r, active: r["dense_pred"] if active and r["dense_pred"] != r["compressed_pred"] else r["compressed_pred"],
            prob_fn=lambda r, active: onehot_prob(r["n_choices"], r["dense_pred"] if active and r["dense_pred"] != r["compressed_pred"] else r["compressed_pred"]),
            name=f"oracle_decision_q{int(q*100)}",
            tau=tau,
        ))

        # Oracle distribution repair: active uses dense distribution, inactive compressed.
        out_rows.append(eval_with_preds(
            ev,
            pred_fn=lambda r, active: r["dense_pred"] if active else r["compressed_pred"],
            prob_fn=lambda r, active: softmax(r["dense_scores"]) if active else softmax(r["compressed_scores"]),
            name=f"oracle_distribution_q{int(q*100)}",
            tau=tau,
        ))

        # Conservative oracle: only fix active damage cases, leave repair cases unchanged.
        out_rows.append(eval_with_preds(
            ev,
            pred_fn=lambda r, active: r["dense_pred"]
            if active and (r["dense_pred"] == r["gold"]) and (r["compressed_pred"] != r["gold"])
            else r["compressed_pred"],
            prob_fn=lambda r, active: onehot_prob(
                r["n_choices"],
                r["dense_pred"] if active and (r["dense_pred"] == r["gold"]) and (r["compressed_pred"] != r["gold"]) else r["compressed_pred"]
            ),
            name=f"oracle_damage_only_q{int(q*100)}",
            tau=tau,
        ))

    oracle_df = pd.DataFrame(out_rows)
    for c in [
        "active_rate", "dense_acc", "repaired_acc", "acc_delta", "pred_flip",
        "hidden_churn", "damage_rate", "repair_rate", "wrong_to_wrong_flip_rate",
        "correct_to_correct_flip_rate", "prob_js", "gold_prob_shift"
    ]:
        oracle_df[c + "_pct"] = oracle_df[c] * 100.0

    oracle_out = OUT / f"{target_id}_oracle_repair_upper_bounds.csv"
    oracle_df.to_csv(oracle_out, index=False)

    show_gate = [
        "split", "q", "tau", "active_rate_pct", "mismatch_recall_pct",
        "active_mismatch_rate_pct", "inactive_mismatch_rate_pct",
        "active_damage_rate_pct", "active_repair_rate_pct",
    ]

    show_oracle = [
        "method", "active_rate_pct", "dense_acc_pct", "repaired_acc_pct",
        "acc_delta_pct", "pred_flip_pct", "hidden_churn_pct",
        "damage_rate_pct", "repair_rate_pct", "prob_js_pct", "nll_shift"
    ]

    print("\n===== GATE QUALITY =====")
    print(gate_df[show_gate].to_string(index=False))

    print("\n===== ORACLE UPPER BOUNDS =====")
    print(oracle_df[show_oracle].to_string(index=False))

    print("\n===== BEST ORACLE BY HIDDEN CHURN =====")
    print(oracle_df[show_oracle].sort_values(["hidden_churn_pct", "pred_flip_pct"]).head(12).to_string(index=False))

    print("\nWROTE", gate_out)
    print("WROTE", oracle_out)


if __name__ == "__main__":
    main()
