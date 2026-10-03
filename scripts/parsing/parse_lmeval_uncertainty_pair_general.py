import os
import glob
import json
import argparse
import numpy as np
from collections import Counter

def latest(pattern):
    xs = sorted(glob.glob(pattern, recursive=True))
    if not xs:
        raise FileNotFoundError(pattern)
    return xs[-1]

def load_jsonl(p):
    rows = []
    with open(p) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def load_result(p, task):
    obj = json.load(open(p))
    return obj["results"][task]

def scores_from_row(row):
    return np.array([float(x[0]) for x in row["filtered_resps"]], dtype=np.float64)

def choices_from_row(row):
    args = row["arguments"]
    return [
        args[k]["arg_1"]
        for k in sorted(args.keys(), key=lambda x: int(x.split("_")[-1]))
    ]

def normalize_choice_for_len(c):
    return c[1:] if isinstance(c, str) and c.startswith(" ") else c

def target_indices(row, choices):
    n = len(choices)
    t = row.get("target")

    def one(x):
        if isinstance(x, bool):
            return int(x) if n == 2 else None
        if isinstance(x, int):
            return x if 0 <= x < n else None
        if isinstance(x, float) and x.is_integer():
            x = int(x)
            return x if 0 <= x < n else None
        if isinstance(x, str):
            s = x.strip()
            if s.lstrip("-").isdigit():
                v = int(s)
                return v if 0 <= v < n else None

            raw = [str(c) for c in choices]
            stripped = [str(c).strip() for c in choices]
            lstripped = [str(c).lstrip() for c in choices]
            normed = [str(c).strip().lower() for c in choices]
            s_norm = s.lower()
            if x in raw:
                return raw.index(x)
            if s in stripped:
                return stripped.index(s)
            if s in lstripped:
                return lstripped.index(s)
            if s_norm in normed:
                return normed.index(s_norm)
        return None

    if isinstance(t, list):
        out = []
        for x in t:
            v = one(x)
            if v is not None:
                out.append(v)
        return sorted(set(out))

    v = one(t)
    if v is not None:
        return [v]

    return []

def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    return float(-np.sum(p * np.log(p + 1e-12)))

def margin(p):
    s = np.sort(np.asarray(p, dtype=np.float64))[::-1]
    return float(s[0] - s[1]) if len(s) >= 2 else 0.0

def js(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return float(0.5 * (
        np.sum(p * (np.log(p + 1e-12) - np.log(m + 1e-12))) +
        np.sum(q * (np.log(q + 1e-12) - np.log(m + 1e-12)))
    ))

def available_modes(rows):
    modes = []
    if rows and all("acc" in r for r in rows):
        modes.append("acc_raw")
    if rows and all("acc_norm" in r for r in rows):
        modes.append("acc_norm_char")
    return modes

def row_metrics(row, mode):
    raw_scores = scores_from_row(row)
    choices = choices_from_row(row)
    targets = target_indices(row, choices)

    if mode == "acc_raw":
        scores = raw_scores
        official_key = "acc"
    elif mode == "acc_norm_char":
        lens = np.array([max(len(normalize_choice_for_len(c)), 1) for c in choices], dtype=np.float64)
        scores = raw_scores / lens
        official_key = "acc_norm"
    else:
        raise ValueError(mode)

    probs = softmax(scores)
    pred = int(np.argmax(scores))
    official_correct = float(row[official_key])
    correct = bool(pred in targets) if targets else None
    target_inferred_from_official_acc = False

    # Fallback for ambiguous binary tasks such as WinoGrande.
    # Some lm-eval samples store target as a text span whose exact mapping to logged continuations
    # is not trivially recoverable. Since the official sample already stores acc, binary gold index
    # can be recovered exactly from (pred, official acc).
    if (correct is None or abs(float(correct) - official_correct) > 1e-9) and len(choices) == 2 and official_correct in (0.0, 1.0):
        inferred = pred if official_correct == 1.0 else 1 - pred
        targets = [int(inferred)]
        correct = bool(pred in targets)
        target_inferred_from_official_acc = True

    gold_prob = float(np.sum([probs[i] for i in targets])) if targets else float("nan")

    return {
        "doc_id": int(row["doc_id"]),
        "doc_hash": row.get("doc_hash"),
        "id": row["doc"].get("id", row.get("doc_hash", str(row["doc_id"]))) if isinstance(row.get("doc"), dict) else row.get("doc_hash", str(row["doc_id"])),
        "target_raw": row.get("target"),
        "targets": targets,
        "pred": pred,
        "correct": correct,
        "official_correct": official_correct,
        "target_inferred_from_official_acc": target_inferred_from_official_acc,
        "scores": [float(x) for x in scores],
        "raw_scores": [float(x) for x in raw_scores],
        "choices": choices,
        "choice_lengths_char": [int(len(normalize_choice_for_len(c))) for c in choices],
        "probs": [float(x) for x in probs],
        "confidence": float(np.max(probs)),
        "margin": margin(probs),
        "entropy": entropy(probs),
        "gold_prob": gold_prob,
    }

def summarize_one(rows, mode):
    ms = [row_metrics(r, mode) for r in rows]
    pred_counts = Counter([m["pred"] for m in ms])
    official_mismatch = [
        m for m in ms
        if m["correct"] is not None and abs(float(m["correct"]) - float(m["official_correct"])) > 1e-9
    ]
    unresolved_targets = [m for m in ms if m["correct"] is None]

    correct_vals = [m["correct"] for m in ms if m["correct"] is not None]
    return {
        "n": len(ms),
        "n_with_resolved_target": len(correct_vals),
        "acc_reproduced": float(np.mean(correct_vals)) if correct_vals else float("nan"),
        "official_sample_acc_mean": float(np.mean([m["official_correct"] for m in ms])),
        "official_mismatch_count": len(official_mismatch),
        "unresolved_target_count": len(unresolved_targets),
        "pred_counts": {str(k): int(v) for k, v in sorted(pred_counts.items())},
    }, ms, official_mismatch, unresolved_targets

def key_for(m):
    return m.get("doc_hash") or str(m["doc_id"])

def compare(fp_ms, comp_ms):
    by_id_fp = {key_for(m): m for m in fp_ms}
    by_id_comp = {key_for(m): m for m in comp_ms}
    ids = [key_for(m) for m in fp_ms if key_for(m) in by_id_comp]

    rows = []
    for ex_id in ids:
        f = by_id_fp[ex_id]
        c = by_id_comp[ex_id]
        pf = np.asarray(f["probs"], dtype=np.float64)
        pc = np.asarray(c["probs"], dtype=np.float64)

        if len(pf) != len(pc):
            continue

        rows.append({
            "key": ex_id,
            "doc_id": f["doc_id"],
            "target_raw": f["target_raw"],
            "targets": f["targets"],
            "fp_pred": f["pred"],
            "compressed_pred": c["pred"],
            "fp_correct": f["correct"],
            "compressed_correct": c["correct"],
            "pred_flip": bool(f["pred"] != c["pred"]),
            "prob_js": js(pf, pc),
            "abs_conf_shift": abs(c["confidence"] - f["confidence"]),
            "conf_shift": c["confidence"] - f["confidence"],
            "abs_margin_shift": abs(c["margin"] - f["margin"]),
            "margin_shift": c["margin"] - f["margin"],
            "abs_entropy_shift": abs(c["entropy"] - f["entropy"]),
            "entropy_shift": c["entropy"] - f["entropy"],
            "abs_gold_prob_shift": abs(c["gold_prob"] - f["gold_prob"]) if not np.isnan(c["gold_prob"]) and not np.isnan(f["gold_prob"]) else float("nan"),
            "gold_prob_shift": c["gold_prob"] - f["gold_prob"] if not np.isnan(c["gold_prob"]) and not np.isnan(f["gold_prob"]) else float("nan"),
            "fp_probs": f["probs"],
            "compressed_probs": c["probs"],
            "choices": f["choices"],
        })

    def mean(k):
        vals = [r[k] for r in rows]
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "n": len(rows),
        "fp_accuracy": mean("fp_correct"),
        "compressed_accuracy": mean("compressed_correct"),
        "pred_flip_rate": mean("pred_flip"),
        "mean_prob_js": mean("prob_js"),
        "mean_abs_conf_shift": mean("abs_conf_shift"),
        "mean_conf_shift": mean("conf_shift"),
        "mean_abs_margin_shift": mean("abs_margin_shift"),
        "mean_margin_shift": mean("margin_shift"),
        "mean_abs_entropy_shift": mean("abs_entropy_shift"),
        "mean_entropy_shift": mean("entropy_shift"),
        "mean_abs_gold_prob_shift": mean("abs_gold_prob_shift"),
        "mean_gold_prob_shift": mean("gold_prob_shift"),
    }, rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense_dir", required=True)
    ap.add_argument("--compressed_dir", required=True)
    ap.add_argument("--compressed_name", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    dense_sample = latest(os.path.join(args.dense_dir, "**", f"samples_{args.task}_*.jsonl"))
    comp_sample = latest(os.path.join(args.compressed_dir, "**", f"samples_{args.task}_*.jsonl"))
    dense_result = latest(os.path.join(args.dense_dir, "**", "results_*.json"))
    comp_result = latest(os.path.join(args.compressed_dir, "**", "results_*.json"))

    dense_rows = load_jsonl(dense_sample)
    comp_rows = load_jsonl(comp_sample)

    modes = sorted(set(available_modes(dense_rows)).intersection(available_modes(comp_rows)))
    if not modes:
        raise RuntimeError(f"No supported modes for task={args.task}")

    out = {
        "paths": {
            "dense_sample": dense_sample,
            "compressed_sample": comp_sample,
            "dense_result": dense_result,
            "compressed_result": comp_result,
        },
        "task": args.task,
        "compressed_name": args.compressed_name,
        "available_modes": modes,
        "official_aggregate": {
            "dense": load_result(dense_result, args.task),
            args.compressed_name: load_result(comp_result, args.task),
        },
        "modes": {},
    }

    os.makedirs(args.out_dir, exist_ok=True)

    for mode in modes:
        fp_sum, fp_ms, fp_mismatch, fp_unresolved = summarize_one(dense_rows, mode)
        comp_sum, comp_ms, comp_mismatch, comp_unresolved = summarize_one(comp_rows, mode)
        drift_sum, drift_rows = compare(fp_ms, comp_ms)

        out["modes"][mode] = {
            "fp_summary": fp_sum,
            "compressed_summary": comp_sum,
            "drift_summary": drift_sum,
            "fp_mismatch_examples": fp_mismatch[:5],
            "compressed_mismatch_examples": comp_mismatch[:5],
            "fp_unresolved_target_examples": fp_unresolved[:5],
            "compressed_unresolved_target_examples": comp_unresolved[:5],
        }

        with open(os.path.join(args.out_dir, f"drift_rows_{mode}.json"), "w") as f:
            json.dump(drift_rows, f, indent=2)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 100)
    print("OFFICIAL UNCERTAINTY SUMMARY")
    print("task:", args.task)
    print("compressed_name:", args.compressed_name)
    print("modes:", modes)
    print("official_aggregate:")
    print(json.dumps(out["official_aggregate"], indent=2))
    for mode, obj in out["modes"].items():
        print("=" * 100)
        print(mode)
        print("fp_summary:", json.dumps(obj["fp_summary"], indent=2))
        print("compressed_summary:", json.dumps(obj["compressed_summary"], indent=2))
        print("drift_summary:", json.dumps(obj["drift_summary"], indent=2))
    print("PARSE_GENERAL_OK")

if __name__ == "__main__":
    main()
