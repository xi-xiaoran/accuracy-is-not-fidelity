import argparse
import os, glob, json, math
import numpy as np
from transformers import AutoTokenizer

PARSER = argparse.ArgumentParser(
    description="Recompute ARC metrics from user-supplied lm-eval sample logs."
)
PARSER.add_argument("--dense-model", default="meta-llama/Llama-3.1-8B")
PARSER.add_argument(
    "--compressed-model",
    required=True,
    help="Local compressed checkpoint directory or Hugging Face model identifier.",
)
PARSER.add_argument("--dense-samples", help="Dense samples_arc_challenge JSONL.")
PARSER.add_argument("--compressed-samples", help="Compressed samples_arc_challenge JSONL.")
PARSER.add_argument("--dense-results", help="Dense lm-eval aggregate results JSON.")
PARSER.add_argument("--compressed-results", help="Compressed lm-eval aggregate results JSON.")
PARSER.add_argument(
    "--output",
    default="generated/reproduce_lmeval_arc_metrics.json",
    help="Destination JSON for the protocol audit.",
)
ARGS = PARSER.parse_args()

DENSE_MODEL = ARGS.dense_model
SVD_DIR = ARGS.compressed_model

dense_sample = ARGS.dense_samples or sorted(glob.glob("results/official_lmeval/logsamples_arc5/dense/**/samples_arc_challenge_*.jsonl", recursive=True))[-1]
asvd_sample = ARGS.compressed_samples or sorted(glob.glob("results/official_lmeval/logsamples_arc5/asvd_keep0p8/**/samples_arc_challenge_*.jsonl", recursive=True))[-1]

dense_result = ARGS.dense_results or sorted(glob.glob("results/official_lmeval/logsamples_arc5/dense/**/results_*.json", recursive=True))[-1]
asvd_result = ARGS.compressed_results or sorted(glob.glob("results/official_lmeval/logsamples_arc5/asvd_keep0p8/**/results_*.json", recursive=True))[-1]

print("dense_sample:", dense_sample)
print("asvd_sample:", asvd_sample)
print("dense_result:", dense_result)
print("asvd_result:", asvd_result)

dense_tok = AutoTokenizer.from_pretrained(DENSE_MODEL, use_fast=True)
svd_tok = AutoTokenizer.from_pretrained(SVD_DIR, trust_remote_code=True, use_fast=True)

def load_jsonl(p):
    rows = []
    with open(p) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def official_result(p):
    obj = json.load(open(p))
    r = obj["results"]["arc_challenge"]
    return {
        "acc": r.get("acc,none"),
        "acc_norm": r.get("acc_norm,none"),
    }

def extract_scores(row):
    # filtered_resps: [[score, greedy_match], ...]
    vals = []
    for x in row["filtered_resps"]:
        vals.append(float(x[0]))
    return vals

def choice_texts(row):
    args = row["arguments"]
    out = []
    for k in sorted(args.keys(), key=lambda x: int(x.split("_")[-1])):
        out.append(args[k]["arg_1"])
    return out

def target_idx(row):
    return int(row["target"])

def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

def summarize(rows, tok, name):
    n = len(rows)

    acc_raw = []
    acc_norm_tok_with_space = []
    acc_norm_tok_no_space = []
    acc_norm_char = []
    acc_norm_word = []

    details = []

    for row in rows:
        scores = extract_scores(row)
        choices = choice_texts(row)
        gold = target_idx(row)

        lens_tok_with_space = [len(tok(c, add_special_tokens=False).input_ids) for c in choices]
        lens_tok_no_space = [len(tok(c.lstrip(), add_special_tokens=False).input_ids) for c in choices]
        lens_char = [max(len(c), 1) for c in choices]
        lens_word = [max(len(c.split()), 1) for c in choices]

        pred_raw = int(np.argmax(scores))
        pred_tok_ws = int(np.argmax([s / max(l, 1) for s, l in zip(scores, lens_tok_with_space)]))
        pred_tok_ns = int(np.argmax([s / max(l, 1) for s, l in zip(scores, lens_tok_no_space)]))
        pred_char = int(np.argmax([s / l for s, l in zip(scores, lens_char)]))
        pred_word = int(np.argmax([s / l for s, l in zip(scores, lens_word)]))

        acc_raw.append(pred_raw == gold)
        acc_norm_tok_with_space.append(pred_tok_ws == gold)
        acc_norm_tok_no_space.append(pred_tok_ns == gold)
        acc_norm_char.append(pred_char == gold)
        acc_norm_word.append(pred_word == gold)

        details.append({
            "doc_id": row["doc_id"],
            "gold": gold,
            "official_acc": row.get("acc"),
            "official_acc_norm": row.get("acc_norm"),
            "scores": scores,
            "choices": choices,
            "lens_tok_with_space": lens_tok_with_space,
            "lens_tok_no_space": lens_tok_no_space,
            "lens_char": lens_char,
            "lens_word": lens_word,
            "pred_raw": pred_raw,
            "pred_tok_with_space": pred_tok_ws,
            "pred_tok_no_space": pred_tok_ns,
            "pred_char": pred_char,
            "pred_word": pred_word,
        })

    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    summary = {
        "n": n,
        "raw_argmax_acc": mean(acc_raw),
        "tok_with_space_norm_acc": mean(acc_norm_tok_with_space),
        "tok_no_space_norm_acc": mean(acc_norm_tok_no_space),
        "char_norm_acc": mean(acc_norm_char),
        "word_norm_acc": mean(acc_norm_word),
        "official_sample_acc_mean": mean([float(r.get("acc")) for r in rows]),
        "official_sample_acc_norm_mean": mean([float(r.get("acc_norm")) for r in rows]),
    }

    print("\n" + "=" * 100)
    print(name)
    print(json.dumps(summary, indent=2))

    print("\nPer example:")
    for d in details:
        print("-" * 100)
        print(json.dumps(d, indent=2)[:5000])

    return summary, details

dense_rows = load_jsonl(dense_sample)
asvd_rows = load_jsonl(asvd_sample)

print("\nOfficial aggregate JSON:")
print("dense", official_result(dense_result))
print("asvd", official_result(asvd_result))

dense_summary, dense_details = summarize(dense_rows, dense_tok, "dense")
asvd_summary, asvd_details = summarize(asvd_rows, svd_tok, "asvd_keep0p8")

output_path = os.path.abspath(os.path.expanduser(ARGS.output))
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump({
        "dense_official": official_result(dense_result),
        "asvd_official": official_result(asvd_result),
        "dense_summary": dense_summary,
        "asvd_summary": asvd_summary,
        "dense_details": dense_details,
        "asvd_details": asvd_details,
    }, f, indent=2)

print("\nREPRODUCE_DONE")
