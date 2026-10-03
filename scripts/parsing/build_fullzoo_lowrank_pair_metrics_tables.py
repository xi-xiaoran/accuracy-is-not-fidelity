import csv
import json
import math
from pathlib import Path
from collections import defaultdict

TASKS = [
    "boolq",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "piqa",
    "hellaswag",
    "openbookqa",
]

COMP_CSV = Path("configs/fullzoo_lowrank_severity_full_eval_targets.csv")
DENSE_CSV = Path("configs/fullzoo_lowrank_severity_dense_targets.csv")

OUT_DIR = Path("results/summary_tables")
OUT_TASK = OUT_DIR / "fullzoo_lowrank_pair_metrics_by_task_normchar.csv"
OUT_CONFIG = OUT_DIR / "fullzoo_lowrank_pair_metrics_by_config_normchar.csv"
OUT_METHOD_KEEP = OUT_DIR / "fullzoo_lowrank_pair_metrics_by_method_keep_normchar.csv"

EPS = 1e-12

def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def find_sample_file(outdir: Path, task: str) -> Path:
    files = sorted(outdir.rglob(f"samples_{task}_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No samples for task={task} in {outdir}")
    if len(files) > 1:
        # Usually there is only one. If multiple exist, use the largest as safest.
        files = sorted(files, key=lambda p: p.stat().st_size, reverse=True)
    return files[0]


def first_number(x):
    if isinstance(x, bool):
        return None

    if isinstance(x, (int, float)):
        return float(x)

    if isinstance(x, str):
        s = x.strip()
        if s.lower() in {"true", "false", "none", "nan", ""}:
            return None
        try:
            return float(s)
        except Exception:
            return None

    if isinstance(x, dict):
        for k in ["loglikelihood", "ll", "score", "value"]:
            if k in x:
                v = first_number(x[k])
                if v is not None:
                    return v
        for v in x.values():
            z = first_number(v)
            if z is not None:
                return z

    if isinstance(x, (list, tuple)):
        for v in x:
            z = first_number(v)
            if z is not None:
                return z

    return None

def extract_raw_scores(sample):
    for key in ["filtered_resps", "resps", "model_resps", "responses"]:
        if key not in sample:
            continue
        obj = sample[key]
        if not isinstance(obj, list) or len(obj) < 2:
            continue
        scores = [first_number(v) for v in obj]
        if all(v is not None for v in scores):
            return scores
    raise RuntimeError(f"Cannot extract choice scores. keys={list(sample.keys())}")


def extract_choice_texts(sample):
    args = sample.get("arguments", None)
    texts = []

    # lm-eval current format:
    # {"gen_args_0": {"arg_0": prompt, "arg_1": choice}, ...}
    if isinstance(args, dict):
        def order_key(k):
            try:
                return int(str(k).split("_")[-1])
            except Exception:
                return str(k)

        for k in sorted(args.keys(), key=order_key):
            v = args[k]
            if isinstance(v, dict):
                if "arg_1" in v:
                    texts.append(str(v["arg_1"]))
                elif 1 in v:
                    texts.append(str(v[1]))
                else:
                    vals = list(v.values())
                    if len(vals) >= 2:
                        texts.append(str(vals[1]))
                    elif vals:
                        texts.append(str(vals[0]))
            elif isinstance(v, (list, tuple)) and len(v) >= 2:
                texts.append(str(v[1]))
            else:
                texts.append(str(v))
        return texts

    # Older/list format.
    if isinstance(args, list):
        for a in args:
            if isinstance(a, (list, tuple)) and len(a) >= 2:
                texts.append(str(a[1]))
            elif isinstance(a, dict) and "arg_1" in a:
                texts.append(str(a["arg_1"]))
            else:
                texts.append(str(a))
        return texts

    return texts

def norm_char_scores(sample):
    raw = extract_raw_scores(sample)
    texts = extract_choice_texts(sample)
    if len(texts) != len(raw):
        lens = [1 for _ in raw]
    else:
        lens = [max(1, len(t.strip())) for t in texts]
    return [s / l for s, l in zip(raw, lens)]

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    z = sum(exps)
    return [e / z for e in exps]

def entropy(p):
    return -sum(x * math.log(max(x, EPS)) for x in p)

def margin(p):
    if len(p) < 2:
        return 0.0
    xs = sorted(p, reverse=True)
    return xs[0] - xs[1]

def kl(p, q):
    return sum(pi * (math.log(max(pi, EPS)) - math.log(max(qi, EPS))) for pi, qi in zip(p, q))

def js(p, q):
    m = [(a + b) / 2 for a, b in zip(p, q)]
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def argmax(xs):
    return max(range(len(xs)), key=lambda i: xs[i])

def metric_value(sample, name):
    if name in sample:
        try:
            return float(sample[name])
        except Exception:
            pass
    metrics = sample.get("metrics", {})
    if isinstance(metrics, dict) and name in metrics:
        v = metrics[name]
        if isinstance(v, (list, tuple)):
            v = v[0]
        try:
            return float(v)
        except Exception:
            pass
    return None


def extract_gold(sample, n_choices):
    target = sample.get("target", None)

    if isinstance(target, int) and 0 <= target < n_choices:
        return target

    if isinstance(target, str):
        t_raw = target.strip()
        t = t_raw.lower()

        # Current lm-eval often stores target as numeric string, e.g. "2".
        if t_raw.lstrip("-").isdigit():
            idx = int(t_raw)
            if 0 <= idx < n_choices:
                return idx

        choices = [x.strip().lower() for x in extract_choice_texts(sample)]
        if t in choices:
            return choices.index(t)

        # Common letter answer format.
        letters = "abcdefghijklmnopqrstuvwxyz"
        if len(t) == 1 and t in letters[:n_choices]:
            return letters.index(t)

    doc = sample.get("doc", {})
    if isinstance(doc, dict):
        for key in ["gold", "label", "answer", "answerKey", "target"]:
            if key in doc:
                v = doc[key]
                if isinstance(v, int) and 0 <= v < n_choices:
                    return v
                if isinstance(v, str):
                    vv_raw = v.strip()
                    vv = vv_raw.lower()

                    if vv_raw.lstrip("-").isdigit():
                        idx = int(vv_raw)
                        if 0 <= idx < n_choices:
                            return idx

                    letters = "abcdefghijklmnopqrstuvwxyz"
                    if len(vv) == 1 and vv in letters[:n_choices]:
                        return letters.index(vv)

                    choices = [x.strip().lower() for x in extract_choice_texts(sample)]
                    if vv in choices:
                        return choices.index(vv)

    return None

def sample_key(sample, fallback_i):
    # Prefer stable lm-eval hashes when available.
    parts = []
    for k in ["doc_id", "doc_hash", "prompt_hash", "target_hash"]:
        if k in sample:
            parts.append(str(sample[k]))
    if parts:
        return "||".join(parts)
    return str(fallback_i)

def mean(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")

def load_task_samples(outdir: Path, task: str):
    p = find_sample_file(outdir, task)
    rows = read_jsonl(p)
    m = {}
    for i, s in enumerate(rows):
        k = sample_key(s, i)
        m[k] = s
    return p, rows, m


def summarize_pair(dense_rows, comp_rows):
    dense_map = {sample_key(s, i): s for i, s in enumerate(dense_rows)}
    comp_map = {sample_key(s, i): s for i, s in enumerate(comp_rows)}
    keys = sorted(set(dense_map) & set(comp_map))

    acc_d = []
    acc_c = []
    accn_d = []
    accn_c = []

    flips = []
    prob_js = []
    ent_d = []
    ent_c = []
    marg_d = []
    marg_c = []
    goldp_d = []
    goldp_c = []
    nll_d = []
    nll_c = []

    n_gold = 0
    n_score_aligned = 0
    n_score_failed = 0

    for k in keys:
        ds = dense_map[k]
        cs = comp_map[k]

        # Official lm-eval per-sample metrics. These are available even for BoolQ.
        vd = metric_value(ds, "acc")
        vc = metric_value(cs, "acc")
        if vd is not None and vc is not None:
            acc_d.append(vd)
            acc_c.append(vc)

        vdn = metric_value(ds, "acc_norm")
        vcn = metric_value(cs, "acc_norm")
        if vdn is not None and vcn is not None:
            accn_d.append(vdn)
            accn_c.append(vcn)

        # Choice-score uncertainty metrics. Some tasks/sample formats, especially BoolQ,
        # may not expose all choice loglikelihoods in log_samples. In that case, keep
        # official accuracy but mark uncertainty metrics as NaN for this task.
        try:
            d_scores = norm_char_scores(ds)
            c_scores = norm_char_scores(cs)
        except Exception:
            n_score_failed += 1
            continue

        if len(d_scores) != len(c_scores) or len(d_scores) < 2:
            n_score_failed += 1
            continue

        n_score_aligned += 1

        dp = softmax(d_scores)
        cp = softmax(c_scores)
        d_pred = argmax(d_scores)
        c_pred = argmax(c_scores)

        flips.append(1.0 if d_pred != c_pred else 0.0)
        prob_js.append(js(dp, cp))
        ent_d.append(entropy(dp))
        ent_c.append(entropy(cp))
        marg_d.append(margin(dp))
        marg_c.append(margin(cp))

        gold = extract_gold(ds, len(d_scores))
        if gold is not None and 0 <= gold < len(d_scores):
            n_gold += 1
            goldp_d.append(dp[gold])
            goldp_c.append(cp[gold])
            nll_d.append(-math.log(max(dp[gold], EPS)))
            nll_c.append(-math.log(max(cp[gold], EPS)))

    dense_acc = mean(acc_d)
    comp_acc = mean(acc_c)

    dense_acc_norm = mean(accn_d) if accn_d else dense_acc
    comp_acc_norm = mean(accn_c) if accn_c else comp_acc

    return {
        "n_aligned": len(keys),
        "n_score_aligned": n_score_aligned,
        "n_score_failed": n_score_failed,
        "n_gold": n_gold,

        "dense_acc": dense_acc,
        "compressed_acc": comp_acc,
        "acc_delta": comp_acc - dense_acc if not (math.isnan(comp_acc) or math.isnan(dense_acc)) else float("nan"),

        "dense_acc_norm": dense_acc_norm,
        "compressed_acc_norm": comp_acc_norm,
        "acc_norm_delta": comp_acc_norm - dense_acc_norm if not (math.isnan(comp_acc_norm) or math.isnan(dense_acc_norm)) else float("nan"),

        "pred_flip": mean(flips),
        "prob_js": mean(prob_js),

        "dense_entropy": mean(ent_d),
        "compressed_entropy": mean(ent_c),
        "entropy_shift": mean([c - d for c, d in zip(ent_c, ent_d)]),

        "dense_margin": mean(marg_d),
        "compressed_margin": mean(marg_c),
        "margin_shift": mean([c - d for c, d in zip(marg_c, marg_d)]),

        "dense_gold_prob": mean(goldp_d),
        "compressed_gold_prob": mean(goldp_c),
        "gold_prob_shift": mean([c - d for c, d in zip(goldp_c, goldp_d)]),

        "dense_nll": mean(nll_d),
        "compressed_nll": mean(nll_c),
        "nll_shift": mean([c - d for c, d in zip(nll_c, nll_d)]),
    }

def group_mean(rows, group_keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in group_keys)].append(r)

    metric_cols = [
        "n_aligned", "n_score_aligned", "n_score_failed", "n_gold",
        "dense_acc", "compressed_acc", "acc_delta",
        "dense_acc_norm", "compressed_acc_norm", "acc_norm_delta",
        "pred_flip", "prob_js",
        "dense_entropy", "compressed_entropy", "entropy_shift",
        "dense_margin", "compressed_margin", "margin_shift",
        "dense_gold_prob", "compressed_gold_prob", "gold_prob_shift",
        "dense_nll", "compressed_nll", "nll_shift",
    ]

    out = []
    for key, xs in sorted(groups.items()):
        row = {k: v for k, v in zip(group_keys, key)}
        row["n_rows"] = len(xs)
        for c in metric_cols:
            vals = [float(x[c]) for x in xs if x.get(c, "") != ""]
            row[c] = mean(vals)
        out.append(row)
    return out

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("WROTE", path, "N_ROWS", len(rows))

def main():
    dense_targets = {r["model_key"]: r for r in csv.DictReader(open(DENSE_CSV))}
    comp_targets = list(csv.DictReader(open(COMP_CSV)))

    task_rows = []

    for i, cr in enumerate(comp_targets, 1):
        model_key = cr["model_key"]
        dr = dense_targets[model_key]

        for task in TASKS:
            dense_file, dense_rows, _ = load_task_samples(Path(dr["output_dir"]), task)
            comp_file, comp_rows, _ = load_task_samples(Path(cr["output_dir"]), task)

            m = summarize_pair(dense_rows, comp_rows)

            row = {
                "target_id": cr["target_id"],
                "model_key": cr["model_key"],
                "dense_model_hf": cr["dense_model_hf"],
                "method_key": cr["method_key"],
                "method_family": cr["method_family"],
                "keep_tag": cr["keep_tag"],
                "keep_ratio": cr["keep_ratio"],
                "task": task,
                **m,
                "dense_sample_file": str(dense_file),
                "compressed_sample_file": str(comp_file),
            }
            task_rows.append(row)

        if i % 25 == 0:
            print(f"PROGRESS {i}/{len(comp_targets)}", flush=True)

    write_csv(OUT_TASK, task_rows)

    config_rows = group_mean(
        task_rows,
        ["target_id", "model_key", "dense_model_hf", "method_key", "method_family", "keep_tag", "keep_ratio"],
    )
    write_csv(OUT_CONFIG, config_rows)

    method_keep_rows = group_mean(
        task_rows,
        ["method_key", "method_family", "keep_tag", "keep_ratio"],
    )
    write_csv(OUT_METHOD_KEEP, method_keep_rows)

    print()
    print("EXPECTED_BY_TASK_ROWS", 235 * 7)
    print("ACTUAL_BY_TASK_ROWS", len(task_rows))
    print("EXPECTED_BY_CONFIG_ROWS", 235)
    print("ACTUAL_BY_CONFIG_ROWS", len(config_rows))

if __name__ == "__main__":
    main()
