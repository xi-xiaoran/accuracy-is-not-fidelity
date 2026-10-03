import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(".").resolve()
ACT = ROOT / "results/mechanism_activation_v2"
OUT = ROOT / "results/mechanism_choice_v1"
OUT.mkdir(parents=True, exist_ok=True)

from scripts.mechanism.mech_common_v1 import TASKS, latest_sample_file, locate_dense_run_dir, prompt_text

DENSE_MODEL_FALLBACK = {
    "llama2_7b": "meta-llama/Llama-2-7b-hf",
    "llama31_8b": "meta-llama/Llama-3.1-8B",
    "llama31_8b_instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama_7b": "huggyllama/llama-7b",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_8b_base": "Qwen/Qwen3-8B-Base",
}

def patch_transformers_torch_load_safety():
    try:
        import transformers.utils.import_utils as iu
        iu.check_torch_load_is_safe = lambda *args, **kwargs: None
    except Exception:
        pass
    try:
        import transformers.modeling_utils as mu
        mu.check_torch_load_is_safe = lambda *args, **kwargs: None
    except Exception:
        pass

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def bool01(x):
    if pd.isna(x):
        return 0
    if isinstance(x, str):
        return int(x.lower() in ["1", "true", "yes"])
    return int(bool(x))

def sorted_gen_arg_items(d):
    items = []
    for k, v in d.items():
        if not isinstance(v, dict):
            continue
        if "arg_0" in v and "arg_1" in v:
            try:
                order = int(str(k).split("_")[-1])
            except Exception:
                order = len(items)
            items.append((order, str(v["arg_0"]), str(v["arg_1"])))
    items.sort(key=lambda x: x[0])
    return [(a, b) for _, a, b in items]

def collect_string_pairs(obj, out):
    """
    Robustly extract lm-eval loglikelihood requests as (context, continuation).
    Handles both:
      - {"gen_args_0": {"arg_0": ctx, "arg_1": cont}, ...}
      - nested list/tuple forms
    """
    if obj is None:
        return

    if isinstance(obj, dict):
        # Main lm-eval sample format in our logs.
        direct = sorted_gen_arg_items(obj)
        if direct:
            out.extend(direct)
            return

        # Single request dict.
        if "arg_0" in obj and "arg_1" in obj:
            out.append((str(obj["arg_0"]), str(obj["arg_1"])))
            return

        # Recurse through possible containers.
        for k in ["arguments", "args", "request", "requests"]:
            if k in obj:
                collect_string_pairs(obj[k], out)
        return

    if isinstance(obj, (list, tuple)):
        if len(obj) >= 2 and isinstance(obj[0], str) and isinstance(obj[1], str):
            out.append((obj[0], obj[1]))
            return
        for z in obj:
            collect_string_pairs(z, out)

def normalize_continuation(c):
    c = str(c)
    if c == "":
        return c
    return c if c.startswith(" ") else " " + c

def extract_doc_choices(sample: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """
    Fallback extraction from task docs.
    This should rarely be used because arguments are the real lm-eval scoring requests.
    """
    doc = sample.get("doc", {})
    if not isinstance(doc, dict):
        return []

    ctx = prompt_text(sample)
    conts = []

    # ARC/OpenBook style: choices = {"text": [...], "label": [...]}
    choices = doc.get("choices", None)
    if isinstance(choices, dict):
        if isinstance(choices.get("text", None), list):
            conts = [normalize_continuation(x) for x in choices["text"]]
        elif isinstance(choices.get("label", None), list):
            conts = [normalize_continuation(x) for x in choices["label"]]

    # HellaSwag style.
    if not conts:
        for key in ["choices", "endings"]:
            v = doc.get(key, None)
            if isinstance(v, list):
                conts = [normalize_continuation(x) for x in v]
                break

    # BoolQ style.
    if not conts and "label" in doc and "passage" in doc and "question" in doc:
        conts = [" False", " True"]

    # PIQA style.
    if not conts and "sol1" in doc and "sol2" in doc:
        conts = [normalize_continuation(doc["sol1"]), normalize_continuation(doc["sol2"])]

    # WinoGrande style: blank is replaced inside context, continuation is shared suffix.
    if not conts and "sentence" in doc and "option1" in doc and "option2" in doc:
        sent = str(doc["sentence"])
        opt1 = str(doc["option1"])
        opt2 = str(doc["option2"])
        target = str(sample.get("target", "")).strip()
        if "_" in sent:
            full1 = sent.replace("_", opt1)
            full2 = sent.replace("_", opt2)
            if target and full1.endswith(target):
                ctx1 = full1[: -len(target)].rstrip()
                ctx2 = full2[: -len(target)].rstrip()
                cont = " " + target
                return [(ctx1, cont, ctx1 + cont), (ctx2, cont, ctx2 + cont)]
            return [(full1, "", full1), (full2, "", full2)]

    return [(ctx, c, ctx + c) for c in conts]

def extract_choice_texts(sample: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """
    Return list of (context, continuation, full_text).
    Prefer lm-eval request arguments because they preserve exact scoring contexts.
    """
    pairs = []
    for key in ["arguments", "args", "requests", "request"]:
        if key in sample:
            collect_string_pairs(sample.get(key), pairs)

    dedup = []
    seen = set()
    for ctx, cont in pairs:
        ctx = str(ctx)
        cont = str(cont)
        key = (ctx, cont)
        if key not in seen:
            seen.add(key)
            dedup.append((ctx, cont, ctx + cont))

    if len(dedup) >= 2:
        return dedup

    return extract_doc_choices(sample)

def parse_choice_idx(v, continuations):
    n = len(continuations)
    if pd.isna(v):
        return -1
    if isinstance(v, (int, np.integer)):
        return int(v) if 0 <= int(v) < n else -1
    if isinstance(v, float) and np.isfinite(v):
        iv = int(v)
        return iv if abs(v - iv) < 1e-6 and 0 <= iv < n else -1
    s = str(v).strip()
    if s == "":
        return -1
    try:
        iv = int(float(s))
        if 0 <= iv < n:
            return iv
    except Exception:
        pass
    if len(s) == 1 and s.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        iv = ord(s.upper()) - ord("A")
        if 0 <= iv < n:
            return iv

    ss = s.strip()
    for i, c in enumerate(continuations):
        cc = str(c).strip()
        if ss == cc or ss in [cc[:1], cc[:2]]:
            return i
    return -1

def load_dense_samples_for_model(model_key):
    run_dir = locate_dense_run_dir(model_key)
    task_to_rows = {}
    for task in TASKS:
        sf = latest_sample_file(run_dir, task)
        if sf is not None:
            task_to_rows[task] = read_jsonl(sf)
    return task_to_rows

def build_index():
    targets = pd.read_csv(ACT / "targets_fullzoo.csv").sort_values("array_idx").reset_index(drop=True)
    trusted = pd.read_csv(ACT / "trusted_eval_index.csv")
    dense_models = pd.read_csv(ACT / "dense_models.csv")

    targets.to_csv(OUT / "targets_fullzoo.csv", index=False)
    trusted.to_csv(OUT / "trusted_eval_index.csv", index=False)
    dense_models.to_csv(OUT / "dense_models.csv", index=False)

    dense_samples = {}
    choice_rows = []
    sample_choice_meta = []

    # unique dense prompt rows
    prompts = trusted[["model_key", "task", "source_idx", "dense_uid"]].drop_duplicates().reset_index(drop=True)

    missing_choice = []
    for _, r in prompts.iterrows():
        model_key = str(r["model_key"])
        task = str(r["task"])
        source_idx = int(r["source_idx"])
        dense_uid = str(r["dense_uid"])

        if model_key not in dense_samples:
            dense_samples[model_key] = load_dense_samples_for_model(model_key)

        rows = dense_samples[model_key].get(task, [])
        if source_idx < 0 or source_idx >= len(rows):
            missing_choice.append((dense_uid, "bad_source_idx"))
            continue

        sample = rows[source_idx]
        choices = extract_choice_texts(sample)
        if len(choices) < 2:
            missing_choice.append((dense_uid, f"n_choices={len(choices)}"))
            continue

        for ci, (ctx, cont, full) in enumerate(choices):
            choice_uid = f"{dense_uid}::choice{ci}"
            choice_rows.append({
                "choice_uid": choice_uid,
                "dense_uid": dense_uid,
                "model_key": model_key,
                "task": task,
                "source_idx": source_idx,
                "choice_idx": ci,
                "context": ctx,
                "continuation": cont,
                "full_text": full,
            })

        sample_choice_meta.append({
            "dense_uid": dense_uid,
            "model_key": model_key,
            "task": task,
            "source_idx": source_idx,
            "n_choices": len(choices),
        })

    choice_df = pd.DataFrame(choice_rows)
    sample_meta = pd.DataFrame(sample_choice_meta)
    choice_df.to_csv(OUT / "dense_choice_index.csv", index=False)
    sample_meta.to_csv(OUT / "sample_choice_meta.csv", index=False)
    pd.DataFrame(missing_choice, columns=["dense_uid", "reason"]).to_csv(OUT / "audits/missing_choice_extraction.csv", index=False)

    # Add parsed choice indices into trusted rows.
    cont_map = choice_df.groupby("dense_uid")["continuation"].apply(list).to_dict()
    n_map = sample_meta.set_index("dense_uid")["n_choices"].to_dict()

    dense_pred_idx = []
    comp_pred_idx = []
    gold_idx = []
    n_choices = []
    for _, r in trusted.iterrows():
        du = str(r["dense_uid"])
        conts = cont_map.get(du, [])
        n_choices.append(int(n_map.get(du, 0)))
        dense_pred_idx.append(parse_choice_idx(r.get("dense_pred", np.nan), conts))
        comp_pred_idx.append(parse_choice_idx(r.get("compressed_pred", np.nan), conts))
        gold_idx.append(parse_choice_idx(r.get("gold", np.nan), conts))

    trusted2 = trusted.copy()
    trusted2["n_choices"] = n_choices
    trusted2["dense_pred_idx"] = dense_pred_idx
    trusted2["compressed_pred_idx"] = comp_pred_idx
    trusted2["gold_idx"] = gold_idx
    trusted2.to_csv(OUT / "trusted_eval_index_with_choice.csv", index=False)

    print("WROTE", OUT / "targets_fullzoo.csv", targets.shape)
    print("WROTE", OUT / "dense_choice_index.csv", choice_df.shape)
    print("WROTE", OUT / "sample_choice_meta.csv", sample_meta.shape)
    print("WROTE", OUT / "trusted_eval_index_with_choice.csv", trusted2.shape)
    print("MISSING_CHOICE", len(missing_choice))
    print("\nCHOICE_COUNTS_BY_TASK")
    print(sample_meta.groupby(["model_key", "task"])["n_choices"].describe().to_string())
    print("\nPARSED_IDX_VALID_RATE")
    for c in ["dense_pred_idx", "compressed_pred_idx", "gold_idx"]:
        print(c, float((trusted2[c] >= 0).mean()), int((trusted2[c] >= 0).sum()), "/", len(trusted2))

def batch_hidden_last(model, tok, texts, batch_size=2, max_length=1024):
    device = next(model.parameters()).device
    outs = []
    with torch.no_grad():
        for st in range(0, len(texts), batch_size):
            batch = texts[st:st+batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states[1:]
            last_idx = enc["attention_mask"].sum(dim=1) - 1
            arr_layers = []
            for h in hs:
                v = h[torch.arange(h.shape[0], device=device), last_idx]
                arr_layers.append(v.detach().float().cpu().numpy())
            arr = np.stack(arr_layers, axis=0).transpose(1, 0, 2).astype(np.float16)
            outs.append(arr)
            del out, hs, enc
            torch.cuda.empty_cache()
    return np.concatenate(outs, axis=0)

def run_dense_cache(array_idx, batch_size, max_length):
    dense_models = pd.read_csv(OUT / "dense_models.csv")
    r = dense_models.iloc[int(array_idx)]
    model_key = str(r["model_key"])
    model_ref = str(r["dense_model_ref"])

    choice = pd.read_csv(OUT / "dense_choice_index.csv")
    choice = choice[choice["model_key"].astype(str) == model_key].copy().reset_index(drop=True)

    out_npz = OUT / "dense_cache" / f"{model_key}_choice_last_hidden.npz"
    out_meta = OUT / "dense_cache" / f"{model_key}_choice_meta.csv"
    if out_npz.exists() and out_meta.exists():
        print("SKIP_DENSE_EXISTS", model_key, out_npz)
        return

    print("DENSE_CHOICE_MODEL", model_key, model_ref, "N_CHOICES", len(choice))
    tok = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    patch_transformers_torch_load_safety()
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    hidden = batch_hidden_last(model, tok, choice["full_text"].astype(str).tolist(), batch_size=batch_size, max_length=max_length)
    np.savez_compressed(out_npz, hidden=hidden)
    choice.to_csv(out_meta, index=False)
    print("WROTE", out_npz, hidden.shape)
    print("WROTE", out_meta, choice.shape)

def compute_rel_cos(dense, comp):
    N, L, D = dense.shape
    rel = np.empty((N, L), dtype=np.float32)
    cos = np.empty((N, L), dtype=np.float32)
    eps = 1e-8
    for l in range(L):
        a = dense[:, l, :].astype(np.float32)
        b = comp[:, l, :].astype(np.float32)
        diff = b - a
        rel[:, l] = np.linalg.norm(diff, axis=1) / (np.linalg.norm(a, axis=1) + eps)
        dot = np.sum(a * b, axis=1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + eps
        cos[:, l] = 1.0 - dot / den
    return rel, cos

def category_from_row(r):
    flip = bool01(r.get("flip", 0))
    if flip:
        if bool01(r.get("damage", 0)): return "damage"
        if bool01(r.get("repair", 0)): return "repair"
        if bool01(r.get("wrong_to_wrong", 0)): return "wrong_to_wrong"
        if bool01(r.get("correct_to_correct", 0)): return "correct_to_correct_flip"
        return "flip_other"
    dc = bool01(r.get("dense_correct", 0))
    cc = bool01(r.get("compressed_correct", 0))
    if dc and cc: return "stable_correct"
    if (not dc) and (not cc): return "stable_wrong"
    return "stable_other"

def run_target(array_idx, batch_size, max_length, overwrite=False):
    targets = pd.read_csv(OUT / "targets_fullzoo.csv")
    t = targets.iloc[int(array_idx)]
    tid = str(t["target_id"])
    model_key = str(t["model_key"])
    comp_path = str(t["compressed_path"])

    out_sample = OUT / "by_target" / f"{tid}_choice_sample_summary.csv"
    out_layer = OUT / "by_target" / f"{tid}_choice_layer_summary.csv"
    if out_sample.exists() and out_layer.exists() and not overwrite:
        print("SKIP_TARGET_EXISTS", tid)
        return

    trusted = pd.read_csv(OUT / "trusted_eval_index_with_choice.csv")
    idx = trusted[trusted["target_id"].astype(str) == tid].copy().reset_index(drop=True)

    choice_all = pd.read_csv(OUT / "dense_choice_index.csv")
    choice_model = choice_all[choice_all["model_key"].astype(str) == model_key].copy().reset_index(drop=True)

    dense_npz = OUT / "dense_cache" / f"{model_key}_choice_last_hidden.npz"
    dense_meta_p = OUT / "dense_cache" / f"{model_key}_choice_meta.csv"
    if not dense_npz.exists() or not dense_meta_p.exists():
        raise FileNotFoundError(f"Missing dense choice cache for {model_key}")

    dense_meta = pd.read_csv(dense_meta_p).reset_index(drop=True)
    dense_hidden_all = np.load(dense_npz)["hidden"]
    pos_by_choice_uid = {u: i for i, u in enumerate(dense_meta["choice_uid"].astype(str).tolist())}

    # Target uses all choice rows for this model, but only choices whose dense_uid appears in this target.
    target_dense_uids = set(idx["dense_uid"].astype(str).tolist())
    choice = choice_model[choice_model["dense_uid"].astype(str).isin(target_dense_uids)].copy().reset_index(drop=True)
    dense_positions = [pos_by_choice_uid[str(u)] for u in choice["choice_uid"].astype(str)]
    dense_hidden = dense_hidden_all[dense_positions]

    print("TARGET", tid)
    print("MODEL_KEY", model_key)
    print("COMP_PATH", comp_path)
    print("N_SAMPLES", len(idx), "N_CHOICES", len(choice))
    print("CHOICE_COUNT_DIST")
    print(choice.groupby("dense_uid").size().describe().to_string())
    print("CATEGORY_COUNTS")
    print(idx.apply(category_from_row, axis=1).value_counts().to_string())

    tok = AutoTokenizer.from_pretrained(comp_path, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    patch_transformers_torch_load_safety()
    model = AutoModelForCausalLM.from_pretrained(
        comp_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    comp_hidden = batch_hidden_last(model, tok, choice["full_text"].astype(str).tolist(), batch_size=batch_size, max_length=max_length)

    if dense_hidden.shape != comp_hidden.shape:
        raise RuntimeError(f"shape mismatch dense={dense_hidden.shape} comp={comp_hidden.shape}")

    rel, cos = compute_rel_cos(dense_hidden, comp_hidden)
    L = rel.shape[1]

    # Map sample -> choice positions inside current choice table
    local_choice_pos = {}
    for i, r in choice.iterrows():
        local_choice_pos.setdefault(str(r["dense_uid"]), {})[int(r["choice_idx"])] = i

    sample_rows = []
    # Sample-level arrays for layer summary
    per_sample_layer = {
        "mean_rel": [], "max_rel": [], "std_rel": [],
        "mean_cos": [], "max_cos": [], "std_cos": [],
        "dense_pred_rel": [], "comp_pred_rel": [], "gold_rel": [],
        "dense_pred_cos": [], "comp_pred_cos": [], "gold_cos": [],
    }
    sample_meta_rows = []

    for _, r in idx.iterrows():
        du = str(r["dense_uid"])
        cmap = local_choice_pos.get(du, {})
        positions = [cmap[k] for k in sorted(cmap.keys())]
        if len(positions) == 0:
            continue

        rr = rel[positions]
        cc = cos[positions]

        def take_choice(metric_arr, choice_idx):
            ci = int(choice_idx)
            if ci in cmap:
                return metric_arr[cmap[ci]]
            return np.full((L,), np.nan, dtype=np.float32)

        dense_idx = int(r.get("dense_pred_idx", -1))
        comp_idx = int(r.get("compressed_pred_idx", -1))
        gold_idx = int(r.get("gold_idx", -1))

        dense_rel_l = take_choice(rel, dense_idx)
        comp_rel_l = take_choice(rel, comp_idx)
        gold_rel_l = take_choice(rel, gold_idx)
        dense_cos_l = take_choice(cos, dense_idx)
        comp_cos_l = take_choice(cos, comp_idx)
        gold_cos_l = take_choice(cos, gold_idx)

        mean_rel_l = np.nanmean(rr, axis=0)
        max_rel_l = np.nanmax(rr, axis=0)
        std_rel_l = np.nanstd(rr, axis=0)
        mean_cos_l = np.nanmean(cc, axis=0)
        max_cos_l = np.nanmax(cc, axis=0)
        std_cos_l = np.nanstd(cc, axis=0)

        cat = category_from_row(r)
        br = pd.to_numeric(pd.Series([r.get("boundary_ratio", np.nan)]), errors="coerce").iloc[0]

        row = {
            "target_id": tid,
            "model_key": model_key,
            "method_key": t.get("method_key", ""),
            "keep_tag": t.get("keep_tag", ""),
            "task": r["task"],
            "dense_uid": du,
            "category": cat,
            "flip": bool01(r.get("flip", 0)),
            "damage": bool01(r.get("damage", 0)),
            "repair": bool01(r.get("repair", 0)),
            "wrong_to_wrong": bool01(r.get("wrong_to_wrong", 0)),
            "boundary_ratio": br,
            "n_choices": len(positions),
            "dense_pred_idx": dense_idx,
            "compressed_pred_idx": comp_idx,
            "gold_idx": gold_idx,
            "mean_choice_rel_l2": float(np.nanmean(rr)),
            "max_choice_rel_l2": float(np.nanmax(rr)),
            "std_choice_rel_l2": float(np.nanmean(std_rel_l)),
            "mean_choice_cos": float(np.nanmean(cc)),
            "max_choice_cos": float(np.nanmax(cc)),
            "std_choice_cos": float(np.nanmean(std_cos_l)),
            "dense_pred_rel_l2": float(np.nanmean(dense_rel_l)),
            "compressed_pred_rel_l2": float(np.nanmean(comp_rel_l)),
            "gold_rel_l2": float(np.nanmean(gold_rel_l)),
            "dense_pred_cos": float(np.nanmean(dense_cos_l)),
            "compressed_pred_cos": float(np.nanmean(comp_cos_l)),
            "gold_cos": float(np.nanmean(gold_cos_l)),
            "winner_delta_rel_l2": float(np.nanmean(comp_rel_l) - np.nanmean(dense_rel_l)),
            "winner_delta_cos": float(np.nanmean(comp_cos_l) - np.nanmean(dense_cos_l)),
            "last_mean_choice_rel_l2": float(np.nanmean(mean_rel_l[-1])),
            "last_std_choice_rel_l2": float(std_rel_l[-1]),
            "last_mean_choice_cos": float(np.nanmean(mean_cos_l[-1])),
            "last_std_choice_cos": float(std_cos_l[-1]),
        }
        sample_rows.append(row)

        sample_meta_rows.append(row)
        per_sample_layer["mean_rel"].append(mean_rel_l)
        per_sample_layer["max_rel"].append(max_rel_l)
        per_sample_layer["std_rel"].append(std_rel_l)
        per_sample_layer["mean_cos"].append(mean_cos_l)
        per_sample_layer["max_cos"].append(max_cos_l)
        per_sample_layer["std_cos"].append(std_cos_l)
        per_sample_layer["dense_pred_rel"].append(dense_rel_l)
        per_sample_layer["comp_pred_rel"].append(comp_rel_l)
        per_sample_layer["gold_rel"].append(gold_rel_l)
        per_sample_layer["dense_pred_cos"].append(dense_cos_l)
        per_sample_layer["comp_pred_cos"].append(comp_cos_l)
        per_sample_layer["gold_cos"].append(gold_cos_l)

    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(out_sample, index=False)

    meta_df = pd.DataFrame(sample_meta_rows)
    arrs = {k: np.stack(v, axis=0) for k, v in per_sample_layer.items()}
    layer_rows = []
    for l in range(L):
        base = meta_df[["category", "flip", "boundary_ratio"]].copy()
        for k, arr in arrs.items():
            base[k] = arr[:, l]

        groups = [("all", base)]
        groups += [(str(k), g) for k, g in base.groupby("category", dropna=False)]
        groups += [
            ("flip", base[base["flip"] == 1]),
            ("nonflip", base[base["flip"] == 0]),
            ("bar_lt1", base[pd.to_numeric(base["boundary_ratio"], errors="coerce") < 1]),
            ("bar_ge1", base[pd.to_numeric(base["boundary_ratio"], errors="coerce") >= 1]),
        ]

        for gname, g in groups:
            if len(g) == 0:
                continue
            layer_rows.append({
                "target_id": tid,
                "model_key": model_key,
                "method_key": t.get("method_key", ""),
                "keep_tag": t.get("keep_tag", ""),
                "layer": l,
                "group": gname,
                "n": len(g),
                "mean_choice_rel_l2": float(g["mean_rel"].mean()),
                "std_choice_rel_l2": float(g["std_rel"].mean()),
                "max_choice_rel_l2": float(g["max_rel"].mean()),
                "mean_choice_cos": float(g["mean_cos"].mean()),
                "std_choice_cos": float(g["std_cos"].mean()),
                "max_choice_cos": float(g["max_cos"].mean()),
                "dense_pred_rel_l2": float(g["dense_pred_rel"].mean()),
                "compressed_pred_rel_l2": float(g["comp_pred_rel"].mean()),
                "gold_rel_l2": float(g["gold_rel"].mean()),
                "dense_pred_cos": float(g["dense_pred_cos"].mean()),
                "compressed_pred_cos": float(g["comp_pred_cos"].mean()),
                "gold_cos": float(g["gold_cos"].mean()),
            })

    pd.DataFrame(layer_rows).to_csv(out_layer, index=False)
    print("WROTE", out_sample, sample_df.shape)
    print("WROTE", out_layer, len(layer_rows))

def collect():
    by = OUT / "by_target"
    sample_files = sorted(by.glob("*_choice_sample_summary.csv"))
    layer_files = sorted(by.glob("*_choice_layer_summary.csv"))
    print("N_SAMPLE_FILES", len(sample_files))
    print("N_LAYER_FILES", len(layer_files))

    if sample_files:
        s = pd.concat([pd.read_csv(p) for p in sample_files], ignore_index=True)
        s.to_csv(OUT / "fullzoo_choice_sample_summary.csv", index=False)
        print("WROTE", OUT / "fullzoo_choice_sample_summary.csv", s.shape)

        cat = s.groupby(["model_key", "method_key", "keep_tag", "category"], dropna=False).agg(
            n=("dense_uid", "count"),
            mean_choice_rel_l2=("mean_choice_rel_l2", "mean"),
            std_choice_rel_l2=("std_choice_rel_l2", "mean"),
            mean_choice_cos=("mean_choice_cos", "mean"),
            std_choice_cos=("std_choice_cos", "mean"),
            dense_pred_rel_l2=("dense_pred_rel_l2", "mean"),
            compressed_pred_rel_l2=("compressed_pred_rel_l2", "mean"),
            winner_delta_rel_l2=("winner_delta_rel_l2", "mean"),
        ).reset_index()
        cat.to_csv(OUT / "fullzoo_choice_category_summary.csv", index=False)
        print("WROTE", OUT / "fullzoo_choice_category_summary.csv", cat.shape)

    if layer_files:
        l = pd.concat([pd.read_csv(p) for p in layer_files], ignore_index=True)
        l.to_csv(OUT / "fullzoo_choice_layer_summary.csv", index=False)
        print("WROTE", OUT / "fullzoo_choice_layer_summary.csv", l.shape)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build_index", "dense_cache", "target", "collect"])
    ap.add_argument("--array_idx", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.mode == "build_index":
        build_index()
    elif args.mode == "dense_cache":
        run_dense_cache(args.array_idx, args.batch_size, args.max_length)
    elif args.mode == "target":
        run_target(args.array_idx, args.batch_size, args.max_length, args.overwrite)
    elif args.mode == "collect":
        collect()

if __name__ == "__main__":
    main()
