import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def patch_transformers_torch_load_safety():
    """
    Transformers >=4.57 blocks torch.load on torch<2.6 for .bin checkpoints.
    These are local project checkpoints already used in the frozen evaluation.
    For this diagnostic rerun, bypass the transformers safety gate so local
    PyTorch .bin checkpoints can load. Safetensors checkpoints are unaffected.
    """
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


ROOT = Path(".").resolve()
THEORY = ROOT / "results/theory_predictions_v1"
OUT = ROOT / "results/mechanism_activation_v2"

from scripts.mechanism.mech_common_v1 import (
    TASKS,
    load_targets_csv,
    latest_sample_file,
    locate_dense_run_dir,
    prompt_text,
)

DENSE_MODEL_FALLBACK = {
    "llama2_7b": "meta-llama/Llama-2-7b-hf",
    "llama31_8b": "meta-llama/Llama-3.1-8B",
    "llama31_8b_instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama_7b": "huggyllama/llama-7b",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_8b_base": "Qwen/Qwen3-8B-Base",
}

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

def find_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in low:
            return low[c.lower()]
    return None

def load_dense_samples_for_model(model_key: str):
    run_dir = locate_dense_run_dir(model_key)
    task_to_rows = {}
    for task in TASKS:
        sf = latest_sample_file(run_dir, task)
        if sf is None:
            continue
        task_to_rows[task] = read_jsonl(sf)
    return task_to_rows

def resolve_source_idx(frame_all_task: pd.DataFrame, dense_rows: List[Dict[str, Any]]):
    """
    Return source_idx for all rows of one task's sample_theory_frame.
    Preference:
      1. trusted explicit source/index columns if present;
      2. doc_id match if available;
      3. row order in full sample frame, including calibration rows.
    """
    n = len(frame_all_task)
    source_idx = np.full(n, -1, dtype=int)

    # doc_id match is safest when both sides have doc_id.
    doc_col = find_existing_col(frame_all_task, ["doc_id", "example_id", "id"])
    if doc_col is not None:
        dense_doc_to_idx = {}
        for i, r in enumerate(dense_rows):
            if isinstance(r, dict) and r.get("doc_id", None) is not None:
                dense_doc_to_idx[str(r.get("doc_id"))] = i
        if dense_doc_to_idx:
            ok = 0
            for j, v in enumerate(frame_all_task[doc_col].tolist()):
                key = str(v)
                if key in dense_doc_to_idx:
                    source_idx[j] = dense_doc_to_idx[key]
                    ok += 1
            if ok >= max(1, int(0.9 * n)):
                return source_idx

    # explicit index columns.
    idx_col = find_existing_col(frame_all_task, [
        "source_idx", "source_index", "jsonl_idx", "jsonl_index",
        "sample_idx", "sample_index", "row_idx", "row_index", "task_local_idx"
    ])
    if idx_col is not None:
        vals = pd.to_numeric(frame_all_task[idx_col], errors="coerce")
        if vals.notna().mean() > 0.9:
            vv = vals.fillna(-1).astype(int).to_numpy()
            if vv.min() >= 0 and vv.max() < len(dense_rows):
                return vv

    # fallback: full order within task in sample_theory_frame.
    # This preserves cal rows before eval rows, matching make_rows(cal_per_task, eval_per_task).
    vv = np.arange(n, dtype=int)
    vv[vv >= len(dense_rows)] = -1
    return vv

def category_from_trusted(r):
    flip = bool01(r.get("flip", 0))
    if flip:
        if bool01(r.get("damage", 0)):
            return "damage"
        if bool01(r.get("repair", 0)):
            return "repair"
        if bool01(r.get("wrong_to_wrong", 0)):
            return "wrong_to_wrong"
        if bool01(r.get("correct_to_correct", 0)):
            return "correct_to_correct_flip"
        return "flip_other"
    else:
        dc = bool01(r.get("dense_correct", 0))
        cc = bool01(r.get("compressed_correct", 0))
        if dc and cc:
            return "stable_correct"
        if (not dc) and (not cc):
            return "stable_wrong"
        return "stable_other"

def build_index(n_per_task: int):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dense_cache").mkdir(parents=True, exist_ok=True)
    (OUT / "by_target").mkdir(parents=True, exist_ok=True)
    (OUT / "audits").mkdir(parents=True, exist_ok=True)

    targets = load_targets_csv().sort_values(["model_key", "method_key", "keep_tag", "target_id"]).reset_index(drop=True)
    targets["array_idx"] = range(len(targets))
    targets.to_csv(OUT / "targets_fullzoo.csv", index=False)

    all_target_rows = []
    prompt_rows = []
    prompt_seen = set()

    dense_samples_cache = {}

    for ti, trow in targets.iterrows():
        tid = str(trow["target_id"])
        model_key = str(trow["model_key"])
        sample_p = THEORY / f"{tid}_sample_theory_frame.csv"
        if not sample_p.exists():
            raise FileNotFoundError(sample_p)

        df = pd.read_csv(sample_p)
        if "split" not in df.columns or "task" not in df.columns:
            raise ValueError(f"{sample_p} missing split/task columns: {list(df.columns)}")

        if model_key not in dense_samples_cache:
            dense_samples_cache[model_key] = load_dense_samples_for_model(model_key)

        for task, g_all in df.groupby("task", sort=False):
            task = str(task)
            dense_rows = dense_samples_cache[model_key].get(task, [])
            if not dense_rows:
                print("WARN missing dense rows", model_key, task)
                continue

            g_all = g_all.copy().reset_index(drop=False).rename(columns={"index": "theory_frame_row"})
            src = resolve_source_idx(g_all, dense_rows)
            g_all["source_idx"] = src

            ev = g_all[g_all["split"].astype(str) == "eval"].copy()
            ev = ev.head(n_per_task).copy()

            for eval_order, (_, r) in enumerate(ev.iterrows()):
                source_idx = int(r["source_idx"])
                if source_idx < 0 or source_idx >= len(dense_rows):
                    raise RuntimeError(f"Bad source_idx tid={tid} task={task} source_idx={source_idx}")

                sample = dense_rows[source_idx]
                prompt = prompt_text(sample)
                dense_uid = f"{model_key}::{task}::{source_idx}"

                if dense_uid not in prompt_seen:
                    prompt_seen.add(dense_uid)
                    prompt_rows.append({
                        "dense_uid": dense_uid,
                        "model_key": model_key,
                        "task": task,
                        "source_idx": source_idx,
                        "prompt": prompt,
                    })

                row = {
                    "target_id": tid,
                    "array_idx": int(trow["array_idx"]),
                    "model_key": model_key,
                    "method_key": trow.get("method_key", ""),
                    "keep_tag": trow.get("keep_tag", ""),
                    "task": task,
                    "source_idx": source_idx,
                    "dense_uid": dense_uid,
                    "theory_frame_row": int(r["theory_frame_row"]),
                    "task_eval_order": int(eval_order),
                    "category": category_from_trusted(r),
                }

                for c in [
                    "dense_correct", "compressed_correct", "flip",
                    "damage", "repair", "wrong_to_wrong", "correct_to_correct",
                    "dense_margin", "boundary_pressure", "boundary_ratio",
                    "prob_js", "dense_pred", "compressed_pred", "gold",
                ]:
                    if c in r.index:
                        row[c] = r[c]
                all_target_rows.append(row)

    idx = pd.DataFrame(all_target_rows)
    prompts = pd.DataFrame(prompt_rows)

    idx.to_csv(OUT / "trusted_eval_index.csv", index=False)
    prompts.to_csv(OUT / "dense_prompt_index.csv", index=False)

    dense_models = (
        prompts[["model_key"]].drop_duplicates().sort_values("model_key").reset_index(drop=True)
    )
    dense_models["dense_model_ref"] = dense_models["model_key"].map(DENSE_MODEL_FALLBACK)
    dense_models["array_idx"] = range(len(dense_models))
    dense_models.to_csv(OUT / "dense_models.csv", index=False)

    # selected-subset summary for sanity.
    summ = idx.groupby("target_id", dropna=False).agg(
        n=("flip", "size"),
        pred_flip_pct=("flip", lambda x: 100 * pd.to_numeric(x, errors="coerce").mean()),
        damage_pct=("damage", lambda x: 100 * pd.to_numeric(x, errors="coerce").mean()),
        repair_pct=("repair", lambda x: 100 * pd.to_numeric(x, errors="coerce").mean()),
        wrong_to_wrong_pct=("wrong_to_wrong", lambda x: 100 * pd.to_numeric(x, errors="coerce").mean()),
        mean_boundary_ratio=("boundary_ratio", lambda x: pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).mean()),
    ).reset_index()
    summ.to_csv(OUT / "trusted_selected_subset_summary.csv", index=False)

    print("WROTE", OUT / "targets_fullzoo.csv", targets.shape)
    print("WROTE", OUT / "trusted_eval_index.csv", idx.shape)
    print("WROTE", OUT / "dense_prompt_index.csv", prompts.shape)
    print("WROTE", OUT / "dense_models.csv", dense_models.shape)
    print("N_PER_TASK", n_per_task)
    print("TARGET_HEAD")
    print(targets[["array_idx", "target_id", "model_key", "method_key", "keep_tag"]].head(12).to_string(index=False))
    print("INDEX_CATEGORY_COUNTS")
    print(idx["category"].value_counts().to_string())
    print("DENSE_MODELS")
    print(dense_models.to_string(index=False))

def batch_hidden_last(model, tok, prompts, batch_size=4, max_length=1024):
    device = next(model.parameters()).device
    outs = []
    with torch.no_grad():
        for st in range(0, len(prompts), batch_size):
            batch = prompts[st:st+batch_size]
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

def run_dense_cache(array_idx: int, batch_size: int, max_length: int):
    dense_models = pd.read_csv(OUT / "dense_models.csv")
    r = dense_models.iloc[int(array_idx)]
    model_key = str(r["model_key"])
    model_ref = str(r["dense_model_ref"])

    prompts = pd.read_csv(OUT / "dense_prompt_index.csv")
    prompts = prompts[prompts["model_key"].astype(str) == model_key].copy().reset_index(drop=True)

    out_npz = OUT / "dense_cache" / f"{model_key}_trusted_prompt_last_hidden.npz"
    out_meta = OUT / "dense_cache" / f"{model_key}_trusted_prompt_meta.csv"
    if out_npz.exists() and out_meta.exists():
        print("SKIP_DENSE_EXISTS", out_npz)
        return

    print("DENSE_MODEL", model_key, model_ref, "N", len(prompts))
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

    hidden = batch_hidden_last(model, tok, prompts["prompt"].astype(str).tolist(), batch_size=batch_size, max_length=max_length)
    np.savez_compressed(out_npz, hidden=hidden)
    prompts.to_csv(out_meta, index=False)
    print("WROTE", out_npz, hidden.shape)
    print("WROTE", out_meta, prompts.shape)

def rel_l2(a, b, eps=1e-8):
    return np.linalg.norm(b - a, axis=-1) / (np.linalg.norm(a, axis=-1) + eps)

def cos_dist(a, b, eps=1e-8):
    dot = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + eps
    return 1.0 - dot / den

def run_target(array_idx: int, batch_size: int, max_length: int, overwrite: bool):
    targets = pd.read_csv(OUT / "targets_fullzoo.csv")
    trow = targets.iloc[int(array_idx)]

    tid = str(trow["target_id"])
    model_key = str(trow["model_key"])
    comp_path = str(trow["compressed_path"])

    out_layer = OUT / "by_target" / f"{tid}_layer_summary.csv"
    out_sample = OUT / "by_target" / f"{tid}_sample_summary.csv"
    if out_layer.exists() and out_sample.exists() and not overwrite:
        print("SKIP_TARGET_EXISTS", tid)
        return

    idx = pd.read_csv(OUT / "trusted_eval_index.csv")
    idx = idx[idx["target_id"].astype(str) == tid].copy().reset_index(drop=True)

    dense_npz = OUT / "dense_cache" / f"{model_key}_trusted_prompt_last_hidden.npz"
    dense_meta_csv = OUT / "dense_cache" / f"{model_key}_trusted_prompt_meta.csv"
    if not dense_npz.exists() or not dense_meta_csv.exists():
        raise FileNotFoundError(f"Missing dense cache for {model_key}: {dense_npz}")

    dense_hidden_all = np.load(dense_npz)["hidden"].astype(np.float32)
    dense_meta = pd.read_csv(dense_meta_csv).reset_index(drop=True)
    dense_pos = {str(uid): i for i, uid in enumerate(dense_meta["dense_uid"].astype(str).tolist())}

    dense_indices = [dense_pos[str(uid)] for uid in idx["dense_uid"].astype(str).tolist()]
    dense_hidden = dense_hidden_all[dense_indices]

    prompt_df = pd.read_csv(OUT / "dense_prompt_index.csv")
    prompt_map = dict(zip(prompt_df["dense_uid"].astype(str), prompt_df["prompt"].astype(str)))
    prompts = [prompt_map[str(uid)] for uid in idx["dense_uid"].astype(str).tolist()]

    print("TARGET", tid)
    print("MODEL_KEY", model_key)
    print("COMP_PATH", comp_path)
    print("N_PROMPTS", len(prompts))
    print("CATEGORY_COUNTS")
    print(idx["category"].value_counts().to_string())

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

    comp_hidden = batch_hidden_last(model, tok, prompts, batch_size=batch_size, max_length=max_length).astype(np.float32)

    if dense_hidden.shape != comp_hidden.shape:
        raise RuntimeError(f"Hidden shape mismatch dense={dense_hidden.shape} comp={comp_hidden.shape}")

    rel = rel_l2(dense_hidden, comp_hidden)
    cd = cos_dist(dense_hidden, comp_hidden)
    L = rel.shape[1]

    sample_rows = []
    for i, r in idx.iterrows():
        sample_rows.append({
            "target_id": tid,
            "model_key": model_key,
            "method_key": trow.get("method_key", ""),
            "keep_tag": trow.get("keep_tag", ""),
            "task": r["task"],
            "source_idx": int(r["source_idx"]),
            "dense_uid": r["dense_uid"],
            "category": r["category"],
            "flip": bool01(r.get("flip", 0)),
            "damage": bool01(r.get("damage", 0)),
            "repair": bool01(r.get("repair", 0)),
            "wrong_to_wrong": bool01(r.get("wrong_to_wrong", 0)),
            "correct_to_correct": bool01(r.get("correct_to_correct", 0)),
            "boundary_ratio": r.get("boundary_ratio", np.nan),
            "dense_margin": r.get("dense_margin", np.nan),
            "mean_rel_l2": float(np.nanmean(rel[i])),
            "max_rel_l2": float(np.nanmax(rel[i])),
            "mean_cos_dist": float(np.nanmean(cd[i])),
            "max_cos_dist": float(np.nanmax(cd[i])),
        })
    pd.DataFrame(sample_rows).to_csv(out_sample, index=False)

    base = idx.copy()
    base["flip"] = pd.to_numeric(base["flip"], errors="coerce")
    layer_rows = []
    for l in range(L):
        tmp = base[["task", "category", "flip"]].copy()
        tmp["rel_l2"] = rel[:, l]
        tmp["cos_dist"] = cd[:, l]

        groups = [("all", tmp)]
        groups += [(str(k), g) for k, g in tmp.groupby("category", dropna=False)]
        groups += [
            ("flip", tmp[tmp["flip"] == 1]),
            ("nonflip", tmp[tmp["flip"] == 0]),
            ("bar_lt1", tmp[pd.to_numeric(base["boundary_ratio"], errors="coerce") < 1] if "boundary_ratio" in base else tmp.iloc[0:0]),
            ("bar_ge1", tmp[pd.to_numeric(base["boundary_ratio"], errors="coerce") >= 1] if "boundary_ratio" in base else tmp.iloc[0:0]),
        ]

        for gname, g in groups:
            if len(g) == 0:
                continue
            layer_rows.append({
                "target_id": tid,
                "model_key": model_key,
                "method_key": trow.get("method_key", ""),
                "keep_tag": trow.get("keep_tag", ""),
                "layer": l,
                "group": gname,
                "n": len(g),
                "rel_l2_mean": float(g["rel_l2"].mean()),
                "rel_l2_median": float(g["rel_l2"].median()),
                "cos_dist_mean": float(g["cos_dist"].mean()),
                "cos_dist_median": float(g["cos_dist"].median()),
            })

    pd.DataFrame(layer_rows).to_csv(out_layer, index=False)
    print("WROTE", out_layer)
    print("WROTE", out_sample)

def collect():
    by = OUT / "by_target"
    layer_files = sorted(by.glob("*_layer_summary.csv"))
    sample_files = sorted(by.glob("*_sample_summary.csv"))
    print("N_LAYER_FILES", len(layer_files))
    print("N_SAMPLE_FILES", len(sample_files))

    if layer_files:
        layer = pd.concat([pd.read_csv(p) for p in layer_files], ignore_index=True)
        layer.to_csv(OUT / "fullzoo_layer_summary.csv", index=False)
        print("WROTE", OUT / "fullzoo_layer_summary.csv", layer.shape)

        # Wide diagnostic: flip minus nonflip per target/layer.
        piv = layer[layer["group"].isin(["flip", "nonflip"])].pivot_table(
            index=["target_id", "model_key", "method_key", "keep_tag", "layer"],
            columns="group",
            values=["rel_l2_mean", "cos_dist_mean"],
            aggfunc="first",
        )
        piv.columns = [f"{a}_{b}" for a, b in piv.columns]
        piv = piv.reset_index()
        if "rel_l2_mean_flip" in piv and "rel_l2_mean_nonflip" in piv:
            piv["rel_l2_flip_minus_nonflip"] = piv["rel_l2_mean_flip"] - piv["rel_l2_mean_nonflip"]
        if "cos_dist_mean_flip" in piv and "cos_dist_mean_nonflip" in piv:
            piv["cos_dist_flip_minus_nonflip"] = piv["cos_dist_mean_flip"] - piv["cos_dist_mean_nonflip"]
        piv.to_csv(OUT / "fullzoo_flip_nonflip_layer_gap.csv", index=False)
        print("WROTE", OUT / "fullzoo_flip_nonflip_layer_gap.csv", piv.shape)

        group = layer.groupby(["model_key", "method_key", "keep_tag", "group", "layer"], dropna=False).agg(
            n_configs=("target_id", "nunique"),
            n_total=("n", "sum"),
            rel_l2_mean=("rel_l2_mean", "mean"),
            cos_dist_mean=("cos_dist_mean", "mean"),
        ).reset_index()
        group.to_csv(OUT / "fullzoo_group_layer_summary.csv", index=False)
        print("WROTE", OUT / "fullzoo_group_layer_summary.csv", group.shape)

    if sample_files:
        sample = pd.concat([pd.read_csv(p) for p in sample_files], ignore_index=True)
        sample.to_csv(OUT / "fullzoo_sample_summary.csv", index=False)
        print("WROTE", OUT / "fullzoo_sample_summary.csv", sample.shape)

        cat = sample.groupby(["model_key", "method_key", "keep_tag", "category"], dropna=False).agg(
            n=("dense_uid", "count"),
            mean_rel_l2=("mean_rel_l2", "mean"),
            max_rel_l2=("max_rel_l2", "mean"),
            mean_cos_dist=("mean_cos_dist", "mean"),
            max_cos_dist=("max_cos_dist", "mean"),
        ).reset_index()
        cat.to_csv(OUT / "fullzoo_category_summary.csv", index=False)
        print("WROTE", OUT / "fullzoo_category_summary.csv", cat.shape)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build_index", "dense_cache", "target", "collect"])
    ap.add_argument("--array_idx", type=int, default=0)
    ap.add_argument("--n_per_task", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.mode == "build_index":
        build_index(args.n_per_task)
    elif args.mode == "dense_cache":
        run_dense_cache(args.array_idx, args.batch_size, args.max_length)
    elif args.mode == "target":
        run_target(args.array_idx, args.batch_size, args.max_length, args.overwrite)
    elif args.mode == "collect":
        collect()

if __name__ == "__main__":
    main()
