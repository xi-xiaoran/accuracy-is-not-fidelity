from pathlib import Path
import argparse
import json
import re
import gc
import traceback
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

ROOT = Path(".").resolve()
PKG = ROOT / "results/paper_packages/stageB1c_effective_weight_residual_fullzoo_v1_20260702"
DATA = PKG / "data"
OUT_DIR = DATA / "per_target"
SUM_DIR = DATA / "per_target_summary"
SHAPE_DIR = DATA / "per_target_shape"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUM_DIR.mkdir(parents=True, exist_ok=True)
SHAPE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_RE = re.compile(
    r"^model\.layers\.(\d+)\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
)

def module_type_from_name(name):
    for m in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        if name.endswith("." + m):
            return m
    return "other"

def layer_index_from_name(name):
    m = re.search(r"model\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1

def is_target_module_name(name):
    return TARGET_RE.match(name) is not None

def load_model(path):
    return AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

def dense_linear_map(model):
    out = {}
    for name, m in model.named_modules():
        if is_target_module_name(name) and isinstance(m, nn.Linear):
            out[name] = m
    return out

def compressed_target_map(model):
    out = {}
    for name, m in model.named_modules():
        if not is_target_module_name(name):
            continue
        cls = m.__class__.__name__
        if isinstance(m, nn.Linear) or cls in {"ASVDLinear", "LowRankLinear", "SVDLinear"}:
            out[name] = m
    return out

def materialize_weight(m):
    cls = m.__class__.__name__

    if hasattr(m, "ALinear") and hasattr(m, "BLinear"):
        A = m.ALinear.weight.detach().float()
        B = m.BLinear.weight.detach().float()
        W = A @ B
        return W, "ALinear@BLinear", int(B.shape[0])

    if hasattr(m, "u_proj") and hasattr(m, "v_proj"):
        U = m.u_proj.weight.detach().float()
        V = m.v_proj.weight.detach().float()
        W = U @ V
        return W, "u_proj@v_proj", int(V.shape[0])

    if isinstance(m, nn.Linear):
        W = m.weight.detach().float()
        return W, "Linear", None

    return None, f"unsupported:{cls}", None

def stats(dense_w, comp_w):
    dense = dense_w.float()
    comp = comp_w.float()
    diff = comp - dense
    dense_sq = torch.sum(dense * dense).item()
    comp_sq = torch.sum(comp * comp).item()
    diff_sq = torch.sum(diff * diff).item()
    dot = torch.sum(dense * comp).item()
    dense_norm = dense_sq ** 0.5
    comp_norm = comp_sq ** 0.5
    diff_norm = diff_sq ** 0.5
    rel = diff_norm / (dense_norm + 1e-12)
    cos = dot / ((dense_norm * comp_norm) + 1e-12)
    mean_abs = torch.mean(torch.abs(diff)).item()
    max_abs = torch.max(torch.abs(diff)).item()
    return {
        "dense_sq": dense_sq,
        "compressed_sq": comp_sq,
        "diff_sq": diff_sq,
        "dense_norm": dense_norm,
        "compressed_norm": comp_norm,
        "diff_norm": diff_norm,
        "rel_frob": rel,
        "cosine": cos,
        "mean_abs_diff": mean_abs,
        "max_abs_diff": max_abs,
    }

def shape_ratio(dense_shape, comp_shape):
    if dense_shape is None or comp_shape is None:
        return {}
    dense_params = int(np.prod(dense_shape))
    comp_params = int(np.prod(comp_shape))
    out = {
        "dense_shape": "x".join(map(str, dense_shape)),
        "compressed_shape": "x".join(map(str, comp_shape)),
        "dense_params": dense_params,
        "compressed_params": comp_params,
        "param_retention": comp_params / dense_params if dense_params else np.nan,
    }
    if len(dense_shape) == 2 and len(comp_shape) == 2:
        out["out_dim_retention"] = comp_shape[0] / dense_shape[0] if dense_shape[0] else np.nan
        out["in_dim_retention"] = comp_shape[1] / dense_shape[1] if dense_shape[1] else np.nan
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-index", type=int, required=True)
    args = parser.parse_args()

    torch.set_num_threads(2)

    targets = pd.read_csv(DATA / "stageB1c_targets.csv")
    row = targets.iloc[args.target_index].to_dict()

    target_id = row["target_id"]
    residual_csv = OUT_DIR / f"{target_id}_effective_residual.csv"
    shape_csv = SHAPE_DIR / f"{target_id}_shape_profile.csv"
    summary_json = SUM_DIR / f"{target_id}_summary.json"

    if residual_csv.exists() and shape_csv.exists() and summary_json.exists():
        print(f"SKIP existing {target_id}")
        return

    status = {
        "target_index": int(args.target_index),
        "target_id": target_id,
        "model_key": row["model_key"],
        "method_key": row["method_key"],
        "keep_ratio": row.get("keep_ratio", ""),
        "residual_mode": row["residual_mode"],
        "base_snapshot_path": row["base_snapshot_path"],
        "compressed_path": row["compressed_path"],
        "ok": False,
        "error": "",
    }

    residual_rows = []
    shape_rows = []

    try:
        dense = load_model(row["base_snapshot_path"])
        comp = load_model(row["compressed_path"])

        status["dense_model_class"] = dense.__class__.__name__
        status["compressed_model_class"] = comp.__class__.__name__

        dense_map = dense_linear_map(dense)
        comp_map = compressed_target_map(comp)

        status["n_dense_target_modules"] = len(dense_map)
        status["n_compressed_target_modules"] = len(comp_map)

        for name, dm in sorted(dense_map.items()):
            mt = module_type_from_name(name)
            layer_idx = layer_index_from_name(name)

            base_rec = {
                "target_id": target_id,
                "model_key": row["model_key"],
                "method_key": row["method_key"],
                "keep_ratio": row.get("keep_ratio", ""),
                "module_name": name,
                "module_type": mt,
                "layer_index": layer_idx,
            }

            dense_w = dm.weight.detach().float()
            dense_shape = tuple(dense_w.shape)

            cm = comp_map.get(name)
            if cm is None:
                rec = dict(base_rec)
                rec.update({
                    "status": "missing_compressed_module",
                    "dense_shape": "x".join(map(str, dense_shape)),
                    "compressed_shape": "",
                })
                shape_rows.append(rec)
                continue

            comp_w, materialization, rank = materialize_weight(cm)
            comp_shape = tuple(comp_w.shape) if comp_w is not None else None

            srec = dict(base_rec)
            srec.update({
                "compressed_module_class": cm.__class__.__name__,
                "materialization": materialization,
                "rank": rank,
            })
            srec.update(shape_ratio(dense_shape, comp_shape))
            shape_rows.append(srec)

            # ModeGPT is structural width pruning. Do not mix it into residual correlations.
            if row["residual_mode"] == "shape_profile_only":
                if comp_w is not None:
                    del comp_w
                del dense_w
                continue

            rec = dict(base_rec)
            rec.update({
                "compressed_module_class": cm.__class__.__name__,
                "materialization": materialization,
                "rank": rank,
                "dense_shape": "x".join(map(str, dense_shape)),
                "compressed_shape": "x".join(map(str, comp_shape)) if comp_shape else "",
                "ok": False,
                "error": "",
            })

            if comp_w is None:
                rec["error"] = "unsupported_materialization"
                residual_rows.append(rec)
                del dense_w
                continue

            if tuple(comp_w.shape) != dense_shape:
                rec["error"] = "shape_mismatch"
                residual_rows.append(rec)
                del dense_w, comp_w
                continue

            st = stats(dense_w, comp_w)
            rec.update(st)
            rec["n_params"] = int(np.prod(dense_shape))
            rec["ok"] = True
            residual_rows.append(rec)

            del dense_w, comp_w
            gc.collect()

        res_df = pd.DataFrame(residual_rows)
        shape_df = pd.DataFrame(shape_rows)

        res_df.to_csv(residual_csv, index=False)
        shape_df.to_csv(shape_csv, index=False)

        ok_df = res_df[res_df["ok"] == True].copy() if len(res_df) else pd.DataFrame()
        status["n_residual_rows"] = len(res_df)
        status["n_residual_ok"] = len(ok_df)
        status["n_shape_rows"] = len(shape_df)

        if len(res_df):
            status["n_shape_mismatch"] = int((res_df.get("error", "") == "shape_mismatch").sum())
            status["n_factorized_modules"] = int(res_df["materialization"].isin(["ALinear@BLinear", "u_proj@v_proj"]).sum())
            status["n_linear_modules"] = int((res_df["materialization"] == "Linear").sum())
        else:
            status["n_shape_mismatch"] = 0
            status["n_factorized_modules"] = 0
            status["n_linear_modules"] = 0

        if len(ok_df):
            dense_sq = ok_df["dense_sq"].sum()
            diff_sq = ok_df["diff_sq"].sum()
            status["global_rel_frob"] = float((diff_sq / (dense_sq + 1e-12)) ** 0.5)
            status["mean_module_rel_frob"] = float(ok_df["rel_frob"].mean())
            status["median_module_rel_frob"] = float(ok_df["rel_frob"].median())
            status["mean_cosine"] = float(ok_df["cosine"].mean())
        else:
            status["global_rel_frob"] = None
            status["mean_module_rel_frob"] = None
            status["median_module_rel_frob"] = None
            status["mean_cosine"] = None

        del dense, comp
        gc.collect()

        status["ok"] = True

    except Exception as e:
        status["ok"] = False
        status["error"] = repr(e)
        status["traceback"] = traceback.format_exc()

    summary_json.write_text(json.dumps(status, indent=2, sort_keys=True))
    print(json.dumps(status, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
