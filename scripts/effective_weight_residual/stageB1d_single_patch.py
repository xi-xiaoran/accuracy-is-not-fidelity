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
PKG = ROOT / "results/paper_packages/stageB1d_effective_weight_patch_dobi_basis_v1_20260702"
DATA = PKG / "data"
OUT_DIR = DATA / "per_target"
SUM_DIR = DATA / "per_target_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUM_DIR.mkdir(parents=True, exist_ok=True)

TARGET_RE = re.compile(
    r"^model\.layers\.(\d+)\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
)

def is_target_module_name(name):
    return TARGET_RE.match(name) is not None

def module_type_from_name(name):
    for m in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        if name.endswith("." + m):
            return m
    return "other"

def layer_index_from_name(name):
    m = re.search(r"model\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1

def load_model(path):
    return AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

def dense_linear_map(model):
    return {
        name: m for name, m in model.named_modules()
        if is_target_module_name(name) and isinstance(m, nn.Linear)
    }

def compressed_map(model):
    out = {}
    for name, m in model.named_modules():
        if not is_target_module_name(name):
            continue
        cls = m.__class__.__name__
        if isinstance(m, nn.Linear) or cls in {"DobiSVDLinear", "LowRankLinear", "ASVDLinear", "SVDLinear"}:
            out[name] = m
    return out

def materialize_weight(m, dense_shape):
    cls = m.__class__.__name__

    if isinstance(m, nn.Linear):
        return m.weight.detach().float(), "Linear", None

    if hasattr(m, "u_proj") and hasattr(m, "v_proj"):
        U = m.u_proj.weight.detach().float()
        V = m.v_proj.weight.detach().float()
        W = U @ V
        return W, "u_proj@v_proj", int(V.shape[0])

    if hasattr(m, "ALinear") and hasattr(m, "BLinear"):
        A = m.ALinear.weight.detach().float()
        B = m.BLinear.weight.detach().float()

        # Try both directions. Different low-rank implementations name A/B differently.
        candidates = []
        if A.ndim == 2 and B.ndim == 2 and A.shape[1] == B.shape[0]:
            candidates.append(("ALinear@BLinear", A @ B, int(A.shape[1])))
        if A.ndim == 2 and B.ndim == 2 and B.shape[1] == A.shape[0]:
            candidates.append(("BLinear@ALinear", B @ A, int(B.shape[1])))

        for name, W, rank in candidates:
            if tuple(W.shape) == tuple(dense_shape):
                return W, name, rank

        # Return first candidate for diagnostics if none matches.
        if candidates:
            name, W, rank = candidates[0]
            return W, "shape_mismatch_" + name, rank

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
    return {
        "dense_sq": dense_sq,
        "compressed_sq": comp_sq,
        "diff_sq": diff_sq,
        "dense_norm": dense_norm,
        "compressed_norm": comp_norm,
        "diff_norm": diff_norm,
        "rel_frob": diff_norm / (dense_norm + 1e-12),
        "cosine": dot / ((dense_norm * comp_norm) + 1e-12),
        "mean_abs_diff": torch.mean(torch.abs(diff)).item(),
        "max_abs_diff": torch.max(torch.abs(diff)).item(),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-index", type=int, required=True)
    args = parser.parse_args()

    torch.set_num_threads(2)

    targets = pd.read_csv(DATA / "stageB1d_patch_targets.csv")
    row = targets.iloc[args.patch_index].to_dict()
    tid = row["target_id"]

    out_csv = OUT_DIR / f"{tid}_effective_residual_patch.csv"
    out_json = SUM_DIR / f"{tid}_summary.json"

    if out_csv.exists() and out_json.exists():
        print(f"SKIP {tid}")
        return

    status = {
        "patch_index": int(args.patch_index),
        "target_id": tid,
        "model_key": row["model_key"],
        "method_key": row["method_key"],
        "keep_ratio": row.get("keep_ratio", ""),
        "patch_reason": row["patch_reason"],
        "base_snapshot_path": row["base_snapshot_path"],
        "compressed_path": row["compressed_path"],
        "ok": False,
        "error": "",
    }

    rows = []
    try:
        dense = load_model(row["base_snapshot_path"])
        comp = load_model(row["compressed_path"])
        status["dense_model_class"] = dense.__class__.__name__
        status["compressed_model_class"] = comp.__class__.__name__

        dense_map = dense_linear_map(dense)
        comp_map = compressed_map(comp)
        status["n_dense_target_modules"] = len(dense_map)
        status["n_compressed_target_modules"] = len(comp_map)

        for name, dm in sorted(dense_map.items()):
            dense_w = dm.weight.detach().float()
            dense_shape = tuple(dense_w.shape)
            cm = comp_map.get(name)

            rec = {
                "target_id": tid,
                "model_key": row["model_key"],
                "method_key": row["method_key"],
                "keep_ratio": row.get("keep_ratio", ""),
                "module_name": name,
                "module_type": module_type_from_name(name),
                "layer_index": layer_index_from_name(name),
                "dense_shape": "x".join(map(str, dense_shape)),
                "ok": False,
                "error": "",
            }

            if cm is None:
                rec["error"] = "missing_compressed_module"
                rows.append(rec)
                del dense_w
                continue

            comp_w, materialization, rank = materialize_weight(cm, dense_shape)
            rec["compressed_module_class"] = cm.__class__.__name__
            rec["materialization"] = materialization
            rec["rank"] = rank
            rec["compressed_shape"] = "x".join(map(str, tuple(comp_w.shape))) if comp_w is not None else ""

            if comp_w is None:
                rec["error"] = "unsupported_materialization"
                rows.append(rec)
                del dense_w
                continue

            if tuple(comp_w.shape) != dense_shape:
                rec["error"] = "shape_mismatch"
                rows.append(rec)
                del dense_w, comp_w
                continue

            rec.update(stats(dense_w, comp_w))
            rec["n_params"] = int(np.prod(dense_shape))
            rec["ok"] = True
            rows.append(rec)

            del dense_w, comp_w
            gc.collect()

        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)

        ok_df = df[df["ok"] == True].copy()
        status["n_residual_rows"] = len(df)
        status["n_residual_ok"] = len(ok_df)
        status["n_shape_mismatch"] = int((df.get("error", "") == "shape_mismatch").sum()) if len(df) else 0
        status["materialization_counts"] = df["materialization"].value_counts(dropna=False).to_dict() if "materialization" in df.columns else {}

        if len(ok_df):
            dense_sq = ok_df["dense_sq"].sum()
            diff_sq = ok_df["diff_sq"].sum()
            status["global_rel_frob"] = float((diff_sq / (dense_sq + 1e-12)) ** 0.5)
            status["mean_module_rel_frob"] = float(ok_df["rel_frob"].mean())
            status["median_module_rel_frob"] = float(ok_df["rel_frob"].median())
            status["mean_cosine"] = float(ok_df["cosine"].mean())
        else:
            status["global_rel_frob"] = None

        del dense, comp
        gc.collect()

        status["ok"] = True

    except Exception as e:
        status["ok"] = False
        status["error"] = repr(e)
        status["traceback"] = traceback.format_exc()

    out_json.write_text(json.dumps(status, indent=2, sort_keys=True))
    print(json.dumps(status, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
