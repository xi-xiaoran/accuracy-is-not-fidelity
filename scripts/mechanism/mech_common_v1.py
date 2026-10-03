import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
SNAP = Path(os.environ.get("CHECKPOINT_REPO_ROOT", "checkpoints")).resolve()
OUT = ROOT / "results/mechanism_activation_v1"

TASKS = [
    "boolq",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "piqa",
    "hellaswag",
    "openbookqa",
]

DENSE_MODEL_FALLBACK = {
    "llama2_7b": "meta-llama/Llama-2-7b-hf",
    "llama31_8b": "meta-llama/Llama-3.1-8B",
    "llama31_8b_instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama_7b": "huggyllama/llama-7b",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_8b_base": "Qwen/Qwen3-8B-Base",
}

def norm_keep(x: Any) -> str:
    s = str(x)
    if s.startswith("keep"):
        return s
    try:
        v = float(s)
        return f"keep0p{int(round(v*10))}"
    except Exception:
        return s.replace(".", "p")

def find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        if n.lower() in low:
            return low[n.lower()]
    return None

def first_number(x: Any) -> float:
    if isinstance(x, (int, float, np.number)):
        return float(x)
    if isinstance(x, str):
        return float(x)
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return float("nan")
        return first_number(x[0])
    raise ValueError(f"Cannot parse numeric score from {type(x)}: {repr(x)[:200]}")

def choice_texts(sample: Dict[str, Any]) -> List[str]:
    args = sample.get("arguments", {})
    out = []
    for k in sorted(args.keys(), key=lambda z: int(str(z).split("_")[-1]) if str(z).split("_")[-1].isdigit() else str(z)):
        v = args[k]
        if isinstance(v, dict) and "arg_1" in v:
            out.append(str(v["arg_1"]))
    if out:
        return out
    doc = sample.get("doc", {})
    ch = doc.get("choices", None)
    if isinstance(ch, dict) and "text" in ch:
        return [str(x) for x in ch["text"]]
    if isinstance(ch, list):
        return [str(x) for x in ch]
    return []

def prompt_text(sample: Dict[str, Any]) -> str:
    args = sample.get("arguments", {})
    if isinstance(args, dict):
        if "gen_args_0" in args and isinstance(args["gen_args_0"], dict) and "arg_0" in args["gen_args_0"]:
            return str(args["gen_args_0"]["arg_0"])
        for k in sorted(args.keys()):
            v = args[k]
            if isinstance(v, dict) and "arg_0" in v:
                return str(v["arg_0"])
    doc = sample.get("doc", {})
    if "question" in doc:
        return "Question: " + str(doc["question"]) + "\nAnswer:"
    return str(doc)

def score_vector_normchar(sample: Dict[str, Any]) -> np.ndarray:
    raw = sample.get("filtered_resps", None)
    if raw is None:
        raw = sample.get("resps", None)
    if raw is None:
        raise RuntimeError(f"Cannot find filtered_resps/resps in sample keys={list(sample.keys())}")
    scores = np.array([first_number(x) for x in raw], dtype=np.float64)
    texts = choice_texts(sample)
    if len(texts) == len(scores):
        lens = np.array([max(1, len(t)) for t in texts], dtype=np.float64)
        scores = scores / lens
    return scores

def pred_from_sample(sample: Dict[str, Any]) -> int:
    s = score_vector_normchar(sample)
    return int(np.nanargmax(s))

def gold_from_sample(sample: Dict[str, Any]) -> Optional[int]:
    t = sample.get("target", None)
    if isinstance(t, int):
        return int(t)
    if isinstance(t, str):
        st = t.strip()
        if st.isdigit():
            return int(st)
        # A/B/C/D label fallback
        if len(st) == 1 and st.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return ord(st.upper()) - ord("A")
    doc = sample.get("doc", {})
    ak = doc.get("answerKey", None)
    if isinstance(ak, str):
        ak = ak.strip()
        if ak.isdigit():
            return int(ak)
        if len(ak) == 1 and ak.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return ord(ak.upper()) - ord("A")
    return None

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def latest_sample_file(run_dir: Path, task: str) -> Optional[Path]:
    if not run_dir.exists():
        return None
    files = sorted(run_dir.rglob(f"samples_{task}_*.jsonl"))
    return files[-1] if files else None

def locate_dense_run_dir(model_key: str) -> Path:
    base = ROOT / "results/official_lmeval/fullzoo_lowrank_severity_dense_full"
    cand = base / model_key
    if cand.exists():
        # Usually one nested dir under model_key.
        sub_files = list(cand.rglob("samples_*.jsonl"))
        if sub_files:
            return cand
    # fallback search
    hits = []
    for p in base.rglob("samples_*.jsonl"):
        if model_key in str(p):
            hits.append(p.parent)
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Cannot locate dense samples for {model_key} under {base}")

def locate_compressed_run_dir(target_id: str) -> Path:
    base = ROOT / "results/official_lmeval/fullzoo_lowrank_severity_compressed_full"
    cand = base / target_id
    if cand.exists():
        return cand
    hits = []
    for p in base.rglob("samples_*.jsonl"):
        if target_id in str(p):
            hits.append(p.parent)
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Cannot locate compressed samples for {target_id} under {base}")

def resolve_local_compressed_path(row: pd.Series) -> str:
    # Prefer explicit local/model path columns if present.
    for c in [
        "model_path", "local_path", "pretrained", "checkpoint_path",
        "hf_path", "resolved_path", "model_ref"
    ]:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            s = str(row[c]).strip()
            if s.startswith("/") or Path(s).exists():
                return s
    # Then use remote prefix/path-in-repo style columns.
    for c in [
        "remote_prefix", "path_in_repo", "public_path", "checkpoint",
        "checkpoint_path_in_repo", "hf_repo_path"
    ]:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            s = str(row[c]).strip()
            if not s.startswith("full_checkpoint_zoo"):
                # Some configs store method-style path; keep only if it exists under SNAP.
                candidate = SNAP / s
            else:
                candidate = SNAP / s
            if candidate.exists():
                return str(candidate)
    # Last resort: infer from model/method/keep.
    model = str(row.get("model_key", row.get("model", "")))
    method = str(row.get("method_key", row.get("method", "")))
    keep = norm_keep(row.get("keep_ratio", row.get("keep", "")))
    candidate = SNAP / "full_checkpoint_zoo" / model / method / keep
    if candidate.exists():
        return str(candidate)
    # SVDLLM naming sometimes uses nested method with slash already in method.
    raise FileNotFoundError(f"Cannot resolve compressed path for row={row.to_dict()}")

def load_targets_csv() -> pd.DataFrame:
    p = ROOT / "configs/fullzoo_lowrank_severity_full_eval_targets.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)

    model_col = find_col(df, ["model_key", "model"])
    method_col = find_col(df, ["method_key", "method"])
    keep_col = find_col(df, ["keep_ratio", "keep"])

    if model_col is None or method_col is None or keep_col is None:
        raise ValueError(f"Cannot infer model/method/keep columns from {list(df.columns)}")

    out = df.copy()
    out["model_key"] = out[model_col].astype(str)
    out["method_key"] = out[method_col].astype(str)
    out["keep_ratio_raw"] = out[keep_col]
    out["keep_tag"] = out[keep_col].map(norm_keep)
    out["target_id"] = out.apply(
        lambda r: f"{r['model_key']}_{str(r['method_key']).replace('/', '_')}_{r['keep_tag']}",
        axis=1,
    )

    # If existing explicit target_id exists, prefer it.
    tid_col = find_col(df, ["target_id", "compressed_name", "config_id"])
    if tid_col is not None:
        out["target_id"] = df[tid_col].astype(str)

    paths = []
    for _, r in out.iterrows():
        paths.append(resolve_local_compressed_path(r))
    out["compressed_path"] = paths
    out["dense_model_ref"] = out["model_key"].map(DENSE_MODEL_FALLBACK)
    return out

def category_from_preds(dense_pred: int, comp_pred: int, gold: Optional[int]) -> str:
    if dense_pred == comp_pred:
        if gold is None:
            return "stable_same_pred"
        if dense_pred == gold:
            return "correct_to_correct"
        return "wrong_stable"
    if gold is None:
        return "flip_unknown_gold"
    d_ok = dense_pred == gold
    c_ok = comp_pred == gold
    if d_ok and not c_ok:
        return "damage"
    if (not d_ok) and c_ok:
        return "repair"
    if (not d_ok) and (not c_ok):
        return "wrong_to_wrong"
    # different pred but both correct can happen with multi-gold tasks, rare
    return "correct_to_correct_flip"

def rel_l2(a: np.ndarray, b: np.ndarray, eps=1e-8) -> np.ndarray:
    num = np.linalg.norm(b - a, axis=-1)
    den = np.linalg.norm(a, axis=-1) + eps
    return num / den

def cos_dist(a: np.ndarray, b: np.ndarray, eps=1e-8) -> np.ndarray:
    dot = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + eps
    cos = dot / den
    return 1.0 - cos
