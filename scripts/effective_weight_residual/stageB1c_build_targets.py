import argparse
from pathlib import Path

import pandas as pd


def repo_to_cache_dir(repo_id):
    return "models--" + repo_id.replace("/", "--")


PARSER = argparse.ArgumentParser(
    description="Build effective-weight residual targets from a checkpoint manifest."
)
PARSER.add_argument("--root", type=Path, default=Path("."), help="Analysis root directory.")
PARSER.add_argument(
    "--hf_hub",
    type=Path,
    default=Path("cache/huggingface/hub"),
    help="Hugging Face hub cache directory.",
)
PARSER.add_argument(
    "--summary",
    type=Path,
    default=Path("results/theory_predictions_v1_fullzoo/fullzoo_theory_summary_by_config.csv"),
    help="Configuration-level theory summary CSV.",
)
PARSER.add_argument(
    "--output_dir",
    type=Path,
    default=Path("results/paper_packages/stageB1c_effective_weight_residual_fullzoo/data"),
    help="Directory for generated target manifests.",
)
PARSER.add_argument(
    "--checkpoint_repo",
    required=True,
    help="Hugging Face repository containing the compressed checkpoint tree.",
)
ARGS = PARSER.parse_args()

ROOT = ARGS.root.resolve()
HF_HUB = (ARGS.hf_hub if ARGS.hf_hub.is_absolute() else ROOT / ARGS.hf_hub).resolve()
SUMMARY = (ARGS.summary if ARGS.summary.is_absolute() else ROOT / ARGS.summary).resolve()
DATA = (ARGS.output_dir if ARGS.output_dir.is_absolute() else ROOT / ARGS.output_dir).resolve()
DATA.mkdir(parents=True, exist_ok=True)
SUBMISSION_CACHE = HF_HUB / repo_to_cache_dir(ARGS.checkpoint_repo)

BASE_REPO_MAP = {
    "llama_7b": "huggyllama/llama-7b",
    "llama2_7b": "meta-llama/Llama-2-7b-hf",
    "llama31_8b": "meta-llama/Llama-3.1-8B",
    "llama31_8b_instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_8b_base": "Qwen/Qwen3-8B-Base",
}

def newest_snapshot(cache_dir):
    cache_dir = Path(cache_dir)
    snaps = []
    if (cache_dir / "snapshots").exists():
        snaps = sorted([p for p in (cache_dir / "snapshots").iterdir() if p.is_dir()])
    return snaps[-1] if snaps else None

summary = pd.read_csv(SUMMARY)

submission_snapshot = newest_snapshot(SUBMISSION_CACHE)
if submission_snapshot is None:
    raise RuntimeError(f"Cannot find submission snapshot under {SUBMISSION_CACHE}")

base_rows = []
for model_key in sorted(summary["model_key"].astype(str).unique()):
    repo = BASE_REPO_MAP[model_key]
    snap = newest_snapshot(HF_HUB / repo_to_cache_dir(repo))
    if snap is None:
        raise RuntimeError(f"No base snapshot for {model_key}: {repo}")
    base_rows.append({
        "model_key": model_key,
        "base_repo": repo,
        "base_snapshot_path": str(snap),
    })

base_df = pd.DataFrame(base_rows)
base_df.to_csv(DATA / "base_snapshot_mapping.csv", index=False)
base_map = dict(zip(base_df["model_key"], base_df["base_snapshot_path"]))

targets = []
for _, r in summary.iterrows():
    target_id = str(r["target_id"])
    model_key = str(r["model_key"])
    method_key = str(r["method_key"])
    remote_prefix = str(r["remote_prefix"])
    compressed_path = submission_snapshot / remote_prefix

    if method_key == "modegpt":
        residual_mode = "shape_profile_only"
    else:
        residual_mode = "effective_weight_residual"

    targets.append({
        "target_index": len(targets),
        "target_id": target_id,
        "model_key": model_key,
        "method_key": method_key,
        "keep_ratio": r.get("keep_ratio", ""),
        "remote_prefix": remote_prefix,
        "base_snapshot_path": base_map[model_key],
        "compressed_path": str(compressed_path),
        "compressed_exists": compressed_path.exists(),
        "residual_mode": residual_mode,
    })

df = pd.DataFrame(targets)
df.to_csv(DATA / "stageB1c_targets.csv", index=False)

print("submission_snapshot", submission_snapshot)
print("targets", len(df))
print("compressed_exists", df["compressed_exists"].sum(), "/", len(df))
print(df.groupby(["method_key", "residual_mode"])["compressed_exists"].agg(["count", "sum", "mean"]).to_string())
