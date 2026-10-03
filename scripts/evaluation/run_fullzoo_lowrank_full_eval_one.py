import argparse
import csv
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download


TASKS = [
    "boolq",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "piqa",
    "hellaswag",
    "openbookqa",
]

def task_has_samples(outdir: Path, task: str) -> bool:
    return outdir.exists() and any(outdir.rglob(f"samples_{task}_*.jsonl"))

def all_done(outdir: Path) -> bool:
    return all(task_has_samples(outdir, t) for t in TASKS)

def resolve_model(row, checkpoint_repo):
    eval_model = row.get("eval_model", "").strip()
    if eval_model:
        return eval_model

    remote_prefix = row.get("remote_prefix", "").strip()
    if remote_prefix:
        if not checkpoint_repo:
            raise ValueError(
                "A row uses remote_prefix, but --checkpoint_repo was not provided."
            )
        local = snapshot_download(
            repo_id=checkpoint_repo,
            repo_type="model",
            allow_patterns=[remote_prefix + "/**", remote_prefix + "/*"],
        )
        ckpt_dir = Path(local) / remote_prefix
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint dir not found after snapshot_download: {ckpt_dir}")
        return str(ckpt_dir)

    dense_hf = row.get("dense_model_hf", "").strip()
    if dense_hf:
        return dense_hf

    raise RuntimeError(f"Cannot resolve model for row: {row}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_csv", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--batch_size", default="auto")
    ap.add_argument(
        "--checkpoint_repo",
        default=None,
        help=(
            "Hugging Face repository containing remote_prefix entries. "
            "Required only when the selected row uses remote_prefix."
        ),
    )
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.target_csv)))
    row = rows[args.index]

    target_id = row["target_id"]
    outdir = Path(row["output_dir"])

    print("TARGET", target_id, flush=True)
    print("MODEL_KEY", row["model_key"], flush=True)
    print("METHOD", row["method_key"], flush=True)
    print("KEEP", row["keep_tag"], flush=True)
    print("OUTDIR", outdir, flush=True)

    if all_done(outdir):
        print("FULL_EVAL_SKIP_EXISTS", target_id, flush=True)
        return

    model_ref = resolve_model(row, args.checkpoint_repo)
    print("MODEL_REF", model_ref, flush=True)

    cmd = [
        "lm-eval", "run",
        "--model", "hf",
        "--model_args", f"pretrained={model_ref},dtype=float16",
        "--tasks", ",".join(TASKS),
        "--output_path", str(outdir),
        "--batch_size", args.batch_size,
        "--device", "cuda:0",
        "--num_fewshot", "0",
        "--trust_remote_code",
        "--log_samples",
    ]

    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    missing = [t for t in TASKS if not task_has_samples(outdir, t)]
    if missing:
        raise RuntimeError(f"FULL_EVAL finished but missing sample jsonl for tasks: {missing}")

    print("FULL_EVAL_OK", target_id, flush=True)

if __name__ == "__main__":
    main()
