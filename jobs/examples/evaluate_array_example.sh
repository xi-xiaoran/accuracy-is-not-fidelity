#!/usr/bin/env bash
# Generic example only: add site-approved account/partition settings yourself.
#SBATCH --job-name=code-release-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-0

set -euo pipefail

RELEASE_ROOT=${RELEASE_ROOT:?Set RELEASE_ROOT to this extracted release}
WORKSPACE_ROOT=${WORKSPACE_ROOT:?Set WORKSPACE_ROOT to your analysis workspace}
MANIFEST_CSV=${MANIFEST_CSV:?Set MANIFEST_CSV to your manifest}
CHECKPOINT_REPO=${CHECKPOINT_REPO:?Set CHECKPOINT_REPO to an accessible repository}

cd "$WORKSPACE_ROOT"
export PYTHONPATH="$RELEASE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -m scripts.evaluation.run_fullzoo_lowrank_full_eval_one \
  --target_csv "$MANIFEST_CSV" \
  --index "${SLURM_ARRAY_TASK_ID:-0}" \
  --checkpoint_repo "$CHECKPOINT_REPO" \
  --batch_size auto
