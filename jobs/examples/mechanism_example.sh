#!/usr/bin/env bash
# Generic example only: choose resources appropriate for your model and site.
#SBATCH --job-name=code-release-mechanism
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00

set -euo pipefail

RELEASE_ROOT=${RELEASE_ROOT:?Set RELEASE_ROOT to this extracted release}
WORKSPACE_ROOT=${WORKSPACE_ROOT:?Set WORKSPACE_ROOT to your analysis workspace}

cd "$WORKSPACE_ROOT"
export PYTHONPATH="$RELEASE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Inspect the selected entry point before running a costly extraction.
python -m scripts.mechanism.mech_choice_v1 --help
