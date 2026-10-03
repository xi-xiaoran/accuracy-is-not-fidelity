# Accuracy Is Not Fidelity

Code for evaluating **decision churn in low-rank compressed language models**. The repository provides evaluation, parsing, metric, matching, boundary-diagnostic, mechanism, robustness, and table/figure-generation pipelines used by the accompanying paper.

This is a **code-only release**. Model weights, benchmark data, model outputs, activation caches, frozen result files, paper tables, and figures are intentionally excluded.

## Repository structure

```text
.
├── configs/                  # Portable YAML schema examples
├── jobs/examples/            # Generic SLURM examples
├── scripts/
│   ├── evaluation/           # lm-eval checkpoint evaluation
│   ├── parsing/              # Sample parsing, alignment, and protocol checks
│   ├── metrics/              # Churn, JS, theory, and selective-repair analyses
│   ├── accuracy_matched/     # Accuracy-matched pairs and bootstrap analyses
│   ├── boundary_diagnostics/ # BAR and decision-boundary diagnostics
│   ├── mechanism/            # Prompt and answer-path representation analyses
│   ├── effective_weight_residual/
│   ├── score_correction/
│   ├── high_retention/
│   ├── sample_selection/
│   └── tables_and_figures/
├── tests/                    # Synthetic unit tests
├── requirements.txt
├── requirements-eval-torch24.txt
├── requirements-modegpt-torch26.txt
└── LICENSE
```

## Installation

Python 3.10 is recommended.

For CPU-side analysis, parsing, tables, figures, and tests:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the primary evaluation environment:

```bash
python3.10 -m venv .venv-eval
source .venv-eval/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-eval-torch24.txt
```

MoDeGPT-related workflows use a separate PyTorch 2.6 environment:

```bash
python3.10 -m venv .venv-modegpt
source .venv-modegpt/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-modegpt-torch26.txt
```

The experiments were run on Rocky Linux 9.4 with Python 3.10.20 and CUDA 12.4. The primary environment used PyTorch 2.4.0+cu124; the MoDeGPT workflow used PyTorch 2.6.0. GPU classes used across the experiments included NVIDIA A100 40/80 GB, H200 141 GB, and L40S 48-GB-class devices.

## External assets

Users must obtain and license the following separately:

1. Dense base models and compressed checkpoints.
2. Public benchmark datasets loaded through `lm-eval-harness`.
3. User-generated lm-eval aggregate JSON files and sample JSONL logs.
4. Intermediate summaries required by the table and figure scripts.

### Checkpoint provenance

The 235 compressed checkpoints used in the main audit were drawn from the following anonymous artifact release:

- Code, manifests, and evaluation utilities:  
  `https://anonymous.4open.science/r/lowrankarena-7883`
- Compressed checkpoint repository:  
  `https://huggingface.co/neurips-ed2026-anon-checkpoints/submission-checkpoints/tree/main`

Checkpoint weights are not redistributed in this code-only package. Users should obtain the corresponding artifacts from the anonymous release above and provide their local paths or repository prefixes through the evaluation manifest. Dense source models are referenced by their public model identifiers in the manifests used by the evaluation pipeline.

The benchmark suite uses BoolQ, ARC-Easy, ARC-Challenge, WinoGrande, PIQA, HellaSwag, and OpenBookQA. No model weights, tokenizers, datasets, or third-party outputs are redistributed here.

## Working-directory convention

Most pipeline scripts resolve `configs/`, `results/`, and `logs/` relative to the current working directory. A typical setup is:

```bash
export REPO_ROOT=/path/to/AAAI2027_Anonymous_Code
export WORKSPACE_ROOT=/path/to/your/analysis-workspace
cd "$WORKSPACE_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CHECKPOINT_REPO_ROOT=/path/to/compressed/checkpoints
```

Scripts with explicit path arguments document them through `--help`.

## Input schemas

The YAML files in `configs/` provide portable schema examples and contain no experimental observations.

### Evaluation manifest

The evaluation runner accepts a CSV with one row per checkpoint. Required columns are:

- `target_id`
- `model_key`
- `method_key`
- `keep_tag`
- `output_dir`

Each row must provide one model locator:

- `eval_model`: local path or public model identifier;
- `remote_prefix`: path inside `--checkpoint_repo`; or
- `dense_model_hf`: public dense-model identifier.

### lm-eval samples

The parser expects one `samples_<task>_*.jsonl` file beneath each model output directory. Records require `doc_id`, `target`, `arguments`, and choice responses in `filtered_resps`; compatible fallback response fields are handled by the parser. Optional `acc` and `acc_norm` fields support protocol checks.

Dense and compressed samples are aligned by a stable key derived from task and document content. Alignment checks fail on missing or inconsistent records.

### Configuration-level metrics

Accuracy-matching inputs use the fields documented in `configs/analysis_example.yaml`, including checkpoint identity, method grouping, dense/compressed accuracy, accuracy change, PredFlip, HiddenChurn, Damage, Repair, Wrong-to-wrong, JS drift, and BAR-related quantities. Percentage-valued fields use percentage-point units.

The one-to-one matcher is deterministic and outcome-blind: candidates are ordered by absolute accuracy-change gap and stable checkpoint identifiers only.

### Mechanism and effective-weight inputs

Mechanism correlations require `metric_col`, `target`, and `spearman`, with optional `metric_name`, `n`, and `pearson` fields. The table builder selects prompt and answer-path signals through exact feature and target keys rather than fuzzy matching.

Effective-weight tables use correlation fields `scope`, `feature`, `metric`, and `spearman`, plus coverage fields such as `method_key` and `has_effective_residual`.

## Quick start

Evaluate one checkpoint from a manifest:

```bash
python -m scripts.evaluation.run_fullzoo_lowrank_full_eval_one \
  --target_csv /path/to/checkpoint_manifest.csv \
  --index 0 \
  --checkpoint_repo neurips-ed2026-anon-checkpoints/submission-checkpoints \
  --batch_size auto
```

Parse dense/compressed outputs and build aggregate metrics:

```bash
python -m scripts.parsing.build_fullzoo_lowrank_pair_metrics_tables
```

Run accuracy-matched analysis:

```bash
python -m scripts.accuracy_matched.build_accuracy_matched_v1 \
  /path/to/configuration_metrics.csv \
  /path/to/output/accuracy_matched
```

Inspect GPU-heavy mechanism entry points before running them:

```bash
python -m scripts.mechanism.mech_activation_v2_trusted --help
python -m scripts.mechanism.mech_choice_v1 --help
python -m scripts.effective_weight_residual.stageB1c_single_effective_residual --help
```

Generate tables and figures from user-supplied intermediate results:

```bash
python -m scripts.tables_and_figures.build_manuscript_tables \
  --project-root "$WORKSPACE_ROOT" \
  --output-dir "$WORKSPACE_ROOT/generated_tables"

python -m scripts.tables_and_figures.plot_mechanism_figures \
  --data-dir "$WORKSPACE_ROOT/mechanism_figure_inputs" \
  --output-dir "$WORKSPACE_ROOT/results/mechanism_figures"
```

Generic scheduler examples are available in `jobs/examples/`.

## Analysis-to-code map

| Component | Main entry points |
|---|---|
| Checkpoint evaluation | `scripts/evaluation/run_fullzoo_lowrank_full_eval_one.py` |
| lm-eval parsing and alignment | `scripts/parsing/build_fullzoo_lowrank_pair_metrics_tables.py`, `scripts/parsing/parse_lmeval_uncertainty_pair_general.py` |
| Protocol reproduction | `scripts/parsing/reproduce_lmeval_metrics_from_samples.py`, `scripts/parsing/read_official_protocol.py` |
| PredFlip, HiddenChurn, decomposition, and JS drift | `scripts/metrics/churn_metrics.py`, `scripts/tables_and_figures/analyze_fullzoo_decision_churn_v2.py` |
| Margin, entropy, BAR, and boundary buckets | `scripts/tables_and_figures/analyze_fullzoo_decision_boundary_v3.py`, `scripts/boundary_diagnostics/` |
| Accuracy-matched analyses | `scripts/accuracy_matched/build_accuracy_matched_v1.py` |
| Bootstrap and sample-selection robustness | `scripts/accuracy_matched/stage19_bootstrap_sample_selection_20260719_204124.py`, `scripts/sample_selection/` |
| Prompt representation | `scripts/mechanism/mech_activation_v2_trusted.py`, `scripts/mechanism/analyze_mech_activation_v2_phase1.py` |
| Answer-path representation | `scripts/mechanism/mech_choice_v1.py`, `scripts/mechanism/analyze_mech_choice_v1.py` |
| Effective-weight residuals | `scripts/effective_weight_residual/` |
| Score correction and repair feasibility | `scripts/score_correction/` |
| High-retention analyses and budget audits | `scripts/high_retention/` |
| Tables and figures | `scripts/tables_and_figures/` |

## Testing

Run the synthetic test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests use small synthetic arrays and do not contain paper results or research observations.

## Scope

This repository supports source inspection and rerunning the pipeline with independently obtained assets. Because experimental inputs and outputs are excluded, cloning the repository alone does not regenerate the paper's numerical results.

Third-party libraries, models, checkpoints, and datasets remain subject to their respective licenses and terms.

## License

The repository code is released under the MIT License. See `LICENSE`.
