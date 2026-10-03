from pathlib import Path
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM
import csv
import json
import os

ROOT = Path(".").resolve()

FINAL_OUT = (
    ROOT
    / "results/stage18_official_asvd_hifi_v1"
    / "budget_audit_v1"
)
TMP_OUT = FINAL_OUT.with_name(
    f".budget_audit_v1.tmp.{os.getpid()}"
)

models = [
    {
        "name": "asvd85",
        "keep_ratio_label": 0.85,
        "path": ROOT / "outputs/stage18_official_asvd_hifi_v1"
        / "02_asvd85_compat_model",
    },
    {
        "name": "asvd90",
        "keep_ratio_label": 0.90,
        "path": ROOT / "outputs/stage18_official_asvd_hifi_v1"
        / "03_asvd90_compat_model",
    },
    {
        "name": "asvd95",
        "keep_ratio_label": 0.95,
        "path": ROOT / "outputs/stage18_official_asvd_hifi_v1"
        / "01_asvd95_compat_model",
    },
]

assert not FINAL_OUT.exists()
assert not TMP_OUT.exists()
TMP_OUT.mkdir(parents=True)

dense_snapshot = Path(
    snapshot_download(
        repo_id="meta-llama/Llama-2-7b-hf",
        local_files_only=True,
    )
)

def build_meta_model(path, trust_remote_code):
    config = AutoConfig.from_pretrained(
        path,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=trust_remote_code,
        )

    n_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert all(
        parameter.is_meta
        for parameter in model.parameters()
    )

    return config, model, n_params

def weight_files(path):
    files = sorted(path.glob("*.safetensors"))

    if not files:
        files = sorted(path.glob("pytorch_model*.bin"))

    assert files, f"No weight files in {path}"
    return files

dense_config, dense_model, dense_params = build_meta_model(
    dense_snapshot,
    trust_remote_code=False,
)

dense_weights = weight_files(dense_snapshot)
dense_weight_bytes = sum(
    path.stat().st_size
    for path in dense_weights
)

rows = []

for item in models:
    config, model, n_params = build_meta_model(
        item["path"],
        trust_remote_code=True,
    )

    weights = weight_files(item["path"])
    weight_bytes = sum(
        path.stat().st_size
        for path in weights
    )

    parameter_ratio = n_params / dense_params
    storage_ratio = weight_bytes / dense_weight_bytes

    row = {
        "model_name": item["name"],
        "keep_ratio_label": item["keep_ratio_label"],
        "dense_parameters": dense_params,
        "compressed_parameters": n_params,
        "actual_parameter_ratio": parameter_ratio,
        "parameter_reduction_ratio": 1.0 - parameter_ratio,
        "dense_weight_bytes": dense_weight_bytes,
        "compressed_weight_bytes": weight_bytes,
        "actual_storage_ratio": storage_ratio,
        "storage_reduction_ratio": 1.0 - storage_ratio,
        "parameter_label_abs_error": abs(
            parameter_ratio - item["keep_ratio_label"]
        ),
        "storage_label_abs_error": abs(
            storage_ratio - item["keep_ratio_label"]
        ),
        "model_class": type(model).__name__,
        "n_weight_files": len(weights),
    }

    rows.append(row)

assert dense_params > 6_000_000_000
assert len(rows) == 3

assert all(
    row["parameter_label_abs_error"] < 0.01
    for row in rows
)

assert all(
    row["storage_label_abs_error"] < 0.01
    for row in rows
)

assert [
    row["actual_parameter_ratio"]
    for row in rows
] == sorted(
    row["actual_parameter_ratio"]
    for row in rows
)

csv_path = TMP_OUT / "stage18_asvd_actual_budget.csv"

with csv_path.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(rows[0]),
    )
    writer.writeheader()
    writer.writerows(rows)

manifest = {
    "dense_repo": "meta-llama/Llama-2-7b-hf",
    "dense_snapshot": str(dense_snapshot),
    "dense_model_class": type(dense_model).__name__,
    "dense_parameters": dense_params,
    "dense_weight_files": [
        path.name for path in dense_weights
    ],
    "dense_weight_bytes": dense_weight_bytes,
    "validation": "PASS",
}

(TMP_OUT / "budget_audit_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)

TMP_OUT.rename(FINAL_OUT)

print("===== OFFICIAL ASVD ACTUAL BUDGET =====")
print(
    "DENSE_PARAMETERS:",
    dense_params,
)
print(
    "DENSE_WEIGHT_GB:",
    round(dense_weight_bytes / 1e9, 6),
)

for row in rows:
    print()
    print("MODEL:", row["model_name"])
    print(
        "LABEL_KEEP:",
        f"{row['keep_ratio_label']:.4f}",
    )
    print(
        "ACTUAL_PARAMETER_KEEP:",
        f"{row['actual_parameter_ratio']:.6f}",
    )
    print(
        "PARAMETER_REDUCTION:",
        f"{row['parameter_reduction_ratio']:.6f}",
    )
    print(
        "ACTUAL_STORAGE_KEEP:",
        f"{row['actual_storage_ratio']:.6f}",
    )
    print(
        "STORAGE_REDUCTION:",
        f"{row['storage_reduction_ratio']:.6f}",
    )
    print(
        "COMPRESSED_PARAMETERS:",
        row["compressed_parameters"],
    )
    print(
        "WEIGHT_GB:",
        round(
            row["compressed_weight_bytes"] / 1e9,
            6,
        ),
    )

print()
print("NO_WEIGHT_TENSORS_LOADED")
print("NO_GPU_USED")
print("ACTUAL_PARAMETER_LABEL_ALIGNMENT_PASS")
print("ACTUAL_STORAGE_LABEL_ALIGNMENT_PASS")
print("STAGE18_STEP19_BUDGET_AUDIT_COMPLETE")
