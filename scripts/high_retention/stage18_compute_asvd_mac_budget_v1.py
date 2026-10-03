from pathlib import Path
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM
import pandas as pd
import torch
import json
import os

ROOT = Path(".").resolve()
FINAL_OUT = (
    ROOT / "results/stage18_official_asvd_hifi_v1/mac_budget_v1"
)
TMP_OUT = FINAL_OUT.with_name(
    f".mac_budget_v1.tmp.{os.getpid()}"
)

assert not FINAL_OUT.exists()
assert not TMP_OUT.exists()
TMP_OUT.mkdir(parents=True)

dense = Path(snapshot_download(
    "meta-llama/Llama-2-7b-hf",
    local_files_only=True,
))

models = {
    "dense": dense,
    "asvd85": (
        ROOT / "outputs/stage18_official_asvd_hifi_v1"
        / "02_asvd85_compat_model"
    ),
    "asvd90": (
        ROOT / "outputs/stage18_official_asvd_hifi_v1"
        / "03_asvd90_compat_model"
    ),
    "asvd95": (
        ROOT / "outputs/stage18_official_asvd_hifi_v1"
        / "01_asvd95_compat_model"
    ),
}

sequence_lengths = [512, 2048, 4096]
rows = []

for label, path in models.items():
    config = AutoConfig.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=True,
    )

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
        )

    counts = {
        "embedding_numel": 0,
        "attention_linear_macs_per_token": 0,
        "mlp_linear_macs_per_token": 0,
        "lm_head_macs_per_token": 0,
        "other_linear_macs_per_token": 0,
    }
    factorized_parents = set()

    for module_name, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters(
            recurse=False
        ):
            if parameter.ndim != 2:
                continue

            numel = int(parameter.numel())

            if isinstance(module, torch.nn.Embedding):
                counts["embedding_numel"] += numel
                continue

            if not isinstance(module, torch.nn.Linear):
                raise RuntimeError(
                    f"Unexpected 2D module: {module_name} "
                    f"{module.__class__.__name__}"
                )

            if module_name == "lm_head":
                key = "lm_head_macs_per_token"
            elif ".self_attn." in module_name:
                key = "attention_linear_macs_per_token"
            elif ".mlp." in module_name:
                key = "mlp_linear_macs_per_token"
            else:
                key = "other_linear_macs_per_token"

            counts[key] += numel

            if module_name.endswith((".ALinear", ".BLinear")):
                factorized_parents.add(
                    module_name.rsplit(".", 1)[0]
                )

    block_linear = (
        counts["attention_linear_macs_per_token"]
        + counts["mlp_linear_macs_per_token"]
        + counts["other_linear_macs_per_token"]
    )
    total_linear = (
        block_linear
        + counts["lm_head_macs_per_token"]
    )

    row = {
        "model": label,
        "total_parameters": int(sum(
            parameter.numel()
            for parameter in model.parameters()
        )),
        "factorized_projection_count": len(
            factorized_parents
        ),
        **counts,
        "block_linear_macs_per_token": block_linear,
        "total_linear_macs_per_token": total_linear,
        "num_hidden_layers": int(
            config.num_hidden_layers
        ),
        "hidden_size": int(config.hidden_size),
    }

    # Standard dense-matmul proxy:
    # linear projections/logits plus QK^T and AV.
    # Excludes embedding lookup, norms, activations,
    # softmax, and memory/system overhead.
    for length in sequence_lengths:
        attention_pairwise = (
            2
            * config.num_hidden_layers
            * config.hidden_size
            * length
            * length
        )
        row[f"standard_mac_proxy_s{length}"] = (
            total_linear * length
            + attention_pairwise
        )

    rows.append(row)

df = pd.DataFrame(rows)
dense_row = df[df["model"] == "dense"].iloc[0]

for column in [
    "total_parameters",
    "block_linear_macs_per_token",
    "total_linear_macs_per_token",
    *[
        f"standard_mac_proxy_s{length}"
        for length in sequence_lengths
    ],
]:
    df[f"{column}_keep_ratio"] = (
        df[column] / float(dense_row[column])
    )
    df[f"{column}_reduction"] = (
        1.0 - df[f"{column}_keep_ratio"]
    )

assert int(dense_row["factorized_projection_count"]) == 0
assert int(dense_row["total_linear_macs_per_token"]) == 6607077376
assert int(dense_row["lm_head_macs_per_token"]) == 131072000

expected_factorized = {
    "asvd85": 120,
    "asvd90": 90,
    "asvd95": 36,
}
for model, expected in expected_factorized.items():
    actual = int(
        df.loc[
            df["model"] == model,
            "factorized_projection_count",
        ].iloc[0]
    )
    assert actual == expected, (model, actual, expected)

df.to_csv(
    TMP_OUT / "stage18_asvd_mac_budget.csv",
    index=False,
)

manifest = {
    "definition": (
        "One MAC per weight element for each executed "
        "Linear operation. FLOP ratios are identical when "
        "one MAC is counted as two FLOPs."
    ),
    "standard_mac_proxy": (
        "total_linear_macs_per_token*S + "
        "2*num_layers*hidden_size*S^2"
    ),
    "excluded": [
        "embedding lookup",
        "normalization",
        "activation functions",
        "softmax",
        "memory traffic",
        "kernel and system overhead",
    ],
    "sequence_lengths": sequence_lengths,
    "status": "PASS",
}

(TMP_OUT / "mac_budget_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)

TMP_OUT.rename(FINAL_OUT)

show_columns = [
    "model",
    "factorized_projection_count",
    "total_parameters_keep_ratio",
    "block_linear_macs_per_token_keep_ratio",
    "block_linear_macs_per_token_reduction",
    "total_linear_macs_per_token_keep_ratio",
    "total_linear_macs_per_token_reduction",
    "standard_mac_proxy_s512_reduction",
    "standard_mac_proxy_s2048_reduction",
    "standard_mac_proxy_s4096_reduction",
]

print("===== ASVD COMPUTE BUDGET =====")
print(
    df[show_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print()
print("WROTE:", FINAL_OUT)
print("EXACT_LINEAR_MAC_ACCOUNTING_PASS")
print("STANDARD_ATTENTION_MAC_PROXY_PASS")
print("NO_WEIGHT_TENSORS_LOADED")
print("NO_GPU_USED")
print("STAGE18_STEP31_MAC_BUDGET_COMPLETE")
