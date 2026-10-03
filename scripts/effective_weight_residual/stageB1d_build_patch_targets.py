from pathlib import Path
import re
import pandas as pd

ROOT = Path(".").resolve()
B1C = ROOT / "results/paper_packages/stageB1c_effective_weight_residual_fullzoo_v1_20260702"
PKG = ROOT / "results/paper_packages/stageB1d_effective_weight_patch_dobi_basis_v1_20260702"
DATA = PKG / "data"
DATA.mkdir(parents=True, exist_ok=True)

targets = pd.read_csv(B1C / "data/stageB1c_targets.csv")
base_map = pd.read_csv(B1C / "data/base_snapshot_mapping.csv")
base_dict = dict(zip(base_map["model_key"], base_map["base_snapshot_path"]))

def keep_from_target_id(tid):
    m = re.search(r"keep0p\d+", str(tid))
    if not m:
        raise RuntimeError(f"Cannot parse keep from {tid}")
    return m.group(0)

patch_rows = []
for _, r in targets.iterrows():
    tid = str(r["target_id"])
    model_key = str(r["model_key"])
    method_key = str(r["method_key"])

    patch_reason = None
    comp_path = Path(str(r["compressed_path"]))

    if method_key == "dobisvd":
        patch_reason = "dobisvd_BLinear_at_ALinear"
    elif method_key == "basis_sharing" and model_key in {"llama2_7b", "llama_7b"}:
        keep = keep_from_target_id(tid)
        comp_path = ROOT / f"outputs/reconstructed_checkpoints/{model_key}_basis_sharing_{keep}_standard_hf"
        patch_reason = "basis_reconstructed_standard_hf"

    if patch_reason is None:
        continue

    patch_rows.append({
        "patch_index": len(patch_rows),
        "target_id": tid,
        "model_key": model_key,
        "method_key": method_key,
        "keep_ratio": r.get("keep_ratio", ""),
        "base_snapshot_path": base_dict[model_key],
        "compressed_path": str(comp_path),
        "compressed_exists": comp_path.exists(),
        "patch_reason": patch_reason,
    })

df = pd.DataFrame(patch_rows)
df.to_csv(DATA / "stageB1d_patch_targets.csv", index=False)

print("patch targets", len(df))
print(df.groupby(["method_key", "patch_reason"])["compressed_exists"].agg(["count", "sum", "mean"]).to_string())
print(df[df["compressed_exists"] == False].to_string(index=False))
