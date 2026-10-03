import argparse
import json
import shutil
import subprocess

from huggingface_hub import HfApi, hf_hub_download

PARSER = argparse.ArgumentParser(
    description="Inspect protocol documentation and selected result files in a checkpoint repository."
)
PARSER.add_argument(
    "--repo_id",
    required=True,
    help="Hugging Face model repository to inspect.",
)
ARGS = PARSER.parse_args()
repo_id = ARGS.repo_id
api = HfApi()

print("=" * 100)
print("Official docs and scripts in checkpoint repo")

files = api.list_repo_files(repo_id, repo_type="model")
doc_files = [
    f for f in files
    if f in ["README.md", "MANIFEST.csv", "eval_artifacts/README.md", "full_checkpoint_zoo/MANIFEST_ONLY_FOR_NOW.md"]
    or f.startswith("scripts/")
]
for f in doc_files:
    print(f)

print("\n" + "=" * 100)
print("Preview official docs")

for f in doc_files:
    if f.endswith((".md", ".txt", ".py", ".sh", ".csv")):
        print("\n" + "=" * 100)
        print("FILE:", f)
        try:
            local = hf_hub_download(repo_id=repo_id, repo_type="model", filename=f)
            text = open(local, errors="ignore").read()
            print(text[:8000])
        except Exception as e:
            print("READ_FAIL:", type(e).__name__, e)

print("\n" + "=" * 100)
print("Check local evaluator commands")

for cmd in ["lm-eval", "lm_eval"]:
    path = shutil.which(cmd)
    print(cmd, "->", path)
    if path:
        try:
            out = subprocess.run([cmd, "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
            print(out.stdout[:2000])
        except Exception as e:
            print("HELP_FAIL:", type(e).__name__, e)

print("\nPython lm_eval import:")
try:
    import lm_eval
    print("lm_eval import OK")
    print("lm_eval version:", getattr(lm_eval, "__version__", "NA"))
except Exception as e:
    print("lm_eval import FAIL:", type(e).__name__, e)

print("\n" + "=" * 100)
print("Parse official MCQ metrics")

target_jsons = [
    "eval_artifacts/results/llama31_8b/mcq/llama31-8b-dense.json",
    "eval_artifacts/results/llama31_8b/mcq/llama31-8b-asvd-0.8.json",
    "eval_artifacts/results/llama31_8b/mcq/llama31-8b-asvd-0.6.json",
    "eval_artifacts/results/llama31_8b/mcq/llama31-8b-basis-sharing-0.8.json",
    "eval_artifacts/results/llama31_8b/mcq/llama31-8b-dobi-0.8.json",
    "eval_artifacts/results/llama31_8b/mcq/llama31-8b-modegpt-0.8.json",
    "eval_artifacts/results/llama31_8b/mcq/llama31-8b-svdllm-v2-hetero-0.8.json",
]

def walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, path + "/" + str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, path + "/" + str(i))
    else:
        yield path, obj

for f in target_jsons:
    print("\n" + "=" * 100)
    print("RESULT:", f)
    try:
        local = hf_hub_download(repo_id=repo_id, repo_type="model", filename=f)
        obj = json.load(open(local))
        print("command:", obj.get("artifacts", {}).get("command"))
        print("backend:", obj.get("backend"))
        print("config:", json.dumps(obj.get("config", {}), indent=2)[:3000])

        # Print paths relevant to task metrics.
        wanted_tasks = ["arc_challenge", "arc_easy", "boolq", "winogrande", "piqa", "hellaswag", "openbookqa"]
        wanted_metric_words = ["acc_norm", "acc", "macro", "mean"]
        hits = []
        for p, v in walk(obj):
            ps = p.lower()
            if any(t in ps for t in wanted_tasks) and any(m in ps for m in wanted_metric_words):
                if isinstance(v, (int, float, str, bool)) or v is None:
                    hits.append((p, v))
        for p, v in hits[:120]:
            print(p, "=", v)

    except Exception as e:
        print("PARSE_FAIL:", type(e).__name__, e)
