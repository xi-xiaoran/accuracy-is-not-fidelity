import os
import inspect
import glob
import lm_eval

root = os.path.dirname(inspect.getfile(lm_eval))
print("lm_eval root:", root)

print("\n===== candidate ARC files =====")
patterns = [
    os.path.join(root, "tasks", "**", "*arc*"),
    os.path.join(root, "tasks", "**", "*ai2*"),
    os.path.join(root, "tasks", "**", "*multiple*"),
]
seen = []
for pat in patterns:
    for p in glob.glob(pat, recursive=True):
        if p not in seen:
            seen.append(p)
            print(p)

print("\n===== grep acc_norm / process_results / doc_to_choice around ARC files =====")
for p in seen:
    if os.path.isfile(p) and p.endswith((".py", ".yaml", ".yml")):
        try:
            txt = open(p, errors="ignore").read()
        except Exception:
            continue
        low = txt.lower()
        if "arc_challenge" in low or "acc_norm" in low or "process_results" in low or "doc_to_choice" in low:
            print("\n" + "=" * 100)
            print("FILE:", p)
            lines = txt.splitlines()
            for i, line in enumerate(lines, start=1):
                if any(k in line.lower() for k in ["arc_challenge", "acc_norm", "process_results", "doc_to_choice", "doc_to_target", "doc_to_text", "metric_list", "choice", "choices"]):
                    lo = max(1, i - 4)
                    hi = min(len(lines), i + 8)
                    print("-" * 80)
                    print(f"around line {i}")
                    for j in range(lo, hi + 1):
                        print(f"{j:04d}: {lines[j-1]}")

print("\n===== task manager listing for arc_challenge =====")
try:
    from lm_eval.tasks import TaskManager
    tm = TaskManager()
    print(tm.match_tasks(["arc_challenge"]))
except Exception as e:
    print("TaskManager fail:", type(e).__name__, e)
