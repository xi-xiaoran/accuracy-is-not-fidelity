import sys
import re
import time
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from scripts.parsing.build_fullzoo_lowrank_pair_metrics_tables import COMP_CSV

ROOT = Path(".").resolve()
OUT = ROOT / "results/theory_predictions_v1"
FULL = ROOT / "results/theory_predictions_v1_fullzoo"
LOGD = ROOT / "logs/theory_predictions_v1_fullzoo"

FULL.mkdir(parents=True, exist_ok=True)
LOGD.mkdir(parents=True, exist_ok=True)


def safe_name(x):
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(x))


def expected_files(target_id):
    p = OUT
    return [
        p / f"{target_id}_sample_theory_frame.csv",
        p / f"{target_id}_boundary_ratio_buckets.csv",
        p / f"{target_id}_boundary_ratio_by_task.csv",
        p / f"{target_id}_correction_direction_rank.csv",
        p / f"{target_id}_projection_eval.csv",
        p / f"{target_id}_projection_energy.csv",
    ]


def run_one(target_id, cal_per_task, eval_per_task, force=False):
    start = time.time()
    log_path = LOGD / f"{safe_name(target_id)}.log"

    if (not force) and all(p.exists() and p.stat().st_size > 0 for p in expected_files(target_id)):
        return {
            "target_id": target_id,
            "status": "skipped_exists",
            "returncode": 0,
            "seconds": 0.0,
            "log": str(log_path),
        }

    cmd = [
        sys.executable,
        "-m", "scripts.metrics.analyze_theory_predictions_v1",
        "--target_id", target_id,
        "--cal_per_task", str(cal_per_task),
        "--eval_per_task", str(eval_per_task),
    ]

    with open(log_path, "w") as f:
        f.write("CMD: " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    ok_files = all(p.exists() and p.stat().st_size > 0 for p in expected_files(target_id))
    status = "ok" if proc.returncode == 0 and ok_files else "failed"

    return {
        "target_id": target_id,
        "status": status,
        "returncode": proc.returncode,
        "seconds": time.time() - start,
        "log": str(log_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal_per_task", type=int, default=128)
    ap.add_argument("--eval_per_task", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--target_file", default=None)
    args = ap.parse_args()

    if args.target_file:
        ids = [
            x.strip()
            for x in Path(args.target_file).read_text().splitlines()
            if x.strip() and not x.strip().startswith("#")
        ]
    else:
        comp = pd.read_csv(COMP_CSV)
        ids = comp["target_id"].dropna().astype(str).drop_duplicates().tolist()

    if args.limit is not None:
        ids = ids[:args.limit]

    target_list = FULL / "target_ids_fullzoo.txt"
    target_list.write_text("\n".join(ids) + "\n")

    print("N_TARGETS", len(ids), flush=True)
    print("TARGET_LIST", target_list, flush=True)
    print("WORKERS", args.workers, flush=True)
    print("CAL_PER_TASK", args.cal_per_task, flush=True)
    print("EVAL_PER_TASK", args.eval_per_task, flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(run_one, tid, args.cal_per_task, args.eval_per_task, args.force): tid
            for tid in ids
        }

        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            print(
                f"[{i}/{len(ids)}] {row['status']} "
                f"{row['target_id']} "
                f"return={row['returncode']} "
                f"sec={row['seconds']:.1f} "
                f"log={row['log']}",
                flush=True,
            )

            pd.DataFrame(rows).to_csv(FULL / "status_fullzoo_running.csv", index=False)

    df = pd.DataFrame(rows)
    status_csv = FULL / "status_fullzoo.csv"
    df.to_csv(status_csv, index=False)

    print()
    print("===== STATUS SUMMARY =====")
    print(df["status"].value_counts(dropna=False).to_string())
    print("WROTE", status_csv)

    failed = df[df["status"] != "ok"]
    if len(failed):
        failed_csv = FULL / "failed_fullzoo.csv"
        failed.to_csv(failed_csv, index=False)
        print("FAILED", len(failed))
        print("WROTE", failed_csv)
    else:
        print("ALL_OK")


if __name__ == "__main__":
    main()
