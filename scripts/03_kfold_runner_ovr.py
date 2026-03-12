"""
scripts/03_kfold_runner_ovr.py
==============================
Batch runner for OVR (one-vs-rest) k-fold cross-validation.

Identical to 03_kfold_runner.py but calls 02_train_ovr.py instead.

Usage:
    python scripts/03_kfold_runner_ovr.py \
        --data_dir data/competition \
        --job_config configs/train_jobs_ovr.json

    # Quick test
    python scripts/03_kfold_runner_ovr.py \
        --data_dir data/competition \
        --job_config configs/train_jobs_ovr.json \
        --quick --folds 0
"""

import subprocess
import sys
import time
from pathlib import Path

N_FOLDS = 5
TRAIN_SCRIPT = "scripts/02_train_ovr.py"


def run_fold(data_dir: str, job: dict, fold: int, quick: bool = False):
    label = job["label"]
    output_dir = Path("models") / f"{label}_folds" / f"fold_{fold}"

    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--model_config", job["config"],
        "--data_dir", data_dir,
        "--crop_dir", job["crop_dir"],
        "--val_fold", str(fold),
        "--output_dir", str(output_dir),
        "--run_name", f"{label}_fold{fold}",
    ]
    if quick:
        cmd.append("--quick")

    print(f"\n{'═'*70}")
    print(f"  {label}  |  fold {fold}/{N_FOLDS-1}  |  output: {output_dir}")
    print(f"{'═'*70}\n")

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n  → {label} fold {fold}: {status} ({elapsed/60:.1f} min)")
    return result.returncode == 0


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description="OVR K-fold CV batch runner")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--job_config", type=str,
                        default="configs/train_jobs_ovr.json")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--jobs", type=str, nargs="+", default=None)
    args = parser.parse_args()

    folds = args.folds if args.folds is not None else list(range(N_FOLDS))

    with open(args.job_config, 'r') as f:
        jobs = json.load(f)

    if args.jobs:
        jobs = [j for j in jobs if j["label"] in args.jobs]
        if not jobs:
            print(f"No matching jobs found.")
            return

    total_runs = len(jobs) * len(folds)
    print(f"\n{'═'*70}")
    print(f"  OVR K-FOLD BATCH RUNNER")
    print(f"  Script: {TRAIN_SCRIPT}")
    print(f"  Models: {[j['label'] for j in jobs]}")
    print(f"  Folds:  {folds}")
    print(f"  Total:  {total_runs} training runs")
    print(f"{'═'*70}")

    results = []
    t_total = time.time()

    for job in jobs:
        for fold in folds:
            success = run_fold(args.data_dir, job, fold, args.quick)
            results.append({"label": job["label"], "fold": fold,
                            "success": success})

    total_time = time.time() - t_total
    passed = sum(1 for r in results if r["success"])
    failed = [r for r in results if not r["success"]]

    print(f"\n\n{'═'*70}")
    print(f"  BATCH COMPLETE")
    print(f"  {passed}/{total_runs} succeeded  |  {total_time/60:.0f} min")
    if failed:
        print(f"\n  FAILED RUNS:")
        for r in failed:
            print(f"    {r['label']} fold {r['fold']}")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
