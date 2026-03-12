"""
scripts/06_kfold_runner.py
==========================
Batch runner for k-fold cross-validation.

Trains each model config across all folds sequentially. Outputs are organized as:
    <base_output_dir>/<run_label>_folds/fold_0/
    <base_output_dir>/<run_label>_folds/fold_1/
    ...
    <base_output_dir>/<run_label>_folds/fold_4/

Usage:
    python scripts/06_kfold_runner.py --data_dir data/competition

    # Quick test (1 fold, minimal epochs)
    python scripts/06_kfold_runner.py --data_dir data/competition --quick --folds 0
"""

import subprocess
import sys
import time
from pathlib import Path

N_FOLDS = 5

# ═══════════════════════════════════════════════════════════════════════


def run_fold(data_dir: str, job: dict, fold: int, quick: bool = False):
    """Run a single training fold as a subprocess."""

    label = job["label"]
    output_dir = Path("models") / f"{label}_folds" / f"fold_{fold}"

    cmd = [
        sys.executable, "scripts/02_train.py",
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
    parser = argparse.ArgumentParser(description="K-fold CV batch runner")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--job_config", type=str, default='configs/train_jobs.json',
                        help="Run jobs configuration")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode for testing")
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="Specific folds to run (default: all)")
    parser.add_argument("--jobs", type=str, nargs="+", default=None,
                        help="Specific job labels to run (default: all)")
    args = parser.parse_args()

    folds = args.folds if args.folds is not None else list(range(N_FOLDS))

    with open(args.job_config, 'r') as json_jobs:
        jobs = json.load(json_jobs)

    if args.jobs:
        jobs = [j for j in job if j["label"] in args.jobs]
        if not jobs:
            print(f"No matching jobs. Available: {[j['label'] for j in jobs]}")
            return

    total_runs = len(jobs) * len(folds)
    print(f"\n{'═'*70}")
    print(f"  K-FOLD BATCH RUNNER")
    print(f"  Models: {[j['label'] for j in jobs]}")
    print(f"  Folds:  {folds}")
    print(f"  Total:  {total_runs} training runs")
    print(f"{'═'*70}")

    results = []
    t_total = time.time()

    for job in jobs:
        for fold in folds:
            success = run_fold(args.data_dir, job, fold, args.quick)
            results.append({
                "label": job["label"],
                "fold": fold,
                "success": success,
            })

    # ── Summary ───────────────────────────────────────────────────────
    total_time = time.time() - t_total
    passed = sum(1 for r in results if r["success"])
    failed = [r for r in results if not r["success"]]

    print(f"\n\n{'═'*70}")
    print(f"  BATCH COMPLETE")
    print(f"  {passed}/{total_runs} succeeded  |  {total_time/60:.0f} min total")
    if failed:
        print(f"\n  FAILED RUNS:")
        for r in failed:
            print(f"    {r['label']} fold {r['fold']}")
    print(f"{'═'*70}")

    # Print output structure
    print(f"\n  Output structure:")
    for job in jobs:
        label = job["label"]
        fold_dir = Path("models") / f"{label}_folds"
        print(f"    {fold_dir}/")
        for fold in folds:
            d = fold_dir / f"fold_{fold}"
            exists = "✓" if d.exists() else "✗"
            print(f"      fold_{fold}/  {exists}")


if __name__ == "__main__":
    main()
