"""
src/mlflow_utils.py
===================
Optional MLflow integration. Gracefully no-ops if mlflow is not installed.
"""

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def setup_run(experiment_name: str, run_name: str | None = None,
              params: dict | None = None):
    """Start an MLflow run if available. Returns whether MLflow is active."""
    if not HAS_MLFLOW:
        return False

    import time
    mlflow.set_experiment(experiment_name)
    run_name = run_name or f"run_{time.strftime('%m%d_%H%M')}"
    mlflow.start_run(run_name=run_name)

    if params:
        flat = {}
        for k, v in params.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat[f"{k}.{k2}"] = v2
            elif isinstance(v, (tuple, list)):
                flat[k] = str(v)
            else:
                flat[k] = v
        mlflow.log_params(flat)

    print(f"MLflow run: {run_name}")
    return True


def log_metrics(metrics: dict, step: int | None = None):
    if HAS_MLFLOW:
        mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str):
    if HAS_MLFLOW:
        mlflow.log_artifact(str(path))


def end_run():
    if HAS_MLFLOW:
        mlflow.end_run()
