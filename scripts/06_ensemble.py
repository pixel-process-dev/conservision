"""
scripts/07_ensemble.py
======================
Test and persist ensemble approaches on OOF validation predictions.

All methods that involve fitting use nested CV on the site-aware fold
structure to produce honest (unbiased) log loss estimates. This means
fit on folds {0..4}\\{k}, evaluate on fold k, for each k.

Methods that cannot be properly nested (greedy_fold, meta_fold) are
excluded because the site-aware fold structure makes within-fold
evaluation systematically optimistic — inner train/val share the same
sites, but the hidden test set has entirely unseen sites.

Methods implemented:
    1. simple_average      — equal-weight probability average (no fitting)
    2. model_weights       — optimized per-model scalar weights (nested CV)
    3. class_model_weights — optimized per-model per-class weights (nested CV)
    4. temperature         — per-model temperature scaling + average (nested CV)
    5. greedy_model        — forward model selection (nested CV)
    6. meta_model          — logistic regression stacking (nested CV)

Each method saves per-image predictions to output_dir/predictions/<method>.csv
for post-hoc comparison.

Usage:
    python scripts/07_ensemble.py --config configs/ensemble.json

Example ensemble.json:
    {
        "models_dir": "models/",
        "models": [
            "swinv2_.1_folds",
            "dinov2_.1_folds",
            "convnext_.1_folds",
            "eva02_1_folds"
        ],
        "pred_file": "oof_predictions.csv",
        "output_dir": "results/ensemble_001",
        "methods": {
            "simple_average": true,
            "model_weights": true,
            "class_model_weights": true,
            "temperature": {
                "grid": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
            },
            "greedy_model": true,
            "meta_model": { "C": [0.01, 0.1, 1.0, 10.0] }
        }
    }
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, f1_score, accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_classes, cls2idx as make_cls2idx

CLASSES = get_classes()
NC = len(CLASSES)
CLS2IDX = make_cls2idx(CLASSES)
PROB_COLS = [f"{c}_prob" for c in CLASSES]
LOGIT_COLS = [f"{c}_logit" for c in CLASSES]


# ═════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═════════════════════════════════════════════════════════════════════════════

def clip_and_norm(p):
    """Clip probabilities and renormalize rows."""
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return p / p.sum(axis=1, keepdims=True)


def score(y_true, probas):
    """Compute log loss, F1 macro, accuracy from integer labels + probas."""
    p = clip_and_norm(probas)
    pred = p.argmax(axis=1)
    return {
        "log_loss": float(log_loss(y_true, p, labels=list(range(NC)))),
        "f1_macro": float(f1_score(y_true, pred, average="macro",
                                   zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }


def softmax(logits, temperature=1.0):
    """Temperature-scaled softmax on (N, C) array."""
    x = logits / temperature
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _expand_proba(raw_proba, seen_classes, n_classes):
    """Expand predict_proba to full class set (handles missing classes)."""
    if len(seen_classes) == n_classes:
        return raw_proba
    n_samples = raw_proba.shape[0]
    full = np.full((n_samples, n_classes), 1e-7)
    for i, c in enumerate(seen_classes):
        full[:, c] = raw_proba[:, i]
    return full / full.sum(axis=1, keepdims=True)


def _build_pred_df(ids, labels, folds, probas, method_name):
    """Build a per-image prediction DataFrame for saving."""
    df = pd.DataFrame({"id": ids, "fold": folds})
    df["true_label"] = [CLASSES[i] for i in labels]
    pred_idx = probas.argmax(axis=1)
    df["pred_label"] = [CLASSES[i] for i in pred_idx]
    for i, c in enumerate(CLASSES):
        df[f"{c}_prob"] = probas[:, i]
    df["method"] = method_name
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Data loading + alignment
# ═════════════════════════════════════════════════════════════════════════════

def load_oof(models_dir, model_names, pred_file="oof_predictions.csv"):
    """Load and align OOF predictions from multiple models.

    Returns:
        labels, probas, logits, folds, ids, sites
    """
    models_dir = Path(models_dir)
    frames = {}
    for name in model_names:
        path = models_dir / name / pred_file
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        frames[name] = pd.read_csv(path)

    ref_name = model_names[0]
    ref = frames[ref_name].sort_values("id").reset_index(drop=True)
    ref_ids = set(ref["id"])

    for name, df in frames.items():
        model_ids = set(df["id"])
        if model_ids != ref_ids:
            missing = ref_ids - model_ids
            extra = model_ids - ref_ids
            raise ValueError(
                f"{name} id mismatch: {len(missing)} missing, "
                f"{len(extra)} extra vs {ref_name}")

    ids = ref["id"].values
    labels = ref["true_label"].map(CLS2IDX).values
    folds = ref["fold"].values
    sites = ref["site"].values if "site" in ref.columns else None

    probas = {}
    logits = {}
    for name, df in frames.items():
        df = df.set_index("id").loc[ids].reset_index()
        probas[name] = df[PROB_COLS].values
        logits[name] = df[LOGIT_COLS].values

    N = len(ids)
    n_folds = len(np.unique(folds))
    print(f"Loaded {len(model_names)} models, {N} images, {n_folds} folds")
    if sites is not None:
        n_sites = len(np.unique(sites))
        print(f"Sites: {n_sites} total")
        for k in sorted(np.unique(folds)):
            fold_sites = np.unique(sites[folds == k])
            print(f"  fold {k}: {len(fold_sites)} sites, "
                  f"{(folds == k).sum()} images")

    print()
    for name in model_names:
        s = score(labels, probas[name])
        print(f"  {name:<35} ll={s['log_loss']:.4f}  "
              f"f1={s['f1_macro']:.4f}  acc={s['accuracy']:.4f}")

    return labels, probas, logits, folds, ids, sites


# ═════════════════════════════════════════════════════════════════════════════
# Method 1: Simple average
# ═════════════════════════════════════════════════════════════════════════════

def run_simple_average(labels, probas, model_names, ids, folds):
    """Equal-weight average of all model probabilities. No fitting."""
    avg = np.mean([probas[m] for m in model_names], axis=0)
    s = score(labels, avg)
    result = {
        "method": "simple_average",
        "variant": "all_models",
        "eval": "exact (no fitting)",
        "n_models": len(model_names),
        "models_used": ",".join(model_names),
        "params": "",
        **s,
    }
    pred_df = _build_pred_df(ids, labels, folds, avg, "simple_average")
    return [result], pred_df


# ═════════════════════════════════════════════════════════════════════════════
# Method 2: Optimized model weights (nested CV)
# ═════════════════════════════════════════════════════════════════════════════

def run_model_weights(labels, probas, model_names, folds, ids):
    """Optimize per-model scalar weights using nested CV on fold structure."""
    M = len(model_names)
    prob_stack = np.stack([probas[m] for m in model_names], axis=0)
    unique_folds = np.unique(folds)

    oof_ensemble = np.zeros_like(prob_stack[0])
    fold_weights = {}

    for k in unique_folds:
        train_mask = folds != k
        val_mask = folds == k
        w = _optimize_weights(labels[train_mask],
                              prob_stack[:, train_mask, :])
        fold_weights[int(k)] = dict(zip(model_names,
                                        [round(float(x), 4) for x in w]))
        oof_ensemble[val_mask] = np.tensordot(
            w, prob_stack[:, val_mask, :], axes=([0], [0]))

    nested_score = score(labels, oof_ensemble)

    mean_w = np.mean([list(fw.values()) for fw in fold_weights.values()],
                     axis=0)
    w_str = ",".join(f"{m}={mean_w[i]:.3f}"
                     for i, m in enumerate(model_names))

    result = {
        "method": "model_weights",
        "variant": "nested_cv",
        "eval": "nested CV on site-aware folds",
        "n_models": M,
        "models_used": ",".join(model_names),
        "params": w_str,
        **nested_score,
    }

    pred_df = _build_pred_df(ids, labels, folds, oof_ensemble,
                             "model_weights")
    details = {"fold_weights": fold_weights,
               "mean_weights": dict(zip(model_names,
                                        [round(float(x), 4)
                                         for x in mean_w]))}
    return [result], pred_df, details


def _optimize_weights(y, prob_stack):
    """Find model weights minimizing log loss. prob_stack: (M, N, C)."""
    M = prob_stack.shape[0]

    def objective(w):
        w_pos = np.abs(w)
        w_norm = w_pos / w_pos.sum()
        avg = np.tensordot(w_norm, prob_stack, axes=([0], [0]))
        return log_loss(y, clip_and_norm(avg), labels=list(range(NC)))

    best_loss = np.inf
    best_w = np.ones(M) / M
    starts = [np.ones(M) / M]
    for i in range(M):
        s = np.ones(M) * 0.01
        s[i] = 1.0
        starts.append(s)

    for w0 in starts:
        res = minimize(objective, w0, method="L-BFGS-B",
                       bounds=[(0.001, None)] * M,
                       options={"maxiter": 200, "ftol": 1e-8})
        if res.fun < best_loss:
            best_loss = res.fun
            best_w = np.abs(res.x)

    return best_w / best_w.sum()


# ═════════════════════════════════════════════════════════════════════════════
# Method 3: Class-model weights (nested CV)
# ═════════════════════════════════════════════════════════════════════════════

def run_class_model_weights(labels, probas, model_names, folds, ids):
    """Optimized per-class per-model weights with nested CV."""
    M = len(model_names)
    prob_stack = np.stack([probas[m] for m in model_names], axis=0)
    unique_folds = np.unique(folds)

    oof_ensemble = np.zeros_like(prob_stack[0])

    for k in unique_folds:
        train_mask = folds != k
        val_mask = folds == k
        W = _optimize_class_weights(labels[train_mask],
                                    prob_stack[:, train_mask, :])
        probs_val = prob_stack[:, val_mask, :]
        for c in range(NC):
            oof_ensemble[val_mask, c] = sum(
                W[m, c] * probs_val[m, :, c] for m in range(M))

    nested_score = score(labels, oof_ensemble)

    # Full fit for weight reporting only (not used in reported score)
    W_full = _optimize_class_weights(labels, prob_stack)
    class_weights = {m: {CLASSES[c]: round(float(W_full[i, c]), 4)
                         for c in range(NC)}
                     for i, m in enumerate(model_names)}

    result = {
        "method": "class_model_weights",
        "variant": "nested_cv",
        "eval": "nested CV on site-aware folds",
        "n_models": M,
        "models_used": ",".join(model_names),
        "params": f"{M}x{NC} weight matrix (see details)",
        **nested_score,
    }

    pred_df = _build_pred_df(ids, labels, folds, oof_ensemble,
                             "class_model_weights")
    return [result], pred_df, {"class_weights": class_weights}


def _optimize_class_weights(y, prob_stack):
    """Optimize per-class per-model weights. Returns (M, C) matrix."""
    M, N, C = prob_stack.shape
    W = np.ones((M, C)) / M

    for c in range(C):
        class_probs = prob_stack[:, :, c]

        def objective(w):
            w_pos = np.abs(w)
            w_norm = w_pos / w_pos.sum()
            blended = w_norm @ class_probs
            ensemble = np.zeros((N, C))
            for c2 in range(C):
                if c2 == c:
                    ensemble[:, c2] = blended
                else:
                    ensemble[:, c2] = np.mean(prob_stack[:, :, c2], axis=0)
            return log_loss(y, clip_and_norm(ensemble), labels=list(range(C)))

        w0 = np.ones(M) / M
        res = minimize(objective, w0, method="L-BFGS-B",
                       bounds=[(0.001, None)] * M,
                       options={"maxiter": 200, "ftol": 1e-8})
        w = np.abs(res.x)
        W[:, c] = w / w.sum()

    return W


# ═════════════════════════════════════════════════════════════════════════════
# Method 4: Temperature scaling (nested CV + uniform grid)
# ═════════════════════════════════════════════════════════════════════════════

def run_temperature(labels, logits, model_names, folds, ids, cfg):
    """Per-model temperature + ensemble, both nested CV and uniform grid."""
    grid = cfg.get("grid", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
    unique_folds = np.unique(folds)
    M = len(model_names)
    results = []
    details = {}
    pred_dfs = {}

    # ── Per-model optimal T with nested CV ───────────────────────────────
    oof_cal = np.zeros((len(labels), NC))
    fold_temps = {}

    for k in unique_folds:
        train_mask = folds != k
        val_mask = folds == k
        k_temps = {}
        for name in model_names:
            best_t, best_ll = 1.0, np.inf
            for t in grid:
                cal = softmax(logits[name][train_mask], t)
                ll = log_loss(labels[train_mask], clip_and_norm(cal),
                              labels=list(range(NC)))
                if ll < best_ll:
                    best_ll, best_t = ll, t
            k_temps[name] = best_t
        fold_temps[int(k)] = k_temps

        cal_probs = [softmax(logits[name][val_mask], k_temps[name])
                     for name in model_names]
        oof_cal[val_mask] = np.mean(cal_probs, axis=0)

    nested_score = score(labels, oof_cal)

    mean_temps = {}
    for name in model_names:
        mean_temps[name] = round(
            np.mean([fold_temps[k][name] for k in fold_temps]), 2)
    t_str = ",".join(f"{m}={mean_temps[m]}" for m in model_names)

    results.append({
        "method": "temperature",
        "variant": "per_model_optimal+avg (nested_cv)",
        "eval": "nested CV on site-aware folds",
        "n_models": M,
        "models_used": ",".join(model_names),
        "params": t_str,
        **nested_score,
    })
    pred_dfs["temperature_nested_cv"] = _build_pred_df(
        ids, labels, folds, oof_cal, "temperature_nested_cv")

    details["fold_temps"] = fold_temps
    details["mean_temps"] = mean_temps

    # ── Uniform T grid (no fitting — exact scores) ───────────────────────
    for t in grid:
        cal_probs = [softmax(logits[m], t) for m in model_names]
        avg = np.mean(cal_probs, axis=0)
        s = score(labels, avg)
        variant = f"uniform_T={t}+avg"
        results.append({
            "method": "temperature",
            "variant": variant,
            "eval": "exact (fixed T, no fitting)",
            "n_models": M,
            "models_used": ",".join(model_names),
            "params": f"T={t}",
            **s,
        })
        pred_dfs[f"temperature_T{t}"] = _build_pred_df(
            ids, labels, folds, avg, variant)

    return results, pred_dfs, details


# ═════════════════════════════════════════════════════════════════════════════
# Method 5: Greedy model selection (nested CV)
# ═════════════════════════════════════════════════════════════════════════════

def run_greedy_model(labels, probas, model_names, folds, ids):
    """Forward selection with nested CV.

    For each held-out fold k: run greedy selection on remaining K-1 folds,
    then evaluate the selected subset on fold k. The full-data selection
    order is reported for interpretability only — the score is from
    nested CV.
    """
    unique_folds = np.unique(folds)

    # ── Nested CV ────────────────────────────────────────────────────────
    oof_greedy = np.zeros((len(labels), NC))
    fold_selections = {}

    for k in unique_folds:
        train_mask = folds != k
        val_mask = folds == k
        y_train = labels[train_mask]

        selected = _greedy_select(
            y_train, {m: probas[m][train_mask] for m in model_names},
            model_names)
        fold_selections[int(k)] = selected

        avg_val = np.mean([probas[m][val_mask] for m in selected], axis=0)
        oof_greedy[val_mask] = avg_val

    nested_score = score(labels, oof_greedy)

    # ── Full-data selection order (interpretability) ─────────────────────
    full_selected = _greedy_select(labels, probas, model_names)
    full_order = full_selected.copy()

    result = {
        "method": "greedy_model",
        "variant": f"nested_cv ({len(full_selected)} on full data)",
        "eval": "nested CV on site-aware folds",
        "n_models": len(full_selected),
        "models_used": ",".join(full_selected),
        "params": f"full_order={full_order}",
        **nested_score,
    }

    pred_df = _build_pred_df(ids, labels, folds, oof_greedy, "greedy_model")
    details = {
        "full_data_selection": full_order,
        "fold_selections": fold_selections,
    }
    return [result], pred_df, details


def _greedy_select(y, probas_dict, model_names):
    """Run greedy forward selection on given data. Returns ordered list."""
    selected = []
    remaining = list(model_names)
    best_ll = np.inf
    while remaining:
        best_add, best_add_ll = None, np.inf
        for candidate in remaining:
            trial = selected + [candidate]
            avg = np.mean([probas_dict[m] for m in trial], axis=0)
            ll = log_loss(y, clip_and_norm(avg), labels=list(range(NC)))
            if ll < best_add_ll:
                best_add_ll, best_add = ll, candidate
        if best_add_ll >= best_ll and selected:
            break
        selected.append(best_add)
        remaining.remove(best_add)
        best_ll = best_add_ll
    return selected


# ═════════════════════════════════════════════════════════════════════════════
# Method 6: Meta-learner (model level, nested CV)
# ═════════════════════════════════════════════════════════════════════════════

def run_meta_model(labels, probas, model_names, folds, ids, cfg):
    """Logistic regression stacking with nested CV.

    Each fold is a held-out set of sites the meta-learner has never seen,
    making this an honest test of cross-site generalization.
    """
    M = len(model_names)
    C_values = cfg.get("C", [0.01, 0.1, 1.0, 10.0])
    X = np.hstack([probas[m] for m in model_names])
    unique_folds = np.unique(folds)
    results = []
    pred_dfs = {}

    for C_val in C_values:
        oof_meta = np.zeros((len(labels), NC))

        for k in unique_folds:
            train_mask = folds != k
            val_mask = folds == k

            clf = LogisticRegression(
                C=C_val, max_iter=1000, solver="lbfgs",
            )
            clf.fit(X[train_mask], labels[train_mask])
            raw = clf.predict_proba(X[val_mask])
            oof_meta[val_mask] = _expand_proba(raw, clf.classes_, NC)

        s = score(labels, oof_meta)
        variant = f"logreg_C={C_val}"
        results.append({
            "method": "meta_model",
            "variant": variant,
            "eval": "nested CV on site-aware folds",
            "n_models": M,
            "models_used": ",".join(model_names),
            "params": f"C={C_val}, features={M}x{NC}={M*NC}",
            **s,
        })
        pred_dfs[f"meta_model_C{C_val}"] = _build_pred_df(
            ids, labels, folds, oof_meta, variant)

    return results, pred_dfs


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

def run_ensemble(config_path):
    """Main entry: load config, run methods, save results."""
    with open(config_path) as f:
        cfg = json.load(f)

    models_dir = cfg["models_dir"]
    model_names = cfg["models"]
    pred_file = cfg.get("pred_file", "oof_predictions.csv")
    output_dir = Path(cfg.get("output_dir", "results/ensemble"))
    methods_cfg = cfg.get("methods", {})

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    t_start = time.time()

    print(f"{'═'*65}")
    print(f"Ensemble testing — {len(model_names)} models")
    print(f"{'═'*65}\n")

    # Warn about excluded methods
    for excluded in ("greedy_fold", "meta_fold"):
        if _enabled(methods_cfg, excluded):
            print(f"  ⚠ '{excluded}' excluded: within-fold evaluation is "
                  f"systematically optimistic with site-aware splits.\n"
                  f"    Use meta_model or greedy_model (nested CV) instead.\n")

    # ── Load data ────────────────────────────────────────────────────────
    labels, probas, logits, folds, ids, sites = load_oof(
        models_dir, model_names, pred_file)

    all_results = []
    all_details = {}

    # ── Individual model baselines ───────────────────────────────────────
    print(f"\n{'─'*65}")
    print("Individual model baselines")
    print(f"{'─'*65}")
    for name in model_names:
        s = score(labels, probas[name])
        all_results.append({
            "method": "individual",
            "variant": name,
            "eval": "OOF (exact)",
            "n_models": 1,
            "models_used": name,
            "params": "",
            **s,
        })
        df = _build_pred_df(ids, labels, folds, probas[name],
                            f"individual_{name}")
        df.to_csv(pred_dir / f"individual_{name}.csv", index=False)

    # ── Simple average ───────────────────────────────────────────────────
    if _enabled(methods_cfg, "simple_average"):
        print(f"\n{'─'*65}")
        print("Simple average")
        print(f"{'─'*65}")
        res, pred_df = run_simple_average(
            labels, probas, model_names, ids, folds)
        all_results.extend(res)
        pred_df.to_csv(pred_dir / "simple_average.csv", index=False)
        _print_results(res)

    # ── Model weights ────────────────────────────────────────────────────
    if _enabled(methods_cfg, "model_weights"):
        print(f"\n{'─'*65}")
        print("Model weights (optimized, nested CV)")
        print(f"{'─'*65}")
        res, pred_df, det = run_model_weights(
            labels, probas, model_names, folds, ids)
        all_results.extend(res)
        all_details["model_weights"] = det
        pred_df.to_csv(pred_dir / "model_weights.csv", index=False)
        _print_results(res)
        print(f"  Mean weights: {det['mean_weights']}")

    # ── Class-model weights ──────────────────────────────────────────────
    if _enabled(methods_cfg, "class_model_weights"):
        print(f"\n{'─'*65}")
        print("Class-model weights (optimized, nested CV)")
        print(f"{'─'*65}")
        res, pred_df, det = run_class_model_weights(
            labels, probas, model_names, folds, ids)
        all_results.extend(res)
        all_details["class_model_weights"] = det
        pred_df.to_csv(pred_dir / "class_model_weights.csv", index=False)
        _print_results(res)

    # ── Temperature ──────────────────────────────────────────────────────
    if _enabled(methods_cfg, "temperature"):
        print(f"\n{'─'*65}")
        print("Temperature scaling")
        print(f"{'─'*65}")
        temp_cfg = methods_cfg.get("temperature", {})
        if isinstance(temp_cfg, bool):
            temp_cfg = {}
        res, pred_dfs, det = run_temperature(
            labels, logits, model_names, folds, ids, temp_cfg)
        all_results.extend(res)
        all_details["temperature"] = det
        for name, df in pred_dfs.items():
            df.to_csv(pred_dir / f"{name}.csv", index=False)
        _print_results(res)

    # ── Greedy model (nested CV) ─────────────────────────────────────────
    if _enabled(methods_cfg, "greedy_model"):
        print(f"\n{'─'*65}")
        print("Greedy model selection (nested CV)")
        print(f"{'─'*65}")
        res, pred_df, det = run_greedy_model(
            labels, probas, model_names, folds, ids)
        all_results.extend(res)
        all_details["greedy_model"] = det
        pred_df.to_csv(pred_dir / "greedy_model.csv", index=False)
        _print_results(res)
        print(f"  Full-data order: {det['full_data_selection']}")
        print(f"  Per-fold selections:")
        for k, sel in det["fold_selections"].items():
            print(f"    fold {k}: {sel}")

    # ── Meta model ───────────────────────────────────────────────────────
    if _enabled(methods_cfg, "meta_model"):
        print(f"\n{'─'*65}")
        print("Meta-learner (model level, nested CV)")
        print(f"{'─'*65}")
        meta_cfg = methods_cfg.get("meta_model", {})
        if isinstance(meta_cfg, bool):
            meta_cfg = {}
        res, pred_dfs = run_meta_model(
            labels, probas, model_names, folds, ids, meta_cfg)
        all_results.extend(res)
        for name, df in pred_dfs.items():
            df.to_csv(pred_dir / f"{name}.csv", index=False)
        _print_results(res)

    # ── Save results ─────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("log_loss").reset_index(drop=True)
    results_df.insert(0, "rank", range(1, len(results_df) + 1))

    csv_path = output_dir / "ensemble_results.csv"
    results_df.to_csv(csv_path, index=False)

    # Summary JSON
    ensemble_only = results_df[results_df["method"] != "individual"]
    best_ens = ensemble_only.iloc[0] if not ensemble_only.empty else None
    indiv = results_df[results_df["method"] == "individual"]
    best_indiv_ll = float(indiv["log_loss"].min()) if not indiv.empty else np.inf

    summary = {
        "config": cfg,
        "n_models": len(model_names),
        "n_images": len(labels),
        "n_folds": len(np.unique(folds)),
        "n_tests": len(results_df),
        "elapsed_seconds": round(elapsed, 1),
        "best_individual": {
            "model": (indiv.loc[indiv["log_loss"].idxmin(), "variant"]
                      if not indiv.empty else ""),
            "log_loss": round(best_indiv_ll, 6),
        },
        "best_ensemble": {
            "method": best_ens["method"],
            "variant": best_ens["variant"],
            "eval": best_ens["eval"],
            "log_loss": round(float(best_ens["log_loss"]), 6),
            "f1_macro": round(float(best_ens["f1_macro"]), 6),
            "accuracy": round(float(best_ens["accuracy"]), 6),
            "n_models": int(best_ens["n_models"]),
            "models_used": best_ens["models_used"],
        } if best_ens is not None else {},
        "improvement": round(
            best_indiv_ll - float(best_ens["log_loss"]), 6
        ) if best_ens is not None else 0.0,
        "details": all_details,
    }

    json_path = output_dir / "ensemble_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n\n{'═'*65}")
    print(f"RESULTS — {len(results_df)} tests ({elapsed:.0f}s)")
    print(f"All scores: nested CV or parameter-free (no leakage)")
    print(f"{'═'*65}")
    print(f"\n{'rank':>4}  {'method':<25} {'variant':<35} "
          f"{'log_loss':>9} {'f1':>7} {'acc':>7}")
    print(f"{'─'*4}  {'─'*25} {'─'*35} {'─'*9} {'─'*7} {'─'*7}")

    shown = set()
    for _, row in results_df.head(15).iterrows():
        _print_row(row)
        shown.add(row.name)

    if not indiv.empty and not all(i in shown for i in indiv.index):
        print(f"  {'...'}")
        for _, row in indiv.iterrows():
            if row.name not in shown:
                _print_row(row)

    if best_ens is not None:
        best_ens_ll = float(best_ens["log_loss"])
        delta = best_indiv_ll - best_ens_ll
        print(f"\n  Best individual:  {best_indiv_ll:.6f}")
        print(f"  Best ensemble:    {best_ens_ll:.6f}")
        print(f"  Improvement:      {delta:+.6f}")
        print(f"\n  Recommended: {best_ens['method']} / {best_ens['variant']}")
        print(f"  Models: {best_ens['models_used']}")

    print(f"\n-> {csv_path}")
    print(f"-> {json_path}")
    print(f"-> {pred_dir}/ ({len(list(pred_dir.glob('*.csv')))} files)")


def _enabled(methods_cfg, key):
    v = methods_cfg.get(key, False)
    if isinstance(v, bool):
        return v
    if isinstance(v, dict):
        return v.get("enabled", True)
    return bool(v)


def _print_results(results):
    for r in results:
        print(f"  {r['variant']:<45} "
              f"ll={r['log_loss']:.4f}  f1={r['f1_macro']:.4f}  "
              f"acc={r['accuracy']:.4f}")


def _print_row(row):
    print(f"{int(row['rank']):>4}  {row['method']:<25} "
          f"{str(row['variant']):<35} "
          f"{row['log_loss']:>9.6f} {row['f1_macro']:>7.4f} "
          f"{row['accuracy']:>7.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Test ensemble approaches on OOF predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to ensemble config JSON")
    args = parser.parse_args()

    run_ensemble(args.config)


if __name__ == "__main__":
    main()
