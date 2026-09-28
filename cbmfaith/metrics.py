"""Numerically guarded CBM audit metrics.

These functions measure observable route behavior. They do not identify causal
mediation unless an external structural causal model justifies that interpretation.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-7


def as_1d(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def probabilities_to_logits(probabilities) -> np.ndarray:
    p = np.clip(as_1d(probabilities, "probabilities"), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def safe_auc(y_true, score) -> float | None:
    y = as_1d(y_true, "y_true").astype(int)
    s = as_1d(score, "score")
    if np.unique(y).size != 2:
        return None
    return float(roc_auc_score(y, s))


def diagnostic_metrics(y_true, probability) -> dict:
    y = as_1d(y_true, "y_true").astype(int)
    p = as_1d(probability, "probability")
    if len(y) != len(p):
        raise ValueError("target/probability length mismatch")
    if np.any(p < -EPS) or np.any(p > 1 + EPS):
        raise ValueError("probability must lie in [0, 1]")
    # Keep exact ranks at 0 and 1. EPS clipping belongs only in logit
    # conversion; using it here creates artificial ties and changes AUROC.
    p = np.clip(p, 0.0, 1.0)
    return {
        "n": len(y),
        "positives": int(y.sum()),
        "auroc": safe_auc(y, p),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def route_reliance(concept_logit, residual_logit, combined_logit) -> dict:
    c = as_1d(concept_logit, "concept_logit")
    r = as_1d(residual_logit, "residual_logit")
    z = as_1d(combined_logit, "combined_logit")
    if not (len(c) == len(r) == len(z)):
        raise ValueError("route length mismatch")
    share = np.abs(r) / (np.abs(c) + np.abs(r) + EPS)
    correlation = None if np.ptp(z) == 0 or np.ptp(r) == 0 else spearmanr(z, r).statistic
    return {
        "definition": (
            "gauge-dependent absolute-logit route ratio; not variance decomposition, "
            "information fraction, or causal attribution"
        ),
        "normalized_residual_share_mean": float(share.mean()),
        "normalized_residual_share_median": float(np.median(share)),
        "residual_dominant_fraction": float(np.mean(share > 0.5)),
        "combined_residual_spearman": float(correlation) if correlation is not None and np.isfinite(correlation) else None,
    }


def fit_gauge_reference(concept_logit, method: str = "median") -> dict:
    """Fit an additive-logit gauge using development/OOF concept logits only.

    The returned offset must be frozen before applying it to an external test set.
    Subtracting the offset from the concept route and adding it to the residual
    route preserves their sum exactly.
    """
    c = as_1d(concept_logit, "concept_logit")
    if method == "median":
        offset = float(np.median(c))
    elif method == "mean":
        offset = float(np.mean(c))
    else:
        raise ValueError("gauge method must be 'median' or 'mean'")
    return {
        "method": method,
        "offset": offset,
        "n_reference": len(c),
        "definition": "development/OOF concept-logit location; freeze before test evaluation",
    }


def apply_gauge(concept_logit, residual_logit, offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply a sum-preserving additive-logit gauge transformation."""
    c = as_1d(concept_logit, "concept_logit")
    r = as_1d(residual_logit, "residual_logit")
    if len(c) != len(r):
        raise ValueError("route length mismatch")
    if not np.isfinite(offset):
        raise ValueError("gauge offset must be finite")
    return c - float(offset), r + float(offset)


def gauge_fixed_route_reliance(
    concept_logit, residual_logit, combined_logit, *, offset: float
) -> dict:
    """Route reliance under a declared, externally supplied gauge reference."""
    c, r = apply_gauge(concept_logit, residual_logit, offset)
    report = route_reliance(c, r, combined_logit)
    report["gauge"] = {
        "offset": float(offset),
        "application": "concept'=concept-offset; residual'=residual+offset",
        "sum_preserved_max_abs_error": float(
            np.max(np.abs((c + r) - (as_1d(concept_logit, "concept_logit") + as_1d(residual_logit, "residual_logit"))))
        ),
    }
    return report


def concept_metrics(labels, probabilities, names: list[str]) -> dict:
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != len(names):
        raise ValueError("concept arrays/names do not align")
    rows = {}
    for j, name in enumerate(names):
        valid = np.isfinite(y[:, j]) & np.isfinite(p[:, j])
        rows[name] = safe_auc(y[valid, j], p[valid, j]) if valid.any() else None
    finite = [value for value in rows.values() if value is not None]
    return {"per_concept_auroc": rows, "macro_auroc": float(np.mean(finite)) if finite else None}


def replacement_metrics(original_logit, replaced_logits) -> dict:
    original = as_1d(original_logit, "original_logit")
    replaced = np.asarray(replaced_logits, dtype=float)
    if replaced.ndim == 1:
        replaced = replaced[:, None]
    if replaced.shape[0] != len(original) or not np.isfinite(replaced).all():
        raise ValueError("replacement logits are invalid")
    delta = np.abs(replaced - original[:, None])
    return {
        "definition": "controlled model-level intervention; not an identified causal do-intervention",
        "mean_absolute_logit_change": float(delta.mean()),
        "median_absolute_logit_change": float(np.median(delta)),
        "per_replacement_mean_absolute_logit_change": [float(x) for x in delta.mean(0)],
    }


def operational_completeness(y_true, concept_probability, blackbox_probability) -> dict:
    concept_auc = safe_auc(y_true, concept_probability)
    blackbox_auc = safe_auc(y_true, blackbox_probability)
    if concept_auc is None or blackbox_auc is None or abs(blackbox_auc - 0.5) < EPS:
        value = None
    else:
        value = float((concept_auc - 0.5) / (blackbox_auc - 0.5))
    return {
        "definition": "(concept-route AUROC - 0.5)/(black-box AUROC - 0.5); not information-theoretic completeness",
        "value": value,
    }


def residual_concept_leakage(residual_features, concept_labels, names: list[str], seed: int = 0, groups=None) -> dict:
    x = np.asarray(residual_features, dtype=float)
    labels = np.asarray(concept_labels, dtype=float)
    if x.ndim != 2 or labels.ndim != 2 or len(x) != len(labels):
        raise ValueError("residual features/concept labels do not align")
    group_array = None if groups is None else np.asarray(groups).reshape(-1)
    if group_array is not None and len(group_array) != len(x):
        raise ValueError("groups do not align with residual features")
    rows = {}
    for j, name in enumerate(names):
        valid = np.isfinite(labels[:, j]) & np.isfinite(x).all(1)
        y = labels[valid, j].astype(int)
        counts = np.bincount(y, minlength=2)
        folds = int(min(5, counts.min()))
        if folds < 2:
            rows[name] = None
            continue
        if group_array is None:
            cv = StratifiedKFold(folds, shuffle=True, random_state=seed + j)
            cv_groups = None
        else:
            group_counts = [len(np.unique(group_array[valid][y == cls])) for cls in (0, 1)]
            folds = min(folds, *group_counts)
            if folds < 2:
                rows[name] = None
                continue
            cv = StratifiedGroupKFold(folds, shuffle=True, random_state=seed + j)
            cv_groups = group_array[valid]
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
        score = cross_val_predict(model, x[valid], y, cv=cv, groups=cv_groups, method="predict_proba")[:, 1]
        rows[name] = float(roc_auc_score(y, score))
    finite = [value for value in rows.values() if value is not None]
    return {
        "definition": (
            "grouped cross-validated decodability from residual representation; association, not causal leakage"
            if group_array is not None else
            "cross-validated decodability from residual representation; association, not causal leakage"
        ),
        "per_concept_auroc": rows,
        "macro_auroc": float(np.mean(finite)) if finite else None,
    }
