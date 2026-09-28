"""Schema and orchestrator for a standardized CBM faithfulness audit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import (
    concept_metrics,
    diagnostic_metrics,
    gauge_fixed_route_reliance,
    operational_completeness,
    probabilities_to_logits,
    replacement_metrics,
    residual_concept_leakage,
    route_reliance,
)


@dataclass
class AuditInputs:
    y_true: np.ndarray
    combined_probability: np.ndarray
    concept_probability: np.ndarray | None = None
    residual_probability: np.ndarray | None = None
    concept_logit: np.ndarray | None = None
    residual_logit: np.ndarray | None = None
    concept_labels: np.ndarray | None = None
    concept_predictions: np.ndarray | None = None
    concept_names: list[str] | None = None
    replaced_combined_logits: np.ndarray | None = None
    blackbox_probability: np.ndarray | None = None
    residual_features: np.ndarray | None = None
    gauge_offset: float | None = None
    groups: np.ndarray | None = None


def audit_faithfulness(inputs: AuditInputs, seed: int = 0) -> dict:
    report = {
        "schema_version": "0.3.0",
        "combined_diagnosis": diagnostic_metrics(inputs.y_true, inputs.combined_probability),
        "causal_scope": "observational route audit unless a separate causal identification argument is supplied",
    }
    combined_logit = probabilities_to_logits(inputs.combined_probability)
    if inputs.concept_probability is not None:
        report["concept_route_diagnosis"] = diagnostic_metrics(inputs.y_true, inputs.concept_probability)
    if inputs.residual_probability is not None:
        report["residual_route_diagnosis"] = diagnostic_metrics(inputs.y_true, inputs.residual_probability)
    if ((inputs.concept_logit is not None and inputs.residual_logit is not None) or
            (inputs.concept_probability is not None and inputs.residual_probability is not None)):
        report["residual_reliance"] = route_reliance(
            inputs.concept_logit if inputs.concept_logit is not None else probabilities_to_logits(inputs.concept_probability),
            inputs.residual_logit if inputs.residual_logit is not None else probabilities_to_logits(inputs.residual_probability),
            combined_logit,
        )
        if inputs.gauge_offset is not None:
            report["residual_reliance_gauge_fixed"] = gauge_fixed_route_reliance(
                inputs.concept_logit if inputs.concept_logit is not None else probabilities_to_logits(inputs.concept_probability),
                inputs.residual_logit if inputs.residual_logit is not None else probabilities_to_logits(inputs.residual_probability),
                combined_logit,
                offset=inputs.gauge_offset,
            )
    if inputs.concept_labels is not None and inputs.concept_predictions is not None:
        names = inputs.concept_names or [f"concept_{j}" for j in range(inputs.concept_labels.shape[1])]
        report["semantic_recovery"] = concept_metrics(inputs.concept_labels, inputs.concept_predictions, names)
    if inputs.replaced_combined_logits is not None:
        report["label_replacement"] = replacement_metrics(combined_logit, inputs.replaced_combined_logits)
    if inputs.concept_probability is not None and inputs.blackbox_probability is not None:
        report["operational_completeness"] = operational_completeness(
            inputs.y_true, inputs.concept_probability, inputs.blackbox_probability
        )
    if inputs.residual_features is not None and inputs.concept_labels is not None:
        names = inputs.concept_names or [f"concept_{j}" for j in range(inputs.concept_labels.shape[1])]
        report["residual_concept_decodability"] = residual_concept_leakage(
            inputs.residual_features, inputs.concept_labels, names, seed, groups=inputs.groups
        )
    return report
