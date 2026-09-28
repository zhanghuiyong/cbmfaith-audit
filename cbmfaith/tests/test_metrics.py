import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import validate

from cbmfaith import AuditInputs, audit_faithfulness
from cbmfaith.metrics import (
    apply_gauge,
    as_1d,
    concept_metrics,
    diagnostic_metrics,
    fit_gauge_reference,
    gauge_fixed_route_reliance,
    operational_completeness,
    probabilities_to_logits,
    replacement_metrics,
    residual_concept_leakage,
    route_reliance,
    safe_auc,
)
from cbmfaith.validation import schema_path, validate_report


def test_residual_dominance_and_semantics_are_separate():
    y = np.array([0, 0, 1, 1])
    concept = np.array([0.49, 0.49, 0.51, 0.51])
    residual = np.array([0.1, 0.2, 0.8, 0.9])
    labels = y[:, None]
    report = audit_faithfulness(AuditInputs(
        y_true=y,
        combined_probability=residual,
        concept_probability=concept,
        residual_probability=residual,
        concept_labels=labels,
        concept_predictions=labels * 0.8 + 0.1,
        concept_names=["visible_feature"],
    ))
    assert report["semantic_recovery"]["macro_auroc"] == 1.0
    assert report["residual_reliance"]["normalized_residual_share_median"] > 0.9


def test_saturated_probabilities_preserve_auc_ranking():
    from sklearn.metrics import roc_auc_score

    from cbmfaith.metrics import diagnostic_metrics

    y = np.array([0, 1, 0, 1])
    p = np.array([1e-12, 2e-12, 1 - 2e-12, 1 - 1e-12])
    assert diagnostic_metrics(y, p)["auroc"] == roc_auc_score(y, p)


def test_gauge_application_preserves_combined_logit():
    c = np.array([-1.0, 0.2, 2.0])
    r = np.array([0.7, -0.4, 1.0])
    cg, rg = apply_gauge(c, r, 0.35)
    np.testing.assert_allclose(cg + rg, c + r)


def test_gauge_reference_is_development_median():
    reference = fit_gauge_reference([4.0, 1.0, 2.0], "median")
    assert reference["offset"] == 2.0
    assert reference["n_reference"] == 3


def test_gauge_reference_rejects_unknown_method():
    with pytest.raises(ValueError):
        fit_gauge_reference([1.0, 2.0], "mode")


def test_gauge_fixed_report_records_zero_sum_error():
    c = np.array([2.0, 3.0, 4.0])
    r = np.array([-1.0, 0.5, 1.0])
    report = gauge_fixed_route_reliance(c, r, c + r, offset=3.0)
    assert report["gauge"]["sum_preserved_max_abs_error"] < 1e-12


def test_raw_route_share_is_explicitly_gauge_dependent():
    report = route_reliance([1.0, 2.0], [3.0, 4.0], [4.0, 6.0])
    assert "gauge-dependent" in report["definition"]


def test_probability_logit_round_trip_inside_open_interval():
    p = np.array([0.1, 0.5, 0.9])
    z = probabilities_to_logits(p)
    np.testing.assert_allclose(1 / (1 + np.exp(-z)), p)


def test_diagnostic_metrics_rejects_out_of_range_probability():
    with pytest.raises(ValueError):
        diagnostic_metrics([0, 1], [0.2, 1.2])


def test_diagnostic_metrics_rejects_length_mismatch():
    with pytest.raises(ValueError):
        diagnostic_metrics([0, 1], [0.2])


def test_safe_auc_returns_none_for_single_class():
    assert safe_auc([1, 1, 1], [0.1, 0.2, 0.3]) is None


def test_concept_metrics_handles_constant_concept():
    labels = np.array([[0, 1], [1, 1], [0, 1], [1, 1]])
    probabilities = np.array([[0.1, 0.2], [0.9, 0.2], [0.2, 0.2], [0.8, 0.2]])
    report = concept_metrics(labels, probabilities, ["variable", "constant"])
    assert report["per_concept_auroc"]["variable"] == 1.0
    assert report["per_concept_auroc"]["constant"] is None


def test_replacement_metrics_accepts_multiple_interventions():
    report = replacement_metrics([0.0, 0.0], [[1.0, -1.0], [0.5, -0.5]])
    assert len(report["per_replacement_mean_absolute_logit_change"]) == 2
    assert "model-level" in report["definition"]


def test_operational_completeness_is_none_at_chance_blackbox():
    report = operational_completeness([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], [0.5] * 4)
    assert report["value"] is None


def test_nonfinite_values_fail_closed():
    with pytest.raises(ValueError):
        as_1d([0.0, np.nan], "bad")


def test_audit_emits_gauge_fixed_section_when_offset_supplied():
    y = np.array([0, 0, 1, 1])
    report = audit_faithfulness(AuditInputs(
        y_true=y,
        combined_probability=np.array([0.1, 0.2, 0.8, 0.9]),
        concept_logit=np.array([0.1, 0.1, 0.2, 0.2]),
        residual_logit=np.array([-2.3, -1.5, 1.2, 2.0]),
        gauge_offset=0.15,
    ))
    assert "residual_reliance_gauge_fixed" in report


def test_audit_output_validates_against_packaged_schema():
    y = np.array([0, 0, 1, 1])
    report = audit_faithfulness(AuditInputs(
        y_true=y,
        combined_probability=np.array([0.1, 0.2, 0.8, 0.9]),
        concept_probability=np.array([0.2, 0.3, 0.7, 0.8]),
        residual_probability=np.array([0.1, 0.2, 0.8, 0.9]),
    ))
    schema_path = Path(__file__).parents[1] / "schema" / "benchmark-output.schema.json"
    validate(report, json.loads(schema_path.read_text()))


def test_offline_validator_accepts_audit_and_schema_is_packaged():
    report = audit_faithfulness(AuditInputs(
        y_true=np.array([0, 0, 1, 1]),
        combined_probability=np.array([0.1, 0.2, 0.8, 0.9]),
    ))
    assert validate_report(report) == []
    assert schema_path().exists()


def test_grouped_residual_decodability_preserves_biological_units():
    groups = np.repeat(np.arange(8), 2)
    labels = np.repeat([0, 0, 0, 0, 1, 1, 1, 1], 2)[:, None]
    features = np.column_stack([labels[:, 0] + np.linspace(0, 0.01, len(labels)), np.arange(len(labels)) % 2])
    report = residual_concept_leakage(features, labels, ["c"], seed=3, groups=groups)
    assert report["per_concept_auroc"]["c"] > 0.9
    assert "grouped" in report["definition"]


def test_grouped_residual_decodability_rejects_misaligned_groups():
    with pytest.raises(ValueError):
        residual_concept_leakage(np.ones((4, 2)), np.ones((4, 1)), ["c"], groups=[1, 2])


def test_known_residual_bypass_is_detected_by_route_metrics():
    rng = np.random.default_rng(20260904)
    y = np.repeat([0, 1], 128)
    concept_logit = 0.05 * rng.normal(size=len(y))
    residual_logit = 3.0 * (2 * y - 1) + 0.1 * rng.normal(size=len(y))
    combined = concept_logit + residual_logit
    report = audit_faithfulness(AuditInputs(
        y_true=y,
        combined_probability=1 / (1 + np.exp(-combined)),
        concept_logit=concept_logit,
        residual_logit=residual_logit,
    ))
    assert report["residual_reliance"]["normalized_residual_share_median"] > 0.95
    assert report["residual_reliance"]["combined_residual_spearman"] > 0.95


def test_interaction_displacement_preserves_prediction_but_changes_attribution():
    rng = np.random.default_rng(7)
    c = rng.choice([-1.0, 1.0], size=(256, 2))
    linear = c[:, 0] + 0.5 * c[:, 1]
    interaction = 1.2 * c[:, 0] * c[:, 1]
    total = linear + interaction
    all_concept = route_reliance(total, np.zeros(len(total)), total)
    displaced = route_reliance(linear, interaction, total)
    assert all_concept["normalized_residual_share_median"] == 0.0
    assert displaced["normalized_residual_share_median"] > 0.0


def test_route_metrics_do_not_certify_semantic_purity():
    # A target-contaminated concept score has no residual bypass, illustrating the
    # benchmark's deliberately documented certification boundary.
    y = np.repeat([0, 1], 64)
    contaminated_concept = 4.0 * (2 * y - 1)
    report = audit_faithfulness(AuditInputs(
        y_true=y,
        combined_probability=1 / (1 + np.exp(-contaminated_concept)),
        concept_logit=contaminated_concept,
        residual_logit=np.zeros(len(y)),
    ))
    assert report["combined_diagnosis"]["auroc"] == 1.0
    assert report["residual_reliance"]["normalized_residual_share_median"] == 0.0
