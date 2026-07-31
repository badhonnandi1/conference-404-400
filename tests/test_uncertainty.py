
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.analysis.uncertainty import (
    binary_classification_metrics,
    confusion_matrix_counts,
    fold_metrics_and_variance,
    source_clustered_bootstrap,
)
from src.analysis.reporting import validate_primary_method_comparison


def _rows() -> list[dict[str, str]]:
    return [
        {"source_id": "A", "held_out_source": "A", "expected_label": "normal", "observed_label": "normal"},
        {"source_id": "A", "held_out_source": "A", "expected_label": "abnormal", "observed_label": "abnormal"},
        {"source_id": "B", "held_out_source": "B", "expected_label": "normal", "observed_label": "abnormal"},
        {"source_id": "B", "held_out_source": "B", "expected_label": "abnormal", "observed_label": "normal"},
        {"source_id": "B", "held_out_source": "B", "expected_label": "abnormal", "observed_label": "abnormal"},
    ]


def test_metric_formulas_include_authentication_far_and_frr() -> None:
    metrics = binary_classification_metrics(_rows())
    assert (metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"]) == (2, 1, 1, 1)
    assert metrics["FAR"] == pytest.approx(1 / 3)
    assert metrics["FRR"] == pytest.approx(1 / 2)
    assert metrics["balanced_accuracy"] == pytest.approx(((2 / 3) + (1 / 2)) / 2)
    assert confusion_matrix_counts(_rows()).tolist() == [[1, 1], [1, 2]]


def test_undefined_denominators_remain_nan() -> None:
    metrics = binary_classification_metrics(
        [{"expected_label": "normal", "observed_label": "normal"}]
    )
    assert math.isnan(float(metrics["precision"]))
    assert math.isnan(float(metrics["recall"]))
    assert math.isnan(float(metrics["FAR"]))
    assert metrics["specificity"] == 1.0
    assert metrics["FRR"] == 0.0


def test_source_cluster_bootstrap_is_fixed_seed_deterministic() -> None:
    summaries_a, distribution_a = source_clustered_bootstrap(_rows(), repetitions=40, seed=404)
    summaries_b, distribution_b = source_clustered_bootstrap(_rows(), repetitions=40, seed=404)
    assert summaries_a == summaries_b
    assert distribution_a == distribution_b
    for summary in summaries_a:
        assert summary["ci_2_5_percentile"] <= summary["ci_97_5_percentile"]
        assert "point_estimate" in summary
        assert "bootstrap_mean" in summary


def test_source_cluster_bootstrap_preserves_cluster_multiplicity() -> None:
    _, distribution = source_clustered_bootstrap(_rows(), repetitions=30, seed=404)
    saw_repeated_cluster = False
    source_sizes = {"A": 2, "B": 3}
    for sample in distribution:
        multiplicity = json.loads(sample["sampled_source_multiplicity"])
        assert sum(multiplicity.values()) == 2
        expected_total = sum(multiplicity[source] * size for source, size in source_sizes.items())
        assert sample["total"] == expected_total
        saw_repeated_cluster |= 2 in multiplicity.values()
    assert saw_repeated_cluster


def test_fold_variance_uses_sample_standard_deviation() -> None:
    fold_rows, variance = fold_metrics_and_variance(_rows())
    accuracy_values = np.asarray([float(row["accuracy"]) for row in fold_rows])
    accuracy_summary = next(row for row in variance if row["metric"] == "accuracy")
    assert accuracy_summary["fold_sample_standard_deviation"] == pytest.approx(
        np.std(accuracy_values, ddof=1)
    )


def test_primary_comparison_uses_phash_and_proposed_workflow_not_sha256() -> None:
    validate_primary_method_comparison(
        [
            {"method": "OpenCV pHash"},
            {"method": "Proposed hybrid authentication workflow"},
        ]
    )
    with pytest.raises(ValueError, match="Primary comparison"):
        validate_primary_method_comparison(
            [
                {"method": "SHA-256"},
                {"method": "Proposed hybrid authentication workflow"},
            ]
        )
