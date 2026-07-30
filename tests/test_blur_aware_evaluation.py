from __future__ import annotations

import pytest

from scripts.run_blur_aware_final_evaluation import (
    ThresholdModel,
    compression_profile,
    evaluate_predictions,
    structural_issue,
    threshold_from_values,
    video_score_rows,
)


def _video(label: str = "normal") -> dict[str, object]:
    return {
        "source_id": "SRC_VID01",
        "video_id": "VID01_480P",
        "filename": "vid01(480).mp4",
        "transformation_type": "resize_480p",
        "expected_category": "benign",
        "expected_label": label,
        "width": 854,
        "height": 480,
        "file_extension": ".mp4",
    }


def test_threshold_from_values_is_deterministic() -> None:
    values = [0.1, 0.2, 0.2, 0.3]

    assert threshold_from_values(values, margin_multiplier=3.0, one_unit=0.01) == threshold_from_values(
        list(reversed(values)),
        margin_multiplier=3.0,
        one_unit=0.01,
    )


def test_metric_counts_are_deterministic_and_include_far_frr() -> None:
    predictions = [
        {"expected_label": "abnormal", "observed_label": "abnormal"},
        {"expected_label": "abnormal", "observed_label": "normal"},
        {"expected_label": "normal", "observed_label": "abnormal"},
        {"expected_label": "normal", "observed_label": "normal"},
    ]

    metrics = evaluate_predictions(predictions)

    assert metrics["TP"] == 1
    assert metrics["TN"] == 1
    assert metrics["FP"] == 1
    assert metrics["FN"] == 1
    assert metrics["FAR"] == pytest.approx(0.5)
    assert metrics["FRR"] == pytest.approx(0.5)


def test_structural_issue_detects_missing_and_extra_segments() -> None:
    structural, missing, extra = structural_issue([0, 1, 2], [0, 2, 3])

    assert structural
    assert missing == 1
    assert extra == 1


def test_video_score_rows_handles_failed_or_structural_video() -> None:
    model = ThresholdModel(
        score_threshold=0.5,
        blur_loss_threshold=0.5,
        margin_multiplier=3.0,
        weights={"resnet": 0.4, "temporal": 0.3, "spatial": 0.3},
        profile_thresholds={},
        profile_counts={},
    )

    prediction, segment_rows = video_score_rows(
        algorithm="V2_FIXED",
        fold_id="LOSO_VID01",
        rows=[],
        video=_video("normal"),
        threshold_model=model,
        structural=True,
        missing_segments=1,
        extra_segments=0,
    )

    assert prediction["observed_label"] == "abnormal"
    assert prediction["structural_issue"]
    assert segment_rows == []


def test_compression_profile_uses_extension_and_resolution_without_label() -> None:
    assert compression_profile(_video()) == ".mp4:480p"
