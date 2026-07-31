
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from src.analysis.uncertainty import binary_classification_metrics
from src.baselines.opencv_phash import fit_robust_threshold, video_decision


CALIBRATION_TRANSFORMATIONS = {
    "trusted_reference",
    "avi_conversion",
    "mov_conversion",
    "resize_480p",
    "resize_720p",
}


def _registry_by_video(registry: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapped = {str(row["video_id"]): row for row in registry}
    if len(mapped) != len(registry):
        raise ValueError("Video registry contains duplicate video IDs.")
    return mapped


def calibration_evidence(
    *,
    registry: Sequence[dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    training_sources: Iterable[str],
) -> tuple[list[float], list[str], int]:
    """Return benign video scores and segment count for only the named source groups."""

    sources = {str(value) for value in training_sources}
    scores: list[float] = []
    video_ids: list[str] = []
    segment_count = 0
    for row in registry:
        if str(row["source_id"]) not in sources:
            continue
        if str(row["transformation_type"]) not in CALIBRATION_TRANSFORMATIONS:
            continue
        video_id = str(row["video_id"])
        comparison = comparisons[video_id]
        score = comparison.get("video_phash_score")
        if score is None or not math.isfinite(float(score)):
            raise ValueError(f"Benign calibration video has no valid pHash score: {video_id}.")
        scores.append(float(score))
        video_ids.append(video_id)
        segment_count += sum(
            bool(segment.get("segment_valid")) for segment in comparison.get("segment_rows", [])
        )
    if not scores:
        raise ValueError("No benign pHash calibration scores were found.")
    return scores, video_ids, segment_count


def fit_source_excluded_threshold(
    *,
    held_out_source: str,
    all_sources: Sequence[str],
    registry: Sequence[dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    margin_multiplier: float,
    logical_bit_length: int,
) -> dict[str, Any]:
    """Fit one threshold after excluding the complete held-out source group."""

    training_sources = sorted(source for source in all_sources if source != held_out_source)
    if held_out_source in training_sources:
        raise AssertionError("Held-out source leaked into pHash threshold training.")
    scores, video_ids, segment_count = calibration_evidence(
        registry=registry,
        comparisons=comparisons,
        training_sources=training_sources,
    )
    fitted = fit_robust_threshold(
        scores,
        margin_multiplier=margin_multiplier,
        logical_bit_length=logical_bit_length,
    )
    return {
        **fitted,
        "held_out_source": held_out_source,
        "training_sources": training_sources,
        "benign_calibration_video_ids": video_ids,
        "benign_calibration_segment_count": segment_count,
        "held_out_used_for_fitting": False,
        "fallback_behavior": "none",
    }


def predict_source(
    *,
    source_id: str,
    registry: Sequence[dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    """Apply a frozen threshold to every video in one source group."""

    predictions: list[dict[str, Any]] = []
    for video in registry:
        if str(video["source_id"]) != source_id:
            continue
        video_id = str(video["video_id"])
        comparison = comparisons[video_id]
        decision = video_decision(
            video_score=comparison.get("video_phash_score"),
            threshold=float(threshold),
            structural_issue=bool(comparison["structural_issue"]),
            invalid_comparison=bool(comparison["invalid_comparison"]),
            segment_rows=comparison["segment_rows"],
        )
        predictions.append(
            {
                "method": "OpenCV pHash",
                "fold_id": f"LOSO_{source_id.replace('SRC_', '')}",
                "held_out_source": source_id,
                "source_id": source_id,
                "video_id": video_id,
                "reference_video_id": str(video["reference_video_id"]),
                "filename": str(video["filename"]),
                "transformation_type": str(video["transformation_type"]),
                "expected_category": str(video["expected_category"]),
                "expected_label": str(video["expected_label"]),
                "observed_label": decision["observed_label"],
                "correct": str(video["expected_label"]) == decision["observed_label"],
                "video_phash_score": comparison.get("video_phash_score"),
                "applied_threshold": float(threshold),
                "abnormal_phash_segment_ids": decision["abnormal_segment_ids"],
                "structural_issue": bool(comparison["structural_issue"]),
                "missing_segment_count": int(comparison["missing_segment_count"]),
                "extra_segment_count": int(comparison["extra_segment_count"]),
                "missing_segment_ids": comparison["missing_segment_ids"],
                "extra_segment_ids": comparison["extra_segment_ids"],
                "invalid_segment_ids": comparison["invalid_segment_ids"],
                "invalid_comparison": bool(comparison["invalid_comparison"]),
                "decision_reason": decision["decision_reason"],
            }
        )
    return predictions


def select_margin_inner_sourcewise(
    *,
    outer_training_sources: Sequence[str],
    registry: Sequence[dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    margin_grid: Sequence[float],
    logical_bit_length: int,
) -> tuple[float, list[dict[str, Any]]]:
    """Select only the pHash margin using inner held-out source groups."""

    candidates: list[dict[str, Any]] = []
    for margin in sorted(float(value) for value in margin_grid):
        predictions: list[dict[str, Any]] = []
        for inner_held in sorted(outer_training_sources):
            fitted = fit_source_excluded_threshold(
                held_out_source=inner_held,
                all_sources=list(outer_training_sources),
                registry=registry,
                comparisons=comparisons,
                margin_multiplier=margin,
                logical_bit_length=logical_bit_length,
            )
            predictions.extend(
                predict_source(
                    source_id=inner_held,
                    registry=registry,
                    comparisons=comparisons,
                    threshold=float(fitted["threshold"]),
                )
            )
        metrics = binary_classification_metrics(predictions)
        candidates.append({"margin_multiplier": margin, **metrics})

    def ranking(candidate: dict[str, Any]) -> tuple[float, float, float]:
        balanced = float(candidate["balanced_accuracy"])
        f1 = float(candidate["f1"])
        return (
            balanced if math.isfinite(balanced) else -math.inf,
            f1 if math.isfinite(f1) else -math.inf,
            -float(candidate["margin_multiplier"]),
        )

    selected = max(candidates, key=ranking)
    return float(selected["margin_multiplier"]), candidates


def run_outer_source_evaluation(
    *,
    registry: Sequence[dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    margin_grid: Sequence[float],
    logical_bit_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run complete nested source-wise pHash thresholding and held-out evaluation."""

    sources = sorted({str(row["source_id"]) for row in registry})
    predictions: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []
    for held_out in sources:
        outer_training = [source for source in sources if source != held_out]
        selected_margin, inner_candidates = select_margin_inner_sourcewise(
            outer_training_sources=outer_training,
            registry=registry,
            comparisons=comparisons,
            margin_grid=margin_grid,
            logical_bit_length=logical_bit_length,
        )
        fitted = fit_source_excluded_threshold(
            held_out_source=held_out,
            all_sources=sources,
            registry=registry,
            comparisons=comparisons,
            margin_multiplier=selected_margin,
            logical_bit_length=logical_bit_length,
        )
        fitted["fold_id"] = f"LOSO_{held_out.replace('SRC_', '')}"
        fitted["inner_margin_candidates"] = inner_candidates
        thresholds.append(fitted)
        predictions.extend(
            predict_source(
                source_id=held_out,
                registry=registry,
                comparisons=comparisons,
                threshold=float(fitted["threshold"]),
            )
        )
    return predictions, thresholds
