"""Focused tests for the official OpenCV segment-level pHash baseline."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from src.baselines.opencv_phash import (
    FrameHash,
    OpenCVPHash,
    compare_video_frame_hashes,
    fit_robust_threshold,
    phash_hamming_distance,
    video_decision,
)
from src.analysis.phash_evaluation import fit_source_excluded_threshold


def _frame(
    video_id: str,
    segment_id: int,
    sample_index: int,
    timestamp: float,
    digest: np.ndarray | None,
) -> FrameHash:
    return FrameHash(
        source_video_id=video_id,
        segment_id=segment_id,
        sample_index=sample_index,
        timestamp_seconds=timestamp,
        frame_path=f"{video_id}_{segment_id}_{sample_index}.png",
        valid=digest is not None,
        hash_bytes=digest,
        failure_reason=None if digest is not None else "test failure",
        opencv_version=cv2.__version__,
        phash_api="test",
        hash_byte_length=8,
        logical_bit_length=64,
    )


def test_opencv_contrib_phash_is_available_and_validated() -> None:
    assert hasattr(cv2, "img_hash")
    assert hasattr(cv2.img_hash, "pHash") or hasattr(cv2.img_hash, "PHash_create")
    hasher = OpenCVPHash()
    assert hasher.hash_byte_length == 8
    assert hasher.logical_bit_length == 64


def test_phash_output_is_deterministic_uint8_and_identical_distance_is_zero() -> None:
    hasher = OpenCVPHash()
    y, x = np.indices((80, 96), dtype=np.int32)
    image = ((x * 7 + y * 11) % 256).astype(np.uint8)
    first = hasher.compute(image)
    second = hasher.compute(image.copy())
    assert first.dtype == np.uint8
    assert first.shape == (8,)
    assert np.array_equal(first, second)
    raw, normalized, bits = phash_hamming_distance(first, second)
    assert raw == 0
    assert normalized == 0.0
    assert bits == 64


def test_normalized_phash_distance_is_in_closed_unit_interval() -> None:
    zeros = np.zeros(8, dtype=np.uint8)
    ones = np.full(8, 255, dtype=np.uint8)
    raw, normalized, bits = phash_hamming_distance(zeros, ones)
    assert raw == 64
    assert bits == 64
    assert normalized == 1.0


def test_segment_aggregation_and_maximum_video_score() -> None:
    zero = np.zeros(8, dtype=np.uint8)
    one_bit = zero.copy()
    one_bit[0] = 1
    four_bits = zero.copy()
    four_bits[0] = 15
    result = compare_video_frame_hashes(
        reference_hashes=[
            _frame("R", 0, 0, 0.5, zero),
            _frame("R", 0, 1, 1.5, zero),
            _frame("R", 1, 0, 5.5, zero),
        ],
        query_hashes=[
            _frame("Q", 0, 0, 0.5, one_bit),
            _frame("Q", 0, 1, 1.5, four_bits),
            _frame("Q", 1, 0, 5.5, zero),
        ],
        reference_segment_ids=[0, 1],
        query_segment_ids=[0, 1],
    )
    assert result["segment_rows"][0]["matched_frame_hash_count"] == 2
    assert result["segment_rows"][0]["segment_phash_score"] == pytest.approx(2.5 / 64)
    assert result["segment_rows"][0]["median_frame_distance"] == pytest.approx(2.5 / 64)
    assert result["segment_rows"][0]["maximum_frame_distance"] == pytest.approx(4 / 64)
    assert result["video_phash_score"] == pytest.approx(2.5 / 64)


def test_timestamp_correspondence_and_missing_frame_pair_are_not_silent_zeroes() -> None:
    digest = np.zeros(8, dtype=np.uint8)
    result = compare_video_frame_hashes(
        reference_hashes=[_frame("R", 0, 0, 0.5, digest)],
        query_hashes=[_frame("Q", 0, 0, 0.6, digest)],
        reference_segment_ids=[0],
        query_segment_ids=[0],
    )
    segment = result["segment_rows"][0]
    assert not segment["segment_valid"]
    assert segment["segment_phash_score"] is None
    assert segment["missing_frame_hash_count"] == 1
    assert result["invalid_comparison"]
    assert "timestamps" in segment["invalid_reason"]


def test_structural_condition_and_invalid_evidence_fail_closed() -> None:
    digest = np.zeros(8, dtype=np.uint8)
    result = compare_video_frame_hashes(
        reference_hashes=[
            _frame("R", 0, 0, 0.5, digest),
            _frame("R", 1, 0, 5.5, digest),
        ],
        query_hashes=[_frame("Q", 0, 0, 0.5, digest)],
        reference_segment_ids=[0, 1],
        query_segment_ids=[0],
    )
    assert result["structural_issue"]
    assert result["missing_segment_ids"] == [1]
    decision = video_decision(
        video_score=result["video_phash_score"],
        threshold=0.25,
        structural_issue=result["structural_issue"],
        invalid_comparison=result["invalid_comparison"],
        segment_rows=result["segment_rows"],
    )
    assert decision["observed_label"] == "abnormal"
    assert decision["structural_abnormal"]


def test_video_decision_uses_strict_greater_than() -> None:
    segment_rows = [{"segment_id": 0, "segment_valid": True, "segment_phash_score": 0.25}]
    equal = video_decision(
        video_score=0.25,
        threshold=0.25,
        structural_issue=False,
        invalid_comparison=False,
        segment_rows=segment_rows,
    )
    above = video_decision(
        video_score=math.nextafter(0.25, 1.0),
        threshold=0.25,
        structural_issue=False,
        invalid_comparison=False,
        segment_rows=segment_rows,
    )
    assert equal["observed_label"] == "normal"
    assert above["observed_label"] == "abnormal"


def test_robust_threshold_uses_mad_and_one_bit_resolution() -> None:
    fitted = fit_robust_threshold(
        [0.0, 1 / 64, 2 / 64],
        margin_multiplier=1.5,
        logical_bit_length=64,
    )
    assert fitted["maximum_benign_score"] == 2 / 64
    assert fitted["mad"] == 1 / 64
    assert fitted["minimum_score_resolution"] == 1 / 64
    assert fitted["threshold"] == pytest.approx(3.5 / 64)


def test_threshold_fitting_excludes_the_complete_held_out_source() -> None:
    registry = []
    comparisons = {}
    for source_index, source_id in enumerate(("SRC_A", "SRC_B", "SRC_C")):
        for transform_index, transformation in enumerate(
            ("trusted_reference", "avi_conversion", "blur")
        ):
            video_id = f"{source_id}_{transform_index}"
            registry.append(
                {
                    "source_id": source_id,
                    "video_id": video_id,
                    "transformation_type": transformation,
                }
            )
            comparisons[video_id] = {
                "video_phash_score": (source_index + transform_index) / 64,
                "segment_rows": [{"segment_valid": True}],
            }
    fitted = fit_source_excluded_threshold(
        held_out_source="SRC_C",
        all_sources=["SRC_A", "SRC_B", "SRC_C"],
        registry=registry,
        comparisons=comparisons,
        margin_multiplier=1.5,
        logical_bit_length=64,
    )
    assert fitted["training_sources"] == ["SRC_A", "SRC_B"]
    assert not fitted["held_out_used_for_fitting"]
    assert all(not video_id.startswith("SRC_C") for video_id in fitted["benign_calibration_video_ids"])
