"""Interpretable temporal frame-difference feature extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from src.features.temporal_sampling import (
    TemporalFrameRecord,
    TemporalSamplingConfig,
    generate_temporal_timestamps,
)
from src.video.segmentation import SegmentManifest


class TemporalFeatureError(RuntimeError):
    """Raised when temporal feature extraction or aggregation fails."""


PAIR_FEATURE_NAMES = [
    "mean_absolute_difference",
    "absdiff_standard_deviation",
    "normalized_rmse",
    "changed_pixel_ratio",
    "p90_absolute_difference",
    "edge_change_ratio",
]

SEGMENT_AGGREGATIONS = ["mean", "standard_deviation", "maximum"]
SEGMENT_FEATURE_NAMES = [
    f"{prefix}_{name}"
    for prefix in ("mean", "std", "max")
    for name in (
        "mad",
        "absdiff_std",
        "rmse",
        "changed_ratio",
        "p90",
        "edge_change",
    )
]


@dataclass(frozen=True)
class TemporalPairRecord:
    """Serializable pair-level temporal feature record."""

    video_id: str
    segment_id: int
    pair_index: int
    first_requested_timestamp_seconds: float
    second_requested_timestamp_seconds: float
    first_actual_timestamp_seconds: float | None
    second_actual_timestamp_seconds: float | None
    temporal_gap_seconds: float
    feature_row_index: int | None
    feature_values: list[float] | None
    success: bool
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable pair record."""

        return asdict(self)


@dataclass(frozen=True)
class TemporalSegmentRecord:
    """Serializable segment-level temporal aggregation record."""

    video_id: str
    segment_id: int
    start_time_seconds: float
    end_time_seconds: float
    expected_temporal_frame_count: int
    decoded_temporal_frame_count: int
    failed_temporal_frame_count: int
    expected_pair_count: int
    successful_pair_count: int
    failed_pair_count: int
    missing_pair_count: int
    segment_feature_row_index: int | None
    segment_feature_dimension: int
    maximum_discontinuity_pair_index: int | None
    maximum_discontinuity_timestamp_seconds: float | None
    success: bool
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable segment record."""

        return asdict(self)


@dataclass(frozen=True)
class TemporalFeatureResult:
    """Temporal pair and segment feature arrays plus records."""

    pair_features: np.ndarray
    pair_records: list[TemporalPairRecord]
    pair_segment_ids: np.ndarray
    pair_indices: np.ndarray
    pair_start_timestamps: np.ndarray
    pair_end_timestamps: np.ndarray
    segment_ids: np.ndarray
    segment_features: np.ndarray
    segment_records: list[TemporalSegmentRecord]
    segment_successful_pair_counts: np.ndarray
    segment_max_discontinuity_pair_indices: np.ndarray
    segment_max_discontinuity_timestamps: np.ndarray
    warnings: list[str]
    failures: list[str]


def _edge_magnitude(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        frame = np.mean(frame, axis=2)
    sobel_x = cv2.Sobel(frame.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(frame.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(sobel_x, sobel_y)


def calculate_pair_features(
    first_frame: np.ndarray,
    second_frame: np.ndarray,
    changed_pixel_threshold: float = 20.0,
) -> np.ndarray:
    """Calculate six interpretable temporal features for a consecutive frame pair."""

    if first_frame.shape != second_frame.shape:
        raise TemporalFeatureError(
            f"Temporal frames must have matching shapes, got {first_frame.shape} and {second_frame.shape}."
        )
    threshold = changed_pixel_threshold / 255.0
    diff = second_frame.astype(np.float32) - first_frame.astype(np.float32)
    absdiff = np.abs(diff)
    edge_change = np.abs(_edge_magnitude(second_frame) - _edge_magnitude(first_frame))
    features = np.asarray(
        [
            float(np.mean(absdiff)),
            float(np.std(absdiff, ddof=0)),
            float(np.sqrt(np.mean(np.square(diff)))),
            float(np.mean(absdiff > threshold)),
            float(np.percentile(absdiff, 90)),
            float(np.mean(edge_change > threshold)),
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(features)):
        raise TemporalFeatureError("Pair feature calculation produced non-finite values.")
    for ratio_index in (3, 5):
        features[ratio_index] = np.clip(features[ratio_index], 0.0, 1.0)
    return features


def build_temporal_pairs_for_segment(
    frames: list[TemporalFrameRecord],
    config: TemporalSamplingConfig,
    starting_row_index: int = 0,
) -> tuple[np.ndarray, list[TemporalPairRecord]]:
    """Create pair records from adjacent temporal frame requests within one segment."""

    sorted_frames = sorted(frames, key=lambda record: record.frame_index)
    features: list[np.ndarray] = []
    records: list[TemporalPairRecord] = []
    for pair_index in range(max(0, len(sorted_frames) - 1)):
        first = sorted_frames[pair_index]
        second = sorted_frames[pair_index + 1]
        gap = round(second.requested_timestamp_seconds - first.requested_timestamp_seconds, 6)
        if not first.success or not second.success or first.image is None or second.image is None:
            records.append(
                TemporalPairRecord(
                    video_id=first.video_id,
                    segment_id=first.segment_id,
                    pair_index=pair_index,
                    first_requested_timestamp_seconds=first.requested_timestamp_seconds,
                    second_requested_timestamp_seconds=second.requested_timestamp_seconds,
                    first_actual_timestamp_seconds=first.actual_timestamp_seconds,
                    second_actual_timestamp_seconds=second.actual_timestamp_seconds,
                    temporal_gap_seconds=gap,
                    feature_row_index=None,
                    feature_values=None,
                    success=False,
                    error_message="One or both temporal frames failed to decode.",
                )
            )
            continue
        try:
            feature_values = calculate_pair_features(
                first.image,
                second.image,
                changed_pixel_threshold=config.changed_pixel_threshold,
            )
        except TemporalFeatureError as exc:
            records.append(
                TemporalPairRecord(
                    video_id=first.video_id,
                    segment_id=first.segment_id,
                    pair_index=pair_index,
                    first_requested_timestamp_seconds=first.requested_timestamp_seconds,
                    second_requested_timestamp_seconds=second.requested_timestamp_seconds,
                    first_actual_timestamp_seconds=first.actual_timestamp_seconds,
                    second_actual_timestamp_seconds=second.actual_timestamp_seconds,
                    temporal_gap_seconds=gap,
                    feature_row_index=None,
                    feature_values=None,
                    success=False,
                    error_message=str(exc),
                )
            )
            continue
        row_index = starting_row_index + len(features)
        features.append(feature_values)
        records.append(
            TemporalPairRecord(
                video_id=first.video_id,
                segment_id=first.segment_id,
                pair_index=pair_index,
                first_requested_timestamp_seconds=first.requested_timestamp_seconds,
                second_requested_timestamp_seconds=second.requested_timestamp_seconds,
                first_actual_timestamp_seconds=first.actual_timestamp_seconds,
                second_actual_timestamp_seconds=second.actual_timestamp_seconds,
                temporal_gap_seconds=gap,
                feature_row_index=row_index,
                feature_values=[float(value) for value in feature_values],
                success=True,
            )
        )
    if features:
        feature_array = np.vstack(features).astype(np.float32)
    else:
        feature_array = np.empty((0, len(PAIR_FEATURE_NAMES)), dtype=np.float32)
    return feature_array, records


def aggregate_temporal_segment_features(pair_features: np.ndarray) -> np.ndarray:
    """Aggregate successful pair features into the 18-dimensional segment vector."""

    if pair_features.ndim != 2 or pair_features.shape[1] != len(PAIR_FEATURE_NAMES):
        raise TemporalFeatureError(
            f"Expected pair feature shape (n, {len(PAIR_FEATURE_NAMES)}), got {pair_features.shape}."
        )
    if pair_features.shape[0] == 0:
        raise TemporalFeatureError("Cannot aggregate a segment with zero valid temporal pairs.")
    values = np.concatenate(
        [
            np.mean(pair_features, axis=0),
            np.std(pair_features, axis=0, ddof=0),
            np.max(pair_features, axis=0),
        ]
    ).astype(np.float32)
    if values.shape != (len(SEGMENT_FEATURE_NAMES),):
        raise TemporalFeatureError(f"Unexpected segment feature shape: {values.shape}")
    if not np.all(np.isfinite(values)):
        raise TemporalFeatureError("Segment temporal aggregation produced non-finite values.")
    return values


def extract_temporal_features_from_frames(
    video_id: str,
    segment_manifest: SegmentManifest,
    frame_records: list[TemporalFrameRecord],
    config: TemporalSamplingConfig,
) -> TemporalFeatureResult:
    """Build pair and segment temporal features from decoded temporal frames."""

    frames_by_segment: dict[int, list[TemporalFrameRecord]] = {}
    for frame in frame_records:
        frames_by_segment.setdefault(frame.segment_id, []).append(frame)

    all_pair_features: list[np.ndarray] = []
    pair_records: list[TemporalPairRecord] = []
    segment_features: list[np.ndarray] = []
    segment_records: list[TemporalSegmentRecord] = []
    segment_ids: list[int] = []
    segment_success_counts: list[int] = []
    max_pair_indices: list[int] = []
    max_timestamps: list[float] = []
    warnings: list[str] = []
    failures: list[str] = []
    next_pair_row = 0

    for segment in segment_manifest.segments:
        if not segment.is_complete:
            continue
        segment_frames = sorted(
            frames_by_segment.get(segment.segment_id, []),
            key=lambda record: record.frame_index,
        )
        expected_frame_count = len(
            generate_temporal_timestamps(
                segment.start_time_seconds,
                segment.end_time_seconds,
                config.sample_fps,
            )
        )
        expected_pair_count = max(0, expected_frame_count - 1)
        decoded_count = sum(1 for frame in segment_frames if frame.success)
        failed_frame_count = len(segment_frames) - decoded_count
        segment_pair_features, segment_pair_records = build_temporal_pairs_for_segment(
            segment_frames,
            config,
            starting_row_index=next_pair_row,
        )
        next_pair_row += segment_pair_features.shape[0]
        pair_records.extend(segment_pair_records)
        if segment_pair_features.size:
            all_pair_features.append(segment_pair_features)

        successful_pair_count = segment_pair_features.shape[0]
        failed_pair_count = sum(1 for record in segment_pair_records if not record.success)
        missing_pair_count = max(0, expected_pair_count - successful_pair_count)
        if successful_pair_count == 0:
            message = f"Segment {segment.segment_id} has zero valid temporal pairs."
            failures.append(message)
            segment_records.append(
                TemporalSegmentRecord(
                    video_id=video_id,
                    segment_id=segment.segment_id,
                    start_time_seconds=segment.start_time_seconds,
                    end_time_seconds=segment.end_time_seconds,
                    expected_temporal_frame_count=expected_frame_count,
                    decoded_temporal_frame_count=decoded_count,
                    failed_temporal_frame_count=failed_frame_count,
                    expected_pair_count=expected_pair_count,
                    successful_pair_count=successful_pair_count,
                    failed_pair_count=failed_pair_count,
                    missing_pair_count=missing_pair_count,
                    segment_feature_row_index=None,
                    segment_feature_dimension=len(SEGMENT_FEATURE_NAMES),
                    maximum_discontinuity_pair_index=None,
                    maximum_discontinuity_timestamp_seconds=None,
                    success=False,
                    error_message=message,
                )
            )
            raise TemporalFeatureError(message)

        if missing_pair_count > 0:
            warnings.append(
                f"Segment {segment.segment_id} used {successful_pair_count} of "
                f"{expected_pair_count} expected temporal pairs."
            )

        segment_feature = aggregate_temporal_segment_features(segment_pair_features)
        discontinuity_scores = segment_pair_features[:, 0]
        max_local_index = int(np.argmax(discontinuity_scores))
        successful_records = [record for record in segment_pair_records if record.success]
        max_pair_record = successful_records[max_local_index]
        max_timestamp = round(
            (
                max_pair_record.first_requested_timestamp_seconds
                + max_pair_record.second_requested_timestamp_seconds
            )
            / 2.0,
            6,
        )

        segment_row = len(segment_features)
        segment_ids.append(segment.segment_id)
        segment_features.append(segment_feature)
        segment_success_counts.append(successful_pair_count)
        max_pair_indices.append(max_pair_record.pair_index)
        max_timestamps.append(max_timestamp)
        segment_records.append(
            TemporalSegmentRecord(
                video_id=video_id,
                segment_id=segment.segment_id,
                start_time_seconds=segment.start_time_seconds,
                end_time_seconds=segment.end_time_seconds,
                expected_temporal_frame_count=expected_frame_count,
                decoded_temporal_frame_count=decoded_count,
                failed_temporal_frame_count=failed_frame_count,
                expected_pair_count=expected_pair_count,
                successful_pair_count=successful_pair_count,
                failed_pair_count=failed_pair_count,
                missing_pair_count=missing_pair_count,
                segment_feature_row_index=segment_row,
                segment_feature_dimension=len(SEGMENT_FEATURE_NAMES),
                maximum_discontinuity_pair_index=max_pair_record.pair_index,
                maximum_discontinuity_timestamp_seconds=max_timestamp,
                success=True,
            )
        )

    pair_feature_array = (
        np.vstack(all_pair_features).astype(np.float32)
        if all_pair_features
        else np.empty((0, len(PAIR_FEATURE_NAMES)), dtype=np.float32)
    )
    successful_pair_records = [record for record in pair_records if record.success]
    return TemporalFeatureResult(
        pair_features=pair_feature_array,
        pair_records=pair_records,
        pair_segment_ids=np.asarray([record.segment_id for record in successful_pair_records], dtype=np.int64),
        pair_indices=np.asarray([record.pair_index for record in successful_pair_records], dtype=np.int64),
        pair_start_timestamps=np.asarray(
            [record.first_requested_timestamp_seconds for record in successful_pair_records],
            dtype=np.float64,
        ),
        pair_end_timestamps=np.asarray(
            [record.second_requested_timestamp_seconds for record in successful_pair_records],
            dtype=np.float64,
        ),
        segment_ids=np.asarray(segment_ids, dtype=np.int64),
        segment_features=(
            np.vstack(segment_features).astype(np.float32)
            if segment_features
            else np.empty((0, len(SEGMENT_FEATURE_NAMES)), dtype=np.float32)
        ),
        segment_records=segment_records,
        segment_successful_pair_counts=np.asarray(segment_success_counts, dtype=np.int64),
        segment_max_discontinuity_pair_indices=np.asarray(max_pair_indices, dtype=np.int64),
        segment_max_discontinuity_timestamps=np.asarray(max_timestamps, dtype=np.float64),
        warnings=warnings,
        failures=failures,
    )
