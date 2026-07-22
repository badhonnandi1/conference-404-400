"""Segment-level aggregation of frame embeddings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.features.resnet_features import FrameFeatureRecord


class SegmentAggregationError(RuntimeError):
    """Raised when segment feature aggregation cannot continue."""


@dataclass(frozen=True)
class SegmentFeatureRecord:
    """Serializable segment-level feature aggregation record."""

    segment_id: int
    start_time_seconds: float | None
    end_time_seconds: float | None
    expected_sampled_frames: int | None
    successfully_used_frames: int
    failed_or_missing_frames: int
    mean_embedding_row_index: int
    standard_deviation_embedding_row_index: int
    combined_embedding_row_index: int
    mean_embedding_dimension: int
    standard_deviation_embedding_dimension: int
    combined_embedding_dimension: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable segment feature record."""

        return asdict(self)


@dataclass(frozen=True)
class SegmentAggregationResult:
    """Aggregated segment arrays and records."""

    segment_ids: np.ndarray
    mean_embeddings: np.ndarray
    std_embeddings: np.ndarray
    combined_embeddings: np.ndarray
    records: list[SegmentFeatureRecord]
    warnings: list[str]
    failures: list[str]


def load_segment_context(segment_manifest_path: str | Path | None) -> dict[int, dict[str, Any]]:
    """Load segment timing and expected sample counts when a manifest is available."""

    if segment_manifest_path is None:
        return {}
    path = Path(segment_manifest_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    context: dict[int, dict[str, Any]] = {}
    for segment in manifest.get("segments", []):
        context[int(segment["segment_id"])] = segment
    return context


def aggregate_segment_embeddings(
    frame_embeddings: np.ndarray,
    frame_records: list[FrameFeatureRecord],
    segment_manifest_path: str | Path | None = None,
    expected_dimension: int = 512,
) -> SegmentAggregationResult:
    """Aggregate frame embeddings into mean, standard deviation, and combined arrays."""

    if frame_embeddings.ndim != 2 or frame_embeddings.shape[1] != expected_dimension:
        raise SegmentAggregationError(
            f"Expected frame embedding shape (n, {expected_dimension}), got {frame_embeddings.shape}."
        )
    if not np.all(np.isfinite(frame_embeddings)):
        raise SegmentAggregationError("Frame embeddings contain non-finite values.")

    context = load_segment_context(segment_manifest_path)
    successful_records = [record for record in frame_records if record.extraction_success]
    by_segment: dict[int, list[FrameFeatureRecord]] = {}
    for record in successful_records:
        if record.embedding_row_index is None:
            continue
        by_segment.setdefault(record.segment_id, []).append(record)

    candidate_segment_ids = sorted(set(context.keys()) | set(by_segment.keys()))
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    combined: list[np.ndarray] = []
    segment_ids: list[int] = []
    records: list[SegmentFeatureRecord] = []
    warnings: list[str] = []
    failures: list[str] = []

    for segment_id in candidate_segment_ids:
        segment_records = sorted(
            by_segment.get(segment_id, []),
            key=lambda record: (record.requested_timestamp_seconds, record.frame_index),
        )
        if not segment_records:
            message = f"Segment {segment_id} has zero valid frame embeddings."
            failures.append(message)
            raise SegmentAggregationError(message)

        row_indices = [record.embedding_row_index for record in segment_records]
        if any(index is None for index in row_indices):
            raise SegmentAggregationError(f"Segment {segment_id} has a missing embedding row index.")
        segment_frame_embeddings = frame_embeddings[[int(index) for index in row_indices]]
        mean_embedding = np.mean(segment_frame_embeddings, axis=0).astype(np.float32)
        std_embedding = np.std(segment_frame_embeddings, axis=0, ddof=0).astype(np.float32)
        combined_embedding = np.concatenate([mean_embedding, std_embedding]).astype(np.float32)

        if not (
            np.all(np.isfinite(mean_embedding))
            and np.all(np.isfinite(std_embedding))
            and np.all(np.isfinite(combined_embedding))
        ):
            raise SegmentAggregationError(f"Segment {segment_id} aggregation produced non-finite values.")

        segment_context = context.get(segment_id, {})
        expected_count = segment_context.get("expected_sample_count")
        failed_or_missing = 0
        if expected_count is not None:
            failed_or_missing = max(0, int(expected_count) - len(segment_records))
            if failed_or_missing > 0:
                warnings.append(
                    f"Segment {segment_id} used {len(segment_records)} of {expected_count} expected frames."
                )

        mean_row = len(means)
        std_row = len(stds)
        combined_row = len(combined)
        segment_ids.append(segment_id)
        means.append(mean_embedding)
        stds.append(std_embedding)
        combined.append(combined_embedding)
        records.append(
            SegmentFeatureRecord(
                segment_id=segment_id,
                start_time_seconds=segment_context.get("start_time_seconds"),
                end_time_seconds=segment_context.get("end_time_seconds"),
                expected_sampled_frames=expected_count,
                successfully_used_frames=len(segment_records),
                failed_or_missing_frames=failed_or_missing,
                mean_embedding_row_index=mean_row,
                standard_deviation_embedding_row_index=std_row,
                combined_embedding_row_index=combined_row,
                mean_embedding_dimension=expected_dimension,
                standard_deviation_embedding_dimension=expected_dimension,
                combined_embedding_dimension=expected_dimension * 2,
            )
        )

    return SegmentAggregationResult(
        segment_ids=np.asarray(segment_ids, dtype=np.int64),
        mean_embeddings=np.vstack(means).astype(np.float32),
        std_embeddings=np.vstack(stds).astype(np.float32),
        combined_embeddings=np.vstack(combined).astype(np.float32),
        records=records,
        warnings=warnings,
        failures=failures,
    )
