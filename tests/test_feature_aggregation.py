"""Tests for segment feature aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.features.aggregation import SegmentAggregationError, aggregate_segment_embeddings
from src.features.resnet_features import FrameFeatureRecord


def _record(segment_id: int, frame_index: int, row: int) -> FrameFeatureRecord:
    return FrameFeatureRecord(
        video_id="T001",
        segment_id=segment_id,
        frame_index=frame_index,
        requested_timestamp_seconds=segment_id * 5 + frame_index + 0.5,
        actual_timestamp_seconds=segment_id * 5 + frame_index + 0.5,
        frame_path=f"/tmp/frame_{segment_id}_{frame_index}.jpg",
        embedding_row_index=row,
        embedding_dimension=2,
        original_embedding_norm=1.0,
        normalized_embedding_norm=1.0,
        extraction_success=True,
    )


def _segment_manifest(path: Path, expected_count: int = 2, include_empty: bool = False) -> Path:
    segments = [
        {
            "segment_id": 0,
            "start_time_seconds": 0.0,
            "end_time_seconds": 5.0,
            "expected_sample_count": expected_count,
        },
        {
            "segment_id": 1,
            "start_time_seconds": 5.0,
            "end_time_seconds": 10.0,
            "expected_sample_count": expected_count,
        },
    ]
    if include_empty:
        segments.append(
            {
                "segment_id": 2,
                "start_time_seconds": 10.0,
                "end_time_seconds": 15.0,
                "expected_sample_count": expected_count,
            }
        )
    path.write_text(json.dumps({"segments": segments}), encoding="utf-8")
    return path


def test_segment_aggregation_mean_std_combined_and_order(tmp_path: Path) -> None:
    """Aggregation computes population standard deviation and deterministic segment order."""

    embeddings = np.asarray(
        [
            [1.0, 3.0],
            [3.0, 7.0],
            [10.0, 12.0],
            [14.0, 18.0],
        ],
        dtype=np.float32,
    )
    records = [_record(1, 0, 2), _record(0, 1, 1), _record(1, 1, 3), _record(0, 0, 0)]
    manifest_path = _segment_manifest(tmp_path / "segments.json")

    result = aggregate_segment_embeddings(
        frame_embeddings=embeddings,
        frame_records=records,
        segment_manifest_path=manifest_path,
        expected_dimension=2,
    )

    assert result.segment_ids.tolist() == [0, 1]
    assert np.allclose(result.mean_embeddings, [[2.0, 5.0], [12.0, 15.0]])
    assert np.allclose(result.std_embeddings, [[1.0, 2.0], [2.0, 3.0]])
    assert result.combined_embeddings.shape == (2, 4)
    assert result.records[0].combined_embedding_dimension == 4
    assert not result.warnings


def test_partial_frame_warning(tmp_path: Path) -> None:
    """Segments with fewer frames than expected are processed with a warning."""

    embeddings = np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    records = [_record(0, 0, 0), _record(1, 0, 1)]
    manifest_path = _segment_manifest(tmp_path / "segments.json", expected_count=2)

    result = aggregate_segment_embeddings(
        frame_embeddings=embeddings,
        frame_records=records,
        segment_manifest_path=manifest_path,
        expected_dimension=2,
    )

    assert len(result.warnings) == 2
    assert [record.failed_or_missing_frames for record in result.records] == [1, 1]


def test_zero_frame_segment_failure(tmp_path: Path) -> None:
    """A segment with zero valid frame embeddings fails clearly."""

    embeddings = np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    records = [_record(0, 0, 0), _record(1, 0, 1)]
    manifest_path = _segment_manifest(tmp_path / "segments.json", include_empty=True)

    with pytest.raises(SegmentAggregationError, match="zero valid"):
        aggregate_segment_embeddings(
            frame_embeddings=embeddings,
            frame_records=records,
            segment_manifest_path=manifest_path,
            expected_dimension=2,
        )
