"""Tests for segment-ID based digest alignment."""

from __future__ import annotations

import pytest

from src.verification.segment_alignment import SegmentAlignmentError, SegmentDescriptor, align_segments


def _segments(ids: list[int]) -> list[SegmentDescriptor]:
    return [
        SegmentDescriptor(segment_id=segment_id, start_time_microseconds=segment_id * 5_000_000, end_time_microseconds=(segment_id + 1) * 5_000_000)
        for segment_id in ids
    ]


def test_matching_segment_ids_and_deterministic_ordering() -> None:
    result = align_segments(_segments([0, 1, 2]), _segments([0, 1, 2]))
    assert result.alignment_valid
    assert result.comparison_complete
    assert result.matched_pairs == ((0, 0, 0), (1, 1, 1), (2, 2, 2))
    assert [record.segment_id for record in result.records] == [0, 1, 2]


def test_reordered_query_rows_align_by_segment_id() -> None:
    result = align_segments(_segments([0, 1]), _segments([1, 0]))
    assert result.alignment_valid
    assert result.matched_pairs == ((0, 0, 1), (1, 1, 0))


def test_missing_and_extra_segments_are_recorded() -> None:
    result = align_segments(_segments([0, 1]), _segments([1, 2]))
    states = {record.segment_id: record.state for record in result.records}
    assert states[0] == "missing_in_query"
    assert states[1] == "matched"
    assert states[2] == "extra_in_query"
    assert result.missing_segment_count == 1
    assert result.extra_segment_count == 1
    assert not result.alignment_valid
    assert not result.comparison_complete


def test_duplicate_segment_id_rejected() -> None:
    with pytest.raises(SegmentAlignmentError, match="duplicate"):
        align_segments(_segments([0, 0]), _segments([0]))
    with pytest.raises(SegmentAlignmentError, match="duplicate"):
        align_segments(_segments([0]), _segments([0, 0]))


def test_timestamp_mismatch_recorded() -> None:
    query = [SegmentDescriptor(segment_id=0, start_time_microseconds=2_000, end_time_microseconds=5_002_000)]
    result = align_segments(_segments([0]), query, timestamp_tolerance_microseconds=1000)
    assert result.timestamp_mismatch_count == 1
    assert result.records[0].state == "timestamp_mismatch"
    assert not result.alignment_valid
