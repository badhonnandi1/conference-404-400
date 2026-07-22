"""Segment-ID alignment for reference/query digest comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AlignmentState = Literal["matched", "missing_in_query", "extra_in_query", "timestamp_mismatch"]


class SegmentAlignmentError(RuntimeError):
    """Raised when segment alignment inputs are structurally invalid."""


@dataclass(frozen=True)
class SegmentDescriptor:
    """Minimal stable segment metadata needed for digest comparison."""

    segment_id: int
    start_time_microseconds: int
    end_time_microseconds: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-friendly mapping."""

        return {
            "segment_id": self.segment_id,
            "start_time_microseconds": self.start_time_microseconds,
            "end_time_microseconds": self.end_time_microseconds,
        }


@dataclass(frozen=True)
class AlignmentRecord:
    """One segment alignment result."""

    segment_id: int
    state: AlignmentState
    reference_index: int | None
    query_index: int | None
    reference_start_time_microseconds: int | None = None
    reference_end_time_microseconds: int | None = None
    query_start_time_microseconds: int | None = None
    query_end_time_microseconds: int | None = None
    start_time_delta_microseconds: int | None = None
    end_time_delta_microseconds: int | None = None

    def to_dict(self) -> dict[str, int | str | None]:
        """Return a JSON-friendly mapping."""

        return {
            "segment_id": self.segment_id,
            "state": self.state,
            "reference_index": self.reference_index,
            "query_index": self.query_index,
            "reference_start_time_microseconds": self.reference_start_time_microseconds,
            "reference_end_time_microseconds": self.reference_end_time_microseconds,
            "query_start_time_microseconds": self.query_start_time_microseconds,
            "query_end_time_microseconds": self.query_end_time_microseconds,
            "start_time_delta_microseconds": self.start_time_delta_microseconds,
            "end_time_delta_microseconds": self.end_time_delta_microseconds,
        }


@dataclass(frozen=True)
class SegmentAlignmentResult:
    """Complete alignment result for reference and query segments."""

    records: tuple[AlignmentRecord, ...]
    matched_pairs: tuple[tuple[int, int, int], ...]
    reference_segment_count: int
    query_segment_count: int
    matched_segment_count: int
    missing_segment_count: int
    extra_segment_count: int
    timestamp_mismatch_count: int
    alignment_valid: bool
    comparison_complete: bool
    alignment_mode: str

    def to_manifest(self) -> dict[str, object]:
        """Return a JSON-friendly alignment summary and record list."""

        return {
            "alignment_mode": self.alignment_mode,
            "reference_segment_count": self.reference_segment_count,
            "query_segment_count": self.query_segment_count,
            "matched_segment_count": self.matched_segment_count,
            "missing_segment_count": self.missing_segment_count,
            "extra_segment_count": self.extra_segment_count,
            "timestamp_mismatch_count": self.timestamp_mismatch_count,
            "alignment_valid": self.alignment_valid,
            "comparison_complete": self.comparison_complete,
            "records": [record.to_dict() for record in self.records],
        }


def _index_by_segment_id(segments: list[SegmentDescriptor], label: str) -> dict[int, int]:
    seen: dict[int, int] = {}
    duplicates: list[int] = []
    for index, segment in enumerate(segments):
        if segment.segment_id in seen:
            duplicates.append(segment.segment_id)
        seen[segment.segment_id] = index
    if duplicates:
        raise SegmentAlignmentError(f"{label} contains duplicate segment IDs: {sorted(set(duplicates))}.")
    return seen


def align_segments(
    reference_segments: list[SegmentDescriptor],
    query_segments: list[SegmentDescriptor],
    timestamp_tolerance_microseconds: int = 1000,
    alignment_mode: str = "strict",
) -> SegmentAlignmentResult:
    """Align reference and query segments by segment ID and validate timestamps."""

    if timestamp_tolerance_microseconds < 0:
        raise SegmentAlignmentError("Timestamp tolerance must be non-negative.")
    if alignment_mode not in {"strict", "partial"}:
        raise SegmentAlignmentError("alignment_mode must be 'strict' or 'partial'.")
    reference_index = _index_by_segment_id(reference_segments, "Reference")
    query_index = _index_by_segment_id(query_segments, "Query")

    records: list[AlignmentRecord] = []
    matched_pairs: list[tuple[int, int, int]] = []
    missing_count = 0
    mismatch_count = 0
    for segment_id in sorted(reference_index):
        ref_idx = reference_index[segment_id]
        ref = reference_segments[ref_idx]
        qry_idx = query_index.get(segment_id)
        if qry_idx is None:
            missing_count += 1
            records.append(
                AlignmentRecord(
                    segment_id=segment_id,
                    state="missing_in_query",
                    reference_index=ref_idx,
                    query_index=None,
                    reference_start_time_microseconds=ref.start_time_microseconds,
                    reference_end_time_microseconds=ref.end_time_microseconds,
                )
            )
            continue
        qry = query_segments[qry_idx]
        start_delta = int(qry.start_time_microseconds - ref.start_time_microseconds)
        end_delta = int(qry.end_time_microseconds - ref.end_time_microseconds)
        if abs(start_delta) > timestamp_tolerance_microseconds or abs(end_delta) > timestamp_tolerance_microseconds:
            mismatch_count += 1
            records.append(
                AlignmentRecord(
                    segment_id=segment_id,
                    state="timestamp_mismatch",
                    reference_index=ref_idx,
                    query_index=qry_idx,
                    reference_start_time_microseconds=ref.start_time_microseconds,
                    reference_end_time_microseconds=ref.end_time_microseconds,
                    query_start_time_microseconds=qry.start_time_microseconds,
                    query_end_time_microseconds=qry.end_time_microseconds,
                    start_time_delta_microseconds=start_delta,
                    end_time_delta_microseconds=end_delta,
                )
            )
            continue
        records.append(
            AlignmentRecord(
                segment_id=segment_id,
                state="matched",
                reference_index=ref_idx,
                query_index=qry_idx,
                reference_start_time_microseconds=ref.start_time_microseconds,
                reference_end_time_microseconds=ref.end_time_microseconds,
                query_start_time_microseconds=qry.start_time_microseconds,
                query_end_time_microseconds=qry.end_time_microseconds,
                start_time_delta_microseconds=start_delta,
                end_time_delta_microseconds=end_delta,
            )
        )
        matched_pairs.append((segment_id, ref_idx, qry_idx))

    extra_count = 0
    for segment_id in sorted(set(query_index) - set(reference_index)):
        extra_count += 1
        qry_idx = query_index[segment_id]
        qry = query_segments[qry_idx]
        records.append(
            AlignmentRecord(
                segment_id=segment_id,
                state="extra_in_query",
                reference_index=None,
                query_index=qry_idx,
                query_start_time_microseconds=qry.start_time_microseconds,
                query_end_time_microseconds=qry.end_time_microseconds,
            )
        )

    alignment_valid = missing_count == 0 and extra_count == 0 and mismatch_count == 0
    comparison_complete = alignment_valid if alignment_mode == "strict" else bool(matched_pairs)
    return SegmentAlignmentResult(
        records=tuple(records),
        matched_pairs=tuple(matched_pairs),
        reference_segment_count=len(reference_segments),
        query_segment_count=len(query_segments),
        matched_segment_count=len(matched_pairs),
        missing_segment_count=missing_count,
        extra_segment_count=extra_count,
        timestamp_mismatch_count=mismatch_count,
        alignment_valid=alignment_valid,
        comparison_complete=comparison_complete,
        alignment_mode=alignment_mode,
    )
