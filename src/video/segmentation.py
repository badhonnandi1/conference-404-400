"""Logical timestamp-based video segmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from src.video.metadata import ExistingOutputError, InvalidDurationError, VideoMetadata


@dataclass(frozen=True)
class SegmentRecord:
    """A logical, timestamp-based video segment."""

    video_id: str
    segment_id: int
    start_time_seconds: float
    end_time_seconds: float
    duration_seconds: float
    is_complete: bool
    expected_sample_count: int
    source_video_path: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable segment record."""

        return asdict(self)


@dataclass(frozen=True)
class SegmentManifest:
    """Serializable manifest containing all logical segment records."""

    video_id: str
    source_video_path: str
    video_metadata_reference: str | None
    segment_duration_seconds: float
    sample_frames_per_second: float
    incomplete_segment_policy: str
    number_complete_segments: int
    processed_duration_seconds: float
    discarded_duration_seconds: float
    segments: list[SegmentRecord]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable segment manifest."""

        payload = asdict(self)
        payload["segments"] = [segment.to_dict() for segment in self.segments]
        return payload


def expected_sample_count(duration_seconds: float, sample_frames_per_second: float) -> int:
    """Return the number of samples expected for a segment duration."""

    if duration_seconds <= 0 or sample_frames_per_second <= 0:
        return 0
    return int(math.floor(duration_seconds * sample_frames_per_second + 1e-9))


def create_segment_manifest(
    video_id: str,
    metadata: VideoMetadata,
    segment_duration_seconds: float = 5.0,
    sample_frames_per_second: float = 1.0,
    incomplete_segment_policy: str = "discard",
    video_metadata_reference: str | Path | None = None,
) -> SegmentManifest:
    """Create non-overlapping logical segment records from video metadata."""

    if segment_duration_seconds <= 0:
        raise InvalidDurationError("Segment duration must be greater than zero.")
    if sample_frames_per_second <= 0:
        raise InvalidDurationError("Sample frames per second must be greater than zero.")

    duration = float(metadata.duration_seconds)
    if duration <= 0:
        raise InvalidDurationError(
            f"Invalid duration for '{metadata.absolute_path}'. Check that FFprobe can read it."
        )

    policy = incomplete_segment_policy.lower()
    if policy not in {"discard", "keep"}:
        raise ValueError("incomplete_segment_policy must be 'discard' or 'keep'.")

    complete_count = int(math.floor(duration / segment_duration_seconds + 1e-9))
    processed_duration = complete_count * segment_duration_seconds
    remainder = max(0.0, duration - processed_duration)
    discarded_duration = remainder if policy == "discard" else 0.0

    segments: list[SegmentRecord] = []
    for segment_id in range(complete_count):
        start = segment_id * segment_duration_seconds
        end = start + segment_duration_seconds
        segments.append(
            SegmentRecord(
                video_id=video_id,
                segment_id=segment_id,
                start_time_seconds=round(start, 6),
                end_time_seconds=round(end, 6),
                duration_seconds=round(segment_duration_seconds, 6),
                is_complete=True,
                expected_sample_count=expected_sample_count(
                    segment_duration_seconds, sample_frames_per_second
                ),
                source_video_path=metadata.absolute_path,
            )
        )

    if policy == "keep" and remainder > 1e-9:
        start = processed_duration
        segments.append(
            SegmentRecord(
                video_id=video_id,
                segment_id=complete_count,
                start_time_seconds=round(start, 6),
                end_time_seconds=round(duration, 6),
                duration_seconds=round(remainder, 6),
                is_complete=False,
                expected_sample_count=expected_sample_count(remainder, sample_frames_per_second),
                source_video_path=metadata.absolute_path,
            )
        )

    return SegmentManifest(
        video_id=video_id,
        source_video_path=metadata.absolute_path,
        video_metadata_reference=str(video_metadata_reference) if video_metadata_reference else None,
        segment_duration_seconds=segment_duration_seconds,
        sample_frames_per_second=sample_frames_per_second,
        incomplete_segment_policy=policy,
        number_complete_segments=complete_count,
        processed_duration_seconds=round(processed_duration, 6),
        discarded_duration_seconds=round(discarded_duration, 6),
        segments=segments,
    )


def save_segment_manifest(
    manifest: SegmentManifest, output_path: str | Path, overwrite: bool = False
) -> Path:
    """Write a segment manifest as formatted JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise ExistingOutputError(
            f"Segment manifest already exists: {path}. Use --overwrite to replace it."
        )
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2)
        handle.write("\n")
    return path


def load_segment_manifest(path: str | Path) -> SegmentManifest:
    """Load a segment manifest from JSON."""

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    segments = [SegmentRecord(**segment) for segment in data.get("segments", [])]
    data["segments"] = segments
    return SegmentManifest(**data)
