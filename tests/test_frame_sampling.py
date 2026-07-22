"""Tests for deterministic frame sampling helpers."""

from __future__ import annotations

import json

from src.video.frame_sampling import (
    FrameSampleRecord,
    FrameSamplingManifest,
    frame_filename,
    generate_sample_timestamps,
    load_frame_sampling_manifest,
    save_frame_sampling_manifest,
)
from src.video.segmentation import SegmentRecord


def _segment(start: float = 0.0, end: float = 5.0) -> SegmentRecord:
    return SegmentRecord(
        video_id="V001",
        segment_id=0,
        start_time_seconds=start,
        end_time_seconds=end,
        duration_seconds=end - start,
        is_complete=True,
        expected_sample_count=5,
        source_video_path="/tmp/sample.mp4",
    )


def test_sample_timestamp_generation_for_five_second_segment() -> None:
    """A five-second segment sampled at 1 FPS yields mid-interval timestamps."""

    timestamps = generate_sample_timestamps(_segment(), sample_frames_per_second=1)

    assert timestamps == [0.5, 1.5, 2.5, 3.5, 4.5]
    assert len(timestamps) == 5


def test_sample_timestamps_remain_inside_later_segment() -> None:
    """Generated timestamps remain inside shifted segment boundaries."""

    segment = _segment(start=5.0, end=10.0)
    timestamps = generate_sample_timestamps(segment, sample_frames_per_second=1)

    assert timestamps == [5.5, 6.5, 7.5, 8.5, 9.5]
    assert all(segment.start_time_seconds < timestamp < segment.end_time_seconds for timestamp in timestamps)
    assert timestamps == generate_sample_timestamps(segment, sample_frames_per_second=1)


def test_frame_filename_is_deterministic() -> None:
    """Sampled frame filenames include video, segment, frame, and timestamp IDs."""

    assert (
        frame_filename("V001", segment_id=0, frame_index=0, timestamp_seconds=0.5)
        == "V001_segment_000_frame_000_t0000500ms.jpg"
    )


def test_frame_sampling_manifest_serialization(tmp_path) -> None:
    """Frame sampling manifests are saved as valid JSON and can be loaded."""

    manifest = FrameSamplingManifest(
        video_id="V001",
        source_video_path="/tmp/sample.mp4",
        sample_frames_per_second=1,
        frame_records=[
            FrameSampleRecord(
                video_id="V001",
                source_video_path="/tmp/sample.mp4",
                segment_id=0,
                frame_index=0,
                requested_timestamp_seconds=0.5,
                actual_timestamp_seconds=0.5,
                output_frame_path="/tmp/V001_segment_000_frame_000_t0000500ms.jpg",
                success=True,
                frame_width=320,
                frame_height=240,
            )
        ],
    )
    output_path = tmp_path / "V001_frames.json"

    save_frame_sampling_manifest(manifest, output_path)

    with output_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    assert raw["frame_records"][0]["success"] is True
    assert load_frame_sampling_manifest(output_path) == manifest
