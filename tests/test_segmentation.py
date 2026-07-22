"""Tests for logical timestamp segmentation."""

from __future__ import annotations

import json

import pytest

from src.video.metadata import VideoMetadata
from src.video.segmentation import create_segment_manifest, load_segment_manifest, save_segment_manifest


def _metadata(duration: float) -> VideoMetadata:
    return VideoMetadata(
        file_name="sample.mp4",
        absolute_path="/tmp/sample.mp4",
        file_size_bytes=123,
        duration_seconds=duration,
        width=320,
        height=240,
        resolution="320x240",
        average_frame_rate="10/1",
        frame_rate_fps=10.0,
        codec_name="h264",
        codec_long_name=None,
        pixel_format="yuv420p",
        bitrate=None,
        container_format="mov,mp4,m4a,3gp,3g2,mj2",
        number_video_streams=1,
        number_audio_streams=0,
        estimated_frame_count=int(duration * 10),
        can_open_successfully=True,
        extraction_timestamp="2026-07-22T00:00:00+00:00",
    )


@pytest.mark.parametrize(
    ("duration", "expected_segments", "expected_discarded"),
    [
        (5.0, 1, 0.0),
        (10.0, 2, 0.0),
        (17.8, 3, 2.8),
        (4.9, 0, 4.9),
    ],
)
def test_complete_segment_calculation(
    duration: float, expected_segments: int, expected_discarded: float
) -> None:
    """Complete segment counts and discarded duration are deterministic."""

    manifest = create_segment_manifest(
        video_id="V001",
        metadata=_metadata(duration),
        segment_duration_seconds=5,
        sample_frames_per_second=1,
    )

    assert manifest.number_complete_segments == expected_segments
    assert manifest.discarded_duration_seconds == pytest.approx(expected_discarded)
    assert len(manifest.segments) == expected_segments
    for segment in manifest.segments:
        assert segment.duration_seconds == 5
        assert segment.is_complete is True
        assert segment.expected_sample_count == 5


def test_segment_manifest_serialization(tmp_path) -> None:
    """Segment manifests are saved as valid JSON and can be loaded."""

    manifest = create_segment_manifest(
        video_id="V001",
        metadata=_metadata(6.0),
        segment_duration_seconds=5,
        sample_frames_per_second=1,
        video_metadata_reference=tmp_path / "V001_metadata.json",
    )
    output_path = tmp_path / "V001_segments.json"

    save_segment_manifest(manifest, output_path)

    with output_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    assert raw["number_complete_segments"] == 1
    assert raw["discarded_duration_seconds"] == pytest.approx(1.0)
    assert load_segment_manifest(output_path) == manifest
