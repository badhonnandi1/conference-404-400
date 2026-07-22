"""Tests for video metadata helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.video.metadata import VideoMetadata, load_metadata, parse_rational_frame_rate, safe_video_id, save_metadata


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("30/1", 30.0),
        ("30000/1001", pytest.approx(29.97002997)),
        ("25/1", 25.0),
        ("invalid", None),
        (None, None),
        ("0/0", None),
    ],
)
def test_parse_rational_frame_rate(value: str | None, expected: float | None) -> None:
    """Rational FFprobe frame-rate values are parsed safely."""

    assert parse_rational_frame_rate(value) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (Path("/tmp/Camera 1 / Front Door!.mp4"), "Front_Door"),
        ("V001", "V001"),
        ("V001.2", "V001.2"),
        ("unsafe id:*?", "unsafe_id"),
        ("...", "video"),
    ],
)
def test_safe_video_id_generation(source: str | Path, expected: str) -> None:
    """Generated video IDs are safe for filenames and directories."""

    assert safe_video_id(source) == expected


def test_metadata_serialization(tmp_path) -> None:
    """Metadata records are saved as valid JSON and can be loaded."""

    metadata = VideoMetadata(
        file_name="sample.mp4",
        absolute_path="/tmp/sample.mp4",
        file_size_bytes=123,
        duration_seconds=6.0,
        width=320,
        height=240,
        resolution="320x240",
        average_frame_rate="10/1",
        frame_rate_fps=10.0,
        codec_name="h264",
        codec_long_name="H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10",
        pixel_format="yuv420p",
        bitrate=50000,
        container_format="mov,mp4,m4a,3gp,3g2,mj2",
        number_video_streams=1,
        number_audio_streams=0,
        estimated_frame_count=60,
        can_open_successfully=True,
        extraction_timestamp="2026-07-22T00:00:00+00:00",
    )
    output_path = tmp_path / "metadata.json"

    save_metadata(metadata, output_path)

    with output_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    assert raw["file_name"] == "sample.mp4"
    assert load_metadata(output_path) == metadata
