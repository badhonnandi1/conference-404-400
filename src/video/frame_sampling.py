"""OpenCV-based frame sampling from logical video segments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import math
from pathlib import Path
from typing import Any

from src.video.metadata import ExistingOutputError, MissingVideoError, UnsupportedVideoError
from src.video.segmentation import SegmentManifest, SegmentRecord


class FrameSamplingError(RuntimeError):
    """Base class for frame sampling errors."""


class FrameDecodingError(FrameSamplingError):
    """Raised or recorded when OpenCV cannot decode a requested frame."""


@dataclass(frozen=True)
class FrameSampleRecord:
    """Serializable record for one requested frame sample."""

    video_id: str
    source_video_path: str
    segment_id: int
    frame_index: int
    requested_timestamp_seconds: float
    actual_timestamp_seconds: float | None
    output_frame_path: str
    success: bool
    frame_width: int | None
    frame_height: int | None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable frame sample record."""

        return asdict(self)


@dataclass(frozen=True)
class FrameSamplingManifest:
    """Serializable manifest for sampled frame outputs."""

    video_id: str
    source_video_path: str
    sample_frames_per_second: float
    frame_records: list[FrameSampleRecord]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable frame sampling manifest."""

        payload = asdict(self)
        payload["frame_records"] = [record.to_dict() for record in self.frame_records]
        return payload


def generate_sample_timestamps(
    segment: SegmentRecord, sample_frames_per_second: float = 1.0
) -> list[float]:
    """Generate deterministic mid-interval sample timestamps for a segment."""

    if sample_frames_per_second <= 0:
        raise ValueError("sample_frames_per_second must be greater than zero.")
    if segment.duration_seconds <= 0:
        return []

    interval = 1.0 / sample_frames_per_second
    count = int(math.floor(segment.duration_seconds * sample_frames_per_second + 1e-9))
    timestamps: list[float] = []
    for index in range(count):
        timestamp = segment.start_time_seconds + (index + 0.5) * interval
        if segment.start_time_seconds < timestamp < segment.end_time_seconds:
            timestamps.append(round(timestamp, 6))
    return timestamps


def frame_filename(video_id: str, segment_id: int, frame_index: int, timestamp_seconds: float) -> str:
    """Return a deterministic sampled-frame filename."""

    timestamp_ms = int(round(timestamp_seconds * 1000.0))
    return (
        f"{video_id}_segment_{segment_id:03d}_frame_{frame_index:03d}_"
        f"t{timestamp_ms:07d}ms.jpg"
    )


def _measured_timestamp(capture: Any) -> float | None:
    import cv2  # type: ignore[import-not-found]

    timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    if not math.isfinite(timestamp_ms) or timestamp_ms < 0:
        return None
    return round(timestamp_ms / 1000.0, 6)


def sample_frames_from_segments(
    manifest: SegmentManifest,
    output_root: str | Path,
    overwrite: bool = False,
    logger: logging.Logger | None = None,
) -> FrameSamplingManifest:
    """Sample JPEG frames from every complete segment in a manifest."""

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise UnsupportedVideoError(
            "OpenCV is required for frame sampling. "
            "Install dependencies with 'pip install -r requirements.txt'."
        ) from exc

    source_video = Path(manifest.source_video_path)
    if not source_video.exists():
        raise MissingVideoError(
            f"Source video not found: {source_video}. Check the manifest and --video path."
        )

    output_base = Path(output_root) / manifest.video_id
    planned_outputs: list[Path] = []
    for segment in manifest.segments:
        if not segment.is_complete:
            continue
        segment_dir = output_base / f"segment_{segment.segment_id:03d}"
        for frame_index, timestamp in enumerate(
            generate_sample_timestamps(segment, manifest.sample_frames_per_second)
        ):
            planned_outputs.append(
                segment_dir / frame_filename(manifest.video_id, segment.segment_id, frame_index, timestamp)
            )

    if not overwrite:
        existing = [path for path in planned_outputs if path.exists()]
        if existing:
            raise ExistingOutputError(
                f"Sampled frame already exists: {existing[0]}. Use --overwrite to replace sampled frames."
            )

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise UnsupportedVideoError(
            f"OpenCV could not open '{source_video}'. Check that the file is readable and supported."
        )

    records: list[FrameSampleRecord] = []
    try:
        for segment in manifest.segments:
            if not segment.is_complete:
                if logger:
                    logger.info("Skipping incomplete segment %s", segment.segment_id)
                continue

            segment_dir = output_base / f"segment_{segment.segment_id:03d}"
            segment_dir.mkdir(parents=True, exist_ok=True)
            timestamps = generate_sample_timestamps(segment, manifest.sample_frames_per_second)

            for frame_index, timestamp in enumerate(timestamps):
                output_path = segment_dir / frame_filename(
                    manifest.video_id, segment.segment_id, frame_index, timestamp
                )
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                decoded, frame = capture.read()
                actual_timestamp = _measured_timestamp(capture)

                if not decoded or frame is None:
                    message = (
                        f"Failed to decode frame from '{source_video}' at {timestamp:.3f}s. "
                        "Check the source video and timestamp seeking support."
                    )
                    if logger:
                        logger.warning(message)
                    records.append(
                        FrameSampleRecord(
                            video_id=manifest.video_id,
                            source_video_path=str(source_video),
                            segment_id=segment.segment_id,
                            frame_index=frame_index,
                            requested_timestamp_seconds=timestamp,
                            actual_timestamp_seconds=actual_timestamp,
                            output_frame_path=str(output_path),
                            success=False,
                            frame_width=None,
                            frame_height=None,
                            error_message=message,
                        )
                    )
                    continue

                write_success = bool(cv2.imwrite(str(output_path), frame))
                height, width = int(frame.shape[0]), int(frame.shape[1])
                error_message = None
                if not write_success:
                    error_message = (
                        f"Failed to write sampled frame to '{output_path}'. "
                        "Check output permissions and disk space."
                    )
                    if logger:
                        logger.warning(error_message)

                records.append(
                    FrameSampleRecord(
                        video_id=manifest.video_id,
                        source_video_path=str(source_video),
                        segment_id=segment.segment_id,
                        frame_index=frame_index,
                        requested_timestamp_seconds=timestamp,
                        actual_timestamp_seconds=actual_timestamp,
                        output_frame_path=str(output_path),
                        success=write_success,
                        frame_width=width if write_success else None,
                        frame_height=height if write_success else None,
                        error_message=error_message,
                    )
                )
    finally:
        capture.release()

    return FrameSamplingManifest(
        video_id=manifest.video_id,
        source_video_path=str(source_video),
        sample_frames_per_second=manifest.sample_frames_per_second,
        frame_records=records,
    )


def save_frame_sampling_manifest(
    manifest: FrameSamplingManifest, output_path: str | Path, overwrite: bool = False
) -> Path:
    """Write a frame sampling manifest as formatted JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise ExistingOutputError(
            f"Frame sampling manifest already exists: {path}. Use --overwrite to replace it."
        )
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2)
        handle.write("\n")
    return path


def load_frame_sampling_manifest(path: str | Path) -> FrameSamplingManifest:
    """Load a frame sampling manifest from JSON."""

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    records = [FrameSampleRecord(**record) for record in data.get("frame_records", [])]
    data["frame_records"] = records
    return FrameSamplingManifest(**data)
