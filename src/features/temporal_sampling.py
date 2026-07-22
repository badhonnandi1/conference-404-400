"""Dense in-memory temporal frame sampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.video.metadata import MissingVideoError, UnsupportedVideoError
from src.video.segmentation import SegmentManifest, SegmentRecord


class TemporalSamplingError(RuntimeError):
    """Raised when temporal sampling cannot continue."""


@dataclass(frozen=True)
class TemporalSamplingConfig:
    """Configuration for dense temporal frame decoding and preprocessing."""

    sample_fps: float = 4.0
    frame_width: int = 224
    frame_height: int = 224
    grayscale: bool = True
    gaussian_blur_kernel: int = 3
    changed_pixel_threshold: float = 20.0

    def to_dict(self) -> dict[str, int | float | bool]:
        """Return a JSON-serializable configuration dictionary."""

        return asdict(self)


@dataclass
class TemporalFrameRecord:
    """A decoded temporal frame request and optional preprocessed image."""

    video_id: str
    segment_id: int
    frame_index: int
    requested_timestamp_seconds: float
    actual_timestamp_seconds: float | None
    success: bool
    error_message: str | None = None
    image: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable frame record without the image array."""

        return {
            "video_id": self.video_id,
            "segment_id": self.segment_id,
            "frame_index": self.frame_index,
            "requested_timestamp_seconds": self.requested_timestamp_seconds,
            "actual_timestamp_seconds": self.actual_timestamp_seconds,
            "success": self.success,
            "error_message": self.error_message,
        }


def generate_temporal_timestamps(
    start_time_seconds: float,
    end_time_seconds: float,
    sample_fps: float = 4.0,
) -> list[float]:
    """Generate midpoint timestamps for dense temporal sampling."""

    if sample_fps <= 0:
        raise TemporalSamplingError("Temporal sample FPS must be greater than zero.")
    duration = end_time_seconds - start_time_seconds
    if duration <= 0:
        return []
    interval = 1.0 / sample_fps
    count = int(math.floor(duration * sample_fps + 1e-9))
    timestamps: list[float] = []
    for index in range(count):
        timestamp = start_time_seconds + (index + 0.5) * interval
        if start_time_seconds < timestamp < end_time_seconds:
            timestamps.append(round(timestamp, 6))
    return timestamps


def preprocess_temporal_frame(frame_bgr: np.ndarray, config: TemporalSamplingConfig) -> np.ndarray:
    """Convert a decoded BGR frame into a normalized comparison image."""

    if frame_bgr is None or frame_bgr.size == 0:
        raise TemporalSamplingError("Decoded frame is empty.")
    if config.frame_width <= 0 or config.frame_height <= 0:
        raise TemporalSamplingError("Temporal frame dimensions must be greater than zero.")
    if config.grayscale:
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    else:
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        frame,
        (config.frame_width, config.frame_height),
        interpolation=cv2.INTER_AREA,
    )
    if config.gaussian_blur_kernel > 0:
        if config.gaussian_blur_kernel % 2 == 0:
            raise TemporalSamplingError("Gaussian blur kernel must be odd.")
        resized = cv2.GaussianBlur(
            resized,
            (config.gaussian_blur_kernel, config.gaussian_blur_kernel),
            0,
        )
    return resized.astype(np.float32) / 255.0


def decode_temporal_frames_for_segment(
    capture: cv2.VideoCapture,
    segment: SegmentRecord,
    config: TemporalSamplingConfig,
) -> list[TemporalFrameRecord]:
    """Decode and preprocess temporal frames for one logical segment."""

    records: list[TemporalFrameRecord] = []
    for frame_index, timestamp in enumerate(
        generate_temporal_timestamps(
            segment.start_time_seconds,
            segment.end_time_seconds,
            config.sample_fps,
        )
    ):
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        decoded, frame = capture.read()
        timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
        actual_timestamp = (
            round(timestamp_ms / 1000.0, 6)
            if math.isfinite(timestamp_ms) and timestamp_ms >= 0
            else None
        )
        if not decoded or frame is None:
            records.append(
                TemporalFrameRecord(
                    video_id=segment.video_id,
                    segment_id=segment.segment_id,
                    frame_index=frame_index,
                    requested_timestamp_seconds=timestamp,
                    actual_timestamp_seconds=actual_timestamp,
                    success=False,
                    error_message=f"Could not decode temporal frame at {timestamp:.3f}s.",
                )
            )
            continue
        try:
            image = preprocess_temporal_frame(frame, config)
        except TemporalSamplingError as exc:
            records.append(
                TemporalFrameRecord(
                    video_id=segment.video_id,
                    segment_id=segment.segment_id,
                    frame_index=frame_index,
                    requested_timestamp_seconds=timestamp,
                    actual_timestamp_seconds=actual_timestamp,
                    success=False,
                    error_message=str(exc),
                )
            )
            continue
        records.append(
            TemporalFrameRecord(
                video_id=segment.video_id,
                segment_id=segment.segment_id,
                frame_index=frame_index,
                requested_timestamp_seconds=timestamp,
                actual_timestamp_seconds=actual_timestamp,
                success=True,
                image=image,
            )
        )
    return records


def decode_temporal_frames(
    source_video_path: str | Path,
    segment_manifest: SegmentManifest,
    config: TemporalSamplingConfig,
) -> list[TemporalFrameRecord]:
    """Decode temporal frames for every complete segment in a segment manifest."""

    source = Path(source_video_path)
    if not source.exists():
        raise MissingVideoError(f"Source video not found: {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise UnsupportedVideoError(f"OpenCV could not open source video for temporal sampling: {source}")
    try:
        records: list[TemporalFrameRecord] = []
        for segment in segment_manifest.segments:
            if not segment.is_complete:
                continue
            records.extend(decode_temporal_frames_for_segment(capture, segment, config))
        return records
    finally:
        capture.release()
