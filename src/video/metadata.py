"""FFprobe-based video metadata extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any

from src.utils.ffmpeg_utils import get_tool_info


class VideoMetadataError(RuntimeError):
    """Base class for metadata extraction errors."""


class MissingVideoError(VideoMetadataError):
    """Raised when the requested video file does not exist."""


class UnsupportedVideoError(VideoMetadataError):
    """Raised when a video cannot be probed or opened."""


class InvalidDurationError(VideoMetadataError):
    """Raised when a video duration is unavailable or invalid."""


class NoVideoStreamError(VideoMetadataError):
    """Raised when FFprobe finds no video stream."""


class ExistingOutputError(VideoMetadataError):
    """Raised when an output file already exists and overwrite is disabled."""


@dataclass(frozen=True)
class VideoMetadata:
    """Serializable metadata extracted from a video file."""

    file_name: str
    absolute_path: str
    file_size_bytes: int
    duration_seconds: float
    width: int | None
    height: int | None
    resolution: str | None
    average_frame_rate: str | None
    frame_rate_fps: float | None
    codec_name: str | None
    codec_long_name: str | None
    pixel_format: str | None
    bitrate: int | None
    container_format: str | None
    number_video_streams: int
    number_audio_streams: int
    estimated_frame_count: int | None
    can_open_successfully: bool
    extraction_timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable metadata dictionary."""

        return asdict(self)


def parse_rational_frame_rate(value: str | None) -> float | None:
    """Safely parse FFprobe rational frame-rate strings as floating-point FPS."""

    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned or cleaned in {"0/0", "N/A"}:
        return None
    if "/" in cleaned:
        numerator_text, denominator_text = cleaned.split("/", maxsplit=1)
        try:
            numerator = float(numerator_text)
            denominator = float(denominator_text)
        except ValueError:
            return None
        if denominator == 0:
            return None
        fps = numerator / denominator
    else:
        try:
            fps = float(cleaned)
        except ValueError:
            return None
    if not math.isfinite(fps) or fps <= 0:
        return None
    return fps


def safe_video_id(source: str | Path) -> str:
    """Generate a deterministic, path-safe video identifier."""

    if isinstance(source, Path):
        value = source.stem
    else:
        value = str(source)
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return sanitized or "video"


def _optional_int(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _run_ffprobe(video_path: Path) -> dict[str, Any]:
    ffprobe = get_tool_info("ffprobe")
    try:
        result = subprocess.run(
            [
                ffprobe.path,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "no stderr output"
        raise UnsupportedVideoError(
            f"FFprobe could not inspect video '{video_path}'. "
            f"Check that the file is a supported, readable video. stderr: {stderr}"
        ) from exc

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise UnsupportedVideoError(
            f"FFprobe returned invalid JSON for '{video_path}'. "
            "Check the FFprobe installation and video file."
        ) from exc


def _opencv_can_open(video_path: Path) -> bool:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise UnsupportedVideoError(
            "OpenCV is required to confirm video readability. "
            "Install dependencies with 'pip install -r requirements.txt'."
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    try:
        return bool(capture.isOpened())
    finally:
        capture.release()


def extract_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Extract structured metadata from a video using FFprobe and OpenCV."""

    path = Path(video_path).expanduser().resolve()
    if not path.exists():
        raise MissingVideoError(
            f"Video file not found: {path}. Check the path passed with --video."
        )
    if not path.is_file():
        raise MissingVideoError(f"Video path is not a file: {path}. Check the input video path.")

    data = _run_ffprobe(path)
    streams = data.get("streams") or []
    if not isinstance(streams, list):
        raise UnsupportedVideoError(f"FFprobe returned malformed stream data for '{path}'.")

    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if not video_streams:
        raise NoVideoStreamError(
            f"No video stream found in '{path}'. Check that the file contains video data."
        )

    video_stream = video_streams[0]
    format_info = data.get("format") or {}
    duration = _optional_float(format_info.get("duration"))
    if duration is None:
        duration = _optional_float(video_stream.get("duration"))
    if duration is None or duration <= 0:
        raise InvalidDurationError(
            f"Invalid or missing duration for '{path}'. Check that the video is complete and readable."
        )

    width = _optional_int(video_stream.get("width"))
    height = _optional_int(video_stream.get("height"))
    resolution = f"{width}x{height}" if width is not None and height is not None else None

    average_frame_rate = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
    frame_rate_fps = parse_rational_frame_rate(average_frame_rate)
    estimated_frame_count = (
        int(round(duration * frame_rate_fps)) if frame_rate_fps is not None else None
    )
    bitrate = _optional_int(format_info.get("bit_rate")) or _optional_int(video_stream.get("bit_rate"))

    return VideoMetadata(
        file_name=path.name,
        absolute_path=str(path),
        file_size_bytes=path.stat().st_size,
        duration_seconds=duration,
        width=width,
        height=height,
        resolution=resolution,
        average_frame_rate=str(average_frame_rate) if average_frame_rate else None,
        frame_rate_fps=frame_rate_fps,
        codec_name=video_stream.get("codec_name"),
        codec_long_name=video_stream.get("codec_long_name"),
        pixel_format=video_stream.get("pix_fmt"),
        bitrate=bitrate,
        container_format=format_info.get("format_name"),
        number_video_streams=len(video_streams),
        number_audio_streams=len(audio_streams),
        estimated_frame_count=estimated_frame_count,
        can_open_successfully=_opencv_can_open(path),
        extraction_timestamp=datetime.now(timezone.utc).isoformat(),
    )


def save_metadata(metadata: VideoMetadata, output_path: str | Path, overwrite: bool = False) -> Path:
    """Write metadata JSON with UTF-8 encoding and indentation."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise ExistingOutputError(
            f"Metadata output already exists: {path}. Use --overwrite to replace it."
        )
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata.to_dict(), handle, indent=2)
        handle.write("\n")
    return path


def load_metadata(path: str | Path) -> VideoMetadata:
    """Load metadata JSON into a VideoMetadata instance."""

    metadata_path = Path(path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return VideoMetadata(**data)
