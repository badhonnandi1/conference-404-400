"""Configuration loading for the video authentication prototype."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when the project configuration is missing or invalid."""


@dataclass(frozen=True)
class ProjectConfig:
    """Project-level configuration values."""

    name: str
    random_seed: int


@dataclass(frozen=True)
class VideoConfig:
    """Video preprocessing configuration values."""

    segment_duration_seconds: float
    sample_frames_per_second: float
    incomplete_segment_policy: str


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths used by the preprocessing stage."""

    originals: Path
    segments: Path
    sampled_frames: Path
    metadata: Path
    manifests: Path
    logs: Path


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration values."""

    level: str


@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration."""

    project: ProjectConfig
    video: VideoConfig
    paths: PathsConfig
    logging: LoggingConfig
    project_root: Path


def default_project_root() -> Path:
    """Return the project root inferred from this module location."""

    return Path(__file__).resolve().parents[1]


def _require_mapping(data: Any, name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigurationError(f"Configuration section '{name}' must be a mapping.")
    return data


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load the YAML configuration file.

    Relative output paths in the configuration are resolved against the project
    root so commands behave consistently from different working directories.
    """

    project_root = default_project_root()
    path = Path(config_path).expanduser() if config_path else project_root / "configs" / "default.yaml"
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    config = _require_mapping(raw, "root")
    project = _require_mapping(config.get("project"), "project")
    video = _require_mapping(config.get("video"), "video")
    paths = _require_mapping(config.get("paths"), "paths")
    logging = _require_mapping(config.get("logging"), "logging")

    try:
        segment_duration = float(video["segment_duration_seconds"])
        sample_fps = float(video["sample_frames_per_second"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError("Video segment duration and sample FPS must be numeric.") from exc

    if segment_duration <= 0:
        raise ConfigurationError("video.segment_duration_seconds must be greater than zero.")
    if sample_fps <= 0:
        raise ConfigurationError("video.sample_frames_per_second must be greater than zero.")

    policy = str(video.get("incomplete_segment_policy", "discard")).lower()
    if policy not in {"discard", "keep"}:
        raise ConfigurationError("video.incomplete_segment_policy must be 'discard' or 'keep'.")

    try:
        paths_config = PathsConfig(
            originals=_resolve_project_path(project_root, str(paths["originals"])),
            segments=_resolve_project_path(project_root, str(paths["segments"])),
            sampled_frames=_resolve_project_path(project_root, str(paths["sampled_frames"])),
            metadata=_resolve_project_path(project_root, str(paths["metadata"])),
            manifests=_resolve_project_path(project_root, str(paths["manifests"])),
            logs=_resolve_project_path(project_root, str(paths["logs"])),
        )
    except KeyError as exc:
        raise ConfigurationError(f"Missing path configuration key: {exc}") from exc

    return AppConfig(
        project=ProjectConfig(
            name=str(project.get("name", "video-authentication")),
            random_seed=int(project.get("random_seed", 42)),
        ),
        video=VideoConfig(
            segment_duration_seconds=segment_duration,
            sample_frames_per_second=sample_fps,
            incomplete_segment_policy=policy,
        ),
        paths=paths_config,
        logging=LoggingConfig(level=str(logging.get("level", "INFO")).upper()),
        project_root=project_root,
    )


def apply_cli_overrides(
    config: AppConfig,
    segment_duration_seconds: float | None = None,
    sample_frames_per_second: float | None = None,
    keep_incomplete_segment: bool = False,
) -> AppConfig:
    """Return a copy of the configuration with command-line overrides applied."""

    segment_duration = (
        float(segment_duration_seconds)
        if segment_duration_seconds is not None
        else config.video.segment_duration_seconds
    )
    sample_fps = (
        float(sample_frames_per_second)
        if sample_frames_per_second is not None
        else config.video.sample_frames_per_second
    )
    if segment_duration <= 0:
        raise ConfigurationError("--segment-duration must be greater than zero.")
    if sample_fps <= 0:
        raise ConfigurationError("--sample-fps must be greater than zero.")

    policy = "keep" if keep_incomplete_segment else config.video.incomplete_segment_policy
    return replace(
        config,
        video=replace(
            config.video,
            segment_duration_seconds=segment_duration,
            sample_frames_per_second=sample_fps,
            incomplete_segment_policy=policy,
        ),
    )


def ensure_output_directories(config: AppConfig) -> None:
    """Create configured output directories if they are missing."""

    for path in (
        config.paths.originals,
        config.paths.segments,
        config.paths.sampled_frames,
        config.paths.metadata,
        config.paths.manifests,
        config.paths.logs,
    ):
        path.mkdir(parents=True, exist_ok=True)


def resolve_video_path(video: str | Path, project_root: Path) -> Path:
    """Resolve a user-supplied video path.

    Relative paths are first interpreted from the current working directory. If
    that file is not present, they are interpreted from the project root.
    """

    path = Path(video).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (project_root / path).resolve()
