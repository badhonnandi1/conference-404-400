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
    resnet_features: Path
    temporal_features: Path
    normalized_features: Path
    calibration: Path
    digests: Path
    authentication_records: Path
    local_secrets: Path
    logs: Path


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration values."""

    level: str


@dataclass(frozen=True)
class ResNetFeatureConfig:
    """Configuration for ResNet frame feature extraction."""

    architecture: str
    weights: str
    embedding_dimension: int
    batch_size: int
    normalize_frame_embeddings: bool
    segment_aggregation: tuple[str, ...]
    device: str


@dataclass(frozen=True)
class FeaturesConfig:
    """Feature extraction configuration values."""

    resnet: ResNetFeatureConfig
    temporal: "TemporalFeatureConfig"


@dataclass(frozen=True)
class QuantizationStreamConfig:
    """Configuration for one quantized authentication stream."""

    method: str
    bits_per_feature: int
    threshold_source: str | None = None
    gray_codes: dict[str, str] | None = None


@dataclass(frozen=True)
class QuantizationConfig:
    """Configuration for binary digest quantization."""

    version: str
    resnet: QuantizationStreamConfig
    temporal: QuantizationStreamConfig
    bit_order: str
    pack_bits: bool


@dataclass(frozen=True)
class HMACConfig:
    """Configuration for HMAC-protected authentication records."""

    algorithm: str
    digest: str
    minimum_key_bytes: int
    schema_version: int
    timestamp_unit: str
    key_environment_variable: str


@dataclass(frozen=True)
class AuthenticationConfig:
    """Authentication-stage configuration values."""

    hmac: HMACConfig
    quantization: QuantizationConfig


@dataclass(frozen=True)
class TemporalFeatureConfig:
    """Configuration for temporal consistency feature extraction."""

    sample_fps: float
    frame_width: int
    frame_height: int
    grayscale: bool
    gaussian_blur_kernel: int
    changed_pixel_threshold: float
    segment_aggregation: tuple[str, ...]


@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration."""

    project: ProjectConfig
    video: VideoConfig
    features: FeaturesConfig
    authentication: AuthenticationConfig
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
    features = _require_mapping(config.get("features", {}), "features")
    resnet = _require_mapping(features.get("resnet", {}), "features.resnet")
    temporal = _require_mapping(features.get("temporal", {}), "features.temporal")
    authentication = _require_mapping(config.get("authentication", {}), "authentication")
    hmac_config = _require_mapping(authentication.get("hmac", {}), "authentication.hmac")
    quantization = _require_mapping(authentication.get("quantization", {}), "authentication.quantization")
    quant_resnet = _require_mapping(quantization.get("resnet", {}), "authentication.quantization.resnet")
    quant_temporal = _require_mapping(
        quantization.get("temporal", {}),
        "authentication.quantization.temporal",
    )
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
        batch_size = int(resnet.get("batch_size", 8))
        embedding_dimension = int(resnet.get("embedding_dimension", 512))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("ResNet batch size and embedding dimension must be integers.") from exc
    if batch_size <= 0:
        raise ConfigurationError("features.resnet.batch_size must be greater than zero.")
    if embedding_dimension <= 0:
        raise ConfigurationError("features.resnet.embedding_dimension must be greater than zero.")

    aggregation = resnet.get("segment_aggregation", ["mean", "standard_deviation"])
    if not isinstance(aggregation, list) or not all(isinstance(item, str) for item in aggregation):
        raise ConfigurationError("features.resnet.segment_aggregation must be a list of strings.")

    feature_device = str(resnet.get("device", "auto")).lower()
    if feature_device not in {"auto", "cpu", "mps"}:
        raise ConfigurationError("features.resnet.device must be 'auto', 'cpu', or 'mps'.")

    try:
        temporal_sample_fps = float(temporal.get("sample_fps", 4))
        temporal_width = int(temporal.get("frame_width", 224))
        temporal_height = int(temporal.get("frame_height", 224))
        temporal_blur = int(temporal.get("gaussian_blur_kernel", 3))
        changed_pixel_threshold = float(temporal.get("changed_pixel_threshold", 20))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("Temporal feature numeric settings are invalid.") from exc
    if temporal_sample_fps <= 0:
        raise ConfigurationError("features.temporal.sample_fps must be greater than zero.")
    if temporal_width <= 0 or temporal_height <= 0:
        raise ConfigurationError("features.temporal frame dimensions must be greater than zero.")
    if temporal_blur < 0 or temporal_blur % 2 == 0:
        raise ConfigurationError("features.temporal.gaussian_blur_kernel must be zero or an odd positive integer.")
    if changed_pixel_threshold < 0:
        raise ConfigurationError("features.temporal.changed_pixel_threshold must be non-negative.")
    temporal_aggregation = temporal.get("segment_aggregation", ["mean", "standard_deviation", "maximum"])
    if not isinstance(temporal_aggregation, list) or not all(
        isinstance(item, str) for item in temporal_aggregation
    ):
        raise ConfigurationError("features.temporal.segment_aggregation must be a list of strings.")

    quantization_version = str(quantization.get("version", "dev_quantizer_v1"))
    bit_order = str(quantization.get("bit_order", "big")).lower()
    if bit_order not in {"big", "little"}:
        raise ConfigurationError("authentication.quantization.bit_order must be 'big' or 'little'.")
    try:
        resnet_bits = int(quant_resnet.get("bits_per_feature", 1))
        temporal_bits = int(quant_temporal.get("bits_per_feature", 2))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("Quantization bits_per_feature values must be integers.") from exc
    if resnet_bits != 1:
        raise ConfigurationError("Phase 5 requires ResNet bits_per_feature to be 1.")
    if temporal_bits != 2:
        raise ConfigurationError("Phase 5 requires temporal bits_per_feature to be 2.")
    gray_codes = quant_temporal.get(
        "gray_codes",
        {"bin_0": "00", "bin_1": "01", "bin_2": "11", "bin_3": "10"},
    )
    if not isinstance(gray_codes, dict):
        raise ConfigurationError("authentication.quantization.temporal.gray_codes must be a mapping.")

    hmac_algorithm = str(hmac_config.get("algorithm", "HMAC-SHA-256"))
    hmac_digest = str(hmac_config.get("digest", "sha256")).lower()
    timestamp_unit = str(hmac_config.get("timestamp_unit", "microseconds")).lower()
    key_environment_variable = str(
        hmac_config.get("key_environment_variable", "VIDEO_AUTH_HMAC_KEY_HEX")
    )
    try:
        minimum_key_bytes = int(hmac_config.get("minimum_key_bytes", 32))
        hmac_schema_version = int(hmac_config.get("schema_version", 1))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("HMAC minimum key bytes and schema version must be integers.") from exc
    if hmac_algorithm != "HMAC-SHA-256":
        raise ConfigurationError("Phase 6 supports only authentication.hmac.algorithm='HMAC-SHA-256'.")
    if hmac_digest != "sha256":
        raise ConfigurationError("Phase 6 supports only authentication.hmac.digest='sha256'.")
    if minimum_key_bytes < 32:
        raise ConfigurationError("authentication.hmac.minimum_key_bytes must be at least 32.")
    if hmac_schema_version < 1:
        raise ConfigurationError("authentication.hmac.schema_version must be positive.")
    if timestamp_unit != "microseconds":
        raise ConfigurationError("Phase 6 requires authentication.hmac.timestamp_unit='microseconds'.")
    if not key_environment_variable:
        raise ConfigurationError("authentication.hmac.key_environment_variable must not be empty.")

    try:
        paths_config = PathsConfig(
            originals=_resolve_project_path(project_root, str(paths["originals"])),
            segments=_resolve_project_path(project_root, str(paths["segments"])),
            sampled_frames=_resolve_project_path(project_root, str(paths["sampled_frames"])),
            metadata=_resolve_project_path(project_root, str(paths["metadata"])),
            manifests=_resolve_project_path(project_root, str(paths["manifests"])),
            resnet_features=_resolve_project_path(
                project_root, str(paths.get("resnet_features", "data/features/resnet"))
            ),
            temporal_features=_resolve_project_path(
                project_root, str(paths.get("temporal_features", "data/features/temporal"))
            ),
            normalized_features=_resolve_project_path(
                project_root, str(paths.get("normalized_features", "data/features/normalized"))
            ),
            calibration=_resolve_project_path(
                project_root, str(paths.get("calibration", "data/calibration"))
            ),
            digests=_resolve_project_path(project_root, str(paths.get("digests", "data/digests"))),
            authentication_records=_resolve_project_path(
                project_root,
                str(paths.get("authentication_records", "data/authentication_records")),
            ),
            local_secrets=_resolve_project_path(project_root, str(paths.get("local_secrets", "data/secrets"))),
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
        features=FeaturesConfig(
            resnet=ResNetFeatureConfig(
                architecture=str(resnet.get("architecture", "resnet18")),
                weights=str(resnet.get("weights", "DEFAULT")),
                embedding_dimension=embedding_dimension,
                batch_size=batch_size,
                normalize_frame_embeddings=bool(resnet.get("normalize_frame_embeddings", True)),
                segment_aggregation=tuple(item.lower() for item in aggregation),
                device=feature_device,
            ),
            temporal=TemporalFeatureConfig(
                sample_fps=temporal_sample_fps,
                frame_width=temporal_width,
                frame_height=temporal_height,
                grayscale=bool(temporal.get("grayscale", True)),
                gaussian_blur_kernel=temporal_blur,
                changed_pixel_threshold=changed_pixel_threshold,
                segment_aggregation=tuple(item.lower() for item in temporal_aggregation),
            ),
        ),
        authentication=AuthenticationConfig(
            hmac=HMACConfig(
                algorithm=hmac_algorithm,
                digest=hmac_digest,
                minimum_key_bytes=minimum_key_bytes,
                schema_version=hmac_schema_version,
                timestamp_unit=timestamp_unit,
                key_environment_variable=key_environment_variable,
            ),
            quantization=QuantizationConfig(
                version=quantization_version,
                resnet=QuantizationStreamConfig(
                    method=str(quant_resnet.get("method", "median_binary")),
                    threshold_source=str(
                        quant_resnet.get("threshold_source", "normalized_calibration_median")
                    ),
                    bits_per_feature=resnet_bits,
                ),
                temporal=QuantizationStreamConfig(
                    method=str(quant_temporal.get("method", "quartile_gray_code")),
                    bits_per_feature=temporal_bits,
                    gray_codes={str(key): str(value) for key, value in gray_codes.items()},
                ),
                bit_order=bit_order,
                pack_bits=bool(quantization.get("pack_bits", True)),
            )
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
        config.paths.resnet_features,
        config.paths.temporal_features,
        config.paths.normalized_features,
        config.paths.calibration,
        config.paths.digests,
        config.paths.authentication_records,
        config.paths.local_secrets,
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
