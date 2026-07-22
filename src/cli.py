"""Command-line interface for the preprocessing foundation."""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Sequence

from src.config import (
    AppConfig,
    ConfigurationError,
    apply_cli_overrides,
    ensure_output_directories,
    load_config,
    resolve_video_path,
)
from src.features.aggregation import SegmentAggregationError, aggregate_segment_embeddings
from src.features.device import DeviceSelectionError, select_device
from src.features.feature_storage import (
    build_feature_cache_key,
    build_feature_manifest,
    ensure_can_write_features,
    feature_output_paths,
    save_feature_manifest,
    save_feature_npz,
    sha256_file,
)
from src.features.resnet_features import (
    FeatureExtractionError,
    RESNET18_DEFAULT_PREPROCESSING_DESCRIPTION,
    extract_resnet18_frame_features,
)
from src.utils.ffmpeg_utils import FFmpegToolError, check_required_tools
from src.utils.logging_utils import setup_logging
from src.video.frame_sampling import (
    FrameSamplingError,
    sample_frames_from_segments,
    save_frame_sampling_manifest,
)
from src.video.metadata import (
    ExistingOutputError,
    MissingVideoError,
    NoVideoStreamError,
    UnsupportedVideoError,
    VideoMetadata,
    VideoMetadataError,
    extract_video_metadata,
    load_metadata,
    safe_video_id,
    save_metadata,
)
from src.video.segmentation import (
    SegmentManifest,
    create_segment_manifest,
    load_segment_manifest,
    save_segment_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse command parser."""

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, help="Path to YAML configuration file.")
    common.add_argument("--segment-duration", type=float, help="Override segment duration in seconds.")
    common.add_argument("--sample-fps", type=float, help="Override sampled frames per second.")
    common.add_argument(
        "--keep-incomplete-segment",
        action="store_true",
        help="Keep a final incomplete segment in the segment manifest.",
    )
    common.add_argument("--overwrite", action="store_true", help="Replace existing generated outputs.")
    common.add_argument("--verbose", action="store_true", help="Enable verbose console logging.")

    video_common = argparse.ArgumentParser(add_help=False)
    video_common.add_argument("--video", required=True, type=Path, help="Path to source video.")
    video_common.add_argument("--video-id", help="Manual video identifier, for example V001.")

    parser = argparse.ArgumentParser(
        description="Compression-resilient video authentication preprocessing foundation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check-env", parents=[common], help="Check Python, FFmpeg, FFprobe, and OpenCV.")
    subparsers.add_parser(
        "feature-env",
        parents=[common],
        help="Check torch, torchvision, and feature extraction device support.",
    )
    subparsers.add_parser(
        "inspect",
        parents=[common, video_common],
        help="Inspect a video and save metadata JSON.",
    )
    subparsers.add_parser(
        "segment",
        parents=[common, video_common],
        help="Create a logical segment manifest.",
    )
    subparsers.add_parser(
        "sample",
        parents=[common, video_common],
        help="Sample frames from complete logical segments.",
    )
    subparsers.add_parser(
        "preprocess",
        parents=[common, video_common],
        help="Run environment check, inspect, segment, and sample.",
    )
    extract_resnet = subparsers.add_parser(
        "extract-resnet",
        parents=[common],
        help="Extract pretrained ResNet-18 frame and segment features.",
    )
    extract_resnet.add_argument("--video-id", required=True, help="Video identifier, for example V001.")
    extract_resnet.add_argument(
        "--frame-manifest",
        type=Path,
        help="Path to a Phase 1 frame manifest. Defaults to data/manifests/<VIDEO_ID>_frames.json.",
    )
    extract_resnet.add_argument("--batch-size", type=int, help="Feature extraction batch size.")
    extract_resnet.add_argument(
        "--device",
        choices=["auto", "cpu", "mps"],
        help="Feature extraction device. Defaults to configuration.",
    )
    return parser


def _load_runtime_config(args: argparse.Namespace) -> AppConfig:
    config = load_config(args.config)
    config = apply_cli_overrides(
        config,
        segment_duration_seconds=args.segment_duration,
        sample_frames_per_second=args.sample_fps,
        keep_incomplete_segment=args.keep_incomplete_segment,
    )
    ensure_output_directories(config)
    return config


def _resolve_video_id(video: Path, supplied_id: str | None) -> str:
    if supplied_id:
        return safe_video_id(supplied_id)
    return safe_video_id(video)


def _paths_for(config: AppConfig, video_id: str) -> tuple[Path, Path, Path]:
    metadata_path = config.paths.metadata / f"{video_id}_metadata.json"
    segments_path = config.paths.manifests / f"{video_id}_segments.json"
    frames_path = config.paths.manifests / f"{video_id}_frames.json"
    return metadata_path, segments_path, frames_path


def _print_metadata_summary(video_id: str, metadata: VideoMetadata, output_path: Path) -> None:
    print(f"Video ID: {video_id}")
    print(f"File: {metadata.absolute_path}")
    print(f"Duration: {metadata.duration_seconds:.3f} seconds")
    print(f"Resolution: {metadata.resolution or 'unknown'}")
    print(f"Average frame rate: {metadata.average_frame_rate or 'unknown'}")
    print(f"Frame rate: {metadata.frame_rate_fps if metadata.frame_rate_fps is not None else 'unknown'}")
    print(f"Codec: {metadata.codec_name or 'unknown'}")
    print(f"Container: {metadata.container_format or 'unknown'}")
    print(f"Video streams: {metadata.number_video_streams}")
    print(f"Audio streams: {metadata.number_audio_streams}")
    print(f"Estimated frame count: {metadata.estimated_frame_count}")
    print(f"OpenCV can open: {metadata.can_open_successfully}")
    print(f"Saved metadata: {output_path}")


def _inspect_video(
    config: AppConfig,
    video_path: Path,
    video_id: str,
    overwrite: bool,
    save_when_exists: bool,
) -> tuple[VideoMetadata, Path]:
    metadata_path, _, _ = _paths_for(config, video_id)
    metadata = extract_video_metadata(video_path)
    if metadata_path.exists() and not overwrite and not save_when_exists:
        return metadata, metadata_path
    save_metadata(metadata, metadata_path, overwrite=overwrite or save_when_exists)
    return metadata, metadata_path


def _load_or_create_metadata(
    config: AppConfig,
    video_path: Path,
    video_id: str,
    overwrite: bool,
) -> tuple[VideoMetadata, Path]:
    metadata_path, _, _ = _paths_for(config, video_id)
    if metadata_path.exists() and not overwrite:
        return load_metadata(metadata_path), metadata_path
    metadata = extract_video_metadata(video_path)
    save_metadata(metadata, metadata_path, overwrite=overwrite)
    return metadata, metadata_path


def _create_segments(
    config: AppConfig,
    metadata: VideoMetadata,
    metadata_path: Path,
    video_id: str,
    overwrite: bool,
) -> tuple[SegmentManifest, Path]:
    _, segments_path, _ = _paths_for(config, video_id)
    manifest = create_segment_manifest(
        video_id=video_id,
        metadata=metadata,
        segment_duration_seconds=config.video.segment_duration_seconds,
        sample_frames_per_second=config.video.sample_frames_per_second,
        incomplete_segment_policy=config.video.incomplete_segment_policy,
        video_metadata_reference=metadata_path,
    )
    save_segment_manifest(manifest, segments_path, overwrite=overwrite)
    return manifest, segments_path


def _load_or_create_segments(
    config: AppConfig,
    video_path: Path,
    video_id: str,
    overwrite: bool,
) -> tuple[SegmentManifest, Path]:
    metadata_path, segments_path, _ = _paths_for(config, video_id)
    if segments_path.exists() and not overwrite:
        return load_segment_manifest(segments_path), segments_path
    metadata, metadata_path = _load_or_create_metadata(config, video_path, video_id, overwrite)
    return _create_segments(config, metadata, metadata_path, video_id, overwrite)


def run_check_env(config: AppConfig) -> int:
    """Check required runtime dependencies and print a concise report."""

    failures: list[str] = []
    print(f"Python: {platform.python_version()} ({sys.executable})")
    if sys.version_info < (3, 11):
        failures.append("Python 3.11 or newer is required.")

    try:
        tools = check_required_tools()
        for name, info in tools.items():
            print(f"{name}: {info.path} ({info.version})")
    except FFmpegToolError as exc:
        failures.append(str(exc))

    try:
        import cv2  # type: ignore[import-not-found]

        print(f"OpenCV: {cv2.__version__}")
    except ImportError:
        failures.append("OpenCV is not installed. Run 'pip install -r requirements.txt'.")

    if failures:
        print("Environment check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"Configuration loaded: {config.project.name}")
    print("Environment check passed.")
    return 0


def run_feature_env(config: AppConfig, requested_device: str | None = None) -> int:
    """Check torch, torchvision, and feature extraction device support."""

    try:
        import torch
        import torchvision
        from torchvision.models import ResNet18_Weights
    except ImportError:
        print(
            "Feature environment check failed: torch and torchvision are required. "
            "Run '.venv/bin/python -m pip install torch torchvision'.",
            file=sys.stderr,
        )
        return 1

    device_request = requested_device or config.features.resnet.device
    try:
        device_info = select_device(device_request)
    except DeviceSelectionError as exc:
        print(f"Feature environment check failed: {exc}", file=sys.stderr)
        return 1

    device = torch.device(device_info.selected_device)
    tensor = torch.arange(4, dtype=torch.float32, device=device)
    tensor_result = float((tensor * tensor).sum().cpu())
    weights = ResNet18_Weights.DEFAULT

    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"torch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"Architecture: {platform.machine()}")
    print(f"MPS built: {device_info.mps_built}")
    print(f"MPS available: {device_info.mps_available}")
    print(f"Selected device: {device_info.selected_device}")
    print(f"Tensor verification result: {tensor_result}")
    print(f"Model architecture: {config.features.resnet.architecture}")
    print(f"Model weights: {weights.name}")
    print("Model weights availability: torchvision metadata available; weights download occurs on first model load.")
    return 0


def command_inspect(args: argparse.Namespace, config: AppConfig) -> int:
    """Handle the inspect command."""

    video_path = resolve_video_path(args.video, config.project_root)
    video_id = _resolve_video_id(video_path, args.video_id)
    metadata_path, _, _ = _paths_for(config, video_id)
    if metadata_path.exists() and not args.overwrite:
        raise ExistingOutputError(
            f"Metadata output already exists: {metadata_path}. Use --overwrite to replace it."
        )
    metadata = extract_video_metadata(video_path)
    save_metadata(metadata, metadata_path, overwrite=args.overwrite)
    _print_metadata_summary(video_id, metadata, metadata_path)
    return 0


def command_segment(args: argparse.Namespace, config: AppConfig) -> int:
    """Handle the segment command."""

    video_path = resolve_video_path(args.video, config.project_root)
    video_id = _resolve_video_id(video_path, args.video_id)
    metadata, metadata_path = _load_or_create_metadata(config, video_path, video_id, args.overwrite)
    manifest, segments_path = _create_segments(config, metadata, metadata_path, video_id, args.overwrite)
    print(f"Saved segment manifest: {segments_path}")
    print(f"Complete segments: {manifest.number_complete_segments}")
    print(f"Processed duration: {manifest.processed_duration_seconds:.3f} seconds")
    print(f"Discarded duration: {manifest.discarded_duration_seconds:.3f} seconds")
    return 0


def command_sample(args: argparse.Namespace, config: AppConfig) -> int:
    """Handle the sample command."""

    logger = setup_logging(config.paths.logs, config.logging.level, args.verbose)
    video_path = resolve_video_path(args.video, config.project_root)
    video_id = _resolve_video_id(video_path, args.video_id)
    manifest, segments_path = _load_or_create_segments(config, video_path, video_id, args.overwrite)
    _, _, frames_path = _paths_for(config, video_id)
    if frames_path.exists() and not args.overwrite:
        raise ExistingOutputError(
            f"Frame sampling manifest already exists: {frames_path}. Use --overwrite to replace it."
        )
    frame_manifest = sample_frames_from_segments(
        manifest,
        config.paths.sampled_frames,
        overwrite=args.overwrite,
        logger=logger,
    )
    save_frame_sampling_manifest(frame_manifest, frames_path, overwrite=args.overwrite)
    success_count = sum(1 for record in frame_manifest.frame_records if record.success)
    failure_count = len(frame_manifest.frame_records) - success_count
    print(f"Using segment manifest: {segments_path}")
    print(f"Saved frame manifest: {frames_path}")
    print(f"Sampled frames: {success_count}")
    print(f"Failed frame requests: {failure_count}")
    return 0 if failure_count == 0 else 1


def command_extract_resnet(args: argparse.Namespace, config: AppConfig) -> int:
    """Handle pretrained ResNet-18 feature extraction."""

    logger = setup_logging(config.paths.logs, config.logging.level, args.verbose)
    video_id = safe_video_id(args.video_id)
    resnet_config = config.features.resnet
    if resnet_config.architecture != "resnet18":
        raise FeatureExtractionError("Phase 2 supports only features.resnet.architecture='resnet18'.")
    if resnet_config.weights != "DEFAULT":
        raise FeatureExtractionError("Phase 2 supports only features.resnet.weights='DEFAULT'.")

    frame_manifest_path = (
        resolve_video_path(args.frame_manifest, config.project_root)
        if args.frame_manifest
        else config.paths.manifests / f"{video_id}_frames.json"
    )
    if not frame_manifest_path.exists():
        raise FeatureExtractionError(
            f"Frame manifest not found: {frame_manifest_path}. Run Phase 1 sampling first."
        )

    segment_manifest_path = config.paths.manifests / f"{video_id}_segments.json"
    batch_size = args.batch_size or resnet_config.batch_size
    if batch_size <= 0:
        raise FeatureExtractionError("--batch-size must be greater than zero.")
    device_request = args.device or resnet_config.device
    source_checksum = sha256_file(frame_manifest_path)
    outputs = feature_output_paths(config.paths.resnet_features, video_id)
    cache_key = build_feature_cache_key(
        source_frame_manifest_sha256=source_checksum,
        architecture=resnet_config.architecture,
        weight_identifier=resnet_config.weights,
        preprocessing_description=RESNET18_DEFAULT_PREPROCESSING_DESCRIPTION,
        normalize_frame_embeddings=resnet_config.normalize_frame_embeddings,
        embedding_dimension=resnet_config.embedding_dimension,
    )
    if ensure_can_write_features(outputs, args.overwrite, cache_key):
        print(f"Reusing cached ResNet features: {outputs.npz_path}")
        print(f"Feature manifest: {outputs.manifest_path}")
        return 0

    started = perf_counter()
    video_id_from_manifest, bundle, device_info, frame_result, source_frame_failures = (
        extract_resnet18_frame_features(
            frame_manifest_path=frame_manifest_path,
            batch_size=batch_size,
            requested_device=device_request,
            normalize=resnet_config.normalize_frame_embeddings,
            expected_dimension=resnet_config.embedding_dimension,
            model_cache_dir=config.paths.resnet_features / "_model_cache",
        )
    )
    if video_id_from_manifest != video_id:
        raise FeatureExtractionError(
            f"Frame manifest video_id '{video_id_from_manifest}' does not match requested '{video_id}'."
        )

    segment_result = aggregate_segment_embeddings(
        frame_embeddings=frame_result.embeddings,
        frame_records=frame_result.records,
        segment_manifest_path=segment_manifest_path if segment_manifest_path.exists() else None,
        expected_dimension=resnet_config.embedding_dimension,
    )
    save_feature_npz(outputs, frame_result, segment_result, overwrite=args.overwrite)
    npz_checksum = sha256_file(outputs.npz_path)
    total_time = perf_counter() - started
    manifest = build_feature_manifest(
        video_id=video_id,
        source_frame_manifest_path=frame_manifest_path,
        source_frame_manifest_sha256=source_checksum,
        npz_sha256=npz_checksum,
        paths=outputs,
        bundle=bundle,
        device_info=device_info,
        batch_size=batch_size,
        normalize_frame_embeddings=resnet_config.normalize_frame_embeddings,
        frame_result=frame_result,
        segment_result=segment_result,
        total_processing_time_seconds=total_time,
        source_frame_failures=source_frame_failures,
    )
    manifest.update(cache_key)
    save_feature_manifest(manifest, outputs, overwrite=args.overwrite)

    frame_successes = sum(1 for record in frame_result.records if record.extraction_success)
    frame_failures = len(frame_result.records) - frame_successes
    logger.info(
        "Extracted ResNet features for %s: frames=%s segments=%s device=%s",
        video_id,
        frame_successes,
        len(segment_result.records),
        device_info.selected_device,
    )
    print(f"Saved ResNet feature NPZ: {outputs.npz_path}")
    print(f"Saved ResNet feature manifest: {outputs.manifest_path}")
    print(f"Device: {device_info.selected_device}")
    print(f"Frame embeddings: {frame_result.embeddings.shape}")
    print(f"Segment mean embeddings: {segment_result.mean_embeddings.shape}")
    print(f"Segment standard-deviation embeddings: {segment_result.std_embeddings.shape}")
    print(f"Segment combined embeddings: {segment_result.combined_embeddings.shape}")
    print(f"Frame extraction failures: {frame_failures}")
    print(f"Total processing time: {total_time:.3f} seconds")
    return 0 if frame_failures == 0 else 1


def command_preprocess(args: argparse.Namespace, config: AppConfig) -> int:
    """Handle the complete preprocessing command."""

    environment_status = run_check_env(config)
    if environment_status != 0:
        return environment_status

    logger = setup_logging(config.paths.logs, config.logging.level, args.verbose)
    video_path = resolve_video_path(args.video, config.project_root)
    video_id = _resolve_video_id(video_path, args.video_id)

    metadata = extract_video_metadata(video_path)
    metadata_path, _, frames_path = _paths_for(config, video_id)
    if frames_path.exists() and not args.overwrite:
        raise ExistingOutputError(
            f"Frame sampling manifest already exists: {frames_path}. Use --overwrite to replace it."
        )
    save_metadata(metadata, metadata_path, overwrite=args.overwrite)
    segment_manifest, segments_path = _create_segments(
        config, metadata, metadata_path, video_id, args.overwrite
    )
    frame_manifest = sample_frames_from_segments(
        segment_manifest,
        config.paths.sampled_frames,
        overwrite=args.overwrite,
        logger=logger,
    )
    save_frame_sampling_manifest(frame_manifest, frames_path, overwrite=args.overwrite)

    success_count = sum(1 for record in frame_manifest.frame_records if record.success)
    failure_count = len(frame_manifest.frame_records) - success_count
    print("Preprocessing complete.")
    print(f"Video ID: {video_id}")
    print(f"Metadata: {metadata_path}")
    print(f"Segments: {segments_path}")
    print(f"Frames: {frames_path}")
    print(f"Complete segments: {segment_manifest.number_complete_segments}")
    print(f"Processed duration: {segment_manifest.processed_duration_seconds:.3f} seconds")
    print(f"Discarded duration: {segment_manifest.discarded_duration_seconds:.3f} seconds")
    print(f"Sampled frames: {success_count}")
    print(f"Failed frame requests: {failure_count}")
    return 0 if failure_count == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)
    config: AppConfig | None = None

    try:
        config = _load_runtime_config(args)
        if args.command == "check-env":
            return run_check_env(config)
        if args.command == "feature-env":
            return run_feature_env(config)

        setup_logging(config.paths.logs, config.logging.level, args.verbose)
        if args.command == "inspect":
            return command_inspect(args, config)
        if args.command == "segment":
            return command_segment(args, config)
        if args.command == "sample":
            return command_sample(args, config)
        if args.command == "preprocess":
            return command_preprocess(args, config)
        if args.command == "extract-resnet":
            return command_extract_resnet(args, config)
    except (
        ConfigurationError,
        DeviceSelectionError,
        FeatureExtractionError,
        FFmpegToolError,
        SegmentAggregationError,
        VideoMetadataError,
        FrameSamplingError,
        MissingVideoError,
        UnsupportedVideoError,
        NoVideoStreamError,
        ExistingOutputError,
        ValueError,
    ) as exc:
        log_dir = config.paths.logs if config is not None else Path("logs")
        logger = setup_logging(log_dir, "INFO", getattr(args, "verbose", False))
        logger.error("%s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2
