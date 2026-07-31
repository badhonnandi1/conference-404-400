
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Sequence

import numpy as np

from src.authentication.auth_record_storage import (
    AuthenticationRecordError,
    authentication_record_paths,
    inspect_authentication_record,
    protect_digest_record,
    verify_authentication_record_file,
)
from src.authentication.canonicalization import CanonicalizationError
from src.authentication.digest import DigestError, unpack_packed_bits
from src.authentication.digest_storage import (
    build_and_store_digest,
    create_and_store_quantizer,
    digest_output_paths,
    load_digest_npz,
    load_quantization_artifact,
)
from src.authentication.quantization import (
    DEFAULT_QUANTIZATION_ID,
    QUANTIZATION_WARNING,
    QuantizationError,
)
from src.authentication.hmac_auth import (
    HMACAuthenticationError,
    generate_hmac_key_file,
    load_hmac_key,
)
from src.config import (
    AppConfig,
    ConfigurationError,
    apply_cli_overrides,
    ensure_output_directories,
    load_config,
    resolve_video_path,
)
from src.features.aggregation import SegmentAggregationError, aggregate_segment_embeddings
from src.features.alignment import FeatureAlignmentError
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
from src.features.fusion import FeatureFusionError
from src.features.normalization import NormalizationError
from src.features.normalization_storage import (
    DEFAULT_CALIBRATION_ID,
    DEVELOPMENT_NORMALIZATION_WARNING,
    fit_and_store_normalization_artifact,
    load_normalization_artifact,
    load_normalized_npz,
    normalize_and_store_features,
    normalized_output_paths,
)
from src.features.resnet_features import (
    FeatureExtractionError,
    RESNET18_DEFAULT_PREPROCESSING_DESCRIPTION,
    extract_resnet18_frame_features,
)
from src.features.temporal_features import TemporalFeatureError
from src.features.temporal_sampling import TemporalSamplingConfig, TemporalSamplingError
from src.features.temporal_storage import extract_and_store_temporal_features
from src.utils.ffmpeg_utils import FFmpegToolError, check_required_tools
from src.utils.logging_utils import setup_logging
from src.verification.comparison import ComparisonConfig, DiagnosticWeights, DigestComparisonError
from src.verification.comparison_storage import (
    ComparisonStorageError,
    compare_and_store_digests,
    inspect_comparison,
)
from src.verification.hamming import HammingDistanceError
from src.verification.segment_alignment import SegmentAlignmentError
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
    extract_temporal = subparsers.add_parser(
        "extract-temporal",
        parents=[common],
        help="Extract temporal frame-difference features for one video.",
    )
    extract_temporal.add_argument("--video-id", required=True, help="Video identifier, for example V001.")
    extract_temporal.add_argument(
        "--video-path",
        type=Path,
        help="Path to source video. Defaults to the path stored in metadata.",
    )
    extract_temporal.add_argument(
        "--segment-manifest",
        type=Path,
        help="Path to a Phase 1 segment manifest. Defaults to data/manifests/<VIDEO_ID>_segments.json.",
    )
    extract_temporal.add_argument("--frame-width", type=int, help="Temporal preprocessing width.")
    extract_temporal.add_argument("--frame-height", type=int, help="Temporal preprocessing height.")
    extract_temporal.add_argument(
        "--changed-pixel-threshold",
        type=float,
        help="Changed-pixel threshold in 8-bit pixel units.",
    )
    extract_temporal_all = subparsers.add_parser(
        "extract-temporal-all",
        parents=[common],
        help="Extract temporal features for all videos in the development registry.",
    )
    extract_temporal_all.add_argument(
        "--registry",
        type=Path,
        default=Path("data/manifests/development_originals_registry.json"),
        help="Development originals registry JSON path.",
    )
    extract_temporal_all.add_argument("--frame-width", type=int, help="Temporal preprocessing width.")
    extract_temporal_all.add_argument("--frame-height", type=int, help="Temporal preprocessing height.")
    extract_temporal_all.add_argument(
        "--changed-pixel-threshold",
        type=float,
        help="Changed-pixel threshold in 8-bit pixel units.",
    )
    fit_normalization = subparsers.add_parser(
        "fit-normalization",
        parents=[common],
        help="Fit development stream-specific normalization parameters.",
    )
    fit_normalization.add_argument("--video-ids", nargs="+", required=True, help="Video IDs used for calibration.")
    fit_normalization.add_argument(
        "--calibration-id",
        default=DEFAULT_CALIBRATION_ID,
        help="Calibration artifact identifier.",
    )
    fit_normalization.add_argument("--status", default="development", help="Calibration status label.")
    normalize_features = subparsers.add_parser(
        "normalize-features",
        parents=[common],
        help="Normalize aligned ResNet and temporal features for one video.",
    )
    normalize_features.add_argument("--video-id", required=True, help="Video identifier, for example V001.")
    normalize_features.add_argument(
        "--calibration-id",
        default=DEFAULT_CALIBRATION_ID,
        help="Calibration artifact identifier.",
    )
    normalize_features_all = subparsers.add_parser(
        "normalize-features-all",
        parents=[common],
        help="Normalize aligned features for multiple videos.",
    )
    normalize_features_all.add_argument("--video-ids", nargs="+", required=True, help="Video IDs to normalize.")
    normalize_features_all.add_argument(
        "--calibration-id",
        default=DEFAULT_CALIBRATION_ID,
        help="Calibration artifact identifier.",
    )
    inspect_normalization = subparsers.add_parser(
        "inspect-normalization",
        parents=[common],
        help="Inspect a saved normalization artifact.",
    )
    inspect_normalization.add_argument(
        "--calibration-id",
        default=DEFAULT_CALIBRATION_ID,
        help="Calibration artifact identifier.",
    )
    inspect_normalized = subparsers.add_parser(
        "inspect-normalized-features",
        parents=[common],
        help="Inspect normalized feature outputs for one video.",
    )
    inspect_normalized.add_argument("--video-id", required=True, help="Video identifier, for example V001.")
    create_quantizer = subparsers.add_parser(
        "create-quantizer",
        parents=[common],
        help="Create a development quantization artifact from a normalization artifact.",
    )
    create_quantizer.add_argument(
        "--normalization-id",
        default=DEFAULT_CALIBRATION_ID,
        help="Normalization calibration identifier.",
    )
    create_quantizer.add_argument(
        "--quantization-id",
        default=DEFAULT_QUANTIZATION_ID,
        help="Quantization artifact identifier.",
    )
    create_quantizer.add_argument("--status", default="development", help="Quantization status label.")
    build_digest = subparsers.add_parser(
        "build-digest",
        parents=[common],
        help="Build binary authentication digests for one video.",
    )
    build_digest.add_argument("--video-id", required=True, help="Video identifier, for example V001.")
    build_digest.add_argument(
        "--quantization-id",
        default=DEFAULT_QUANTIZATION_ID,
        help="Quantization artifact identifier.",
    )
    build_digests = subparsers.add_parser(
        "build-digests",
        parents=[common],
        help="Build binary authentication digests for multiple videos.",
    )
    build_digests.add_argument("--video-ids", nargs="+", required=True, help="Video IDs to digest.")
    build_digests.add_argument(
        "--quantization-id",
        default=DEFAULT_QUANTIZATION_ID,
        help="Quantization artifact identifier.",
    )
    inspect_quantizer = subparsers.add_parser(
        "inspect-quantizer",
        parents=[common],
        help="Inspect a saved quantization artifact.",
    )
    inspect_quantizer.add_argument(
        "--quantization-id",
        default=DEFAULT_QUANTIZATION_ID,
        help="Quantization artifact identifier.",
    )
    inspect_digest = subparsers.add_parser(
        "inspect-digest",
        parents=[common],
        help="Inspect one video's binary digest outputs.",
    )
    inspect_digest.add_argument("--video-id", required=True, help="Video identifier, for example V001.")
    generate_hmac_key = subparsers.add_parser(
        "generate-hmac-key",
        parents=[common],
        help="Generate a local development HMAC key file.",
    )
    generate_hmac_key.add_argument("--output", required=True, type=Path, help="Output hex key file path.")
    generate_hmac_key.add_argument("--key-id", required=True, help="Non-secret key identifier.")
    generate_hmac_key.add_argument(
        "--key-bytes",
        type=int,
        default=32,
        help="Number of random key bytes to generate. Defaults to 32.",
    )
    protect_digest = subparsers.add_parser(
        "protect-digest",
        parents=[common],
        help="Create an HMAC-protected authentication record for one video's digests.",
    )
    protect_digest.add_argument("--video-id", required=True, help="Video identifier, for example V001.")
    protect_digest.add_argument("--key-file", type=Path, help="Hex-encoded HMAC key file.")
    protect_digest.add_argument("--key-id", help="Non-secret key identifier stored with the record.")
    protect_digests = subparsers.add_parser(
        "protect-digests",
        parents=[common],
        help="Create HMAC-protected authentication records for multiple videos.",
    )
    protect_digests.add_argument("--video-ids", nargs="+", required=True, help="Video IDs to protect.")
    protect_digests.add_argument("--key-file", type=Path, help="Hex-encoded HMAC key file.")
    protect_digests.add_argument("--key-id", help="Non-secret key identifier stored with the records.")
    verify_record = subparsers.add_parser(
        "verify-auth-record",
        parents=[common],
        help="Verify one HMAC-protected authentication record.",
    )
    verify_record.add_argument("--video-id", required=True, help="Video identifier, for example V001.")
    verify_record.add_argument("--record", type=Path, help="Explicit authentication record JSON path.")
    verify_record.add_argument("--key-file", type=Path, help="Hex-encoded HMAC key file.")
    inspect_record = subparsers.add_parser(
        "inspect-auth-record",
        parents=[common],
        help="Inspect one HMAC-protected authentication record without verifying it.",
    )
    inspect_record.add_argument("--video-id", required=True, help="Video identifier, for example V001.")
    inspect_record.add_argument("--record", type=Path, help="Explicit authentication record JSON path.")
    compare_digests = subparsers.add_parser(
        "compare-digests",
        parents=[common],
        help="Compute segment-level Hamming distances between reference and query digests.",
    )
    compare_digests.add_argument("--reference-id", required=True, help="Trusted reference video ID.")
    compare_digests.add_argument("--query-id", required=True, help="Query video ID.")
    compare_digests.add_argument("--key-file", type=Path, help="Hex-encoded HMAC key file.")
    compare_digests.add_argument("--resnet-weight", type=float, help="Temporary ResNet diagnostic weight.")
    compare_digests.add_argument("--temporal-weight", type=float, help="Temporary temporal diagnostic weight.")
    compare_batch = subparsers.add_parser(
        "compare-digests-batch",
        parents=[common],
        help="Compute segment-level Hamming distances for one reference and multiple queries.",
    )
    compare_batch.add_argument("--reference-id", required=True, help="Trusted reference video ID.")
    compare_batch.add_argument("--query-ids", nargs="+", required=True, help="Query video IDs.")
    compare_batch.add_argument("--key-file", type=Path, help="Hex-encoded HMAC key file.")
    compare_batch.add_argument("--resnet-weight", type=float, help="Temporary ResNet diagnostic weight.")
    compare_batch.add_argument("--temporal-weight", type=float, help="Temporary temporal diagnostic weight.")
    inspect_comparison_parser = subparsers.add_parser(
        "inspect-comparison",
        parents=[common],
        help="Inspect a stored digest-comparison result.",
    )
    inspect_comparison_parser.add_argument("--reference-id", required=True, help="Trusted reference video ID.")
    inspect_comparison_parser.add_argument("--query-id", required=True, help="Query video ID.")
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


def _temporal_config_from_args(args: argparse.Namespace, config: AppConfig) -> TemporalSamplingConfig:
    temporal = config.features.temporal
    sample_fps = args.sample_fps if getattr(args, "sample_fps", None) is not None else temporal.sample_fps
    frame_width = args.frame_width if getattr(args, "frame_width", None) is not None else temporal.frame_width
    frame_height = args.frame_height if getattr(args, "frame_height", None) is not None else temporal.frame_height
    threshold = (
        args.changed_pixel_threshold
        if getattr(args, "changed_pixel_threshold", None) is not None
        else temporal.changed_pixel_threshold
    )
    return TemporalSamplingConfig(
        sample_fps=float(sample_fps),
        frame_width=int(frame_width),
        frame_height=int(frame_height),
        grayscale=temporal.grayscale,
        gaussian_blur_kernel=temporal.gaussian_blur_kernel,
        changed_pixel_threshold=float(threshold),
    )


def _source_video_for_temporal(args: argparse.Namespace, config: AppConfig, video_id: str) -> Path:
    if getattr(args, "video_path", None):
        return resolve_video_path(args.video_path, config.project_root)
    metadata_path = config.paths.metadata / f"{video_id}_metadata.json"
    if metadata_path.exists():
        return Path(load_metadata(metadata_path).absolute_path)
    raise TemporalSamplingError(
        f"Source video path not supplied and metadata was not found: {metadata_path}. "
        "Run preprocess first or pass --video-path."
    )


def command_extract_temporal(args: argparse.Namespace, config: AppConfig) -> int:

    setup_logging(config.paths.logs, config.logging.level, args.verbose)
    video_id = safe_video_id(args.video_id)
    source_video_path = _source_video_for_temporal(args, config, video_id)
    segment_manifest_path = (
        resolve_video_path(args.segment_manifest, config.project_root)
        if getattr(args, "segment_manifest", None)
        else config.paths.manifests / f"{video_id}_segments.json"
    )
    if not segment_manifest_path.exists():
        raise TemporalSamplingError(
            f"Segment manifest not found: {segment_manifest_path}. Run preprocessing first."
        )
    temporal_config = _temporal_config_from_args(args, config)
    result, manifest, paths, cache_reused = extract_and_store_temporal_features(
        video_id=video_id,
        source_video_path=source_video_path,
        segment_manifest_path=segment_manifest_path,
        output_root=config.paths.temporal_features,
        config=temporal_config,
        overwrite=args.overwrite,
    )
    if cache_reused:
        print(f"Reusing cached temporal features: {paths.npz_path}")
        print(f"Temporal manifest: {paths.manifest_path}")
        return 0
    assert result is not None and manifest is not None
    successful_pairs = sum(1 for record in result.pair_records if record.success)
    failed_pairs = len(result.pair_records) - successful_pairs
    print(f"Saved temporal feature NPZ: {paths.npz_path}")
    print(f"Saved temporal feature manifest: {paths.manifest_path}")
    print(f"Temporal frames decoded: {sum(record.decoded_temporal_frame_count for record in result.segment_records)}")
    print(f"Pair features: {result.pair_features.shape}")
    print(f"Segment features: {result.segment_features.shape}")
    print(f"Successful pairs: {successful_pairs}")
    print(f"Failed pairs: {failed_pairs}")
    print(f"Maximum-discontinuity timestamps: {result.segment_max_discontinuity_timestamps.tolist()}")
    print(f"Total processing time: {manifest['total_processing_time_seconds']:.3f} seconds")
    return 0 if failed_pairs == 0 else 1


def command_extract_temporal_all(args: argparse.Namespace, config: AppConfig) -> int:

    registry_path = resolve_video_path(args.registry, config.project_root)
    if not registry_path.exists():
        raise TemporalSamplingError(f"Development registry not found: {registry_path}")
    with registry_path.open("r", encoding="utf-8") as handle:
        registry = __import__("json").load(handle)
    videos = registry.get("videos", [])
    if not isinstance(videos, list):
        raise TemporalSamplingError(f"Registry does not contain a videos list: {registry_path}")

    failures = 0
    for record in videos:
        video_id = safe_video_id(str(record["video_id"]))
        source_video_path = Path(str(record["source_path"]))
        segment_manifest_path = config.paths.manifests / f"{video_id}_segments.json"
        temporal_config = _temporal_config_from_args(args, config)
        result, manifest, paths, cache_reused = extract_and_store_temporal_features(
            video_id=video_id,
            source_video_path=source_video_path,
            segment_manifest_path=segment_manifest_path,
            output_root=config.paths.temporal_features,
            config=temporal_config,
            overwrite=args.overwrite,
        )
        if cache_reused:
            print(f"{video_id}: reused cached temporal features at {paths.npz_path}")
            continue
        assert result is not None and manifest is not None
        failed_pairs = sum(1 for pair_record in result.pair_records if not pair_record.success)
        failures += 1 if failed_pairs else 0
        print(
            f"{video_id}: pairs={result.pair_features.shape}, segments={result.segment_features.shape}, "
            f"failed_pairs={failed_pairs}, time={manifest['total_processing_time_seconds']:.3f}s"
        )
    return 0 if failures == 0 else 1


def _safe_video_ids(values: Sequence[str]) -> list[str]:
    return [safe_video_id(value) for value in values]


def command_fit_normalization(args: argparse.Namespace, config: AppConfig) -> int:

    video_ids = _safe_video_ids(args.video_ids)
    artifact, aligned_sets = fit_and_store_normalization_artifact(
        video_ids=video_ids,
        resnet_root=config.paths.resnet_features,
        temporal_root=config.paths.temporal_features,
        manifests_root=config.paths.manifests,
        calibration_root=config.paths.calibration,
        calibration_id=args.calibration_id,
        status=args.status,
        overwrite=args.overwrite,
    )
    print(f"Saved normalization parameters: {artifact.paths.npz_path}")
    print(f"Saved normalization manifest: {artifact.paths.manifest_path}")
    print(f"Calibration ID: {artifact.calibration_id}")
    print(f"Status: {artifact.manifest['status']}")
    print(f"Development-only: {artifact.manifest['development_only']}")
    print(f"Source videos: {video_ids}")
    print(f"Total calibration segments: {artifact.manifest['total_calibration_segments']}")
    print(f"ResNet dimension: {artifact.resnet_normalizer.feature_dimension}")
    print(f"Temporal dimension: {artifact.temporal_normalizer.feature_dimension}")
    print(f"Zero-IQR ResNet dimensions: {int(np.count_nonzero(artifact.resnet_normalizer.zero_iqr_mask))}")
    print(f"Zero-IQR temporal dimensions: {int(np.count_nonzero(artifact.temporal_normalizer.zero_iqr_mask))}")
    print(DEVELOPMENT_NORMALIZATION_WARNING)
    if any(aligned.warnings for aligned in aligned_sets):
        print("Warnings were recorded in the calibration manifest.")
    return 0


def command_normalize_features(args: argparse.Namespace, config: AppConfig) -> int:

    video_id = safe_video_id(args.video_id)
    bundle, manifest, paths, cache_reused = normalize_and_store_features(
        video_id=video_id,
        resnet_root=config.paths.resnet_features,
        temporal_root=config.paths.temporal_features,
        manifests_root=config.paths.manifests,
        calibration_root=config.paths.calibration,
        normalized_root=config.paths.normalized_features,
        calibration_id=args.calibration_id,
        overwrite=args.overwrite,
    )
    if cache_reused:
        print(f"Reusing cached normalized features: {paths.npz_path}")
        print(f"Normalized manifest: {paths.manifest_path}")
        return 0
    assert bundle is not None
    min_value, max_value = bundle.value_range()
    print(f"Saved normalized feature NPZ: {paths.npz_path}")
    print(f"Saved normalized feature manifest: {paths.manifest_path}")
    print(f"Video ID: {video_id}")
    print(f"Calibration ID: {manifest['calibration_id']}")
    print(f"Segment count: {manifest['segment_count']}")
    print(f"ResNet normalized: {bundle.resnet_normalized_features.shape}")
    print(f"Temporal normalized: {bundle.temporal_normalized_features.shape}")
    print(f"Combined normalized: {bundle.combined_normalized_features.shape}")
    print(f"Value range: [{min_value:.6f}, {max_value:.6f}]")
    print(f"Finite values: {bundle.finite()}")
    print(f"Processing time: {manifest['processing_time_seconds']:.6f} seconds")
    return 0 if bundle.finite() else 1


def command_normalize_features_all(args: argparse.Namespace, config: AppConfig) -> int:

    failures = 0
    for video_id in _safe_video_ids(args.video_ids):
        bundle, manifest, paths, cache_reused = normalize_and_store_features(
            video_id=video_id,
            resnet_root=config.paths.resnet_features,
            temporal_root=config.paths.temporal_features,
            manifests_root=config.paths.manifests,
            calibration_root=config.paths.calibration,
            normalized_root=config.paths.normalized_features,
            calibration_id=args.calibration_id,
            overwrite=args.overwrite,
        )
        if cache_reused:
            print(f"{video_id}: reused cached normalized features at {paths.npz_path}")
            continue
        assert bundle is not None
        failures += 0 if bundle.finite() else 1
        print(
            f"{video_id}: resnet={bundle.resnet_normalized_features.shape}, "
            f"temporal={bundle.temporal_normalized_features.shape}, "
            f"combined={bundle.combined_normalized_features.shape}, "
            f"finite={bundle.finite()}, time={manifest['processing_time_seconds']:.6f}s"
        )
    return 0 if failures == 0 else 1


def command_inspect_normalization(args: argparse.Namespace, config: AppConfig) -> int:

    artifact = load_normalization_artifact(config.paths.calibration, args.calibration_id)
    print(f"Calibration ID: {artifact.calibration_id}")
    print(f"Status: {artifact.manifest['status']}")
    print(f"Development-only: {artifact.manifest['development_only']}")
    print(f"Warning: {DEVELOPMENT_NORMALIZATION_WARNING}")
    print(f"Source videos: {artifact.manifest['source_video_ids']}")
    print(f"Total calibration segments: {artifact.manifest['total_calibration_segments']}")
    print(f"ResNet dimension: {artifact.resnet_normalizer.feature_dimension}")
    print(f"Temporal dimension: {artifact.temporal_normalizer.feature_dimension}")
    print(f"Method: {artifact.manifest['normalization_method']}")
    print(f"Epsilon: {artifact.resnet_normalizer.epsilon}")
    print(f"Clipping range: [{artifact.resnet_normalizer.clip_min}, {artifact.resnet_normalizer.clip_max}]")
    print(f"Zero-IQR ResNet dimensions: {int(np.count_nonzero(artifact.resnet_normalizer.zero_iqr_mask))}")
    print(f"Zero-IQR temporal dimensions: {int(np.count_nonzero(artifact.temporal_normalizer.zero_iqr_mask))}")
    print(f"NPZ: {artifact.paths.npz_path}")
    print(f"NPZ checksum: {artifact.npz_sha256}")
    return 0


def command_inspect_normalized_features(args: argparse.Namespace, config: AppConfig) -> int:

    video_id = safe_video_id(args.video_id)
    paths = normalized_output_paths(config.paths.normalized_features, video_id)
    if not paths.manifest_path.exists() or not paths.npz_path.exists():
        raise FileNotFoundError(f"Normalized feature outputs not found for {video_id}: {paths.output_dir}")
    with paths.manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    arrays = load_normalized_npz(paths.npz_path)
    combined = arrays["combined_normalized_features"]
    finite = bool(np.all(np.isfinite(combined)))
    print(f"Video ID: {video_id}")
    print(f"Calibration ID: {manifest['calibration_id']}")
    print(f"Development-only warning: {manifest['development_warning']}")
    print(f"Segment count: {manifest['segment_count']}")
    print(f"ResNet dimension: {manifest['feature_dimensions']['resnet_normalized']}")
    print(f"Temporal dimension: {manifest['feature_dimensions']['temporal_normalized']}")
    print(f"Combined dimension: {manifest['feature_dimensions']['combined_normalized']}")
    print(f"Minimum normalized value: {float(np.min(combined)):.6f}")
    print(f"Maximum normalized value: {float(np.max(combined)):.6f}")
    print(f"Finite values: {finite}")
    settings = manifest["normalization_settings"]
    print(f"Zero-IQR ResNet dimensions: {settings['resnet_zero_iqr_dimension_count']}")
    print(f"Zero-IQR temporal dimensions: {settings['temporal_zero_iqr_dimension_count']}")
    print(f"Source ResNet checksum: {manifest['source_resnet_sha256']}")
    print(f"Source temporal checksum: {manifest['source_temporal_sha256']}")
    print(f"Calibration checksum: {manifest['calibration_npz_sha256']}")
    print(f"Output checksum: {manifest['npz_sha256']}")
    return 0 if finite else 1


def command_create_quantizer(args: argparse.Namespace, config: AppConfig) -> int:

    artifact = create_and_store_quantizer(
        normalization_root=config.paths.calibration,
        quantization_root=config.paths.calibration,
        normalization_id=args.normalization_id,
        quantization_id=args.quantization_id,
        version=config.authentication.quantization.version,
        status=args.status,
        bit_order=config.authentication.quantization.bit_order,
        overwrite=args.overwrite,
    )
    print(f"Saved quantization parameters: {artifact.paths.npz_path}")
    print(f"Saved quantization manifest: {artifact.paths.manifest_path}")
    print(f"Quantization ID: {artifact.quantization_id}")
    print(f"Normalization ID: {artifact.parameters.normalization_id}")
    print(f"Version: {artifact.parameters.version}")
    print(f"Status: {artifact.parameters.status}")
    print(f"Development-only: {artifact.parameters.development_only}")
    print(f"ResNet digest length: {artifact.manifest['digest_dimensions']['resnet']}")
    print(f"Temporal digest length: {artifact.manifest['digest_dimensions']['temporal']}")
    print(f"Hybrid digest length: {artifact.manifest['digest_dimensions']['hybrid']}")
    print(QUANTIZATION_WARNING)
    return 0


def command_build_digest(args: argparse.Namespace, config: AppConfig) -> int:

    video_id = safe_video_id(args.video_id)
    bundle, manifest, paths, cache_reused = build_and_store_digest(
        video_id=video_id,
        normalized_root=config.paths.normalized_features,
        quantization_root=config.paths.calibration,
        digest_root=config.paths.digests,
        quantization_id=args.quantization_id,
        overwrite=args.overwrite,
    )
    if cache_reused:
        print(f"Reusing cached digest: {paths.npz_path}")
        print(f"Digest manifest: {paths.manifest_path}")
        return 0
    assert bundle is not None
    print(f"Saved digest NPZ: {paths.npz_path}")
    print(f"Saved digest manifest: {paths.manifest_path}")
    print(f"Video ID: {video_id}")
    print(f"Segments: {bundle.segment_ids.shape[0]}")
    print(f"ResNet digest: {bundle.resnet_binary_digests.shape}")
    print(f"Temporal digest: {bundle.temporal_binary_digests.shape}")
    print(f"Hybrid digest: {bundle.hybrid_binary_digests.shape}")
    print(f"Packed bytes: resnet={bundle.resnet_packed_digests.shape[1]}, temporal={bundle.temporal_packed_digests.shape[1]}, hybrid={bundle.hybrid_packed_digests.shape[1]}")
    print(f"Pack/unpack round-trip: {bundle.validate_round_trips()}")
    print(f"Processing time: {manifest['processing_time_seconds']:.6f} seconds")
    return 0 if bundle.validate_round_trips() else 1


def command_build_digests(args: argparse.Namespace, config: AppConfig) -> int:

    failures = 0
    for video_id in _safe_video_ids(args.video_ids):
        bundle, manifest, paths, cache_reused = build_and_store_digest(
            video_id=video_id,
            normalized_root=config.paths.normalized_features,
            quantization_root=config.paths.calibration,
            digest_root=config.paths.digests,
            quantization_id=args.quantization_id,
            overwrite=args.overwrite,
        )
        if cache_reused:
            print(f"{video_id}: reused cached digest at {paths.npz_path}")
            continue
        assert bundle is not None
        ok = bundle.validate_round_trips()
        failures += 0 if ok else 1
        print(
            f"{video_id}: resnet={bundle.resnet_binary_digests.shape}, "
            f"temporal={bundle.temporal_binary_digests.shape}, "
            f"hybrid={bundle.hybrid_binary_digests.shape}, "
            f"packed=({bundle.resnet_packed_digests.shape[1]}, {bundle.temporal_packed_digests.shape[1]}, {bundle.hybrid_packed_digests.shape[1]}), "
            f"round_trip={ok}, time={manifest['processing_time_seconds']:.6f}s"
        )
    return 0 if failures == 0 else 1


def command_inspect_quantizer(args: argparse.Namespace, config: AppConfig) -> int:

    artifact = load_quantization_artifact(config.paths.calibration, args.quantization_id)
    manifest = artifact.manifest
    print(f"Quantization ID: {artifact.quantization_id}")
    print(f"Normalization ID: {manifest['normalization_calibration_id']}")
    print(f"Version: {manifest['quantization_version']}")
    print(f"Status: {manifest['status']}")
    print(f"Development-only: {manifest['development_only']}")
    print(f"Warning: {QUANTIZATION_WARNING}")
    print(f"ResNet method: {manifest['resnet_quantization_method']}")
    print(f"Temporal method: {manifest['temporal_quantization_method']}")
    print(f"Gray-code mapping: {manifest['gray_code_mapping']}")
    print(f"Digest dimensions: {manifest['digest_dimensions']}")
    print(f"Stream boundaries: {manifest['stream_boundaries']}")
    print(f"Bit order: {manifest['bit_order']}")
    print(f"NPZ: {artifact.paths.npz_path}")
    print(f"NPZ checksum: {artifact.npz_sha256}")
    return 0


def command_inspect_digest(args: argparse.Namespace, config: AppConfig) -> int:

    video_id = safe_video_id(args.video_id)
    paths = digest_output_paths(config.paths.digests, video_id)
    if not paths.npz_path.exists() or not paths.manifest_path.exists():
        raise FileNotFoundError(f"Digest outputs not found for {video_id}: {paths.output_dir}")
    with paths.manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    arrays = load_digest_npz(paths.npz_path)
    summary = manifest["bit_statistics"]
    resnet_bits = arrays["resnet_binary_digests"]
    temporal_bits = arrays["temporal_binary_digests"]
    hybrid_bits = arrays["hybrid_binary_digests"]
    bit_order = manifest["bit_order"]
    round_trip = (
        np.array_equal(
            unpack_packed_bits(arrays["resnet_packed_digests"], int(arrays["resnet_bit_length"]), bit_order),
            resnet_bits,
        )
        and np.array_equal(
            unpack_packed_bits(arrays["temporal_packed_digests"], int(arrays["temporal_bit_length"]), bit_order),
            temporal_bits,
        )
        and np.array_equal(
            unpack_packed_bits(arrays["hybrid_packed_digests"], int(arrays["hybrid_bit_length"]), bit_order),
            hybrid_bits,
        )
    )
    clipping = manifest["clipping_statistics"]
    print(f"Video ID: {video_id}")
    print(f"Segment count: {manifest['segment_count']}")
    print(f"Normalization ID: {manifest['normalization_calibration_id']}")
    print(f"Quantization ID: {manifest['quantization_id']}")
    print(f"Development-only warning: {manifest['development_warning']}")
    print(f"ResNet digest shape and length: {resnet_bits.shape}, {int(arrays['resnet_bit_length'])}")
    print(f"Temporal digest shape and length: {temporal_bits.shape}, {int(arrays['temporal_bit_length'])}")
    print(f"Hybrid digest shape and length: {hybrid_bits.shape}, {int(arrays['hybrid_bit_length'])}")
    print(
        "Packed byte sizes: "
        f"resnet={arrays['resnet_packed_digests'].shape[1]}, "
        f"temporal={arrays['temporal_packed_digests'].shape[1]}, "
        f"hybrid={arrays['hybrid_packed_digests'].shape[1]}"
    )
    print(f"ResNet zero/one counts: {summary['resnet']['zero_count']}/{summary['resnet']['one_count']}")
    print(f"Temporal zero/one counts: {summary['temporal']['zero_count']}/{summary['temporal']['one_count']}")
    print(f"Hybrid zero/one counts: {summary['hybrid']['zero_count']}/{summary['hybrid']['one_count']}")
    print(f"ResNet one-bit ratio: {summary['resnet']['one_ratio']:.6f}")
    print(f"Temporal one-bit ratio: {summary['temporal']['one_ratio']:.6f}")
    print(f"Hybrid one-bit ratio: {summary['hybrid']['one_ratio']:.6f}")
    print(f"Temporal bin distribution: {summary['temporal_bin_distribution']}")
    print(f"ResNet clipping percentage: {clipping['resnet']['clipping_percentage']:.6f}")
    print(f"Temporal clipping percentage: {clipping['temporal']['clipping_percentage']:.6f}")
    print(f"Combined clipping percentage: {clipping['combined']['clipping_percentage']:.6f}")
    print(f"Pack/unpack validation: {round_trip}")
    print(f"Source normalized checksum: {manifest['source_normalized_feature_sha256']}")
    print(f"Quantizer checksum: {manifest['quantization_artifact_sha256']}")
    print(f"Output checksum: {manifest['npz_sha256']}")
    return 0 if round_trip else 1


def _hmac_key_from_args(args: argparse.Namespace, config: AppConfig):
    return load_hmac_key(
        key_file=resolve_video_path(args.key_file, config.project_root) if getattr(args, "key_file", None) else None,
        key_id=getattr(args, "key_id", None),
        environment_variable=config.authentication.hmac.key_environment_variable,
        minimum_key_bytes=config.authentication.hmac.minimum_key_bytes,
    )


def command_generate_hmac_key(args: argparse.Namespace, config: AppConfig) -> int:

    output_path = resolve_video_path(args.output, config.project_root)
    key_info = generate_hmac_key_file(
        output_path=output_path,
        key_id=args.key_id,
        key_bytes=args.key_bytes,
        overwrite=args.overwrite,
    )
    print(f"Generated HMAC key file: {output_path}")
    print(f"Key ID: {key_info.key_id}")
    print(f"Key source type: {key_info.source_type}")
    print(f"Key length: {key_info.key_length_bytes} bytes")
    print(f"Key fingerprint: {key_info.key_fingerprint}")
    print("Secret key material was not printed.")
    return 0


def command_protect_digest(args: argparse.Namespace, config: AppConfig) -> int:

    video_id = safe_video_id(args.video_id)
    key_info = _hmac_key_from_args(args, config)
    stored = protect_digest_record(
        video_id=video_id,
        digest_root=config.paths.digests,
        authentication_record_root=config.paths.authentication_records,
        key_info=key_info,
        schema_version=config.authentication.hmac.schema_version,
        algorithm=config.authentication.hmac.algorithm,
        overwrite=args.overwrite,
    )
    authentication = stored.record["authentication"]
    print(f"Video ID: {video_id}")
    if stored.cache_reused:
        print(f"Reusing cached authentication record: {stored.paths.record_path}")
    else:
        print(f"Saved authentication record: {stored.paths.record_path}")
    print(f"Algorithm: {authentication['algorithm']}")
    print(f"Key ID: {authentication['key_id']}")
    print(f"Key fingerprint: {authentication['key_fingerprint']}")
    print(f"Key length: {authentication['key_length_bytes']} bytes")
    print(f"Segment count: {stored.record['payload']['segment_count']}")
    print(f"Authentication tag length: {len(authentication['tag_hex'])} hex characters")
    print(f"Canonical payload checksum: {authentication['canonical_payload_sha256']}")
    print(f"Record file checksum: {stored.record_file_sha256}")
    return 0


def command_protect_digests(args: argparse.Namespace, config: AppConfig) -> int:

    key_info = _hmac_key_from_args(args, config)
    for video_id in _safe_video_ids(args.video_ids):
        stored = protect_digest_record(
            video_id=video_id,
            digest_root=config.paths.digests,
            authentication_record_root=config.paths.authentication_records,
            key_info=key_info,
            schema_version=config.authentication.hmac.schema_version,
            algorithm=config.authentication.hmac.algorithm,
            overwrite=args.overwrite,
        )
        authentication = stored.record["authentication"]
        status = "reused" if stored.cache_reused else "saved"
        print(
            f"{video_id}: {status}={stored.paths.record_path}, "
            f"segments={stored.record['payload']['segment_count']}, "
            f"algorithm={authentication['algorithm']}, "
            f"key_id={authentication['key_id']}, "
            f"key_fingerprint={authentication['key_fingerprint']}, "
            f"tag_len={len(authentication['tag_hex'])}, "
            f"payload_sha256={authentication['canonical_payload_sha256']}"
        )
    return 0


def command_verify_auth_record(args: argparse.Namespace, config: AppConfig) -> int:

    video_id = safe_video_id(args.video_id)
    record_path = (
        resolve_video_path(args.record, config.project_root)
        if getattr(args, "record", None)
        else authentication_record_paths(config.paths.authentication_records, video_id).record_path
    )
    key_info = _hmac_key_from_args(args, config)
    result = verify_authentication_record_file(
        record_path,
        key_info=key_info,
        algorithm=config.authentication.hmac.algorithm,
    )
    print(f"Video ID: {result.video_id or video_id}")
    print(f"Record: {record_path}")
    print(f"Record valid: {result.record_valid}")
    print(f"HMAC valid: {result.hmac_valid}")
    print(f"Payload checksum valid: {result.payload_checksum_valid}")
    print(f"Key fingerprint match: {result.key_fingerprint_match}")
    print(f"Schema valid: {result.schema_valid}")
    print(f"Algorithm supported: {result.algorithm_supported}")
    print(f"Key ID: {result.key_id}")
    print(f"Failure reason: {result.failure_reason or 'none'}")
    print(f"Verification timestamp: {result.verification_timestamp}")
    return 0 if result.record_valid else 1


def command_inspect_auth_record(args: argparse.Namespace, config: AppConfig) -> int:

    video_id = safe_video_id(args.video_id)
    record_path = (
        resolve_video_path(args.record, config.project_root)
        if getattr(args, "record", None)
        else authentication_record_paths(config.paths.authentication_records, video_id).record_path
    )
    summary = inspect_authentication_record(record_path)
    print(f"Video ID: {summary['video_id']}")
    print(f"Schema version: {summary['schema_version']}")
    print(f"Record schema version: {summary['record_schema_version']}")
    print(f"Algorithm: {summary['algorithm']}")
    print(f"Key ID: {summary['key_id']}")
    print(f"Key fingerprint: {summary['key_fingerprint']}")
    print(f"Segment count: {summary['segment_count']}")
    print(f"Normalization ID: {summary['normalization_id']}")
    print(f"Quantization ID: {summary['quantization_id']}")
    print(f"Digest lengths: {summary['digest_lengths']}")
    print(f"Payload checksum: {summary['canonical_payload_sha256']}")
    print(f"Record checksum: {summary['record_file_sha256']}")
    print(f"Development-only: {summary['development_only']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Record path: {summary['record_path']}")
    return 0


def _comparison_config_from_args(args: argparse.Namespace, config: AppConfig) -> ComparisonConfig:
    comparison = config.verification.comparison
    resnet_weight = (
        float(args.resnet_weight)
        if getattr(args, "resnet_weight", None) is not None
        else comparison.resnet_weight
    )
    temporal_weight = (
        float(args.temporal_weight)
        if getattr(args, "temporal_weight", None) is not None
        else comparison.temporal_weight
    )
    phase_config = ComparisonConfig(
        resnet_bit_length=comparison.resnet_bit_length,
        temporal_bit_length=comparison.temporal_bit_length,
        hybrid_bit_length=comparison.hybrid_bit_length,
        diagnostic_weights=DiagnosticWeights(resnet=resnet_weight, temporal=temporal_weight),
        tie_tolerance=comparison.tie_tolerance,
        segment_alignment=comparison.segment_alignment,
        timestamp_tolerance_microseconds=comparison.timestamp_tolerance_microseconds,
    )
    phase_config.validate()
    return phase_config


def _print_stored_comparison_summary(stored) -> None:
    summary = stored.result.video_summary
    status = "reused" if stored.cache_reused else "saved"
    print(f"Comparison ID: {stored.result.comparison_id}")
    print(f"Output status: {status}")
    print(f"Manifest: {stored.paths.manifest_path}")
    print(f"NPZ: {stored.paths.npz_path}")
    print(f"Reference HMAC valid: {stored.result.reference_hmac_verification_result['hmac_valid']}")
    print(f"Matched segments: {summary['matched_segment_count']}")
    print(f"Missing segments: {summary['missing_segment_count']}")
    print(f"Extra segments: {summary['extra_segment_count']}")
    print(f"Timestamp mismatches: {summary['timestamp_mismatch_count']}")
    print(f"Alignment valid: {summary['alignment_valid']}")
    print(f"Comparison complete: {summary['comparison_complete']}")
    print(f"Mean ResNet distance: {summary['mean_resnet_normalized_distance']}")
    print(f"Mean temporal distance: {summary['mean_temporal_normalized_distance']}")
    print(f"Mean flat hybrid distance: {summary['mean_flat_hybrid_normalized_distance']}")
    print(f"Mean balanced diagnostic score: {summary['mean_balanced_diagnostic_score']}")
    print(stored.result.warnings[0])


def command_compare_digests(args: argparse.Namespace, config: AppConfig) -> int:

    key_info = _hmac_key_from_args(args, config)
    comparison_config = _comparison_config_from_args(args, config)
    stored = compare_and_store_digests(
        reference_id=safe_video_id(args.reference_id),
        query_id=safe_video_id(args.query_id),
        authentication_record_root=config.paths.authentication_records,
        digest_root=config.paths.digests,
        comparison_root=config.paths.comparisons,
        key_info=key_info,
        config=comparison_config,
        algorithm=config.authentication.hmac.algorithm,
        overwrite=args.overwrite,
    )
    _print_stored_comparison_summary(stored)
    return 0


def command_compare_digests_batch(args: argparse.Namespace, config: AppConfig) -> int:

    key_info = _hmac_key_from_args(args, config)
    comparison_config = _comparison_config_from_args(args, config)
    for query_id in _safe_video_ids(args.query_ids):
        stored = compare_and_store_digests(
            reference_id=safe_video_id(args.reference_id),
            query_id=query_id,
            authentication_record_root=config.paths.authentication_records,
            digest_root=config.paths.digests,
            comparison_root=config.paths.comparisons,
            key_info=key_info,
            config=comparison_config,
            algorithm=config.authentication.hmac.algorithm,
            overwrite=args.overwrite,
        )
        summary = stored.result.video_summary
        status = "reused" if stored.cache_reused else "saved"
        print(
            f"{stored.result.comparison_id}: {status}, "
            f"matched={summary['matched_segment_count']}, "
            f"missing={summary['missing_segment_count']}, "
            f"extra={summary['extra_segment_count']}, "
            f"max_resnet={summary['maximum_resnet_normalized_distance']}, "
            f"max_temporal={summary['maximum_temporal_normalized_distance']}, "
            f"max_balanced={summary['maximum_balanced_diagnostic_score']}"
        )
    print("No acceptance threshold was applied.")
    return 0


def command_inspect_comparison(args: argparse.Namespace, config: AppConfig) -> int:

    reference_id = safe_video_id(args.reference_id)
    query_id = safe_video_id(args.query_id)
    manifest = inspect_comparison(config.paths.comparisons, reference_id, query_id)
    summary = manifest["video_level_summary"]
    print(f"Comparison ID: {manifest['comparison_id']}")
    print(f"Reference ID: {manifest['reference_video_id']}")
    print(f"Query ID: {manifest['query_video_id']}")
    print(f"Reference HMAC valid: {manifest['reference_hmac_verification_result']['hmac_valid']}")
    print(f"Normalization ID: {manifest['normalization_id']}")
    print(f"Quantization ID: {manifest['quantization_id']}")
    print(f"Matched segments: {summary['matched_segment_count']}")
    print(f"Missing segments: {summary['missing_segment_count']}")
    print(f"Extra segments: {summary['extra_segment_count']}")
    print(f"Timestamp mismatches: {summary['timestamp_mismatch_count']}")
    for segment in manifest["per_segment_results"]:
        print(
            f"Segment {segment['segment_id']}: "
            f"resnet={segment['resnet_normalized_distance']:.12f}, "
            f"temporal={segment['temporal_normalized_distance']:.12f}, "
            f"flat_hybrid={segment['flat_hybrid_normalized_distance']:.12f}, "
            f"balanced={segment['development_diagnostic_score']:.12f}, "
            f"attribution={segment['relative_stream_attribution']}"
        )
    print(f"Max ResNet segment: {summary['segment_id_with_maximum_resnet_distance']}")
    print(f"Max temporal segment: {summary['segment_id_with_maximum_temporal_distance']}")
    print(f"Max balanced-score segment: {summary['segment_id_with_maximum_balanced_diagnostic_score']}")
    print(manifest["no_threshold_warning"])
    return 0


def command_preprocess(args: argparse.Namespace, config: AppConfig) -> int:

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
        if args.command == "extract-temporal":
            return command_extract_temporal(args, config)
        if args.command == "extract-temporal-all":
            return command_extract_temporal_all(args, config)
        if args.command == "fit-normalization":
            return command_fit_normalization(args, config)
        if args.command == "normalize-features":
            return command_normalize_features(args, config)
        if args.command == "normalize-features-all":
            return command_normalize_features_all(args, config)
        if args.command == "inspect-normalization":
            return command_inspect_normalization(args, config)
        if args.command == "inspect-normalized-features":
            return command_inspect_normalized_features(args, config)
        if args.command == "create-quantizer":
            return command_create_quantizer(args, config)
        if args.command == "build-digest":
            return command_build_digest(args, config)
        if args.command == "build-digests":
            return command_build_digests(args, config)
        if args.command == "inspect-quantizer":
            return command_inspect_quantizer(args, config)
        if args.command == "inspect-digest":
            return command_inspect_digest(args, config)
        if args.command == "generate-hmac-key":
            return command_generate_hmac_key(args, config)
        if args.command == "protect-digest":
            return command_protect_digest(args, config)
        if args.command == "protect-digests":
            return command_protect_digests(args, config)
        if args.command == "verify-auth-record":
            return command_verify_auth_record(args, config)
        if args.command == "inspect-auth-record":
            return command_inspect_auth_record(args, config)
        if args.command == "compare-digests":
            return command_compare_digests(args, config)
        if args.command == "compare-digests-batch":
            return command_compare_digests_batch(args, config)
        if args.command == "inspect-comparison":
            return command_inspect_comparison(args, config)
    except (
        AuthenticationRecordError,
        CanonicalizationError,
        ComparisonStorageError,
        ConfigurationError,
        DeviceSelectionError,
        DigestError,
        DigestComparisonError,
        FeatureExtractionError,
        FeatureAlignmentError,
        FeatureFusionError,
        FFmpegToolError,
        HammingDistanceError,
        HMACAuthenticationError,
        NormalizationError,
        QuantizationError,
        SegmentAggregationError,
        SegmentAlignmentError,
        TemporalFeatureError,
        TemporalSamplingError,
        VideoMetadataError,
        FrameSamplingError,
        MissingVideoError,
        UnsupportedVideoError,
        NoVideoStreamError,
        ExistingOutputError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        log_dir = config.paths.logs if config is not None else Path("logs")
        logger = setup_logging(log_dir, "INFO", getattr(args, "verbose", False))
        logger.error("%s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2
