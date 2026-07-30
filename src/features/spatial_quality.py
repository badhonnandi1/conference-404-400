"""Blur-aware spatial quality feature extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from src.features.feature_storage import sha256_file
from src.video.frame_sampling import generate_sample_timestamps
from src.video.metadata import ExistingOutputError, MissingVideoError, UnsupportedVideoError
from src.video.segmentation import SegmentManifest


SPATIAL_FRAME_WIDTH = 224
SPATIAL_FRAME_HEIGHT = 224
SPATIAL_METRIC_NAMES = [
    "variance_laplacian",
    "tenengrad_sharpness",
    "edge_density",
    "high_frequency_dct_energy_ratio",
    "local_contrast",
]
SPATIAL_AGGREGATIONS = ["mean", "standard_deviation", "minimum", "p10", "median"]
SPATIAL_SEGMENT_DIMENSION = len(SPATIAL_METRIC_NAMES) * len(SPATIAL_AGGREGATIONS)
SPATIAL_SEGMENT_FEATURE_NAMES = [
    f"{aggregation}_{metric}"
    for aggregation in SPATIAL_AGGREGATIONS
    for metric in SPATIAL_METRIC_NAMES
]
SPATIAL_PREPROCESSING_DESCRIPTION = (
    "BGR/RGB frame to grayscale float32 in [0,1], resize to 224x224 with INTER_AREA, "
    "then compute Laplacian variance, Tenengrad sharpness, Canny edge density, "
    "high-frequency DCT energy ratio, and local contrast."
)


class SpatialQualityError(RuntimeError):
    """Raised when spatial quality features cannot be extracted safely."""


@dataclass(frozen=True)
class SpatialQualityOutputPaths:
    """Output paths for one video's spatial quality artifacts."""

    output_dir: Path
    npz_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class SpatialFrameRecord:
    """Serializable per-frame spatial quality extraction record."""

    video_id: str
    segment_id: int
    frame_index: int
    requested_timestamp_seconds: float
    actual_timestamp_seconds: float | None
    feature_row_index: int | None
    feature_values: list[float] | None
    success: bool
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record."""

        return asdict(self)


@dataclass(frozen=True)
class SpatialSegmentRecord:
    """Serializable per-segment spatial quality aggregation record."""

    video_id: str
    segment_id: int
    start_time_seconds: float
    end_time_seconds: float
    expected_frame_count: int
    decoded_frame_count: int
    failed_frame_count: int
    segment_feature_row_index: int | None
    segment_feature_dimension: int
    success: bool
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record."""

        return asdict(self)


@dataclass(frozen=True)
class SpatialQualityResult:
    """Frame-level metrics and 25-dimensional segment features."""

    frame_features: np.ndarray
    frame_records: list[SpatialFrameRecord]
    frame_segment_ids: np.ndarray
    frame_indices: np.ndarray
    frame_requested_timestamps: np.ndarray
    frame_actual_timestamps: np.ndarray
    segment_ids: np.ndarray
    segment_features: np.ndarray
    segment_records: list[SpatialSegmentRecord]
    warnings: list[str]
    failures: list[str]


def spatial_quality_output_paths(output_root: str | Path, video_id: str) -> SpatialQualityOutputPaths:
    """Return deterministic output paths for one video's spatial quality features."""

    output_dir = Path(output_root) / video_id
    return SpatialQualityOutputPaths(
        output_dir=output_dir,
        npz_path=output_dir / f"{video_id}_spatial_quality_features.npz",
        manifest_path=output_dir / f"{video_id}_spatial_quality_manifest.json",
    )


def preprocess_spatial_frame(frame: np.ndarray) -> np.ndarray:
    """Convert a decoded frame to deterministic 224x224 grayscale float32."""

    values = np.asarray(frame)
    if values.ndim == 3:
        gray = cv2.cvtColor(values, cv2.COLOR_BGR2GRAY)
    elif values.ndim == 2:
        gray = values
    else:
        raise SpatialQualityError(f"Unsupported frame shape: {values.shape}.")
    resized = cv2.resize(gray, (SPATIAL_FRAME_WIDTH, SPATIAL_FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
    if resized.dtype == np.uint8:
        normalized = resized.astype(np.float32) / 255.0
    else:
        normalized = resized.astype(np.float32)
        max_value = float(np.max(normalized)) if normalized.size else 0.0
        if max_value > 1.5:
            normalized = normalized / 255.0
    normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)
    if not np.all(np.isfinite(normalized)):
        raise SpatialQualityError("Spatial preprocessing produced non-finite values.")
    return normalized


def _edge_magnitude(gray: np.ndarray) -> np.ndarray:
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(sobel_x, sobel_y)


def _high_frequency_dct_energy_ratio(gray: np.ndarray, low_frequency_size: int = 32) -> float:
    dct = cv2.dct(gray.astype(np.float32))
    energy = np.square(dct, dtype=np.float32)
    total = float(np.sum(energy))
    if total <= 1.0e-12:
        return 0.0
    low = int(max(1, min(low_frequency_size, gray.shape[0], gray.shape[1])))
    high_energy = total - float(np.sum(energy[:low, :low]))
    return float(np.clip(high_energy / total, 0.0, 1.0))


def _local_contrast(gray: np.ndarray, window_size: int = 15) -> float:
    mean = cv2.blur(gray, (window_size, window_size))
    mean_square = cv2.blur(np.square(gray, dtype=np.float32), (window_size, window_size))
    variance = np.maximum(mean_square - np.square(mean, dtype=np.float32), 0.0)
    return float(np.mean(np.sqrt(variance, dtype=np.float32)))


def calculate_spatial_quality_metrics(frame: np.ndarray) -> np.ndarray:
    """Calculate five blur-aware spatial quality metrics for one frame."""

    gray = preprocess_spatial_frame(frame)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    edge_magnitude = _edge_magnitude(gray)
    edges = cv2.Canny((gray * 255.0).astype(np.uint8), 50, 150)
    metrics = np.asarray(
        [
            float(np.var(laplacian)),
            float(np.mean(np.square(edge_magnitude, dtype=np.float32))),
            float(np.mean(edges > 0)),
            _high_frequency_dct_energy_ratio(gray),
            _local_contrast(gray),
        ],
        dtype=np.float32,
    )
    if metrics.shape != (len(SPATIAL_METRIC_NAMES),):
        raise SpatialQualityError(f"Unexpected spatial metric shape: {metrics.shape}.")
    if not np.all(np.isfinite(metrics)):
        raise SpatialQualityError("Spatial quality metrics contain non-finite values.")
    metrics[2] = np.clip(metrics[2], 0.0, 1.0)
    metrics[3] = np.clip(metrics[3], 0.0, 1.0)
    return metrics


def aggregate_spatial_segment_features(frame_features: np.ndarray) -> np.ndarray:
    """Aggregate frame metrics into the 25-dimensional segment vector."""

    matrix = np.asarray(frame_features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(SPATIAL_METRIC_NAMES):
        raise SpatialQualityError(
            f"Expected frame feature shape (n, {len(SPATIAL_METRIC_NAMES)}), got {matrix.shape}."
        )
    if matrix.shape[0] == 0:
        raise SpatialQualityError("Cannot aggregate spatial quality for a segment with zero valid frames.")
    values = np.concatenate(
        [
            np.mean(matrix, axis=0),
            np.std(matrix, axis=0, ddof=0),
            np.min(matrix, axis=0),
            np.percentile(matrix, 10, axis=0),
            np.median(matrix, axis=0),
        ]
    ).astype(np.float32)
    if values.shape != (SPATIAL_SEGMENT_DIMENSION,):
        raise SpatialQualityError(f"Unexpected spatial segment feature shape: {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise SpatialQualityError("Spatial segment aggregation produced non-finite values.")
    return values


def _measured_timestamp(capture: Any) -> float | None:
    timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    if not np.isfinite(timestamp_ms) or timestamp_ms < 0:
        return None
    return round(timestamp_ms / 1000.0, 6)


def extract_spatial_quality_from_video(
    video_id: str,
    source_video_path: str | Path,
    segment_manifest: SegmentManifest,
) -> SpatialQualityResult:
    """Decode deterministic segment frames and extract spatial quality features."""

    source = Path(source_video_path)
    if not source.exists():
        raise MissingVideoError(f"Source video not found: {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise UnsupportedVideoError(f"OpenCV could not open '{source}'.")

    frame_features: list[np.ndarray] = []
    frame_records: list[SpatialFrameRecord] = []
    segment_features: list[np.ndarray] = []
    segment_records: list[SpatialSegmentRecord] = []
    segment_ids: list[int] = []
    warnings: list[str] = []
    failures: list[str] = []
    try:
        for segment in segment_manifest.segments:
            if not segment.is_complete:
                continue
            timestamps = generate_sample_timestamps(segment, segment_manifest.sample_frames_per_second)
            segment_rows: list[np.ndarray] = []
            failed_count = 0
            for frame_index, timestamp in enumerate(timestamps):
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                decoded, frame = capture.read()
                actual = _measured_timestamp(capture)
                if not decoded or frame is None:
                    failed_count += 1
                    message = f"Failed to decode spatial-quality frame at {timestamp:.3f}s."
                    frame_records.append(
                        SpatialFrameRecord(
                            video_id=video_id,
                            segment_id=segment.segment_id,
                            frame_index=frame_index,
                            requested_timestamp_seconds=timestamp,
                            actual_timestamp_seconds=actual,
                            feature_row_index=None,
                            feature_values=None,
                            success=False,
                            error_message=message,
                        )
                    )
                    continue
                try:
                    features = calculate_spatial_quality_metrics(frame)
                except SpatialQualityError as exc:
                    failed_count += 1
                    frame_records.append(
                        SpatialFrameRecord(
                            video_id=video_id,
                            segment_id=segment.segment_id,
                            frame_index=frame_index,
                            requested_timestamp_seconds=timestamp,
                            actual_timestamp_seconds=actual,
                            feature_row_index=None,
                            feature_values=None,
                            success=False,
                            error_message=str(exc),
                        )
                    )
                    continue
                row_index = len(frame_features)
                frame_features.append(features)
                segment_rows.append(features)
                frame_records.append(
                    SpatialFrameRecord(
                        video_id=video_id,
                        segment_id=segment.segment_id,
                        frame_index=frame_index,
                        requested_timestamp_seconds=timestamp,
                        actual_timestamp_seconds=actual,
                        feature_row_index=row_index,
                        feature_values=[float(value) for value in features],
                        success=True,
                    )
                )
            if not segment_rows:
                message = f"Segment {segment.segment_id} has zero valid spatial-quality frames."
                failures.append(message)
                segment_records.append(
                    SpatialSegmentRecord(
                        video_id=video_id,
                        segment_id=segment.segment_id,
                        start_time_seconds=segment.start_time_seconds,
                        end_time_seconds=segment.end_time_seconds,
                        expected_frame_count=len(timestamps),
                        decoded_frame_count=0,
                        failed_frame_count=failed_count,
                        segment_feature_row_index=None,
                        segment_feature_dimension=SPATIAL_SEGMENT_DIMENSION,
                        success=False,
                        error_message=message,
                    )
                )
                continue
            features = aggregate_spatial_segment_features(np.vstack(segment_rows))
            segment_ids.append(segment.segment_id)
            segment_features.append(features)
            if failed_count:
                warnings.append(f"Segment {segment.segment_id} had {failed_count} failed spatial frame decodes.")
            segment_records.append(
                SpatialSegmentRecord(
                    video_id=video_id,
                    segment_id=segment.segment_id,
                    start_time_seconds=segment.start_time_seconds,
                    end_time_seconds=segment.end_time_seconds,
                    expected_frame_count=len(timestamps),
                    decoded_frame_count=len(segment_rows),
                    failed_frame_count=failed_count,
                    segment_feature_row_index=len(segment_features) - 1,
                    segment_feature_dimension=SPATIAL_SEGMENT_DIMENSION,
                    success=True,
                )
            )
    finally:
        capture.release()

    if failures:
        raise SpatialQualityError("; ".join(failures))

    frame_array = (
        np.vstack(frame_features).astype(np.float32)
        if frame_features
        else np.empty((0, len(SPATIAL_METRIC_NAMES)), dtype=np.float32)
    )
    segment_array = (
        np.vstack(segment_features).astype(np.float32)
        if segment_features
        else np.empty((0, SPATIAL_SEGMENT_DIMENSION), dtype=np.float32)
    )
    successful_records = [record for record in frame_records if record.success]
    return SpatialQualityResult(
        frame_features=frame_array,
        frame_records=frame_records,
        frame_segment_ids=np.asarray([record.segment_id for record in successful_records], dtype=np.int64),
        frame_indices=np.asarray([record.frame_index for record in successful_records], dtype=np.int64),
        frame_requested_timestamps=np.asarray(
            [record.requested_timestamp_seconds for record in successful_records],
            dtype=np.float64,
        ),
        frame_actual_timestamps=np.asarray(
            [
                np.nan if record.actual_timestamp_seconds is None else record.actual_timestamp_seconds
                for record in successful_records
            ],
            dtype=np.float64,
        ),
        segment_ids=np.asarray(segment_ids, dtype=np.int64),
        segment_features=segment_array,
        segment_records=segment_records,
        warnings=warnings,
        failures=failures,
    )


def build_spatial_cache_key(source_video_path: str | Path, segment_manifest_path: str | Path) -> dict[str, Any]:
    """Build cache metadata for spatial quality extraction."""

    return {
        "source_video_sha256": sha256_file(source_video_path),
        "source_segment_manifest_sha256": sha256_file(segment_manifest_path),
        "preprocessing_description": SPATIAL_PREPROCESSING_DESCRIPTION,
        "frame_dimensions": [SPATIAL_FRAME_WIDTH, SPATIAL_FRAME_HEIGHT],
        "metric_names": SPATIAL_METRIC_NAMES,
        "segment_aggregations": SPATIAL_AGGREGATIONS,
        "segment_feature_dimension": SPATIAL_SEGMENT_DIMENSION,
    }


def spatial_manifest_matches(manifest: dict[str, Any], cache_key: dict[str, Any]) -> bool:
    """Return whether an existing spatial manifest is reusable."""

    return all(manifest.get(key) == value for key, value in cache_key.items())


def ensure_can_write_spatial(
    paths: SpatialQualityOutputPaths,
    overwrite: bool,
    cache_key: dict[str, Any],
) -> bool:
    """Return True when an existing spatial quality output can be reused."""

    if not paths.npz_path.exists() and not paths.manifest_path.exists():
        return False
    if overwrite:
        return False
    if paths.npz_path.exists() and paths.manifest_path.exists():
        with paths.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if spatial_manifest_matches(manifest, cache_key):
            return True
    raise ExistingOutputError(
        f"Spatial quality outputs already exist under {paths.output_dir}. "
        "Use --overwrite to regenerate because cache metadata did not match."
    )


def save_spatial_quality_npz(
    paths: SpatialQualityOutputPaths,
    result: SpatialQualityResult,
    overwrite: bool = False,
) -> Path:
    """Save spatial quality arrays in compressed NumPy format."""

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    if paths.npz_path.exists() and not overwrite:
        raise ExistingOutputError(f"Spatial quality NPZ already exists: {paths.npz_path}.")
    np.savez_compressed(
        paths.npz_path,
        frame_features=result.frame_features.astype(np.float32),
        frame_segment_ids=result.frame_segment_ids.astype(np.int64),
        frame_indices=result.frame_indices.astype(np.int64),
        frame_requested_timestamps=result.frame_requested_timestamps.astype(np.float64),
        frame_actual_timestamps=result.frame_actual_timestamps.astype(np.float64),
        segment_ids=result.segment_ids.astype(np.int64),
        segment_features=result.segment_features.astype(np.float32),
    )
    return paths.npz_path


def save_spatial_quality_manifest(
    manifest: dict[str, Any],
    paths: SpatialQualityOutputPaths,
    overwrite: bool = False,
) -> Path:
    """Save a spatial quality manifest as JSON."""

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    if paths.manifest_path.exists() and not overwrite:
        raise ExistingOutputError(f"Spatial quality manifest already exists: {paths.manifest_path}.")
    with paths.manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return paths.manifest_path


def build_spatial_quality_manifest(
    video_id: str,
    source_video_path: str | Path,
    segment_manifest_path: str | Path,
    result: SpatialQualityResult,
    paths: SpatialQualityOutputPaths,
    npz_sha256: str,
    processing_time_seconds: float,
) -> dict[str, Any]:
    """Build JSON metadata for spatial quality artifacts."""

    return {
        "video_id": video_id,
        "source_video_path": str(Path(source_video_path).resolve()),
        "segment_manifest_path": str(Path(segment_manifest_path).resolve()),
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "preprocessing_description": SPATIAL_PREPROCESSING_DESCRIPTION,
        "frame_dimensions": [SPATIAL_FRAME_WIDTH, SPATIAL_FRAME_HEIGHT],
        "metric_names": SPATIAL_METRIC_NAMES,
        "segment_aggregations": SPATIAL_AGGREGATIONS,
        "segment_feature_names": SPATIAL_SEGMENT_FEATURE_NAMES,
        "segment_feature_dimension": SPATIAL_SEGMENT_DIMENSION,
        "frame_count": int(result.frame_features.shape[0]),
        "segment_count": int(result.segment_features.shape[0]),
        "frame_records": [record.to_dict() for record in result.frame_records],
        "segment_records": [record.to_dict() for record in result.segment_records],
        "output_npz_path": str(paths.npz_path.resolve()),
        "npz_sha256": npz_sha256,
        "processing_time_seconds": float(processing_time_seconds),
        "warnings": list(result.warnings),
        "failures": list(result.failures),
    }


def load_spatial_quality_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load spatial quality arrays from NPZ."""

    with np.load(Path(path), allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def extract_and_store_spatial_quality(
    video_id: str,
    source_video_path: str | Path,
    segment_manifest_path: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
) -> tuple[SpatialQualityResult | None, dict[str, Any], SpatialQualityOutputPaths, bool]:
    """Extract or reuse one video's spatial quality features."""

    paths = spatial_quality_output_paths(output_root, video_id)
    cache_key = build_spatial_cache_key(source_video_path, segment_manifest_path)
    if ensure_can_write_spatial(paths, overwrite, cache_key):
        with paths.manifest_path.open("r", encoding="utf-8") as handle:
            return None, json.load(handle), paths, True
    from src.video.segmentation import load_segment_manifest

    segment_manifest = load_segment_manifest(segment_manifest_path)
    started = perf_counter()
    result = extract_spatial_quality_from_video(video_id, source_video_path, segment_manifest)
    save_spatial_quality_npz(paths, result, overwrite=overwrite)
    npz_checksum = sha256_file(paths.npz_path)
    manifest = build_spatial_quality_manifest(
        video_id=video_id,
        source_video_path=source_video_path,
        segment_manifest_path=segment_manifest_path,
        result=result,
        paths=paths,
        npz_sha256=npz_checksum,
        processing_time_seconds=perf_counter() - started,
    )
    manifest.update(cache_key)
    save_spatial_quality_manifest(manifest, paths, overwrite=overwrite)
    return result, manifest, paths, False
