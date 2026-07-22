"""Feature stream alignment for segment-level ResNet and temporal features."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.features.feature_storage import sha256_file
from src.video.segmentation import load_segment_manifest


RESNET_SEGMENT_DIMENSION = 1024
TEMPORAL_SEGMENT_DIMENSION = 18


class FeatureAlignmentError(RuntimeError):
    """Raised when ResNet and temporal segment features cannot be aligned."""


@dataclass(frozen=True)
class AlignedFeatureSet:
    """Validated and segment-aligned feature streams for one video."""

    video_id: str
    segment_ids: np.ndarray
    segment_start_times: np.ndarray
    segment_end_times: np.ndarray
    resnet_features: np.ndarray
    temporal_features: np.ndarray
    resnet_source_path: Path
    temporal_source_path: Path
    resnet_source_sha256: str
    temporal_source_sha256: str
    warnings: list[str]

    @property
    def segment_count(self) -> int:
        """Return the number of aligned segments."""

        return int(self.segment_ids.shape[0])


def default_resnet_feature_path(resnet_root: str | Path, video_id: str) -> Path:
    """Return the default ResNet feature NPZ path for a video ID."""

    return Path(resnet_root) / video_id / f"{video_id}_resnet_features.npz"


def default_temporal_feature_path(temporal_root: str | Path, video_id: str) -> Path:
    """Return the default temporal feature NPZ path for a video ID."""

    return Path(temporal_root) / video_id / f"{video_id}_temporal_features.npz"


def _load_npz(path: Path, video_id: str, stream_name: str) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FeatureAlignmentError(f"{stream_name} feature file not found for {video_id}: {path}")
    try:
        with np.load(path) as payload:
            return {name: payload[name] for name in payload.files}
    except OSError as exc:
        raise FeatureAlignmentError(f"Could not load {stream_name} feature file for {video_id}: {path}") from exc


def _load_json(path: Path, video_id: str, stream_name: str) -> dict[str, Any]:
    if not path.exists():
        raise FeatureAlignmentError(
            f"{stream_name} manifest is required to confirm video ID for {video_id}: {path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    manifest_video_id = data.get("video_id")
    if manifest_video_id != video_id:
        raise FeatureAlignmentError(
            f"{stream_name} manifest video_id '{manifest_video_id}' does not match requested '{video_id}'."
        )
    return data


def _default_manifest_path(npz_path: Path, video_id: str, suffix: str) -> Path:
    return npz_path.with_name(f"{video_id}_{suffix}_manifest.json")


def _require_array(
    payload: dict[str, np.ndarray],
    key: str,
    video_id: str,
    stream_name: str,
) -> np.ndarray:
    if key not in payload:
        raise FeatureAlignmentError(f"{stream_name} feature file for {video_id} is missing array '{key}'.")
    return np.asarray(payload[key])


def _validate_segment_ids(segment_ids: np.ndarray, video_id: str, stream_name: str) -> np.ndarray:
    if segment_ids.ndim != 1:
        raise FeatureAlignmentError(f"{stream_name} segment IDs for {video_id} must be one-dimensional.")
    ids = segment_ids.astype(np.int64, copy=True)
    unique_ids, counts = np.unique(ids, return_counts=True)
    duplicates = unique_ids[counts > 1]
    if duplicates.size:
        raise FeatureAlignmentError(
            f"{stream_name} feature file for {video_id} has duplicate segment IDs: {duplicates.tolist()}."
        )
    return ids


def _validate_features(
    features: np.ndarray,
    segment_ids: np.ndarray,
    expected_dimension: int,
    video_id: str,
    stream_name: str,
) -> np.ndarray:
    if features.ndim != 2:
        raise FeatureAlignmentError(f"{stream_name} features for {video_id} must be a 2D array.")
    if features.shape[0] != segment_ids.shape[0]:
        raise FeatureAlignmentError(
            f"{stream_name} feature row count for {video_id} does not match segment ID count."
        )
    if features.shape[1] != expected_dimension:
        raise FeatureAlignmentError(
            f"{stream_name} feature dimension for {video_id} must be {expected_dimension}, got {features.shape[1]}."
        )
    values = features.astype(np.float32, copy=True)
    if not np.all(np.isfinite(values)):
        bad_rows = np.unique(np.argwhere(~np.isfinite(values))[:, 0]).tolist()
        bad_segments = segment_ids[bad_rows].tolist()
        raise FeatureAlignmentError(
            f"{stream_name} features for {video_id} contain non-finite values in segments {bad_segments}."
        )
    return values


def _segment_times_from_phase1_manifest(path: Path | None, video_id: str) -> dict[int, tuple[float, float]]:
    if path is None or not path.exists():
        return {}
    manifest = load_segment_manifest(path)
    if manifest.video_id != video_id:
        raise FeatureAlignmentError(
            f"Segment manifest video_id '{manifest.video_id}' does not match requested '{video_id}'."
        )
    times: dict[int, tuple[float, float]] = {}
    for segment in manifest.segments:
        if not segment.is_complete:
            continue
        if segment.segment_id in times:
            raise FeatureAlignmentError(
                f"Segment manifest for {video_id} has duplicate segment ID {segment.segment_id}."
            )
        times[segment.segment_id] = (segment.start_time_seconds, segment.end_time_seconds)
    return times


def _segment_times_from_temporal_manifest(data: dict[str, Any]) -> dict[int, tuple[float, float]]:
    times: dict[int, tuple[float, float]] = {}
    for record in data.get("segment_records", []):
        if not record.get("success", True):
            continue
        segment_id = int(record["segment_id"])
        times[segment_id] = (
            float(record["start_time_seconds"]),
            float(record["end_time_seconds"]),
        )
    return times


def load_aligned_features(
    video_id: str,
    resnet_feature_path: str | Path,
    temporal_feature_path: str | Path,
    segment_manifest_path: str | Path | None = None,
) -> AlignedFeatureSet:
    """Load, validate, and align ResNet and temporal segment features."""

    resnet_path = Path(resnet_feature_path).resolve()
    temporal_path = Path(temporal_feature_path).resolve()
    resnet_manifest = _load_json(
        _default_manifest_path(resnet_path, video_id, "resnet"),
        video_id,
        "ResNet",
    )
    temporal_manifest = _load_json(
        _default_manifest_path(temporal_path, video_id, "temporal"),
        video_id,
        "Temporal",
    )

    resnet_payload = _load_npz(resnet_path, video_id, "ResNet")
    temporal_payload = _load_npz(temporal_path, video_id, "Temporal")
    resnet_ids = _validate_segment_ids(
        _require_array(resnet_payload, "segment_ids", video_id, "ResNet"),
        video_id,
        "ResNet",
    )
    temporal_ids = _validate_segment_ids(
        _require_array(temporal_payload, "segment_ids", video_id, "Temporal"),
        video_id,
        "Temporal",
    )
    resnet_features = _validate_features(
        _require_array(resnet_payload, "segment_combined_embeddings", video_id, "ResNet"),
        resnet_ids,
        RESNET_SEGMENT_DIMENSION,
        video_id,
        "ResNet",
    )
    temporal_features = _validate_features(
        _require_array(temporal_payload, "segment_features", video_id, "Temporal"),
        temporal_ids,
        TEMPORAL_SEGMENT_DIMENSION,
        video_id,
        "Temporal",
    )

    resnet_set = set(resnet_ids.tolist())
    temporal_set = set(temporal_ids.tolist())
    missing_temporal = sorted(resnet_set - temporal_set)
    missing_resnet = sorted(temporal_set - resnet_set)
    if missing_temporal:
        raise FeatureAlignmentError(
            f"Temporal features for {video_id} are missing segment IDs: {missing_temporal}."
        )
    if missing_resnet:
        raise FeatureAlignmentError(
            f"ResNet features for {video_id} are missing segment IDs: {missing_resnet}."
        )

    sorted_segment_ids = np.asarray(sorted(resnet_set), dtype=np.int64)
    resnet_index = {int(segment_id): index for index, segment_id in enumerate(resnet_ids.tolist())}
    temporal_index = {int(segment_id): index for index, segment_id in enumerate(temporal_ids.tolist())}
    if sorted_segment_ids.size:
        ordered_resnet = np.vstack(
            [resnet_features[resnet_index[int(segment_id)]] for segment_id in sorted_segment_ids]
        )
        ordered_temporal = np.vstack(
            [temporal_features[temporal_index[int(segment_id)]] for segment_id in sorted_segment_ids]
        )
    else:
        ordered_resnet = np.empty((0, RESNET_SEGMENT_DIMENSION), dtype=np.float32)
        ordered_temporal = np.empty((0, TEMPORAL_SEGMENT_DIMENSION), dtype=np.float32)

    warnings: list[str] = []
    phase1_times = _segment_times_from_phase1_manifest(
        Path(segment_manifest_path).resolve() if segment_manifest_path else None,
        video_id,
    )
    temporal_times = _segment_times_from_temporal_manifest(temporal_manifest)
    times = phase1_times or temporal_times
    starts: list[float] = []
    ends: list[float] = []
    for segment_id in sorted_segment_ids.tolist():
        if int(segment_id) in times:
            start, end = times[int(segment_id)]
        else:
            start, end = np.nan, np.nan
            warnings.append(f"Segment {int(segment_id)} for {video_id} has no available start/end timestamps.")
        starts.append(float(start))
        ends.append(float(end))

    return AlignedFeatureSet(
        video_id=video_id,
        segment_ids=sorted_segment_ids,
        segment_start_times=np.asarray(starts, dtype=np.float64),
        segment_end_times=np.asarray(ends, dtype=np.float64),
        resnet_features=ordered_resnet.astype(np.float32, copy=False),
        temporal_features=ordered_temporal.astype(np.float32, copy=False),
        resnet_source_path=resnet_path,
        temporal_source_path=temporal_path,
        resnet_source_sha256=sha256_file(resnet_path),
        temporal_source_sha256=sha256_file(temporal_path),
        warnings=warnings
        + list(resnet_manifest.get("warnings", []))
        + list(temporal_manifest.get("warnings", [])),
    )
