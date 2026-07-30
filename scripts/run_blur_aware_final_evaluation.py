#!/usr/bin/env python3
"""Run the blur-aware final six-source evaluation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

_REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str((_REPO_ROOT_FOR_IMPORTS / "data" / "tmp" / "matplotlib_cache").resolve()),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = _REPO_ROOT_FOR_IMPORTS
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_multisource_versions_evaluation import (
    BENIGN_TRANSFORMATIONS,
    TAMPERED_TRANSFORMATIONS,
    augment_registry_metadata,
    discover_sources,
    metadata_comparison,
    sha256_baseline,
    write_csv_rows,
    write_json,
)
from src.authentication.blur_aware_v2 import (
    DEFAULT_V2_WEIGHTS,
    PIPELINE_V2_ID,
    V2_CONTINUOUS_DIMENSIONS,
    V2_DIGEST_LENGTHS,
    V2_HYBRID_DIGEST_LENGTH,
    V2_SCHEMA_VERSION,
    V2_SPATIAL_DIGEST_LENGTH,
    V2_TEMPORAL_DIGEST_LENGTH,
    build_v2_authentication_record,
    build_v2_digest_bundle,
    compare_v2_digest_bundles,
    derive_v2_quantization_parameters,
    fit_three_stream_normalizers,
    save_json,
    save_v2_digest_npz,
    v2_digest_payload,
    verify_v2_authentication_record,
)
from src.authentication.hmac_auth import load_hmac_key
from src.authentication.quantization import (
    assign_temporal_bins,
    build_hybrid_digest,
    gray_encode_temporal_bins,
    quantize_resnet_binary,
)
from src.config import load_config
from src.features.feature_storage import feature_output_paths, sha256_file
from src.features.normalization import RobustNormalizer
from src.features.spatial_quality import (
    SPATIAL_METRIC_NAMES,
    SPATIAL_SEGMENT_DIMENSION,
    SPATIAL_SEGMENT_FEATURE_NAMES,
    calculate_spatial_quality_metrics,
    extract_and_store_spatial_quality,
    load_spatial_quality_npz,
    spatial_quality_output_paths,
)
from src.features.temporal_storage import temporal_output_paths
from src.video.metadata import safe_video_id
from src.video.segmentation import load_segment_manifest
from src.verification.hamming import hamming_distance


EXPERIMENT_ID = "blur_aware_final_evaluation"
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "Dataset -this is the final"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "reports" / EXPERIMENT_ID
DEFAULT_CONFIG = REPO_ROOT / "configs" / "multisource_versions_evaluation.yaml"
SPATIAL_FEATURE_ROOT = REPO_ROOT / "data" / "features" / "spatial_quality"
STOP_FREE_BYTES = int(1.5 * 1024**3)
CALIBRATION_TRANSFORMATIONS = {
    "trusted_reference",
    "avi_conversion",
    "mov_conversion",
    "resize_480p",
    "resize_720p",
}
NORMAL_LABELS = {"normal"}
ABNORMAL_LABELS = {"abnormal"}
V1_WEIGHT_GRID = [
    {"resnet": 0.5, "temporal": 0.5},
    {"resnet": 0.7, "temporal": 0.3},
    {"resnet": 0.3, "temporal": 0.7},
]
V2_WEIGHT_GRID = [
    dict(DEFAULT_V2_WEIGHTS),
    {"resnet": 0.3, "temporal": 0.2, "spatial": 0.5},
    {"resnet": 0.5, "temporal": 0.25, "spatial": 0.25},
]
MARGIN_GRID = [1.5, 3.0]
COMMAND_ENV = {
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "VECLIB_MAXIMUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "TOKENIZERS_PARALLELISM": "false",
    "CUDA_VISIBLE_DEVICES": "",
}


class FinalEvaluationError(RuntimeError):
    """Raised when the final evaluation cannot proceed safely."""


@dataclass(frozen=True)
class RawFeatureSet:
    """Aligned raw feature streams for one video."""

    video_id: str
    source_id: str
    segment_ids: np.ndarray
    segment_start_times: np.ndarray
    segment_end_times: np.ndarray
    resnet: np.ndarray
    temporal: np.ndarray
    spatial: np.ndarray


@dataclass(frozen=True)
class V1DigestBits:
    """V1 digest bits built from fold-specific normalizers."""

    video_id: str
    segment_ids: np.ndarray
    resnet_bits: np.ndarray
    temporal_bits: np.ndarray
    hybrid_bits: np.ndarray


@dataclass(frozen=True)
class ThresholdModel:
    """Segment-level thresholds selected inside a fold."""

    score_threshold: float
    blur_loss_threshold: float
    margin_multiplier: float
    weights: dict[str, float]
    profile_thresholds: dict[str, float]
    profile_counts: dict[str, int]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id() -> str:
    return "blur_aware_final_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def disk_free_bytes() -> int:
    return shutil.disk_usage("/").free


def check_disk_stop() -> None:
    free = disk_free_bytes()
    if free < STOP_FREE_BYTES:
        raise FinalEvaluationError(f"Free disk space dropped below 1.5 GiB stop condition: {free} bytes.")


def append_runtime_log(output_dir: Path, row: dict[str, Any]) -> None:
    path = output_dir / "runtime_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "experiment_id",
        "run_id",
        "source_id",
        "video_id",
        "stage",
        "start_time",
        "end_time",
        "duration_seconds",
        "status",
        "cache_reused",
        "failure_reason",
        "free_space_bytes",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def controlled_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(COMMAND_ENV)
    env["MPLCONFIGDIR"] = str((REPO_ROOT / "data" / "tmp" / "matplotlib_cache").resolve())
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return env


def run_command(
    *,
    command: list[str],
    source_id: str,
    video_id: str,
    stage: str,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    check_disk_stop()
    start_time = utc_now()
    start = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=controlled_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    duration = time.perf_counter() - start
    status = "completed" if result.returncode == 0 else "failed"
    failure = "" if result.returncode == 0 else (result.stderr.strip() or result.stdout.strip() or str(result.returncode))[-500:]
    cache_reused = "Reusing cached" in result.stdout or "reused cached" in result.stdout
    row = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "source_id": source_id,
        "video_id": video_id,
        "stage": stage,
        "start_time": start_time,
        "end_time": utc_now(),
        "duration_seconds": round(duration, 6),
        "status": status,
        "cache_reused": cache_reused,
        "failure_reason": failure,
        "free_space_bytes": disk_free_bytes(),
    }
    append_runtime_log(output_dir, row)
    if result.returncode != 0:
        raise FinalEvaluationError(f"{stage} failed for {video_id}: {failure}")
    return {**row, "stdout": result.stdout.strip(), "stderr": result.stderr.strip(), "command": " ".join(command)}


def sleep_after(
    *,
    stage: str,
    seconds: float,
    source_id: str,
    output_dir: Path,
    run_id: str,
    cache_reused: bool,
) -> None:
    if cache_reused or seconds <= 0:
        return
    start = time.perf_counter()
    start_time = utc_now()
    time.sleep(seconds)
    append_runtime_log(
        output_dir,
        {
            "experiment_id": EXPERIMENT_ID,
            "run_id": run_id,
            "source_id": source_id,
            "video_id": "SYSTEM",
            "stage": f"sleep_after_{stage}",
            "start_time": start_time,
            "end_time": utc_now(),
            "duration_seconds": round(time.perf_counter() - start, 6),
            "status": "completed",
            "cache_reused": False,
            "failure_reason": "",
            "free_space_bytes": disk_free_bytes(),
        },
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise FinalEvaluationError(f"Expected JSON object: {path}")
    return data


def stage_cached(config: Any, video: dict[str, Any], stage: str) -> bool:
    video_id = video["video_id"]
    if stage == "preprocess":
        metadata = config.paths.metadata / f"{video_id}_metadata.json"
        segments = config.paths.manifests / f"{video_id}_segments.json"
        frames = config.paths.manifests / f"{video_id}_frames.json"
        if not (metadata.exists() and segments.exists() and frames.exists()):
            return False
        try:
            return read_json(metadata).get("absolute_path") == video["absolute_path"]
        except (OSError, json.JSONDecodeError, FinalEvaluationError):
            return False
    if stage == "resnet":
        paths = feature_output_paths(config.paths.resnet_features, video_id)
        return paths.npz_path.exists() and paths.manifest_path.exists()
    if stage == "temporal":
        paths = temporal_output_paths(config.paths.temporal_features, video_id)
        return paths.npz_path.exists() and paths.manifest_path.exists()
    if stage == "spatial_quality":
        paths = spatial_quality_output_paths(SPATIAL_FEATURE_ROOT, video_id)
        return paths.npz_path.exists() and paths.manifest_path.exists()
    return False


def record_cached(output_dir: Path, run_id: str, source_id: str, video_id: str, stage: str) -> dict[str, Any]:
    row = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "source_id": source_id,
        "video_id": video_id,
        "stage": stage,
        "start_time": utc_now(),
        "end_time": utc_now(),
        "duration_seconds": 0.0,
        "status": "completed",
        "cache_reused": True,
        "failure_reason": "",
        "free_space_bytes": disk_free_bytes(),
    }
    append_runtime_log(output_dir, row)
    return row


def extract_raw_features_for_video(
    *,
    video: dict[str, Any],
    config_path: Path,
    config: Any,
    output_dir: Path,
    run_id: str,
) -> list[dict[str, Any]]:
    python = str(REPO_ROOT / ".venv" / "bin" / "python")
    rows: list[dict[str, Any]] = []
    stages = [
        (
            "preprocess",
            [
                python,
                "main.py",
                "preprocess",
                "--config",
                str(config_path),
                "--video",
                video["relative_path"],
                "--video-id",
                video["video_id"],
            ],
            8.0,
        ),
        (
            "resnet",
            [
                python,
                "main.py",
                "extract-resnet",
                "--config",
                str(config_path),
                "--video-id",
                video["video_id"],
                "--batch-size",
                "2",
                "--device",
                "cpu",
            ],
            12.0,
        ),
        (
            "temporal",
            [
                python,
                "main.py",
                "extract-temporal",
                "--config",
                str(config_path),
                "--video-id",
                video["video_id"],
                "--video-path",
                video["relative_path"],
            ],
            8.0,
        ),
    ]
    for stage, command, pause in stages:
        if stage_cached(config, video, stage):
            row = record_cached(output_dir, run_id, video["source_id"], video["video_id"], stage)
        else:
            row = run_command(
                command=command,
                source_id=video["source_id"],
                video_id=video["video_id"],
                stage=stage,
                output_dir=output_dir,
                run_id=run_id,
            )
        rows.append(row)
        sleep_after(
            stage=stage,
            seconds=pause,
            source_id=video["source_id"],
            output_dir=output_dir,
            run_id=run_id,
            cache_reused=bool(row["cache_reused"]),
        )

    if stage_cached(config, video, "spatial_quality"):
        row = record_cached(output_dir, run_id, video["source_id"], video["video_id"], "spatial_quality")
    else:
        check_disk_stop()
        start_time = utc_now()
        start = time.perf_counter()
        try:
            _, _, _, cache_reused = extract_and_store_spatial_quality(
                video_id=video["video_id"],
                source_video_path=Path(video["absolute_path"]),
                segment_manifest_path=config.paths.manifests / f"{video['video_id']}_segments.json",
                output_root=SPATIAL_FEATURE_ROOT,
                overwrite=False,
            )
            status = "completed"
            failure = ""
        except Exception as exc:
            status = "failed"
            failure = str(exc)[-500:]
            cache_reused = False
        row = {
            "experiment_id": EXPERIMENT_ID,
            "run_id": run_id,
            "source_id": video["source_id"],
            "video_id": video["video_id"],
            "stage": "spatial_quality",
            "start_time": start_time,
            "end_time": utc_now(),
            "duration_seconds": round(time.perf_counter() - start, 6),
            "status": status,
            "cache_reused": cache_reused,
            "failure_reason": failure,
            "free_space_bytes": disk_free_bytes(),
        }
        append_runtime_log(output_dir, row)
        if status != "completed":
            raise FinalEvaluationError(f"spatial_quality failed for {video['video_id']}: {failure}")
    rows.append(row)
    sleep_after(
        stage="spatial_quality",
        seconds=5.0,
        source_id=video["source_id"],
        output_dir=output_dir,
        run_id=run_id,
        cache_reused=bool(row["cache_reused"]),
    )
    return rows


def ensure_raw_feature_cache(
    *,
    registry: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    config_path: Path,
    output_dir: Path,
    run_id: str,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    rows: list[dict[str, Any]] = []
    valid_sources = {row["source_id"] for row in source_rows if row["status"] == "valid"}
    for source_id in sorted(valid_sources):
        source_videos = sorted(
            [row for row in registry if row["source_id"] == source_id],
            key=lambda row: (row["transformation_type"], row["filename"].lower()),
        )
        for video in source_videos:
            rows.extend(
                extract_raw_features_for_video(
                    video=video,
                    config_path=config_path,
                    config=config,
                    output_dir=output_dir,
                    run_id=run_id,
                )
            )
        sleep_after(
            stage="source_complete",
            seconds=30.0,
            source_id=source_id,
            output_dir=output_dir,
            run_id=run_id,
            cache_reused=False,
        )
    return rows


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def load_raw_feature_set(video: dict[str, Any], config: Any) -> RawFeatureSet:
    video_id = video["video_id"]
    resnet_paths = feature_output_paths(config.paths.resnet_features, video_id)
    temporal_paths = temporal_output_paths(config.paths.temporal_features, video_id)
    spatial_paths = spatial_quality_output_paths(SPATIAL_FEATURE_ROOT, video_id)
    segment_manifest = load_segment_manifest(config.paths.manifests / f"{video_id}_segments.json")
    resnet = _npz(resnet_paths.npz_path)
    temporal = _npz(temporal_paths.npz_path)
    spatial = load_spatial_quality_npz(spatial_paths.npz_path)
    ids = sorted(
        set(np.asarray(resnet["segment_ids"], dtype=np.int64).tolist())
        & set(np.asarray(temporal["segment_ids"], dtype=np.int64).tolist())
        & set(np.asarray(spatial["segment_ids"], dtype=np.int64).tolist())
    )
    if not ids:
        raise FinalEvaluationError(f"No common feature segment IDs for {video_id}.")
    resnet_index = {int(value): index for index, value in enumerate(np.asarray(resnet["segment_ids"], dtype=np.int64))}
    temporal_index = {int(value): index for index, value in enumerate(np.asarray(temporal["segment_ids"], dtype=np.int64))}
    spatial_index = {int(value): index for index, value in enumerate(np.asarray(spatial["segment_ids"], dtype=np.int64))}
    segment_times = {
        int(segment.segment_id): (float(segment.start_time_seconds), float(segment.end_time_seconds))
        for segment in segment_manifest.segments
        if segment.is_complete
    }
    starts = [segment_times[int(segment_id)][0] for segment_id in ids]
    ends = [segment_times[int(segment_id)][1] for segment_id in ids]
    return RawFeatureSet(
        video_id=video_id,
        source_id=video["source_id"],
        segment_ids=np.asarray(ids, dtype=np.int64),
        segment_start_times=np.asarray(starts, dtype=np.float64),
        segment_end_times=np.asarray(ends, dtype=np.float64),
        resnet=np.vstack([resnet["segment_combined_embeddings"][resnet_index[int(segment_id)]] for segment_id in ids]).astype(np.float32),
        temporal=np.vstack([temporal["segment_features"][temporal_index[int(segment_id)]] for segment_id in ids]).astype(np.float32),
        spatial=np.vstack([spatial["segment_features"][spatial_index[int(segment_id)]] for segment_id in ids]).astype(np.float32),
    )


def fit_v1_normalizers(
    training_video_ids: list[str],
    features: dict[str, RawFeatureSet],
) -> tuple[RobustNormalizer, RobustNormalizer]:
    resnet = np.vstack([features[video_id].resnet for video_id in training_video_ids])
    temporal = np.vstack([features[video_id].temporal for video_id in training_video_ids])
    return RobustNormalizer.fit(resnet), RobustNormalizer.fit(temporal)


def build_v1_digest(video_id: str, feature: RawFeatureSet, resnet_norm: RobustNormalizer, temporal_norm: RobustNormalizer) -> V1DigestBits:
    resnet_values = resnet_norm.transform(feature.resnet)
    temporal_values = temporal_norm.transform(feature.temporal)
    resnet_bits = quantize_resnet_binary(resnet_values, np.zeros(feature.resnet.shape[1], dtype=np.float32))
    temporal_bins = assign_temporal_bins(
        temporal_values,
        ((temporal_norm.q1 - temporal_norm.median) / temporal_norm.safe_scale).astype(np.float32),
        np.zeros(feature.temporal.shape[1], dtype=np.float32),
        ((temporal_norm.q3 - temporal_norm.median) / temporal_norm.safe_scale).astype(np.float32),
    )
    temporal_bits = gray_encode_temporal_bins(temporal_bins)
    hybrid_bits = build_hybrid_digest(resnet_bits, temporal_bits)
    return V1DigestBits(
        video_id=video_id,
        segment_ids=feature.segment_ids.copy(),
        resnet_bits=resnet_bits,
        temporal_bits=temporal_bits,
        hybrid_bits=hybrid_bits,
    )


def compare_v1(
    reference: V1DigestBits,
    query: V1DigestBits,
    weights: dict[str, float],
) -> list[dict[str, Any]]:
    ref_index = {int(segment_id): index for index, segment_id in enumerate(reference.segment_ids.tolist())}
    query_index = {int(segment_id): index for index, segment_id in enumerate(query.segment_ids.tolist())}
    rows: list[dict[str, Any]] = []
    for segment_id in sorted(set(ref_index) & set(query_index)):
        ri = ref_index[segment_id]
        qi = query_index[segment_id]
        resnet = hamming_distance(reference.resnet_bits[ri], query.resnet_bits[qi])
        temporal = hamming_distance(reference.temporal_bits[ri], query.temporal_bits[qi])
        hybrid = hamming_distance(reference.hybrid_bits[ri], query.hybrid_bits[qi])
        if hybrid.raw_distance != resnet.raw_distance + temporal.raw_distance:
            raise FinalEvaluationError("V1 hybrid raw distance invariant failed.")
        score = (
            weights["resnet"] * resnet.normalized_distance
            + weights["temporal"] * temporal.normalized_distance
        ) / max(weights["resnet"] + weights["temporal"], 1.0e-12)
        rows.append(
            {
                "segment_id": int(segment_id),
                "resnet_raw_distance": int(resnet.raw_distance),
                "temporal_raw_distance": int(temporal.raw_distance),
                "hybrid_raw_distance": int(hybrid.raw_distance),
                "resnet_normalized_distance": float(resnet.normalized_distance),
                "temporal_normalized_distance": float(temporal.normalized_distance),
                "hybrid_normalized_distance": float(hybrid.normalized_distance),
                "score": float(score),
            }
        )
    return rows


def median_absolute_deviation(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def threshold_from_values(values: list[float], margin_multiplier: float, one_unit: float) -> float:
    if not values:
        return 1.0
    matrix = np.asarray(values, dtype=np.float64)
    return float(min(1.0, max(0.0, float(np.max(matrix)) + max(margin_multiplier * median_absolute_deviation(matrix), one_unit))))


def compression_profile(video: dict[str, Any]) -> str:
    width = video.get("width")
    height = video.get("height")
    ext = str(video.get("file_extension") or Path(video["filename"]).suffix).lower()
    if width and height:
        pixels = int(width) * int(height)
        if pixels <= 900 * 520:
            bucket = "480p"
        elif pixels <= 1300 * 760:
            bucket = "720p"
        else:
            bucket = "source_res"
    else:
        bucket = "unknown_res"
    return f"{ext or 'unknown'}:{bucket}"


def normal_training_videos(registry: list[dict[str, Any]], source_ids: list[str]) -> list[str]:
    return [
        row["video_id"]
        for row in registry
        if row["source_id"] in source_ids and row["transformation_type"] in CALIBRATION_TRANSFORMATIONS
    ]


def reference_by_source(registry: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for row in registry:
        if row["expected_category"] == "reference":
            refs[row["source_id"]] = row
    return refs


def structural_issue(reference_segments: np.ndarray, query_segments: np.ndarray) -> tuple[bool, int, int]:
    ref = set(np.asarray(reference_segments, dtype=np.int64).tolist())
    qry = set(np.asarray(query_segments, dtype=np.int64).tolist())
    return bool(ref - qry or qry - ref), len(ref - qry), len(qry - ref)


def build_fold_digests(
    *,
    fold_id: str,
    training_sources: list[str],
    registry: list[dict[str, Any]],
    features: dict[str, RawFeatureSet],
    output_dir: Path,
    key_info: Any | None = None,
    held_out_source: str | None = None,
) -> tuple[dict[str, V1DigestBits], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    train_video_ids = normal_training_videos(registry, training_sources)
    if not train_video_ids:
        raise FinalEvaluationError(f"{fold_id} has no benign/reference training videos.")
    v1_resnet_norm, v1_temporal_norm = fit_v1_normalizers(train_video_ids, features)
    three = fit_three_stream_normalizers(
        normalization_id=f"{fold_id}_THREE_STREAM_NORMALIZATION_V2",
        training_video_ids=train_video_ids,
        training_source_ids=training_sources,
        resnet_features=np.vstack([features[video_id].resnet for video_id in train_video_ids]),
        temporal_features=np.vstack([features[video_id].temporal for video_id in train_video_ids]),
        spatial_features=np.vstack([features[video_id].spatial for video_id in train_video_ids]),
    )
    quantizer = derive_v2_quantization_parameters(three, quantization_id=f"{fold_id}_QUANTIZATION_V2")
    fold_dir = output_dir / "fold_artifacts" / fold_id
    save_json(fold_dir / "normalization_manifest.json", three.to_manifest())
    save_json(fold_dir / "quantization_manifest.json", quantizer.to_manifest())
    v1_digests: dict[str, V1DigestBits] = {}
    v2_digests: dict[str, Any] = {}
    normalized: dict[str, dict[str, np.ndarray]] = {}
    hmac_rows: list[dict[str, Any]] = []
    for row in registry:
        feature = features[row["video_id"]]
        v1_digests[row["video_id"]] = build_v1_digest(row["video_id"], feature, v1_resnet_norm, v1_temporal_norm)
        rn, tn, sn, combined = (
            three.resnet.transform(feature.resnet),
            three.temporal.transform(feature.temporal),
            three.spatial.transform(feature.spatial),
            None,
        )
        combined = np.concatenate([rn, tn, sn], axis=1).astype(np.float32)
        normalized[row["video_id"]] = {
            "resnet": rn,
            "temporal": tn,
            "spatial": sn,
            "combined": combined,
        }
        bundle = build_v2_digest_bundle(
            video_id=row["video_id"],
            segment_ids=feature.segment_ids,
            segment_start_times=feature.segment_start_times,
            segment_end_times=feature.segment_end_times,
            resnet_normalized_features=rn,
            temporal_normalized_features=tn,
            spatial_normalized_features=sn,
            parameters=quantizer,
        )
        v2_digests[row["video_id"]] = bundle
        if held_out_source and row["source_id"] == held_out_source:
            digest_dir = fold_dir / "digests" / row["video_id"]
            save_v2_digest_npz(digest_dir / f"{row['video_id']}_v2_digests.npz", bundle)
            save_json(
                digest_dir / f"{row['video_id']}_v2_digest_manifest.json",
                {
                    "video_id": row["video_id"],
                    "fold_id": fold_id,
                    "pipeline_id": PIPELINE_V2_ID,
                    "normalization_id": three.normalization_id,
                    "quantization_id": quantizer.quantization_id,
                    "continuous_dimensions": V2_CONTINUOUS_DIMENSIONS,
                    "digest_lengths": V2_DIGEST_LENGTHS,
                    "pack_unpack_round_trip": bundle.validate_round_trips(),
                },
            )
    if key_info is not None and held_out_source:
        refs = reference_by_source(registry)
        reference = refs[held_out_source]
        bundle = v2_digests[reference["video_id"]]
        payload = v2_digest_payload(
            bundle=bundle,
            normalization_id=three.normalization_id,
            quantization_id=quantizer.quantization_id,
            source_video_sha256=str(reference.get("sha256", "")),
        )
        record = build_v2_authentication_record(payload, key_info)
        auth_path = fold_dir / "authentication_records" / reference["video_id"] / f"{reference['video_id']}_v2_authentication_record.json"
        save_json(auth_path, record)
        verification = verify_v2_authentication_record(record, key_info)
        hmac_rows.append(
            {
                "fold_id": fold_id,
                "held_out_source": held_out_source,
                "reference_video_id": reference["video_id"],
                "record_path": str(auth_path),
                **verification,
            }
        )
    return v1_digests, v2_digests, normalized, {"normalizers": three, "quantizer": quantizer}, hmac_rows


def video_score_rows(
    *,
    algorithm: str,
    fold_id: str,
    rows: list[dict[str, Any]],
    video: dict[str, Any],
    threshold_model: ThresholdModel,
    structural: bool,
    missing_segments: int,
    extra_segments: int,
    adaptive: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    abnormal_segments = []
    scored_rows = []
    profile = compression_profile(video)
    threshold = threshold_model.profile_thresholds.get(profile, threshold_model.score_threshold) if adaptive else threshold_model.score_threshold
    for row in rows:
        score = float(row["score"])
        blur_loss = float(row.get("blur_loss", 0.0))
        abnormal = score > threshold or blur_loss > threshold_model.blur_loss_threshold
        segment_row = {
            "algorithm": algorithm,
            "fold_id": fold_id,
            "source_id": video["source_id"],
            "video_id": video["video_id"],
            "filename": video["filename"],
            "transformation_type": video["transformation_type"],
            "expected_label": video["expected_label"],
            "compression_profile": profile,
            "segment_score_threshold": threshold,
            "blur_loss_threshold": threshold_model.blur_loss_threshold,
            "segment_abnormal": abnormal,
            **row,
        }
        scored_rows.append(segment_row)
        if abnormal:
            abnormal_segments.append(int(row["segment_id"]))
    observed = "abnormal" if structural or abnormal_segments else "normal"
    expected = video["expected_label"]
    return (
        {
            "algorithm": algorithm,
            "fold_id": fold_id,
            "held_out_source": video["source_id"],
            "source_id": video["source_id"],
            "video_id": video["video_id"],
            "filename": video["filename"],
            "transformation_type": video["transformation_type"],
            "expected_category": video["expected_category"],
            "expected_label": expected,
            "observed_label": observed,
            "correct": expected == observed,
            "abnormal_segment_ids": abnormal_segments,
            "structural_issue": structural,
            "missing_segment_count": missing_segments,
            "extra_segment_count": extra_segments,
            "max_score": max([float(row["score"]) for row in rows], default=0.0),
            "max_blur_loss": max([float(row.get("blur_loss", 0.0)) for row in rows], default=0.0),
            "compression_profile": profile,
            "score_threshold": threshold,
            "blur_loss_threshold": threshold_model.blur_loss_threshold,
            "weights": threshold_model.weights,
            "run_timestamp": utc_now(),
        },
        scored_rows,
    )


def comparisons_for_source(
    *,
    source_id: str,
    registry: list[dict[str, Any]],
    features: dict[str, RawFeatureSet],
    v1_digests: dict[str, V1DigestBits],
    v2_digests: dict[str, Any],
    weights_v1: dict[str, float],
    weights_v2: dict[str, float],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, tuple[bool, int, int]]]:
    refs = reference_by_source(registry)
    ref_video = refs[source_id]
    ref_id = ref_video["video_id"]
    v1_rows_by_video: dict[str, list[dict[str, Any]]] = {}
    v2_rows_by_video: dict[str, list[dict[str, Any]]] = {}
    structural_by_video: dict[str, tuple[bool, int, int]] = {}
    for video in [row for row in registry if row["source_id"] == source_id]:
        video_id = video["video_id"]
        structural_by_video[video_id] = structural_issue(features[ref_id].segment_ids, features[video_id].segment_ids)
        v1_rows_by_video[video_id] = compare_v1(v1_digests[ref_id], v1_digests[video_id], weights_v1)
        v2_comparisons = compare_v2_digest_bundles(
            v2_digests[ref_id],
            v2_digests[video_id],
            features[ref_id].spatial,
            features[video_id].spatial,
            weights_v2,
        )
        v2_rows_by_video[video_id] = [
            {
                "segment_id": item.segment_id,
                "resnet_raw_distance": item.resnet_raw_distance,
                "temporal_raw_distance": item.temporal_raw_distance,
                "spatial_raw_distance": item.spatial_raw_distance,
                "hybrid_raw_distance": item.hybrid_raw_distance,
                "resnet_normalized_distance": item.resnet_normalized_distance,
                "temporal_normalized_distance": item.temporal_normalized_distance,
                "spatial_normalized_distance": item.spatial_normalized_distance,
                "hybrid_normalized_distance": item.hybrid_normalized_distance,
                "score": item.weighted_score,
                "blur_loss": item.blur_loss,
                "stream_attribution": item.stream_attribution,
            }
            for item in v2_comparisons
        ]
    return v1_rows_by_video, v2_rows_by_video, structural_by_video


def threshold_model_from_training(
    *,
    registry: list[dict[str, Any]],
    training_sources: list[str],
    rows_by_source: dict[str, dict[str, list[dict[str, Any]]]],
    weights: dict[str, float],
    margin_multiplier: float,
    include_blur_loss: bool,
) -> ThresholdModel:
    score_values = []
    blur_values = []
    profile_values: dict[str, list[float]] = {}
    row_by_id = {row["video_id"]: row for row in registry}
    for source_id in training_sources:
        for video_id, rows in rows_by_source[source_id].items():
            video = row_by_id[video_id]
            if video["transformation_type"] not in CALIBRATION_TRANSFORMATIONS:
                continue
            profile = compression_profile(video)
            for row in rows:
                value = float(row["score"])
                score_values.append(value)
                profile_values.setdefault(profile, []).append(value)
                if include_blur_loss:
                    blur_values.append(float(row.get("blur_loss", 0.0)))
    one_unit = 1.0 / (V2_HYBRID_DIGEST_LENGTH if include_blur_loss else 1060)
    base = threshold_from_values(score_values, margin_multiplier, one_unit)
    blur = threshold_from_values(blur_values, margin_multiplier, 1.0e-6) if include_blur_loss else 1.0
    return ThresholdModel(
        score_threshold=base,
        blur_loss_threshold=blur,
        margin_multiplier=margin_multiplier,
        weights=weights,
        profile_thresholds={
            profile: threshold_from_values(values, margin_multiplier, one_unit)
            for profile, values in profile_values.items()
        },
        profile_counts={profile: len(values) for profile, values in profile_values.items()},
    )


def evaluate_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in predictions if row["expected_label"] in {"normal", "abnormal"} and row["observed_label"] in {"normal", "abnormal"}]
    tp = sum(1 for row in rows if row["expected_label"] == "abnormal" and row["observed_label"] == "abnormal")
    tn = sum(1 for row in rows if row["expected_label"] == "normal" and row["observed_label"] == "normal")
    fp = sum(1 for row in rows if row["expected_label"] == "normal" and row["observed_label"] == "abnormal")
    fn = sum(1 for row in rows if row["expected_label"] == "abnormal" and row["observed_label"] == "normal")
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "total": total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2 if total else 0.0,
        "FAR": fn / (tp + fn) if tp + fn else 0.0,
        "FRR": fp / (tn + fp) if tn + fp else 0.0,
    }


def select_parameters_inner(
    *,
    algorithm: str,
    outer_training_sources: list[str],
    registry: list[dict[str, Any]],
    features: dict[str, RawFeatureSet],
    include_blur_loss: bool,
) -> ThresholdModel:
    best: tuple[float, float, ThresholdModel] | None = None
    weight_grid = V2_WEIGHT_GRID if include_blur_loss else V1_WEIGHT_GRID
    for weights in weight_grid:
        for margin in MARGIN_GRID:
            inner_predictions: list[dict[str, Any]] = []
            for inner_held in outer_training_sources:
                inner_train = [source for source in outer_training_sources if source != inner_held]
                if len(inner_train) < 2:
                    continue
                v1, v2, _, _, _ = build_fold_digests(
                    fold_id=f"INNER_{algorithm}_{safe_video_id(inner_held)}",
                    training_sources=inner_train,
                    registry=registry,
                    features=features,
                    output_dir=DEFAULT_OUTPUT_DIR / "inner_scratch",
                )
                source_ids = inner_train + [inner_held]
                rows_by_source: dict[str, dict[str, list[dict[str, Any]]]] = {}
                for source_id in source_ids:
                    v1_rows, v2_rows, structural = comparisons_for_source(
                        source_id=source_id,
                        registry=registry,
                        features=features,
                        v1_digests=v1,
                        v2_digests=v2,
                        weights_v1=weights if not include_blur_loss else V1_WEIGHT_GRID[0],
                        weights_v2=weights if include_blur_loss else V2_WEIGHT_GRID[0],
                    )
                    rows_by_source[source_id] = v2_rows if include_blur_loss else v1_rows
                model = threshold_model_from_training(
                    registry=registry,
                    training_sources=inner_train,
                    rows_by_source=rows_by_source,
                    weights=weights,
                    margin_multiplier=margin,
                    include_blur_loss=include_blur_loss,
                )
                for video in [row for row in registry if row["source_id"] == inner_held]:
                    ref = reference_by_source(registry)[inner_held]
                    structural = structural_issue(features[ref["video_id"]].segment_ids, features[video["video_id"]].segment_ids)
                    prediction, _ = video_score_rows(
                        algorithm=algorithm,
                        fold_id=f"INNER_{inner_held}",
                        rows=rows_by_source[inner_held][video["video_id"]],
                        video=video,
                        threshold_model=model,
                        structural=structural[0],
                        missing_segments=structural[1],
                        extra_segments=structural[2],
                    )
                    inner_predictions.append(prediction)
            metrics = evaluate_predictions(inner_predictions)
            candidate = (float(metrics["balanced_accuracy"]), float(metrics["f1"]), ThresholdModel(0, 0, margin, weights, {}, {}))
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        return ThresholdModel(1.0, 1.0, 3.0, weight_grid[0], {}, {})
    return best[2]


def run_nested_evaluation(
    *,
    registry: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    features: dict[str, RawFeatureSet],
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    valid_sources = sorted(row["source_id"] for row in source_rows if row["status"] == "valid")
    key_file = REPO_ROOT / "data" / "secrets" / "DEV_HMAC_KEY_V1.hex"
    key_info = load_hmac_key(key_file=key_file, key_id="DEV_HMAC_KEY_V1")
    predictions: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    hmac_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    self_rows: list[dict[str, Any]] = []
    for held_out in valid_sources:
        fold_id = "LOSO_" + held_out.replace("SRC_", "")
        training_sources = [source for source in valid_sources if source != held_out]
        selected_v1 = select_parameters_inner(
            algorithm="V1_FIXED",
            outer_training_sources=training_sources,
            registry=registry,
            features=features,
            include_blur_loss=False,
        )
        selected_v2 = select_parameters_inner(
            algorithm="V2_FIXED",
            outer_training_sources=training_sources,
            registry=registry,
            features=features,
            include_blur_loss=True,
        )
        v1, v2, _, _, fold_hmac = build_fold_digests(
            fold_id=fold_id,
            training_sources=training_sources,
            registry=registry,
            features=features,
            output_dir=output_dir,
            key_info=key_info,
            held_out_source=held_out,
        )
        hmac_rows.extend(fold_hmac)
        all_v1_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
        all_v2_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
        structural_by_source: dict[str, dict[str, tuple[bool, int, int]]] = {}
        for source_id in valid_sources:
            v1_rows, v2_rows, structural = comparisons_for_source(
                source_id=source_id,
                registry=registry,
                features=features,
                v1_digests=v1,
                v2_digests=v2,
                weights_v1=selected_v1.weights,
                weights_v2=selected_v2.weights,
            )
            all_v1_rows[source_id] = v1_rows
            all_v2_rows[source_id] = v2_rows
            structural_by_source[source_id] = structural
        v1_model = threshold_model_from_training(
            registry=registry,
            training_sources=training_sources,
            rows_by_source=all_v1_rows,
            weights=selected_v1.weights,
            margin_multiplier=selected_v1.margin_multiplier,
            include_blur_loss=False,
        )
        v2_model = threshold_model_from_training(
            registry=registry,
            training_sources=training_sources,
            rows_by_source=all_v2_rows,
            weights=selected_v2.weights,
            margin_multiplier=selected_v2.margin_multiplier,
            include_blur_loss=True,
        )
        threshold_rows.extend(
            [
                {
                    "fold_id": fold_id,
                    "algorithm": "V1_FIXED",
                    "held_out_source": held_out,
                    "training_sources": training_sources,
                    "training_calibration_videos": normal_training_videos(registry, training_sources),
                    "score_threshold": v1_model.score_threshold,
                    "blur_loss_threshold": "",
                    "weights": v1_model.weights,
                    "margin_multiplier": v1_model.margin_multiplier,
                    "profile_thresholds": {},
                    "held_out_used_for_fitting": held_out in training_sources,
                },
                {
                    "fold_id": fold_id,
                    "algorithm": "V2_FIXED",
                    "held_out_source": held_out,
                    "training_sources": training_sources,
                    "training_calibration_videos": normal_training_videos(registry, training_sources),
                    "score_threshold": v2_model.score_threshold,
                    "blur_loss_threshold": v2_model.blur_loss_threshold,
                    "weights": v2_model.weights,
                    "margin_multiplier": v2_model.margin_multiplier,
                    "profile_thresholds": {},
                    "held_out_used_for_fitting": held_out in training_sources,
                },
                {
                    "fold_id": fold_id,
                    "algorithm": "V2_ADAPTIVE",
                    "held_out_source": held_out,
                    "training_sources": training_sources,
                    "training_calibration_videos": normal_training_videos(registry, training_sources),
                    "score_threshold": v2_model.score_threshold,
                    "blur_loss_threshold": v2_model.blur_loss_threshold,
                    "weights": v2_model.weights,
                    "margin_multiplier": v2_model.margin_multiplier,
                    "profile_thresholds": v2_model.profile_thresholds,
                    "profile_counts": v2_model.profile_counts,
                    "held_out_used_for_fitting": held_out in training_sources,
                },
            ]
        )
        leakage_rows.append(
            {
                "fold_id": fold_id,
                "held_out_source": held_out,
                "training_sources": training_sources,
                "held_out_excluded_from_normalization": held_out not in training_sources,
                "held_out_excluded_from_quantization": held_out not in training_sources,
                "held_out_excluded_from_thresholds": held_out not in training_sources,
                "tampered_used_for_normalization": False,
                "tampered_used_for_quantization": False,
                "tampered_used_for_adaptive_model": False,
            }
        )
        for video in [row for row in registry if row["source_id"] == held_out]:
            structural = structural_by_source[held_out][video["video_id"]]
            for algorithm, rows, model, adaptive in (
                ("V1_FIXED", all_v1_rows[held_out][video["video_id"]], v1_model, False),
                ("V2_FIXED", all_v2_rows[held_out][video["video_id"]], v2_model, False),
                ("V2_ADAPTIVE", all_v2_rows[held_out][video["video_id"]], v2_model, True),
            ):
                prediction, scored = video_score_rows(
                    algorithm=algorithm,
                    fold_id=fold_id,
                    rows=rows,
                    video=video,
                    threshold_model=model,
                    structural=structural[0],
                    missing_segments=structural[1],
                    extra_segments=structural[2],
                    adaptive=adaptive,
                )
                predictions.append(prediction)
                segment_rows.extend(scored)
            if video["expected_category"] == "reference":
                self_rows.append(
                    {
                        "fold_id": fold_id,
                        "video_id": video["video_id"],
                        "v1_self_zero": all(int(row["hybrid_raw_distance"]) == 0 for row in all_v1_rows[held_out][video["video_id"]]),
                        "v2_self_zero": all(int(row["hybrid_raw_distance"]) == 0 and float(row.get("blur_loss", 0.0)) == 0.0 for row in all_v2_rows[held_out][video["video_id"]]),
                    }
                )
    algorithms = sorted({row["algorithm"] for row in predictions})
    overall = [{**evaluate_predictions([row for row in predictions if row["algorithm"] == algorithm]), "algorithm": algorithm} for algorithm in algorithms]
    per_source = []
    for algorithm in algorithms:
        for source_id in valid_sources:
            rows = [row for row in predictions if row["algorithm"] == algorithm and row["source_id"] == source_id]
            per_source.append({**evaluate_predictions(rows), "algorithm": algorithm, "source_id": source_id})
    per_tamper = []
    for algorithm in algorithms:
        for transformation in sorted(TAMPERED_TRANSFORMATIONS):
            rows = [row for row in predictions if row["algorithm"] == algorithm and row["transformation_type"] == transformation]
            detected = sum(1 for row in rows if row["observed_label"] == "abnormal")
            per_tamper.append(
                {
                    "algorithm": algorithm,
                    "transformation_type": transformation,
                    "detected": detected,
                    "total": len(rows),
                    "detection_rate": detected / len(rows) if rows else 0.0,
                }
            )
    per_benign = []
    for algorithm in algorithms:
        for transformation in sorted(BENIGN_TRANSFORMATIONS):
            rows = [row for row in predictions if row["algorithm"] == algorithm and row["transformation_type"] == transformation]
            rejected = sum(1 for row in rows if row["observed_label"] == "abnormal")
            per_benign.append(
                {
                    "algorithm": algorithm,
                    "transformation_type": transformation,
                    "false_rejections": rejected,
                    "total": len(rows),
                    "false_rejection_rate": rejected / len(rows) if rows else 0.0,
                }
            )
    false_positives = [row for row in predictions if row["expected_label"] == "normal" and row["observed_label"] == "abnormal"]
    false_negatives = [row for row in predictions if row["expected_label"] == "abnormal" and row["observed_label"] == "normal"]
    return {
        "predictions": predictions,
        "segment_rows": segment_rows,
        "threshold_rows": threshold_rows,
        "hmac_rows": hmac_rows,
        "leakage_rows": leakage_rows,
        "self_rows": self_rows,
        "overall": overall,
        "per_source": per_source,
        "per_tamper": per_tamper,
        "per_benign": per_benign,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def synthetic_blur_behavior() -> list[dict[str, Any]]:
    rng = np.random.default_rng(42)
    base = np.indices((224, 224)).sum(axis=0) % 2
    image = (base.astype(np.float32) * 180.0 + 40.0 + rng.normal(0, 6, size=(224, 224))).clip(0, 255).astype(np.uint8)
    rows = []
    for kernel in [1, 3, 5, 9, 15]:
        frame = image if kernel == 1 else cv2_gaussian(image, kernel)
        metrics = calculate_spatial_quality_metrics(frame)
        rows.append(
            {
                "kernel": kernel,
                **{name: float(value) for name, value in zip(SPATIAL_SEGMENT_FEATURE_NAMES[:5], metrics, strict=True)},
            }
        )
    return rows


def cv2_gaussian(image: np.ndarray, kernel: int) -> np.ndarray:
    import cv2

    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def runtime_summary_rows(output_dir: Path, run_id: str) -> list[dict[str, Any]]:
    path = output_dir / "runtime_log.csv"
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("run_id") == run_id:
                rows.append(row)
    by_stage: dict[str, dict[str, Any]] = {}
    for row in rows:
        stage = row["stage"]
        item = by_stage.setdefault(stage, {"stage": stage, "runs": 0, "total_duration_seconds": 0.0, "completed": 0, "failed": 0})
        item["runs"] += 1
        item["completed"] += row["status"] == "completed"
        item["failed"] += row["status"] == "failed"
        item["total_duration_seconds"] += float(row.get("duration_seconds") or 0.0)
    return list(by_stage.values())


def write_all_tables(output_dir: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv_rows(table_dir / f"{name}.csv", rows)


def build_figures(output_dir: Path, results: dict[str, Any]) -> dict[str, str]:
    figures = output_dir / "figures"
    timelines = figures / "timelines"
    localization = figures / "localization"
    for folder in (figures, timelines, localization):
        folder.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    overall_df = pd.DataFrame(results["overall"])
    if not overall_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        overall_df.set_index("algorithm")[["accuracy", "precision", "recall", "f1"]].plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Metric")
        ax.set_title("V1 vs V2 Outer-Fold Metrics")
        fig.tight_layout()
        path = figures / "overall_metrics.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths["overall_metrics"] = str(path)
    tamper_df = pd.DataFrame(results["per_tamper"])
    if not tamper_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        tamper_df.pivot(index="transformation_type", columns="algorithm", values="detection_rate").plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Detection rate")
        ax.set_title("Per-Tamper Detection")
        fig.tight_layout()
        path = figures / "per_tamper_detection.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths["per_tamper_detection"] = str(path)
    benign_df = pd.DataFrame(results["per_benign"])
    if not benign_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        benign_df.pivot(index="transformation_type", columns="algorithm", values="false_rejection_rate").plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("False rejection rate")
        ax.set_title("Benign False Rejections")
        fig.tight_layout()
        path = figures / "per_benign_false_rejection.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths["per_benign_false_rejection"] = str(path)
    segment_df = pd.DataFrame(results["segment_rows"])
    for video_id, rows in segment_df.groupby("video_id") if not segment_df.empty else []:
        fig, ax = plt.subplots(figsize=(8, 3))
        for algorithm, alg_rows in rows.groupby("algorithm"):
            ax.plot(alg_rows["segment_id"], alg_rows["score"], marker="o", label=algorithm)
        ax.set_title(video_id)
        ax.set_xlabel("Segment")
        ax.set_ylabel("Score")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = timelines / f"{safe_video_id(video_id)}_timeline.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
    paths["timelines"] = str(timelines)
    for row in results["false_positives"][:6] + results["false_negatives"][:6]:
        rows = segment_df[(segment_df["video_id"] == row["video_id"]) & (segment_df["algorithm"] == row["algorithm"])]
        if rows.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(rows["segment_id"].astype(str), rows["score"])
        ax.axhline(float(row["score_threshold"]), color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{row['algorithm']} {row['video_id']} localization")
        ax.set_xlabel("Segment")
        ax.set_ylabel("Score")
        fig.tight_layout()
        path = localization / f"{safe_video_id(row['algorithm'] + '_' + row['video_id'])}_localization.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
    paths["localization"] = str(localization)
    return paths


def build_html_report(output_dir: Path, summary: dict[str, Any]) -> Path:
    html = output_dir / "blur_aware_final_evaluation_report.html"
    overall = pd.DataFrame(summary["overall_metrics"])
    per_tamper = pd.DataFrame(summary["per_tamper_metrics"])
    per_benign = pd.DataFrame(summary["per_benign_metrics"])
    failures = pd.DataFrame(summary["false_positives"] + summary["false_negatives"])
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Blur-Aware Final Evaluation</title>",
        "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;margin:16px 0}td,th{border:1px solid #ccc;padding:6px 8px}th{background:#eee}.note{color:#444}</style>",
        "</head><body>",
        "<h1>Blur-Aware Final Six-Source Evaluation</h1>",
        f"<p class='note'>Run ID: {summary['run_id']} | Pipeline: {PIPELINE_V2_ID}</p>",
        "<h2>Dataset</h2>",
        pd.DataFrame([summary["dataset"]]).to_html(index=False),
        "<h2>Overall Metrics</h2>",
        overall.to_html(index=False) if not overall.empty else "<p>No metrics.</p>",
        "<h2>Per-Tamper Detection</h2>",
        per_tamper.to_html(index=False) if not per_tamper.empty else "<p>No tamper metrics.</p>",
        "<h2>Per-Benign False Rejection</h2>",
        per_benign.to_html(index=False) if not per_benign.empty else "<p>No benign metrics.</p>",
        "<h2>False Positives and False Negatives</h2>",
        failures.to_html(index=False) if not failures.empty else "<p>No false positives or false negatives.</p>",
        "<h2>Claims</h2>",
        f"<p>{summary['safe_dataset_specific_claim']}</p>",
        "<h2>Limitations</h2><ul>",
        *[f"<li>{item}</li>" for item in summary["limitations"]],
        "</ul></body></html>",
    ]
    html.write_text("\n".join(parts), encoding="utf-8")
    return html


def validate_dataset_summary(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "dataset_validation_report.json"
    if path.exists():
        data = read_json(path)
        return {
            "sources": data.get("source_count"),
            "total_videos": data.get("video_count"),
            "valid_videos": data.get("valid_video_count"),
            "corrupt": data.get("corrupt_video_count"),
            "references": data.get("category_counts", {}).get("reference"),
            "benign": data.get("category_counts", {}).get("benign"),
            "tampered": data.get("category_counts", {}).get("tampered"),
        }
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--skip-feature-extraction", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir / "tables"
    (output_dir / "figures" / "timelines").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures" / "localization").mkdir(parents=True, exist_ok=True)
    run_id = make_run_id()
    started = time.perf_counter()
    input_root = args.input_root.resolve()
    config_path = args.config.resolve()
    source_rows, registry, skipped = discover_sources(input_root, output_dir, run_id)
    registry = augment_registry_metadata(
        registry=registry,
        output_dir=output_dir,
        current_run_id=run_id,
        cpu_threads=2,
        dry_run=False,
    )
    if not args.skip_feature_extraction:
        ensure_raw_feature_cache(
            registry=registry,
            source_rows=source_rows,
            config_path=config_path,
            output_dir=output_dir,
            run_id=run_id,
        )
    config = load_config(config_path)
    features = {row["video_id"]: load_raw_feature_set(row, config) for row in registry}
    feature_rows = [
        {
            "video_id": feature.video_id,
            "source_id": feature.source_id,
            "segment_count": int(feature.segment_ids.shape[0]),
            "resnet_shape": list(feature.resnet.shape),
            "temporal_shape": list(feature.temporal.shape),
            "spatial_shape": list(feature.spatial.shape),
            "finite": bool(np.all(np.isfinite(feature.resnet)) and np.all(np.isfinite(feature.temporal)) and np.all(np.isfinite(feature.spatial))),
        }
        for feature in features.values()
    ]
    results = run_nested_evaluation(
        registry=registry,
        source_rows=source_rows,
        features=features,
        output_dir=output_dir,
        run_id=run_id,
    )
    synthetic_rows = synthetic_blur_behavior()
    runtime_rows = runtime_summary_rows(output_dir, run_id)
    write_all_tables(
        output_dir,
        {
            "dataset_validation": [validate_dataset_summary(output_dir)],
            "source_registry": source_rows,
            "video_inventory": registry,
            "metadata_comparison": metadata_comparison(registry, run_id),
            "sha256_baseline": sha256_baseline(registry, run_id),
            "raw_feature_inventory": feature_rows,
            "spatial_blur_behavior": synthetic_rows,
            "diagnostic_predictions": results["predictions"],
            "segment_distances": results["segment_rows"],
            "nested_thresholds": results["threshold_rows"],
            "leakage_checks": results["leakage_rows"],
            "hmac_verification": results["hmac_rows"],
            "self_comparisons": results["self_rows"],
            "overall_metrics": results["overall"],
            "per_source_metrics": results["per_source"],
            "per_tamper_metrics": results["per_tamper"],
            "per_benign_metrics": results["per_benign"],
            "false_positives": results["false_positives"],
            "false_negatives": results["false_negatives"],
            "runtime_summary": runtime_rows,
        },
    )
    figure_paths = build_figures(output_dir, results)
    total_runtime = time.perf_counter() - started
    dataset = validate_dataset_summary(output_dir)
    summary = {
        "run_id": run_id,
        "experiment_id": EXPERIMENT_ID,
        "pipeline_id": PIPELINE_V2_ID,
        "schema_version": V2_SCHEMA_VERSION,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False).stdout.strip(),
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "dataset": dataset,
        "spatial_metrics": SPATIAL_METRIC_NAMES,
        "spatial_segment_dimension": SPATIAL_SEGMENT_DIMENSION,
        "continuous_dimensions": V2_CONTINUOUS_DIMENSIONS,
        "digest_lengths": V2_DIGEST_LENGTHS,
        "outer_folds": 6,
        "inner_folds_per_outer": 5,
        "leakage_checks": results["leakage_rows"],
        "hmac_references": results["hmac_rows"],
        "self_comparisons": results["self_rows"],
        "overall_metrics": results["overall"],
        "per_source_metrics": results["per_source"],
        "per_tamper_metrics": results["per_tamper"],
        "per_benign_metrics": results["per_benign"],
        "false_positives": results["false_positives"],
        "false_negatives": results["false_negatives"],
        "runtime": {
            "total_seconds": total_runtime,
            "runtime_summary": runtime_rows,
        },
        "reports": {
            "tables": str(table_dir),
            "figures": str(output_dir / "figures"),
            "timelines": str(output_dir / "figures" / "timelines"),
            "localization": str(output_dir / "figures" / "localization"),
            **figure_paths,
        },
        "safe_dataset_specific_claim": "On this repaired six-source dataset, V2 adds blur-aware spatial quality features and was evaluated with held-out source groups only.",
        "limitations": [
            "Only six source videos are available, so results are dataset-specific.",
            "Outer-fold outcomes were not used for normalization, quantization, adaptive profile fitting, or threshold tuning.",
            "Adaptive thresholds use observed benign compression profiles; broader codec and camera generalization is not established.",
            "Universal generalization is not supported.",
        ],
        "universal_generalization": "Not supported",
        "free_space_bytes_after": disk_free_bytes(),
    }
    html = build_html_report(output_dir, summary)
    summary["reports"]["html"] = str(html)
    write_json(output_dir / "blur_aware_final_evaluation_summary.json", summary)
    print(f"Blur-aware final evaluation complete: {output_dir / 'blur_aware_final_evaluation_summary.json'}")
    print(f"HTML report: {html}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FinalEvaluationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
