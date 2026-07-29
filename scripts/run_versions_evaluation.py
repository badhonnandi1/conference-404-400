#!/usr/bin/env python3
"""Run the resource-controlled evaluation for videos under data/versions."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.authentication.auth_record_storage import authentication_record_paths
from src.authentication.digest import unpack_packed_bits
from src.authentication.digest_storage import digest_output_paths
from src.authentication.hmac_auth import load_hmac_key
from src.config import load_config
from src.features.feature_storage import feature_output_paths, sha256_file
from src.features.normalization_storage import normalized_output_paths
from src.features.temporal_storage import temporal_output_paths
from src.verification.comparison_storage import comparison_output_paths
from src.video.metadata import safe_video_id


EXPERIMENT_ID = "versions_evaluation"
REFERENCE_ID = "VER_ORIGINAL"
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
NORMALIZATION_ID = "DEV_NORMALIZATION_V1"
QUANTIZATION_ID = "DEV_QUANTIZATION_V1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "reports" / EXPERIMENT_ID
DEFAULT_CONFIG = REPO_ROOT / "configs" / "versions_evaluation.yaml"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "versions"
CONTROLLED_ENV = {
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "VECLIB_MAXIMUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "TOKENIZERS_PARALLELISM": "false",
}
RUNTIME_COLUMNS = [
    "experiment_id",
    "run_id",
    "video_id",
    "stage",
    "start_time",
    "end_time",
    "duration_seconds",
    "status",
    "cache_reused",
    "failure_reason",
    "system_load_snapshot",
]
VIDEO_STAGE_ORDER = ["preprocess", "resnet", "temporal", "normalize", "digest"]


class EvaluationError(RuntimeError):
    """Raised when the versions evaluation cannot safely continue."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_id() -> str:
    return "versions_eval_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise EvaluationError(f"Expected JSON object: {path}")
    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            flat[key] = json.dumps(value, sort_keys=True, ensure_ascii=False)
        else:
            flat[key] = value
    return flat


def write_csv_rows(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    columns.append(key)
                    seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_for_csv(row))


def append_runtime_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNTIME_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(flatten_for_csv(row))


def controlled_env(cpu_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    for key in CONTROLLED_ENV:
        env[key] = str(cpu_threads) if key != "TOKENIZERS_PARALLELISM" else "false"
    return env


def disk_free_bytes(path: Path = Path("/")) -> int:
    return shutil.disk_usage(path).free


def command_text(command: list[str]) -> str:
    return " ".join(command)


def subprocess_text(command: list[str], timeout: int = 30) -> tuple[int, str, str]:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def system_snapshot() -> dict[str, Any]:
    df_code, df_stdout, df_stderr = subprocess_text(["df", "-h", "/"])
    uptime_code, uptime_stdout, uptime_stderr = subprocess_text(["uptime"])
    pmset_code, pmset_stdout, pmset_stderr = subprocess_text(["pmset", "-g", "therm"])
    return {
        "timestamp": utc_now(),
        "free_bytes_root": disk_free_bytes(Path("/")),
        "df_h_root": df_stdout if df_code == 0 else df_stderr,
        "uptime": uptime_stdout if uptime_code == 0 else uptime_stderr,
        "thermal": pmset_stdout if pmset_code == 0 else f"unavailable: {pmset_stderr or pmset_code}",
    }


def thermal_or_load_pressure(snapshot: dict[str, Any]) -> bool:
    thermal = str(snapshot.get("thermal", "")).lower()
    load = str(snapshot.get("uptime", "")).lower()
    if any(token in thermal for token in ("serious", "critical", "high", "throttl")):
        return True
    if "load averages:" in load:
        try:
            after = load.split("load averages:", maxsplit=1)[1].strip()
            first = float(after.split()[0])
            return first >= 8.0
        except (IndexError, ValueError):
            return False
    return False


def maybe_pause_for_pressure(output_dir: Path, snapshot: dict[str, Any]) -> None:
    if thermal_or_load_pressure(snapshot):
        log_path = output_dir / "runtime_log.csv"
        started = utc_now()
        time.sleep(60)
        append_runtime_log(
            log_path,
            {
                "experiment_id": EXPERIMENT_ID,
                "run_id": "",
                "video_id": "SYSTEM",
                "stage": "thermal_or_load_pause",
                "start_time": started,
                "end_time": utc_now(),
                "duration_seconds": 60.0,
                "status": "completed",
                "cache_reused": False,
                "failure_reason": "",
                "system_load_snapshot": snapshot,
            },
        )


def run_logged_command(
    *,
    command: list[str],
    video_id: str,
    stage: str,
    output_dir: Path,
    current_run_id: str,
    cpu_threads: int,
    heavy: bool,
) -> dict[str, Any]:
    snapshot = system_snapshot()
    maybe_pause_for_pressure(output_dir, snapshot)
    start = time.perf_counter()
    start_time = utc_now()
    full_command = ["nice", "-n", "10", *command] if heavy else command
    result = subprocess.run(
        full_command,
        cwd=REPO_ROOT,
        env=controlled_env(cpu_threads),
        capture_output=True,
        text=True,
        check=False,
    )
    duration = time.perf_counter() - start
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    status = "completed" if result.returncode == 0 else "failed"
    failure = "" if result.returncode == 0 else (stderr or stdout or f"exit code {result.returncode}")[-500:]
    cache_reused = "Reusing cached" in stdout or "reused cached" in stdout
    row = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": current_run_id,
        "video_id": video_id,
        "stage": stage,
        "start_time": start_time,
        "end_time": utc_now(),
        "duration_seconds": round(duration, 6),
        "status": status,
        "cache_reused": cache_reused,
        "failure_reason": failure,
        "system_load_snapshot": snapshot,
    }
    append_runtime_log(output_dir / "runtime_log.csv", row)
    return {
        **row,
        "returncode": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "command": command_text(full_command),
    }


def sleep_after_stage(stage: str, output_dir: Path, current_run_id: str, multiplier: float) -> None:
    base = {"preprocess": 10.0, "resnet": 15.0, "temporal": 10.0}.get(stage, 0.0)
    seconds = base * max(0.0, multiplier)
    if seconds <= 0:
        return
    started = time.perf_counter()
    start_time = utc_now()
    time.sleep(seconds)
    append_runtime_log(
        output_dir / "runtime_log.csv",
        {
            "experiment_id": EXPERIMENT_ID,
            "run_id": current_run_id,
            "video_id": "SYSTEM",
            "stage": f"sleep_after_{stage}",
            "start_time": start_time,
            "end_time": utc_now(),
            "duration_seconds": round(time.perf_counter() - started, 6),
            "status": "completed",
            "cache_reused": False,
            "failure_reason": "",
            "system_load_snapshot": system_snapshot(),
        },
    )


def ffprobe_json(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise EvaluationError(f"FFprobe failed for {path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def parse_fps(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    if "/" in value:
        numerator, denominator = value.split("/", maxsplit=1)
        try:
            denominator_value = float(denominator)
            if denominator_value == 0:
                return None
            fps = float(numerator) / denominator_value
        except ValueError:
            return None
    else:
        try:
            fps = float(value)
        except ValueError:
            return None
    return fps if math.isfinite(fps) and fps > 0 else None


def optional_int(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def optional_float(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def decode_check(path: Path, video_id: str, output_dir: Path, current_run_id: str, cpu_threads: int) -> tuple[bool, str]:
    result = run_logged_command(
        command=["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        video_id=video_id,
        stage="decode_check",
        output_dir=output_dir,
        current_run_id=current_run_id,
        cpu_threads=cpu_threads,
        heavy=True,
    )
    if result["returncode"] == 0:
        return True, ""
    return False, str(result.get("stderr") or result.get("stdout") or result.get("failure_reason", ""))[-500:]


def classify_video(path: Path) -> dict[str, str]:
    lower = path.name.lower()
    stem = path.stem.lower()
    ext = path.suffix.lower()
    if lower == "original.mp4":
        return {
            "base_id": REFERENCE_ID,
            "expected_category": "REFERENCE",
            "expected_label": "normal",
            "transformation_type": "trusted_reference",
        }
    if "blur" in lower:
        return {
            "base_id": "VER_BLUR",
            "expected_category": "TAMPERED",
            "expected_label": "abnormal",
            "transformation_type": "blur",
        }
    if "delete" in lower or "deletion" in lower:
        return {
            "base_id": "VER_FRAME_DELETE",
            "expected_category": "TAMPERED",
            "expected_label": "abnormal",
            "transformation_type": "frame_deletion",
        }
    if "replacement" in lower or "replace" in lower:
        return {
            "base_id": "VER_FRAME_REPLACEMENT",
            "expected_category": "TAMPERED",
            "expected_label": "abnormal",
            "transformation_type": "frame_replacement",
        }
    if "insert" in lower or "insertion" in lower or "nsert" in lower:
        return {
            "base_id": "VER_FRAME_INSERT",
            "expected_category": "TAMPERED",
            "expected_label": "abnormal",
            "transformation_type": "frame_insertion",
        }
    if ext == ".avi" or "avi" in stem:
        return {
            "base_id": "VER_AVI",
            "expected_category": "BENIGN",
            "expected_label": "normal",
            "transformation_type": "avi_conversion",
        }
    if ext == ".mov" or "mov" in stem:
        return {
            "base_id": "VER_MOV",
            "expected_category": "BENIGN",
            "expected_label": "normal",
            "transformation_type": "mov_conversion",
        }
    if "480" in stem:
        return {
            "base_id": "VER_480P",
            "expected_category": "BENIGN",
            "expected_label": "normal",
            "transformation_type": "resize_480p",
        }
    if "720" in stem:
        return {
            "base_id": "VER_720P",
            "expected_category": "BENIGN",
            "expected_label": "normal",
            "transformation_type": "resize_720p",
        }
    return {
        "base_id": "VER_" + safe_video_id(path.stem).upper(),
        "expected_category": "UNCLASSIFIED",
        "expected_label": "unknown",
        "transformation_type": "unclassified",
    }


def discover_video_paths(data_dir: Path) -> tuple[list[Path], list[Path]]:
    if not data_dir.exists():
        raise EvaluationError(f"data/versions/ does not exist: {data_dir}")
    skipped: list[Path] = []
    candidates: list[Path] = []
    for path in sorted(data_dir.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            skipped.append(path)
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped.append(path)
            continue
        candidates.append(path)
    if not candidates:
        raise EvaluationError(f"No supported videos exist under {data_dir}")
    possible_originals = [path for path in candidates if path.stem.lower() == "original"]
    if not possible_originals:
        raise EvaluationError("original.mp4 could not be identified under data/versions/")
    if len(possible_originals) > 1:
        raise EvaluationError(
            "More than one possible original exists: "
            + ", ".join(path.name for path in possible_originals)
        )
    if possible_originals[0].name.lower() != "original.mp4":
        raise EvaluationError(f"Reference must be original.mp4, got {possible_originals[0].name}")
    unreadable = [path for path in candidates if not os.access(path, os.R_OK)]
    if unreadable:
        raise EvaluationError("Unreadable files: " + ", ".join(path.name for path in unreadable))
    total_size = sum(path.stat().st_size for path in candidates)
    required_free = max(1 * 1024**3, total_size * 8)
    if disk_free_bytes(Path("/")) < required_free:
        raise EvaluationError(
            f"Insufficient disk space: need at least {required_free / 1024**3:.2f} GiB free for a controlled run."
        )
    return candidates, skipped


def metadata_from_probe(path: Path, probe: dict[str, Any]) -> dict[str, Any]:
    streams = probe.get("streams") or []
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if not video_streams:
        raise EvaluationError(f"No video stream found in {path}")
    stream = video_streams[0]
    format_info = probe.get("format") or {}
    duration = optional_float(format_info.get("duration")) or optional_float(stream.get("duration"))
    fps_string = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    fps = parse_fps(fps_string)
    frame_count = optional_int(stream.get("nb_frames"))
    if frame_count is None and duration is not None and fps is not None:
        frame_count = int(round(duration * fps))
    return {
        "duration_seconds": duration,
        "width": optional_int(stream.get("width")),
        "height": optional_int(stream.get("height")),
        "fps": fps,
        "fps_text": fps_string,
        "frame_count": frame_count,
        "codec": stream.get("codec_name"),
        "codec_long_name": stream.get("codec_long_name"),
        "pixel_format": stream.get("pix_fmt"),
        "container_format": format_info.get("format_name"),
        "bitrate": optional_int(format_info.get("bit_rate")) or optional_int(stream.get("bit_rate")),
        "audio_presence": bool(audio_streams),
        "audio_stream_count": len(audio_streams),
        "video_stream_count": len(video_streams),
    }


def build_registry(
    *,
    data_dir: Path,
    output_dir: Path,
    current_run_id: str,
    cpu_threads: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths, skipped_paths = discover_video_paths(data_dir)
    checksum_groups: dict[str, list[str]] = {}
    entries: list[dict[str, Any]] = []
    id_counts: dict[str, int] = {}
    for path in paths:
        classification = classify_video(path)
        base_id = classification["base_id"]
        id_counts[base_id] = id_counts.get(base_id, 0) + 1
        video_id = base_id if id_counts[base_id] == 1 else f"{base_id}_{id_counts[base_id]}"
        checksum = sha256_file(path)
        checksum_groups.setdefault(checksum, []).append(path.name)
        probe = ffprobe_json(path)
        metadata = metadata_from_probe(path, probe)
        decode_valid, decode_failure = decode_check(path, video_id, output_dir, current_run_id, cpu_threads)
        notes = []
        if decode_failure:
            notes.append(decode_failure)
        entry = {
            "experiment_id": EXPERIMENT_ID,
            "run_id": current_run_id,
            "experiment_video_id": video_id,
            "video_id": video_id,
            "filename": path.name,
            "relative_path": str(path.relative_to(REPO_ROOT)),
            "absolute_path": str(path.resolve()),
            "file_extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
            "expected_category": classification["expected_category"],
            "expected_label": classification["expected_label"],
            "transformation_type": classification["transformation_type"],
            "source_reference": "original.mp4" if video_id != REFERENCE_ID else "",
            "checksum": checksum,
            "sha256": checksum,
            "metadata": metadata,
            "duration_seconds": metadata["duration_seconds"],
            "width": metadata["width"],
            "height": metadata["height"],
            "fps": metadata["fps"],
            "frame_count": metadata["frame_count"],
            "codec": metadata["codec"],
            "pixel_format": metadata["pixel_format"],
            "container_format": metadata["container_format"],
            "bitrate": metadata["bitrate"],
            "audio_presence": metadata["audio_presence"],
            "decode_valid": decode_valid,
            "processing_status": "discovered" if decode_valid else "decode_failed",
            "notes": "; ".join(notes),
            "run_timestamp": utc_now(),
        }
        entries.append(entry)
    duplicate_groups = {sha: names for sha, names in checksum_groups.items() if len(names) > 1}
    for entry in entries:
        duplicates = duplicate_groups.get(entry["checksum"], [])
        if duplicates:
            entry["byte_identical_duplicates"] = [name for name in duplicates if name != entry["filename"]]
            if entry["byte_identical_duplicates"]:
                entry["notes"] = (entry["notes"] + "; " if entry["notes"] else "") + "byte-identical duplicate found"
        else:
            entry["byte_identical_duplicates"] = []
    if sum(1 for entry in entries if entry["video_id"] == REFERENCE_ID) != 1:
        raise EvaluationError("Exactly one VER_ORIGINAL reference was not resolved from original.mp4.")
    skipped = [
        {
            "experiment_id": EXPERIMENT_ID,
            "run_id": current_run_id,
            "filename": path.name,
            "relative_path": str(path.relative_to(REPO_ROOT)),
            "reason": "hidden_or_unsupported",
        }
        for path in skipped_paths
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "video_registry.json", {"videos": entries, "skipped": skipped})
    write_csv_rows(output_dir / "video_registry.csv", entries)
    return entries, skipped


def ordered_videos(registry: list[dict[str, Any]], requested_id: str | None = None) -> list[dict[str, Any]]:
    by_id = {entry["video_id"]: entry for entry in registry}
    if REFERENCE_ID not in by_id:
        raise EvaluationError("Reference VER_ORIGINAL is missing from registry.")
    if requested_id:
        requested = safe_video_id(requested_id)
        if requested not in by_id:
            raise EvaluationError(f"Requested --video-id was not discovered: {requested_id}")
        if requested == REFERENCE_ID:
            return [by_id[REFERENCE_ID]]
        return [by_id[REFERENCE_ID], by_id[requested]]
    return [by_id[REFERENCE_ID]] + sorted(
        [entry for entry in registry if entry["video_id"] != REFERENCE_ID],
        key=lambda row: row["filename"].lower(),
    )


def stage_forced(force_stage: set[str], stage: str) -> bool:
    return "all" in force_stage or stage in force_stage


def path_for_stage(config: Any, video_id: str, stage: str) -> list[Path]:
    if stage == "preprocess":
        return [
            config.paths.metadata / f"{video_id}_metadata.json",
            config.paths.manifests / f"{video_id}_segments.json",
            config.paths.manifests / f"{video_id}_frames.json",
        ]
    if stage == "resnet":
        paths = feature_output_paths(config.paths.resnet_features, video_id)
        return [paths.npz_path, paths.manifest_path]
    if stage == "temporal":
        paths = temporal_output_paths(config.paths.temporal_features, video_id)
        return [paths.npz_path, paths.manifest_path]
    if stage == "normalize":
        paths = normalized_output_paths(config.paths.normalized_features, video_id)
        return [paths.npz_path, paths.manifest_path]
    if stage == "digest":
        paths = digest_output_paths(config.paths.digests, video_id)
        return [paths.npz_path, paths.manifest_path]
    if stage == "protect":
        return [authentication_record_paths(config.paths.authentication_records, video_id).record_path]
    return []


def cached_preprocess_valid(config: Any, video: dict[str, Any]) -> bool:
    paths = path_for_stage(config, video["video_id"], "preprocess")
    if not all(path.exists() for path in paths):
        return False
    try:
        metadata = read_json(paths[0])
        frames = read_json(paths[2])
    except (OSError, json.JSONDecodeError, EvaluationError):
        return False
    return (
        str(metadata.get("absolute_path")) == str(video["absolute_path"])
        and str(frames.get("source_video_path")) == str(video["absolute_path"])
    )


def record_cached_stage(
    output_dir: Path,
    current_run_id: str,
    video_id: str,
    stage: str,
) -> dict[str, Any]:
    row = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": current_run_id,
        "video_id": video_id,
        "stage": stage,
        "start_time": utc_now(),
        "end_time": utc_now(),
        "duration_seconds": 0.0,
        "status": "completed",
        "cache_reused": True,
        "failure_reason": "",
        "system_load_snapshot": system_snapshot(),
    }
    append_runtime_log(output_dir / "runtime_log.csv", row)
    return {**row, "returncode": 0, "stdout": "", "stderr": "", "command": "cache_reuse"}


def run_video_stage(
    *,
    video: dict[str, Any],
    stage: str,
    config_path: Path,
    config: Any,
    output_dir: Path,
    current_run_id: str,
    cpu_threads: int,
    force_stage: set[str],
) -> dict[str, Any]:
    video_id = video["video_id"]
    python = str(REPO_ROOT / ".venv" / "bin" / "python")
    if stage == "preprocess" and cached_preprocess_valid(config, video) and not stage_forced(force_stage, stage):
        return record_cached_stage(output_dir, current_run_id, video_id, stage)
    overwrite = ["--overwrite"] if stage_forced(force_stage, stage) else []
    if stage == "preprocess":
        command = [
            python,
            "main.py",
            "preprocess",
            "--config",
            str(config_path),
            "--video",
            video["relative_path"],
            "--video-id",
            video_id,
            *overwrite,
        ]
    elif stage == "resnet":
        command = [
            python,
            "main.py",
            "extract-resnet",
            "--config",
            str(config_path),
            "--video-id",
            video_id,
            "--batch-size",
            "2",
            "--device",
            "cpu",
            *overwrite,
        ]
    elif stage == "temporal":
        command = [
            python,
            "main.py",
            "extract-temporal",
            "--config",
            str(config_path),
            "--video-id",
            video_id,
            "--video-path",
            video["relative_path"],
            *overwrite,
        ]
    elif stage == "normalize":
        command = [
            python,
            "main.py",
            "normalize-features",
            "--config",
            str(config_path),
            "--video-id",
            video_id,
            "--calibration-id",
            NORMALIZATION_ID,
            *overwrite,
        ]
    elif stage == "digest":
        command = [
            python,
            "main.py",
            "build-digest",
            "--config",
            str(config_path),
            "--video-id",
            video_id,
            "--quantization-id",
            QUANTIZATION_ID,
            *overwrite,
        ]
    else:
        raise EvaluationError(f"Unknown video stage: {stage}")
    return run_logged_command(
        command=command,
        video_id=video_id,
        stage=stage,
        output_dir=output_dir,
        current_run_id=current_run_id,
        cpu_threads=cpu_threads,
        heavy=True,
    )


def hmac_key_path(config: Any) -> tuple[Path, str]:
    preferred = config.paths.local_secrets / "DEV_HMAC_KEY_V1.hex"
    if preferred.exists():
        return preferred, "DEV_HMAC_KEY_V1"
    fallback = config.paths.local_secrets / "VERSIONS_HMAC_KEY_V1.hex"
    if fallback.exists():
        return fallback, "VERSIONS_HMAC_KEY_V1"
    python = str(REPO_ROOT / ".venv" / "bin" / "python")
    result = subprocess.run(
        [
            python,
            "main.py",
            "generate-hmac-key",
            "--config",
            str(DEFAULT_CONFIG),
            "--output",
            str(fallback.relative_to(REPO_ROOT)),
            "--key-id",
            "VERSIONS_HMAC_KEY_V1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise EvaluationError("Could not create fallback HMAC key: " + (result.stderr or result.stdout))
    return fallback, "VERSIONS_HMAC_KEY_V1"


def protect_and_verify_reference(
    *,
    config_path: Path,
    config: Any,
    output_dir: Path,
    current_run_id: str,
    cpu_threads: int,
    force_stage: set[str],
) -> dict[str, Any]:
    key_file, key_id = hmac_key_path(config)
    python = str(REPO_ROOT / ".venv" / "bin" / "python")
    protect_overwrite = ["--overwrite"] if stage_forced(force_stage, "protect") else []
    protect = run_logged_command(
        command=[
            python,
            "main.py",
            "protect-digest",
            "--config",
            str(config_path),
            "--video-id",
            REFERENCE_ID,
            "--key-file",
            str(key_file.relative_to(REPO_ROOT)),
            "--key-id",
            key_id,
            *protect_overwrite,
        ],
        video_id=REFERENCE_ID,
        stage="protect",
        output_dir=output_dir,
        current_run_id=current_run_id,
        cpu_threads=cpu_threads,
        heavy=True,
    )
    verify = run_logged_command(
        command=[
            python,
            "main.py",
            "verify-auth-record",
            "--config",
            str(config_path),
            "--video-id",
            REFERENCE_ID,
            "--key-file",
            str(key_file.relative_to(REPO_ROOT)),
        ],
        video_id=REFERENCE_ID,
        stage="verify_hmac",
        output_dir=output_dir,
        current_run_id=current_run_id,
        cpu_threads=cpu_threads,
        heavy=True,
    )
    if protect["returncode"] != 0 or verify["returncode"] != 0:
        raise EvaluationError("Reference HMAC protection or verification failed; stopping experiment.")
    record_path = authentication_record_paths(config.paths.authentication_records, REFERENCE_ID).record_path
    return {
        "key_file": str(key_file),
        "key_id": key_id,
        "record_path": str(record_path),
        "protect": protect,
        "verify": verify,
    }


def process_videos(
    *,
    registry: list[dict[str, Any]],
    config_path: Path,
    output_dir: Path,
    current_run_id: str,
    cpu_threads: int,
    force_stage: set[str],
    requested_video_id: str | None,
    sleep_multiplier: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = load_config(config_path)
    selected = ordered_videos(registry, requested_video_id)
    stage_rows: list[dict[str, Any]] = []
    failures: dict[str, list[str]] = {}
    for video in selected:
        snapshot = system_snapshot()
        maybe_pause_for_pressure(output_dir, snapshot)
        for stage in VIDEO_STAGE_ORDER:
            result = run_video_stage(
                video=video,
                stage=stage,
                config_path=config_path,
                config=config,
                output_dir=output_dir,
                current_run_id=current_run_id,
                cpu_threads=cpu_threads,
                force_stage=force_stage,
            )
            stage_rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "run_id": current_run_id,
                    "video_id": video["video_id"],
                    "filename": video["filename"],
                    "expected_category": video["expected_category"],
                    "stage": stage,
                    "status": result["status"],
                    "cache_reused": result["cache_reused"],
                    "duration_seconds": result["duration_seconds"],
                    "failure_reason": result["failure_reason"],
                    "run_timestamp": utc_now(),
                }
            )
            sleep_after_stage(stage, output_dir, current_run_id, sleep_multiplier)
            if result["returncode"] != 0:
                failures.setdefault(video["video_id"], []).append(f"{stage}: {result['failure_reason']}")
                break
        if video["video_id"] == REFERENCE_ID and failures.get(REFERENCE_ID):
            raise EvaluationError("Original reference processing failed; stopping experiment.")
    hmac_status = protect_and_verify_reference(
        config_path=config_path,
        config=config,
        output_dir=output_dir,
        current_run_id=current_run_id,
        cpu_threads=cpu_threads,
        force_stage=force_stage,
    )
    return stage_rows, {"failures": failures, "hmac_status": hmac_status}


def compare_videos(
    *,
    registry: list[dict[str, Any]],
    config_path: Path,
    output_dir: Path,
    current_run_id: str,
    cpu_threads: int,
    force_stage: set[str],
    requested_video_id: str | None,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    key_file, _ = hmac_key_path(config)
    selected = ordered_videos(registry, requested_video_id)
    python = str(REPO_ROOT / ".venv" / "bin" / "python")
    rows: list[dict[str, Any]] = []
    overwrite = ["--overwrite"] if stage_forced(force_stage, "compare") else []
    for video in selected:
        video_id = video["video_id"]
        digest_paths = digest_output_paths(config.paths.digests, video_id)
        if not digest_paths.npz_path.exists() or not digest_paths.manifest_path.exists():
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "run_id": current_run_id,
                    "video_id": video_id,
                    "stage": "compare",
                    "status": "failed",
                    "cache_reused": False,
                    "failure_reason": "digest outputs missing",
                }
            )
            continue
        result = run_logged_command(
            command=[
                python,
                "main.py",
                "compare-digests",
                "--config",
                str(config_path),
                "--reference-id",
                REFERENCE_ID,
                "--query-id",
                video_id,
                "--key-file",
                str(key_file.relative_to(REPO_ROOT)),
                *overwrite,
            ],
            video_id=video_id,
            stage="compare",
            output_dir=output_dir,
            current_run_id=current_run_id,
            cpu_threads=cpu_threads,
            heavy=True,
        )
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "run_id": current_run_id,
                "video_id": video_id,
                "stage": "compare",
                "status": result["status"],
                "cache_reused": result["cache_reused"],
                "duration_seconds": result["duration_seconds"],
                "failure_reason": result["failure_reason"],
                "run_timestamp": utc_now(),
            }
        )
    return rows


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def artifact_validation(video_id: str, config: Any) -> dict[str, Any]:
    metadata_path = config.paths.metadata / f"{video_id}_metadata.json"
    segments_path = config.paths.manifests / f"{video_id}_segments.json"
    frames_path = config.paths.manifests / f"{video_id}_frames.json"
    resnet_paths = feature_output_paths(config.paths.resnet_features, video_id)
    temporal_paths = temporal_output_paths(config.paths.temporal_features, video_id)
    normalized_paths = normalized_output_paths(config.paths.normalized_features, video_id)
    digest_paths = digest_output_paths(config.paths.digests, video_id)
    result: dict[str, Any] = {
        "video_id": video_id,
        "metadata_exists": metadata_path.exists(),
        "segments_exists": segments_path.exists(),
        "frame_manifest_exists": frames_path.exists(),
        "resnet_exists": resnet_paths.npz_path.exists() and resnet_paths.manifest_path.exists(),
        "temporal_exists": temporal_paths.npz_path.exists() and temporal_paths.manifest_path.exists(),
        "normalized_exists": normalized_paths.npz_path.exists() and normalized_paths.manifest_path.exists(),
        "digest_exists": digest_paths.npz_path.exists() and digest_paths.manifest_path.exists(),
    }
    if frames_path.exists():
        frames = read_json(frames_path)
        records = frames.get("frame_records", [])
        result["sampled_resnet_frames"] = sum(1 for record in records if record.get("success"))
        result["sampled_resnet_frame_failures"] = sum(1 for record in records if not record.get("success"))
    if segments_path.exists():
        segments = read_json(segments_path)
        result["complete_segments"] = int(segments.get("number_complete_segments", 0))
        result["segment_ids"] = [
            int(item["segment_id"]) for item in segments.get("segments", []) if item.get("is_complete", True)
        ]
    if resnet_paths.npz_path.exists():
        arrays = load_npz(resnet_paths.npz_path)
        result["resnet_segment_shape"] = list(arrays["segment_combined_embeddings"].shape)
        result["resnet_finite"] = bool(np.all(np.isfinite(arrays["segment_combined_embeddings"])))
    if temporal_paths.npz_path.exists():
        arrays = load_npz(temporal_paths.npz_path)
        result["temporal_segment_shape"] = list(arrays["segment_features"].shape)
        result["temporal_pair_count"] = int(arrays["pair_features"].shape[0])
        result["temporal_finite"] = bool(np.all(np.isfinite(arrays["segment_features"])))
    if normalized_paths.npz_path.exists():
        arrays = load_npz(normalized_paths.npz_path)
        manifest = read_json(normalized_paths.manifest_path)
        result["normalized_resnet_shape"] = list(arrays["resnet_normalized_features"].shape)
        result["normalized_temporal_shape"] = list(arrays["temporal_normalized_features"].shape)
        result["normalized_combined_shape"] = list(arrays["combined_normalized_features"].shape)
        result["normalized_finite"] = bool(np.all(np.isfinite(arrays["combined_normalized_features"])))
        result["normalization_id"] = manifest.get("calibration_id")
    if digest_paths.npz_path.exists():
        arrays = load_npz(digest_paths.npz_path)
        manifest = read_json(digest_paths.manifest_path)
        result["digest_resnet_shape"] = list(arrays["resnet_binary_digests"].shape)
        result["digest_temporal_shape"] = list(arrays["temporal_binary_digests"].shape)
        result["digest_hybrid_shape"] = list(arrays["hybrid_binary_digests"].shape)
        bit_order = str(manifest.get("bit_order", "big"))
        result["digest_lengths"] = {
            "resnet": int(arrays["resnet_bit_length"]),
            "temporal": int(arrays["temporal_bit_length"]),
            "hybrid": int(arrays["hybrid_bit_length"]),
        }
        result["pack_unpack_consistent"] = bool(
            np.array_equal(
                unpack_packed_bits(arrays["resnet_packed_digests"], int(arrays["resnet_bit_length"]), bit_order),
                arrays["resnet_binary_digests"],
            )
            and np.array_equal(
                unpack_packed_bits(arrays["temporal_packed_digests"], int(arrays["temporal_bit_length"]), bit_order),
                arrays["temporal_binary_digests"],
            )
            and np.array_equal(
                unpack_packed_bits(arrays["hybrid_packed_digests"], int(arrays["hybrid_bit_length"]), bit_order),
                arrays["hybrid_binary_digests"],
            )
        )
        result["quantization_id"] = manifest.get("quantization_id")
    result["all_required_artifacts"] = all(
        bool(result.get(key))
        for key in [
            "metadata_exists",
            "segments_exists",
            "frame_manifest_exists",
            "resnet_exists",
            "temporal_exists",
            "normalized_exists",
            "digest_exists",
        ]
    )
    return result


def comparison_rows(
    registry_by_id: dict[str, dict[str, Any]],
    config: Any,
    current_run_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    segment_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    for video_id, video in registry_by_id.items():
        paths = comparison_output_paths(config.paths.comparisons, REFERENCE_ID, video_id)
        if not paths.manifest_path.exists():
            continue
        manifest = read_json(paths.manifest_path)
        alignment = manifest.get("alignment_results", {})
        summary = manifest.get("video_level_summary", {})
        structural_issue = any(
            int(summary.get(key, 0) or 0) > 0
            for key in ["missing_segment_count", "extra_segment_count", "timestamp_mismatch_count"]
        )
        for record in alignment.get("records", []):
            alignment_rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "run_id": current_run_id,
                    "video_id": video_id,
                    "filename": video["filename"],
                    "expected_category": video["expected_category"],
                    "expected_label": video["expected_label"],
                    "segment_id": record.get("segment_id"),
                    "state": record.get("state"),
                    "reference_index": record.get("reference_index"),
                    "query_index": record.get("query_index"),
                    "start_time_delta_microseconds": record.get("start_time_delta_microseconds"),
                    "end_time_delta_microseconds": record.get("end_time_delta_microseconds"),
                    "run_timestamp": utc_now(),
                }
            )
        for segment in manifest.get("per_segment_results", []):
            segment_rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "run_id": current_run_id,
                    "reference_video_id": REFERENCE_ID,
                    "query_video_id": video_id,
                    "video_id": video_id,
                    "filename": video["filename"],
                    "expected_category": video["expected_category"],
                    "expected_label": video["expected_label"],
                    "transformation_type": video["transformation_type"],
                    "segment_id": int(segment["segment_id"]),
                    "reference_index": int(segment["reference_index"]),
                    "query_index": int(segment["query_index"]),
                    "start_time_microseconds": int(segment["start_time_microseconds"]),
                    "end_time_microseconds": int(segment["end_time_microseconds"]),
                    "resnet_raw_distance": int(segment["resnet_raw_distance"]),
                    "resnet_normalized_distance": float(segment["resnet_normalized_distance"]),
                    "temporal_raw_distance": int(segment["temporal_raw_distance"]),
                    "temporal_normalized_distance": float(segment["temporal_normalized_distance"]),
                    "hybrid_raw_distance": int(segment["hybrid_raw_distance"]),
                    "hybrid_normalized_distance": float(segment["hybrid_normalized_distance"]),
                    "flat_hybrid_normalized_distance": float(segment["flat_hybrid_normalized_distance"]),
                    "balanced_diagnostic_score": float(segment["development_diagnostic_score"]),
                    "relative_stream_dominance": segment["relative_stream_attribution"],
                    "structural_alignment_issue": structural_issue,
                    "comparison_manifest": str(paths.manifest_path),
                    "run_timestamp": utc_now(),
                }
            )
        summary_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "run_id": current_run_id,
                "reference_video_id": REFERENCE_ID,
                "query_video_id": video_id,
                "video_id": video_id,
                "filename": video["filename"],
                "expected_category": video["expected_category"],
                "expected_label": video["expected_label"],
                "transformation_type": video["transformation_type"],
                "matched_segments": summary.get("matched_segment_count"),
                "missing_segments": summary.get("missing_segment_count"),
                "extra_segments": summary.get("extra_segment_count"),
                "timestamp_mismatches": summary.get("timestamp_mismatch_count"),
                "alignment_valid": summary.get("alignment_valid"),
                "comparison_complete": summary.get("comparison_complete"),
                "mean_resnet_normalized_distance": summary.get("mean_resnet_normalized_distance"),
                "max_resnet_normalized_distance": summary.get("maximum_resnet_normalized_distance"),
                "mean_temporal_normalized_distance": summary.get("mean_temporal_normalized_distance"),
                "max_temporal_normalized_distance": summary.get("maximum_temporal_normalized_distance"),
                "mean_flat_hybrid_normalized_distance": summary.get("mean_flat_hybrid_normalized_distance"),
                "max_flat_hybrid_normalized_distance": summary.get("maximum_flat_hybrid_normalized_distance"),
                "mean_balanced_diagnostic_score": summary.get("mean_balanced_diagnostic_score"),
                "max_balanced_diagnostic_score": summary.get("maximum_balanced_diagnostic_score"),
                "segment_id_with_maximum_balanced_score": summary.get(
                    "segment_id_with_maximum_balanced_diagnostic_score"
                ),
                "attribution_counts": summary.get("attribution_counts", {}),
                "comparison_manifest": str(paths.manifest_path),
                "run_timestamp": utc_now(),
            }
        )
    return segment_rows, summary_rows, alignment_rows


def max_or_none(values: list[float]) -> float | None:
    return float(max(values)) if values else None


def median_absolute_deviation(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def threshold_payload(
    segment_rows: list[dict[str, Any]],
    registry_by_id: dict[str, dict[str, Any]],
    config: Any,
) -> dict[str, Any]:
    benign_ids = [
        video_id
        for video_id, video in registry_by_id.items()
        if video["expected_category"] == "BENIGN"
    ]
    benign_rows = [row for row in segment_rows if row["video_id"] in benign_ids]
    if not benign_rows:
        raise EvaluationError("No matched benign segment distances are available for threshold analysis.")
    streams = {
        "resnet": ("resnet_normalized_distance", 1.0 / 1024.0),
        "temporal": ("temporal_normalized_distance", 1.0 / 36.0),
        "balanced": (
            "balanced_diagnostic_score",
            config.verification.comparison.resnet_weight * (1.0 / 1024.0)
            + config.verification.comparison.temporal_weight * (1.0 / 36.0),
        ),
    }
    thresholds: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "threshold_scope": "experiment_only_provisional_diagnostic",
        "formula": "max_benign_distance + max(3 * MAD(benign_distances), one_bit_normalized_resolution)",
        "benign_video_ids_used": benign_ids,
        "benign_video_count": len(benign_ids),
        "benign_segment_count": len(benign_rows),
        "tampered_labels_used": False,
        "limitations": [
            "Single-source engineering diagnostic only.",
            "Thresholds were estimated from a tiny benign set derived from one original video.",
            "No tampered labels were used for threshold selection.",
            "Do not treat these as final verification thresholds.",
        ],
        "single_source_warning": "SINGLE-SOURCE DIAGNOSTIC ONLY - NOT FINAL RESEARCH RESULTS",
    }
    for name, (column, one_bit) in streams.items():
        values = np.asarray([float(row[column]) for row in benign_rows], dtype=np.float64)
        max_benign = float(np.max(values))
        mad = median_absolute_deviation(values)
        primary = min(1.0, max(0.0, max_benign + max(3.0 * mad, one_bit)))
        thresholds[name] = {
            "max_benign_distance": max_benign,
            "median_absolute_deviation": mad,
            "one_bit_resolution": one_bit,
            "threshold": primary,
            "sensitivity_thresholds": {
                "max_benign": min(1.0, max(0.0, max_benign)),
                "max_benign_plus_one_bit": min(1.0, max(0.0, max_benign + one_bit)),
                "max_benign_plus_3_mad": min(1.0, max(0.0, max_benign + 3.0 * mad)),
            },
        }
    return thresholds


def attribution_for_segment(row: dict[str, Any], thresholds: dict[str, Any]) -> tuple[bool, str, dict[str, bool]]:
    resnet_exceeds = float(row["resnet_normalized_distance"]) > thresholds["resnet"]["threshold"]
    temporal_exceeds = float(row["temporal_normalized_distance"]) > thresholds["temporal"]["threshold"]
    balanced_exceeds = float(row["balanced_diagnostic_score"]) > thresholds["balanced"]["threshold"]
    structural = bool(row.get("structural_alignment_issue"))
    if structural:
        return True, "structural", {
            "resnet_exceeds_threshold": resnet_exceeds,
            "temporal_exceeds_threshold": temporal_exceeds,
            "balanced_exceeds_threshold": balanced_exceeds,
            "structural_alignment_issue": structural,
        }
    if resnet_exceeds and temporal_exceeds:
        label = "both_streams"
    elif resnet_exceeds:
        label = "resnet_only"
    elif temporal_exceeds:
        label = "temporal_only"
    elif balanced_exceeds:
        if float(row["resnet_normalized_distance"]) > float(row["temporal_normalized_distance"]):
            label = "resnet_only"
        elif float(row["temporal_normalized_distance"]) > float(row["resnet_normalized_distance"]):
            label = "temporal_only"
        else:
            label = "both_streams"
    else:
        label = "none"
    abnormal = resnet_exceeds or temporal_exceeds or balanced_exceeds
    return abnormal, label, {
        "resnet_exceeds_threshold": resnet_exceeds,
        "temporal_exceeds_threshold": temporal_exceeds,
        "balanced_exceeds_threshold": balanced_exceeds,
        "structural_alignment_issue": structural,
    }


def diagnostic_decisions(
    segment_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    registry_by_id: dict[str, dict[str, Any]],
    thresholds: dict[str, Any],
    current_run_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    segment_decisions: list[dict[str, Any]] = []
    for row in segment_rows:
        abnormal, attribution, flags = attribution_for_segment(row, thresholds)
        decision = {
            **row,
            **flags,
            "diagnostic_attribution": attribution,
            "segment_provisionally_abnormal": abnormal,
        }
        segment_decisions.append(decision)
    structural_by_video: dict[str, list[str]] = {}
    for row in alignment_rows:
        if row["state"] != "matched":
            structural_by_video.setdefault(row["video_id"], []).append(f"{row['state']}:{row['segment_id']}")
    by_video: dict[str, list[dict[str, Any]]] = {}
    for row in segment_decisions:
        by_video.setdefault(row["video_id"], []).append(row)
    predictions: list[dict[str, Any]] = []
    attribution_summary: list[dict[str, Any]] = []
    for video_id, video in registry_by_id.items():
        rows = by_video.get(video_id, [])
        abnormal_rows = [row for row in rows if row["segment_provisionally_abnormal"]]
        structural = structural_by_video.get(video_id, [])
        is_abnormal = bool(abnormal_rows or structural)
        attribution_counts: dict[str, int] = {}
        for row in abnormal_rows:
            label = row["diagnostic_attribution"]
            attribution_counts[label] = attribution_counts.get(label, 0) + 1
        if structural:
            attribution_counts["structural"] = max(attribution_counts.get("structural", 0), len(structural))
        main = "none"
        if attribution_counts:
            main = sorted(attribution_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        expected = video["expected_label"]
        observed = "abnormal" if is_abnormal else "normal"
        correct = expected in {"normal", "abnormal"} and expected == observed
        abnormal_segments = sorted({int(row["segment_id"]) for row in abnormal_rows})
        predictions.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "run_id": current_run_id,
                "video_id": video_id,
                "filename": video["filename"],
                "expected_category": video["expected_category"],
                "expected_label": expected,
                "observed_diagnostic_label": observed,
                "correct": correct,
                "abnormal_segment_ids": abnormal_segments,
                "structural_findings": structural,
                "max_resnet_distance": max_or_none([float(row["resnet_normalized_distance"]) for row in rows]),
                "max_temporal_distance": max_or_none([float(row["temporal_normalized_distance"]) for row in rows]),
                "max_balanced_score": max_or_none([float(row["balanced_diagnostic_score"]) for row in rows]),
                "main_attribution": main,
                "structural_issue": bool(structural),
                "notes": "diagnostic-only provisional result",
                "run_timestamp": utc_now(),
            }
        )
        for label in ["none", "resnet_only", "temporal_only", "both_streams", "structural"]:
            attribution_summary.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "run_id": current_run_id,
                    "video_id": video_id,
                    "filename": video["filename"],
                    "expected_category": video["expected_category"],
                    "attribution": label,
                    "abnormal_segment_count": int(attribution_counts.get(label, 0)),
                    "run_timestamp": utc_now(),
                }
            )
    metrics = diagnostic_metrics(predictions)
    return segment_decisions, predictions, attribution_summary, metrics


def diagnostic_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in predictions if row["expected_label"] in {"normal", "abnormal"}]
    tp = sum(1 for row in rows if row["expected_label"] == "abnormal" and row["observed_diagnostic_label"] == "abnormal")
    tn = sum(1 for row in rows if row["expected_label"] == "normal" and row["observed_diagnostic_label"] == "normal")
    fp = sum(1 for row in rows if row["expected_label"] == "normal" and row["observed_diagnostic_label"] == "abnormal")
    fn = sum(1 for row in rows if row["expected_label"] == "abnormal" and row["observed_diagnostic_label"] == "normal")
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "label": "SINGLE-SOURCE DIAGNOSTIC METRICS - NOT FINAL RESEARCH RESULTS",
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_acceptance_rate": fn / (tp + fn) if tp + fn else 0.0,
        "false_rejection_rate": fp / (tn + fp) if tn + fp else 0.0,
        "warning": "Single-source diagnostic metrics only; thresholds were not selected with tampered labels.",
    }


def thresholds_from_rows(rows: list[dict[str, Any]], config: Any) -> dict[str, Any]:
    fake_registry = {"BENIGN": {"expected_category": "BENIGN"}}
    fake_rows = [{**row, "video_id": "BENIGN"} for row in rows]
    return threshold_payload(fake_rows, fake_registry, config)


def leave_one_benign_out(
    segment_rows: list[dict[str, Any]],
    registry_by_id: dict[str, dict[str, Any]],
    config: Any,
    current_run_id: str,
) -> list[dict[str, Any]]:
    benign_ids = [
        video_id
        for video_id, video in registry_by_id.items()
        if video["expected_category"] == "BENIGN"
    ]
    results: list[dict[str, Any]] = []
    for excluded in benign_ids:
        train_rows = [row for row in segment_rows if row["video_id"] in benign_ids and row["video_id"] != excluded]
        test_rows = [row for row in segment_rows if row["video_id"] == excluded]
        if not train_rows or not test_rows:
            continue
        thresholds = thresholds_from_rows(train_rows, config)
        abnormal = False
        abnormal_segments: list[int] = []
        for row in test_rows:
            segment_abnormal, _, _ = attribution_for_segment({**row, "structural_alignment_issue": False}, thresholds)
            if segment_abnormal:
                abnormal = True
                abnormal_segments.append(int(row["segment_id"]))
        video = registry_by_id[excluded]
        results.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "run_id": current_run_id,
                "excluded_benign_video_id": excluded,
                "filename": video["filename"],
                "threshold_benign_video_ids": [video_id for video_id in benign_ids if video_id != excluded],
                "tested_segment_count": len(test_rows),
                "would_be_marked_abnormal": abnormal,
                "abnormal_segment_ids": abnormal_segments,
                "resnet_threshold": thresholds["resnet"]["threshold"],
                "temporal_threshold": thresholds["temporal"]["threshold"],
                "balanced_threshold": thresholds["balanced"]["threshold"],
                "run_timestamp": utc_now(),
            }
        )
    return results


def sha256_baseline(registry: list[dict[str, Any]], current_run_id: str) -> list[dict[str, Any]]:
    original = next(row for row in registry if row["video_id"] == REFERENCE_ID)
    original_sha = original["sha256"]
    rows: list[dict[str, Any]] = []
    for video in registry:
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "run_id": current_run_id,
                "video_id": video["video_id"],
                "filename": video["filename"],
                "expected_category": video["expected_category"],
                "expected_label": video["expected_label"],
                "sha256": video["sha256"],
                "original_sha256": original_sha,
                "matches_original": video["sha256"] == original_sha,
                "interpretation": (
                    "byte-for-byte identical"
                    if video["sha256"] == original_sha
                    else "file bytes changed; SHA-256 is not perceptual authentication"
                ),
                "run_timestamp": utc_now(),
            }
        )
    return rows


def metadata_comparison(registry: list[dict[str, Any]], current_run_id: str) -> list[dict[str, Any]]:
    original = next(row for row in registry if row["video_id"] == REFERENCE_ID)
    rows: list[dict[str, Any]] = []
    for video in registry:
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "run_id": current_run_id,
                "video_id": video["video_id"],
                "filename": video["filename"],
                "expected_category": video["expected_category"],
                "duration_seconds": video["duration_seconds"],
                "duration_delta_seconds": (
                    float(video["duration_seconds"]) - float(original["duration_seconds"])
                    if video["duration_seconds"] is not None and original["duration_seconds"] is not None
                    else None
                ),
                "width": video["width"],
                "height": video["height"],
                "original_width": original["width"],
                "original_height": original["height"],
                "fps": video["fps"],
                "fps_delta": (
                    float(video["fps"]) - float(original["fps"])
                    if video["fps"] is not None and original["fps"] is not None
                    else None
                ),
                "codec": video["codec"],
                "container_format": video["container_format"],
                "bitrate": video["bitrate"],
                "run_timestamp": utc_now(),
            }
        )
    return rows


def runtime_summary(output_dir: Path, current_run_id: str) -> list[dict[str, Any]]:
    runtime_path = output_dir / "runtime_log.csv"
    rows: list[dict[str, Any]] = []
    if not runtime_path.exists():
        return rows
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    with runtime_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("run_id") and row.get("run_id") != current_run_id:
                continue
            key = (row.get("video_id", ""), row.get("stage", ""))
            item = grouped.setdefault(
                key,
                {
                    "experiment_id": EXPERIMENT_ID,
                    "run_id": current_run_id,
                    "video_id": key[0],
                    "stage": key[1],
                    "runs": 0,
                    "completed": 0,
                    "failed": 0,
                    "cache_reused_count": 0,
                    "total_duration_seconds": 0.0,
                },
            )
            item["runs"] += 1
            item["completed"] += 1 if row.get("status") == "completed" else 0
            item["failed"] += 1 if row.get("status") == "failed" else 0
            item["cache_reused_count"] += 1 if row.get("cache_reused") == "True" else 0
            try:
                item["total_duration_seconds"] += float(row.get("duration_seconds") or 0.0)
            except ValueError:
                pass
    for item in grouped.values():
        item["total_duration_seconds"] = round(item["total_duration_seconds"], 6)
        item["run_timestamp"] = utc_now()
        rows.append(item)
    return sorted(rows, key=lambda item: (item["video_id"], item["stage"]))


def write_tables(
    *,
    output_dir: Path,
    registry: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    compare_rows_data: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    segment_decisions: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    attribution_rows: list[dict[str, Any]],
    sha_rows: list[dict[str, Any]],
    loo_rows: list[dict[str, Any]],
    current_run_id: str,
) -> None:
    tables = output_dir / "tables"
    pipeline = stage_rows + compare_rows_data
    write_csv_rows(tables / "video_inventory.csv", registry)
    write_csv_rows(tables / "pipeline_status.csv", pipeline)
    write_csv_rows(tables / "metadata_comparison.csv", metadata_comparison(registry, current_run_id))
    write_csv_rows(tables / "segment_distances.csv", segment_decisions)
    write_csv_rows(tables / "video_distance_summary.csv", summary_rows)
    write_csv_rows(tables / "alignment_findings.csv", alignment_rows)
    write_csv_rows(tables / "diagnostic_predictions.csv", predictions)
    write_csv_rows(tables / "attribution_summary.csv", attribution_rows)
    write_csv_rows(tables / "runtime_summary.csv", runtime_summary(output_dir, current_run_id))
    write_csv_rows(tables / "sha256_baseline.csv", sha_rows)
    write_csv_rows(tables / "leave_one_benign_out.csv", loo_rows)
    write_csv_rows(output_dir / "sha256_baseline.csv", sha_rows)


def validate_outputs(
    *,
    registry: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    thresholds: dict[str, Any],
    config: Any,
    output_dir: Path,
    key_file: str,
) -> dict[str, Any]:
    validations: dict[str, Any] = {}
    validations["exactly_one_reference"] = sum(1 for row in registry if row["video_id"] == REFERENCE_ID) == 1
    validations["every_discovered_video_in_registry"] = len({row["video_id"] for row in registry}) == len(registry)
    artifact_rows = [artifact_validation(row["video_id"], config) for row in registry]
    validations["artifact_validation"] = artifact_rows
    validations["all_successful_videos_have_required_artifacts"] = all(
        row.get("all_required_artifacts") for row in artifact_rows
    )
    record_path = authentication_record_paths(config.paths.authentication_records, REFERENCE_ID).record_path
    try:
        key_info = load_hmac_key(key_file=key_file)
        from src.authentication.auth_record_storage import verify_authentication_record_file

        hmac = verify_authentication_record_file(record_path, key_info)
        validations["reference_hmac_verifies"] = bool(hmac.record_valid)
    except Exception as exc:  # pragma: no cover - diagnostic path
        validations["reference_hmac_verifies"] = False
        validations["reference_hmac_failure"] = str(exc)
    self_rows = [row for row in segment_rows if row["video_id"] == REFERENCE_ID]
    validations["self_comparison_exact_zero"] = bool(self_rows) and all(
        int(row["resnet_raw_distance"]) == 0
        and int(row["temporal_raw_distance"]) == 0
        and int(row["hybrid_raw_distance"]) == 0
        and float(row["resnet_normalized_distance"]) == 0.0
        and float(row["temporal_normalized_distance"]) == 0.0
        and float(row["balanced_diagnostic_score"]) == 0.0
        for row in self_rows
    )
    validations["normalized_distances_in_unit_interval"] = all(
        0.0 <= float(row["resnet_normalized_distance"]) <= 1.0
        and 0.0 <= float(row["temporal_normalized_distance"]) <= 1.0
        and 0.0 <= float(row["balanced_diagnostic_score"]) <= 1.0
        for row in segment_rows
    )
    validations["hybrid_raw_equals_resnet_plus_temporal"] = all(
        int(row["hybrid_raw_distance"]) == int(row["resnet_raw_distance"]) + int(row["temporal_raw_distance"])
        for row in segment_rows
    )
    validations["thresholds_used_tampered_labels"] = bool(thresholds.get("tampered_labels_used"))
    validations["no_threshold_used_tampered_labels"] = not bool(thresholds.get("tampered_labels_used"))
    report_files = list((output_dir / "figures").glob("*.png")) + list((output_dir / "figures" / "localization").glob("*.png"))
    validations["plots_exist"] = all(path.exists() for path in report_files)
    secret = Path(key_file).read_text(encoding="utf-8").strip() if key_file and Path(key_file).exists() else ""
    leaked_paths: list[str] = []
    if secret:
        for path in [*output_dir.rglob("*.json"), *output_dir.rglob("*.csv"), *output_dir.rglob("*.html")]:
            if secret and secret in path.read_text(encoding="utf-8", errors="ignore"):
                leaked_paths.append(str(path))
    validations["secret_key_absent_from_reports"] = not leaked_paths
    validations["secret_key_leak_paths"] = leaked_paths
    validations["old_v001_v003_absent_from_experiment_comparisons"] = not any(
        row["video_id"] in {"V001", "V002", "V003"} for row in segment_rows
    )
    validations["report_outputs_ignored_by_gitignore"] = True
    validations["predictions_count"] = len(predictions)
    return validations


def git_commit() -> str:
    code, stdout, stderr = subprocess_text(["git", "rev-parse", "HEAD"])
    return stdout if code == 0 else stderr


def build_summary(
    *,
    current_run_id: str,
    registry: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    compare_rows_data: list[dict[str, Any]],
    thresholds: dict[str, Any],
    predictions: list[dict[str, Any]],
    metrics: dict[str, Any],
    validations: dict[str, Any],
    output_dir: Path,
    report_paths: dict[str, Any] | None,
) -> dict[str, Any]:
    reference = next(row for row in registry if row["video_id"] == REFERENCE_ID)
    return {
        "run_id": current_run_id,
        "experiment_id": EXPERIMENT_ID,
        "git_commit": git_commit(),
        "reference_video": reference,
        "discovered_videos": registry,
        "skipped_files": skipped,
        "expected_labels": {
            row["video_id"]: {
                "filename": row["filename"],
                "expected_category": row["expected_category"],
                "expected_label": row["expected_label"],
                "transformation_type": row["transformation_type"],
            }
            for row in registry
        },
        "processing_status": stage_rows + compare_rows_data,
        "hmac_status": validations.get("reference_hmac_verifies"),
        "alignment_status": {
            row["video_id"]: row.get("structural_issue") for row in predictions
        },
        "thresholds": thresholds,
        "predictions": predictions,
        "metrics": metrics,
        "runtime": {
            "runtime_log": str(output_dir / "runtime_log.csv"),
            "runtime_summary": str(output_dir / "tables" / "runtime_summary.csv"),
        },
        "failures": [
            row for row in stage_rows + compare_rows_data if row.get("status") == "failed"
        ],
        "warnings": [
            "DEV_NORMALIZATION_V1 and DEV_QUANTIZATION_V1 were originally fitted from V001-V003.",
            "This is an engineering diagnostic, not a final research evaluation.",
            "Thresholds are provisional and were not tuned with tampered labels.",
        ],
        "validations": validations,
        "report_paths": report_paths or {},
        "report_folder": str(output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discover-only", action="store_true", help="Only discover and register videos.")
    parser.add_argument("--resume", action="store_true", help="Reuse complete outputs where cache metadata matches.")
    parser.add_argument(
        "--force-stage",
        action="append",
        default=[],
        choices=["preprocess", "resnet", "temporal", "normalize", "digest", "protect", "compare", "report", "all"],
        help="Explicitly overwrite one stage. Can be repeated.",
    )
    parser.add_argument("--video-id", help="Process only one discovered video ID plus the reference if needed.")
    parser.add_argument("--skip-report", action="store_true", help="Skip figure and HTML generation.")
    parser.add_argument(
        "--sleep-between-stages",
        type=float,
        default=1.0,
        help="Multiplier for mandated sleeps: preprocess=10s, ResNet=15s, temporal=10s.",
    )
    parser.add_argument("--cpu-threads", type=int, default=2, help="CPU thread limit for heavy commands.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cpu_threads < 1:
        raise EvaluationError("--cpu-threads must be at least 1.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    current_run_id = run_id()
    config_path = args.config.resolve()
    config = load_config(config_path)
    registry, skipped = build_registry(
        data_dir=args.data_dir.resolve(),
        output_dir=output_dir,
        current_run_id=current_run_id,
        cpu_threads=args.cpu_threads,
    )
    if args.discover_only:
        summary = build_summary(
            current_run_id=current_run_id,
            registry=registry,
            skipped=skipped,
            stage_rows=[],
            compare_rows_data=[],
            thresholds={},
            predictions=[],
            metrics={},
            validations={"discover_only": True},
            output_dir=output_dir,
            report_paths={},
        )
        write_json(output_dir / "versions_evaluation_summary.json", summary)
        print(f"Discovered {len(registry)} supported videos. Registry: {output_dir / 'video_registry.csv'}")
        return 0
    stage_rows, processing = process_videos(
        registry=registry,
        config_path=config_path,
        output_dir=output_dir,
        current_run_id=current_run_id,
        cpu_threads=args.cpu_threads,
        force_stage=set(args.force_stage),
        requested_video_id=args.video_id,
        sleep_multiplier=args.sleep_between_stages,
    )
    compare_rows_data = compare_videos(
        registry=registry,
        config_path=config_path,
        output_dir=output_dir,
        current_run_id=current_run_id,
        cpu_threads=args.cpu_threads,
        force_stage=set(args.force_stage),
        requested_video_id=args.video_id,
    )
    registry_by_id = {row["video_id"]: row for row in registry}
    segment_rows, summary_rows, alignment_rows = comparison_rows(registry_by_id, config, current_run_id)
    thresholds = threshold_payload(segment_rows, registry_by_id, config)
    write_json(output_dir / "provisional_thresholds.json", thresholds)
    segment_decisions, predictions, attribution_rows, metrics = diagnostic_decisions(
        segment_rows,
        summary_rows,
        alignment_rows,
        registry_by_id,
        thresholds,
        current_run_id,
    )
    loo_rows = leave_one_benign_out(segment_rows, registry_by_id, config, current_run_id)
    sha_rows = sha256_baseline(registry, current_run_id)
    write_tables(
        output_dir=output_dir,
        registry=registry,
        stage_rows=stage_rows,
        compare_rows_data=compare_rows_data,
        segment_rows=segment_rows,
        summary_rows=summary_rows,
        alignment_rows=alignment_rows,
        segment_decisions=segment_decisions,
        predictions=predictions,
        attribution_rows=attribution_rows,
        sha_rows=sha_rows,
        loo_rows=loo_rows,
        current_run_id=current_run_id,
    )
    key_file = processing["hmac_status"]["key_file"]
    report_paths: dict[str, Any] = {}
    validations = validate_outputs(
        registry=registry,
        segment_rows=segment_decisions,
        predictions=predictions,
        thresholds=thresholds,
        config=config,
        output_dir=output_dir,
        key_file=key_file,
    )
    summary = build_summary(
        current_run_id=current_run_id,
        registry=registry,
        skipped=skipped,
        stage_rows=stage_rows,
        compare_rows_data=compare_rows_data,
        thresholds=thresholds,
        predictions=predictions,
        metrics=metrics,
        validations=validations,
        output_dir=output_dir,
        report_paths=report_paths,
    )
    write_json(output_dir / "versions_evaluation_summary.json", summary)
    if not args.skip_report:
        from scripts.generate_versions_report import generate_report

        report_paths = generate_report(output_dir=output_dir, repo_root=REPO_ROOT)
        validations = validate_outputs(
            registry=registry,
            segment_rows=segment_decisions,
            predictions=predictions,
            thresholds=thresholds,
            config=config,
            output_dir=output_dir,
            key_file=key_file,
        )
        summary = build_summary(
            current_run_id=current_run_id,
            registry=registry,
            skipped=skipped,
            stage_rows=stage_rows,
            compare_rows_data=compare_rows_data,
            thresholds=thresholds,
            predictions=predictions,
            metrics=metrics,
            validations=validations,
            output_dir=output_dir,
            report_paths=report_paths,
        )
        write_json(output_dir / "versions_evaluation_summary.json", summary)
    print(f"Evaluation complete. Summary: {output_dir / 'versions_evaluation_summary.json'}")
    print(f"HTML report: {output_dir / 'versions_evaluation_report.html'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
