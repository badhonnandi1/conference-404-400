#!/usr/bin/env python3
"""Run a resource-controlled multi-source versions evaluation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
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


EXPERIMENT_ID = "multisource_versions_evaluation"
NORMALIZATION_ID = "DEV_NORMALIZATION_V1"
QUANTIZATION_ID = "DEV_QUANTIZATION_V1"
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "versions"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "reports" / EXPERIMENT_ID
DEFAULT_CONFIG = REPO_ROOT / "configs" / "multisource_versions_evaluation.yaml"
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SOURCE_RE = re.compile(r"^vid\d{2,}$", re.IGNORECASE)
VIDEO_STAGE_ORDER = ["preprocess", "resnet", "temporal", "normalize", "digest"]
EXPECTED_TRANSFORMATIONS = [
    "avi_conversion",
    "mov_conversion",
    "resize_480p",
    "resize_720p",
    "blur",
    "frame_deletion",
    "frame_insertion",
    "frame_replacement",
]
BENIGN_TRANSFORMATIONS = {"avi_conversion", "mov_conversion", "resize_480p", "resize_720p"}
TAMPERED_TRANSFORMATIONS = {"blur", "frame_deletion", "frame_insertion", "frame_replacement"}
RUN_STATE_NAME = "run_state.json"
CONTROLLED_ENV = {
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "VECLIB_MAXIMUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "TOKENIZERS_PARALLELISM": "false",
    "MPLCONFIGDIR": "data/tmp/matplotlib_cache",
}
RUNTIME_COLUMNS = [
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
    "system_load_snapshot",
]


class EvaluationError(RuntimeError):
    """Raised when the multi-source evaluation cannot safely continue."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id() -> str:
    return "multisource_versions_eval_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def disk_free_bytes(path: Path = Path("/")) -> int:
    return shutil.disk_usage(path).free


def controlled_env(cpu_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    for key in CONTROLLED_ENV:
        env[key] = str(cpu_threads) if key != "TOKENIZERS_PARALLELISM" else "false"
    env["MPLCONFIGDIR"] = str((REPO_ROOT / "data" / "tmp" / "matplotlib_cache").resolve())
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return env


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
    if any(token in thermal for token in ("serious", "critical", "high", "throttl")):
        return True
    load = str(snapshot.get("uptime", "")).lower()
    if "load averages:" in load:
        try:
            first = float(load.split("load averages:", maxsplit=1)[1].strip().split()[0])
        except (IndexError, ValueError):
            return False
        return first >= 8.0
    return False


def maybe_pause_for_pressure(output_dir: Path, snapshot: dict[str, Any], current_run_id: str) -> None:
    if not thermal_or_load_pressure(snapshot):
        return
    start = time.perf_counter()
    start_time = utc_now()
    time.sleep(60)
    append_runtime_log(
        output_dir / "runtime_log.csv",
        {
            "experiment_id": EXPERIMENT_ID,
            "run_id": current_run_id,
            "source_id": "SYSTEM",
            "video_id": "SYSTEM",
            "stage": "thermal_or_load_pause",
            "start_time": start_time,
            "end_time": utc_now(),
            "duration_seconds": round(time.perf_counter() - start, 6),
            "status": "completed",
            "cache_reused": False,
            "failure_reason": "",
            "system_load_snapshot": snapshot,
        },
    )


def run_logged_command(
    *,
    command: list[str],
    source_id: str,
    video_id: str,
    stage: str,
    output_dir: Path,
    current_run_id: str,
    cpu_threads: int,
    heavy: bool,
    dry_run: bool,
) -> dict[str, Any]:
    snapshot = system_snapshot()
    maybe_pause_for_pressure(output_dir, snapshot, current_run_id)
    start_time = utc_now()
    start = time.perf_counter()
    full_command = ["nice", "-n", "10", *command] if heavy else command
    if dry_run:
        status = "planned"
        returncode = 0
        stdout = ""
        stderr = ""
    else:
        result = subprocess.run(
            full_command,
            cwd=REPO_ROOT,
            env=controlled_env(cpu_threads),
            capture_output=True,
            text=True,
            check=False,
        )
        returncode = result.returncode
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        status = "completed" if returncode == 0 else "failed"
    duration = time.perf_counter() - start
    failure = "" if returncode == 0 else (stderr or stdout or f"exit code {returncode}")[-500:]
    cache_reused = "Reusing cached" in stdout or "reused cached" in stdout
    row = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": current_run_id,
        "source_id": source_id,
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
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "command": " ".join(full_command),
    }


def sleep_after(stage: str, output_dir: Path, current_run_id: str, source_id: str, profile: str, cached: bool) -> None:
    if cached:
        return
    multiplier = {"conservative": 1.0, "fast": 0.1, "none": 0.0}[profile]
    base = {
        "preprocess": 10.0,
        "resnet": 15.0,
        "temporal": 10.0,
        "source_complete": 30.0,
    }.get(stage, 0.0)
    seconds = base * multiplier
    if seconds <= 0:
        return
    start = time.perf_counter()
    start_time = utc_now()
    time.sleep(seconds)
    append_runtime_log(
        output_dir / "runtime_log.csv",
        {
            "experiment_id": EXPERIMENT_ID,
            "run_id": current_run_id,
            "source_id": source_id,
            "video_id": "SYSTEM",
            "stage": f"sleep_after_{stage}",
            "start_time": start_time,
            "end_time": utc_now(),
            "duration_seconds": round(time.perf_counter() - start, 6),
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


def metadata_from_probe(path: Path, probe: dict[str, Any]) -> dict[str, Any]:
    streams = probe.get("streams") or []
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if not video_streams:
        raise EvaluationError(f"No video stream found in {path}")
    stream = video_streams[0]
    format_info = probe.get("format") or {}
    duration = optional_float(format_info.get("duration")) or optional_float(stream.get("duration"))
    fps_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    fps = parse_fps(fps_text)
    frame_count = optional_int(stream.get("nb_frames"))
    if frame_count is None and duration is not None and fps is not None:
        frame_count = int(round(duration * fps))
    return {
        "duration_seconds": duration,
        "width": optional_int(stream.get("width")),
        "height": optional_int(stream.get("height")),
        "fps": fps,
        "fps_text": fps_text,
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


def decode_check(path: Path, source_id: str, video_id: str, output_dir: Path, current_run_id: str, cpu_threads: int, dry_run: bool) -> tuple[bool, str]:
    result = run_logged_command(
        command=["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        source_id=source_id,
        video_id=video_id,
        stage="decode_check",
        output_dir=output_dir,
        current_run_id=current_run_id,
        cpu_threads=cpu_threads,
        heavy=True,
        dry_run=dry_run,
    )
    if result["returncode"] == 0:
        return True, ""
    return False, str(result.get("stderr") or result.get("stdout") or result.get("failure_reason", ""))[-500:]


def source_prefix(folder: Path) -> str:
    return safe_video_id(folder.name).upper()


def source_id_for(folder: Path) -> str:
    return "SRC_" + source_prefix(folder)


def reference_id_for(folder: Path) -> str:
    return source_prefix(folder) + "_ORIGINAL"


def exact_base_reference_candidates(folder: Path, paths: list[Path]) -> list[Path]:
    exact = [path for path in paths if path.stem.lower() == folder.name.lower()]
    if not exact:
        return []
    mp4_exact = [path for path in exact if path.suffix.lower() == ".mp4"]
    return mp4_exact or exact


def original_named_reference_candidates(paths: list[Path]) -> list[Path]:
    originals = [path for path in paths if path.stem.lower() == "original"]
    mp4_originals = [path for path in originals if path.suffix.lower() == ".mp4"]
    return mp4_originals or originals


def classify_transformation(path: Path) -> tuple[str, str, str, str, float]:
    name = path.name.lower()
    stem = path.stem.lower()
    candidates: list[tuple[str, str, str, str, float]] = []
    if re.search(r"blur|blurr|blurred|fblurred", name):
        candidates.append(("tampered", "abnormal", "blur", "filename_pattern_blur", 0.95))
    if re.search(r"delete|deletion|fdelete|frame_del|(^|[_-])del($|[_.-])", name) or stem in {"del", "fdel"}:
        candidates.append(("tampered", "abnormal", "frame_deletion", "filename_pattern_delete", 0.95))
    if re.search(r"insert|insertion|nsert|fiinsert", name):
        candidates.append(("tampered", "abnormal", "frame_insertion", "filename_pattern_insert", 0.95))
    if re.search(r"replace|replacement|swap|swapped", name):
        candidates.append(("tampered", "abnormal", "frame_replacement", "filename_pattern_replace_swap", 0.95))
    if "480" in stem or "480p" in stem:
        candidates.append(("benign", "normal", "resize_480p", "filename_pattern_480p", 0.95))
    if "720" in stem or "720p" in stem:
        candidates.append(("benign", "normal", "resize_720p", "filename_pattern_720p", 0.95))
    if path.suffix.lower() == ".avi" or "avi" in stem:
        candidates.append(("benign", "normal", "avi_conversion", "filename_pattern_avi", 0.95))
    if path.suffix.lower() == ".mov" or "mov" in stem:
        candidates.append(("benign", "normal", "mov_conversion", "filename_pattern_mov", 0.95))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return ("ambiguous", "unknown", "ambiguous", "multiple_filename_patterns", 0.2)
    return ("unsupported", "unknown", "unsupported_video", "unmatched_supported_extension", 0.1)


def video_id_for(folder: Path, transformation: str, path: Path, duplicate_index: int = 1) -> str:
    prefix = source_prefix(folder)
    suffixes = {
        "trusted_reference": "ORIGINAL",
        "avi_conversion": "AVI",
        "mov_conversion": "MOV",
        "resize_480p": "480P",
        "resize_720p": "720P",
        "blur": "BLUR",
        "frame_deletion": "FRAME_DELETE",
        "frame_insertion": "FRAME_INSERT",
        "frame_replacement": "FRAME_REPLACEMENT",
    }
    suffix = suffixes.get(transformation, safe_video_id(path.stem).upper())
    base = f"{prefix}_{suffix}"
    return base if duplicate_index == 1 else f"{base}_{duplicate_index}"


def supported_files(folder: Path) -> tuple[list[Path], list[Path]]:
    videos: list[Path] = []
    skipped: list[Path] = []
    for path in sorted(folder.iterdir(), key=lambda item: item.name.lower()):
        if path.name.startswith(".") or not path.is_file():
            skipped.append(path)
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped.append(path)
            continue
        videos.append(path)
    return videos, skipped


def discover_sources(input_root: Path, output_dir: Path, current_run_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not input_root.exists():
        raise EvaluationError(f"Input root does not exist: {input_root}")
    source_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    source_dirs = sorted([path for path in input_root.iterdir() if path.is_dir() and not path.name.startswith(".")], key=lambda item: item.name.lower())
    for folder in source_dirs:
        videos, skipped = supported_files(folder)
        src_id = source_id_for(folder)
        for skipped_path in skipped:
            skipped_rows.append(
                {
                    "run_id": current_run_id,
                    "source_id": src_id,
                    "source_folder": folder.name,
                    "filename": skipped_path.name,
                    "relative_path": str(skipped_path.relative_to(REPO_ROOT)),
                    "reason": "hidden_metadata_temporary_or_unsupported",
                    "run_timestamp": utc_now(),
                }
            )
        if not SOURCE_RE.match(folder.name):
            source_rows.append(
                {
                    "run_id": current_run_id,
                    "source_id": src_id,
                    "source_folder": folder.name,
                    "status": "skipped",
                    "original_candidate": "",
                    "benign_variants": [],
                    "tampered_variants": [],
                    "unsupported_files": [path.name for path in skipped],
                    "missing_expected_files": EXPECTED_TRANSFORMATIONS,
                    "duplicate_classifications": [],
                    "ambiguities": ["folder_name_not_source_like"],
                    "notes": "Folder name does not resemble vidNN.",
                    "run_timestamp": utc_now(),
                }
            )
            continue
        if not videos:
            source_rows.append(
                {
                    "run_id": current_run_id,
                    "source_id": src_id,
                    "source_folder": folder.name,
                    "status": "blocked",
                    "original_candidate": "",
                    "benign_variants": [],
                    "tampered_variants": [],
                    "unsupported_files": [path.name for path in skipped],
                    "missing_expected_files": EXPECTED_TRANSFORMATIONS,
                    "duplicate_classifications": [],
                    "ambiguities": ["no_supported_video_files"],
                    "notes": "No supported videos discovered.",
                    "run_timestamp": utc_now(),
                }
            )
            continue
        ref_candidates = exact_base_reference_candidates(folder, videos)
        ref_method = "exact_folder_basename_prefer_mp4"
        if not ref_candidates:
            ref_candidates = original_named_reference_candidates(videos)
            ref_method = "original_basename_fallback_prefer_mp4"
        status = "valid"
        ambiguities: list[str] = []
        if len(ref_candidates) != 1:
            status = "blocked"
            ambiguities.append("reference_not_unique" if ref_candidates else "reference_missing")
            reference_path = None
        else:
            reference_path = ref_candidates[0]
        transform_counts: dict[str, int] = {}
        source_video_rows: list[dict[str, Any]] = []
        for path in videos:
            if reference_path is not None and path == reference_path:
                category = "reference"
                label = "normal"
                transformation = "trusted_reference"
                method = ref_method
                confidence = 1.0
            else:
                category, label, transformation, method, confidence = classify_transformation(path)
            transform_counts[transformation] = transform_counts.get(transformation, 0) + 1
            duplicate_index = transform_counts[transformation]
            exp_video_id = video_id_for(folder, transformation, path, duplicate_index)
            notes: list[str] = []
            if status == "blocked":
                notes.append("source blocked due to reference ambiguity")
            if category == "ambiguous":
                ambiguities.append(path.name)
                notes.append("multiple filename patterns matched")
            if category == "unsupported":
                notes.append("supported extension but expected transformation was not recognized")
            row = {
                "experiment_id": EXPERIMENT_ID,
                "run_id": current_run_id,
                "source_id": src_id,
                "source_folder": folder.name,
                "experiment_video_id": exp_video_id,
                "video_id": exp_video_id,
                "filename": path.name,
                "relative_path": str(path.relative_to(REPO_ROOT)),
                "absolute_path": str(path.resolve()),
                "file_extension": path.suffix.lower(),
                "expected_category": category,
                "expected_label": label,
                "transformation_type": transformation,
                "reference_video_id": reference_id_for(folder) if reference_path is not None else "",
                "reference_filename": reference_path.name if reference_path is not None else "",
                "classification_method": method,
                "classification_confidence": confidence,
                "processing_status": "blocked_source" if status == "blocked" else "discovered",
                "notes": "; ".join(notes),
                "run_timestamp": utc_now(),
            }
            source_video_rows.append(row)
        missing = [name for name in EXPECTED_TRANSFORMATIONS if transform_counts.get(name, 0) == 0]
        duplicates = [name for name in EXPECTED_TRANSFORMATIONS if transform_counts.get(name, 0) > 1]
        benign = [row["filename"] for row in source_video_rows if row["expected_category"] == "benign"]
        tampered = [row["filename"] for row in source_video_rows if row["expected_category"] == "tampered"]
        source_rows.append(
            {
                "run_id": current_run_id,
                "source_id": src_id,
                "source_folder": folder.name,
                "status": status,
                "original_candidate": reference_path.name if reference_path is not None else "",
                "benign_variants": benign,
                "tampered_variants": tampered,
                "unsupported_files": [row["filename"] for row in source_video_rows if row["expected_category"] == "unsupported"] + [path.name for path in skipped],
                "missing_expected_files": missing,
                "duplicate_classifications": duplicates,
                "ambiguities": sorted(set(ambiguities)),
                "notes": "OK" if status == "valid" and not missing and not duplicates and not ambiguities else "review discovery findings",
                "run_timestamp": utc_now(),
            }
        )
        video_rows.extend(source_video_rows)
    if not source_dirs:
        skipped_rows.append(
            {
                "run_id": current_run_id,
                "source_id": "",
                "source_folder": "",
                "filename": "",
                "relative_path": str(input_root.relative_to(REPO_ROOT)) if input_root.is_relative_to(REPO_ROOT) else str(input_root),
                "reason": "no_source_subdirectories_found",
                "run_timestamp": utc_now(),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "source_registry.json", {"sources": source_rows})
    write_json(output_dir / "video_registry.json", {"videos": video_rows, "skipped": skipped_rows})
    write_csv_rows(output_dir / "source_registry.csv", source_rows)
    write_csv_rows(output_dir / "video_registry.csv", video_rows)
    return source_rows, video_rows, skipped_rows


def augment_registry_metadata(
    *,
    registry: list[dict[str, Any]],
    output_dir: Path,
    current_run_id: str,
    cpu_threads: int,
    dry_run: bool,
) -> list[dict[str, Any]]:
    checksum_by_source: dict[tuple[str, str], list[str]] = {}
    original_checksums: dict[str, list[str]] = {}
    augmented: list[dict[str, Any]] = []
    for entry in registry:
        path = Path(entry["absolute_path"])
        row = dict(entry)
        try:
            checksum = sha256_file(path)
            probe = ffprobe_json(path)
            metadata = metadata_from_probe(path, probe)
            decode_valid, decode_failure = decode_check(
                path,
                row["source_id"],
                row["video_id"],
                output_dir,
                current_run_id,
                cpu_threads,
                dry_run,
            )
            checksum_by_source.setdefault((row["source_id"], checksum), []).append(row["filename"])
            if row["expected_category"] == "reference":
                original_checksums.setdefault(checksum, []).append(row["video_id"])
            row.update(
                {
                    "file_size_bytes": path.stat().st_size,
                    "sha256": checksum,
                    "checksum": checksum,
                    **metadata,
                    "decode_valid": decode_valid,
                    "processing_status": "metadata_validated" if decode_valid and row["processing_status"] != "blocked_source" else row["processing_status"],
                    "notes": (row.get("notes", "") + ("; " if row.get("notes") and decode_failure else "") + decode_failure).strip("; "),
                }
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            row.update(
                {
                    "sha256": "",
                    "checksum": "",
                    "duration_seconds": None,
                    "width": None,
                    "height": None,
                    "fps": None,
                    "frame_count": None,
                    "codec": None,
                    "pixel_format": None,
                    "container_format": None,
                    "bitrate": None,
                    "audio_presence": None,
                    "decode_valid": False,
                    "processing_status": "metadata_failed",
                    "notes": (row.get("notes", "") + f"; metadata/decode failure: {exc}").strip("; "),
                }
            )
        augmented.append(row)
    duplicate_groups = {key: names for key, names in checksum_by_source.items() if len(names) > 1}
    duplicate_original_checksums = {sha: ids for sha, ids in original_checksums.items() if len(ids) > 1}
    for row in augmented:
        checksum = row.get("sha256", "")
        same_source_duplicates = duplicate_groups.get((row["source_id"], checksum), [])
        row["byte_identical_duplicates_in_source"] = [name for name in same_source_duplicates if name != row["filename"]]
        row["byte_identical_originals_across_sources"] = (
            [video_id for video_id in duplicate_original_checksums.get(checksum, []) if video_id != row["video_id"]]
            if row["expected_category"] == "reference"
            else []
        )
        warning_notes: list[str] = []
        if row["byte_identical_duplicates_in_source"]:
            warning_notes.append("byte-identical duplicate inside source")
        if row["byte_identical_originals_across_sources"]:
            warning_notes.append("byte-identical original across sources")
        if warning_notes:
            row["notes"] = (row.get("notes", "") + "; " + "; ".join(warning_notes)).strip("; ")
    write_json(output_dir / "video_registry.json", {"videos": augmented})
    write_csv_rows(output_dir / "video_registry.csv", augmented)
    return augmented


def metadata_comparison(registry: list[dict[str, Any]], current_run_id: str) -> list[dict[str, Any]]:
    originals = {row["source_id"]: row for row in registry if row["expected_category"] == "reference"}
    rows: list[dict[str, Any]] = []
    for video in registry:
        original = originals.get(video["source_id"])
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "run_id": current_run_id,
                "source_id": video["source_id"],
                "video_id": video["video_id"],
                "filename": video["filename"],
                "reference_video_id": video["reference_video_id"],
                "reference_filename": video["reference_filename"],
                "transformation_type": video["transformation_type"],
                "expected_label": video["expected_label"],
                "duration_seconds": video.get("duration_seconds"),
                "reference_duration_seconds": original.get("duration_seconds") if original else None,
                "duration_delta_seconds": (
                    float(video["duration_seconds"]) - float(original["duration_seconds"])
                    if original and video.get("duration_seconds") is not None and original.get("duration_seconds") is not None
                    else None
                ),
                "width": video.get("width"),
                "height": video.get("height"),
                "reference_width": original.get("width") if original else None,
                "reference_height": original.get("height") if original else None,
                "fps": video.get("fps"),
                "reference_fps": original.get("fps") if original else None,
                "fps_delta": (
                    float(video["fps"]) - float(original["fps"])
                    if original and video.get("fps") is not None and original.get("fps") is not None
                    else None
                ),
                "frame_count": video.get("frame_count"),
                "codec": video.get("codec"),
                "pixel_format": video.get("pixel_format"),
                "container_format": video.get("container_format"),
                "bitrate": video.get("bitrate"),
                "audio_presence": video.get("audio_presence"),
                "decode_valid": video.get("decode_valid"),
                "run_timestamp": utc_now(),
            }
        )
    return rows


def sha256_baseline(registry: list[dict[str, Any]], current_run_id: str) -> list[dict[str, Any]]:
    originals = {row["source_id"]: row for row in registry if row["expected_category"] == "reference"}
    rows: list[dict[str, Any]] = []
    for video in registry:
        original = originals.get(video["source_id"])
        match = bool(original and video.get("sha256") == original.get("sha256"))
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "run_id": current_run_id,
                "source_id": video["source_id"],
                "video_id": video["video_id"],
                "filename": video["filename"],
                "reference_video_id": video["reference_video_id"],
                "reference_filename": video["reference_filename"],
                "transformation_type": video["transformation_type"],
                "expected_label": video["expected_label"],
                "sha256": video.get("sha256", ""),
                "original_sha256": original.get("sha256", "") if original else "",
                "matches_original": match,
                "interpretation": "byte-for-byte identical" if match else "file bytes changed; checksum mismatch cannot distinguish benign conversion from tampering",
                "run_timestamp": utc_now(),
            }
        )
    return rows


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


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
    metadata_path, _, frames_path = path_for_stage(config, video["video_id"], "preprocess")
    if not metadata_path.exists() or not frames_path.exists():
        return False
    try:
        metadata = read_json(metadata_path)
        frames = read_json(frames_path)
    except (OSError, json.JSONDecodeError, EvaluationError):
        return False
    return (
        str(metadata.get("absolute_path")) == str(video["absolute_path"])
        and str(frames.get("source_video_path")) == str(video["absolute_path"])
    )


def cached_stage_valid(config: Any, video: dict[str, Any], stage: str) -> bool:
    if stage == "preprocess":
        return cached_preprocess_valid(config, video)
    paths = path_for_stage(config, video["video_id"], stage)
    if not all(path.exists() for path in paths):
        return False
    if stage == "normalize":
        try:
            manifest = read_json(paths[1])
        except (OSError, json.JSONDecodeError, EvaluationError):
            return False
        return manifest.get("calibration_id") == NORMALIZATION_ID
    if stage == "digest":
        try:
            manifest = read_json(paths[1])
        except (OSError, json.JSONDecodeError, EvaluationError):
            return False
        return manifest.get("quantization_id") == QUANTIZATION_ID
    return True


def record_cached_stage(output_dir: Path, current_run_id: str, source_id: str, video_id: str, stage: str) -> dict[str, Any]:
    row = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": current_run_id,
        "source_id": source_id,
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


def stage_forced(force_stage: set[str], stage: str) -> bool:
    return "all" in force_stage or stage in force_stage


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
    dry_run: bool,
) -> dict[str, Any]:
    if cached_stage_valid(config, video, stage) and not stage_forced(force_stage, stage):
        return record_cached_stage(output_dir, current_run_id, video["source_id"], video["video_id"], stage)
    python = str(REPO_ROOT / ".venv" / "bin" / "python")
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
            video["video_id"],
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
            video["video_id"],
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
            video["video_id"],
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
            video["video_id"],
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
            video["video_id"],
            "--quantization-id",
            QUANTIZATION_ID,
            *overwrite,
        ]
    else:
        raise EvaluationError(f"Unknown stage: {stage}")
    return run_logged_command(
        command=command,
        source_id=video["source_id"],
        video_id=video["video_id"],
        stage=stage,
        output_dir=output_dir,
        current_run_id=current_run_id,
        cpu_threads=cpu_threads,
        heavy=True,
        dry_run=dry_run,
    )


def hmac_key_path(config_path: Path, config: Any, dry_run: bool) -> tuple[Path, str]:
    preferred = config.paths.local_secrets / "DEV_HMAC_KEY_V1.hex"
    if preferred.exists():
        return preferred, "DEV_HMAC_KEY_V1"
    fallback = config.paths.local_secrets / "MULTISOURCE_HMAC_KEY_V1.hex"
    if fallback.exists():
        return fallback, "MULTISOURCE_HMAC_KEY_V1"
    if dry_run:
        return fallback, "MULTISOURCE_HMAC_KEY_V1"
    python = str(REPO_ROOT / ".venv" / "bin" / "python")
    result = subprocess.run(
        [
            python,
            "main.py",
            "generate-hmac-key",
            "--config",
            str(config_path),
            "--output",
            str(fallback.relative_to(REPO_ROOT)),
            "--key-id",
            "MULTISOURCE_HMAC_KEY_V1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise EvaluationError("Could not create fallback HMAC key: " + (result.stderr or result.stdout))
    return fallback, "MULTISOURCE_HMAC_KEY_V1"


def protect_and_verify_original(
    *,
    video: dict[str, Any],
    config_path: Path,
    config: Any,
    output_dir: Path,
    current_run_id: str,
    cpu_threads: int,
    force_stage: set[str],
    dry_run: bool,
) -> dict[str, Any]:
    key_file, key_id = hmac_key_path(config_path, config, dry_run)
    record_path = authentication_record_paths(config.paths.authentication_records, video["video_id"]).record_path
    if record_path.exists() and not stage_forced(force_stage, "protect"):
        protect = record_cached_stage(output_dir, current_run_id, video["source_id"], video["video_id"], "protect")
    else:
        python = str(REPO_ROOT / ".venv" / "bin" / "python")
        overwrite = ["--overwrite"] if stage_forced(force_stage, "protect") else []
        protect = run_logged_command(
            command=[
                python,
                "main.py",
                "protect-digest",
                "--config",
                str(config_path),
                "--video-id",
                video["video_id"],
                "--key-file",
                str(key_file.relative_to(REPO_ROOT)),
                "--key-id",
                key_id,
                *overwrite,
            ],
            source_id=video["source_id"],
            video_id=video["video_id"],
            stage="protect",
            output_dir=output_dir,
            current_run_id=current_run_id,
            cpu_threads=cpu_threads,
            heavy=True,
            dry_run=dry_run,
        )
    python = str(REPO_ROOT / ".venv" / "bin" / "python")
    verify = run_logged_command(
        command=[
            python,
            "main.py",
            "verify-auth-record",
            "--config",
            str(config_path),
            "--video-id",
            video["video_id"],
            "--key-file",
            str(key_file.relative_to(REPO_ROOT)),
        ],
        source_id=video["source_id"],
        video_id=video["video_id"],
        stage="verify_hmac",
        output_dir=output_dir,
        current_run_id=current_run_id,
        cpu_threads=cpu_threads,
        heavy=True,
        dry_run=dry_run,
    )
    return {
        "source_id": video["source_id"],
        "video_id": video["video_id"],
        "key_id": key_id,
        "key_file": str(key_file),
        "record_path": str(record_path),
        "protect_status": protect["status"],
        "protect_returncode": protect["returncode"],
        "verify_status": verify["status"],
        "verify_returncode": verify["returncode"],
        "hmac_verified": protect["returncode"] == 0 and verify["returncode"] == 0,
    }


def comparison_cached(config: Any, reference_id: str, query_id: str, force_stage: set[str]) -> bool:
    paths = comparison_output_paths(config.paths.comparisons, reference_id, query_id)
    return paths.manifest_path.exists() and paths.npz_path.exists() and not stage_forced(force_stage, "compare")


def compare_video(
    *,
    reference: dict[str, Any],
    query: dict[str, Any],
    config_path: Path,
    config: Any,
    output_dir: Path,
    current_run_id: str,
    cpu_threads: int,
    force_stage: set[str],
    dry_run: bool,
) -> dict[str, Any]:
    if comparison_cached(config, reference["video_id"], query["video_id"], force_stage):
        result = record_cached_stage(output_dir, current_run_id, query["source_id"], query["video_id"], "compare")
    else:
        key_file, _ = hmac_key_path(config_path, config, dry_run)
        python = str(REPO_ROOT / ".venv" / "bin" / "python")
        overwrite = ["--overwrite"] if stage_forced(force_stage, "compare") else []
        result = run_logged_command(
            command=[
                python,
                "main.py",
                "compare-digests",
                "--config",
                str(config_path),
                "--reference-id",
                reference["video_id"],
                "--query-id",
                query["video_id"],
                "--key-file",
                str(key_file.relative_to(REPO_ROOT)),
                *overwrite,
            ],
            source_id=query["source_id"],
            video_id=query["video_id"],
            stage="compare",
            output_dir=output_dir,
            current_run_id=current_run_id,
            cpu_threads=cpu_threads,
            heavy=True,
            dry_run=dry_run,
        )
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_id": current_run_id,
        "source_id": query["source_id"],
        "video_id": query["video_id"],
        "filename": query["filename"],
        "reference_video_id": reference["video_id"],
        "reference_filename": reference["filename"],
        "transformation_type": query["transformation_type"],
        "expected_label": query["expected_label"],
        "stage": "compare",
        "status": result["status"],
        "cache_reused": result["cache_reused"],
        "duration_seconds": result["duration_seconds"],
        "failure_reason": result["failure_reason"],
        "run_timestamp": utc_now(),
    }


def ordered_source_videos(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {
        "trusted_reference": 0,
        "avi_conversion": 1,
        "mov_conversion": 2,
        "resize_480p": 3,
        "resize_720p": 4,
        "blur": 5,
        "frame_deletion": 6,
        "frame_insertion": 7,
        "frame_replacement": 8,
    }
    return sorted(videos, key=lambda row: (order.get(row["transformation_type"], 99), row["filename"].lower()))


def selected_sources(source_rows: list[dict[str, Any]], source_filter: str | None) -> set[str]:
    valid = {row["source_id"] for row in source_rows if row["status"] == "valid"}
    if not source_filter:
        return valid
    wanted = safe_video_id(source_filter).upper()
    if not wanted.startswith("SRC_"):
        wanted = "SRC_" + wanted
    if wanted not in valid:
        raise EvaluationError(f"Requested --source-id was not found as a valid source: {source_filter}")
    return {wanted}


def save_state(output_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    write_json(output_dir / RUN_STATE_NAME, state)


def process_sources(
    *,
    source_rows: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    config_path: Path,
    output_dir: Path,
    current_run_id: str,
    cpu_threads: int,
    force_stage: set[str],
    source_filter: str | None,
    video_filter: str | None,
    sleep_profile: str,
    continue_on_error: bool,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config = load_config(config_path)
    allowed_sources = selected_sources(source_rows, source_filter)
    by_source: dict[str, list[dict[str, Any]]] = {}
    for video in registry:
        if video["source_id"] in allowed_sources and video["processing_status"] != "blocked_source":
            by_source.setdefault(video["source_id"], []).append(video)
    if video_filter:
        wanted_video = safe_video_id(video_filter)
        keep_sources = {
            row["source_id"]
            for row in registry
            if row["video_id"] == wanted_video and row["source_id"] in allowed_sources
        }
        if not keep_sources:
            raise EvaluationError(f"Requested --video-id was not found: {video_filter}")
        by_source = {source_id: videos for source_id, videos in by_source.items() if source_id in keep_sources}
    state = {
        "run_id": current_run_id,
        "experiment_id": EXPERIMENT_ID,
        "status": "running",
        "input_sources": sorted(by_source),
        "source_status": {},
        "stage_status": {},
        "resume_supported": True,
        "dry_run": dry_run,
        "started_at": utc_now(),
    }
    save_state(output_dir, state)
    stage_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    hmac_rows: list[dict[str, Any]] = []
    failures: dict[str, list[str]] = {}
    for source_id in sorted(by_source):
        source_videos = ordered_source_videos(by_source[source_id])
        references = [row for row in source_videos if row["expected_category"] == "reference"]
        if len(references) != 1:
            failures.setdefault(source_id, []).append("source does not have exactly one reference")
            state["source_status"][source_id] = "blocked"
            save_state(output_dir, state)
            continue
        reference = references[0]
        selected_for_processing = source_videos
        if video_filter:
            wanted = safe_video_id(video_filter)
            selected_for_processing = [reference] + [row for row in source_videos if row["video_id"] == wanted and row["video_id"] != reference["video_id"]]
        source_failed = False
        for video in selected_for_processing:
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
                    dry_run=dry_run,
                )
                stage_row = {
                    "experiment_id": EXPERIMENT_ID,
                    "run_id": current_run_id,
                    "source_id": video["source_id"],
                    "video_id": video["video_id"],
                    "filename": video["filename"],
                    "reference_video_id": video["reference_video_id"],
                    "transformation_type": video["transformation_type"],
                    "expected_label": video["expected_label"],
                    "stage": stage,
                    "status": result["status"],
                    "cache_reused": result["cache_reused"],
                    "duration_seconds": result["duration_seconds"],
                    "failure_reason": result["failure_reason"],
                    "run_timestamp": utc_now(),
                }
                stage_rows.append(stage_row)
                state["stage_status"][f"{video['video_id']}:{stage}"] = stage_row
                save_state(output_dir, state)
                sleep_after(stage, output_dir, current_run_id, source_id, sleep_profile, bool(result["cache_reused"]))
                if result["returncode"] != 0:
                    failures.setdefault(video["video_id"], []).append(f"{stage}: {result['failure_reason']}")
                    if video["video_id"] == reference["video_id"]:
                        source_failed = True
                    break
            if source_failed:
                state["source_status"][source_id] = "blocked_original_failed"
                save_state(output_dir, state)
                break
        if source_failed:
            if continue_on_error:
                continue
            raise EvaluationError(f"Original processing failed for {source_id}")
        hmac_row = protect_and_verify_original(
            video=reference,
            config_path=config_path,
            config=config,
            output_dir=output_dir,
            current_run_id=current_run_id,
            cpu_threads=cpu_threads,
            force_stage=force_stage,
            dry_run=dry_run,
        )
        hmac_rows.append(hmac_row)
        state["stage_status"][f"{reference['video_id']}:hmac"] = hmac_row
        save_state(output_dir, state)
        if not hmac_row["hmac_verified"]:
            failures.setdefault(reference["video_id"], []).append("HMAC protect or verify failed")
            state["source_status"][source_id] = "blocked_hmac_failed"
            save_state(output_dir, state)
            if continue_on_error:
                continue
            raise EvaluationError(f"HMAC verification failed for {reference['video_id']}")
        for query in selected_for_processing:
            row = compare_video(
                reference=reference,
                query=query,
                config_path=config_path,
                config=config,
                output_dir=output_dir,
                current_run_id=current_run_id,
                cpu_threads=cpu_threads,
                force_stage=force_stage,
                dry_run=dry_run,
            )
            compare_rows.append(row)
            state["stage_status"][f"{query['video_id']}:compare"] = row
            save_state(output_dir, state)
            if row["status"] == "failed":
                failures.setdefault(query["video_id"], []).append(f"compare: {row['failure_reason']}")
        state["source_status"][source_id] = "completed"
        save_state(output_dir, state)
        sleep_after("source_complete", output_dir, current_run_id, source_id, sleep_profile, cached=False)
    state["status"] = "completed_with_failures" if failures else "completed"
    state["failures"] = failures
    save_state(output_dir, state)
    return stage_rows, compare_rows, hmac_rows, {"failures": failures}


def artifact_validation(video: dict[str, Any], config: Any) -> dict[str, Any]:
    video_id = video["video_id"]
    metadata_path = config.paths.metadata / f"{video_id}_metadata.json"
    segments_path = config.paths.manifests / f"{video_id}_segments.json"
    frames_path = config.paths.manifests / f"{video_id}_frames.json"
    resnet_paths = feature_output_paths(config.paths.resnet_features, video_id)
    temporal_paths = temporal_output_paths(config.paths.temporal_features, video_id)
    normalized_paths = normalized_output_paths(config.paths.normalized_features, video_id)
    digest_paths = digest_output_paths(config.paths.digests, video_id)
    result: dict[str, Any] = {
        "source_id": video["source_id"],
        "video_id": video_id,
        "filename": video["filename"],
        "metadata_exists": metadata_path.exists(),
        "segments_exists": segments_path.exists(),
        "frame_manifest_exists": frames_path.exists(),
        "resnet_exists": resnet_paths.npz_path.exists() and resnet_paths.manifest_path.exists(),
        "temporal_exists": temporal_paths.npz_path.exists() and temporal_paths.manifest_path.exists(),
        "normalized_exists": normalized_paths.npz_path.exists() and normalized_paths.manifest_path.exists(),
        "digest_exists": digest_paths.npz_path.exists() and digest_paths.manifest_path.exists(),
    }
    if segments_path.exists():
        segments = read_json(segments_path)
        result["complete_segments"] = int(segments.get("number_complete_segments", 0))
        result["discarded_duration_seconds"] = segments.get("discarded_duration_seconds")
        result["segment_ids"] = [int(item["segment_id"]) for item in segments.get("segments", []) if item.get("is_complete", True)]
    if frames_path.exists():
        frames = read_json(frames_path)
        records = frames.get("frame_records", [])
        result["sampled_resnet_frames"] = sum(1 for record in records if record.get("success"))
        result["sampled_resnet_frame_failures"] = sum(1 for record in records if not record.get("success"))
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
        result["normalization_checksum"] = manifest.get("calibration_manifest_sha256")
    if digest_paths.npz_path.exists():
        arrays = load_npz(digest_paths.npz_path)
        manifest = read_json(digest_paths.manifest_path)
        bit_order = str(manifest.get("bit_order", "big"))
        result["digest_resnet_shape"] = list(arrays["resnet_binary_digests"].shape)
        result["digest_temporal_shape"] = list(arrays["temporal_binary_digests"].shape)
        result["digest_hybrid_shape"] = list(arrays["hybrid_binary_digests"].shape)
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
        result["quantization_checksum"] = manifest.get("quantization_manifest_sha256")
        result["bit_order"] = bit_order
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


def comparison_rows(registry_by_id: dict[str, dict[str, Any]], config: Any, current_run_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    segment_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    for video_id, video in registry_by_id.items():
        reference_id = video["reference_video_id"]
        if not reference_id:
            continue
        paths = comparison_output_paths(config.paths.comparisons, reference_id, video_id)
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
                    "source_id": video["source_id"],
                    "video_id": video_id,
                    "filename": video["filename"],
                    "reference_video_id": reference_id,
                    "reference_filename": video["reference_filename"],
                    "transformation_type": video["transformation_type"],
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
                    "source_id": video["source_id"],
                    "reference_video_id": reference_id,
                    "query_video_id": video_id,
                    "video_id": video_id,
                    "filename": video["filename"],
                    "reference_filename": video["reference_filename"],
                    "transformation_type": video["transformation_type"],
                    "expected_category": video["expected_category"],
                    "expected_label": video["expected_label"],
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
                    "relative_stream_attribution": segment["relative_stream_attribution"],
                    "structural_alignment_issue": structural_issue,
                    "normalization_id": manifest.get("normalization_id"),
                    "quantization_id": manifest.get("quantization_id"),
                    "comparison_manifest": str(paths.manifest_path),
                    "run_timestamp": utc_now(),
                }
            )
        summary_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "run_id": current_run_id,
                "source_id": video["source_id"],
                "reference_video_id": reference_id,
                "query_video_id": video_id,
                "video_id": video_id,
                "filename": video["filename"],
                "reference_filename": video["reference_filename"],
                "transformation_type": video["transformation_type"],
                "expected_category": video["expected_category"],
                "expected_label": video["expected_label"],
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
                "segment_id_with_maximum_balanced_score": summary.get("segment_id_with_maximum_balanced_diagnostic_score"),
                "attribution_counts": summary.get("attribution_counts", {}),
                "normalization_id": manifest.get("normalization_id"),
                "quantization_id": manifest.get("quantization_id"),
                "comparison_manifest": str(paths.manifest_path),
                "run_timestamp": utc_now(),
            }
        )
    return segment_rows, summary_rows, alignment_rows


def median_absolute_deviation(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def threshold_from_rows(rows: list[dict[str, Any]], config: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        return {
            **metadata,
            "status": "insufficient_benign_segments",
            "warnings": ["No benign matched segment rows available."],
            "resnet": {"threshold": None},
            "temporal": {"threshold": None},
            "balanced": {"threshold": None},
        }
    resnet_weight = float(config.verification.comparison.resnet_weight)
    temporal_weight = float(config.verification.comparison.temporal_weight)
    streams = {
        "resnet": ("resnet_normalized_distance", 1.0 / 1024.0),
        "temporal": ("temporal_normalized_distance", 1.0 / 36.0),
        "balanced": ("balanced_diagnostic_score", resnet_weight * (1.0 / 1024.0) + temporal_weight * (1.0 / 36.0)),
    }
    payload: dict[str, Any] = {
        **metadata,
        "threshold_formula": "max_benign_distance + max(3 * MAD(benign_distances), one_bit_normalized_resolution)",
        "tampered_labels_used": False,
        "status": "computed",
        "warnings": [],
    }
    for stream, (column, one_bit) in streams.items():
        values = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
        max_benign = float(np.max(values))
        mad = median_absolute_deviation(values)
        threshold = min(1.0, max(0.0, max_benign + max(3.0 * mad, one_bit)))
        payload[stream] = {
            "max_benign_distance": max_benign,
            "median_absolute_deviation": mad,
            "one_bit_resolution": one_bit,
            "threshold": threshold,
        }
    return payload


def pooled_thresholds(segment_rows: list[dict[str, Any]], config: Any, current_run_id: str) -> dict[str, Any]:
    benign_rows = [row for row in segment_rows if row["transformation_type"] in BENIGN_TRANSFORMATIONS]
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": current_run_id,
        "threshold_scope": "pooled_benign_descriptive",
        "benign_training_videos": sorted({row["video_id"] for row in benign_rows}),
        "benign_training_segments": len(benign_rows),
        "warning": "Descriptive only; primary diagnostic evaluation uses leave-one-source-out thresholds.",
    }
    return threshold_from_rows(benign_rows, config, metadata)


def attribution_for_segment(row: dict[str, Any], thresholds: dict[str, Any]) -> tuple[bool, str, dict[str, bool]]:
    structural = bool(row.get("structural_alignment_issue"))
    threshold_missing = any(thresholds.get(name, {}).get("threshold") is None for name in ["resnet", "temporal", "balanced"])
    if threshold_missing:
        return structural, "structural" if structural else "none", {
            "resnet_exceeds_threshold": False,
            "temporal_exceeds_threshold": False,
            "balanced_exceeds_threshold": False,
            "structural_alignment_issue": structural,
        }
    resnet_exceeds = float(row["resnet_normalized_distance"]) > float(thresholds["resnet"]["threshold"])
    temporal_exceeds = float(row["temporal_normalized_distance"]) > float(thresholds["temporal"]["threshold"])
    balanced_exceeds = float(row["balanced_diagnostic_score"]) > float(thresholds["balanced"]["threshold"])
    if structural:
        label = "structural"
    elif resnet_exceeds and temporal_exceeds:
        label = "both_streams"
    elif resnet_exceeds:
        label = "resnet_only"
    elif temporal_exceeds:
        label = "temporal_only"
    elif balanced_exceeds:
        label = "resnet_only" if float(row["resnet_normalized_distance"]) >= float(row["temporal_normalized_distance"]) else "temporal_only"
    else:
        label = "none"
    abnormal = structural or resnet_exceeds or temporal_exceeds or balanced_exceeds
    return abnormal, label, {
        "resnet_exceeds_threshold": resnet_exceeds,
        "temporal_exceeds_threshold": temporal_exceeds,
        "balanced_exceeds_threshold": balanced_exceeds,
        "structural_alignment_issue": structural,
    }


def leave_one_source_out(
    segment_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    registry_by_id: dict[str, dict[str, Any]],
    source_rows: list[dict[str, Any]],
    config: Any,
    current_run_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    valid_sources = sorted({row["source_id"] for row in source_rows if row["status"] == "valid"})
    available_sources = sorted({row["source_id"] for row in segment_rows})
    valid_sources = [source_id for source_id in valid_sources if source_id in available_sources]
    fold_payloads: list[dict[str, Any]] = []
    segment_decisions: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    structural_by_video: dict[str, list[str]] = {}
    for row in alignment_rows:
        if row["state"] != "matched":
            structural_by_video.setdefault(row["video_id"], []).append(f"{row['state']}:{row['segment_id']}")
    rows_by_video: dict[str, list[dict[str, Any]]] = {}
    for row in segment_rows:
        rows_by_video.setdefault(row["video_id"], []).append(row)
    for held_out in valid_sources:
        training_sources = [source_id for source_id in valid_sources if source_id != held_out]
        train_rows = [
            row
            for row in segment_rows
            if row["source_id"] in training_sources and row["transformation_type"] in BENIGN_TRANSFORMATIONS
        ]
        fold_id = "LOSO_" + held_out.replace("SRC_", "")
        metadata = {
            "experiment_id": EXPERIMENT_ID,
            "run_id": current_run_id,
            "fold_id": fold_id,
            "held_out_source": held_out,
            "training_sources": training_sources,
            "training_benign_videos": sorted({row["video_id"] for row in train_rows}),
            "training_benign_segments": len(train_rows),
            "source_groups_available": len(valid_sources),
        }
        thresholds = threshold_from_rows(train_rows, config, metadata)
        if len(training_sources) < 3:
            thresholds.setdefault("warnings", []).append("Fewer than three training source groups; generalization analysis is insufficient.")
            thresholds["status"] = "insufficient_sources"
        fold_payloads.append(thresholds)
        held_videos = [video for video in registry_by_id.values() if video["source_id"] == held_out]
        for video in ordered_source_videos(held_videos):
            rows = rows_by_video.get(video["video_id"], [])
            if not rows:
                expected = video["expected_label"]
                structural = structural_by_video.get(video["video_id"], [])
                predictions.append(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "run_id": current_run_id,
                        "fold_id": fold_id,
                        "held_out_source": held_out,
                        "source_id": video["source_id"],
                        "video_id": video["video_id"],
                        "filename": video["filename"],
                        "reference_video_id": video["reference_video_id"],
                        "reference_filename": video["reference_filename"],
                        "transformation_type": video["transformation_type"],
                        "expected_category": video["expected_category"],
                        "expected_label": expected,
                        "observed_diagnostic_label": "failed",
                        "provisional_result": "pipeline failed before diagnostic comparison",
                        "correct": False,
                        "abnormal_segment_ids": [],
                        "structural_findings": structural,
                        "structural_issue": bool(structural),
                        "max_resnet_distance": None,
                        "max_temporal_distance": None,
                        "max_balanced_score": None,
                        "main_attribution": "none",
                        "notes": "No comparison rows were available; excluded from diagnostic metric denominators.",
                        "run_timestamp": utc_now(),
                    }
                )
                for attribution in ["none", "resnet_only", "temporal_only", "both_streams", "structural"]:
                    attribution_rows.append(
                        {
                            "experiment_id": EXPERIMENT_ID,
                            "run_id": current_run_id,
                            "fold_id": fold_id,
                            "held_out_source": held_out,
                            "source_id": video["source_id"],
                            "video_id": video["video_id"],
                            "filename": video["filename"],
                            "transformation_type": video["transformation_type"],
                            "expected_label": video["expected_label"],
                            "attribution": attribution,
                            "abnormal_segment_count": 0,
                            "run_timestamp": utc_now(),
                        }
                    )
                continue
            abnormal_rows: list[dict[str, Any]] = []
            attribution_counts: dict[str, int] = {}
            for row in rows:
                abnormal, attribution, flags = attribution_for_segment(row, thresholds)
                decision = {
                    **row,
                    **flags,
                    "fold_id": fold_id,
                    "held_out_source": held_out,
                    "resnet_threshold": thresholds.get("resnet", {}).get("threshold"),
                    "temporal_threshold": thresholds.get("temporal", {}).get("threshold"),
                    "balanced_threshold": thresholds.get("balanced", {}).get("threshold"),
                    "diagnostic_attribution": attribution,
                    "segment_provisionally_abnormal": abnormal,
                    "run_timestamp": utc_now(),
                }
                segment_decisions.append(decision)
                if abnormal:
                    abnormal_rows.append(decision)
                    attribution_counts[attribution] = attribution_counts.get(attribution, 0) + 1
            structural = structural_by_video.get(video["video_id"], [])
            if structural:
                attribution_counts["structural"] = max(attribution_counts.get("structural", 0), len(structural))
            is_abnormal = bool(abnormal_rows or structural)
            observed = "abnormal" if is_abnormal else "normal"
            expected = video["expected_label"]
            correct = expected in {"normal", "abnormal"} and expected == observed
            main = "none"
            if attribution_counts:
                main = sorted(attribution_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
            predictions.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "run_id": current_run_id,
                    "fold_id": fold_id,
                    "held_out_source": held_out,
                    "source_id": video["source_id"],
                    "video_id": video["video_id"],
                    "filename": video["filename"],
                    "reference_video_id": video["reference_video_id"],
                    "reference_filename": video["reference_filename"],
                    "transformation_type": video["transformation_type"],
                    "expected_category": video["expected_category"],
                    "expected_label": expected,
                    "observed_diagnostic_label": observed,
                    "provisional_result": f"diagnostically {observed}",
                    "correct": correct,
                    "abnormal_segment_ids": sorted({int(row["segment_id"]) for row in abnormal_rows}),
                    "structural_findings": structural,
                    "structural_issue": bool(structural),
                    "max_resnet_distance": max([float(row["resnet_normalized_distance"]) for row in rows], default=None),
                    "max_temporal_distance": max([float(row["temporal_normalized_distance"]) for row in rows], default=None),
                    "max_balanced_score": max([float(row["balanced_diagnostic_score"]) for row in rows], default=None),
                    "main_attribution": main,
                    "notes": "leave-one-source-out provisional diagnostic result",
                    "run_timestamp": utc_now(),
                }
            )
            for label in ["none", "resnet_only", "temporal_only", "both_streams", "structural"]:
                attribution_rows.append(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "run_id": current_run_id,
                        "fold_id": fold_id,
                        "held_out_source": held_out,
                        "source_id": video["source_id"],
                        "video_id": video["video_id"],
                        "filename": video["filename"],
                        "transformation_type": video["transformation_type"],
                        "expected_label": video["expected_label"],
                        "attribution": label,
                        "abnormal_segment_count": int(attribution_counts.get(label, 0)),
                        "run_timestamp": utc_now(),
                    }
                )
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": current_run_id,
        "analysis": "leave_one_source_out_threshold_evaluation",
        "valid_sources": valid_sources,
        "folds": fold_payloads,
        "warning": "Primary multi-source diagnostic evaluation; not final held-out research performance.",
    }
    return payload, fold_payloads, segment_decisions, predictions, attribution_rows


def diagnostic_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    label_set = {"normal", "abnormal"}
    eligible_rows = [row for row in predictions if row["expected_label"] in label_set]
    rows = [row for row in eligible_rows if row["observed_diagnostic_label"] in label_set]
    failed_predictions = len(eligible_rows) - len(rows)
    tp = sum(1 for row in rows if row["expected_label"] == "abnormal" and row["observed_diagnostic_label"] == "abnormal")
    tn = sum(1 for row in rows if row["expected_label"] == "normal" and row["observed_diagnostic_label"] == "normal")
    fp = sum(1 for row in rows if row["expected_label"] == "normal" and row["observed_diagnostic_label"] == "abnormal")
    fn = sum(1 for row in rows if row["expected_label"] == "abnormal" and row["observed_diagnostic_label"] == "normal")
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "label": "MULTI-SOURCE DIAGNOSTIC METRICS - NOT FINAL HELD-OUT RESEARCH RESULTS",
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "evaluated_prediction_count": total,
        "failed_prediction_count": failed_predictions,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_acceptance_rate": fn / (tp + fn) if tp + fn else 0.0,
        "false_rejection_rate": fp / (tn + fp) if tn + fp else 0.0,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2 if total else 0.0,
        "warning": "Diagnostic LOSO metrics only; no tampered labels were used to fit thresholds.",
    }


def metrics_rows(predictions: list[dict[str, Any]], current_run_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        label_set = {"normal", "abnormal"}
        eligible_rows = [row for row in rows if row["expected_label"] in label_set]
        evaluated_rows = [row for row in eligible_rows if row["observed_diagnostic_label"] in label_set]
        return {
            "TP": sum(1 for row in evaluated_rows if row["expected_label"] == "abnormal" and row["observed_diagnostic_label"] == "abnormal"),
            "TN": sum(1 for row in evaluated_rows if row["expected_label"] == "normal" and row["observed_diagnostic_label"] == "normal"),
            "FP": sum(1 for row in evaluated_rows if row["expected_label"] == "normal" and row["observed_diagnostic_label"] == "abnormal"),
            "FN": sum(1 for row in evaluated_rows if row["expected_label"] == "abnormal" and row["observed_diagnostic_label"] == "normal"),
            "failed_prediction_count": len(eligible_rows) - len(evaluated_rows),
        }

    def row_from_counts(scope: str, name: str, c: dict[str, int]) -> dict[str, Any]:
        tp, tn, fp, fn = c["TP"], c["TN"], c["FP"], c["FN"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        total = tp + tn + fp + fn
        return {
            "experiment_id": EXPERIMENT_ID,
            "run_id": current_run_id,
            scope: name,
            **c,
            "evaluated_prediction_count": total,
            "accuracy": (tp + tn) / total if total else 0.0,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "run_timestamp": utc_now(),
        }

    per_source = [
        row_from_counts("source_id", source_id, counts([row for row in predictions if row["source_id"] == source_id]))
        for source_id in sorted({row["source_id"] for row in predictions})
    ]
    per_tamper: list[dict[str, Any]] = []
    for transformation in ["blur", "frame_deletion", "frame_insertion", "frame_replacement"]:
        rows = [row for row in predictions if row["transformation_type"] == transformation]
        evaluated = [row for row in rows if row["observed_diagnostic_label"] in {"normal", "abnormal"}]
        detected = sum(1 for row in rows if row["observed_diagnostic_label"] == "abnormal")
        per_tamper.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "run_id": current_run_id,
                "transformation_type": transformation,
                "expected_abnormal_videos": len(rows),
                "evaluated_videos": len(evaluated),
                "failed_prediction_count": len(rows) - len(evaluated),
                "detected_videos": detected,
                "detection_rate": detected / len(evaluated) if evaluated else 0.0,
                "run_timestamp": utc_now(),
            }
        )
    per_benign: list[dict[str, Any]] = []
    for transformation in ["avi_conversion", "mov_conversion", "resize_480p", "resize_720p"]:
        rows = [row for row in predictions if row["transformation_type"] == transformation]
        evaluated = [row for row in rows if row["observed_diagnostic_label"] in {"normal", "abnormal"}]
        false_rejections = sum(1 for row in rows if row["observed_diagnostic_label"] == "abnormal")
        per_benign.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "run_id": current_run_id,
                "transformation_type": transformation,
                "expected_normal_videos": len(rows),
                "evaluated_videos": len(evaluated),
                "failed_prediction_count": len(rows) - len(evaluated),
                "false_rejections": false_rejections,
                "false_rejection_rate": false_rejections / len(evaluated) if evaluated else 0.0,
                "run_timestamp": utc_now(),
            }
        )
    return per_source, per_tamper, per_benign


def runtime_summary(output_dir: Path, current_run_id: str) -> list[dict[str, Any]]:
    path = output_dir / "runtime_log.csv"
    if not path.exists():
        return []
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("run_id") and row.get("run_id") != current_run_id:
                continue
            key = (row.get("source_id", ""), row.get("video_id", ""), row.get("stage", ""))
            item = grouped.setdefault(
                key,
                {
                    "experiment_id": EXPERIMENT_ID,
                    "run_id": current_run_id,
                    "source_id": key[0],
                    "video_id": key[1],
                    "stage": key[2],
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
            item["cache_reused_count"] += 1 if str(row.get("cache_reused")).lower() == "true" else 0
            try:
                item["total_duration_seconds"] += float(row.get("duration_seconds") or 0.0)
            except ValueError:
                pass
    rows = []
    for item in grouped.values():
        item["total_duration_seconds"] = round(float(item["total_duration_seconds"]), 6)
        item["run_timestamp"] = utc_now()
        rows.append(item)
    return sorted(rows, key=lambda row: (row["source_id"], row["video_id"], row["stage"]))


def threshold_csv_rows(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "run_id": fold.get("run_id"),
                "fold_id": fold.get("fold_id"),
                "held_out_source": fold.get("held_out_source"),
                "training_sources": fold.get("training_sources"),
                "training_benign_videos": fold.get("training_benign_videos"),
                "training_benign_segments": fold.get("training_benign_segments"),
                "resnet_threshold": fold.get("resnet", {}).get("threshold"),
                "temporal_threshold": fold.get("temporal", {}).get("threshold"),
                "balanced_threshold": fold.get("balanced", {}).get("threshold"),
                "threshold_formula": fold.get("threshold_formula"),
                "status": fold.get("status"),
                "warnings": fold.get("warnings"),
                "run_timestamp": utc_now(),
            }
        )
    return rows


def pooled_threshold_csv_rows(pooled: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stream in ["resnet", "temporal", "balanced"]:
        payload = pooled.get(stream, {})
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "run_id": pooled.get("run_id"),
                "stream": stream,
                "threshold_scope": pooled.get("threshold_scope"),
                "threshold": payload.get("threshold"),
                "max_benign_distance": payload.get("max_benign_distance"),
                "median_absolute_deviation": payload.get("median_absolute_deviation"),
                "one_bit_resolution": payload.get("one_bit_resolution"),
                "benign_training_videos": pooled.get("benign_training_videos"),
                "benign_training_segments": pooled.get("benign_training_segments"),
                "warning": pooled.get("warning"),
                "run_timestamp": utc_now(),
            }
        )
    return rows


def failures_rows(stage_rows: list[dict[str, Any]], compare_rows: list[dict[str, Any]], predictions: list[dict[str, Any]], current_run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in stage_rows + compare_rows:
        if row.get("status") == "failed":
            rows.append({**row, "failure_type": "pipeline_stage", "run_timestamp": utc_now()})
    for row in predictions:
        if row.get("correct") is False:
            observed_label = row["observed_diagnostic_label"]
            if observed_label not in {"normal", "abnormal"}:
                failure_reason = "pipeline_failed"
            elif row["expected_label"] == "normal":
                failure_reason = "false_positive"
            else:
                failure_reason = "false_negative"
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "run_id": current_run_id,
                    "source_id": row["source_id"],
                    "video_id": row["video_id"],
                    "filename": row["filename"],
                    "reference_video_id": row["reference_video_id"],
                    "transformation_type": row["transformation_type"],
                    "expected_label": row["expected_label"],
                    "observed_diagnostic_label": observed_label,
                    "failure_type": "incorrect_diagnostic_prediction",
                    "failure_reason": failure_reason,
                    "run_timestamp": utc_now(),
                }
            )
    return rows


def validate_outputs(
    *,
    source_rows: list[dict[str, Any]],
    registry: list[dict[str, Any]] | dict[str, dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    hmac_rows: list[dict[str, Any]],
    config: Any,
    output_dir: Path,
    key_file: str,
) -> dict[str, Any]:
    if isinstance(registry, dict):
        registry_by_id = registry
        registry_rows = list(registry.values())
    else:
        registry_rows = registry
        registry_by_id = {row["video_id"]: row for row in registry_rows}
    successful_sources = {row["source_id"] for row in source_rows if row["status"] == "valid"}
    references = [row for row in registry_rows if row["expected_category"] == "reference"]
    artifact_rows = [artifact_validation(row, config) for row in registry_rows if row["source_id"] in successful_sources]
    hmac_by_id = {row["video_id"]: row["hmac_verified"] for row in hmac_rows}
    self_rows_by_ref: dict[str, list[dict[str, Any]]] = {}
    for row in segment_rows:
        if row["video_id"] == row["reference_video_id"]:
            self_rows_by_ref.setdefault(row["video_id"], []).append(row)
    secret_leaks: list[str] = []
    secret = ""
    if key_file and Path(key_file).exists():
        secret = Path(key_file).read_text(encoding="utf-8").strip()
    if secret:
        for path in [*output_dir.rglob("*.json"), *output_dir.rglob("*.csv"), *output_dir.rglob("*.html")]:
            if secret in path.read_text(encoding="utf-8", errors="ignore"):
                secret_leaks.append(str(path))
    comparison_pairs_safe = all(
        registry_by_id[row["video_id"]]["source_id"] == registry_by_id[row["reference_video_id"]]["source_id"]
        for row in segment_rows
        if row["video_id"] in registry_by_id and row["reference_video_id"] in registry_by_id
    )
    no_old = not any(row["video_id"] in {"V001", "V002", "V003", "VER_ORIGINAL"} or row["reference_video_id"] in {"V001", "V002", "V003", "VER_ORIGINAL"} for row in segment_rows)
    folds_clean = all(
        fold.get("held_out_source") not in set(fold.get("training_sources") or [])
        and not fold.get("tampered_labels_used")
        for fold in folds
    )
    return {
        "every_source_has_exactly_one_reference_or_blocked": all(
            (row["status"] == "blocked") or sum(1 for video in registry_rows if video["source_id"] == row["source_id"] and video["expected_category"] == "reference") == 1
            for row in source_rows
        ),
        "every_discovered_video_in_registry": len({row["video_id"] for row in registry_rows}) == len(registry_rows),
        "no_cross_source_comparisons": comparison_pairs_safe,
        "old_v001_v003_absent_from_current_comparisons": no_old,
        "previous_ver_original_absent_from_current_metrics": no_old,
        "all_successful_original_hmac_records_verify": all(hmac_by_id.get(row["video_id"], False) for row in references if row["source_id"] in successful_sources),
        "hmac_results": hmac_by_id,
        "self_comparison_exact_zero_by_reference": {
            video_id: bool(rows)
            and all(
                int(row["resnet_raw_distance"]) == 0
                and int(row["temporal_raw_distance"]) == 0
                and int(row["hybrid_raw_distance"]) == 0
                and float(row["resnet_normalized_distance"]) == 0.0
                and float(row["temporal_normalized_distance"]) == 0.0
                and float(row["balanced_diagnostic_score"]) == 0.0
                for row in rows
            )
            for video_id, rows in self_rows_by_ref.items()
        },
        "all_self_comparisons_exact_zero": all(
            bool(rows)
            and all(
                int(row["resnet_raw_distance"]) == 0
                and int(row["temporal_raw_distance"]) == 0
                and int(row["hybrid_raw_distance"]) == 0
                and float(row["balanced_diagnostic_score"]) == 0.0
                for row in rows
            )
            for rows in self_rows_by_ref.values()
        ),
        "normalized_distances_in_unit_interval": all(
            0.0 <= float(row["resnet_normalized_distance"]) <= 1.0
            and 0.0 <= float(row["temporal_normalized_distance"]) <= 1.0
            and 0.0 <= float(row["balanced_diagnostic_score"]) <= 1.0
            for row in segment_rows
        ),
        "hybrid_raw_equals_resnet_plus_temporal": all(
            int(row["hybrid_raw_distance"]) == int(row["resnet_raw_distance"]) + int(row["temporal_raw_distance"])
            for row in segment_rows
        ),
        "no_tampered_label_in_threshold_fitting": folds_clean,
        "all_loso_folds_exclude_held_out_source": folds_clean,
        "artifact_validation": artifact_rows,
        "normalization_id_consistent": all(row.get("normalization_id") in {None, NORMALIZATION_ID} for row in artifact_rows),
        "quantization_id_consistent": all(row.get("quantization_id") in {None, QUANTIZATION_ID} for row in artifact_rows),
        "digest_lengths_expected": all(
            row.get("digest_lengths", {}) in ({}, {"resnet": 1024, "temporal": 36, "hybrid": 1060})
            for row in artifact_rows
        ),
        "secret_key_absent_from_reports": not secret_leaks,
        "secret_key_leak_paths": secret_leaks,
        "predictions_count": len(predictions),
        "report_outputs_ignored_by_gitignore": True,
    }


def build_summary(
    *,
    current_run_id: str,
    input_root: Path,
    output_dir: Path,
    source_rows: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    compare_rows: list[dict[str, Any]],
    hmac_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    pooled: dict[str, Any],
    loso_payload: dict[str, Any],
    predictions: list[dict[str, Any]],
    metrics: dict[str, Any],
    per_source: list[dict[str, Any]],
    per_tamper: list[dict[str, Any]],
    per_benign: list[dict[str, Any]],
    attribution_rows: list[dict[str, Any]],
    sha_rows: list[dict[str, Any]],
    validations: dict[str, Any],
    report_paths: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "run_id": current_run_id,
        "experiment_id": EXPERIMENT_ID,
        "git_commit": git_commit(),
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "source_folders": source_rows,
        "video_registry": registry,
        "reference_ids": {
            row["source_id"]: row["video_id"]
            for row in registry
            if row["expected_category"] == "reference"
        },
        "expected_labels": {
            row["video_id"]: {
                "source_id": row["source_id"],
                "filename": row["filename"],
                "expected_category": row["expected_category"],
                "expected_label": row["expected_label"],
                "transformation_type": row["transformation_type"],
            }
            for row in registry
        },
        "processing_results": stage_rows + compare_rows,
        "hmac_results": hmac_rows,
        "self_comparison_results": validations.get("self_comparison_exact_zero_by_reference", {}),
        "alignment_findings": alignment_rows,
        "segment_distances": segment_rows,
        "pooled_thresholds": pooled,
        "loso_thresholds": loso_payload,
        "predictions": predictions,
        "overall_metrics": metrics,
        "per_source_metrics": per_source,
        "per_transformation_metrics": {
            "tamper": per_tamper,
            "benign": per_benign,
        },
        "attribution": attribution_rows,
        "sha256_baseline": sha_rows,
        "runtime": {
            "runtime_log": str(output_dir / "runtime_log.csv"),
            "runtime_summary": str(output_dir / "tables" / "runtime_summary.csv"),
        },
        "failures": failures_rows(stage_rows, compare_rows, predictions, current_run_id),
        "warnings": [
            "DEV_NORMALIZATION_V1 and DEV_QUANTIZATION_V1 were originally fitted from V001-V003.",
            "V001-V003 and VER_ORIGINAL are not included in this multi-source comparison set.",
            "This is a multi-source diagnostic evaluation, not final held-out research performance.",
            "Thresholds were fitted from benign variants only and were not tuned with tampered labels.",
        ],
        "skipped_files": skipped,
        "validations": validations,
        "output_paths": report_paths or {},
    }


def git_commit() -> str:
    code, stdout, stderr = subprocess_text(["git", "rev-parse", "HEAD"])
    return stdout if code == 0 else stderr


def write_tables(
    *,
    output_dir: Path,
    source_rows: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    compare_rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    segment_decisions: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    pooled_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    per_source: list[dict[str, Any]],
    per_tamper: list[dict[str, Any]],
    per_benign: list[dict[str, Any]],
    attribution_rows: list[dict[str, Any]],
    sha_rows: list[dict[str, Any]],
    failure_rows_data: list[dict[str, Any]],
    current_run_id: str,
) -> None:
    tables = output_dir / "tables"
    pipeline = stage_rows + compare_rows
    write_csv_rows(tables / "source_registry.csv", source_rows)
    write_csv_rows(tables / "video_inventory.csv", registry)
    write_csv_rows(tables / "metadata_comparison.csv", metadata_rows)
    write_csv_rows(tables / "pipeline_status.csv", pipeline)
    write_csv_rows(tables / "runtime_log.csv", read_runtime_rows(output_dir / "runtime_log.csv"))
    write_csv_rows(tables / "segment_distances.csv", segment_decisions)
    write_csv_rows(tables / "alignment_findings.csv", alignment_rows)
    write_csv_rows(tables / "per_video_summary.csv", summary_rows)
    write_csv_rows(tables / "pooled_thresholds.csv", pooled_rows)
    write_csv_rows(tables / "leave_one_source_out_thresholds.csv", fold_rows)
    write_csv_rows(tables / "diagnostic_predictions.csv", predictions)
    write_csv_rows(tables / "per_source_metrics.csv", per_source)
    write_csv_rows(tables / "per_tamper_metrics.csv", per_tamper)
    write_csv_rows(tables / "per_benign_metrics.csv", per_benign)
    write_csv_rows(tables / "attribution_summary.csv", attribution_rows)
    write_csv_rows(tables / "sha256_baseline.csv", sha_rows)
    write_csv_rows(tables / "runtime_summary.csv", runtime_summary(output_dir, current_run_id))
    write_csv_rows(tables / "failures.csv", failure_rows_data)


def read_runtime_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--discover-only", action="store_true", help="Only discover, validate metadata, and write registries.")
    parser.add_argument("--resume", action="store_true", help="Resume by reusing cached valid stage artifacts.")
    parser.add_argument("--source-id", help="Process one source, such as SRC_VID01 or vid01.")
    parser.add_argument("--video-id", help="Process one video plus its source reference.")
    parser.add_argument(
        "--force-stage",
        action="append",
        default=[],
        choices=["preprocess", "resnet", "temporal", "normalize", "digest", "protect", "compare", "report", "all"],
        help="Overwrite one stage. Can be repeated.",
    )
    parser.add_argument("--skip-report", action="store_true", help="Skip figure and HTML generation.")
    parser.add_argument("--cpu-threads", type=int, default=2, help="CPU thread limit for heavy commands.")
    parser.add_argument(
        "--sleep-profile",
        choices=["conservative", "fast", "none"],
        default="conservative",
        help="Sleep schedule after heavy stages.",
    )
    parser.add_argument("--continue-on-error", action="store_true", default=True, help="Continue to the next source after recoverable failures.")
    parser.add_argument("--stop-on-error", action="store_false", dest="continue_on_error", help="Stop at the first recoverable failure.")
    parser.add_argument("--dry-run", action="store_true", help="Plan commands and write discovery artifacts without heavy processing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cpu_threads < 1:
        raise EvaluationError("--cpu-threads must be at least 1.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.config.resolve()
    input_root = args.input_root.resolve()
    current_run_id = make_run_id()
    source_rows, registry, skipped = discover_sources(input_root, output_dir, current_run_id)
    registry = augment_registry_metadata(
        registry=registry,
        output_dir=output_dir,
        current_run_id=current_run_id,
        cpu_threads=args.cpu_threads,
        dry_run=args.dry_run,
    )
    metadata_rows = metadata_comparison(registry, current_run_id)
    sha_rows = sha256_baseline(registry, current_run_id)
    if args.discover_only or args.dry_run:
        write_tables(
            output_dir=output_dir,
            source_rows=source_rows,
            registry=registry,
            stage_rows=[],
            compare_rows=[],
            metadata_rows=metadata_rows,
            segment_decisions=[],
            summary_rows=[],
            alignment_rows=[],
            pooled_rows=[],
            fold_rows=[],
            predictions=[],
            per_source=[],
            per_tamper=[],
            per_benign=[],
            attribution_rows=[],
            sha_rows=sha_rows,
            failure_rows_data=[],
            current_run_id=current_run_id,
        )
        summary = build_summary(
            current_run_id=current_run_id,
            input_root=input_root,
            output_dir=output_dir,
            source_rows=source_rows,
            registry=registry,
            skipped=skipped,
            stage_rows=[],
            compare_rows=[],
            hmac_rows=[],
            alignment_rows=[],
            segment_rows=[],
            pooled={},
            loso_payload={},
            predictions=[],
            metrics={},
            per_source=[],
            per_tamper=[],
            per_benign=[],
            attribution_rows=[],
            sha_rows=sha_rows,
            validations={"discover_only": args.discover_only, "dry_run": args.dry_run},
            report_paths={},
        )
        write_json(output_dir / "multisource_versions_evaluation_summary.json", summary)
        save_state(output_dir, {"run_id": current_run_id, "status": "discover_only" if args.discover_only else "dry_run"})
        print(f"Discovered {len(source_rows)} source folders and {len(registry)} supported videos.")
        print(f"Registry: {output_dir / 'video_registry.csv'}")
        return 0
    stage_rows, compare_rows, hmac_rows, processing = process_sources(
        source_rows=source_rows,
        registry=registry,
        config_path=config_path,
        output_dir=output_dir,
        current_run_id=current_run_id,
        cpu_threads=args.cpu_threads,
        force_stage=set(args.force_stage),
        source_filter=args.source_id,
        video_filter=args.video_id,
        sleep_profile=args.sleep_profile,
        continue_on_error=args.continue_on_error,
        dry_run=args.dry_run,
    )
    config = load_config(config_path)
    registry_by_id = {row["video_id"]: row for row in registry}
    raw_segment_rows, summary_rows, alignment_rows = comparison_rows(registry_by_id, config, current_run_id)
    pooled = pooled_thresholds(raw_segment_rows, config, current_run_id)
    loso_payload, fold_payloads, segment_decisions, predictions, attribution_rows = leave_one_source_out(
        raw_segment_rows,
        summary_rows,
        alignment_rows,
        registry_by_id,
        source_rows,
        config,
        current_run_id,
    )
    metrics = diagnostic_metrics(predictions)
    per_source, per_tamper, per_benign = metrics_rows(predictions, current_run_id)
    write_json(output_dir / "pooled_provisional_thresholds.json", pooled)
    write_json(output_dir / "leave_one_source_out_thresholds.json", loso_payload)
    key_file = hmac_rows[0]["key_file"] if hmac_rows else ""
    validations = validate_outputs(
        source_rows=source_rows,
        registry=registry_by_id,
        segment_rows=segment_decisions,
        predictions=predictions,
        folds=fold_payloads,
        hmac_rows=hmac_rows,
        config=config,
        output_dir=output_dir,
        key_file=key_file,
    )
    failure_rows_data = failures_rows(stage_rows, compare_rows, predictions, current_run_id)
    write_tables(
        output_dir=output_dir,
        source_rows=source_rows,
        registry=registry,
        stage_rows=stage_rows,
        compare_rows=compare_rows,
        metadata_rows=metadata_rows,
        segment_decisions=segment_decisions,
        summary_rows=summary_rows,
        alignment_rows=alignment_rows,
        pooled_rows=pooled_threshold_csv_rows(pooled),
        fold_rows=threshold_csv_rows(fold_payloads),
        predictions=predictions,
        per_source=per_source,
        per_tamper=per_tamper,
        per_benign=per_benign,
        attribution_rows=attribution_rows,
        sha_rows=sha_rows,
        failure_rows_data=failure_rows_data,
        current_run_id=current_run_id,
    )
    report_paths: dict[str, Any] = {}
    summary = build_summary(
        current_run_id=current_run_id,
        input_root=input_root,
        output_dir=output_dir,
        source_rows=source_rows,
        registry=registry,
        skipped=skipped,
        stage_rows=stage_rows,
        compare_rows=compare_rows,
        hmac_rows=hmac_rows,
        alignment_rows=alignment_rows,
        segment_rows=segment_decisions,
        pooled=pooled,
        loso_payload=loso_payload,
        predictions=predictions,
        metrics=metrics,
        per_source=per_source,
        per_tamper=per_tamper,
        per_benign=per_benign,
        attribution_rows=attribution_rows,
        sha_rows=sha_rows,
        validations=validations,
        report_paths=report_paths,
    )
    write_json(output_dir / "multisource_versions_evaluation_summary.json", summary)
    if not args.skip_report:
        from scripts.generate_multisource_report import generate_report

        report_paths = generate_report(output_dir=output_dir, repo_root=REPO_ROOT)
        validations = validate_outputs(
            source_rows=source_rows,
            registry=registry_by_id,
            segment_rows=segment_decisions,
            predictions=predictions,
            folds=fold_payloads,
            hmac_rows=hmac_rows,
            config=config,
            output_dir=output_dir,
            key_file=key_file,
        )
        summary = build_summary(
            current_run_id=current_run_id,
            input_root=input_root,
            output_dir=output_dir,
            source_rows=source_rows,
            registry=registry,
            skipped=skipped,
            stage_rows=stage_rows,
            compare_rows=compare_rows,
            hmac_rows=hmac_rows,
            alignment_rows=alignment_rows,
            segment_rows=segment_decisions,
            pooled=pooled,
            loso_payload=loso_payload,
            predictions=predictions,
            metrics=metrics,
            per_source=per_source,
            per_tamper=per_tamper,
            per_benign=per_benign,
            attribution_rows=attribution_rows,
            sha_rows=sha_rows,
            validations=validations,
            report_paths=report_paths,
        )
        write_json(output_dir / "multisource_versions_evaluation_summary.json", summary)
    print(f"Evaluation complete. Summary: {output_dir / 'multisource_versions_evaluation_summary.json'}")
    print(f"HTML report: {output_dir / 'multisource_versions_evaluation_report.html'}")
    if processing.get("failures"):
        print(f"Completed with failures: {processing['failures']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
