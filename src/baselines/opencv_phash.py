"""Segment-level video baseline using OpenCV's official perceptual hash."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


class PHashError(RuntimeError):
    """Raised when OpenCV pHash input or output is invalid."""


@dataclass(frozen=True)
class FrameHash:
    """One sampled frame and its OpenCV pHash provenance."""

    source_video_id: str
    segment_id: int
    sample_index: int
    timestamp_seconds: float
    frame_path: str
    valid: bool
    hash_bytes: np.ndarray | None
    failure_reason: str | None
    opencv_version: str
    phash_api: str
    hash_byte_length: int
    logical_bit_length: int

    def provenance_dict(self) -> dict[str, Any]:
        """Return a CSV/JSON-friendly provenance row without changing the hash."""

        return {
            "source_video_id": self.source_video_id,
            "segment_id": self.segment_id,
            "sample_index": self.sample_index,
            "timestamp_seconds": self.timestamp_seconds,
            "frame_path": self.frame_path,
            "valid": self.valid,
            "failure_reason": self.failure_reason,
            "opencv_version": self.opencv_version,
            "phash_api": self.phash_api,
            "hash_byte_length": self.hash_byte_length,
            "logical_bit_length": self.logical_bit_length,
            "hash_hex": (
                np.asarray(self.hash_bytes, dtype=np.uint8).reshape(-1).tobytes().hex()
                if self.hash_bytes is not None
                else ""
            ),
        }


class OpenCVPHash:
    """Validated adapter for OpenCV's official image-hash implementation."""

    def __init__(self) -> None:
        module = getattr(cv2, "img_hash", None)
        if module is None:
            raise PHashError("Installed OpenCV does not expose cv2.img_hash.")
        self._object = None
        if hasattr(module, "pHash"):
            self.api_name = "cv2.img_hash.pHash"
            self._compute = module.pHash
        elif hasattr(module, "PHash_create"):
            self._object = module.PHash_create()
            self.api_name = "cv2.img_hash.PHash_create().compute"
            self._compute = self._object.compute
        elif hasattr(getattr(module, "PHash", None), "create"):
            self._object = module.PHash.create()
            self.api_name = "cv2.img_hash.PHash.create().compute"
            self._compute = self._object.compute
        else:
            raise PHashError("OpenCV img_hash is present but no supported pHash callable exists.")

        probe = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
        first = self.compute(probe)
        second = self.compute(probe.copy())
        if not np.array_equal(first, second):
            raise PHashError("OpenCV pHash is not deterministic for an identical probe image.")
        self.hash_byte_length = int(first.nbytes)
        self.logical_bit_length = self.hash_byte_length * 8

    @staticmethod
    def _validate_image(image: np.ndarray) -> np.ndarray:
        values = np.asarray(image)
        if values.dtype != np.uint8:
            raise PHashError(f"OpenCV pHash input must have dtype uint8, got {values.dtype}.")
        if values.ndim == 2:
            return values
        if values.ndim == 3 and values.shape[2] in {3, 4}:
            return values
        raise PHashError(
            "OpenCV pHash input must be a grayscale, BGR, or BGRA uint8 image; "
            f"got shape {values.shape}."
        )

    def compute(self, image: np.ndarray) -> np.ndarray:
        """Compute and validate the exact byte array returned by OpenCV."""

        values = self._validate_image(image)
        output = np.asarray(self._compute(values))
        if output.dtype != np.uint8:
            raise PHashError(f"OpenCV pHash output must have dtype uint8, got {output.dtype}.")
        if output.ndim != 2 or output.shape[0] != 1 or output.size == 0:
            raise PHashError(f"Unexpected OpenCV pHash output shape: {output.shape}.")
        return np.ascontiguousarray(output.reshape(-1), dtype=np.uint8)

    def hash_image_file(
        self,
        *,
        source_video_id: str,
        segment_id: int,
        sample_index: int,
        timestamp_seconds: float,
        frame_path: str | Path,
    ) -> FrameHash:
        """Load and hash one cached sampled frame, retaining failure provenance."""

        path = Path(frame_path)
        try:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise PHashError(f"OpenCV could not decode sampled frame: {path}")
            digest = self.compute(image)
            return FrameHash(
                source_video_id=source_video_id,
                segment_id=int(segment_id),
                sample_index=int(sample_index),
                timestamp_seconds=float(timestamp_seconds),
                frame_path=str(path),
                valid=True,
                hash_bytes=digest,
                failure_reason=None,
                opencv_version=cv2.__version__,
                phash_api=self.api_name,
                hash_byte_length=int(digest.nbytes),
                logical_bit_length=int(digest.nbytes * 8),
            )
        except (OSError, PHashError, cv2.error) as exc:
            return FrameHash(
                source_video_id=source_video_id,
                segment_id=int(segment_id),
                sample_index=int(sample_index),
                timestamp_seconds=float(timestamp_seconds),
                frame_path=str(path),
                valid=False,
                hash_bytes=None,
                failure_reason=str(exc),
                opencv_version=cv2.__version__,
                phash_api=self.api_name,
                hash_byte_length=self.hash_byte_length,
                logical_bit_length=self.logical_bit_length,
            )

    def hash_frame_manifest(self, manifest_path: str | Path) -> list[FrameHash]:
        """Hash every requested sample in an existing project frame manifest."""

        path = Path(manifest_path)
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        video_id = str(manifest["video_id"])
        hashes: list[FrameHash] = []
        for record in manifest.get("frame_records", []):
            timestamp = float(record["requested_timestamp_seconds"])
            if not bool(record.get("success")):
                hashes.append(
                    FrameHash(
                        source_video_id=video_id,
                        segment_id=int(record["segment_id"]),
                        sample_index=int(record["frame_index"]),
                        timestamp_seconds=timestamp,
                        frame_path=str(record.get("output_frame_path", "")),
                        valid=False,
                        hash_bytes=None,
                        failure_reason=str(record.get("error_message") or "frame sampling failed"),
                        opencv_version=cv2.__version__,
                        phash_api=self.api_name,
                        hash_byte_length=self.hash_byte_length,
                        logical_bit_length=self.logical_bit_length,
                    )
                )
                continue
            hashes.append(
                self.hash_image_file(
                    source_video_id=video_id,
                    segment_id=int(record["segment_id"]),
                    sample_index=int(record["frame_index"]),
                    timestamp_seconds=timestamp,
                    frame_path=str(record["output_frame_path"]),
                )
            )
        return hashes


def phash_hamming_distance(left: np.ndarray, right: np.ndarray) -> tuple[int, float, int]:
    """Return raw, normalized, and logical-bit-length pHash distance."""

    a = np.asarray(left)
    b = np.asarray(right)
    if a.dtype != np.uint8 or b.dtype != np.uint8:
        raise PHashError("pHash arrays must have dtype uint8.")
    a = a.reshape(-1)
    b = b.reshape(-1)
    if a.size == 0 or a.shape != b.shape:
        raise PHashError(f"pHash arrays must have the same non-empty shape, got {a.shape} and {b.shape}.")
    bit_length = int(a.nbytes * 8)
    raw = int(np.unpackbits(np.bitwise_xor(a, b)).sum())
    normalized = raw / bit_length
    if not 0.0 <= normalized <= 1.0:
        raise PHashError(f"Normalized pHash distance is outside [0, 1]: {normalized}.")
    return raw, float(normalized), bit_length


def _frame_map(frames: Iterable[FrameHash]) -> dict[tuple[int, int], FrameHash]:
    mapped: dict[tuple[int, int], FrameHash] = {}
    for frame in frames:
        key = (int(frame.segment_id), int(frame.sample_index))
        if key in mapped:
            raise PHashError(f"Duplicate sampled-frame key: segment={key[0]}, sample={key[1]}.")
        mapped[key] = frame
    return mapped


def compare_video_frame_hashes(
    *,
    reference_hashes: Iterable[FrameHash],
    query_hashes: Iterable[FrameHash],
    reference_segment_ids: Iterable[int],
    query_segment_ids: Iterable[int],
    timestamp_tolerance_seconds: float = 1.0e-9,
) -> dict[str, Any]:
    """Compare corresponding frame hashes and aggregate segment/video scores."""

    ref_map = _frame_map(reference_hashes)
    qry_map = _frame_map(query_hashes)
    ref_segments = {int(value) for value in reference_segment_ids}
    qry_segments = {int(value) for value in query_segment_ids}
    missing_segment_ids = sorted(ref_segments - qry_segments)
    extra_segment_ids = sorted(qry_segments - ref_segments)
    segment_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []

    for segment_id in sorted(ref_segments & qry_segments):
        sample_indices = sorted(
            {sample for seg, sample in ref_map if seg == segment_id}
            | {sample for seg, sample in qry_map if seg == segment_id}
        )
        distances: list[float] = []
        missing_count = 0
        reasons: list[str] = []
        for sample_index in sample_indices:
            ref = ref_map.get((segment_id, sample_index))
            qry = qry_map.get((segment_id, sample_index))
            reason = None
            raw_distance: int | None = None
            normalized_distance: float | None = None
            bit_length: int | None = None
            if ref is None or qry is None:
                reason = "corresponding sample index missing"
            elif not ref.valid or not qry.valid:
                reason = "reference or query frame hash invalid"
            elif not math.isclose(
                ref.timestamp_seconds,
                qry.timestamp_seconds,
                rel_tol=0.0,
                abs_tol=timestamp_tolerance_seconds,
            ):
                reason = "requested timestamps do not correspond"
            else:
                raw_distance, normalized_distance, bit_length = phash_hamming_distance(
                    np.asarray(ref.hash_bytes),
                    np.asarray(qry.hash_bytes),
                )
                distances.append(normalized_distance)
            if reason:
                missing_count += 1
                reasons.append(f"sample {sample_index}: {reason}")
            frame_rows.append(
                {
                    "segment_id": segment_id,
                    "sample_index": sample_index,
                    "reference_timestamp_seconds": ref.timestamp_seconds if ref else None,
                    "query_timestamp_seconds": qry.timestamp_seconds if qry else None,
                    "raw_frame_distance": raw_distance,
                    "normalized_frame_distance": normalized_distance,
                    "logical_bit_length": bit_length,
                    "valid_pair": reason is None,
                    "failure_reason": reason,
                }
            )
        valid = bool(distances)
        segment_rows.append(
            {
                "segment_id": segment_id,
                "segment_valid": valid,
                "segment_phash_score": float(np.mean(distances)) if valid else None,
                "median_frame_distance": float(np.median(distances)) if valid else None,
                "maximum_frame_distance": float(np.max(distances)) if valid else None,
                "matched_frame_hash_count": len(distances),
                "missing_frame_hash_count": missing_count,
                "invalid_reason": "; ".join(reasons) if not valid else None,
            }
        )

    valid_scores = [
        float(row["segment_phash_score"])
        for row in segment_rows
        if row["segment_valid"] and row["segment_phash_score"] is not None
    ]
    invalid_segment_ids = [int(row["segment_id"]) for row in segment_rows if not row["segment_valid"]]
    return {
        "segment_rows": segment_rows,
        "frame_rows": frame_rows,
        "video_phash_score": max(valid_scores) if valid_scores else None,
        "structural_issue": bool(missing_segment_ids or extra_segment_ids),
        "missing_segment_ids": missing_segment_ids,
        "extra_segment_ids": extra_segment_ids,
        "missing_segment_count": len(missing_segment_ids),
        "extra_segment_count": len(extra_segment_ids),
        "invalid_segment_ids": invalid_segment_ids,
        "invalid_comparison": bool(invalid_segment_ids or not valid_scores),
    }


def fit_robust_threshold(
    benign_video_scores: Iterable[float],
    *,
    margin_multiplier: float,
    logical_bit_length: int,
) -> dict[str, float | int]:
    """Fit max-benign plus max(lambda*MAD, one-bit resolution), clipped to [0, 1]."""

    values = np.asarray(list(benign_video_scores), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise PHashError("At least one benign video score is required for threshold fitting.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise PHashError("Benign pHash scores must be finite values in [0, 1].")
    if margin_multiplier < 0.0:
        raise PHashError("pHash threshold margin multiplier must be non-negative.")
    if logical_bit_length <= 0:
        raise PHashError("pHash logical bit length must be positive.")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    maximum = float(np.max(values))
    resolution = 1.0 / int(logical_bit_length)
    margin = max(float(margin_multiplier) * mad, resolution)
    return {
        "threshold": float(np.clip(maximum + margin, 0.0, 1.0)),
        "margin_multiplier": float(margin_multiplier),
        "benign_calibration_video_count": int(values.size),
        "maximum_benign_score": maximum,
        "median_benign_score": median,
        "mad": mad,
        "minimum_score_resolution": resolution,
        "applied_margin": margin,
    }


def video_decision(
    *,
    video_score: float | None,
    threshold: float,
    structural_issue: bool,
    invalid_comparison: bool,
    segment_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Apply strict greater-than scoring and fail closed on unusable frame evidence."""

    if not 0.0 <= threshold <= 1.0:
        raise PHashError("pHash threshold must be in [0, 1].")
    if video_score is not None and (not math.isfinite(video_score) or not 0.0 <= video_score <= 1.0):
        raise PHashError("pHash video score must be finite and in [0, 1].")
    abnormal_segments = sorted(
        int(row["segment_id"])
        for row in segment_rows
        if row.get("segment_valid")
        and row.get("segment_phash_score") is not None
        and float(row["segment_phash_score"]) > threshold
    )
    score_exceeds = video_score is not None and video_score > threshold
    abnormal = bool(score_exceeds or structural_issue or invalid_comparison)
    reasons = []
    if score_exceeds:
        reasons.append("video score strictly exceeds threshold")
    if structural_issue:
        reasons.append("missing or extra segment")
    if invalid_comparison:
        reasons.append("unusable corresponding frame-hash evidence")
    return {
        "observed_label": "abnormal" if abnormal else "normal",
        "score_exceeds_threshold": bool(score_exceeds),
        "structural_abnormal": bool(structural_issue),
        "invalid_comparison_abnormal": bool(invalid_comparison),
        "abnormal_segment_ids": abnormal_segments,
        "decision_reason": "; ".join(reasons) if reasons else "within threshold with complete valid structure",
    }
