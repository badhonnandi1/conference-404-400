"""Storage, cache, and verification helpers for HMAC-protected digest records."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from src.authentication.canonicalization import (
    SCHEMA_VERSION,
    CanonicalizationError,
    build_canonical_payload_from_digest_files,
    canonical_json_bytes,
    canonical_payload_sha256,
)
from src.authentication.hmac_auth import (
    DEFAULT_ALGORITHM,
    HMACAuthenticationError,
    HMACKeyInfo,
    compute_hmac_sha256_hex,
    verify_hmac_sha256_hex,
)
from src.authentication.quantization import QUANTIZATION_WARNING
from src.features.feature_storage import sha256_file
from src.video.metadata import ExistingOutputError


AUTHENTICATION_RECORD_WARNING = (
    "This HMAC-protected record is for development pipeline validation. "
    "Development keys and development calibration artifacts must not be used in production."
)


class AuthenticationRecordError(RuntimeError):
    """Raised when an authentication record cannot be saved, loaded, or verified."""


@dataclass(frozen=True)
class AuthenticationRecordPaths:
    """Output paths for one video's HMAC-protected authentication record."""

    output_dir: Path
    record_path: Path


@dataclass(frozen=True)
class StoredAuthenticationRecord:
    """Saved or cached authentication record metadata."""

    record: dict[str, Any]
    paths: AuthenticationRecordPaths
    record_file_sha256: str
    cache_reused: bool


@dataclass(frozen=True)
class VerificationResult:
    """Structured result returned by HMAC authentication-record verification."""

    record_valid: bool
    hmac_valid: bool
    payload_checksum_valid: bool
    key_fingerprint_match: bool
    schema_valid: bool
    algorithm_supported: bool
    video_id: str | None
    key_id: str | None
    failure_reason: str | None
    verification_timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly verification-result mapping."""

        return asdict(self)


def authentication_record_paths(root: str | Path, video_id: str) -> AuthenticationRecordPaths:
    """Return deterministic authentication-record paths for one video."""

    output_dir = Path(root) / video_id
    return AuthenticationRecordPaths(
        output_dir=output_dir,
        record_path=output_dir / f"{video_id}_authentication_record.json",
    )


def build_authentication_record(
    payload: dict[str, Any],
    key_info: HMACKeyInfo,
    algorithm: str = DEFAULT_ALGORITHM,
) -> dict[str, Any]:
    """Build an HMAC-protected authentication record from a canonical payload."""

    if algorithm != DEFAULT_ALGORITHM:
        raise AuthenticationRecordError(f"Unsupported authentication algorithm: {algorithm}")
    payload_bytes = canonical_json_bytes(payload)
    tag_hex = compute_hmac_sha256_hex(key_info.key, payload_bytes)
    return {
        "payload": payload,
        "authentication": {
            "record_schema_version": SCHEMA_VERSION,
            "algorithm": algorithm,
            "key_id": key_info.key_id,
            "key_fingerprint": key_info.key_fingerprint,
            "key_source_type": key_info.source_type,
            "key_length_bytes": key_info.key_length_bytes,
            "tag_hex": tag_hex,
            "canonical_payload_sha256": canonical_payload_sha256(payload),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "warnings": [AUTHENTICATION_RECORD_WARNING, QUANTIZATION_WARNING],
        },
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and Path(temp_name).exists():
            Path(temp_name).unlink()


def save_authentication_record(
    record: dict[str, Any],
    paths: AuthenticationRecordPaths,
    overwrite: bool = False,
) -> StoredAuthenticationRecord:
    """Atomically save an authentication record and return its file checksum."""

    if paths.record_path.exists() and not overwrite:
        raise ExistingOutputError(
            f"Authentication record already exists: {paths.record_path}. Use --overwrite to replace it."
        )
    _atomic_write_json(paths.record_path, record)
    return StoredAuthenticationRecord(
        record=record,
        paths=paths,
        record_file_sha256=sha256_file(paths.record_path),
        cache_reused=False,
    )


def load_authentication_record(path: str | Path) -> dict[str, Any]:
    """Load an authentication record JSON file."""

    record_path = Path(path)
    if not record_path.exists():
        raise AuthenticationRecordError(f"Authentication record not found: {record_path}")
    with record_path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict):
        raise AuthenticationRecordError(f"Authentication record root must be a JSON object: {record_path}")
    return record


def _validate_record_shape(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = record.get("payload")
    authentication = record.get("authentication")
    if not isinstance(payload, dict):
        raise AuthenticationRecordError("Authentication record is missing a payload object.")
    if not isinstance(authentication, dict):
        raise AuthenticationRecordError("Authentication record is missing an authentication object.")
    required_auth = [
        "algorithm",
        "key_id",
        "key_fingerprint",
        "tag_hex",
        "canonical_payload_sha256",
        "record_schema_version",
    ]
    missing = [field for field in required_auth if field not in authentication]
    if missing:
        raise AuthenticationRecordError(f"Authentication record is missing fields: {missing}.")
    tag_hex = str(authentication.get("tag_hex"))
    if len(tag_hex) != 64:
        raise AuthenticationRecordError("Authentication tag must be 64 hexadecimal characters.")
    try:
        bytes.fromhex(tag_hex)
    except ValueError as exc:
        raise AuthenticationRecordError("Authentication tag is not valid hexadecimal.") from exc
    return payload, authentication


def verify_authentication_record(
    record: dict[str, Any],
    key_info: HMACKeyInfo,
    algorithm: str = DEFAULT_ALGORITHM,
) -> VerificationResult:
    """Verify an HMAC-protected authentication record with the supplied key."""

    checked_at = datetime.now(timezone.utc).isoformat()
    video_id: str | None = None
    key_id: str | None = None
    try:
        payload, authentication = _validate_record_shape(record)
        video_id = str(payload.get("video_id")) if payload.get("video_id") is not None else None
        key_id = str(authentication.get("key_id")) if authentication.get("key_id") is not None else None
        schema_valid = (
            int(authentication.get("record_schema_version")) == SCHEMA_VERSION
            and int(payload.get("schema_version")) == SCHEMA_VERSION
        )
        algorithm_supported = str(authentication.get("algorithm")) == algorithm == DEFAULT_ALGORITHM
        payload_checksum = canonical_payload_sha256(payload)
        payload_checksum_valid = payload_checksum == str(authentication.get("canonical_payload_sha256"))
        key_fingerprint_match = str(authentication.get("key_fingerprint")) == key_info.key_fingerprint
        hmac_valid = (
            algorithm_supported
            and verify_hmac_sha256_hex(key_info.key, canonical_json_bytes(payload), str(authentication.get("tag_hex")))
        )
        failures: list[str] = []
        if not schema_valid:
            failures.append("schema mismatch")
        if not algorithm_supported:
            failures.append("unsupported algorithm")
        if not payload_checksum_valid:
            failures.append("payload checksum mismatch")
        if not key_fingerprint_match:
            failures.append("key fingerprint mismatch")
        if not hmac_valid:
            failures.append("HMAC tag mismatch")
        record_valid = not failures
        return VerificationResult(
            record_valid=record_valid,
            hmac_valid=hmac_valid,
            payload_checksum_valid=payload_checksum_valid,
            key_fingerprint_match=key_fingerprint_match,
            schema_valid=schema_valid,
            algorithm_supported=algorithm_supported,
            video_id=video_id,
            key_id=key_id,
            failure_reason="; ".join(failures) if failures else None,
            verification_timestamp=checked_at,
        )
    except (AuthenticationRecordError, TypeError, ValueError, OverflowError) as exc:
        return VerificationResult(
            record_valid=False,
            hmac_valid=False,
            payload_checksum_valid=False,
            key_fingerprint_match=False,
            schema_valid=False,
            algorithm_supported=False,
            video_id=video_id,
            key_id=key_id,
            failure_reason=str(exc),
            verification_timestamp=checked_at,
        )


def verify_authentication_record_file(
    record_path: str | Path,
    key_info: HMACKeyInfo,
    algorithm: str = DEFAULT_ALGORITHM,
) -> VerificationResult:
    """Load and verify an authentication record file."""

    return verify_authentication_record(load_authentication_record(record_path), key_info, algorithm=algorithm)


def _existing_record_reusable(
    record: dict[str, Any],
    current_payload: dict[str, Any],
    key_info: HMACKeyInfo,
    algorithm: str,
) -> bool:
    if record.get("payload") != current_payload:
        return False
    authentication = record.get("authentication", {})
    if not isinstance(authentication, dict):
        return False
    if authentication.get("algorithm") != algorithm:
        return False
    if authentication.get("key_id") != key_info.key_id:
        return False
    if authentication.get("key_fingerprint") != key_info.key_fingerprint:
        return False
    result = verify_authentication_record(record, key_info, algorithm=algorithm)
    return result.record_valid


def protect_digest_record(
    video_id: str,
    digest_root: str | Path,
    authentication_record_root: str | Path,
    key_info: HMACKeyInfo,
    schema_version: int = SCHEMA_VERSION,
    algorithm: str = DEFAULT_ALGORITHM,
    overwrite: bool = False,
) -> StoredAuthenticationRecord:
    """Create or reuse an HMAC-protected authentication record for one video's digests."""

    paths = authentication_record_paths(authentication_record_root, video_id)
    payload = build_canonical_payload_from_digest_files(
        video_id=video_id,
        digest_root=digest_root,
        schema_version=schema_version,
    )
    if paths.record_path.exists() and not overwrite:
        record = load_authentication_record(paths.record_path)
        if _existing_record_reusable(record, payload, key_info, algorithm):
            return StoredAuthenticationRecord(
                record=record,
                paths=paths,
                record_file_sha256=sha256_file(paths.record_path),
                cache_reused=True,
            )
        raise ExistingOutputError(
            f"Authentication record already exists but cache metadata or HMAC verification did not match: "
            f"{paths.record_path}. Use --overwrite to regenerate it."
        )
    record = build_authentication_record(payload, key_info, algorithm=algorithm)
    return save_authentication_record(record, paths, overwrite=overwrite)


def inspect_authentication_record(record_path: str | Path) -> dict[str, Any]:
    """Return a non-secret summary for one authentication record."""

    path = Path(record_path)
    record = load_authentication_record(path)
    payload, authentication = _validate_record_shape(record)
    return {
        "video_id": payload.get("video_id"),
        "schema_version": payload.get("schema_version"),
        "record_schema_version": authentication.get("record_schema_version"),
        "algorithm": authentication.get("algorithm"),
        "key_id": authentication.get("key_id"),
        "key_fingerprint": authentication.get("key_fingerprint"),
        "segment_count": payload.get("segment_count"),
        "normalization_id": payload.get("normalization_id"),
        "quantization_id": payload.get("quantization_id"),
        "digest_lengths": payload.get("digest_lengths"),
        "canonical_payload_sha256": authentication.get("canonical_payload_sha256"),
        "record_file_sha256": sha256_file(path),
        "development_only": payload.get("development_only"),
        "warnings": authentication.get("warnings", []),
        "record_path": str(path.resolve()),
    }


def authentication_payload_from_record(record_path: str | Path) -> dict[str, Any]:
    """Load and return the authenticated payload from a record file."""

    record = load_authentication_record(record_path)
    payload, _ = _validate_record_shape(record)
    return payload
