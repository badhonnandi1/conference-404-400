"""Tests for HMAC-protected authentication-record storage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.authentication.auth_record_storage import (
    AUTHENTICATION_RECORD_WARNING,
    authentication_record_paths,
    inspect_authentication_record,
    load_authentication_record,
    protect_digest_record,
    save_authentication_record,
    verify_authentication_record,
    verify_authentication_record_file,
)
from src.authentication.canonicalization import (
    CanonicalizationError,
    build_canonical_payload_from_digest_files,
)
from src.authentication.digest import pack_bit_matrix
from src.authentication.digest_storage import digest_output_paths
from src.authentication.hmac_auth import HMACKeyInfo, key_fingerprint
from src.authentication.quantization import DIGEST_LENGTHS, QUANTIZATION_WARNING, STREAM_BOUNDARIES
from src.features.feature_storage import sha256_file
from src.video.metadata import ExistingOutputError


def _key_info(key_byte: int = 1) -> HMACKeyInfo:
    key = bytes([key_byte]) * 32
    return HMACKeyInfo(
        key=key,
        key_id=f"KEY_{key_byte}",
        source_type="test",
        key_length_bytes=len(key),
        key_fingerprint=key_fingerprint(key),
    )


def _flip_hex_character(value: str) -> str:
    replacement = "0" if value[0].lower() != "0" else "1"
    return replacement + value[1:]


def _write_digest_fixture(root: Path, video_id: str = "T001") -> None:
    paths = digest_output_paths(root / "digests", video_id)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    segment_ids = np.asarray([0, 1], dtype=np.int64)
    starts = np.asarray([0.0, 5.0], dtype=np.float64)
    ends = np.asarray([5.0, 10.0], dtype=np.float64)
    resnet = np.vstack(
        [
            np.zeros(DIGEST_LENGTHS["resnet"], dtype=np.uint8),
            np.arange(DIGEST_LENGTHS["resnet"], dtype=np.uint16) % 2,
        ]
    ).astype(np.uint8)
    temporal = np.vstack(
        [
            np.ones(DIGEST_LENGTHS["temporal"], dtype=np.uint8),
            np.arange(DIGEST_LENGTHS["temporal"], dtype=np.uint16) % 2,
        ]
    ).astype(np.uint8)
    hybrid = np.concatenate([resnet, temporal], axis=1)
    resnet_packed, resnet_length, resnet_padding = pack_bit_matrix(resnet, "big")
    temporal_packed, temporal_length, temporal_padding = pack_bit_matrix(temporal, "big")
    hybrid_packed, hybrid_length, hybrid_padding = pack_bit_matrix(hybrid, "big")
    np.savez_compressed(
        paths.npz_path,
        segment_ids=segment_ids,
        segment_start_times=starts,
        segment_end_times=ends,
        resnet_binary_digests=resnet,
        temporal_bin_indices=np.zeros((2, 18), dtype=np.uint8),
        temporal_binary_digests=temporal,
        hybrid_binary_digests=hybrid,
        resnet_packed_digests=resnet_packed,
        temporal_packed_digests=temporal_packed,
        hybrid_packed_digests=hybrid_packed,
        resnet_bit_length=np.asarray(resnet_length, dtype=np.int64),
        temporal_bit_length=np.asarray(temporal_length, dtype=np.int64),
        hybrid_bit_length=np.asarray(hybrid_length, dtype=np.int64),
    )
    manifest = {
        "video_id": video_id,
        "normalization_calibration_id": "DEV_NORMALIZATION_V1",
        "quantization_id": "DEV_QUANTIZATION_V1",
        "development_only": True,
        "development_warning": QUANTIZATION_WARNING,
        "source_normalized_feature_sha256": "n" * 64,
        "source_calibration_sha256": "c" * 64,
        "quantization_artifact_sha256": "q" * 64,
        "segment_count": 2,
        "segments": [
            {"segment_id": 0, "start_time_seconds": 0.0, "end_time_seconds": 5.0},
            {"segment_id": 1, "start_time_seconds": 5.0, "end_time_seconds": 10.0},
        ],
        "digest_dimensions": DIGEST_LENGTHS,
        "stream_boundaries": STREAM_BOUNDARIES,
        "bit_order": "big",
        "padding_bit_counts": {
            "resnet": resnet_padding,
            "temporal": temporal_padding,
            "hybrid": hybrid_padding,
        },
        "pack_unpack_round_trip": True,
        "output_npz_path": str(paths.npz_path.resolve()),
        "npz_sha256": sha256_file(paths.npz_path),
        "warnings": [QUANTIZATION_WARNING],
        "failures": [],
    }
    paths.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_authentication_record_save_load_verify_cache_and_inspect(tmp_path: Path) -> None:
    """Authentication records should save atomically, verify, and cache safely."""

    _write_digest_fixture(tmp_path)
    key_info = _key_info()
    stored = protect_digest_record(
        video_id="T001",
        digest_root=tmp_path / "digests",
        authentication_record_root=tmp_path / "authentication_records",
        key_info=key_info,
        overwrite=True,
    )
    assert not stored.cache_reused
    assert stored.record_file_sha256 == sha256_file(stored.paths.record_path)
    assert not list(stored.paths.output_dir.glob("*.tmp"))
    loaded = load_authentication_record(stored.paths.record_path)
    result = verify_authentication_record(loaded, key_info)
    assert result.record_valid
    assert result.hmac_valid
    assert result.payload_checksum_valid
    assert result.key_fingerprint_match
    assert AUTHENTICATION_RECORD_WARNING in loaded["authentication"]["warnings"]
    assert QUANTIZATION_WARNING in loaded["authentication"]["warnings"]
    secret_hex = key_info.key.hex()
    assert secret_hex not in stored.paths.record_path.read_text(encoding="utf-8")

    summary = inspect_authentication_record(stored.paths.record_path)
    assert summary["video_id"] == "T001"
    assert summary["record_file_sha256"] == sha256_file(stored.paths.record_path)

    cached = protect_digest_record(
        video_id="T001",
        digest_root=tmp_path / "digests",
        authentication_record_root=tmp_path / "authentication_records",
        key_info=key_info,
        overwrite=False,
    )
    assert cached.cache_reused


def test_authentication_record_cache_invalidation_and_digest_checksum_validation(tmp_path: Path) -> None:
    """Changed digest inputs should invalidate an existing authentication record."""

    _write_digest_fixture(tmp_path)
    key_info = _key_info()
    stored = protect_digest_record(
        "T001",
        tmp_path / "digests",
        tmp_path / "authentication_records",
        key_info,
        overwrite=True,
    )
    digest_manifest = digest_output_paths(tmp_path / "digests", "T001").manifest_path
    manifest = json.loads(digest_manifest.read_text(encoding="utf-8"))
    manifest["extra_cache_breaker"] = "changed"
    digest_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ExistingOutputError, match="cache metadata"):
        protect_digest_record(
            "T001",
            tmp_path / "digests",
            tmp_path / "authentication_records",
            key_info,
            overwrite=False,
        )

    manifest["npz_sha256"] = "bad"
    digest_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CanonicalizationError, match="checksum mismatch"):
        build_canonical_payload_from_digest_files("T001", tmp_path / "digests")
    assert stored.paths.record_path.exists()


def test_payload_excludes_absolute_paths_and_formatting_changes_verify(tmp_path: Path) -> None:
    """Authenticated payload avoids machine-specific paths and ignores outer JSON formatting."""

    _write_digest_fixture(tmp_path)
    key_info = _key_info()
    stored = protect_digest_record("T001", tmp_path / "digests", tmp_path / "authentication_records", key_info)
    payload_text = json.dumps(stored.record["payload"], sort_keys=True)
    assert str(tmp_path) not in payload_text

    record = load_authentication_record(stored.paths.record_path)
    stored.paths.record_path.write_text(json.dumps(record, indent=4), encoding="utf-8")
    assert verify_authentication_record_file(stored.paths.record_path, key_info).record_valid


def test_modified_payload_tag_schema_and_wrong_key_fail(tmp_path: Path) -> None:
    """Payload, tag, schema, and key changes should be detected."""

    _write_digest_fixture(tmp_path)
    key_info = _key_info()
    stored = protect_digest_record("T001", tmp_path / "digests", tmp_path / "authentication_records", key_info)
    record = load_authentication_record(stored.paths.record_path)
    wrong_key = _key_info(2)
    assert not verify_authentication_record(record, wrong_key).record_valid

    modified_payload = json.loads(json.dumps(record))
    modified_payload["payload"]["segments"][0]["resnet_packed_digest_hex"] = "01"
    assert not verify_authentication_record(modified_payload, key_info).record_valid

    modified_tag = json.loads(json.dumps(record))
    modified_tag["authentication"]["tag_hex"] = _flip_hex_character(
        modified_tag["authentication"]["tag_hex"]
    )
    assert not verify_authentication_record(modified_tag, key_info).record_valid

    missing_schema = json.loads(json.dumps(record))
    del missing_schema["authentication"]["record_schema_version"]
    assert not verify_authentication_record(missing_schema, key_info).schema_valid


def test_save_authentication_record_refuses_overwrite(tmp_path: Path) -> None:
    """Direct record saves should refuse accidental overwrites."""

    paths = authentication_record_paths(tmp_path / "records", "T001")
    record = {"payload": {"schema_version": 1}, "authentication": {"record_schema_version": 1}}
    save_authentication_record(record, paths, overwrite=True)
    with pytest.raises(ExistingOutputError):
        save_authentication_record(record, paths, overwrite=False)
