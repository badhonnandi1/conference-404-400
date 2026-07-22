"""Tests for Phase 7 segment-level digest comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.authentication.auth_record_storage import protect_digest_record
from src.authentication.digest import pack_bit_matrix
from src.authentication.digest_storage import digest_output_paths
from src.authentication.hmac_auth import HMACKeyInfo, key_fingerprint
from src.authentication.quantization import DIGEST_LENGTHS, QUANTIZATION_WARNING, STREAM_BOUNDARIES
from src.features.feature_storage import sha256_file
from src.verification.comparison import (
    ATTRIBUTION_CODES,
    ComparisonConfig,
    DiagnosticWeights,
    DigestComparisonError,
    balanced_diagnostic_score,
    compare_digests,
    relative_stream_attribution,
)


def _key_info() -> HMACKeyInfo:
    key = b"\x07" * 32
    return HMACKeyInfo(
        key=key,
        key_id="TEST_KEY",
        source_type="test",
        key_length_bytes=len(key),
        key_fingerprint=key_fingerprint(key),
    )


def _bits(segment_count: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((segment_count, DIGEST_LENGTHS["resnet"]), dtype=np.uint8),
        np.zeros((segment_count, DIGEST_LENGTHS["temporal"]), dtype=np.uint8),
    )


def write_digest_fixture(
    root: Path,
    video_id: str,
    segment_ids: list[int] | None = None,
    resnet_bits: np.ndarray | None = None,
    temporal_bits: np.ndarray | None = None,
    starts: list[float] | None = None,
    ends: list[float] | None = None,
    normalization_id: str = "DEV_NORMALIZATION_V1",
    quantization_id: str = "DEV_QUANTIZATION_V1",
) -> None:
    """Write a deterministic Phase 5-style digest fixture."""

    segment_ids = segment_ids or [0, 1]
    segment_count = len(segment_ids)
    default_resnet, default_temporal = _bits(segment_count)
    resnet_bits = default_resnet if resnet_bits is None else np.asarray(resnet_bits, dtype=np.uint8)
    temporal_bits = default_temporal if temporal_bits is None else np.asarray(temporal_bits, dtype=np.uint8)
    hybrid_bits = np.concatenate([resnet_bits, temporal_bits], axis=1)
    starts = starts or [float(segment_id * 5) for segment_id in segment_ids]
    ends = ends or [start + 5.0 for start in starts]
    paths = digest_output_paths(root / "digests", video_id)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    resnet_packed, resnet_length, resnet_padding = pack_bit_matrix(resnet_bits, "big")
    temporal_packed, temporal_length, temporal_padding = pack_bit_matrix(temporal_bits, "big")
    hybrid_packed, hybrid_length, hybrid_padding = pack_bit_matrix(hybrid_bits, "big")
    np.savez_compressed(
        paths.npz_path,
        segment_ids=np.asarray(segment_ids, dtype=np.int64),
        segment_start_times=np.asarray(starts, dtype=np.float64),
        segment_end_times=np.asarray(ends, dtype=np.float64),
        resnet_binary_digests=resnet_bits,
        temporal_bin_indices=np.zeros((segment_count, 18), dtype=np.uint8),
        temporal_binary_digests=temporal_bits,
        hybrid_binary_digests=hybrid_bits,
        resnet_packed_digests=resnet_packed,
        temporal_packed_digests=temporal_packed,
        hybrid_packed_digests=hybrid_packed,
        resnet_bit_length=np.asarray(resnet_length, dtype=np.int64),
        temporal_bit_length=np.asarray(temporal_length, dtype=np.int64),
        hybrid_bit_length=np.asarray(hybrid_length, dtype=np.int64),
    )
    manifest = {
        "video_id": video_id,
        "normalization_calibration_id": normalization_id,
        "quantization_id": quantization_id,
        "development_only": True,
        "development_warning": QUANTIZATION_WARNING,
        "segment_count": segment_count,
        "segments": [
            {"segment_id": int(segment_id), "start_time_seconds": float(start), "end_time_seconds": float(end)}
            for segment_id, start, end in zip(segment_ids, starts, ends, strict=True)
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


def protect_reference(root: Path, video_id: str = "REF") -> None:
    protect_digest_record(
        video_id=video_id,
        digest_root=root / "digests",
        authentication_record_root=root / "authentication_records",
        key_info=_key_info(),
        overwrite=True,
    )


def test_balanced_score_weight_validation_and_attribution() -> None:
    assert balanced_diagnostic_score(0.2, 0.6, DiagnosticWeights(0.25, 0.75)) == pytest.approx(0.5)
    with pytest.raises(DigestComparisonError, match="non-negative"):
        DiagnosticWeights(-0.1, 1.1).validate()
    with pytest.raises(DigestComparisonError, match="sum"):
        DiagnosticWeights(0.4, 0.4).validate()
    assert balanced_diagnostic_score(0.0, 0.0) == 0.0
    assert relative_stream_attribution(0.0, 0.0)[0] == "no_difference"
    assert relative_stream_attribution(0.2, 0.1)[0] == "resnet_dominant"
    assert relative_stream_attribution(0.1, 0.2)[0] == "temporal_dominant"
    assert relative_stream_attribution(0.1, 0.1000000001, tie_tolerance=1e-6)[0] == "approximately_equal"


def test_self_comparison_zero_distances_and_no_classification_fields(tmp_path: Path) -> None:
    write_digest_fixture(tmp_path, "REF")
    protect_reference(tmp_path, "REF")
    result = compare_digests("REF", "REF", tmp_path / "authentication_records", tmp_path / "digests", _key_info(), ComparisonConfig())
    assert result.video_summary["matched_segment_count"] == 2
    assert result.video_summary["maximum_resnet_normalized_distance"] == 0.0
    assert result.video_summary["maximum_temporal_normalized_distance"] == 0.0
    assert result.video_summary["maximum_balanced_diagnostic_score"] == 0.0
    assert {segment.attribution_label for segment in result.segment_results} == {"no_difference"}
    summary_text = json.dumps(result.video_summary)
    assert "authentic" not in summary_text
    assert "tampered" not in summary_text
    assert "accepted" not in summary_text
    assert "rejected" not in summary_text


def test_known_resnet_temporal_and_multiple_bit_changes(tmp_path: Path) -> None:
    write_digest_fixture(tmp_path, "REF")
    protect_reference(tmp_path, "REF")
    resnet, temporal = _bits(2)
    resnet[0, 0] = 1
    temporal[1, 0] = 1
    resnet[1, 0:10] = 1
    temporal[1, 1:5] = 1
    write_digest_fixture(tmp_path, "QRY", resnet_bits=resnet, temporal_bits=temporal)
    result = compare_digests("REF", "QRY", tmp_path / "authentication_records", tmp_path / "digests", _key_info(), ComparisonConfig())
    segment0, segment1 = result.segment_results
    assert segment0.resnet.raw_distance == 1
    assert segment0.resnet.normalized_distance == pytest.approx(1 / 1024)
    assert segment0.temporal.raw_distance == 0
    assert segment0.hybrid.raw_distance == 1
    assert segment0.attribution_label == "resnet_dominant"
    assert segment1.resnet.raw_distance == 10
    assert segment1.temporal.raw_distance == 5
    assert segment1.temporal.normalized_distance == pytest.approx(5 / 36)
    assert segment1.hybrid.raw_distance == 15
    assert segment1.attribution_label == "temporal_dominant"


def test_normalization_and_quantization_mismatch_rejected(tmp_path: Path) -> None:
    write_digest_fixture(tmp_path, "REF")
    protect_reference(tmp_path, "REF")
    write_digest_fixture(tmp_path, "QRY", normalization_id="OTHER")
    with pytest.raises(DigestComparisonError, match="normalization"):
        compare_digests("REF", "QRY", tmp_path / "authentication_records", tmp_path / "digests", _key_info(), ComparisonConfig())
    write_digest_fixture(tmp_path, "QRY", quantization_id="OTHER")
    with pytest.raises(DigestComparisonError, match="quantization"):
        compare_digests("REF", "QRY", tmp_path / "authentication_records", tmp_path / "digests", _key_info(), ComparisonConfig())


def test_missing_segments_do_not_cause_row_shift_and_reordered_segments_work(tmp_path: Path) -> None:
    write_digest_fixture(tmp_path, "REF")
    protect_reference(tmp_path, "REF")
    write_digest_fixture(tmp_path, "QRY", segment_ids=[1])
    missing = compare_digests("REF", "QRY", tmp_path / "authentication_records", tmp_path / "digests", _key_info(), ComparisonConfig())
    assert missing.alignment.missing_segment_count == 1
    assert missing.segment_results[0].segment_id == 1
    assert missing.segment_results[0].resnet.raw_distance == 0

    resnet, temporal = _bits(2)
    write_digest_fixture(tmp_path, "QRY", segment_ids=[1, 0], resnet_bits=resnet[[1, 0]], temporal_bits=temporal[[1, 0]], starts=[5.0, 0.0], ends=[10.0, 5.0])
    reordered = compare_digests("REF", "QRY", tmp_path / "authentication_records", tmp_path / "digests", _key_info(), ComparisonConfig())
    assert reordered.alignment.alignment_valid
    assert [segment.segment_id for segment in reordered.segment_results] == [0, 1]
    assert all(segment.hybrid.raw_distance == 0 for segment in reordered.segment_results)


def test_duplicate_segment_timestamp_mismatch_and_invalid_hmac(tmp_path: Path) -> None:
    write_digest_fixture(tmp_path, "REF")
    protect_reference(tmp_path, "REF")
    write_digest_fixture(tmp_path, "QRY", segment_ids=[0, 0])
    with pytest.raises(DigestComparisonError, match="duplicate"):
        compare_digests("REF", "QRY", tmp_path / "authentication_records", tmp_path / "digests", _key_info(), ComparisonConfig())

    write_digest_fixture(tmp_path, "QRY", starts=[0.002, 5.0], ends=[5.002, 10.0])
    mismatch = compare_digests("REF", "QRY", tmp_path / "authentication_records", tmp_path / "digests", _key_info(), ComparisonConfig())
    assert mismatch.alignment.timestamp_mismatch_count == 1
    assert not mismatch.video_summary["comparison_complete"]

    record_path = tmp_path / "authentication_records" / "REF" / "REF_authentication_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["authentication"]["tag_hex"] = ("0" if record["authentication"]["tag_hex"][0] != "0" else "1") + record["authentication"]["tag_hex"][1:]
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(DigestComparisonError, match="verification failed"):
        compare_digests("REF", "REF", tmp_path / "authentication_records", tmp_path / "digests", _key_info(), ComparisonConfig())
