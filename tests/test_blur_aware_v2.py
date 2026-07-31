from __future__ import annotations

import copy
import math

import numpy as np

from src.authentication.blur_aware_v2 import (
    V2_DIGEST_LENGTHS,
    V2_HYBRID_DIGEST_LENGTH,
    V2_SPATIAL_DIGEST_LENGTH,
    V2_TEMPORAL_DIGEST_LENGTH,
    V2QuantizationParameters,
    build_v2_authentication_record,
    build_v2_digest_bundle,
    compare_v2_digest_bundles,
    gray_encode_bins,
    spatial_blur_loss,
    stream_level_attribution,
    v2_digest_payload,
    verify_v2_authentication_record,
)
from src.authentication.hmac_auth import HMACKeyInfo, key_fingerprint
from src.authentication.quantization import GRAY_CODE_MAPPING, build_hybrid_digest
from src.features.spatial_quality import SPATIAL_SEGMENT_DIMENSION


def _params() -> V2QuantizationParameters:
    return V2QuantizationParameters(
        quantization_id="Q2",
        normalization_id="N2",
        resnet_thresholds=np.zeros(1024, dtype=np.float64),
        temporal_q1_thresholds=np.full(18, -1.0),
        temporal_median_thresholds=np.zeros(18),
        temporal_q3_thresholds=np.ones(18),
        spatial_q1_thresholds=np.full(SPATIAL_SEGMENT_DIMENSION, -1.0),
        spatial_median_thresholds=np.zeros(SPATIAL_SEGMENT_DIMENSION),
        spatial_q3_thresholds=np.ones(SPATIAL_SEGMENT_DIMENSION),
        gray_code_table=np.asarray([GRAY_CODE_MAPPING[index] for index in range(4)], dtype=np.uint8),
    )


def _bundle(video_id: str, offset: float = 0.0):
    segment_ids = np.asarray([0, 1], dtype=np.int64)
    starts = np.asarray([0.0, 5.0])
    ends = np.asarray([5.0, 10.0])
    resnet = np.full((2, 1024), offset, dtype=np.float32)
    temporal = np.full((2, 18), offset, dtype=np.float32)
    spatial = np.full((2, SPATIAL_SEGMENT_DIMENSION), offset, dtype=np.float32)
    return build_v2_digest_bundle(
        video_id=video_id,
        segment_ids=segment_ids,
        segment_start_times=starts,
        segment_end_times=ends,
        resnet_normalized_features=resnet,
        temporal_normalized_features=temporal,
        spatial_normalized_features=spatial,
        parameters=_params(),
    )


def test_v2_digest_lengths_and_pack_unpack_round_trip() -> None:
    bundle = _bundle("VID")

    assert bundle.resnet_binary_digests.shape == (2, 1024)
    assert bundle.temporal_binary_digests.shape == (2, V2_TEMPORAL_DIGEST_LENGTH)
    assert bundle.spatial_binary_digests.shape == (2, V2_SPATIAL_DIGEST_LENGTH)
    assert bundle.hybrid_binary_digests.shape == (2, V2_HYBRID_DIGEST_LENGTH)
    assert V2_DIGEST_LENGTHS == {"resnet": 1024, "temporal": 36, "spatial": 50, "hybrid": 1110}
    assert bundle.validate_round_trips()


def test_generic_gray_code_adjacency_for_spatial_bins() -> None:
    table = np.asarray([GRAY_CODE_MAPPING[index] for index in range(4)], dtype=np.uint8)
    bits = gray_encode_bins(np.asarray([[0, 1, 2, 3]], dtype=np.uint8), table)

    assert bits.tolist() == [[0, 0, 0, 1, 1, 1, 1, 0]]


def test_v1_hybrid_compatibility_still_concatenates_two_streams() -> None:
    resnet = np.ones((1, 1024), dtype=np.uint8)
    temporal = np.zeros((1, 36), dtype=np.uint8)

    hybrid = build_hybrid_digest(resnet, temporal)

    assert hybrid.shape == (1, 1060)
    assert hybrid.shape[1] != V2_HYBRID_DIGEST_LENGTH


def test_v2_hybrid_distance_invariant() -> None:
    reference = _bundle("REF", offset=0.0)
    query = _bundle("QUERY", offset=2.0)
    ref_spatial = np.ones((2, SPATIAL_SEGMENT_DIMENSION), dtype=np.float32)
    query_spatial = ref_spatial * 0.5

    rows = compare_v2_digest_bundles(reference, query, ref_spatial, query_spatial)

    assert rows
    for row in rows:
        assert row.hybrid_raw_distance == row.resnet_raw_distance + row.temporal_raw_distance + row.spatial_raw_distance


def test_blur_loss_direction_and_attribution() -> None:
    reference = np.ones((2, SPATIAL_SEGMENT_DIMENSION), dtype=np.float32)
    blurred = reference * 0.25
    sharpened = reference * 1.25

    assert np.all(spatial_blur_loss(reference, blurred) > 0)
    assert np.all(spatial_blur_loss(reference, sharpened) == 0)
    assert stream_level_attribution(0.01, 0.01, 0.02, 0.8) == "blur_loss_dominant"


def test_v2_hmac_record_verifies_and_detects_payload_changes() -> None:
    bundle = _bundle("VID", offset=0.0)
    key = bytes.fromhex("12" * 32)
    key_info = HMACKeyInfo(
        key=key,
        key_id="TEST_KEY",
        source_type="test",
        key_length_bytes=len(key),
        key_fingerprint=key_fingerprint(key),
    )
    payload = v2_digest_payload(
        bundle=bundle,
        normalization_id="N2",
        quantization_id="Q2",
        source_video_sha256="abc",
    )

    record = build_v2_authentication_record(payload, key_info)

    assert verify_v2_authentication_record(record, key_info)["record_valid"]
    record["payload"]["video_id"] = "OTHER"
    assert not verify_v2_authentication_record(record, key_info)["record_valid"]


def test_v2_hmac_rejects_reference_integrity_and_malformed_record_changes() -> None:
    bundle = _bundle("VID", offset=0.0)
    key = bytes.fromhex("12" * 32)
    key_info = HMACKeyInfo(
        key=key,
        key_id="TEST_KEY",
        source_type="test",
        key_length_bytes=len(key),
        key_fingerprint=key_fingerprint(key),
    )
    wrong_key = bytes.fromhex("34" * 32)
    wrong_key_info = HMACKeyInfo(
        key=wrong_key,
        key_id="WRONG_KEY",
        source_type="test",
        key_length_bytes=len(wrong_key),
        key_fingerprint=key_fingerprint(wrong_key),
    )
    payload = v2_digest_payload(
        bundle=bundle,
        normalization_id="N2",
        quantization_id="Q2",
        source_video_sha256="abc",
    )
    record = build_v2_authentication_record(payload, key_info)

    assert not verify_v2_authentication_record(record, wrong_key_info)["record_valid"]

    modified_tag = copy.deepcopy(record)
    modified_tag["authentication"]["tag_hex"] = "00" * 32
    assert not verify_v2_authentication_record(modified_tag, key_info)["record_valid"]

    modified_timestamp = copy.deepcopy(record)
    modified_timestamp["payload"]["segments"][0]["end_time_microseconds"] += 1
    assert not verify_v2_authentication_record(modified_timestamp, key_info)["record_valid"]

    reordered_segments = copy.deepcopy(record)
    reordered_segments["payload"]["segments"].reverse()
    assert not verify_v2_authentication_record(reordered_segments, key_info)["record_valid"]

    invalid_schema = copy.deepcopy(record)
    invalid_schema["payload"]["schema_version"] = 999
    assert not verify_v2_authentication_record(invalid_schema, key_info)["record_valid"]

    malformed_payload = copy.deepcopy(record)
    malformed_payload["payload"]["non_finite"] = math.nan
    result = verify_v2_authentication_record(malformed_payload, key_info)
    assert not result["record_valid"]
    assert "malformed canonical payload" in result["failure_reason"]
