"""Tests for canonical HMAC authentication payload serialization."""

from __future__ import annotations

import json
import math

import pytest

from src.authentication.canonicalization import (
    canonical_json_bytes,
    canonical_payload_sha256,
    seconds_to_microseconds,
)


def _payload() -> dict:
    return {
        "schema_version": 1,
        "video_id": "V001",
        "segment_count": 2,
        "development_only": True,
        "digest_lengths": {"resnet": 1024, "temporal": 36, "hybrid": 1060},
        "segments": [
            {
                "segment_id": 0,
                "start_time_microseconds": 0,
                "end_time_microseconds": 5_000_000,
                "resnet_packed_digest_hex": "00",
            },
            {
                "segment_id": 1,
                "start_time_microseconds": 5_000_000,
                "end_time_microseconds": 10_000_000,
                "resnet_packed_digest_hex": "ff",
            },
        ],
    }


def test_canonical_serialization_is_deterministic_and_sorted() -> None:
    """Insertion order should not affect canonical bytes."""

    payload = _payload()
    reordered = {
        "segments": payload["segments"],
        "digest_lengths": {"hybrid": 1060, "resnet": 1024, "temporal": 36},
        "development_only": True,
        "segment_count": 2,
        "video_id": "V001",
        "schema_version": 1,
    }
    assert canonical_json_bytes(payload) == canonical_json_bytes(reordered)
    assert canonical_json_bytes(payload) == canonical_json_bytes(payload)
    assert canonical_payload_sha256(payload) == canonical_payload_sha256(reordered)


def test_canonical_serialization_is_segment_order_sensitive() -> None:
    """Changing stable list order should change canonical bytes."""

    payload = _payload()
    changed = json.loads(json.dumps(payload))
    changed["segments"] = list(reversed(changed["segments"]))
    assert canonical_json_bytes(payload) != canonical_json_bytes(changed)


def test_canonical_serialization_is_digest_and_timestamp_sensitive() -> None:
    """Digest-byte or timestamp edits must change canonical bytes."""

    payload = _payload()
    digest_changed = json.loads(json.dumps(payload))
    digest_changed["segments"][0]["resnet_packed_digest_hex"] = "01"
    timestamp_changed = json.loads(json.dumps(payload))
    timestamp_changed["segments"][0]["end_time_microseconds"] += 1
    assert canonical_json_bytes(payload) != canonical_json_bytes(digest_changed)
    assert canonical_json_bytes(payload) != canonical_json_bytes(timestamp_changed)


def test_nan_and_infinity_are_rejected() -> None:
    """Canonical JSON must not allow NaN or Infinity."""

    payload = _payload()
    payload["bad"] = math.nan
    with pytest.raises(ValueError):
        canonical_json_bytes(payload)
    payload["bad"] = math.inf
    with pytest.raises(ValueError):
        canonical_json_bytes(payload)


def test_integer_timestamp_conversion_and_unicode() -> None:
    """Timestamps are converted to microseconds and Unicode survives UTF-8 serialization."""

    assert seconds_to_microseconds(0.125) == 125_000
    assert seconds_to_microseconds(5.000001) == 5_000_001
    payload = _payload()
    payload["note"] = "café"
    assert "café".encode("utf-8") in canonical_json_bytes(payload)
