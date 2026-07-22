"""Tests for HMAC-SHA-256 key handling and tag verification."""

from __future__ import annotations

from pathlib import Path
import stat

import pytest

from src.authentication.hmac_auth import (
    HMACAuthenticationError,
    compute_hmac_sha256,
    compute_hmac_sha256_hex,
    decode_hex_key,
    generate_hmac_key_file,
    key_fingerprint,
    load_hmac_key,
    load_hmac_key_from_environment,
    load_hmac_key_from_file,
    verify_hmac_sha256_hex,
)


def test_valid_and_long_hex_keys_decode() -> None:
    """Valid keys of at least 32 bytes should decode."""

    assert decode_hex_key("aa" * 32) == bytes.fromhex("aa" * 32)
    assert len(decode_hex_key("bb" * 64)) == 64


@pytest.mark.parametrize("value", ["", "aa", "abc", "zz" * 32, "aa aa"])
def test_invalid_hex_keys_are_rejected(value: str) -> None:
    """Empty, short, odd-length, and malformed keys should fail."""

    with pytest.raises(HMACAuthenticationError):
        decode_hex_key(value)


def test_key_file_loading_environment_loading_and_fingerprint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Key file and environment loading should expose only non-secret metadata."""

    key_hex = "01" * 32
    key_file = tmp_path / "DEV_HMAC_KEY_V1.hex"
    key_file.write_text(key_hex + "\n", encoding="utf-8")
    from_file = load_hmac_key_from_file(key_file, key_id="DEV_HMAC_KEY_V1")
    assert from_file.key == bytes.fromhex(key_hex)
    assert from_file.key_id == "DEV_HMAC_KEY_V1"
    assert from_file.key_fingerprint == key_fingerprint(from_file.key)
    assert key_hex not in str(from_file.public_summary())

    monkeypatch.setenv("VIDEO_AUTH_HMAC_KEY_HEX", key_hex)
    from_env = load_hmac_key_from_environment(key_id="ENV_KEY")
    assert from_env.key == from_file.key
    assert from_env.source_type == "environment"
    assert load_hmac_key(key_file=key_file).source_type == "key_file"


def test_generate_hmac_key_file_permissions_and_no_overwrite(tmp_path: Path) -> None:
    """Generated key files should be private where chmod is supported."""

    key_file = tmp_path / "secrets" / "DEV_HMAC_KEY_V1.hex"
    info = generate_hmac_key_file(key_file, key_id="DEV_HMAC_KEY_V1")
    assert key_file.exists()
    assert info.key_length_bytes == 32
    assert len(key_file.read_text(encoding="utf-8").strip()) == 64
    mode = stat.S_IMODE(key_file.stat().st_mode)
    assert mode == 0o600
    with pytest.raises(HMACAuthenticationError, match="already exists"):
        generate_hmac_key_file(key_file, key_id="DEV_HMAC_KEY_V1")


def test_hmac_generation_and_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    """HMAC tags are deterministic and verification uses compare_digest."""

    key = bytes.fromhex("11" * 32)
    other_key = bytes.fromhex("22" * 32)
    payload = b'{"stable":true}'
    tag = compute_hmac_sha256_hex(key, payload)
    assert len(compute_hmac_sha256(key, payload)) == 32
    assert len(tag) == 64
    assert tag == compute_hmac_sha256_hex(key, payload)
    assert tag != compute_hmac_sha256_hex(other_key, payload)
    assert verify_hmac_sha256_hex(key, payload, tag)
    assert not verify_hmac_sha256_hex(other_key, payload, tag)
    assert not verify_hmac_sha256_hex(key, b'{"stable":false}', tag)

    called = {"value": False}

    def fake_compare_digest(left: bytes, right: bytes) -> bool:
        called["value"] = True
        return left == right

    monkeypatch.setattr("src.authentication.hmac_auth.hmac.compare_digest", fake_compare_digest)
    assert verify_hmac_sha256_hex(key, payload, tag)
    assert called["value"]
