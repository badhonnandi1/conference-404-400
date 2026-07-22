"""HMAC-SHA-256 key handling, tag generation, and constant-time verification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import os
from pathlib import Path
import re
import secrets


DEFAULT_ALGORITHM = "HMAC-SHA-256"
DEFAULT_DIGEST = "sha256"
DEFAULT_KEY_ENVIRONMENT_VARIABLE = "VIDEO_AUTH_HMAC_KEY_HEX"
DEFAULT_MINIMUM_KEY_BYTES = 32


class HMACAuthenticationError(RuntimeError):
    """Raised when HMAC generation, verification, or key handling fails."""


@dataclass(frozen=True)
class HMACKeyInfo:
    """Loaded HMAC key material plus non-secret identifying metadata."""

    key: bytes
    key_id: str
    source_type: str
    key_length_bytes: int
    key_fingerprint: str

    def public_summary(self) -> dict[str, str | int]:
        """Return non-secret key metadata safe for manifests or CLI output."""

        return {
            "key_id": self.key_id,
            "key_source_type": self.source_type,
            "key_length_bytes": self.key_length_bytes,
            "key_fingerprint": self.key_fingerprint,
        }


def key_fingerprint(key: bytes) -> str:
    """Return a non-secret truncated SHA-256 fingerprint for key identification."""

    return hashlib.sha256(key).hexdigest()[:16]


def decode_hex_key(hex_value: str, minimum_key_bytes: int = DEFAULT_MINIMUM_KEY_BYTES) -> bytes:
    """Decode and validate a hexadecimal HMAC key."""

    cleaned = hex_value.strip()
    if not cleaned:
        raise HMACAuthenticationError("HMAC key is empty.")
    if len(cleaned) % 2:
        raise HMACAuthenticationError("HMAC key hex must contain an even number of characters.")
    if re.fullmatch(r"[0-9a-fA-F]+", cleaned) is None:
        raise HMACAuthenticationError("HMAC key contains non-hexadecimal characters.")
    key = bytes.fromhex(cleaned)
    if len(key) < minimum_key_bytes:
        raise HMACAuthenticationError(
            f"HMAC key must contain at least {minimum_key_bytes} bytes; got {len(key)} bytes."
        )
    return key


def load_hmac_key_from_file(
    path: str | Path,
    key_id: str | None = None,
    minimum_key_bytes: int = DEFAULT_MINIMUM_KEY_BYTES,
) -> HMACKeyInfo:
    """Load a hex-encoded HMAC key from an explicit local key file."""

    key_path = Path(path).expanduser()
    if not key_path.exists():
        raise HMACAuthenticationError(f"HMAC key file not found: {key_path}")
    hex_value = key_path.read_text(encoding="utf-8")
    key = decode_hex_key(hex_value, minimum_key_bytes=minimum_key_bytes)
    return HMACKeyInfo(
        key=key,
        key_id=key_id or key_path.stem,
        source_type="key_file",
        key_length_bytes=len(key),
        key_fingerprint=key_fingerprint(key),
    )


def load_hmac_key_from_environment(
    environment_variable: str = DEFAULT_KEY_ENVIRONMENT_VARIABLE,
    key_id: str | None = None,
    minimum_key_bytes: int = DEFAULT_MINIMUM_KEY_BYTES,
) -> HMACKeyInfo:
    """Load a hex-encoded HMAC key from an environment variable."""

    value = os.environ.get(environment_variable)
    if value is None:
        raise HMACAuthenticationError(f"HMAC key environment variable is not set: {environment_variable}")
    key = decode_hex_key(value, minimum_key_bytes=minimum_key_bytes)
    return HMACKeyInfo(
        key=key,
        key_id=key_id or environment_variable,
        source_type="environment",
        key_length_bytes=len(key),
        key_fingerprint=key_fingerprint(key),
    )


def load_hmac_key(
    key_file: str | Path | None = None,
    key_id: str | None = None,
    environment_variable: str = DEFAULT_KEY_ENVIRONMENT_VARIABLE,
    minimum_key_bytes: int = DEFAULT_MINIMUM_KEY_BYTES,
) -> HMACKeyInfo:
    """Load an HMAC key from a key file or, if omitted, from the environment."""

    if key_file is not None:
        return load_hmac_key_from_file(key_file, key_id=key_id, minimum_key_bytes=minimum_key_bytes)
    return load_hmac_key_from_environment(
        environment_variable=environment_variable,
        key_id=key_id,
        minimum_key_bytes=minimum_key_bytes,
    )


def generate_hmac_key_file(
    output_path: str | Path,
    key_id: str,
    key_bytes: int = DEFAULT_MINIMUM_KEY_BYTES,
    overwrite: bool = False,
) -> HMACKeyInfo:
    """Generate and store a cryptographically random hex-encoded HMAC key."""

    if key_bytes < DEFAULT_MINIMUM_KEY_BYTES:
        raise HMACAuthenticationError(f"Generated HMAC keys must be at least {DEFAULT_MINIMUM_KEY_BYTES} bytes.")
    output = Path(output_path).expanduser()
    if output.exists() and not overwrite:
        raise HMACAuthenticationError(f"HMAC key file already exists: {output}. Use --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(key_bytes)
    output.write_text(key.hex() + "\n", encoding="utf-8")
    try:
        os.chmod(output, 0o600)
    except OSError:
        pass
    return HMACKeyInfo(
        key=key,
        key_id=key_id,
        source_type="generated_key_file",
        key_length_bytes=len(key),
        key_fingerprint=key_fingerprint(key),
    )


def compute_hmac_sha256(key: bytes, payload_bytes: bytes) -> bytes:
    """Return the raw HMAC-SHA-256 tag bytes for canonical payload bytes."""

    return hmac.new(key, payload_bytes, hashlib.sha256).digest()


def compute_hmac_sha256_hex(key: bytes, payload_bytes: bytes) -> str:
    """Return the lowercase hexadecimal HMAC-SHA-256 tag for canonical payload bytes."""

    return compute_hmac_sha256(key, payload_bytes).hex()


def verify_hmac_sha256_hex(key: bytes, payload_bytes: bytes, expected_tag_hex: str) -> bool:
    """Verify a hex HMAC-SHA-256 tag using constant-time comparison."""

    try:
        expected = bytes.fromhex(expected_tag_hex)
    except ValueError:
        return False
    calculated = compute_hmac_sha256(key, payload_bytes)
    return hmac.compare_digest(calculated, expected)
