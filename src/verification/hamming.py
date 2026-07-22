"""Exact Hamming-distance helpers for binary digest arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.authentication.digest import DigestError, unpack_packed_bits


class HammingDistanceError(RuntimeError):
    """Raised when Hamming-distance inputs are invalid."""


@dataclass(frozen=True)
class HammingDistanceResult:
    """Raw and normalized Hamming distance for one bit stream."""

    raw_distance: int
    bit_length: int
    normalized_distance: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-friendly result."""

        return {
            "raw_distance": self.raw_distance,
            "bit_length": self.bit_length,
            "normalized_distance": self.normalized_distance,
        }


def validate_binary_vector(values: np.ndarray, name: str = "bit array") -> np.ndarray:
    """Return a copied one-dimensional uint8 binary vector."""

    vector = np.asarray(values, dtype=np.uint8)
    if vector.ndim != 1:
        raise HammingDistanceError(f"{name} must be one-dimensional.")
    if not np.all(np.isin(vector, [0, 1])):
        raise HammingDistanceError(f"{name} contains values other than 0 and 1.")
    return vector.copy()


def hamming_distance(reference_bits: np.ndarray, query_bits: np.ndarray) -> HammingDistanceResult:
    """Compute exact raw and normalized Hamming distance for two equal-length bit vectors."""

    reference = validate_binary_vector(reference_bits, "reference bits")
    query = validate_binary_vector(query_bits, "query bits")
    if reference.shape != query.shape:
        raise HammingDistanceError(f"Bit vectors must have equal shape, got {reference.shape} and {query.shape}.")
    bit_length = int(reference.size)
    raw = int(np.count_nonzero(reference != query))
    normalized = float(raw / bit_length) if bit_length else 0.0
    if not 0.0 <= normalized <= 1.0:
        raise HammingDistanceError("Normalized Hamming distance fell outside [0, 1].")
    return HammingDistanceResult(raw_distance=raw, bit_length=bit_length, normalized_distance=normalized)


def packed_hamming_distance(
    reference_packed: np.ndarray,
    query_packed: np.ndarray,
    bit_length: int,
    bit_order: str = "big",
) -> HammingDistanceResult:
    """Compute Hamming distance from row-level packed bytes while ignoring padding bits."""

    if bit_length < 0:
        raise HammingDistanceError("bit_length must be non-negative.")
    reference = np.asarray(reference_packed, dtype=np.uint8)
    query = np.asarray(query_packed, dtype=np.uint8)
    if reference.ndim != 1 or query.ndim != 1:
        raise HammingDistanceError("Packed Hamming distance expects one-dimensional packed byte arrays.")
    if reference.shape != query.shape:
        raise HammingDistanceError(f"Packed arrays must have equal shape, got {reference.shape} and {query.shape}.")
    try:
        reference_bits = unpack_packed_bits(reference.reshape(1, -1), bit_length, bit_order)[0]
        query_bits = unpack_packed_bits(query.reshape(1, -1), bit_length, bit_order)[0]
    except DigestError as exc:
        raise HammingDistanceError(str(exc)) from exc
    return hamming_distance(reference_bits, query_bits)
