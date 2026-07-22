"""Tests for exact Hamming-distance helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.authentication.digest import pack_bit_matrix
from src.verification.hamming import HammingDistanceError, hamming_distance, packed_hamming_distance


def test_equal_arrays_produce_zero_distance() -> None:
    bits = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    result = hamming_distance(bits, bits.copy())
    assert result.raw_distance == 0
    assert result.normalized_distance == 0.0


def test_one_bit_and_all_bits_changed() -> None:
    reference = np.asarray([0, 0, 0, 0], dtype=np.uint8)
    one_changed = np.asarray([1, 0, 0, 0], dtype=np.uint8)
    all_changed = np.asarray([1, 1, 1, 1], dtype=np.uint8)
    assert hamming_distance(reference, one_changed).raw_distance == 1
    full = hamming_distance(reference, all_changed)
    assert full.raw_distance == 4
    assert full.normalized_distance == 1.0


def test_normalized_distance_range_and_unequal_lengths() -> None:
    result = hamming_distance(np.asarray([0, 0, 1]), np.asarray([0, 1, 1]))
    assert 0.0 <= result.normalized_distance <= 1.0
    with pytest.raises(HammingDistanceError, match="equal shape"):
        hamming_distance(np.asarray([0, 1]), np.asarray([0, 1, 0]))


def test_non_binary_values_rejected() -> None:
    with pytest.raises(HammingDistanceError, match="other than 0 and 1"):
        hamming_distance(np.asarray([0, 2]), np.asarray([0, 1]))


def test_packed_and_unpacked_methods_agree_with_padding_ignored() -> None:
    reference = np.asarray([[1, 0, 1, 0, 1]], dtype=np.uint8)
    query = np.asarray([[1, 1, 1, 0, 0]], dtype=np.uint8)
    reference_packed, bit_length, _ = pack_bit_matrix(reference, "big")
    query_packed, _, _ = pack_bit_matrix(query, "big")
    unpacked = hamming_distance(reference[0], query[0])
    packed = packed_hamming_distance(reference_packed[0], query_packed[0], bit_length, "big")
    assert packed == unpacked
    assert packed.raw_distance == 2

    altered_padding = query_packed.copy()
    altered_padding[0, -1] ^= np.uint8(0b00000111)
    assert packed_hamming_distance(reference_packed[0], altered_padding[0], bit_length, "big") == unpacked
