"""Tests for digest packing and diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from src.authentication.digest import (
    calculate_clipping_statistics,
    clipping_statistics_for_stream,
    pack_bit_matrix,
    unpack_packed_bits,
)


def test_pack_round_trip_lengths_and_padding() -> None:
    """Packed ResNet, temporal, and hybrid digests round-trip exactly."""

    rng = np.random.default_rng(42)
    for bit_length, expected_bytes in [(1024, 128), (36, 5), (1060, 133)]:
        bits = rng.integers(0, 2, size=(3, bit_length), dtype=np.uint8)
        packed, stored_length, padding = pack_bit_matrix(bits, bit_order="big")
        unpacked = unpack_packed_bits(packed, stored_length, bit_order="big")

        assert packed.shape == (3, expected_bytes)
        assert stored_length == bit_length
        assert padding == expected_bytes * 8 - bit_length
        assert np.array_equal(unpacked, bits)


def test_pack_little_bit_order_round_trip() -> None:
    """Little-endian bit packing is explicit and round-trips when requested."""

    bits = np.asarray([[1, 0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)
    packed, bit_length, _ = pack_bit_matrix(bits, bit_order="little")
    assert np.array_equal(unpack_packed_bits(packed, bit_length, bit_order="little"), bits)


def test_clipping_statistics_counts_percentages_and_tolerance() -> None:
    """Clipping diagnostics count lower and upper boundary hits with tolerance."""

    values = np.asarray([[-5.0, -4.9999995, 0.0, 4.9999995, 5.0]], dtype=np.float32)
    stats = clipping_statistics_for_stream(values, clip_min=-5, clip_max=5, tolerance=1e-6)

    assert stats["total_values"] == 5
    assert stats["lower_clip_count"] == 2
    assert stats["upper_clip_count"] == 2
    assert stats["clipped_value_count"] == 4
    assert stats["clipping_percentage"] == pytest.approx(80.0)
    assert stats["per_segment_clipping_percentage"] == [pytest.approx(80.0)]


def test_combined_clipping_statistics_no_clipping() -> None:
    """No-clipping cases produce zero percentages."""

    stats = calculate_clipping_statistics(
        np.zeros((2, 4), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    )

    assert stats["resnet"]["clipping_percentage"] == 0.0
    assert stats["temporal"]["clipping_percentage"] == 0.0
    assert stats["combined"]["clipping_percentage"] == 0.0
