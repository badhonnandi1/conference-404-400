"""Tests for binary and Gray-code quantization."""

from __future__ import annotations

import numpy as np
import pytest

from src.authentication.quantization import (
    HYBRID_DIGEST_LENGTH,
    RESNET_DIGEST_LENGTH,
    TEMPORAL_DIGEST_LENGTH,
    QuantizationError,
    assign_temporal_bins,
    build_hybrid_digest,
    gray_encode_temporal_bins,
    quantize_resnet_binary,
)


def test_resnet_binary_threshold_behavior_and_shape() -> None:
    """Below threshold maps to 0; exact threshold and above map to 1."""

    values = np.zeros((3, RESNET_DIGEST_LENGTH), dtype=np.float32)
    values[0, :] = -0.1
    values[1, :] = 0.0
    values[2, :] = 0.1
    thresholds = np.zeros(RESNET_DIGEST_LENGTH, dtype=np.float32)

    bits = quantize_resnet_binary(values, thresholds)

    assert bits.shape == (3, RESNET_DIGEST_LENGTH)
    assert np.all(bits[0] == 0)
    assert np.all(bits[1] == 1)
    assert np.all(bits[2] == 1)
    assert set(np.unique(bits).tolist()) == {0, 1}
    assert np.array_equal(bits, quantize_resnet_binary(values, thresholds))


def test_resnet_wrong_dimension_rejected() -> None:
    """ResNet quantization rejects incorrect feature dimensions."""

    with pytest.raises(QuantizationError, match="dimension"):
        quantize_resnet_binary(np.zeros((1, 1023), dtype=np.float32), np.zeros(RESNET_DIGEST_LENGTH))


def test_temporal_bin_boundaries_and_zero_iqr_finite() -> None:
    """Temporal bins use deterministic quartile boundary behavior."""

    q1 = np.full(18, -1.0, dtype=np.float32)
    median = np.zeros(18, dtype=np.float32)
    q3 = np.ones(18, dtype=np.float32)
    values = np.asarray(
        [
            [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] + [0.0] * 11,
        ],
        dtype=np.float32,
    )

    bins = assign_temporal_bins(values, q1, median, q3)

    assert bins.shape == (1, 18)
    assert bins[0, :7].tolist() == [0, 1, 1, 2, 2, 3, 3]
    assert set(np.unique(bins).tolist()).issubset({0, 1, 2, 3})

    zero_iqr_bins = assign_temporal_bins(np.zeros((1, 18), dtype=np.float32), median, median, median)
    assert zero_iqr_bins.shape == (1, 18)
    assert np.all(np.isfinite(zero_iqr_bins))


def test_gray_code_mapping_and_adjacency() -> None:
    """Temporal bins are encoded with two-bit Gray codes."""

    bins = np.asarray([[0, 1, 2, 3] + [0] * 14], dtype=np.uint8)
    bits = gray_encode_temporal_bins(bins)

    assert bits.shape == (1, TEMPORAL_DIGEST_LENGTH)
    assert bits[0, :8].tolist() == [0, 0, 0, 1, 1, 1, 1, 0]
    codes = bits[0, :8].reshape(4, 2)
    hamming_adjacent = [int(np.sum(codes[index] != codes[index + 1])) for index in range(3)]
    assert hamming_adjacent == [1, 1, 1]


def test_hybrid_digest_order_and_boundaries() -> None:
    """Hybrid digest concatenates ResNet bits before temporal bits."""

    resnet = np.ones((2, RESNET_DIGEST_LENGTH), dtype=np.uint8)
    temporal = np.zeros((2, TEMPORAL_DIGEST_LENGTH), dtype=np.uint8)
    hybrid = build_hybrid_digest(resnet, temporal)

    assert hybrid.shape == (2, HYBRID_DIGEST_LENGTH)
    assert np.all(hybrid[:, :RESNET_DIGEST_LENGTH] == 1)
    assert np.all(hybrid[:, RESNET_DIGEST_LENGTH:] == 0)
    assert set(np.unique(hybrid).tolist()) == {0, 1}
