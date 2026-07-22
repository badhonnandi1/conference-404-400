"""Tests for robust median/IQR stream normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.features.normalization import NormalizationError, RobustNormalizer


def test_fit_calculates_median_quartiles_and_iqr() -> None:
    """Robust fitting stores per-dimension median, quartiles, and IQR."""

    values = np.asarray([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float32)
    normalizer = RobustNormalizer.fit(values)

    assert np.allclose(normalizer.q1, [1.75, 17.5])
    assert np.allclose(normalizer.median, [2.5, 25.0])
    assert np.allclose(normalizer.q3, [3.25, 32.5])
    assert np.allclose(normalizer.iqr, [1.5, 15.0])
    assert normalizer.feature_dimension == 2
    assert normalizer.n_samples == 4


def test_transform_applies_robust_scaling_without_mutation() -> None:
    """Transform uses fitted parameters and does not mutate input values."""

    values = np.asarray([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float32)
    original = values.copy()
    normalizer = RobustNormalizer.fit(values)
    transformed = normalizer.transform(values)

    assert np.allclose(values, original)
    assert np.allclose(transformed[0], [-1.0, -1.0])
    assert np.allclose(transformed[-1], [1.0, 1.0])
    assert np.all(np.isfinite(transformed))


def test_zero_iqr_handling_and_clipping() -> None:
    """Zero-IQR dimensions use epsilon and extreme values are clipped."""

    values = np.asarray([[1, 0], [1, 1], [1, 2]], dtype=np.float32)
    normalizer = RobustNormalizer.fit(values, epsilon=1e-8, clip_min=-5, clip_max=5)
    transformed = normalizer.transform(np.asarray([[1, 100]], dtype=np.float32))

    assert normalizer.zero_iqr_mask.tolist() == [True, False]
    assert transformed.shape == (1, 2)
    assert transformed[0, 0] == pytest.approx(0.0)
    assert transformed[0, 1] == pytest.approx(5.0)
    assert np.all(np.isfinite(transformed))


def test_wrong_dimension_and_nonfinite_inputs_are_rejected() -> None:
    """Transform rejects wrong dimensions and fitting rejects non-finite values."""

    normalizer = RobustNormalizer.fit(np.ones((3, 2), dtype=np.float32))
    with pytest.raises(NormalizationError, match="Expected feature dimension"):
        normalizer.transform(np.ones((1, 3), dtype=np.float32))
    with pytest.raises(NormalizationError, match="non-finite"):
        RobustNormalizer.fit(np.asarray([[1.0, np.nan]], dtype=np.float32))


def test_save_load_consistency(tmp_path: Path) -> None:
    """Saved and loaded normalizers produce identical transformations."""

    values = np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.float32)
    normalizer = RobustNormalizer.fit(values)
    path = normalizer.save(tmp_path / "normalizer.npz")
    loaded = RobustNormalizer.load(path)

    assert np.allclose(loaded.transform(values), normalizer.transform(values))
    assert loaded.epsilon == normalizer.epsilon
    assert loaded.clip_min == normalizer.clip_min
    assert loaded.clip_max == normalizer.clip_max
