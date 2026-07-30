from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.features.spatial_quality import (
    SPATIAL_SEGMENT_DIMENSION,
    aggregate_spatial_segment_features,
    calculate_spatial_quality_metrics,
    preprocess_spatial_frame,
)


def _textured_frame() -> np.ndarray:
    rng = np.random.default_rng(42)
    grid = np.indices((128, 128)).sum(axis=0) % 2
    values = grid.astype(np.float32) * 190.0 + 30.0 + rng.normal(0, 4, size=(128, 128))
    return np.clip(values, 0, 255).astype(np.uint8)


def test_spatial_metrics_shape_and_finiteness() -> None:
    frame = _textured_frame()

    metrics = calculate_spatial_quality_metrics(frame)

    assert metrics.shape == (5,)
    assert np.all(np.isfinite(metrics))
    assert 0.0 <= metrics[2] <= 1.0
    assert 0.0 <= metrics[3] <= 1.0


def test_stronger_gaussian_blur_reduces_sharpness_features() -> None:
    frame = _textured_frame()
    kernels = [1, 3, 5, 9, 15]
    values = []
    for kernel in kernels:
        blurred = frame if kernel == 1 else cv2.GaussianBlur(frame, (kernel, kernel), 0)
        values.append(calculate_spatial_quality_metrics(blurred))
    matrix = np.vstack(values)

    assert matrix[-1, 0] < matrix[0, 0]
    assert matrix[-1, 1] < matrix[0, 1]
    assert matrix[-1, 3] < matrix[0, 3]


def test_resize_and_compression_like_tolerance_is_smaller_than_blur() -> None:
    frame = _textured_frame()
    resized = cv2.resize(cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA), frame.shape[::-1])
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    assert ok
    jpeg = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)

    base_metrics = calculate_spatial_quality_metrics(frame)
    resized_delta = np.mean(np.abs(base_metrics - calculate_spatial_quality_metrics(resized)))
    jpeg_delta = np.mean(np.abs(base_metrics - calculate_spatial_quality_metrics(jpeg)))
    blur_delta = np.mean(np.abs(base_metrics - calculate_spatial_quality_metrics(blurred)))

    assert resized_delta < blur_delta
    assert jpeg_delta < blur_delta


def test_spatial_segment_aggregation_dimension() -> None:
    features = np.asarray(
        [
            [10.0, 20.0, 0.5, 0.25, 0.1],
            [8.0, 16.0, 0.4, 0.20, 0.08],
            [6.0, 12.0, 0.3, 0.15, 0.06],
        ],
        dtype=np.float32,
    )

    segment = aggregate_spatial_segment_features(features)

    assert segment.shape == (SPATIAL_SEGMENT_DIMENSION,)
    assert np.allclose(segment[:5], np.mean(features, axis=0))
    assert np.allclose(segment[-5:], np.median(features, axis=0))


def test_preprocess_rejects_invalid_frame_rank() -> None:
    with pytest.raises(Exception):
        preprocess_spatial_frame(np.zeros((2, 2, 2, 2), dtype=np.uint8))
