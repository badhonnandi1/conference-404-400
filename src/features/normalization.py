"""Robust per-stream feature normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


NORMALIZATION_METHOD = "median_iqr"
DEFAULT_EPSILON = 1e-8
DEFAULT_CLIP_MIN = -5.0
DEFAULT_CLIP_MAX = 5.0


class NormalizationError(RuntimeError):
    """Raised when robust normalization cannot be fitted or applied."""


@dataclass(frozen=True)
class RobustNormalizer:
    """Per-dimension robust normalizer using median and interquartile range."""

    median: np.ndarray
    q1: np.ndarray
    q3: np.ndarray
    iqr: np.ndarray
    safe_scale: np.ndarray
    zero_iqr_mask: np.ndarray
    epsilon: float = DEFAULT_EPSILON
    clip_min: float = DEFAULT_CLIP_MIN
    clip_max: float = DEFAULT_CLIP_MAX
    feature_dimension: int = 0
    n_samples: int = 0
    method: str = NORMALIZATION_METHOD

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        epsilon: float = DEFAULT_EPSILON,
        clip_min: float = DEFAULT_CLIP_MIN,
        clip_max: float = DEFAULT_CLIP_MAX,
    ) -> "RobustNormalizer":
        """Fit robust normalization parameters from a 2D feature matrix."""

        matrix = _validate_input_matrix(values, "fit")
        if matrix.shape[0] == 0:
            raise NormalizationError("Cannot fit normalization with zero calibration samples.")
        if epsilon <= 0:
            raise NormalizationError("Normalization epsilon must be greater than zero.")
        if clip_min >= clip_max:
            raise NormalizationError("Normalization clip_min must be less than clip_max.")

        q1 = np.percentile(matrix, 25, axis=0).astype(np.float64)
        median = np.percentile(matrix, 50, axis=0).astype(np.float64)
        q3 = np.percentile(matrix, 75, axis=0).astype(np.float64)
        iqr = (q3 - q1).astype(np.float64)
        zero_iqr_mask = iqr <= float(epsilon)
        safe_scale = np.maximum(iqr, float(epsilon)).astype(np.float64)
        return cls(
            median=median,
            q1=q1,
            q3=q3,
            iqr=iqr,
            safe_scale=safe_scale,
            zero_iqr_mask=zero_iqr_mask,
            epsilon=float(epsilon),
            clip_min=float(clip_min),
            clip_max=float(clip_max),
            feature_dimension=int(matrix.shape[1]),
            n_samples=int(matrix.shape[0]),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply fitted robust normalization without modifying the input array."""

        self._validate_fitted()
        matrix = _validate_input_matrix(values, "transform")
        if matrix.shape[1] != self.feature_dimension:
            raise NormalizationError(
                f"Expected feature dimension {self.feature_dimension}, got {matrix.shape[1]}."
            )
        normalized = (matrix - self.median) / self.safe_scale
        clipped = np.clip(normalized, self.clip_min, self.clip_max).astype(np.float32)
        if not np.all(np.isfinite(clipped)):
            raise NormalizationError("Normalization produced non-finite values.")
        return clipped

    @classmethod
    def fit_transform(
        cls,
        values: np.ndarray,
        epsilon: float = DEFAULT_EPSILON,
        clip_min: float = DEFAULT_CLIP_MIN,
        clip_max: float = DEFAULT_CLIP_MAX,
    ) -> tuple["RobustNormalizer", np.ndarray]:
        """Fit a normalizer and return the transformed matrix."""

        normalizer = cls.fit(values, epsilon=epsilon, clip_min=clip_min, clip_max=clip_max)
        return normalizer, normalizer.transform(values)

    def save(self, path: str | Path) -> Path:
        """Save this normalizer to a standalone NPZ file."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **self.to_arrays())
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "RobustNormalizer":
        """Load a standalone normalizer NPZ file."""

        with np.load(Path(path), allow_pickle=False) as payload:
            arrays = {name: payload[name] for name in payload.files}
        return cls.from_arrays(arrays)

    def to_arrays(self, prefix: str = "") -> dict[str, np.ndarray]:
        """Return arrays suitable for storage in an NPZ file."""

        return {
            f"{prefix}median": self.median.astype(np.float64),
            f"{prefix}q1": self.q1.astype(np.float64),
            f"{prefix}q3": self.q3.astype(np.float64),
            f"{prefix}iqr": self.iqr.astype(np.float64),
            f"{prefix}safe_scale": self.safe_scale.astype(np.float64),
            f"{prefix}zero_iqr_mask": self.zero_iqr_mask.astype(bool),
            f"{prefix}epsilon": np.asarray(self.epsilon, dtype=np.float64),
            f"{prefix}clip_min": np.asarray(self.clip_min, dtype=np.float64),
            f"{prefix}clip_max": np.asarray(self.clip_max, dtype=np.float64),
            f"{prefix}feature_dimension": np.asarray(self.feature_dimension, dtype=np.int64),
            f"{prefix}n_samples": np.asarray(self.n_samples, dtype=np.int64),
            f"{prefix}method": np.asarray(self.method),
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray], prefix: str = "") -> "RobustNormalizer":
        """Build a normalizer from arrays loaded out of an NPZ file."""

        required = [
            "median",
            "q1",
            "q3",
            "iqr",
            "safe_scale",
            "zero_iqr_mask",
            "epsilon",
            "clip_min",
            "clip_max",
            "feature_dimension",
            "n_samples",
        ]
        missing = [name for name in required if f"{prefix}{name}" not in arrays]
        if missing:
            raise NormalizationError(f"Normalizer storage is missing arrays: {missing}.")
        method_value: Any = arrays.get(f"{prefix}method", np.asarray(NORMALIZATION_METHOD))
        method = str(method_value.tolist() if hasattr(method_value, "tolist") else method_value)
        normalizer = cls(
            median=np.asarray(arrays[f"{prefix}median"], dtype=np.float64),
            q1=np.asarray(arrays[f"{prefix}q1"], dtype=np.float64),
            q3=np.asarray(arrays[f"{prefix}q3"], dtype=np.float64),
            iqr=np.asarray(arrays[f"{prefix}iqr"], dtype=np.float64),
            safe_scale=np.asarray(arrays[f"{prefix}safe_scale"], dtype=np.float64),
            zero_iqr_mask=np.asarray(arrays[f"{prefix}zero_iqr_mask"], dtype=bool),
            epsilon=float(np.asarray(arrays[f"{prefix}epsilon"]).item()),
            clip_min=float(np.asarray(arrays[f"{prefix}clip_min"]).item()),
            clip_max=float(np.asarray(arrays[f"{prefix}clip_max"]).item()),
            feature_dimension=int(np.asarray(arrays[f"{prefix}feature_dimension"]).item()),
            n_samples=int(np.asarray(arrays[f"{prefix}n_samples"]).item()),
            method=method,
        )
        normalizer._validate_fitted()
        return normalizer

    def summary(self) -> dict[str, int | float | str]:
        """Return JSON-friendly parameter summary fields."""

        self._validate_fitted()
        return {
            "method": self.method,
            "epsilon": self.epsilon,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "feature_dimension": self.feature_dimension,
            "n_samples": self.n_samples,
            "zero_iqr_dimension_count": int(np.count_nonzero(self.zero_iqr_mask)),
        }

    def _validate_fitted(self) -> None:
        vectors = [
            self.median,
            self.q1,
            self.q3,
            self.iqr,
            self.safe_scale,
            self.zero_iqr_mask,
        ]
        if self.feature_dimension <= 0:
            raise NormalizationError("Normalizer is not fitted.")
        if any(vector.shape != (self.feature_dimension,) for vector in vectors):
            raise NormalizationError("Normalizer parameter dimensions are inconsistent.")
        if self.n_samples <= 0:
            raise NormalizationError("Normalizer has no calibration samples.")
        if not np.all(np.isfinite(self.median)) or not np.all(np.isfinite(self.safe_scale)):
            raise NormalizationError("Normalizer parameters contain non-finite values.")
        if np.any(self.safe_scale <= 0):
            raise NormalizationError("Normalizer safe_scale values must be greater than zero.")


def _validate_input_matrix(values: np.ndarray, operation: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise NormalizationError(f"Normalization {operation} expects a 2D feature matrix.")
    if not np.all(np.isfinite(matrix)):
        raise NormalizationError(f"Normalization {operation} input contains non-finite values.")
    return matrix.copy()
