"""Development binary quantization for normalized segment features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.features.alignment import RESNET_SEGMENT_DIMENSION, TEMPORAL_SEGMENT_DIMENSION
from src.features.normalization_storage import LoadedNormalizationArtifact


DEFAULT_QUANTIZATION_ID = "DEV_QUANTIZATION_V1"
DEFAULT_QUANTIZATION_VERSION = "dev_quantizer_v1"
DEFAULT_BIT_ORDER = "big"
RESNET_DIGEST_LENGTH = RESNET_SEGMENT_DIMENSION
TEMPORAL_BITS_PER_FEATURE = 2
TEMPORAL_DIGEST_LENGTH = TEMPORAL_SEGMENT_DIMENSION * TEMPORAL_BITS_PER_FEATURE
HYBRID_DIGEST_LENGTH = RESNET_DIGEST_LENGTH + TEMPORAL_DIGEST_LENGTH
GRAY_CODE_MAPPING = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 1),
    3: (1, 0),
}
QUANTIZATION_WARNING = (
    "This quantization artifact depends on a normalization model fitted using only "
    "three original development videos. It is intended for pipeline validation and "
    "must be regenerated using the final calibration split before experimental evaluation."
)
STREAM_BOUNDARIES = {
    "resnet": {"start": 0, "end_exclusive": RESNET_DIGEST_LENGTH},
    "temporal": {"start": RESNET_DIGEST_LENGTH, "end_exclusive": HYBRID_DIGEST_LENGTH},
}
DIGEST_LENGTHS = {
    "resnet": RESNET_DIGEST_LENGTH,
    "temporal": TEMPORAL_DIGEST_LENGTH,
    "hybrid": HYBRID_DIGEST_LENGTH,
}


class QuantizationError(RuntimeError):
    """Raised when quantization parameters or inputs are invalid."""


@dataclass(frozen=True)
class QuantizationParameters:
    """Fitted development quantization thresholds and metadata."""

    quantization_id: str
    version: str
    normalization_id: str
    normalization_npz_sha256: str
    resnet_thresholds: np.ndarray
    temporal_q1_thresholds: np.ndarray
    temporal_median_thresholds: np.ndarray
    temporal_q3_thresholds: np.ndarray
    temporal_gray_code_table: np.ndarray
    bit_order: str = DEFAULT_BIT_ORDER
    status: str = "development"
    development_only: bool = True

    def validate(self) -> None:
        """Validate parameter dimensions and finite thresholds."""

        if self.resnet_thresholds.shape != (RESNET_DIGEST_LENGTH,):
            raise QuantizationError(
                f"ResNet thresholds must have shape ({RESNET_DIGEST_LENGTH},), got {self.resnet_thresholds.shape}."
            )
        for name, values in (
            ("temporal_q1_thresholds", self.temporal_q1_thresholds),
            ("temporal_median_thresholds", self.temporal_median_thresholds),
            ("temporal_q3_thresholds", self.temporal_q3_thresholds),
        ):
            if values.shape != (TEMPORAL_SEGMENT_DIMENSION,):
                raise QuantizationError(
                    f"{name} must have shape ({TEMPORAL_SEGMENT_DIMENSION},), got {values.shape}."
                )
            if not np.all(np.isfinite(values)):
                raise QuantizationError(f"{name} contains non-finite values.")
        if not np.all(np.isfinite(self.resnet_thresholds)):
            raise QuantizationError("ResNet thresholds contain non-finite values.")
        if self.temporal_gray_code_table.shape != (4, 2):
            raise QuantizationError("Temporal Gray-code table must have shape (4, 2).")
        if not np.all(np.isin(self.temporal_gray_code_table, [0, 1])):
            raise QuantizationError("Temporal Gray-code table must contain only 0 and 1.")
        if self.bit_order not in {"big", "little"}:
            raise QuantizationError("Bit order must be 'big' or 'little'.")

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Return NPZ arrays for storage."""

        self.validate()
        boundaries = np.asarray(
            [
                STREAM_BOUNDARIES["resnet"]["start"],
                STREAM_BOUNDARIES["resnet"]["end_exclusive"],
                STREAM_BOUNDARIES["temporal"]["start"],
                STREAM_BOUNDARIES["temporal"]["end_exclusive"],
            ],
            dtype=np.int64,
        )
        lengths = np.asarray(
            [
                DIGEST_LENGTHS["resnet"],
                DIGEST_LENGTHS["temporal"],
                DIGEST_LENGTHS["hybrid"],
            ],
            dtype=np.int64,
        )
        return {
            "resnet_thresholds": self.resnet_thresholds.astype(np.float64),
            "temporal_q1_thresholds": self.temporal_q1_thresholds.astype(np.float64),
            "temporal_median_thresholds": self.temporal_median_thresholds.astype(np.float64),
            "temporal_q3_thresholds": self.temporal_q3_thresholds.astype(np.float64),
            "temporal_gray_code_table": self.temporal_gray_code_table.astype(np.uint8),
            "stream_boundaries": boundaries,
            "digest_lengths": lengths,
        }

    @classmethod
    def from_arrays(
        cls,
        arrays: dict[str, np.ndarray],
        quantization_id: str,
        version: str,
        normalization_id: str,
        normalization_npz_sha256: str,
        bit_order: str = DEFAULT_BIT_ORDER,
        status: str = "development",
        development_only: bool = True,
    ) -> "QuantizationParameters":
        """Build quantization parameters from loaded NPZ arrays."""

        required = [
            "resnet_thresholds",
            "temporal_q1_thresholds",
            "temporal_median_thresholds",
            "temporal_q3_thresholds",
            "temporal_gray_code_table",
        ]
        missing = [name for name in required if name not in arrays]
        if missing:
            raise QuantizationError(f"Quantization storage is missing arrays: {missing}.")
        params = cls(
            quantization_id=quantization_id,
            version=version,
            normalization_id=normalization_id,
            normalization_npz_sha256=normalization_npz_sha256,
            resnet_thresholds=np.asarray(arrays["resnet_thresholds"], dtype=np.float64),
            temporal_q1_thresholds=np.asarray(arrays["temporal_q1_thresholds"], dtype=np.float64),
            temporal_median_thresholds=np.asarray(arrays["temporal_median_thresholds"], dtype=np.float64),
            temporal_q3_thresholds=np.asarray(arrays["temporal_q3_thresholds"], dtype=np.float64),
            temporal_gray_code_table=np.asarray(arrays["temporal_gray_code_table"], dtype=np.uint8),
            bit_order=bit_order,
            status=status,
            development_only=development_only,
        )
        params.validate()
        return params


def derive_quantization_parameters(
    artifact: LoadedNormalizationArtifact,
    quantization_id: str = DEFAULT_QUANTIZATION_ID,
    version: str = DEFAULT_QUANTIZATION_VERSION,
    status: str = "development",
    bit_order: str = DEFAULT_BIT_ORDER,
) -> QuantizationParameters:
    """Derive development quantization thresholds from a saved normalization artifact."""

    resnet = artifact.resnet_normalizer
    temporal = artifact.temporal_normalizer
    resnet_thresholds = ((resnet.median - resnet.median) / resnet.safe_scale).astype(np.float64)
    temporal_q1 = ((temporal.q1 - temporal.median) / temporal.safe_scale).astype(np.float64)
    temporal_median = ((temporal.median - temporal.median) / temporal.safe_scale).astype(np.float64)
    temporal_q3 = ((temporal.q3 - temporal.median) / temporal.safe_scale).astype(np.float64)
    gray_table = np.asarray([GRAY_CODE_MAPPING[index] for index in range(4)], dtype=np.uint8)
    params = QuantizationParameters(
        quantization_id=quantization_id,
        version=version,
        normalization_id=artifact.calibration_id,
        normalization_npz_sha256=artifact.npz_sha256,
        resnet_thresholds=resnet_thresholds,
        temporal_q1_thresholds=temporal_q1,
        temporal_median_thresholds=temporal_median,
        temporal_q3_thresholds=temporal_q3,
        temporal_gray_code_table=gray_table,
        bit_order=bit_order,
        status=status,
        development_only=True,
    )
    params.validate()
    return params


def validate_normalized_matrix(
    values: np.ndarray,
    expected_dimension: int,
    stream_name: str,
) -> np.ndarray:
    """Return a copied finite 2D normalized feature matrix."""

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise QuantizationError(f"{stream_name} normalized features must be a 2D array.")
    if matrix.shape[1] != expected_dimension:
        raise QuantizationError(
            f"{stream_name} normalized feature dimension must be {expected_dimension}, got {matrix.shape[1]}."
        )
    if not np.all(np.isfinite(matrix)):
        raise QuantizationError(f"{stream_name} normalized features contain non-finite values.")
    return matrix.copy()


def quantize_resnet_binary(
    values: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Quantize normalized ResNet features to one bit per feature.

    Exact threshold equality maps to bit 1.
    """

    matrix = validate_normalized_matrix(values, RESNET_SEGMENT_DIMENSION, "ResNet")
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if thresholds.shape != (RESNET_DIGEST_LENGTH,):
        raise QuantizationError(f"ResNet threshold shape is invalid: {thresholds.shape}.")
    bits = (matrix >= thresholds.reshape(1, -1)).astype(np.uint8)
    if not np.all(np.isin(bits, [0, 1])):
        raise QuantizationError("ResNet binary quantization produced non-binary values.")
    return bits


def assign_temporal_bins(
    values: np.ndarray,
    q1_thresholds: np.ndarray,
    median_thresholds: np.ndarray,
    q3_thresholds: np.ndarray,
) -> np.ndarray:
    """Assign normalized temporal features to quartile-derived bins."""

    matrix = validate_normalized_matrix(values, TEMPORAL_SEGMENT_DIMENSION, "Temporal")
    q1 = np.asarray(q1_thresholds, dtype=np.float32).reshape(1, -1)
    median = np.asarray(median_thresholds, dtype=np.float32).reshape(1, -1)
    q3 = np.asarray(q3_thresholds, dtype=np.float32).reshape(1, -1)
    for name, thresholds in (("q1", q1), ("median", median), ("q3", q3)):
        if thresholds.shape != (1, TEMPORAL_SEGMENT_DIMENSION):
            raise QuantizationError(f"Temporal {name} threshold shape is invalid: {thresholds.shape}.")
        if not np.all(np.isfinite(thresholds)):
            raise QuantizationError(f"Temporal {name} thresholds contain non-finite values.")
    bins = np.zeros(matrix.shape, dtype=np.uint8)
    bins[matrix >= q1] = 1
    bins[matrix >= median] = 2
    bins[matrix >= q3] = 3
    if not np.all(np.isin(bins, [0, 1, 2, 3])):
        raise QuantizationError("Temporal bin assignment produced invalid bin values.")
    return bins


def gray_encode_temporal_bins(
    bin_indices: np.ndarray,
    gray_code_table: np.ndarray | None = None,
) -> np.ndarray:
    """Encode temporal bin indices as two-bit Gray codes."""

    bins = np.asarray(bin_indices)
    if bins.ndim != 2 or bins.shape[1] != TEMPORAL_SEGMENT_DIMENSION:
        raise QuantizationError(
            f"Temporal bin indices must have shape (segments, {TEMPORAL_SEGMENT_DIMENSION}), got {bins.shape}."
        )
    if not np.all(np.isin(bins, [0, 1, 2, 3])):
        raise QuantizationError("Temporal bin indices must be 0, 1, 2, or 3.")
    table = (
        np.asarray([GRAY_CODE_MAPPING[index] for index in range(4)], dtype=np.uint8)
        if gray_code_table is None
        else np.asarray(gray_code_table, dtype=np.uint8)
    )
    if table.shape != (4, 2) or not np.all(np.isin(table, [0, 1])):
        raise QuantizationError("Temporal Gray-code table must have shape (4, 2) and binary values.")
    bits = table[bins.astype(np.int64)].reshape(bins.shape[0], TEMPORAL_DIGEST_LENGTH).astype(np.uint8)
    if not np.all(np.isin(bits, [0, 1])):
        raise QuantizationError("Temporal Gray-code encoding produced non-binary values.")
    return bits


def build_hybrid_digest(resnet_bits: np.ndarray, temporal_bits: np.ndarray) -> np.ndarray:
    """Concatenate ResNet and temporal bits in deterministic stream order."""

    resnet = np.asarray(resnet_bits, dtype=np.uint8)
    temporal = np.asarray(temporal_bits, dtype=np.uint8)
    if resnet.ndim != 2 or resnet.shape[1] != RESNET_DIGEST_LENGTH:
        raise QuantizationError(f"ResNet bit matrix must have shape (segments, {RESNET_DIGEST_LENGTH}).")
    if temporal.ndim != 2 or temporal.shape[1] != TEMPORAL_DIGEST_LENGTH:
        raise QuantizationError(f"Temporal bit matrix must have shape (segments, {TEMPORAL_DIGEST_LENGTH}).")
    if resnet.shape[0] != temporal.shape[0]:
        raise QuantizationError("ResNet and temporal digest segment counts must match.")
    hybrid = np.concatenate([resnet, temporal], axis=1).astype(np.uint8)
    if hybrid.shape != (resnet.shape[0], HYBRID_DIGEST_LENGTH):
        raise QuantizationError(f"Unexpected hybrid digest shape: {hybrid.shape}.")
    if not np.all(np.isin(hybrid, [0, 1])):
        raise QuantizationError("Hybrid digest contains non-binary values.")
    return hybrid


def gray_code_mapping_for_manifest() -> dict[str, str]:
    """Return the default Gray-code mapping as JSON-friendly strings."""

    return {
        f"bin_{index}": "".join(str(bit) for bit in GRAY_CODE_MAPPING[index])
        for index in range(4)
    }


def stream_boundaries_for_manifest() -> dict[str, dict[str, int]]:
    """Return digest stream boundary metadata."""

    return {name: dict(values) for name, values in STREAM_BOUNDARIES.items()}


def digest_lengths_for_manifest() -> dict[str, int]:
    """Return digest lengths by stream."""

    return dict(DIGEST_LENGTHS)


def quantization_metadata(params: QuantizationParameters) -> dict[str, Any]:
    """Return JSON-friendly quantization metadata."""

    params.validate()
    return {
        "quantization_id": params.quantization_id,
        "quantization_version": params.version,
        "normalization_calibration_id": params.normalization_id,
        "normalization_artifact_checksum": params.normalization_npz_sha256,
        "status": params.status,
        "development_only": params.development_only,
        "resnet_quantization_method": "median_binary",
        "temporal_quantization_method": "quartile_gray_code",
        "threshold_derivation_method": "normalized calibration median and quartiles",
        "gray_code_mapping": gray_code_mapping_for_manifest(),
        "feature_dimensions": {
            "resnet": RESNET_SEGMENT_DIMENSION,
            "temporal": TEMPORAL_SEGMENT_DIMENSION,
        },
        "digest_dimensions": digest_lengths_for_manifest(),
        "stream_boundaries": stream_boundaries_for_manifest(),
        "bit_order": params.bit_order,
        "padding_rules": "zero-pad packed bit rows to the next whole byte; remove padding by recorded bit length",
    }
