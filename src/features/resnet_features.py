"""Pretrained ResNet-18 frame embedding extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torchvision.models import ResNet18_Weights, resnet18

from src.features.device import DeviceInfo, select_device, torch_device


class FeatureExtractionError(RuntimeError):
    """Raised when frame feature extraction cannot continue."""


RESNET18_DEFAULT_PREPROCESSING_DESCRIPTION = (
    "ResNet18_Weights.DEFAULT transforms: RGB conversion, resize, center crop, "
    "tensor conversion, and ImageNet normalization."
)


@dataclass(frozen=True)
class FrameInputRecord:
    """A successful Phase 1 frame record prepared for feature extraction."""

    video_id: str
    segment_id: int
    frame_index: int
    requested_timestamp_seconds: float
    actual_timestamp_seconds: float | None
    frame_path: str


@dataclass(frozen=True)
class FrameFeatureRecord:
    """Serializable per-frame feature extraction record."""

    video_id: str
    segment_id: int
    frame_index: int
    requested_timestamp_seconds: float
    actual_timestamp_seconds: float | None
    frame_path: str
    embedding_row_index: int | None
    embedding_dimension: int | None
    original_embedding_norm: float | None
    normalized_embedding_norm: float | None
    extraction_success: bool
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable frame feature record."""

        return asdict(self)


@dataclass(frozen=True)
class ResNetModelBundle:
    """Loaded ResNet-18 feature extractor and preprocessing transform."""

    model: torch.nn.Module
    transform: Callable[[Image.Image], torch.Tensor]
    architecture: str
    weight_identifier: str
    preprocessing_description: str
    embedding_dimension: int
    model_loading_time_seconds: float


@dataclass(frozen=True)
class FrameExtractionResult:
    """Frame embeddings and associated extraction records."""

    embeddings: np.ndarray
    records: list[FrameFeatureRecord]
    failures: list[str]
    warnings: list[str]


def load_frame_manifest_records(frame_manifest_path: str | Path) -> tuple[str, list[FrameInputRecord], list[dict[str, Any]]]:
    """Load successful frame records from a Phase 1 frame manifest.

    Records are sorted deterministically by segment ID, requested timestamp, and
    frame index. Duplicate successful records are rejected.
    """

    path = Path(frame_manifest_path)
    if not path.exists():
        raise FeatureExtractionError(f"Frame manifest not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    video_id = str(manifest.get("video_id") or "")
    if not video_id:
        raise FeatureExtractionError(f"Frame manifest is missing video_id: {path}")

    raw_records = manifest.get("frame_records")
    if not isinstance(raw_records, list):
        raise FeatureExtractionError(f"Frame manifest has no frame_records list: {path}")

    successful: list[FrameInputRecord] = []
    failed_records: list[dict[str, Any]] = []
    seen: set[tuple[int, int, float, str]] = set()
    for raw in raw_records:
        if not raw.get("success"):
            failed_records.append(raw)
            continue
        frame_path = str(raw.get("output_frame_path") or "")
        key = (
            int(raw["segment_id"]),
            int(raw["frame_index"]),
            float(raw["requested_timestamp_seconds"]),
            frame_path,
        )
        if key in seen:
            raise FeatureExtractionError(
                f"Duplicate successful frame record in {path}: segment={key[0]} "
                f"frame={key[1]} timestamp={key[2]} path={key[3]}"
            )
        seen.add(key)
        successful.append(
            FrameInputRecord(
                video_id=video_id,
                segment_id=key[0],
                frame_index=key[1],
                requested_timestamp_seconds=key[2],
                actual_timestamp_seconds=(
                    float(raw["actual_timestamp_seconds"])
                    if raw.get("actual_timestamp_seconds") is not None
                    else None
                ),
                frame_path=frame_path,
            )
        )

    successful.sort(
        key=lambda record: (
            record.segment_id,
            record.requested_timestamp_seconds,
            record.frame_index,
        )
    )
    return video_id, successful, failed_records


def load_image_rgb(frame_path: str | Path) -> Image.Image:
    """Load a frame image and return an RGB PIL image."""

    path = Path(frame_path)
    if not path.exists():
        raise FeatureExtractionError(f"Frame image not found: {path}")
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise FeatureExtractionError(
            f"Frame image could not be decoded as an image: {path}"
        ) from exc


def normalize_embeddings(
    embeddings: np.ndarray, epsilon: float = 1e-12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """L2-normalize embeddings with zero-norm protection."""

    if embeddings.ndim != 2:
        raise FeatureExtractionError("Embeddings must be a 2D array.")
    norms = np.linalg.norm(embeddings, axis=1).astype(np.float32)
    denominators = np.maximum(norms, epsilon).reshape(-1, 1)
    normalized = (embeddings / denominators).astype(np.float32)
    normalized_norms = np.linalg.norm(normalized, axis=1).astype(np.float32)
    return normalized, norms, normalized_norms


def validate_embedding_array(embeddings: np.ndarray, expected_dimension: int) -> None:
    """Validate embedding shape and numerical finiteness."""

    if embeddings.ndim != 2 or embeddings.shape[1] != expected_dimension:
        raise FeatureExtractionError(
            f"Expected embeddings with dimension {expected_dimension}, got shape {embeddings.shape}."
        )
    if not np.all(np.isfinite(embeddings)):
        raise FeatureExtractionError("Embedding output contains non-finite values.")


def load_resnet18_feature_extractor(
    device_info: DeviceInfo,
    weights_identifier: str = "DEFAULT",
    expected_dimension: int = 512,
    model_cache_dir: str | Path | None = None,
) -> ResNetModelBundle:
    """Load pretrained ResNet-18 with the classifier replaced by identity."""

    if weights_identifier != "DEFAULT":
        raise FeatureExtractionError("Only ResNet18_Weights.DEFAULT is supported in Phase 2.")

    started = perf_counter()
    weights = ResNet18_Weights.DEFAULT
    if model_cache_dir is not None:
        cache_dir = Path(model_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(cache_dir))
    try:
        model = resnet18(weights=weights)
    except (OSError, RuntimeError) as exc:
        raise FeatureExtractionError(
            "Could not load pretrained ResNet-18 weights. Check network access, "
            "disk space, and write permissions for the configured model cache. "
            f"Details: {exc}"
        ) from exc
    model.fc = torch.nn.Identity()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(torch_device(device_info))
    elapsed = perf_counter() - started

    return ResNetModelBundle(
        model=model,
        transform=weights.transforms(),
        architecture="resnet18",
        weight_identifier="DEFAULT",
        preprocessing_description=RESNET18_DEFAULT_PREPROCESSING_DESCRIPTION,
        embedding_dimension=expected_dimension,
        model_loading_time_seconds=elapsed,
    )


def _chunks(values: list[FrameInputRecord], size: int) -> Iterable[list[FrameInputRecord]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def extract_embeddings_for_frame_records(
    frame_records: list[FrameInputRecord],
    model: torch.nn.Module,
    transform: Callable[[Image.Image], torch.Tensor],
    device_info: DeviceInfo,
    batch_size: int,
    expected_dimension: int = 512,
    normalize: bool = True,
    epsilon: float = 1e-12,
) -> FrameExtractionResult:
    """Extract one embedding per valid frame record using batched inference."""

    if batch_size <= 0:
        raise FeatureExtractionError("Batch size must be greater than zero.")

    device = torch_device(device_info)
    embeddings: list[np.ndarray] = []
    records: list[FrameFeatureRecord] = []
    warnings: list[str] = []
    failures: list[str] = []
    next_row_index = 0

    for batch in _chunks(frame_records, batch_size):
        valid_batch: list[tuple[FrameInputRecord, torch.Tensor]] = []
        for record in batch:
            try:
                image = load_image_rgb(record.frame_path)
                valid_batch.append((record, transform(image)))
            except FeatureExtractionError as exc:
                message = f"{exc} (segment={record.segment_id}, frame={record.frame_index})"
                failures.append(message)
                records.append(
                    FrameFeatureRecord(
                        video_id=record.video_id,
                        segment_id=record.segment_id,
                        frame_index=record.frame_index,
                        requested_timestamp_seconds=record.requested_timestamp_seconds,
                        actual_timestamp_seconds=record.actual_timestamp_seconds,
                        frame_path=record.frame_path,
                        embedding_row_index=None,
                        embedding_dimension=None,
                        original_embedding_norm=None,
                        normalized_embedding_norm=None,
                        extraction_success=False,
                        error_message=message,
                    )
                )

        if not valid_batch:
            continue

        input_tensor = torch.stack([tensor for _record, tensor in valid_batch]).to(device)
        with torch.inference_mode():
            output = model(input_tensor)
        output_array = output.detach().cpu().numpy().astype(np.float32)
        validate_embedding_array(output_array, expected_dimension)

        stored_array = output_array
        original_norms = np.linalg.norm(output_array, axis=1).astype(np.float32)
        normalized_norms = original_norms
        if normalize:
            stored_array, original_norms, normalized_norms = normalize_embeddings(
                output_array, epsilon=epsilon
            )

        validate_embedding_array(stored_array, expected_dimension)
        for index, (record, _tensor) in enumerate(valid_batch):
            embedding = stored_array[index]
            if not np.any(np.abs(embedding) > 0):
                warnings.append(
                    f"Embedding is entirely zero for frame {record.frame_path}."
                )
            embeddings.append(embedding)
            records.append(
                FrameFeatureRecord(
                    video_id=record.video_id,
                    segment_id=record.segment_id,
                    frame_index=record.frame_index,
                    requested_timestamp_seconds=record.requested_timestamp_seconds,
                    actual_timestamp_seconds=record.actual_timestamp_seconds,
                    frame_path=record.frame_path,
                    embedding_row_index=next_row_index,
                    embedding_dimension=expected_dimension,
                    original_embedding_norm=float(original_norms[index]),
                    normalized_embedding_norm=float(normalized_norms[index]),
                    extraction_success=True,
                )
            )
            next_row_index += 1

    records.sort(
        key=lambda record: (
            record.segment_id,
            record.requested_timestamp_seconds,
            record.frame_index,
        )
    )
    if embeddings:
        embedding_array = np.vstack(embeddings).astype(np.float32)
    else:
        embedding_array = np.empty((0, expected_dimension), dtype=np.float32)
    return FrameExtractionResult(
        embeddings=embedding_array,
        records=records,
        failures=failures,
        warnings=warnings,
    )


def extract_resnet18_frame_features(
    frame_manifest_path: str | Path,
    batch_size: int = 8,
    requested_device: str = "auto",
    normalize: bool = True,
    expected_dimension: int = 512,
    model_cache_dir: str | Path | None = None,
) -> tuple[str, ResNetModelBundle, DeviceInfo, FrameExtractionResult, list[dict[str, Any]]]:
    """Load frame records, load ResNet-18, and extract frame embeddings."""

    video_id, frame_records, source_failures = load_frame_manifest_records(frame_manifest_path)
    device_info = select_device(requested_device)
    bundle = load_resnet18_feature_extractor(
        device_info=device_info,
        weights_identifier="DEFAULT",
        expected_dimension=expected_dimension,
        model_cache_dir=model_cache_dir,
    )
    result = extract_embeddings_for_frame_records(
        frame_records=frame_records,
        model=bundle.model,
        transform=bundle.transform,
        device_info=device_info,
        batch_size=batch_size,
        expected_dimension=expected_dimension,
        normalize=normalize,
    )
    return video_id, bundle, device_info, result, source_failures
