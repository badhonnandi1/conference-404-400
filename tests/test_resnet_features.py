"""Tests for ResNet frame feature extraction helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from src.features.device import DeviceSelectionError, select_device
from src.features.resnet_features import (
    FeatureExtractionError,
    FrameInputRecord,
    extract_embeddings_for_frame_records,
    load_resnet18_feature_extractor,
    load_frame_manifest_records,
    load_image_rgb,
    normalize_embeddings,
    validate_embedding_array,
)


class StubEmbeddingModel(torch.nn.Module):
    """Small deterministic model used to avoid pretrained-weight downloads in tests."""

    def __init__(self, embedding_dimension: int) -> None:
        super().__init__()
        self.embedding_dimension = embedding_dimension

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        values = torch.arange(
            1,
            batch.shape[0] * self.embedding_dimension + 1,
            dtype=torch.float32,
            device=batch.device,
        )
        return values.reshape(batch.shape[0], self.embedding_dimension)


class TinyModel(torch.nn.Module):
    """Minimal module used to test model-loading setup without downloading weights."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(1, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.ones((batch.shape[0], 512), dtype=torch.float32, device=batch.device)


def _write_image(path: Path, mode: str = "RGB") -> None:
    image = Image.new(mode, (8, 8), color=128)
    image.save(path)


def _frame_record(path: Path, segment_id: int = 0, frame_index: int = 0) -> FrameInputRecord:
    return FrameInputRecord(
        video_id="T001",
        segment_id=segment_id,
        frame_index=frame_index,
        requested_timestamp_seconds=frame_index + 0.5,
        actual_timestamp_seconds=frame_index + 0.5,
        frame_path=str(path),
    )


def test_device_auto_chooses_mps_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto device selection prefers MPS when it is available."""

    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    info = select_device("auto")

    assert info.selected_device == "mps"
    assert info.mps_built is True
    assert info.mps_available is True


def test_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto device selection falls back to CPU when MPS is unavailable."""

    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    info = select_device("auto")

    assert info.selected_device == "cpu"


def test_invalid_device_raises_clear_error() -> None:
    """Invalid device requests are rejected."""

    with pytest.raises(DeviceSelectionError, match="Invalid device"):
        select_device("cuda")


def test_load_image_rgb_valid_and_rgb_conversion(tmp_path: Path) -> None:
    """Valid images are decoded and converted to RGB."""

    path = tmp_path / "gray.jpg"
    _write_image(path, mode="L")

    image = load_image_rgb(path)

    assert image.mode == "RGB"
    assert image.size == (8, 8)


def test_load_image_rgb_missing_or_corrupt(tmp_path: Path) -> None:
    """Missing and corrupt images raise clear feature extraction errors."""

    with pytest.raises(FeatureExtractionError, match="not found"):
        load_image_rgb(tmp_path / "missing.jpg")

    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_text("not an image", encoding="utf-8")
    with pytest.raises(FeatureExtractionError, match="could not be decoded"):
        load_image_rgb(corrupt)


def test_embedding_validation_and_normalization() -> None:
    """Embedding validation checks shape, finite values, and normalization."""

    embeddings = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normalized, original_norms, normalized_norms = normalize_embeddings(embeddings)

    assert np.allclose(original_norms, [5.0, 0.0])
    assert np.allclose(normalized[0], [0.6, 0.8])
    assert np.allclose(normalized[1], [0.0, 0.0])
    assert np.allclose(normalized_norms, [1.0, 0.0])
    validate_embedding_array(normalized, expected_dimension=2)

    with pytest.raises(FeatureExtractionError, match="non-finite"):
        validate_embedding_array(np.asarray([[np.nan, 1.0]], dtype=np.float32), 2)
    with pytest.raises(FeatureExtractionError, match="Expected embeddings"):
        validate_embedding_array(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32), 2)


def test_extract_embeddings_with_stub_model(tmp_path: Path) -> None:
    """Batched extraction records row indices, dimensions, and normalized norms."""

    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    _write_image(first)
    _write_image(second)
    records = [_frame_record(first, frame_index=0), _frame_record(second, frame_index=1)]
    device_info = select_device("cpu")

    result = extract_embeddings_for_frame_records(
        frame_records=records,
        model=StubEmbeddingModel(embedding_dimension=4),
        transform=lambda _image: torch.ones((3, 8, 8), dtype=torch.float32),
        device_info=device_info,
        batch_size=2,
        expected_dimension=4,
        normalize=True,
    )

    assert result.embeddings.shape == (2, 4)
    assert np.all(np.isfinite(result.embeddings))
    assert np.allclose(np.linalg.norm(result.embeddings, axis=1), 1.0)
    assert [record.embedding_row_index for record in result.records] == [0, 1]
    assert all(record.embedding_dimension == 4 for record in result.records)
    assert not result.failures


def test_resnet_loader_uses_project_model_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ResNet loader uses a caller-provided cache directory for model weights."""

    old_hub_dir = torch.hub.get_dir()
    cache_dir = tmp_path / "torch-cache"

    def fake_resnet18(weights):
        assert weights is not None
        assert torch.hub.get_dir() == str(cache_dir)
        return TinyModel()

    monkeypatch.setattr("src.features.resnet_features.resnet18", fake_resnet18)
    try:
        bundle = load_resnet18_feature_extractor(
            device_info=select_device("cpu"),
            model_cache_dir=cache_dir,
        )
    finally:
        torch.hub.set_dir(old_hub_dir)

    assert cache_dir.exists()
    assert bundle.architecture == "resnet18"
    assert bundle.weight_identifier == "DEFAULT"


def test_resnet_loader_reports_weight_loading_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weight-loading errors include the underlying exception details."""

    old_hub_dir = torch.hub.get_dir()

    def failing_resnet18(weights):
        raise OSError("network unavailable")

    monkeypatch.setattr("src.features.resnet_features.resnet18", failing_resnet18)
    try:
        with pytest.raises(FeatureExtractionError, match="network unavailable"):
            load_resnet18_feature_extractor(
                device_info=select_device("cpu"),
                model_cache_dir=tmp_path / "torch-cache",
            )
    finally:
        torch.hub.set_dir(old_hub_dir)


def test_frame_manifest_loading_sorts_and_rejects_duplicates(tmp_path: Path) -> None:
    """Frame manifests are read deterministically and duplicate successful records fail."""

    image_path = tmp_path / "frame.jpg"
    _write_image(image_path)
    manifest = {
        "video_id": "T001",
        "frame_records": [
            {
                "segment_id": 1,
                "frame_index": 1,
                "requested_timestamp_seconds": 6.5,
                "actual_timestamp_seconds": 6.5,
                "output_frame_path": str(image_path),
                "success": True,
            },
            {
                "segment_id": 0,
                "frame_index": 0,
                "requested_timestamp_seconds": 0.5,
                "actual_timestamp_seconds": 0.5,
                "output_frame_path": str(image_path),
                "success": True,
            },
        ],
    }
    manifest_path = tmp_path / "frames.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    video_id, records, failures = load_frame_manifest_records(manifest_path)

    assert video_id == "T001"
    assert [record.segment_id for record in records] == [0, 1]
    assert failures == []

    manifest["frame_records"].append(manifest["frame_records"][1])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FeatureExtractionError, match="Duplicate"):
        load_frame_manifest_records(manifest_path)
