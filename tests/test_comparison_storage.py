"""Tests for Phase 7 comparison result storage."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.feature_storage import sha256_file
from src.verification.comparison import ComparisonConfig, NO_THRESHOLD_WARNING
from src.verification.comparison_storage import (
    compare_and_store_digests,
    comparison_output_paths,
    inspect_comparison,
    load_comparison_npz,
)
from src.video.metadata import ExistingOutputError
from tests.test_comparison import _bits, _key_info, protect_reference, write_digest_fixture


def test_comparison_npz_manifest_cache_and_inspection(tmp_path) -> None:
    write_digest_fixture(tmp_path, "REF")
    protect_reference(tmp_path, "REF")
    write_digest_fixture(tmp_path, "QRY")
    stored = compare_and_store_digests(
        "REF",
        "QRY",
        tmp_path / "authentication_records",
        tmp_path / "digests",
        tmp_path / "comparisons",
        _key_info(),
        ComparisonConfig(),
        overwrite=True,
    )
    assert not stored.cache_reused
    assert stored.paths.npz_path.exists()
    assert stored.paths.manifest_path.exists()
    assert stored.manifest["output_npz_checksum"] == sha256_file(stored.paths.npz_path)
    assert NO_THRESHOLD_WARNING in stored.manifest["warnings"]
    arrays = load_comparison_npz(stored.paths.npz_path)
    assert arrays["matched_segment_ids"].tolist() == [0, 1]
    assert arrays["resnet_raw_distances"].tolist() == [0, 0]
    assert arrays["balanced_diagnostic_scores"].shape == (2,)

    cached = compare_and_store_digests(
        "REF",
        "QRY",
        tmp_path / "authentication_records",
        tmp_path / "digests",
        tmp_path / "comparisons",
        _key_info(),
        ComparisonConfig(),
        overwrite=False,
    )
    assert cached.cache_reused
    manifest = inspect_comparison(tmp_path / "comparisons", "REF", "QRY")
    assert manifest["comparison_id"] == "REF__vs__QRY"


def test_cache_invalidation_when_query_digest_changes(tmp_path) -> None:
    write_digest_fixture(tmp_path, "REF")
    protect_reference(tmp_path, "REF")
    write_digest_fixture(tmp_path, "QRY")
    compare_and_store_digests(
        "REF",
        "QRY",
        tmp_path / "authentication_records",
        tmp_path / "digests",
        tmp_path / "comparisons",
        _key_info(),
        ComparisonConfig(),
        overwrite=True,
    )
    resnet, temporal = _bits(2)
    resnet[0, 0] = 1
    write_digest_fixture(tmp_path, "QRY", resnet_bits=resnet, temporal_bits=temporal)
    with pytest.raises(ExistingOutputError, match="cache metadata"):
        compare_and_store_digests(
            "REF",
            "QRY",
            tmp_path / "authentication_records",
            tmp_path / "digests",
            tmp_path / "comparisons",
            _key_info(),
            ComparisonConfig(),
            overwrite=False,
        )


def test_cache_invalidation_when_configuration_changes(tmp_path) -> None:
    write_digest_fixture(tmp_path, "REF")
    protect_reference(tmp_path, "REF")
    write_digest_fixture(tmp_path, "QRY")
    compare_and_store_digests(
        "REF",
        "QRY",
        tmp_path / "authentication_records",
        tmp_path / "digests",
        tmp_path / "comparisons",
        _key_info(),
        ComparisonConfig(),
        overwrite=True,
    )
    with pytest.raises(ExistingOutputError, match="cache metadata"):
        compare_and_store_digests(
            "REF",
            "QRY",
            tmp_path / "authentication_records",
            tmp_path / "digests",
            tmp_path / "comparisons",
            _key_info(),
            ComparisonConfig(timestamp_tolerance_microseconds=0),
            overwrite=False,
        )


def test_comparison_output_paths_are_deterministic(tmp_path) -> None:
    paths = comparison_output_paths(tmp_path, "V001", "V002")
    assert paths.output_dir == tmp_path / "V001__vs__V002"
    assert paths.npz_path.name == "comparison_results.npz"
    assert paths.manifest_path.name == "comparison_manifest.json"
