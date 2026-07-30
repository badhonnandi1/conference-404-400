from pathlib import Path

from scripts.run_multisource_versions_evaluation import classify_transformation


def test_classify_short_deletion_variant_filename() -> None:
    category, label, transformation, method, confidence = classify_transformation(Path("del.mp4"))

    assert category == "tampered"
    assert label == "abnormal"
    assert transformation == "frame_deletion"
    assert method == "filename_pattern_delete"
    assert confidence == 0.95


def test_classify_short_prefixed_deletion_variant_filename() -> None:
    _, _, transformation, method, _ = classify_transformation(Path("fdel.mp4"))

    assert transformation == "frame_deletion"
    assert method == "filename_pattern_delete"
