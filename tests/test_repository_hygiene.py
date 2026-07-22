"""Tests for repository hygiene rules."""

from __future__ import annotations

from pathlib import Path


def test_generated_artifacts_are_gitignored() -> None:
    """Generated Phase 1 artifacts should not be committed accidentally."""

    gitignore = Path(".gitignore").read_text(encoding="utf-8").splitlines()

    required_patterns = {
        ".venv/",
        "logs/*.log",
        "data/originals/*",
        "data/segments/*",
        "data/sampled_frames/*",
        "data/metadata/*.json",
        "data/manifests/*.json",
        "data/features/*",
        "data/features/resnet/",
        "data/features/temporal/",
        "data/features/normalized/",
        "data/features/resnet/_model_cache/",
        "data/calibration/",
        "data/tmp/*",
        "*.ckpt",
        "*.pth",
        "*.pt",
        ".cache/",
    }
    assert required_patterns.issubset(set(gitignore))
