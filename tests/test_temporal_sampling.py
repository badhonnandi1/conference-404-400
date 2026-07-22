"""Tests for dense temporal sampling helpers."""

from __future__ import annotations

import numpy as np

from src.features.temporal_sampling import (
    TemporalFrameRecord,
    TemporalSamplingConfig,
    generate_temporal_timestamps,
    preprocess_temporal_frame,
)
from src.features.temporal_features import build_temporal_pairs_for_segment


def test_temporal_timestamp_generation_for_five_second_segment() -> None:
    """A five-second segment at 4 FPS yields 20 deterministic midpoint timestamps."""

    timestamps = generate_temporal_timestamps(0.0, 5.0, sample_fps=4)

    assert len(timestamps) == 20
    assert timestamps[:4] == [0.125, 0.375, 0.625, 0.875]
    assert timestamps[-1] == 4.875
    assert all(0.0 < timestamp < 5.0 for timestamp in timestamps)
    assert timestamps == generate_temporal_timestamps(0.0, 5.0, sample_fps=4)


def test_temporal_pair_creation_stays_within_segment() -> None:
    """Twenty temporal frames produce nineteen pairs inside a single segment."""

    frames = [
        TemporalFrameRecord(
            video_id="T001",
            segment_id=0,
            frame_index=index,
            requested_timestamp_seconds=timestamp,
            actual_timestamp_seconds=timestamp,
            success=True,
            image=np.full((4, 4), index / 20.0, dtype=np.float32),
        )
        for index, timestamp in enumerate(generate_temporal_timestamps(0.0, 5.0, 4))
    ]

    pair_features, pair_records = build_temporal_pairs_for_segment(
        frames,
        TemporalSamplingConfig(sample_fps=4, frame_width=4, frame_height=4),
    )

    assert pair_features.shape == (19, 6)
    assert len(pair_records) == 19
    assert {record.segment_id for record in pair_records} == {0}


def test_temporal_preprocessing_shape_and_range() -> None:
    """Decoded frames are converted to normalized fixed-size arrays."""

    frame = np.zeros((16, 12, 3), dtype=np.uint8)
    frame[:, :, 1] = 255
    config = TemporalSamplingConfig(frame_width=8, frame_height=6, gaussian_blur_kernel=3)

    processed = preprocess_temporal_frame(frame, config)

    assert processed.shape == (6, 8)
    assert processed.dtype == np.float32
    assert float(processed.min()) >= 0.0
    assert float(processed.max()) <= 1.0
