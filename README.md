# Compression-Resilient Cryptographic Authentication for Secure Surveillance Video Integrity

This repository contains the first five implementation stages of a research prototype for authenticating surveillance video integrity under compression. The current code prepares videos for later digest generation by inspecting source files, creating logical time segments, sampling deterministic frames, extracting pretrained ResNet-18 frame and segment features, measuring lightweight temporal consistency inside each segment, aligning the streams, applying stream-specific robust normalization, and converting normalized features into development binary digests.

## Current Scope

Implemented in this phase:

- FFmpeg and FFprobe availability checks.
- FFprobe metadata extraction into structured JSON.
- Safe video identifier generation.
- Non-overlapping timestamp segmentation with five-second defaults.
- Default discard policy for final incomplete segments.
- OpenCV frame sampling at one frame per second.
- JSON manifests for metadata, segments, and sampled frames.
- Pretrained torchvision ResNet-18 frame feature extraction.
- Segment-level mean and standard-deviation feature aggregation.
- Dense temporal frame sampling at four frames per second.
- Interpretable frame-to-frame temporal difference features.
- Segment-level temporal mean, standard-deviation, and maximum aggregation.
- Segment-level ResNet/temporal feature alignment.
- Stream-specific robust median/IQR normalization.
- Development calibration artifact creation and inspection.
- Normalized ResNet, temporal, and combined feature outputs.
- Development binary quantization artifact creation and inspection.
- ResNet, temporal, and hybrid binary digest generation.
- Packed-byte digest storage with padding metadata and round-trip validation.
- Clipping diagnostics for normalized features before quantization.
- Compressed NumPy feature storage and JSON feature manifests.
- Apple Silicon MPS selection with CPU fallback.
- Argparse command-line interface.
- Unit tests for deterministic preprocessing and feature helpers.

Not implemented yet:

- Optical flow or temporal learned features.
- Video compression variants.
- Tampered-video generation.
- Perceptual hashes or SHA-256 baselines.
- HMAC protection.
- Hamming-distance digest comparison, thresholds, or verification.
- Segment-level tamper decisions.
- Augmentation, metrics, ROC curves, plots, or web interfaces.

## Environment Setup

Target environment:

- macOS on Apple Silicon or Linux.
- Python 3.11 or newer.
- FFmpeg and FFprobe installed separately.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install FFmpeg on macOS:

```bash
brew install ffmpeg
```

## Project Structure

```text
video-authentication/
├── configs/
│   └── default.yaml
├── data/
│   ├── originals/
│   ├── segments/
│   ├── sampled_frames/
│   ├── metadata/
│   └── manifests/
├── logs/
├── src/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── authentication/
│   │   ├── __init__.py
│   │   ├── digest.py
│   │   ├── digest_storage.py
│   │   └── quantization.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── aggregation.py
│   │   ├── alignment.py
│   │   ├── device.py
│   │   ├── feature_storage.py
│   │   ├── fusion.py
│   │   ├── normalization.py
│   │   ├── normalization_storage.py
│   │   ├── resnet_features.py
│   │   ├── temporal_features.py
│   │   ├── temporal_sampling.py
│   │   └── temporal_storage.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging_utils.py
│   │   └── ffmpeg_utils.py
│   └── video/
│       ├── __init__.py
│       ├── metadata.py
│       ├── segmentation.py
│       └── frame_sampling.py
├── tests/
│   ├── __init__.py
│   ├── test_digest.py
│   ├── test_digest_storage.py
│   ├── test_feature_aggregation.py
│   ├── test_feature_alignment.py
│   ├── test_feature_fusion.py
│   ├── test_feature_normalization.py
│   ├── test_feature_storage.py
│   ├── test_metadata.py
│   ├── test_normalization_storage.py
│   ├── test_quantization.py
│   ├── test_repository_hygiene.py
│   ├── test_resnet_features.py
│   ├── test_segmentation.py
│   ├── test_temporal_features.py
│   ├── test_temporal_sampling.py
│   ├── test_temporal_storage.py
│   └── test_frame_sampling.py
├── .gitignore
├── requirements.txt
├── README.md
└── main.py
```

## Configuration

The default configuration is loaded from `configs/default.yaml`.

Defaults:

- Segment duration: `5` seconds.
- Frame sampling rate: `1` frame per second.
- Final incomplete segment policy: `discard`.
- Output paths under `data/`.
- Logs under `logs/`.
- ResNet-18 architecture with `ResNet18_Weights.DEFAULT`.
- Frame embedding dimension: `512`.
- Segment representations: mean `512`, standard deviation `512`, combined `1024`.
- Frame embeddings are L2-normalized by default.
- Feature device: `auto`, choosing MPS when available and CPU otherwise.
- Temporal sampling rate: `4` frames per second.
- Temporal preprocessing: grayscale, resize to `224x224`, `3x3` Gaussian blur, float32 `[0, 1]`.
- Temporal segment vector: `18` values from six pair features aggregated by mean, population standard deviation, and maximum.
- Normalized feature output path: `data/features/normalized`.
- Development calibration path: `data/calibration`.
- Digest output path: `data/digests`.
- Development quantizer version: `dev_quantizer_v1`.
- ResNet quantization: one bit per normalized feature.
- Temporal quantization: four bins with two-bit Gray code.
- Hybrid digest length: `1060` bits.

## CLI Commands

Run commands from the `video-authentication/` directory.

Check the environment:

```bash
python main.py check-env
```

Inspect a video:

```bash
python main.py inspect \
  --video data/originals/sample.mp4 \
  --video-id V001
```

Create a logical segment manifest:

```bash
python main.py segment \
  --video data/originals/sample.mp4 \
  --video-id V001
```

Sample frames from complete segments:

```bash
python main.py sample \
  --video data/originals/sample.mp4 \
  --video-id V001
```

Run the complete preprocessing stage:

```bash
python main.py preprocess \
  --video data/originals/sample.mp4 \
  --video-id V001
```

Check the feature extraction environment:

```bash
python main.py feature-env
```

Extract ResNet-18 features from a Phase 1 frame manifest:

```bash
python main.py extract-resnet \
  --video-id V001 \
  --overwrite
```

Use explicit feature extraction options:

```bash
python main.py extract-resnet \
  --video-id V001 \
  --frame-manifest data/manifests/V001_frames.json \
  --batch-size 8 \
  --device auto \
  --overwrite
```

Extract temporal consistency features from a Phase 1 segment manifest:

```bash
python main.py extract-temporal \
  --video-id V001 \
  --sample-fps 4 \
  --overwrite
```

When metadata is not available or a custom source path is needed:

```bash
python main.py extract-temporal \
  --video-id V001 \
  --video-path data/originals/sample.mp4 \
  --segment-manifest data/manifests/V001_segments.json \
  --frame-width 224 \
  --frame-height 224 \
  --changed-pixel-threshold 20 \
  --overwrite
```

Run temporal extraction for the development originals registry:

```bash
python main.py extract-temporal-all \
  --registry data/manifests/development_originals_registry.json \
  --overwrite
```

Fit the current development normalization artifact:

```bash
python main.py fit-normalization \
  --video-ids V001 V002 V003 \
  --calibration-id DEV_NORMALIZATION_V1 \
  --status development \
  --overwrite
```

Normalize one video:

```bash
python main.py normalize-features \
  --video-id V001 \
  --calibration-id DEV_NORMALIZATION_V1 \
  --overwrite
```

Normalize the development set:

```bash
python main.py normalize-features-all \
  --video-ids V001 V002 V003 \
  --calibration-id DEV_NORMALIZATION_V1 \
  --overwrite
```

Inspect calibration and normalized outputs:

```bash
python main.py inspect-normalization \
  --calibration-id DEV_NORMALIZATION_V1

python main.py inspect-normalized-features \
  --video-id V001
```

Create the development quantizer:

```bash
python main.py create-quantizer \
  --normalization-id DEV_NORMALIZATION_V1 \
  --quantization-id DEV_QUANTIZATION_V1 \
  --status development \
  --overwrite
```

Build one digest:

```bash
python main.py build-digest \
  --video-id V001 \
  --quantization-id DEV_QUANTIZATION_V1 \
  --overwrite
```

Build digests for the development set:

```bash
python main.py build-digests \
  --video-ids V001 V002 V003 \
  --quantization-id DEV_QUANTIZATION_V1 \
  --overwrite
```

Inspect quantizer and digest outputs:

```bash
python main.py inspect-quantizer \
  --quantization-id DEV_QUANTIZATION_V1

python main.py inspect-digest \
  --video-id V001
```

Optional arguments:

```bash
--config path/to/config.yaml
--segment-duration 5
--sample-fps 1
--keep-incomplete-segment
--overwrite
--verbose
```

ResNet-specific options:

```bash
--batch-size 8
--device auto
--device cpu
--device mps
```

Temporal-specific options:

```bash
--video-path data/originals/sample.mp4
--segment-manifest data/manifests/V001_segments.json
--sample-fps 4
--frame-width 224
--frame-height 224
--changed-pixel-threshold 20
```

## Example Workflow

Generate a small temporary synthetic video for local testing:

```bash
ffmpeg -y \
  -f lavfi -i testsrc=duration=6:size=320x240:rate=10 \
  -pix_fmt yuv420p \
  data/originals/synthetic_6s.mp4
```

Run preprocessing:

```bash
python main.py preprocess \
  --video data/originals/synthetic_6s.mp4 \
  --video-id V001 \
  --overwrite
```

Expected default behavior for a six-second video:

- One complete five-second segment.
- One second discarded.
- Five requested frame samples at `0.5`, `1.5`, `2.5`, `3.5`, and `4.5` seconds.

## Outputs

Metadata JSON:

```text
data/metadata/V001_metadata.json
```

Segment manifest:

```text
data/manifests/V001_segments.json
```

Frame manifest:

```text
data/manifests/V001_frames.json
```

Sampled frames:

```text
data/sampled_frames/V001/segment_000/
```

Frame filenames are deterministic. Example:

```text
V001_segment_000_frame_000_t0000500ms.jpg
```

ResNet feature arrays:

```text
data/features/resnet/V001/V001_resnet_features.npz
```

ResNet feature manifest:

```text
data/features/resnet/V001/V001_resnet_manifest.json
```

The NPZ contains:

- `frame_embeddings`: one 512-dimensional vector per sampled frame.
- `frame_segment_ids`, `frame_indices`, `frame_requested_timestamps`, `frame_actual_timestamps`.
- `segment_ids`.
- `segment_mean_embeddings`: one 512-dimensional mean vector per segment.
- `segment_std_embeddings`: one 512-dimensional population-standard-deviation vector per segment.
- `segment_combined_embeddings`: concatenated mean and standard-deviation vectors with 1024 dimensions.

The JSON feature manifest records source checksums, torch/torchvision versions, preprocessing details, selected device, timing, frame records, segment records, output checksum, warnings, and failures. Full embedding arrays are intentionally stored only in the NPZ, not JSON.

Temporal feature arrays:

```text
data/features/temporal/V001/V001_temporal_features.npz
```

Temporal feature manifest:

```text
data/features/temporal/V001/V001_temporal_manifest.json
```

The temporal NPZ contains:

- `pair_features`: one six-dimensional row per successful consecutive temporal frame pair.
- `pair_segment_ids`, `pair_indices`, `pair_start_timestamps`, `pair_end_timestamps`.
- `segment_ids`.
- `segment_features`: one 18-dimensional temporal vector per complete segment.
- `segment_successful_pair_counts`.
- `segment_max_discontinuity_pair_indices`.
- `segment_max_discontinuity_timestamps`.
- `feature_names`: ordered segment feature names.

The temporal JSON manifest records source-video and segment-manifest checksums, sampling and preprocessing configuration, pair feature definitions, ordered segment feature names, segment records, pair records, output checksum, timings, warnings, and failures. Full numerical arrays are stored in the NPZ rather than duplicated in JSON.

Development normalization artifact:

```text
data/calibration/DEV_NORMALIZATION_V1/normalization_parameters.npz
data/calibration/DEV_NORMALIZATION_V1/normalization_manifest.json
```

The normalization NPZ contains stream-specific arrays such as `resnet_median`, `resnet_q1`, `resnet_q3`, `resnet_iqr`, `resnet_safe_scale`, `resnet_zero_iqr_mask`, and equivalent `temporal_*` arrays. The manifest records the calibration ID, source videos, source segment IDs, feature checksums, dimensions, normalization method, clipping range, zero-IQR counts, software versions, output checksum, warnings, and limitations.

Normalized feature arrays:

```text
data/features/normalized/V001/V001_normalized_features.npz
data/features/normalized/V001/V001_normalized_manifest.json
```

The normalized NPZ contains:

- `segment_ids`, `segment_start_times`, `segment_end_times`.
- `resnet_raw_features`: `(segments, 1024)`.
- `temporal_raw_features`: `(segments, 18)`.
- `resnet_normalized_features`: `(segments, 1024)`.
- `temporal_normalized_features`: `(segments, 18)`.
- `combined_normalized_features`: `(segments, 1042)`.

Stream boundaries are preserved in the manifest: ResNet uses combined indices `0-1023`, and temporal features use `1024-1041`. The combined array is storage convenience only; later verification must compare streams separately before deciding weighting.

Development quantization artifact:

```text
data/calibration/DEV_QUANTIZATION_V1/quantization_parameters.npz
data/calibration/DEV_QUANTIZATION_V1/quantization_manifest.json
```

The quantization NPZ contains:

- `resnet_thresholds`: 1024 one-bit thresholds.
- `temporal_q1_thresholds`, `temporal_median_thresholds`, `temporal_q3_thresholds`: 18 thresholds each.
- `temporal_gray_code_table`: bin-to-bit mapping.
- `stream_boundaries`.
- `digest_lengths`.

Digest outputs:

```text
data/digests/V001/V001_digests.npz
data/digests/V001/V001_digest_manifest.json
```

The digest NPZ contains:

- `segment_ids`, `segment_start_times`, `segment_end_times`.
- `resnet_binary_digests`: `(segments, 1024)`.
- `temporal_bin_indices`: `(segments, 18)`.
- `temporal_binary_digests`: `(segments, 36)`.
- `hybrid_binary_digests`: `(segments, 1060)`.
- `resnet_packed_digests`: `(segments, 128)`.
- `temporal_packed_digests`: `(segments, 5)`.
- `hybrid_packed_digests`: `(segments, 133)`.
- `resnet_bit_length`, `temporal_bit_length`, `hybrid_bit_length`.

The digest manifest records source normalized-feature checksums, calibration and quantizer checksums, digest dimensions, stream boundaries, bit order, padding counts, bit-one ratios, temporal bin distributions, clipping statistics, output checksum, warnings, and failures. Full bit arrays are intentionally stored in NPZ, not duplicated in JSON.

## ResNet-18 Feature Extraction

Phase 2 uses `torchvision.models.resnet18` with `ResNet18_Weights.DEFAULT`. The final fully connected classification layer is replaced with an identity layer so the model returns the 512-dimensional representation after global average pooling instead of ImageNet class logits.

The preprocessing transform comes from the selected torchvision weights and includes RGB conversion, resize, center crop, tensor conversion, and ImageNet normalization. Original sampled JPEG files are never resized, overwritten, or otherwise modified.

Device selection order is:

1. MPS when available.
2. CPU otherwise.

CUDA-specific code is intentionally not used.

## Temporal Consistency Features

Phase 3 adds a second, independent feature stream for frame-to-frame consistency inside each five-second segment. It is designed to expose interpretable motion and discontinuity signals that later phases can protect cryptographically, not to make final tamper decisions by itself.

Temporal sampling uses four frames per second because one frame per second is too sparse for insertion, deletion, duplication, or abrupt replacement signals. For a five-second segment this requests 20 midpoint timestamps, such as `0.125`, `0.375`, `0.625`, and so on, producing up to 19 consecutive frame pairs. Temporal frames are decoded directly from the source video and kept in memory during calculation; they are not saved as permanent JPEGs by default.

Each decoded frame is preprocessed only for comparison:

- Convert BGR to grayscale.
- Resize to `224x224`.
- Apply a light `3x3` Gaussian blur.
- Convert to float32 values in `[0, 1]`.

The six pair-level features are:

- Mean absolute difference.
- Standard deviation of absolute difference.
- Normalized root mean squared difference.
- Changed-pixel ratio using `changed_pixel_threshold / 255`.
- 90th-percentile absolute difference.
- Sobel edge-change ratio.

Each complete segment aggregates those six pair features using mean, population standard deviation, and maximum. The default temporal segment vector therefore has `6 x 3 = 18` dimensions. Optical flow is intentionally not implemented yet.

## Alignment and Normalization

Phase 4 aligns ResNet and temporal segment features before any normalization or concatenation. Alignment checks the requested video ID, required arrays, duplicate segment IDs, missing segment IDs, expected dimensions, finite values, deterministic segment ordering, and available segment start/end timestamps. This prevents a ResNet vector from segment `1` being paired with a temporal vector from segment `2`, which would invalidate later digest generation.

ResNet and temporal streams are normalized separately because their meanings and scales are different. ResNet contributes a 1024-dimensional representation from learned visual embeddings, while temporal contributes 18 interpretable frame-difference statistics. Fitting one shared scaler across both would mix unlike distributions and hide stream boundaries.

The normalizer uses robust per-dimension median/IQR scaling:

```text
normalized_value = (value - median) / max(IQR, epsilon)
```

The default epsilon is `1e-8`, and normalized values are clipped to `[-5, 5]`. Zero-IQR dimensions are recorded in the calibration manifest and safely scaled by epsilon instead of producing division-by-zero errors. With only nine development segments from V001-V003, zero-IQR dimensions can occur and are reported rather than treated as automatic failures.

`DEV_NORMALIZATION_V1` is a development-only artifact. It was fitted using only three original videos and must be replaced before final compression-resilience or tamper-detection experiments. It is useful for validating the pipeline shape, storage, and reproducibility, not for final research claims.

## Quantization and Digests

Phase 5 converts normalized continuous segment features into compact binary digests. Continuous values are not directly suitable for cryptographic authentication records because tiny numeric perturbations from decoding, compression, or hardware differences can alter exact floating-point values. Quantization creates deterministic bit strings that later phases can protect and compare.

ResNet quantization uses one bit per normalized feature. Each of the 1024 normalized ResNet dimensions is compared against a calibration-derived threshold. Because Phase 4 centers each fitted calibration median at zero, the development threshold vector is currently all zeros. Exact-threshold values map to bit `1`, which is deterministic and tested.

Temporal quantization preserves more temporal detail by using four bins per feature. For each of the 18 normalized temporal dimensions, thresholds are derived from the saved Phase 4 calibration parameters:

```text
normalized_q1 = (q1 - median) / safe_scale
normalized_median = 0
normalized_q3 = (q3 - median) / safe_scale
```

Temporal bins are encoded with two-bit Gray code:

```text
Bin 0 -> 00
Bin 1 -> 01
Bin 2 -> 11
Bin 3 -> 10
```

Adjacent bins differ by one bit, which is useful near bin boundaries. The temporal stream therefore produces `18 x 2 = 36` bits per segment.

The hybrid digest concatenates streams in this order:

```text
[ResNet bits | Temporal bits]
```

Stream boundaries are recorded as ResNet `[0, 1024)` and temporal `[1024, 1060)`. The flat 1060-bit hybrid digest is for storage and inspection only. Later verification should compare ResNet and temporal distances separately before applying any weighting policy.

Digests are packed with `numpy.packbits` using `bit_order: big`. ResNet packs to 128 bytes, temporal packs to 5 bytes with four padding bits, and hybrid packs to 133 bytes with four padding bits. The original bit length and padding-bit counts are stored so unpacking can remove padding and exactly reproduce the original bit arrays.

`DEV_QUANTIZATION_V1` is development-only. It depends on `DEV_NORMALIZATION_V1`, which used only three original videos. It must be regenerated after the final calibration split is prepared. No HMAC security, Hamming-distance comparison, verification thresholds, compression-resilience result, or tamper-detection result is claimed in this phase.

## Testing

Run unit tests:

```bash
pytest
```

Run the environment check:

```bash
python main.py check-env
```

Run the feature environment check:

```bash
python main.py feature-env
```

Optional integration flow:

```bash
ffmpeg -y \
  -f lavfi -i testsrc=duration=6:size=320x240:rate=10 \
  -pix_fmt yuv420p \
  data/originals/synthetic_6s.mp4

python main.py inspect --video data/originals/synthetic_6s.mp4 --video-id V001 --overwrite
python main.py segment --video data/originals/synthetic_6s.mp4 --video-id V001 --overwrite
python main.py sample --video data/originals/synthetic_6s.mp4 --video-id V001 --overwrite
python main.py preprocess --video data/originals/synthetic_6s.mp4 --video-id V001 --overwrite
python main.py extract-resnet --video-id V001 --overwrite
python main.py extract-temporal --video-id V001 --overwrite
python main.py fit-normalization --video-ids V001 V002 V003 --calibration-id DEV_NORMALIZATION_V1 --status development --overwrite
python main.py normalize-features-all --video-ids V001 V002 V003 --calibration-id DEV_NORMALIZATION_V1 --overwrite
python main.py create-quantizer --normalization-id DEV_NORMALIZATION_V1 --quantization-id DEV_QUANTIZATION_V1 --status development --overwrite
python main.py build-digests --video-ids V001 V002 V003 --quantization-id DEV_QUANTIZATION_V1 --overwrite
```

Optional Phase 3 synthetic sanity checks should use temporary files under `data/tmp/`:

```bash
ffmpeg -y \
  -f lavfi -i color=c=gray:s=320x240:r=10:d=6 \
  -c:v libx264 \
  -pix_fmt yuv420p \
  data/tmp/phase3_static_6s.mp4

ffmpeg -y \
  -f lavfi -i color=c=black:s=320x240:r=10:d=3 \
  -f lavfi -i color=c=white:s=320x240:r=10:d=3 \
  -filter_complex "[0:v][1:v]concat=n=2:v=1:a=0[outv]" \
  -map "[outv]" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  data/tmp/phase3_abrupt_6s.mp4
```

The static video should produce near-zero temporal differences. The abrupt-change video should produce its maximum discontinuity near the known transition time.

Remove temporary integration artefacts when finished:

```bash
rm -f data/originals/synthetic_6s.mp4
rm -rf data/sampled_frames/V001
rm -rf data/features/resnet/V001
rm -rf data/features/temporal/V001
rm -rf data/features/normalized/V001
rm -rf data/calibration/DEV_NORMALIZATION_V1
rm -rf data/calibration/DEV_QUANTIZATION_V1
rm -rf data/digests/V001
rm -f data/metadata/V001_metadata.json data/manifests/V001_segments.json data/manifests/V001_frames.json
```

## Troubleshooting

`ffmpeg` or `ffprobe` is missing:

- Install FFmpeg with `brew install ffmpeg` on macOS.
- Confirm `ffmpeg -version` and `ffprobe -version` work in the same shell.

OpenCV is missing:

- Activate the virtual environment.
- Run `pip install -r requirements.txt`.

Video cannot be inspected:

- Check that the `--video` path exists.
- Check that FFprobe supports the container and codec.
- Check that the file is not truncated or permission-restricted.

Frame sampling fails:

- Confirm OpenCV can open the file.
- Try a standard MP4/H.264 test file.
- Use `--overwrite` when re-running commands with the same video ID.

ResNet extraction fails:

- Confirm `torch` and `torchvision` are installed in the active environment.
- Run `python main.py feature-env`.
- Confirm the Phase 1 frame manifest exists and references readable JPEG files.
- The first ResNet run may need network access to download pretrained weights.
- Downloaded weights are cached under ignored `data/features/resnet/_model_cache/`.

Temporal extraction fails:

- Confirm the Phase 1 segment manifest exists.
- Confirm the source video path in metadata still points to an existing readable file.
- Confirm OpenCV can seek and decode frames from the source container.
- Use `--overwrite` only when intentionally regenerating non-matching temporal outputs.

Normalization fails:

- Confirm ResNet and temporal feature NPZ files exist for the same video ID.
- Confirm segment IDs match between streams.
- Run `inspect-normalization` to confirm the calibration artifact exists.
- Use `--overwrite` only when intentionally replacing stale normalized outputs.

Digest generation fails:

- Confirm normalized feature outputs exist for the video ID.
- Confirm `DEV_QUANTIZATION_V1` exists with `inspect-quantizer`.
- Confirm the quantizer normalization ID matches the normalized feature calibration ID.
- Use `--overwrite` only when intentionally replacing stale digest outputs.

## Reproducibility Notes

- Video IDs are deterministic when generated from filenames.
- Segment boundaries are timestamp-based and non-overlapping.
- The default policy discards incomplete final segments and records discarded duration.
- Sample timestamps are deterministic midpoints of fixed sampling intervals.
- Frame feature records are sorted by segment ID, requested timestamp, and frame index.
- ResNet frame embeddings are L2-normalized by default with zero-norm protection.
- Segment standard deviation uses population standard deviation.
- Feature NPZ and source frame manifests are checksummed for cache bookkeeping only.
- Temporal timestamps are deterministic midpoint samples at the configured FPS.
- Temporal pair features and segment feature names use a fixed ordered list.
- Temporal cache reuse requires matching source checksum, segment manifest checksum, sampling rate, frame dimensions, grayscale flag, blur kernel, changed-pixel threshold, pair feature list, and aggregation list.
- Alignment sorts segment IDs deterministically and rejects missing or duplicate segment IDs.
- Normalization is fitted separately for ResNet and temporal streams.
- Normalization cache reuse requires matching source checksums, calibration checksum, method, epsilon, clipping range, and feature dimensions.
- ResNet digest bits use deterministic `value >= threshold` behavior.
- Temporal digest bins use deterministic quartile boundary behavior and Gray-code encoding.
- Packed digest round trips are validated before writing manifests.
- Digest cache reuse requires matching normalized feature checksum, calibration checksum, quantizer checksum, quantization version, bit order, stream boundaries, and padding policy.
- JSON outputs use UTF-8 and stable indentation for reviewability.
- No research dataset, generated videos, sampled frames, feature arrays, model cache files, or logs should be committed.
- The current normalization artifact uses only three original videos and must not be used for final experimental results.
- The current quantization artifact depends on only three original videos and must be regenerated after the final calibration split is prepared.
- Compressed datasets, tampered datasets, optical flow, HMAC authentication, Hamming-distance verification, threshold selection, and verification accuracy remain future phases.
