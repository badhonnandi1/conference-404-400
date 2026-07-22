# Compression-Resilient Cryptographic Authentication for Secure Surveillance Video Integrity

This repository contains the first two implementation stages of a research prototype for authenticating surveillance video integrity under compression. The current code prepares videos for later digest generation by inspecting source files, creating logical time segments, sampling deterministic frames, and extracting pretrained ResNet-18 frame and segment features.

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
- Compressed NumPy feature storage and JSON feature manifests.
- Apple Silicon MPS selection with CPU fallback.
- Argparse command-line interface.
- Unit tests for deterministic preprocessing and feature helpers.

Not implemented yet:

- Temporal differences, optical flow, or temporal learned features.
- Quantization into binary digests.
- Perceptual hashes or SHA-256 baselines.
- HMAC protection.
- Digest comparison, thresholds, or verification.
- Segment-level tamper decisions.
- Tampering generation, augmentation, metrics, plots, or web interfaces.

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
│   ├── features/
│   │   ├── __init__.py
│   │   ├── aggregation.py
│   │   ├── device.py
│   │   ├── feature_storage.py
│   │   └── resnet_features.py
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
│   ├── test_feature_aggregation.py
│   ├── test_feature_storage.py
│   ├── test_metadata.py
│   ├── test_repository_hygiene.py
│   ├── test_resnet_features.py
│   ├── test_segmentation.py
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

## ResNet-18 Feature Extraction

Phase 2 uses `torchvision.models.resnet18` with `ResNet18_Weights.DEFAULT`. The final fully connected classification layer is replaced with an identity layer so the model returns the 512-dimensional representation after global average pooling instead of ImageNet class logits.

The preprocessing transform comes from the selected torchvision weights and includes RGB conversion, resize, center crop, tensor conversion, and ImageNet normalization. Original sampled JPEG files are never resized, overwritten, or otherwise modified.

Device selection order is:

1. MPS when available.
2. CPU otherwise.

CUDA-specific code is intentionally not used.

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
```

Remove temporary integration artefacts when finished:

```bash
rm -f data/originals/synthetic_6s.mp4
rm -rf data/sampled_frames/V001
rm -rf data/features/resnet/V001
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

## Reproducibility Notes

- Video IDs are deterministic when generated from filenames.
- Segment boundaries are timestamp-based and non-overlapping.
- The default policy discards incomplete final segments and records discarded duration.
- Sample timestamps are deterministic midpoints of fixed sampling intervals.
- Frame feature records are sorted by segment ID, requested timestamp, and frame index.
- ResNet frame embeddings are L2-normalized by default with zero-norm protection.
- Segment standard deviation uses population standard deviation.
- Feature NPZ and source frame manifests are checksummed for cache bookkeeping only.
- JSON outputs use UTF-8 and stable indentation for reviewability.
- No research dataset, generated videos, sampled frames, feature arrays, model cache files, or logs should be committed.
- Compression robustness, quantization, HMAC authentication, and verification remain future phases.
