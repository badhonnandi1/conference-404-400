# Compression-Resilient Cryptographic Authentication for Secure Surveillance Video Integrity

This repository contains the first implementation stage of a research prototype for authenticating surveillance video integrity under compression. The current code prepares videos for later digest generation by inspecting source files, creating logical time segments, and sampling deterministic frames from each complete segment.

## Current Scope

Implemented in this phase:

- FFmpeg and FFprobe availability checks.
- FFprobe metadata extraction into structured JSON.
- Safe video identifier generation.
- Non-overlapping timestamp segmentation with five-second defaults.
- Default discard policy for final incomplete segments.
- OpenCV frame sampling at one frame per second.
- JSON manifests for metadata, segments, and sampled frames.
- Argparse command-line interface.
- Unit tests for deterministic preprocessing helpers.

Not implemented yet:

- ResNet-18 feature extraction.
- Temporal differences, optical flow, or learned features.
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
│   ├── test_metadata.py
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

Optional arguments:

```bash
--config path/to/config.yaml
--segment-duration 5
--sample-fps 1
--keep-incomplete-segment
--overwrite
--verbose
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

## Testing

Run unit tests:

```bash
pytest
```

Run the environment check:

```bash
python main.py check-env
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
```

Remove temporary integration artefacts when finished:

```bash
rm -f data/originals/synthetic_6s.mp4
rm -rf data/sampled_frames/V001
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

## Reproducibility Notes

- Video IDs are deterministic when generated from filenames.
- Segment boundaries are timestamp-based and non-overlapping.
- The default policy discards incomplete final segments and records discarded duration.
- Sample timestamps are deterministic midpoints of fixed sampling intervals.
- JSON outputs use UTF-8 and stable indentation for reviewability.
- No research dataset, generated videos, sampled frames, or logs should be committed.
