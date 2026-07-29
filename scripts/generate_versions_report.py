#!/usr/bin/env python3
"""Generate figures and a self-contained HTML report for the versions evaluation."""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
import textwrap
from typing import Any

import cv2
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/video_auth_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


EXPERIMENT_ID = "versions_evaluation"
REFERENCE_ID = "VER_ORIGINAL"


def read_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return data


def read_table(output_dir: Path, name: str) -> pd.DataFrame:
    path = output_dir / "tables" / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def savefig(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def short_labels(values: list[Any]) -> list[str]:
    return [str(value).replace("VER_", "").replace("_", "\n") for value in values]


def thresholds(summary: dict[str, Any]) -> dict[str, float]:
    payload = summary.get("thresholds", {})
    return {
        "resnet": float(payload.get("resnet", {}).get("threshold", 0.0)),
        "temporal": float(payload.get("temporal", {}).get("threshold", 0.0)),
        "balanced": float(payload.get("balanced", {}).get("threshold", 0.0)),
    }


def figure_inventory(inventory: pd.DataFrame, figures: Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    labels = short_labels(inventory["video_id"].tolist())
    x = np.arange(len(inventory))
    axes[0, 0].bar(x - 0.18, inventory["width"].fillna(0), width=0.36, label="width")
    axes[0, 0].bar(x + 0.18, inventory["height"].fillna(0), width=0.36, label="height")
    axes[0, 0].set_title("Resolution")
    axes[0, 0].set_xticks(x, labels, rotation=0)
    axes[0, 0].set_ylabel("Pixels")
    axes[0, 0].legend()
    axes[0, 1].bar(x, inventory["duration_seconds"].fillna(0))
    axes[0, 1].set_title("Duration")
    axes[0, 1].set_xticks(x, labels, rotation=0)
    axes[0, 1].set_ylabel("Seconds")
    axes[1, 0].bar(x, inventory["fps"].fillna(0))
    axes[1, 0].set_title("Frame Rate")
    axes[1, 0].set_xticks(x, labels, rotation=0)
    axes[1, 0].set_ylabel("FPS")
    bitrate_mbps = inventory["bitrate"].fillna(0).astype(float) / 1_000_000.0
    axes[1, 1].bar(x, bitrate_mbps)
    for index, codec in enumerate(inventory["codec"].fillna("unknown")):
        axes[1, 1].text(index, bitrate_mbps.iloc[index], str(codec), ha="center", va="bottom", fontsize=7)
    axes[1, 1].set_title("Bitrate and Codec")
    axes[1, 1].set_xticks(x, labels, rotation=0)
    axes[1, 1].set_ylabel("Mbps")
    fig.suptitle("01 Video Inventory")
    return savefig(fig, figures / "01_video_inventory.png")


def figure_pipeline(pipeline: pd.DataFrame, figures: Path) -> str:
    stage_order = ["preprocess", "resnet", "temporal", "normalize", "digest", "compare"]
    data = pipeline[pipeline["video_id"] != "SYSTEM"].copy()
    videos = sorted(data["video_id"].dropna().unique().tolist())
    matrix = np.full((len(videos), len(stage_order)), np.nan)
    for row_i, video_id in enumerate(videos):
        for col_i, stage in enumerate(stage_order):
            matches = data[(data["video_id"] == video_id) & (data["stage"] == stage)]
            if matches.empty:
                continue
            status = str(matches.iloc[-1]["status"])
            matrix[row_i, col_i] = 1 if status == "completed" else 0
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = ListedColormap(["#cc3d3d", "#3b8c4a"])
    ax.imshow(np.nan_to_num(matrix, nan=0), aspect="auto", vmin=0, vmax=1, cmap=cmap)
    for row_i in range(matrix.shape[0]):
        for col_i in range(matrix.shape[1]):
            value = matrix[row_i, col_i]
            text = "NA" if np.isnan(value) else ("OK" if value == 1 else "FAIL")
            ax.text(col_i, row_i, text, ha="center", va="center", color="white", fontsize=8)
    ax.set_yticks(np.arange(len(videos)), short_labels(videos))
    ax.set_xticks(np.arange(len(stage_order)), stage_order, rotation=30, ha="right")
    ax.set_title("02 Pipeline Completion")
    return savefig(fig, figures / "02_pipeline_completion.png")


def figure_distance_bars(summary: pd.DataFrame, thresh: dict[str, float], figures: Path, mode: str) -> str:
    columns = {
        "max": [
            "max_resnet_normalized_distance",
            "max_temporal_normalized_distance",
            "max_balanced_diagnostic_score",
        ],
        "mean": [
            "mean_resnet_normalized_distance",
            "mean_temporal_normalized_distance",
            "mean_balanced_diagnostic_score",
        ],
    }[mode]
    labels = short_labels(summary["video_id"].tolist())
    x = np.arange(len(summary))
    width = 0.26
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width, summary[columns[0]].fillna(0), width=width, label="ResNet")
    ax.bar(x, summary[columns[1]].fillna(0), width=width, label="Temporal")
    ax.bar(x + width, summary[columns[2]].fillna(0), width=width, label="Balanced")
    if mode == "max":
        ax.axhline(thresh["resnet"], color="#1f77b4", linestyle="--", linewidth=1, label="ResNet threshold")
        ax.axhline(thresh["temporal"], color="#ff7f0e", linestyle="--", linewidth=1, label="Temporal threshold")
        ax.axhline(thresh["balanced"], color="#2ca02c", linestyle="--", linewidth=1, label="Balanced threshold")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized distance / score")
    ax.set_title("03 Max Distance by Video" if mode == "max" else "04 Mean Distance by Video")
    ax.legend(ncols=3, fontsize=8)
    filename = "03_max_distance_by_video.png" if mode == "max" else "04_mean_distance_by_video.png"
    return savefig(fig, figures / filename)


def figure_heatmap(segments: pd.DataFrame, figures: Path) -> str:
    pivot = segments.pivot_table(
        index="video_id",
        columns="segment_id",
        values="balanced_diagnostic_score",
        aggfunc="max",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * len(pivot))))
    data = pivot.to_numpy(dtype=float)
    image = ax.imshow(data, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(np.arange(len(pivot.index)), short_labels(pivot.index.tolist()))
    ax.set_xticks(np.arange(len(pivot.columns)), [str(int(col)) for col in pivot.columns])
    ax.set_xlabel("Segment ID")
    ax.set_title("05 Segment Score Heatmap")
    fig.colorbar(image, ax=ax, label="Balanced diagnostic score")
    return savefig(fig, figures / "05_segment_score_heatmap.png")


def figure_timeline(group: pd.DataFrame, thresh: dict[str, float], figures: Path) -> str:
    video_id = str(group["video_id"].iloc[0])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    group = group.sort_values("segment_id")
    x = group["segment_id"].astype(int)
    ax.plot(x, group["resnet_normalized_distance"], marker="o", label="ResNet")
    ax.plot(x, group["temporal_normalized_distance"], marker="s", label="Temporal")
    ax.plot(x, group["balanced_diagnostic_score"], marker="^", label="Balanced")
    ax.axhline(thresh["resnet"], color="#1f77b4", linestyle="--", linewidth=1)
    ax.axhline(thresh["temporal"], color="#ff7f0e", linestyle="--", linewidth=1)
    ax.axhline(thresh["balanced"], color="#2ca02c", linestyle="--", linewidth=1)
    abnormal = group[group["segment_provisionally_abnormal"].map(parse_bool)]
    if not abnormal.empty:
        ax.scatter(
            abnormal["segment_id"],
            abnormal["balanced_diagnostic_score"],
            s=110,
            facecolors="none",
            edgecolors="#cc3d3d",
            linewidths=2,
            label="Provisional abnormal",
        )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Segment ID")
    ax.set_ylabel("Normalized distance / score")
    ax.set_title(f"Timeline {video_id}")
    ax.legend(ncols=4, fontsize=8)
    return savefig(fig, figures / f"timeline_{video_id}.png")


def figure_expected_observed(predictions: pd.DataFrame, figures: Path) -> str:
    data = predictions.copy()
    data["expected"] = data["expected_label"].map({"normal": "Expected normal", "abnormal": "Expected abnormal"})
    counts = data.pivot_table(
        index="expected",
        columns="observed_diagnostic_label",
        values="video_id",
        aggfunc="count",
        fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", stacked=True, ax=ax, color=["#3b8c4a", "#cc8a24"])
    ax.set_title("06 Expected vs Observed")
    ax.set_ylabel("Video count")
    ax.set_xlabel("")
    ax.legend(title="Observed")
    return savefig(fig, figures / "06_expected_vs_observed.png")


def figure_confusion(predictions: pd.DataFrame, figures: Path) -> str:
    rows = predictions[predictions["expected_label"].isin(["normal", "abnormal"])]
    matrix = np.array(
        [
            [
                int(((rows["expected_label"] == "normal") & (rows["observed_diagnostic_label"] == "normal")).sum()),
                int(((rows["expected_label"] == "normal") & (rows["observed_diagnostic_label"] == "abnormal")).sum()),
            ],
            [
                int(((rows["expected_label"] == "abnormal") & (rows["observed_diagnostic_label"] == "normal")).sum()),
                int(((rows["expected_label"] == "abnormal") & (rows["observed_diagnostic_label"] == "abnormal")).sum()),
            ],
        ],
        dtype=int,
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Observed normal", "Observed abnormal"])
    ax.set_yticks([0, 1], ["Expected normal", "Expected abnormal"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=14)
    ax.set_title("07 Diagnostic Confusion Matrix\nsingle-source diagnostic only")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return savefig(fig, figures / "07_diagnostic_confusion_matrix.png")


def figure_runtime(runtime: pd.DataFrame, figures: Path) -> str:
    data = runtime[(runtime["video_id"] != "SYSTEM") & runtime["stage"].isin(["decode_check", *["preprocess", "resnet", "temporal", "normalize", "digest", "compare"]])].copy()
    if data.empty:
        data = runtime.copy()
    pivot = data.pivot_table(
        index="video_id",
        columns="stage",
        values="total_duration_seconds",
        aggfunc="sum",
        fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("08 Runtime by Stage")
    ax.set_ylabel("Seconds")
    ax.set_xlabel("")
    ax.legend(fontsize=8, ncols=3)
    return savefig(fig, figures / "08_runtime_by_stage.png")


def figure_bitrate_resolution(inventory: pd.DataFrame, summary: pd.DataFrame, figures: Path) -> str:
    merged = pd.merge(inventory, summary, on="video_id", suffixes=("", "_summary"))
    data = merged[merged["expected_category"] == "BENIGN"].copy()
    if data.empty:
        data = merged.copy()
    size = (data["width"].fillna(0).astype(float) * data["height"].fillna(0).astype(float) / 5000).clip(30, 300)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(data["bitrate"].fillna(0).astype(float) / 1_000_000.0, data["max_balanced_diagnostic_score"].fillna(0), s=size)
    for _, row in data.iterrows():
        ax.text(float(row.get("bitrate", 0) or 0) / 1_000_000.0, float(row.get("max_balanced_diagnostic_score", 0) or 0), str(row["video_id"]), fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Bitrate (Mbps)")
    ax.set_ylabel("Max balanced score")
    ax.set_title("09 Bitrate/Resolution vs Distance")
    return savefig(fig, figures / "09_bitrate_resolution_vs_distance.png")


def figure_sha(sha: pd.DataFrame, figures: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    values = sha["matches_original"].map(parse_bool).map({True: 1, False: 0})
    colors = ["#3b8c4a" if value == 1 else "#cc8a24" for value in values]
    ax.bar(short_labels(sha["video_id"].tolist()), values, color=colors)
    ax.set_yticks([0, 1], ["Mismatch", "Match"])
    ax.set_title("10 SHA-256 Baseline")
    ax.set_ylabel("Byte-level match to original")
    return savefig(fig, figures / "10_sha256_baseline.png")


def figure_attribution(attr: pd.DataFrame, figures: Path) -> str:
    pivot = attr.pivot_table(
        index="video_id",
        columns="attribution",
        values="abnormal_segment_count",
        aggfunc="sum",
        fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("11 Stream Attribution")
    ax.set_ylabel("Abnormal segment count")
    ax.set_xlabel("")
    ax.legend(fontsize=8)
    return savefig(fig, figures / "11_stream_attribution.png")


def figure_loo(loo: pd.DataFrame, figures: Path) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    values = loo["would_be_marked_abnormal"].map(parse_bool).astype(int)
    ax.bar(short_labels(loo["excluded_benign_video_id"].tolist()), values, color=["#cc3d3d" if value else "#3b8c4a" for value in values])
    ax.set_yticks([0, 1], ["Stable", "False abnormal"])
    ax.set_title("12 Leave-One-Benign-Out")
    ax.set_ylabel("Excluded benign result")
    return savefig(fig, figures / "12_leave_one_benign_out.png")


def read_frame(path: str | Path, timestamp_seconds: float) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_seconds) * 1000.0)
        success, frame = capture.read()
        if not success or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def placeholder(width: int = 640, height: int = 360) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (40, 40, 40)
    return image


def figure_localization(inventory: pd.DataFrame, segments: pd.DataFrame, figures: Path) -> list[str]:
    local_dir = figures / "localization"
    local_dir.mkdir(parents=True, exist_ok=True)
    original = inventory[inventory["video_id"] == REFERENCE_ID].iloc[0]
    outputs: list[str] = []
    tampered_ids = inventory[inventory["expected_category"] == "TAMPERED"]["video_id"].tolist()
    for video_id in tampered_ids:
        rows = segments[segments["video_id"] == video_id]
        if rows.empty:
            continue
        row = rows.sort_values("balanced_diagnostic_score", ascending=False).iloc[0]
        query = inventory[inventory["video_id"] == video_id].iloc[0]
        midpoint = (float(row["start_time_microseconds"]) + float(row["end_time_microseconds"])) / 2_000_000.0
        original_frame = read_frame(original["absolute_path"], midpoint)
        query_frame = read_frame(query["absolute_path"], midpoint)
        if original_frame is None:
            original_frame = placeholder()
        if query_frame is None:
            query_frame = placeholder(original_frame.shape[1], original_frame.shape[0])
        aligned_query = cv2.resize(query_frame, (original_frame.shape[1], original_frame.shape[0]))
        diff = np.abs(original_frame.astype(np.int16) - aligned_query.astype(np.int16)).astype(np.uint8)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        labels = [
            f"Original\nsegment {int(row['segment_id'])}, {midpoint:.2f}s",
            f"{video_id}\n{query['filename']}",
            "Absolute visual difference\nillustrative only",
        ]
        for ax, image, label in zip(axes, [original_frame, query_frame, diff], labels, strict=True):
            ax.imshow(image)
            ax.set_title(label, fontsize=10)
            ax.axis("off")
        fig.suptitle(
            "\n".join(
                textwrap.wrap(
                    (
                        f"{video_id}: ResNet={float(row['resnet_normalized_distance']):.4f}, "
                        f"Temporal={float(row['temporal_normalized_distance']):.4f}, "
                        f"Balanced={float(row['balanced_diagnostic_score']):.4f}, "
                        f"Attribution={row.get('diagnostic_attribution', 'n/a')}"
                    ),
                    width=120,
                )
            ),
            fontsize=11,
        )
        outputs.append(savefig(fig, local_dir / f"localization_{video_id}.png"))
    return outputs


def image_uri(path: str | Path) -> str:
    payload = Path(path).read_bytes()
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def status_span(label: str, state: str) -> str:
    return f'<span class="status {state}">{label}</span>'


def table_html(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    data = df.copy()
    if columns:
        data = data[[column for column in columns if column in data.columns]]
    if max_rows is not None:
        data = data.head(max_rows)
    return data.to_html(index=False, escape=True, classes="data-table")


def html_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    inventory: pd.DataFrame,
    predictions: pd.DataFrame,
    video_summary: pd.DataFrame,
    alignment: pd.DataFrame,
    segments: pd.DataFrame,
    sha: pd.DataFrame,
    runtime: pd.DataFrame,
    loo: pd.DataFrame,
    figures: list[str],
    localization: list[str],
) -> Path:
    detected_count = len(inventory)
    all_nine = detected_count == 9
    validations = summary.get("validations", {})
    benign = predictions[predictions["expected_category"] == "BENIGN"]
    tampered = predictions[predictions["expected_category"] == "TAMPERED"]
    benign_within = benign[benign["observed_diagnostic_label"] == "normal"]["video_id"].tolist()
    benign_exceeded = benign[benign["observed_diagnostic_label"] == "abnormal"]["video_id"].tolist()
    tampered_detected = tampered[tampered["observed_diagnostic_label"] == "abnormal"]["video_id"].tolist()
    tampered_missed = tampered[tampered["observed_diagnostic_label"] == "normal"]["video_id"].tolist()
    structural = predictions[predictions["structural_issue"].map(parse_bool)] if "structural_issue" in predictions else pd.DataFrame()
    thresholds_payload = summary.get("thresholds", {})
    metrics = summary.get("metrics", {})
    main_stream = (
        predictions[predictions["main_attribution"] != "none"]["main_attribution"].mode().iloc[0]
        if not predictions.empty and (predictions["main_attribution"] != "none").any()
        else "none"
    )
    figure_blocks = "\n".join(
        f'<figure><img src="{image_uri(path)}" alt="{Path(path).stem}"><figcaption>{Path(path).name}</figcaption></figure>'
        for path in figures
        if Path(path).exists()
    )
    localization_blocks = "\n".join(
        f'<figure><img src="{image_uri(path)}" alt="{Path(path).stem}"><figcaption>{Path(path).name}</figcaption></figure>'
        for path in localization
        if Path(path).exists()
    )
    timeline_blocks = "\n".join(
        f'<figure><img src="{image_uri(path)}" alt="{Path(path).stem}"><figcaption>{Path(path).name}</figcaption></figure>'
        for path in figures
        if Path(path).name.startswith("timeline_")
    )
    failure_rows = (
        predictions[predictions["correct"].map(lambda value: not parse_bool(value))]
        if not predictions.empty and "correct" in predictions
        else predictions
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Versions Evaluation Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2328; line-height: 1.45; }}
h1, h2 {{ color: #111827; }}
h1 {{ margin-bottom: 0.2rem; }}
.subtitle {{ color: #4b5563; margin-top: 0; }}
.status {{ display: inline-block; padding: 0.12rem 0.45rem; border-radius: 0.35rem; font-weight: 600; font-size: 0.86rem; }}
.passed {{ background: #dff3e4; color: #17612a; }}
.failed {{ background: #f8d7da; color: #8f1d2c; }}
.amber {{ background: #fff3cd; color: #7a5200; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; margin: 18px 0; }}
.metric {{ border: 1px solid #d1d5db; border-radius: 6px; padding: 12px; background: #fafafa; }}
.metric strong {{ display: block; font-size: 1.3rem; }}
.note {{ border-left: 4px solid #d99b24; padding: 10px 12px; background: #fff8e6; }}
figure {{ margin: 24px 0; }}
figure img {{ max-width: 100%; height: auto; border: 1px solid #d1d5db; }}
figcaption {{ color: #4b5563; font-size: 0.9rem; }}
.data-table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; margin: 12px 0 24px; }}
.data-table th, .data-table td {{ border: 1px solid #d1d5db; padding: 6px 8px; vertical-align: top; }}
.data-table th {{ background: #f3f4f6; text-align: left; }}
code {{ background: #f3f4f6; padding: 0.1rem 0.25rem; border-radius: 0.2rem; }}
</style>
</head>
<body>
<h1>Versions Evaluation Report</h1>
<p class="subtitle">Run ID: <code>{summary.get('run_id')}</code> | Engineering diagnostic only</p>

<h2>1. Executive Summary</h2>
<div class="grid">
<div class="metric"><strong>{detected_count}</strong> videos detected {status_span('all nine detected' if all_nine else 'count differs from nine', 'passed' if all_nine else 'amber')}</div>
<div class="metric"><strong>{validations.get('reference_hmac_verifies')}</strong> original HMAC verification {status_span('engineering passed' if validations.get('reference_hmac_verifies') else 'failed', 'passed' if validations.get('reference_hmac_verifies') else 'failed')}</div>
<div class="metric"><strong>{validations.get('self_comparison_exact_zero')}</strong> self-comparison zero distance {status_span('engineering passed' if validations.get('self_comparison_exact_zero') else 'failed', 'passed' if validations.get('self_comparison_exact_zero') else 'failed')}</div>
<div class="metric"><strong>No</strong> ready for final research claims {status_span('provisional', 'amber')}</div>
</div>
<p>Benign transformations within provisional range: {', '.join(benign_within) or 'none'}.</p>
<p>Benign transformations exceeding provisional range: {', '.join(benign_exceeded) or 'none'}.</p>
<p>Tampered versions detected: {', '.join(tampered_detected) or 'none'}. Missed: {', '.join(tampered_missed) or 'none'}.</p>
<p>Main contributing attribution across abnormal decisions: <strong>{main_stream}</strong>. Structural alignment findings: {', '.join(structural['video_id'].tolist()) if not structural.empty else 'none'}.</p>
<p class="note">The frozen normalization and quantization artifacts were originally fitted on V001-V003. This report is an engineering diagnostic and must not be cited as final tamper-detection accuracy.</p>

<h2>2. Experiment Objective</h2>
<p>Evaluate all supported videos under <code>data/versions/</code> against <code>original.mp4</code> using the existing Phase 1-7 digest pipeline, then add experiment-only thresholds and diagnostics.</p>

<h2>3. Dataset Inventory</h2>
{table_html(inventory, ['video_id', 'filename', 'expected_category', 'expected_label', 'transformation_type', 'duration_seconds', 'width', 'height', 'fps', 'codec', 'container_format', 'decode_valid'])}

<h2>4. Expected Normal and Expected Tampered Variants</h2>
{table_html(predictions, ['video_id', 'filename', 'expected_category', 'expected_label', 'observed_diagnostic_label', 'correct'])}

<h2>5. Pipeline Architecture</h2>
<p>Preprocessing creates metadata, five-second complete segments, and sampled frames. ResNet-18 and temporal streams are extracted on CPU, normalized with <code>DEV_NORMALIZATION_V1</code>, quantized with <code>DEV_QUANTIZATION_V1</code>, protected for the trusted reference with HMAC-SHA-256, and compared segment-by-segment using Hamming distance.</p>

<h2>6. Resource-Controlled Execution Settings</h2>
<p>CPU threads were limited to 2 through OMP/MKL/OpenBLAS/VECLIB/NumExpr environment variables. Heavy commands ran sequentially through <code>nice -n 10</code>. No multiprocessing, GPU/MPS, background jobs, sudo, or powermetrics were used.</p>

<h2>7. Processing Success and Failures</h2>
{table_html(read_table(output_dir, 'pipeline_status.csv'), ['video_id', 'filename', 'stage', 'status', 'cache_reused', 'duration_seconds', 'failure_reason'], max_rows=80)}

<h2>8. Original HMAC Verification</h2>
<p>Reference HMAC valid: <strong>{validations.get('reference_hmac_verifies')}</strong>. Self-comparison zero: <strong>{validations.get('self_comparison_exact_zero')}</strong>.</p>

<h2>9. Segment Alignment Findings</h2>
{table_html(alignment, ['video_id', 'filename', 'segment_id', 'state', 'start_time_delta_microseconds', 'end_time_delta_microseconds'], max_rows=120)}

<h2>10. ResNet Distance Results</h2>
{table_html(video_summary, ['video_id', 'filename', 'max_resnet_normalized_distance', 'mean_resnet_normalized_distance'], max_rows=20)}

<h2>11. Temporal Distance Results</h2>
{table_html(video_summary, ['video_id', 'filename', 'max_temporal_normalized_distance', 'mean_temporal_normalized_distance'], max_rows=20)}

<h2>12. Hybrid and Balanced-Score Results</h2>
{table_html(video_summary, ['video_id', 'filename', 'max_flat_hybrid_normalized_distance', 'max_balanced_diagnostic_score', 'mean_balanced_diagnostic_score'], max_rows=20)}

<h2>13. Provisional Threshold Methodology</h2>
<p>Formula: {thresholds_payload.get('formula')}. Benign segments used: {thresholds_payload.get('benign_segment_count')}. Tampered labels used: {thresholds_payload.get('tampered_labels_used')}.</p>
<pre>{thresholds_payload}</pre>

<h2>14. Per-Video Diagnostic Decisions</h2>
{table_html(predictions, ['video_id', 'filename', 'expected_label', 'observed_diagnostic_label', 'abnormal_segment_ids', 'main_attribution', 'structural_issue', 'correct'])}

<h2>15. Segment-Level Timelines</h2>
{timeline_blocks}

<h2>16. Visual Localization Examples</h2>
<p>These panels illustrate the segment with the highest digest difference for each expected tampered video. They are not pixel-level tamper localization.</p>
{localization_blocks}

<h2>17. SHA-256 Baseline</h2>
{table_html(sha, ['video_id', 'filename', 'matches_original', 'interpretation'])}

<h2>18. Diagnostic Confusion Matrix and Metrics</h2>
<pre>{metrics}</pre>

<h2>19. Runtime Analysis</h2>
{table_html(runtime, ['video_id', 'stage', 'runs', 'completed', 'failed', 'cache_reused_count', 'total_duration_seconds'], max_rows=120)}

<h2>20. Did the Current Pipeline Work as Expected?</h2>
<p>The engineering pipeline processed the trusted reference and produced HMAC-verified segment comparisons. The diagnostic layer separated some expected tampered cases from benign transformations, but any misses or false positives shown above remain factual limitations of this single-source run.</p>

<h2>21. Failure Cases</h2>
{table_html(failure_rows, ['video_id', 'expected_label', 'observed_diagnostic_label', 'abnormal_segment_ids', 'main_attribution', 'notes'])}

<h2>22. Limitations</h2>
<ul>
<li>Only one original source video was evaluated.</li>
<li>The benign threshold set is tiny.</li>
<li>Frozen artifacts were fitted on V001-V003 and are not final calibration artifacts.</li>
<li>No threshold was selected using tampered labels, and no final authenticity decision is claimed.</li>
</ul>

<h2>23. Recommended Next Research Steps</h2>
<p>Run the same workflow on a full multi-source calibration, validation, and held-out test dataset before selecting fixed or adaptive thresholds.</p>

<h2>Figures</h2>
{figure_blocks}
</body>
</html>
"""
    path = output_dir / "versions_evaluation_report.html"
    path.write_text(html, encoding="utf-8")
    return path


def generate_report(output_dir: Path, repo_root: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = read_json(output_dir / "versions_evaluation_summary.json")
    inventory = read_table(output_dir, "video_inventory.csv")
    pipeline = read_table(output_dir, "pipeline_status.csv")
    video_summary = read_table(output_dir, "video_distance_summary.csv")
    alignment = read_table(output_dir, "alignment_findings.csv")
    segments = read_table(output_dir, "segment_distances.csv")
    predictions = read_table(output_dir, "diagnostic_predictions.csv")
    attr = read_table(output_dir, "attribution_summary.csv")
    runtime = read_table(output_dir, "runtime_summary.csv")
    sha = read_table(output_dir, "sha256_baseline.csv")
    loo = read_table(output_dir, "leave_one_benign_out.csv")
    thresh = thresholds(summary)
    generated: list[str] = []
    if not inventory.empty:
        generated.append(figure_inventory(inventory, figures))
    if not pipeline.empty:
        generated.append(figure_pipeline(pipeline, figures))
    if not video_summary.empty:
        generated.append(figure_distance_bars(video_summary, thresh, figures, "max"))
        generated.append(figure_distance_bars(video_summary, thresh, figures, "mean"))
    if not segments.empty:
        generated.append(figure_heatmap(segments, figures))
        for _, group in segments.groupby("video_id"):
            generated.append(figure_timeline(group, thresh, figures))
    if not predictions.empty:
        generated.append(figure_expected_observed(predictions, figures))
        generated.append(figure_confusion(predictions, figures))
    if not runtime.empty:
        generated.append(figure_runtime(runtime, figures))
    if not inventory.empty and not video_summary.empty:
        generated.append(figure_bitrate_resolution(inventory, video_summary, figures))
    if not sha.empty:
        generated.append(figure_sha(sha, figures))
    if not attr.empty:
        generated.append(figure_attribution(attr, figures))
    if not loo.empty:
        generated.append(figure_loo(loo, figures))
    localization = figure_localization(inventory, segments, figures) if not inventory.empty and not segments.empty else []
    html = html_report(
        output_dir=output_dir,
        summary=summary,
        inventory=inventory,
        predictions=predictions,
        video_summary=video_summary,
        alignment=alignment,
        segments=segments,
        sha=sha,
        runtime=runtime,
        loo=loo,
        figures=generated,
        localization=localization,
    )
    missing = [path for path in generated + localization if not Path(path).exists()]
    if missing:
        raise RuntimeError(f"Missing generated figure files: {missing}")
    return {
        "html_report": str(html),
        "figures_folder": str(figures),
        "figures": generated,
        "localization_figures": localization,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/reports/versions_evaluation"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = generate_report(output_dir=args.output_dir, repo_root=args.repo_root)
    print(f"HTML report: {paths['html_report']}")
    print(f"Figures: {paths['figures_folder']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
