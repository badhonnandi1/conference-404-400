#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import html
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "data/tmp/matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


EXPERIMENT_ID = "multisource_versions_evaluation"


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


def label(values: list[Any]) -> list[str]:
    return [str(value).replace("SRC_", "").replace("_", "\n") for value in values]


def figure_dataset_inventory(source_registry: pd.DataFrame, inventory: pd.DataFrame, figures: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    if not source_registry.empty:
        counts = inventory.pivot_table(index="source_id", columns="expected_category", values="video_id", aggfunc="count", fill_value=0)
        counts.plot(kind="bar", stacked=True, ax=axes[0])
        axes[0].set_title("Source Folders and Variant Counts")
        axes[0].set_ylabel("Videos")
        axes[0].set_xlabel("")
    if not inventory.empty:
        transform_counts = inventory["transformation_type"].value_counts().sort_index()
        axes[1].bar(label(transform_counts.index.tolist()), transform_counts.values)
        axes[1].set_title("Transformation Inventory")
        axes[1].set_ylabel("Videos")
        axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle("01 Dataset Inventory")
    return savefig(fig, figures / "01_dataset_inventory.png")


def figure_pipeline_completion(pipeline: pd.DataFrame, inventory: pd.DataFrame, figures: Path) -> str:
    stages = ["preprocess", "resnet", "temporal", "normalize", "digest", "compare"]
    videos = inventory["video_id"].tolist() if not inventory.empty else sorted(pipeline["video_id"].dropna().unique().tolist())
    matrix = np.full((len(videos), len(stages)), np.nan)
    for row_i, video_id in enumerate(videos):
        for col_i, stage in enumerate(stages):
            matches = pipeline[(pipeline["video_id"] == video_id) & (pipeline["stage"] == stage)]
            if matches.empty:
                continue
            status = str(matches.iloc[-1].get("status"))
            matrix[row_i, col_i] = 1 if status == "completed" else 0
    fig, ax = plt.subplots(figsize=(12, max(6, 0.23 * len(videos))))
    cmap = ListedColormap(["#bd3f32", "#3f7f4d"])
    ax.imshow(np.nan_to_num(matrix, nan=0), aspect="auto", vmin=0, vmax=1, cmap=cmap)
    for row_i in range(matrix.shape[0]):
        for col_i in range(matrix.shape[1]):
            value = matrix[row_i, col_i]
            ax.text(col_i, row_i, "NA" if np.isnan(value) else ("OK" if value else "FAIL"), ha="center", va="center", color="white", fontsize=6)
    ax.set_yticks(np.arange(len(videos)), label(videos), fontsize=6)
    ax.set_xticks(np.arange(len(stages)), stages, rotation=30, ha="right")
    ax.set_title("02 Pipeline Completion Matrix")
    return savefig(fig, figures / "02_pipeline_completion_matrix.png")


def figure_metadata_overview(inventory: pd.DataFrame, figures: Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    if inventory.empty:
        return savefig(fig, figures / "03_metadata_overview.png")
    x = np.arange(len(inventory))
    labels = label(inventory["video_id"].tolist())
    axes[0, 0].bar(x, inventory["duration_seconds"].fillna(0))
    axes[0, 0].set_title("Duration")
    axes[0, 0].set_ylabel("Seconds")
    axes[0, 1].bar(x - 0.18, inventory["width"].fillna(0), width=0.36, label="width")
    axes[0, 1].bar(x + 0.18, inventory["height"].fillna(0), width=0.36, label="height")
    axes[0, 1].set_title("Resolution")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].bar(x, inventory["fps"].fillna(0))
    axes[1, 0].set_title("FPS")
    bitrate = inventory["bitrate"].fillna(0).astype(float) / 1_000_000.0
    axes[1, 1].bar(x, bitrate)
    axes[1, 1].set_title("Bitrate and Codec")
    axes[1, 1].set_ylabel("Mbps")
    for i, codec in enumerate(inventory["codec"].fillna("unknown")):
        axes[1, 1].text(i, bitrate.iloc[i], str(codec), rotation=90, fontsize=5, ha="center", va="bottom")
    for ax in axes.ravel():
        ax.set_xticks(x, labels, rotation=90, fontsize=5)
    fig.suptitle("03 Metadata Overview")
    return savefig(fig, figures / "03_metadata_overview.png")


def merge_predictions_summary(predictions: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return summary.copy()
    if summary.empty:
        return predictions.copy()
    return pd.merge(predictions, summary, on=["source_id", "video_id", "filename", "transformation_type", "expected_label"], how="left", suffixes=("", "_summary"))


def figure_max_balanced(summary: pd.DataFrame, fold_thresholds: pd.DataFrame, figures: Path) -> str:
    data = summary.copy()
    fig, ax = plt.subplots(figsize=(15, 6))
    if not data.empty:
        data = data.sort_values(["source_id", "transformation_type"])
        x = np.arange(len(data))
        ax.bar(x, data["max_balanced_diagnostic_score"].fillna(0), color="#5c7c95")
        ax.set_xticks(x, label(data["video_id"].tolist()), rotation=90, fontsize=6)
        for source, group in data.groupby("source_id"):
            match = fold_thresholds[fold_thresholds["held_out_source"] == source] if not fold_thresholds.empty else pd.DataFrame()
            if not match.empty:
                ax.hlines(float(match.iloc[0]["balanced_threshold"]), group.index.min(), group.index.max(), colors="#bd3f32", linestyles="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Max balanced score")
    ax.set_title("04 Max Balanced Distance by Video")
    return savefig(fig, figures / "04_max_balanced_distance_by_video.png")


def figure_stream_distances(summary: pd.DataFrame, figures: Path) -> str:
    data = summary.copy()
    fig, ax = plt.subplots(figsize=(15, 6))
    if not data.empty:
        data = data.sort_values(["source_id", "transformation_type"])
        x = np.arange(len(data))
        width = 0.38
        ax.bar(x - width / 2, data["max_resnet_normalized_distance"].fillna(0), width=width, label="ResNet")
        ax.bar(x + width / 2, data["max_temporal_normalized_distance"].fillna(0), width=width, label="Temporal")
        ax.set_xticks(x, label(data["video_id"].tolist()), rotation=90, fontsize=6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Max normalized distance")
    ax.legend()
    ax.set_title("05 Max Stream Distances")
    return savefig(fig, figures / "05_max_stream_distances.png")


def figure_mean_by_transformation(summary: pd.DataFrame, figures: Path) -> str:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    if not summary.empty:
        grouped = summary.groupby("transformation_type")["mean_balanced_diagnostic_score"].mean().sort_values()
        ax.bar(label(grouped.index.tolist()), grouped.values, color="#7b6c9c")
        ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean balanced score")
    ax.set_title("06 Mean Distance by Transformation")
    return savefig(fig, figures / "06_mean_distance_by_transformation.png")


def figure_segment_heatmap(segments: pd.DataFrame, figures: Path) -> str:
    fig, ax = plt.subplots(figsize=(14, max(6, 0.25 * max(1, segments["video_id"].nunique() if not segments.empty else 1))))
    if not segments.empty:
        pivot = segments.pivot_table(index="video_id", columns="segment_id", values="balanced_diagnostic_score", aggfunc="max").sort_index()
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_yticks(np.arange(len(pivot.index)), label(pivot.index.tolist()), fontsize=6)
        ax.set_xticks(np.arange(len(pivot.columns)), [str(int(col)) for col in pivot.columns])
        fig.colorbar(image, ax=ax, label="Balanced score")
    ax.set_title("07 Segment Score Heatmap")
    ax.set_xlabel("Segment ID")
    return savefig(fig, figures / "07_segment_score_heatmap.png")


def figure_expected_observed(predictions: pd.DataFrame, figures: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    if not predictions.empty:
        counts = predictions.pivot_table(index="expected_label", columns="observed_diagnostic_label", values="video_id", aggfunc="count", fill_value=0)
        counts.plot(kind="bar", stacked=True, ax=ax, color=["#bd3f32", "#3f7f4d"])
    ax.set_title("08 Expected vs Observed")
    ax.set_ylabel("Videos")
    ax.set_xlabel("")
    return savefig(fig, figures / "08_expected_vs_observed.png")


def figure_confusion(predictions: pd.DataFrame, figures: Path) -> str:
    rows = predictions[predictions["expected_label"].isin(["normal", "abnormal"])] if not predictions.empty else pd.DataFrame()
    matrix = np.array(
        [
            [
                int(((rows["expected_label"] == "normal") & (rows["observed_diagnostic_label"] == "normal")).sum()) if not rows.empty else 0,
                int(((rows["expected_label"] == "normal") & (rows["observed_diagnostic_label"] == "abnormal")).sum()) if not rows.empty else 0,
            ],
            [
                int(((rows["expected_label"] == "abnormal") & (rows["observed_diagnostic_label"] == "normal")).sum()) if not rows.empty else 0,
                int(((rows["expected_label"] == "abnormal") & (rows["observed_diagnostic_label"] == "abnormal")).sum()) if not rows.empty else 0,
            ],
        ],
        dtype=int,
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Observed normal", "Observed abnormal"], rotation=20, ha="right")
    ax.set_yticks([0, 1], ["Expected normal", "Expected abnormal"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=14)
    ax.set_title("09 Overall Confusion Matrix\nDiagnostic LOSO Evaluation")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return savefig(fig, figures / "09_overall_confusion_matrix.png")


def figure_per_source_accuracy(per_source: pd.DataFrame, figures: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    if not per_source.empty:
        x = np.arange(len(per_source))
        width = 0.25
        ax.bar(x - width, per_source["accuracy"], width=width, label="Accuracy")
        ax.bar(x, per_source["recall"], width=width, label="Recall")
        ax.bar(x + width, per_source["f1"], width=width, label="F1")
        ax.set_xticks(x, label(per_source["source_id"].tolist()))
        ax.legend()
    ax.set_ylim(0, 1)
    ax.set_title("10 Per-Source Accuracy")
    return savefig(fig, figures / "10_per_source_accuracy.png")


def figure_tamper_detection(per_tamper: pd.DataFrame, figures: Path) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    if not per_tamper.empty:
        ax.bar(label(per_tamper["transformation_type"].tolist()), per_tamper["detection_rate"], color="#8b4f6f")
        ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0, 1)
    ax.set_title("11 Tamper Detection Rate")
    return savefig(fig, figures / "11_tamper_detection_rate.png")


def figure_benign_frr(per_benign: pd.DataFrame, figures: Path) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    if not per_benign.empty:
        ax.bar(label(per_benign["transformation_type"].tolist()), per_benign["false_rejection_rate"], color="#bb7f39")
        ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0, 1)
    ax.set_title("12 Benign False Rejection Rate")
    return savefig(fig, figures / "12_benign_false_rejection_rate.png")


def figure_attribution(attribution: pd.DataFrame, figures: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if not attribution.empty:
        totals = attribution.groupby("attribution")["abnormal_segment_count"].sum().sort_index()
        ax.bar(label(totals.index.tolist()), totals.values, color="#4d7983")
    ax.set_title("13 Stream Attribution")
    ax.set_ylabel("Abnormal segments")
    return savefig(fig, figures / "13_stream_attribution.png")


def figure_thresholds(folds: pd.DataFrame, figures: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    if not folds.empty:
        x = np.arange(len(folds))
        width = 0.25
        ax.bar(x - width, folds["resnet_threshold"], width=width, label="ResNet")
        ax.bar(x, folds["temporal_threshold"], width=width, label="Temporal")
        ax.bar(x + width, folds["balanced_threshold"], width=width, label="Balanced")
        ax.set_xticks(x, label(folds["held_out_source"].tolist()))
        ax.legend()
    ax.set_ylim(0, 1)
    ax.set_title("14 Thresholds by Fold")
    return savefig(fig, figures / "14_thresholds_by_fold.png")


def figure_runtime_by_source(runtime: pd.DataFrame, figures: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    if not runtime.empty:
        data = runtime[~runtime["stage"].astype(str).str.startswith("sleep")].groupby("source_id")["total_duration_seconds"].sum().sort_index()
        ax.bar(label(data.index.tolist()), data.values, color="#58769a")
    ax.set_title("15 Runtime by Source")
    ax.set_ylabel("Seconds")
    return savefig(fig, figures / "15_runtime_by_source.png")


def figure_runtime_by_stage(runtime: pd.DataFrame, figures: Path) -> str:
    fig, ax = plt.subplots(figsize=(11, 5))
    if not runtime.empty:
        data = runtime.groupby("stage")["total_duration_seconds"].sum().sort_values()
        ax.bar(label(data.index.tolist()), data.values, color="#6b7d55")
        ax.tick_params(axis="x", rotation=45)
    ax.set_title("16 Runtime by Stage")
    ax.set_ylabel("Seconds")
    return savefig(fig, figures / "16_runtime_by_stage.png")


def figure_sha256(sha: pd.DataFrame, figures: Path) -> str:
    fig, ax = plt.subplots(figsize=(15, 5))
    if not sha.empty:
        values = sha["matches_original"].map(parse_bool).astype(int)
        ax.bar(label(sha["video_id"].tolist()), values, color=["#3f7f4d" if value else "#bd8b32" for value in values])
        ax.set_yticks([0, 1], ["Mismatch", "Match"])
        ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.set_title("17 SHA-256 Baseline")
    return savefig(fig, figures / "17_sha256_baseline.png")


def figure_blur_analysis(summary: pd.DataFrame, folds: pd.DataFrame, figures: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    blur = summary[summary["transformation_type"] == "blur"] if not summary.empty else pd.DataFrame()
    if not blur.empty:
        x = np.arange(len(blur))
        ax.bar(x, blur["max_balanced_diagnostic_score"].fillna(0), color="#80639a")
        ax.set_xticks(x, label(blur["source_id"].tolist()))
        for i, (_, row) in enumerate(blur.iterrows()):
            match = folds[folds["held_out_source"] == row["source_id"]] if not folds.empty else pd.DataFrame()
            if not match.empty:
                ax.hlines(float(match.iloc[0]["balanced_threshold"]), i - 0.4, i + 0.4, colors="#bd3f32", linestyles="--")
    ax.set_ylim(0, 1)
    ax.set_title("18 Blur Analysis")
    return savefig(fig, figures / "18_blur_analysis.png")


def figure_structural_tamper(alignment: pd.DataFrame, figures: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    if not alignment.empty:
        data = alignment[alignment["transformation_type"].isin(["frame_deletion", "frame_insertion"])]
        data = data[data["state"] != "matched"].groupby(["source_id", "transformation_type"]).size()
        labels = [f"{idx[0]}\n{idx[1]}" for idx in data.index]
        ax.bar(labels, data.values, color="#9a5e4f")
        ax.tick_params(axis="x", rotation=45)
    ax.set_title("19 Structural Tamper Analysis")
    ax.set_ylabel("Missing/extra/timestamp findings")
    return savefig(fig, figures / "19_structural_tamper_analysis.png")


def figure_generalization(predictions: pd.DataFrame, figures: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    if not predictions.empty:
        data = predictions.assign(correct_int=predictions["correct"].map(parse_bool).astype(int))
        pivot = data.pivot_table(index="source_id", columns="correct_int", values="video_id", aggfunc="count", fill_value=0)
        pivot = pivot.rename(columns={0: "incorrect", 1: "correct"})
        pivot.plot(kind="bar", stacked=True, ax=ax, color=["#bd3f32", "#3f7f4d"])
    ax.set_title("20 Generalization Summary")
    ax.set_ylabel("Videos")
    ax.set_xlabel("")
    return savefig(fig, figures / "20_generalization_summary.png")


def figure_timeline(group: pd.DataFrame, figures: Path) -> str:
    video_id = str(group["video_id"].iloc[0])
    source_id = str(group["source_id"].iloc[0])
    transform = str(group["transformation_type"].iloc[0])
    group = group.sort_values("segment_id")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = group["segment_id"].astype(int)
    ax.plot(x, group["resnet_normalized_distance"], marker="o", label="ResNet")
    ax.plot(x, group["temporal_normalized_distance"], marker="s", label="Temporal")
    ax.plot(x, group["balanced_diagnostic_score"], marker="^", label="Balanced")
    ax.axhline(float(group["resnet_threshold"].iloc[0]), color="#1f77b4", linestyle="--", linewidth=1)
    ax.axhline(float(group["temporal_threshold"].iloc[0]), color="#ff7f0e", linestyle="--", linewidth=1)
    ax.axhline(float(group["balanced_threshold"].iloc[0]), color="#2ca02c", linestyle="--", linewidth=1)
    abnormal = group[group["segment_provisionally_abnormal"].map(parse_bool)]
    if not abnormal.empty:
        ax.scatter(abnormal["segment_id"], abnormal["balanced_diagnostic_score"], s=100, facecolors="none", edgecolors="#bd3f32", linewidths=2, label="Abnormal")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Segment ID")
    ax.set_ylabel("Normalized distance / score")
    ax.set_title(f"{source_id} {video_id} {transform}")
    ax.legend(ncols=4, fontsize=8)
    return savefig(fig, figures / "timelines" / f"{video_id}_timeline.png")


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
    image[:, :] = (42, 42, 42)
    return image


def figure_localization(inventory: pd.DataFrame, segments: pd.DataFrame, figures: Path) -> list[str]:
    outputs: list[str] = []
    if inventory.empty or segments.empty:
        return outputs
    originals = inventory[inventory["expected_category"] == "reference"].set_index("source_id")
    tampered = inventory[inventory["expected_category"] == "tampered"]
    for _, query in tampered.iterrows():
        rows = segments[segments["video_id"] == query["video_id"]]
        if rows.empty or query["source_id"] not in originals.index:
            continue
        source_id = str(query["source_id"])
        row = rows.sort_values("balanced_diagnostic_score", ascending=False).iloc[0]
        original = originals.loc[source_id]
        midpoint = (float(row["start_time_microseconds"]) + float(row["end_time_microseconds"])) / 2_000_000.0
        original_frame = read_frame(original["absolute_path"], midpoint)
        if original_frame is None:
            original_frame = placeholder()
        query_frame = read_frame(query["absolute_path"], midpoint)
        if query_frame is None:
            query_frame = placeholder(original_frame.shape[1], original_frame.shape[0])
        aligned_query = cv2.resize(query_frame, (original_frame.shape[1], original_frame.shape[0]))
        diff = np.abs(original_frame.astype(np.int16) - aligned_query.astype(np.int16)).astype(np.uint8)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        panels = [
            (original_frame, f"Original\n{source_id}, segment {int(row['segment_id'])}, {midpoint:.2f}s"),
            (aligned_query, f"Query\n{query['video_id']}"),
            (diff, "Absolute difference\nillustrative only"),
        ]
        for ax, (image, title) in zip(axes, panels, strict=True):
            ax.imshow(image)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(
            f"{query['transformation_type']}: ResNet={float(row['resnet_normalized_distance']):.4f}, "
            f"Temporal={float(row['temporal_normalized_distance']):.4f}, "
            f"Balanced={float(row['balanced_diagnostic_score']):.4f}, "
            f"Attribution={row.get('diagnostic_attribution', 'n/a')}"
        )
        outputs.append(savefig(fig, figures / "localization" / source_id / f"{query['video_id']}_localization.png"))
    return outputs


def image_uri(path: str | Path) -> str:
    payload = Path(path).read_bytes()
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def table_html(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if df.empty:
        return "<p>No rows.</p>"
    data = df.copy()
    if columns:
        data = data[[column for column in columns if column in data.columns]]
    if max_rows is not None:
        data = data.head(max_rows)
    return data.to_html(index=False, escape=True, classes="data-table")


def metric_value(metrics: dict[str, Any], name: str) -> str:
    value = metrics.get(name)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def html_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    source_registry: pd.DataFrame,
    inventory: pd.DataFrame,
    predictions: pd.DataFrame,
    per_source: pd.DataFrame,
    per_tamper: pd.DataFrame,
    per_benign: pd.DataFrame,
    alignment: pd.DataFrame,
    sha: pd.DataFrame,
    runtime: pd.DataFrame,
    figures: list[str],
    timelines: list[str],
    localization: list[str],
) -> Path:
    metrics = summary.get("overall_metrics", {})
    validations = summary.get("validations", {})
    source_count = int(source_registry["source_id"].nunique()) if not source_registry.empty else 0
    valid_sources = int((source_registry["status"] == "valid").sum()) if not source_registry.empty else 0
    processed_videos = len(predictions) if not predictions.empty else 0
    all_refs = int((inventory["expected_category"] == "reference").sum()) if not inventory.empty else 0
    false_pos = predictions[(predictions["expected_label"] == "normal") & (predictions["observed_diagnostic_label"] == "abnormal")] if not predictions.empty else pd.DataFrame()
    false_neg = predictions[(predictions["expected_label"] == "abnormal") & (predictions["observed_diagnostic_label"] == "normal")] if not predictions.empty else pd.DataFrame()
    main_stream = (
        predictions[predictions["main_attribution"] != "none"]["main_attribution"].mode().iloc[0]
        if not predictions.empty and (predictions["main_attribution"] != "none").any()
        else "none"
    )
    figure_blocks = "\n".join(
        f'<figure><img src="{image_uri(path)}" alt="{html.escape(Path(path).stem)}"><figcaption>{html.escape(Path(path).name)}</figcaption></figure>'
        for path in figures
        if Path(path).exists()
    )
    timeline_blocks = "\n".join(
        f'<figure><img src="{image_uri(path)}" alt="{html.escape(Path(path).stem)}"><figcaption>{html.escape(Path(path).name)}</figcaption></figure>'
        for path in timelines
        if Path(path).exists()
    )
    localization_blocks = "\n".join(
        f'<figure><img src="{image_uri(path)}" alt="{html.escape(Path(path).stem)}"><figcaption>{html.escape(Path(path).name)}</figcaption></figure>'
        for path in localization
        if Path(path).exists()
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Multi-Source Versions Evaluation Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2328; line-height: 1.45; }}
h1, h2 {{ color: #111827; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 18px 0; }}
.metric {{ border: 1px solid #d1d5db; border-radius: 6px; padding: 12px; background: #fafafa; }}
.metric strong {{ display: block; font-size: 1.35rem; }}
.green {{ background: #dff3e4; color: #17612a; padding: 0.1rem 0.35rem; border-radius: 4px; }}
.amber {{ background: #fff3cd; color: #7a5200; padding: 0.1rem 0.35rem; border-radius: 4px; }}
.red {{ background: #f8d7da; color: #8f1d2c; padding: 0.1rem 0.35rem; border-radius: 4px; }}
.note {{ border-left: 4px solid #d99b24; padding: 10px 12px; background: #fff8e6; }}
figure img {{ max-width: 100%; height: auto; border: 1px solid #d1d5db; }}
figcaption {{ color: #4b5563; font-size: 0.9rem; }}
.data-table {{ border-collapse: collapse; width: 100%; font-size: 0.84rem; margin: 12px 0 24px; }}
.data-table th, .data-table td {{ border: 1px solid #d1d5db; padding: 6px 8px; vertical-align: top; }}
.data-table th {{ background: #f3f4f6; text-align: left; }}
code {{ background: #f3f4f6; padding: 0.1rem 0.25rem; border-radius: 0.2rem; }}
</style>
</head>
<body>
<h1>Multi-Source Versions Evaluation Report</h1>
<p>Run ID: <code>{html.escape(str(summary.get('run_id')))}</code></p>
<p class="note">MULTI-SOURCE DIAGNOSTIC METRICS - NOT FINAL HELD-OUT RESEARCH RESULTS. Green means engineering validation passed, not proof of authenticity.</p>

<h2>1. Executive Summary</h2>
<div class="grid">
<div class="metric"><strong>{source_count}</strong> source folders detected</div>
<div class="metric"><strong>{valid_sources}</strong> valid source groups</div>
<div class="metric"><strong>{processed_videos}</strong> videos with LOSO predictions</div>
<div class="metric"><strong>{all_refs}</strong> originals identified</div>
<div class="metric"><strong>{validations.get('all_successful_original_hmac_records_verify')}</strong> original HMAC records verify</div>
<div class="metric"><strong>{validations.get('all_self_comparisons_exact_zero')}</strong> self-comparisons zero</div>
<div class="metric"><strong>{metric_value(metrics, 'accuracy')}</strong> overall LOSO diagnostic accuracy</div>
<div class="metric"><strong>{main_stream}</strong> main contributing stream</div>
</div>
<p>Blur/deletion/insertion/replacement rates and benign false-rejection rates are shown in the per-transformation sections below. This run improves diagnostic breadth compared with the previous one-source test because thresholds are evaluated source-wise, but it still cannot be reported as final research performance.</p>

<h2>2. Repository and Experiment Configuration</h2>
<p>Git commit: <code>{html.escape(str(summary.get('git_commit')))}</code>. Input root: <code>{html.escape(str(summary.get('input_root')))}</code>. Output directory: <code>{html.escape(str(summary.get('output_dir')))}</code>.</p>

<h2>3. Six-Source Dataset Inventory</h2>
{table_html(source_registry, ['source_id', 'source_folder', 'status', 'original_candidate', 'missing_expected_files', 'duplicate_classifications', 'ambiguities', 'notes'])}

<h2>4. File Classification Rules</h2>
<p>References prefer exact folder-base MP4 files, with an <code>original.mp4</code> fallback for folders such as <code>vid01</code>. Benign and tampered variants are classified from explicit filename patterns and extensions; ambiguous matches are reported, not silently treated as references.</p>

<h2>5. Pipeline Architecture</h2>
<p>Video discovery, FFprobe metadata, five-second segmentation, CPU ResNet-18 extraction, temporal frame-difference extraction, frozen normalization, frozen quantization, HMAC reference protection, and segment-level Hamming comparison were executed source by source.</p>

<h2>6. Resource-Control Policy</h2>
<p>Heavy stages ran sequentially on CPU with two computational threads, no CUDA, no MPS request, no multiprocessing, reduced process priority, thermal/load snapshots, and mandatory sleeps according to the selected profile.</p>

<h2>7. Processing Status</h2>
{table_html(read_table(output_dir, 'pipeline_status.csv'), ['source_id', 'video_id', 'filename', 'stage', 'status', 'cache_reused', 'duration_seconds', 'failure_reason'], 120)}

<h2>8. Original Reference HMAC Results</h2>
<pre>{html.escape(str(summary.get('hmac_results')))}</pre>

<h2>9. Original Self-Comparison Validation</h2>
<pre>{html.escape(str(validations.get('self_comparison_exact_zero_by_reference')))}</pre>

<h2>10. Benign Transformation Results</h2>
{table_html(predictions[predictions['expected_label'] == 'normal'] if not predictions.empty else predictions, ['source_id', 'video_id', 'transformation_type', 'observed_diagnostic_label', 'correct', 'main_attribution'])}

<h2>11. Tampered Variant Results</h2>
{table_html(predictions[predictions['expected_label'] == 'abnormal'] if not predictions.empty else predictions, ['source_id', 'video_id', 'transformation_type', 'observed_diagnostic_label', 'correct', 'abnormal_segment_ids', 'main_attribution'])}

<h2>12. Pooled Threshold Analysis</h2>
<pre>{html.escape(str(summary.get('pooled_thresholds')))}</pre>

<h2>13. Leave-One-Source-Out Methodology</h2>
<p>For each fold, the held-out source was excluded from threshold fitting. Only benign variants from other sources contributed to thresholds. No tampered segment influenced threshold calculation.</p>

<h2>14. Thresholds by Source Fold</h2>
{table_html(read_table(output_dir, 'leave_one_source_out_thresholds.csv'))}

<h2>15. Overall Diagnostic Metrics</h2>
<pre>{html.escape(str(metrics))}</pre>

<h2>16. Per-Source Metrics</h2>
{table_html(per_source)}

<h2>17. Per-Tampering-Type Results</h2>
{table_html(per_tamper)}

<h2>18. Per-Benign-Transformation Results</h2>
{table_html(per_benign)}

<h2>19. Segment Timelines</h2>
{timeline_blocks}

<h2>20. Localization Examples</h2>
<p>These figures illustrate the highest-scoring five-second segment only; they do not claim pixel-level localization.</p>
{localization_blocks}

<h2>21. Stream Attribution</h2>
{table_html(read_table(output_dir, 'attribution_summary.csv'), ['source_id', 'video_id', 'transformation_type', 'attribution', 'abnormal_segment_count'], 160)}

<h2>22. Alignment and Structural Findings</h2>
{table_html(alignment, ['source_id', 'video_id', 'transformation_type', 'segment_id', 'state', 'start_time_delta_microseconds', 'end_time_delta_microseconds'], 160)}

<h2>23. SHA-256 Baseline</h2>
{table_html(sha, ['source_id', 'video_id', 'filename', 'transformation_type', 'matches_original', 'interpretation'])}

<h2>24. Runtime Analysis</h2>
{table_html(runtime, ['source_id', 'video_id', 'stage', 'runs', 'completed', 'failed', 'cache_reused_count', 'total_duration_seconds'], 160)}

<h2>25. Failure Cases</h2>
{table_html(read_table(output_dir, 'failures.csv'))}

<h2>26. Comparison With Previous Single-Source Result</h2>
<p>The previous workflow used one source and a pooled benign threshold. This report evaluates source-wise generalization with LOSO folds, so confidence in broad behavior is stronger, but it remains diagnostic because the frozen artifacts came from V001-V003 and no final calibration/validation/held-out split has been performed.</p>

<h2>27. Current Limitations</h2>
<p>Frozen DEV artifacts were fitted using V001-V003, thresholds are provisional, compression-adaptive thresholds are not final, and this is not a held-out research test.</p>

<h2>28. Final Conclusion</h2>
<p>The implementation can run isolated multi-source diagnostics and expose failures, false positives, false negatives, attribution, structural issues, and threshold stability. These values must not be claimed as final research performance.</p>

<h2>29. Recommended Next Research Steps</h2>
<p>Build the final source-wise calibration, validation and held-out test split, then refit normalization, quantization and compression-adaptive thresholds using calibration/validation sources only.</p>

<h2>Figures</h2>
{figure_blocks}

<h2>False Positives</h2>
{table_html(false_pos, ['source_id', 'video_id', 'transformation_type', 'observed_diagnostic_label', 'main_attribution'])}

<h2>False Negatives</h2>
{table_html(false_neg, ['source_id', 'video_id', 'transformation_type', 'observed_diagnostic_label', 'main_attribution'])}
</body>
</html>
"""
    path = output_dir / "multisource_versions_evaluation_report.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def generate_report(output_dir: Path, repo_root: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = read_json(output_dir / "multisource_versions_evaluation_summary.json")
    source_registry = read_table(output_dir, "source_registry.csv")
    inventory = read_table(output_dir, "video_inventory.csv")
    pipeline = read_table(output_dir, "pipeline_status.csv")
    metadata = read_table(output_dir, "metadata_comparison.csv")
    segments = read_table(output_dir, "segment_distances.csv")
    alignment = read_table(output_dir, "alignment_findings.csv")
    per_video = read_table(output_dir, "per_video_summary.csv")
    predictions = read_table(output_dir, "diagnostic_predictions.csv")
    folds = read_table(output_dir, "leave_one_source_out_thresholds.csv")
    per_source = read_table(output_dir, "per_source_metrics.csv")
    per_tamper = read_table(output_dir, "per_tamper_metrics.csv")
    per_benign = read_table(output_dir, "per_benign_metrics.csv")
    attribution = read_table(output_dir, "attribution_summary.csv")
    runtime = read_table(output_dir, "runtime_summary.csv")
    sha = read_table(output_dir, "sha256_baseline.csv")
    merged = merge_predictions_summary(predictions, per_video)
    generated = [
        figure_dataset_inventory(source_registry, inventory, figures),
        figure_pipeline_completion(pipeline, inventory, figures),
        figure_metadata_overview(inventory, figures),
        figure_max_balanced(per_video, folds, figures),
        figure_stream_distances(per_video, figures),
        figure_mean_by_transformation(per_video, figures),
        figure_segment_heatmap(segments, figures),
        figure_expected_observed(predictions, figures),
        figure_confusion(predictions, figures),
        figure_per_source_accuracy(per_source, figures),
        figure_tamper_detection(per_tamper, figures),
        figure_benign_frr(per_benign, figures),
        figure_attribution(attribution, figures),
        figure_thresholds(folds, figures),
        figure_runtime_by_source(runtime, figures),
        figure_runtime_by_stage(runtime, figures),
        figure_sha256(sha, figures),
        figure_blur_analysis(per_video, folds, figures),
        figure_structural_tamper(alignment, figures),
        figure_generalization(predictions, figures),
    ]
    timelines: list[str] = []
    if not segments.empty:
        query_segments = segments[segments["transformation_type"] != "trusted_reference"]
        for _, group in query_segments.groupby("video_id"):
            timelines.append(figure_timeline(group, figures))
    localization = figure_localization(inventory, segments, figures)
    report = html_report(
        output_dir=output_dir,
        summary=summary,
        source_registry=source_registry,
        inventory=inventory,
        predictions=predictions,
        per_source=per_source,
        per_tamper=per_tamper,
        per_benign=per_benign,
        alignment=alignment,
        sha=sha,
        runtime=runtime,
        figures=generated,
        timelines=timelines,
        localization=localization,
    )
    missing = [path for path in generated + timelines + localization if not Path(path).exists()]
    if missing:
        raise RuntimeError(f"Missing generated report artifacts: {missing}")
    return {
        "html_report": str(report),
        "figures_folder": str(figures),
        "figures": generated,
        "timelines": timelines,
        "localization_figures": localization,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/reports/multisource_versions_evaluation"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = generate_report(args.output_dir, args.repo_root)
    print(f"HTML report: {paths['html_report']}")
    print(f"Figures: {paths['figures_folder']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
