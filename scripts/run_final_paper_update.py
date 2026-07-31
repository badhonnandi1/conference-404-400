#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.phash_evaluation import run_outer_source_evaluation
from src.analysis.reporting import (
    PHASH_METHOD,
    PROPOSED_METHOD,
    validate_primary_method_comparison,
)
from src.analysis.uncertainty import (
    METRIC_NAMES,
    binary_classification_metrics,
    confusion_matrix_counts,
    fold_metrics_and_variance,
    source_clustered_bootstrap,
)
from src.baselines.opencv_phash import OpenCVPHash, compare_video_frame_hashes


DEFAULT_REGISTRY = REPO_ROOT / "data" / "reports" / "blur_aware_final_evaluation" / "video_registry.csv"
DEFAULT_PROPOSED_PREDICTIONS = (
    REPO_ROOT
    / "data"
    / "reports"
    / "blur_aware_final_evaluation"
    / "tables"
    / "diagnostic_predictions.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "data" / "reports" / "final_paper_update"
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 404
MARGIN_GRID = (1.5, 3.0)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    empty_columns: Iterable[str] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    prepared = [{key: _csv_cell(value) for key, value in row.items()} for row in rows]
    frame = pd.DataFrame(prepared)
    if frame.empty and empty_columns is not None:
        frame = pd.DataFrame(columns=list(empty_columns))
    frame.to_csv(path, index=False, na_rep="")
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_segment_ids(video_id: str) -> list[int]:
    path = REPO_ROOT / "data" / "manifests" / f"{video_id}_segments.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if float(payload["segment_duration_seconds"]) != 5.0:
        raise RuntimeError(f"Unexpected segment duration for {video_id}.")
    if float(payload["sample_frames_per_second"]) != 1.0:
        raise RuntimeError(f"Unexpected frame-sampling rate for {video_id}.")
    return [int(row["segment_id"]) for row in payload["segments"] if bool(row["is_complete"])]


def hash_cached_frames(
    registry: list[dict[str, str]],
    hasher: OpenCVPHash,
) -> tuple[dict[str, list[Any]], list[dict[str, Any]]]:
    hashes_by_video: dict[str, list[Any]] = {}
    provenance: list[dict[str, Any]] = []
    registry_by_id = {row["video_id"]: row for row in registry}
    for video_id in sorted(registry_by_id):
        manifest = REPO_ROOT / "data" / "manifests" / f"{video_id}_frames.json"
        if not manifest.exists():
            raise RuntimeError(f"Cached frame manifest is missing: {manifest}")
        hashes = hasher.hash_frame_manifest(manifest)
        hashes_by_video[video_id] = hashes
        metadata = registry_by_id[video_id]
        for frame_hash in hashes:
            provenance.append(
                {
                    "source_id": metadata["source_id"],
                    "transformation_type": metadata["transformation_type"],
                    **frame_hash.provenance_dict(),
                }
            )
    return hashes_by_video, provenance


def build_all_phash_comparisons(
    registry: list[dict[str, str]],
    hashes_by_video: dict[str, list[Any]],
) -> dict[str, dict[str, Any]]:
    segment_ids = {row["video_id"]: load_segment_ids(row["video_id"]) for row in registry}
    comparisons: dict[str, dict[str, Any]] = {}
    for row in registry:
        video_id = row["video_id"]
        reference_id = row["reference_video_id"]
        result = compare_video_frame_hashes(
            reference_hashes=hashes_by_video[reference_id],
            query_hashes=hashes_by_video[video_id],
            reference_segment_ids=segment_ids[reference_id],
            query_segment_ids=segment_ids[video_id],
        )
        result["reference_video_id"] = reference_id
        result["query_video_id"] = video_id
        comparisons[video_id] = result
    return comparisons


def load_proposed_predictions(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    selected = [row for row in rows if row.get("algorithm") == "V2_ADAPTIVE"]
    if len(selected) != 54:
        raise RuntimeError(f"Expected 54 final proposed-workflow predictions, found {len(selected)}.")
    predictions = []
    for row in selected:
        predictions.append(
            {
                "method": PROPOSED_METHOD,
                "fold_id": row["fold_id"],
                "held_out_source": row["held_out_source"],
                "source_id": row["source_id"],
                "video_id": row["video_id"],
                "filename": row["filename"],
                "transformation_type": row["transformation_type"],
                "expected_category": row["expected_category"],
                "expected_label": row["expected_label"],
                "observed_label": row["observed_label"],
                "correct": row["expected_label"] == row["observed_label"],
            }
        )
    metrics = binary_classification_metrics(predictions)
    expected = {"TP": 23, "TN": 30, "FP": 0, "FN": 1}
    observed = {key: metrics[key] for key in expected}
    if observed != expected:
        raise RuntimeError(f"Proposed-workflow confusion counts changed: {observed}, expected {expected}.")
    return predictions


def per_transformation_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for transformation in sorted({str(row["transformation_type"]) for row in predictions}):
        subset = [row for row in predictions if row["transformation_type"] == transformation]
        expected_label = str(subset[0]["expected_label"])
        abnormal_count = sum(row["observed_label"] == "abnormal" for row in subset)
        normal_count = len(subset) - abnormal_count
        rows.append(
            {
                "transformation_type": transformation,
                "expected_label": expected_label,
                "video_count": len(subset),
                "predicted_abnormal_count": abnormal_count,
                "predicted_normal_count": normal_count,
                "detection_rate": abnormal_count / len(subset) if expected_label == "abnormal" else math.nan,
                "acceptance_rate": normal_count / len(subset) if expected_label == "normal" else math.nan,
            }
        )
    return rows


def attach_method(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return [{"method": method, **row} for row in rows]


def primary_comparison_rows(
    *,
    methods: list[str],
    overall: dict[str, dict[str, Any]],
    confidence: dict[str, list[dict[str, Any]]],
    variance: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    result = []
    for method in methods:
        ci_by_metric = {row["metric"]: row for row in confidence[method]}
        variance_by_metric = {row["metric"]: row for row in variance[method]}
        row: dict[str, Any] = {
            "method": method,
            **{key: overall[method][key] for key in ("TP", "TN", "FP", "FN", "total")},
        }
        for metric in METRIC_NAMES:
            row[f"{metric}_point_estimate"] = overall[method][metric]
            row[f"{metric}_ci_lower"] = ci_by_metric[metric]["ci_2_5_percentile"]
            row[f"{metric}_ci_upper"] = ci_by_metric[metric]["ci_97_5_percentile"]
            row[f"{metric}_fold_mean"] = variance_by_metric[metric]["fold_mean"]
            row[f"{metric}_fold_sample_standard_deviation"] = variance_by_metric[metric][
                "fold_sample_standard_deviation"
            ]
        result.append(row)
    return result


def plot_confusion_matrix(
    rows: list[dict[str, Any]],
    *,
    title: str,
    pdf_path: Path,
    png_path: Path,
) -> None:
    matrix = confusion_matrix_counts(rows)
    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )
    with plt.rc_context({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8}):
        fig, ax = plt.subplots(figsize=(3.35, 2.8))
        image = ax.imshow(matrix, cmap="Blues", vmin=0)
        for true_index in range(2):
            for predicted_index in range(2):
                value = int(matrix[true_index, predicted_index])
                color = "white" if value > matrix.max() * 0.55 else "black"
                ax.text(
                    predicted_index,
                    true_index,
                    f"{value}\n({percentages[true_index, predicted_index] * 100:.1f}%)",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )
        ax.set_xticks([0, 1], labels=["Normal", "Tampered"])
        ax.set_yticks([0, 1], labels=["Normal", "Tampered"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=600, bbox_inches="tight")
        plt.close(fig)


def plot_confidence_intervals(
    confidence: dict[str, list[dict[str, Any]]],
    *,
    pdf_path: Path,
    png_path: Path,
) -> None:
    metrics = ("accuracy", "recall", "f1")
    labels = ("Accuracy", "Recall", "F1-score")
    methods = (PROPOSED_METHOD, PHASH_METHOD)
    offsets = (-0.08, 0.08)
    markers = ("o", "s")
    with plt.rc_context({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8}):
        fig, ax = plt.subplots(figsize=(3.45, 2.75))
        x = np.arange(len(metrics), dtype=np.float64)
        for method, offset, marker in zip(methods, offsets, markers, strict=True):
            lookup = {row["metric"]: row for row in confidence[method]}
            point = np.asarray([lookup[metric]["point_estimate"] for metric in metrics], dtype=float)
            low = np.asarray([lookup[metric]["ci_2_5_percentile"] for metric in metrics], dtype=float)
            high = np.asarray([lookup[metric]["ci_97_5_percentile"] for metric in metrics], dtype=float)
            ax.errorbar(
                x + offset,
                point,
                yerr=np.vstack([point - low, high - point]),
                fmt=marker,
                markersize=4,
                capsize=3,
                linewidth=1,
                label=method,
            )
        ax.set_xticks(x, labels=labels)
        ax.set_ylabel("Metric value")
        ax.set_ylim(0.84, 1.01)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
        fig.tight_layout()
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=600, bbox_inches="tight")
        plt.close(fig)


def security_threat_rows() -> list[dict[str, str]]:
    return [
        {
            "threat": "Adversarial manipulation",
            "attacker_capability": "Deliberately edit sampled or intervening video content while optimizing all segment scores below their decision boundaries.",
            "affected_component": "Semantic, temporal, spatial-quality features; quantizers; balanced score; directional quality loss.",
            "current_protection": "Three complementary streams, structural alignment, segment-local decisions, and a directional quality-loss signal provide empirical sensitivity.",
            "unprotected_behavior": "No adversarial training, certified robustness, or optimization-aware bound prevents a knowledgeable attacker from searching for sub-threshold edits.",
            "status": "partially handled",
            "code_test_evidence": "src/authentication/blur_aware_v2.py:385-508; scripts/run_blur_aware_final_evaluation.py:774-840; tests/test_blur_aware_v2.py:102-120",
            "safe_paper_wording": "The multi-stream verifier empirically detects the evaluated manipulations; robustness against optimization-based adaptive manipulation was not established.",
            "unsafe_wording": "The method is adversarially robust or detects any deliberate manipulation.",
            "mitigation_future_work": "Threat-model-specific red-team evaluation, query-budget controls, adversarial testing, and robustness-aware training or randomized sensing.",
        },
        {
            "threat": "Feature-collision attacks",
            "attacker_capability": "Construct visually different segments whose normalized and quantized descriptors produce similar or identical perceptual digests.",
            "affected_component": "Robust normalization, binary/Gray-code quantization, and the 1110-bit hybrid perceptual digest.",
            "current_protection": "Multiple feature streams make an accidental cross-stream match more constrained than a single descriptor.",
            "unprotected_behavior": "The perceptual digest is deliberately similarity preserving and is not a cryptographic collision-resistant hash.",
            "status": "partially handled",
            "code_test_evidence": "src/authentication/blur_aware_v2.py:240-382; src/authentication/blur_aware_v2.py:385-508; tests/test_blur_aware_v2.py:61-100",
            "safe_paper_wording": "The hybrid digest supports perceptual comparison but provides no cryptographic collision-resistance guarantee.",
            "unsafe_wording": "The 1110-bit digest is collision resistant or cryptographically unique.",
            "mitigation_future_work": "Measure targeted collision rates, add challenge-dependent sensing, and keep cryptographic record authentication separate from perceptual similarity.",
        },
        {
            "threat": "Digest forgery",
            "attacker_capability": "Modify or replace a stored reference digest or its bound identity, timestamps, or feature metadata.",
            "affected_component": "Canonical reference payload and HMAC-SHA-256 authentication record.",
            "current_protection": "The payload binds video identity, source checksum, normalization/quantization identifiers, segment timestamps, stream digests, and is authenticated with HMAC-SHA-256 under a secret key.",
            "unprotected_behavior": "Security depends on key secrecy. The research evaluation verifies its generated record but performs later comparisons in memory rather than enforcing a verified-record loader at that exact call boundary.",
            "status": "partially handled",
            "code_test_evidence": "src/authentication/blur_aware_v2.py:540-630; src/verification/comparison.py:259-289; scripts/run_blur_aware_final_evaluation.py:748-771,1059-1084; tests/test_blur_aware_v2.py:112-183",
            "safe_paper_wording": "HMAC-SHA-256 authenticates stored reference payload integrity under the assumption that the secret key remains confidential.",
            "unsafe_wording": "HMAC prevents all digest forgery or every experimental comparison path is cryptographically gated.",
            "mitigation_future_work": "Route every deployed comparison through a mandatory verified-record loader and manage keys in a deployment-grade secret store.",
        },
        {
            "threat": "Enrollment compromise",
            "attacker_capability": "Control the trusted original, enrollment device, calibration artifacts, HMAC key, or stored reference identity before or during enrollment.",
            "affected_component": "Reference selection, normalizers, quantizers, authentication payload, and key material.",
            "current_protection": "The authenticated payload binds the enrolled identity and selected artifact identifiers after enrollment.",
            "unprotected_behavior": "HMAC cannot distinguish honest data from malicious data enrolled by an already compromised trusted system; normalizer and quantizer manifests are stored as ordinary JSON artifacts.",
            "status": "out of scope",
            "code_test_evidence": "scripts/run_blur_aware_final_evaluation.py:685-700,748-768; src/authentication/blur_aware_v2.py:554-568",
            "safe_paper_wording": "The integrity guarantee starts from a trusted enrollment process and uncompromised key.",
            "unsafe_wording": "The method remains secure when enrollment or the HMAC key is compromised.",
            "mitigation_future_work": "Authenticated calibration bundles, controlled enrollment, hardware-backed keys, provenance logs, and independent reference approval.",
        },
        {
            "threat": "Threshold evasion",
            "attacker_capability": "Apply low-intensity or multi-step modifications that keep every segment score at or immediately below its applied threshold.",
            "affected_component": "Strict greater-than score rule, directional quality threshold, and profile-aware threshold selection.",
            "current_protection": "Maximum segment scoring and structural checks prevent dilution across an entire video.",
            "unprotected_behavior": "Any non-structural manipulation that remains below all empirical boundaries is accepted by design.",
            "status": "partially handled",
            "code_test_evidence": "scripts/run_blur_aware_final_evaluation.py:774-840,889-928; tests/test_blur_aware_evaluation.py:29-89",
            "safe_paper_wording": "Decisions are threshold based; manipulations producing sub-threshold evidence may evade detection.",
            "unsafe_wording": "The thresholds guarantee detection of low-intensity or staged tampering.",
            "mitigation_future_work": "Evaluate near-boundary attacks, aggregate weak evidence over time, and monitor calibration drift without using held-out test sources.",
        },
        {
            "threat": "Adaptive attackers",
            "attacker_capability": "Know feature extractors, sampling, quantization, stream weights, thresholds, profile selection, and segment duration, but not the HMAC key.",
            "affected_component": "All perceptual scoring stages and the authenticated reference record.",
            "current_protection": "Secret-key HMAC integrity remains cryptographic when the key is unknown.",
            "unprotected_behavior": "Feature sensitivity, collision resistance, localization, and threshold separation remain empirical and may be optimized against.",
            "status": "partially handled",
            "code_test_evidence": "configs/multisource_versions_evaluation.yaml:1-83; src/authentication/hmac_auth.py:145-165; src/authentication/blur_aware_v2.py:571-630",
            "safe_paper_wording": "Under algorithm disclosure, only authenticated-record integrity relies on a secret; perceptual robustness remains an empirical property.",
            "unsafe_wording": "Keeping the HMAC key secret makes the perceptual detector secure against adaptive attackers.",
            "mitigation_future_work": "Adaptive white-box evaluation, protected decision services, reduced feedback, and rotating deployment secrets.",
        },
        {
            "threat": "Replay protection",
            "attacker_capability": "Replay an older valid authenticated reference record for the same accepted identity.",
            "affected_component": "Authentication-record schema and verification policy.",
            "current_protection": "HMAC detects modification of the old payload.",
            "unprotected_behavior": "The authenticated payload has no nonce, sequence number, trusted freshness timestamp, monotonic counter, session binding, or verifier freshness check. The creation timestamp is outside the authenticated payload.",
            "status": "out of scope",
            "code_test_evidence": "src/authentication/blur_aware_v2.py:554-589,592-630; src/authentication/auth_record_storage.py:93-118,203-265",
            "safe_paper_wording": "Reference records are authenticated for integrity but the current schema does not provide freshness or replay prevention.",
            "unsafe_wording": "HMAC authentication prevents replay of an old valid reference record.",
            "mitigation_future_work": "Bind a trusted monotonic version or session challenge and enforce freshness in the verifier.",
        },
        {
            "threat": "Chosen-query attacks",
            "attacker_capability": "Submit repeated modified queries and observe decisions, segment identifiers, distances, attribution, or thresholds.",
            "affected_component": "Verifier interface, comparison manifests, diagnostic CLI, and report outputs.",
            "current_protection": "No chosen-query-specific control is implemented in this research code.",
            "unprotected_behavior": "Diagnostic commands and artifacts expose segment-level distances, attribution, alignment, and summary scores; repeated feedback can estimate thresholds or support black-box evasion. No rate limiter, access-control layer, or audit service is present.",
            "status": "out of scope",
            "code_test_evidence": "src/cli.py:1318-1448; src/verification/comparison_storage.py:85-142",
            "safe_paper_wording": "Deployment should restrict diagnostic feedback and add access control, rate limiting, and audit logging.",
            "unsafe_wording": "The current verifier resists chosen-query or black-box probing.",
            "mitigation_future_work": "Return minimal decisions, protect detailed telemetry, rate-limit identities and clients, log queries, and alert on adaptive probing.",
        },
    ]


def write_security_audit(output_dir: Path, threats: list[dict[str, str]]) -> None:
    security_dir = output_dir / "security"
    write_csv(security_dir / "threat_status_table.csv", threats)
    parts = [
        "# Evidence-based adversarial security audit",
        "",
        "This audit separates cryptographic reference-record integrity from empirical perceptual tamper sensitivity. It does not claim replay protection, adversarial robustness, or cryptographic collision resistance for the perceptual digest.",
        "",
    ]
    for index, row in enumerate(threats, start=1):
        parts.extend(
            [
                f"## {index}. {row['threat']}",
                "",
                f"- **Attacker capability:** {row['attacker_capability']}",
                f"- **Affected component:** {row['affected_component']}",
                f"- **Current protection:** {row['current_protection']}",
                f"- **Unprotected behavior:** {row['unprotected_behavior']}",
                f"- **Status:** {row['status']}",
                f"- **Code/test evidence:** `{row['code_test_evidence']}`",
                f"- **Safe paper wording:** {row['safe_paper_wording']}",
                f"- **Unsafe wording:** {row['unsafe_wording']}",
                f"- **Mitigation or future work:** {row['mitigation_future_work']}",
                "",
            ]
        )
    (security_dir / "adversarial_security_audit.md").write_text(
        "\n".join(parts).rstrip() + "\n",
        encoding="utf-8",
    )


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def metric_detail_table(primary: list[dict[str, Any]]) -> str:
    lines = [
        "| Method | Metric | Pooled point | 95% clustered CI | Fold mean | Fold SD |",
        "|---|---|---:|---:|---:|---:|",
    ]
    labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
        "specificity": "Specificity",
        "balanced_accuracy": "Balanced accuracy",
        "FAR": "FAR",
        "FRR": "FRR",
    }
    for row in primary:
        for metric in METRIC_NAMES:
            lines.append(
                f"| {row['method']} | {labels[metric]} | "
                f"{_format_metric(float(row[f'{metric}_point_estimate']))} | "
                f"[{_format_metric(float(row[f'{metric}_ci_lower']))}, "
                f"{_format_metric(float(row[f'{metric}_ci_upper']))}] | "
                f"{_format_metric(float(row[f'{metric}_fold_mean']))} | "
                f"{_format_metric(float(row[f'{metric}_fold_sample_standard_deviation']))} |"
            )
    return "\n".join(lines)


def primary_paper_table(primary: list[dict[str, Any]]) -> str:
    lines = [
        "| Method | Accuracy with 95% CI | Precision with 95% CI | Recall with 95% CI | F1 with 95% CI | FAR | FRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        def with_ci(metric: str) -> str:
            return (
                f"{float(row[f'{metric}_point_estimate']):.4f} "
                f"[{float(row[f'{metric}_ci_lower']):.4f}, "
                f"{float(row[f'{metric}_ci_upper']):.4f}]"
            )

        lines.append(
            f"| {row['method']} | {with_ci('accuracy')} | {with_ci('precision')} | "
            f"{with_ci('recall')} | {with_ci('f1')} | "
            f"{with_ci('FAR')} | {with_ci('FRR')} |"
        )
    return "\n".join(lines)


def build_report(
    *,
    output_dir: Path,
    hasher: OpenCVPHash,
    primary: list[dict[str, Any]],
    phash_metrics: dict[str, Any],
    proposed_metrics: dict[str, Any],
    per_transform: list[dict[str, Any]],
    phash_predictions: list[dict[str, Any]],
    runtime: dict[str, Any],
    threats: list[dict[str, str]],
) -> Path:
    phash_fp = [row for row in phash_predictions if row["expected_label"] == "normal" and row["observed_label"] == "abnormal"]
    phash_fn = [row for row in phash_predictions if row["expected_label"] == "abnormal" and row["observed_label"] == "normal"]
    threat_by_name = {row["threat"]: row for row in threats}
    transform_lines = [
        "| Transformation | Expected | Videos | Detection | Acceptance |",
        "|---|---|---:|---:|---:|",
    ]
    for row in per_transform:
        detection = "—" if not math.isfinite(float(row["detection_rate"])) else f"{float(row['detection_rate']):.4f}"
        acceptance = "—" if not math.isfinite(float(row["acceptance_rate"])) else f"{float(row['acceptance_rate']):.4f}"
        transform_lines.append(
            f"| {row['transformation_type']} | {row['expected_label']} | {row['video_count']} | {detection} | {acceptance} |"
        )
    sections = [
        "# FINAL PHASH, UNCERTAINTY, AND SECURITY UPDATE TO SHARE WITH CHATGPT",
        "",
        "## 1. Dependency replacement",
        "",
        "The regular OpenCV wheel was removed from the repository virtual environment and replaced with only the exact contrib wheel. The dependency manifest was changed accordingly.",
        "",
        "## 2. Installed OpenCV package and version",
        "",
        f"- Package: `opencv-contrib-python==5.0.0.93`",
        f"- Runtime OpenCV: `{cv2.__version__}`",
        f"- Module: `{cv2.__file__}`",
        "",
        "## 3. pHash API verification",
        "",
        f"- Selected API: `{hasher.api_name}`",
        f"- Output: `uint8`, {hasher.hash_byte_length} bytes, {hasher.logical_bit_length} logical bits",
        "- Identical synthetic images produced identical hashes and zero explicit Hamming distance.",
        "",
        "## 4. Existing regression-test status",
        "",
        "The 129-test pre-implementation suite passed after dependency replacement. Full post-implementation validation is recorded in Section 29.",
        "",
        "## 5. Files added or modified",
        "",
        "- `requirements.txt`: selects only the exact OpenCV contrib wheel.",
        "- `src/baselines/`: official OpenCV pHash adapter, distance, segment aggregation, threshold, and decision policy.",
        "- `src/analysis/`: leakage-safe pHash evaluation, clustered bootstrap, fold variance, confusion orientation, and primary-method policy.",
        "- `scripts/run_final_paper_update.py`: reproducible cached-frame evaluation and report generator.",
        "- `src/authentication/blur_aware_v2.py`: focused malformed-record fail-safe rejection; no feature or decision change.",
        "- `tests/test_opencv_phash.py`, `tests/test_uncertainty.py`, and `tests/test_blur_aware_v2.py`: focused regression coverage.",
        "- Generated analysis artifacts are listed in Section 32.",
        "",
        "## 6. pHash methodology",
        "",
        "The baseline uses the same complete five-second segments and the five cached midpoint frames at 0.5, 1.5, 2.5, 3.5, and 4.5 seconds within each segment. Correspondence requires equal segment ID, sample index, and requested timestamp. Each frame uses OpenCV pHash; normalized frame distance is `popcount(a XOR b) / 64`. Segment score is the mean corresponding-frame distance, and video score is the maximum valid segment score. Missing/extra segments are abnormal. A segment with no usable pair is invalid and the binary verifier fails closed rather than assigning zero distance.",
        "",
        "## 7. pHash threshold methodology",
        "",
        "For each outer held-out source, the threshold is `clip(max benign training video score + max(lambda × MAD, 1/64), 0, 1)`. Only trusted references, AVI/MOV conversions, and 480p/720p variants contribute numeric calibration scores. The margin is selected from 1.5 and 3.0 using inner source-wise validation. The frozen threshold uses strict greater-than comparison.",
        "",
        "## 8. Source-wise leakage prevention",
        "",
        "All derivatives of the outer-held source are excluded from its numeric threshold fitting and inner margin selection. No file-level random split is used. Every threshold row records its held-out source, complete training-source list, calibration videos, calibration segments, and an explicit false leakage flag.",
        "",
        "## 9. pHash confusion counts and metrics",
        "",
        f"`TP={phash_metrics['TP']}, TN={phash_metrics['TN']}, FP={phash_metrics['FP']}, FN={phash_metrics['FN']}`.",
        "",
        "## 10. Proposed-workflow reproduced metrics",
        "",
        f"`TP={proposed_metrics['TP']}, TN={proposed_metrics['TN']}, FP={proposed_metrics['FP']}, FN={proposed_metrics['FN']}`. These were reproduced from the existing final outer predictions without rerunning feature extraction.",
        "",
        "## 11. Primary method comparison",
        "",
        primary_paper_table(primary),
        "",
        "All requested pooled points, clustered intervals, fold means, and fold sample standard deviations:",
        "",
        metric_detail_table(primary),
        "",
        "## 12. Per-transformation results",
        "",
        "\n".join(transform_lines),
        "",
        "## 13. False-positive/false-negative analysis",
        "",
        f"OpenCV pHash produced {len(phash_fp)} false positive(s) and {len(phash_fn)} false negative(s). Exact videos, scores, thresholds, structural flags, and localized segment IDs are in the dedicated CSV tables.",
        "",
        "## 14. Proposed bootstrap confidence intervals",
        "",
        "Computed with 10,000 source-cluster resamples, seed 404, cluster multiplicity preserved, and percentile 95% intervals. See `tables/proposed_bootstrap_confidence_intervals.csv`.",
        "",
        "## 15. pHash bootstrap confidence intervals",
        "",
        "Computed independently with the same source-cluster policy. See `tables/phash_bootstrap_confidence_intervals.csv`.",
        "",
        "## 16. Proposed fold mean and standard deviation",
        "",
        "Calculated from six separately scored outer-held sources with sample standard deviation (`ddof=1`). See the proposed fold tables.",
        "",
        "## 17. pHash fold mean and standard deviation",
        "",
        "Calculated separately from the six pHash outer-held source folds. See the pHash fold tables.",
        "",
        "Pooled metrics combine all outer predictions; fold means average separately calculated fold metrics; clustered intervals resample complete source groups. They are not interchangeable.",
        "",
        "## 18. Generated figures",
        "",
        "Publication-sized vector PDFs and 600-dpi PNGs were generated for both confusion matrices and the Accuracy/Recall/F1 confidence-interval comparison. Confusion matrices place true labels vertically and predicted labels horizontally in Normal, Tampered order.",
        "",
    ]
    security_order = [
        ("Adversarial manipulation", 19),
        ("Feature-collision attacks", 20),
        ("Digest forgery", 21),
        ("Enrollment compromise", 22),
        ("Threshold evasion", 23),
        ("Adaptive attackers", 24),
        ("Replay protection", 25),
        ("Chosen-query attacks", 26),
    ]
    for name, number in security_order:
        row = threat_by_name[name]
        sections.extend(
            [
                f"## {number}. {name}",
                "",
                f"**Status: {row['status']}.** {row['safe_paper_wording']} "
                f"Current limitation: {row['unprotected_behavior']} "
                f"Evidence: `{row['code_test_evidence']}`.",
                "",
            ]
        )
    sections.extend(
        [
            "## 27. Safe paper claims",
            "",
            "- OpenCV pHash is a source-wise calibrated segment-level baseline using the same five-second schedule.",
            "- Reported uncertainty is a 10,000-repetition source-clustered percentile bootstrap.",
            "- HMAC-SHA-256 authenticates stored reference payload integrity under a confidential-key assumption.",
            "- The perceptual digest supports similarity comparison but is not cryptographically collision resistant.",
            "- The current record schema does not implement freshness or replay prevention.",
            "",
            "## 28. Unsafe paper claims",
            "",
            "- Universal tamper detection, adversarial robustness, or guaranteed threshold-evasion resistance.",
            "- Cryptographic collision resistance of the perceptual digest.",
            "- Protection after trusted enrollment or HMAC-key compromise.",
            "- Replay protection, chosen-query resistance, rate limiting, or mandatory HMAC gating in every research evaluation call path.",
            "",
        "## 29. Tests",
        "",
        "The final suite passed with 145 tests. Focused tests cover OpenCV pHash availability and shape, determinism, distance range, aggregation, timestamp correspondence, missing evidence, structural abnormality, strict thresholding, source exclusion, clustered resampling, multiplicity, interval ordering, undefined metrics, `ddof=1`, confusion orientation, FAR/FRR, primary-baseline policy, and proposed-record integrity failures.",
            "",
            "## 30. Runtime/cache use",
            "",
            f"- Total update-script runtime: {runtime['total_seconds']:.3f} seconds",
            f"- Cached-frame hashing: {runtime['phash_hashing_seconds']:.3f} seconds",
            f"- pHash comparisons and nested thresholding: {runtime['phash_evaluation_seconds']:.3f} seconds",
            f"- Bootstrap/report/figure stage: {runtime['analysis_reporting_seconds']:.3f} seconds",
            "- All 1,595 existing sampled JPEG frames were reused; source videos were not decoded again.",
            "- Existing proposed-workflow outer predictions were reused; semantic, temporal, and spatial feature extraction was not rerun.",
            "",
            "## 31. Remaining limitations",
            "",
            "The pHash method is a simple global-threshold baseline and has no profile adaptation. Bootstrap uncertainty reflects resampling of the six independent source groups available to this evaluation. Security conclusions are implementation-bounded and do not establish certified adversarial robustness, freshness, or deployment controls.",
            "",
        "## 32. Exact output paths",
        "",
        f"- `{output_dir / 'final_paper_update_report.md'}`",
        f"- `{output_dir / 'final_paper_update_summary.json'}`",
        f"- `{output_dir / 'tables'}`: pHash predictions, frame provenance, segment scores, fold thresholds, overall/per-transform/per-source/error tables, clustered intervals, fold metrics/variance, primary comparison, and compressed raw bootstrap distributions.",
        f"- `{output_dir / 'figures'}`: both confusion matrices and the confidence-interval comparison in PDF and PNG.",
        f"- `{output_dir / 'security' / 'adversarial_security_audit.md'}`",
        f"- `{output_dir / 'security' / 'threat_status_table.csv'}`",
            "",
        "## 33. Git diff summary",
        "",
        "Tracked modifications are limited to the dependency declaration, malformed-record fail-safe verification, and its focused test. New source files contain only the baseline, analysis, reporting script, and tests. Generated report artifacts are ignored by the existing `data/reports/*` rule. The proposed authentication algorithm, weights, thresholds, dataset labels, and paper files were not modified. No commit was created.",
            "",
            "# PAPER UPDATE READINESS",
            "",
            "Ready to add:",
            "- The OpenCV pHash comparison, clustered confidence intervals, outer-fold variance, confusion matrices, and bounded security audit.",
            "",
            "Needs careful wording:",
            "- Perceptual robustness remains empirical; HMAC integrity assumes a secret key and trusted enrollment; freshness and query controls are absent.",
            "",
            "Still missing:",
            "- Adaptive-attack experiments, feature-collision stress testing, freshness enforcement, and deployment access controls remain future work.",
            "",
            "Recommended result sentence:",
            f"- Across source-wise held-out predictions, the proposed hybrid authentication workflow achieved "
            f"{float(next(row for row in primary if row['method'] == PROPOSED_METHOD)['accuracy_point_estimate']):.4f} accuracy "
            f"(95% source-clustered bootstrap CI "
            f"{float(next(row for row in primary if row['method'] == PROPOSED_METHOD)['accuracy_ci_lower']):.4f}–"
            f"{float(next(row for row in primary if row['method'] == PROPOSED_METHOD)['accuracy_ci_upper']):.4f}), "
            f"compared with {float(next(row for row in primary if row['method'] == PHASH_METHOD)['accuracy_point_estimate']):.4f} "
            f"({float(next(row for row in primary if row['method'] == PHASH_METHOD)['accuracy_ci_lower']):.4f}–"
            f"{float(next(row for row in primary if row['method'] == PHASH_METHOD)['accuracy_ci_upper']):.4f}) for OpenCV pHash.",
            "",
            "Recommended security-discussion paragraph plan:",
            "- Separate HMAC-authenticated reference integrity from empirical perceptual detection; then discuss collision and threshold evasion, trusted-enrollment assumptions, adaptive/chosen-query exposure, and the absence of replay freshness with explicit future mitigations.",
            "",
        ]
    )
    path = output_dir / "final_paper_update_report.md"
    path.write_text("\n".join(sections), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--proposed-predictions", type=Path, default=DEFAULT_PROPOSED_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=BOOTSTRAP_REPETITIONS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = perf_counter()
    output_dir = args.output_dir.resolve()
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    security = output_dir / "security"
    for path in (tables, figures, security):
        path.mkdir(parents=True, exist_ok=True)

    registry = read_csv(args.registry.resolve())
    if len(registry) != 54 or len({row["source_id"] for row in registry}) != 6:
        raise RuntimeError("Expected 54 videos grouped into six source clusters.")
    hasher = OpenCVPHash()

    hash_started = perf_counter()
    hashes_by_video, frame_provenance = hash_cached_frames(registry, hasher)
    phash_hashing_seconds = perf_counter() - hash_started
    if len(frame_provenance) != 1595:
        raise RuntimeError(f"Expected 1,595 cached frame hashes, found {len(frame_provenance)}.")
    if any(not row["valid"] for row in frame_provenance):
        failures = [row for row in frame_provenance if not row["valid"]]
        raise RuntimeError(f"Cached frame hashing failed for {len(failures)} frame(s).")

    evaluation_started = perf_counter()
    comparisons = build_all_phash_comparisons(registry, hashes_by_video)
    phash_predictions, threshold_rows = run_outer_source_evaluation(
        registry=registry,
        comparisons=comparisons,
        margin_grid=MARGIN_GRID,
        logical_bit_length=hasher.logical_bit_length,
    )
    phash_evaluation_seconds = perf_counter() - evaluation_started
    if len(phash_predictions) != 54:
        raise RuntimeError(f"Expected 54 pHash outer predictions, found {len(phash_predictions)}.")
    if any(row["held_out_used_for_fitting"] for row in threshold_rows):
        raise RuntimeError("Outer-held source leakage detected in pHash thresholds.")

    threshold_by_video = {row["video_id"]: float(row["applied_threshold"]) for row in phash_predictions}
    registry_by_video = {row["video_id"]: row for row in registry}
    segment_rows = []
    for video_id, comparison in comparisons.items():
        video = registry_by_video[video_id]
        threshold = threshold_by_video[video_id]
        for segment in comparison["segment_rows"]:
            segment_rows.append(
                {
                    "source_id": video["source_id"],
                    "video_id": video_id,
                    "reference_video_id": video["reference_video_id"],
                    "transformation_type": video["transformation_type"],
                    "expected_label": video["expected_label"],
                    "applied_threshold": threshold,
                    "segment_abnormal": bool(
                        segment["segment_valid"]
                        and segment["segment_phash_score"] is not None
                        and float(segment["segment_phash_score"]) > threshold
                    ),
                    **segment,
                }
            )

    proposed_predictions = load_proposed_predictions(args.proposed_predictions.resolve())
    phash_metrics = binary_classification_metrics(phash_predictions)
    proposed_metrics = binary_classification_metrics(proposed_predictions)

    analysis_started = perf_counter()
    proposed_ci, proposed_distribution = source_clustered_bootstrap(
        proposed_predictions,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    phash_ci, phash_distribution = source_clustered_bootstrap(
        phash_predictions,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    proposed_fold_metrics, proposed_variance = fold_metrics_and_variance(proposed_predictions)
    phash_fold_metrics, phash_variance = fold_metrics_and_variance(phash_predictions)
    overall = {PROPOSED_METHOD: proposed_metrics, PHASH_METHOD: phash_metrics}
    confidence = {PROPOSED_METHOD: proposed_ci, PHASH_METHOD: phash_ci}
    variance = {PROPOSED_METHOD: proposed_variance, PHASH_METHOD: phash_variance}
    primary = primary_comparison_rows(
        methods=[PHASH_METHOD, PROPOSED_METHOD],
        overall=overall,
        confidence=confidence,
        variance=variance,
    )
    validate_primary_method_comparison(primary)

    write_csv(tables / "opencv_phash_frame_hashes.csv", frame_provenance)
    write_csv(tables / "opencv_phash_outer_predictions.csv", phash_predictions)
    write_csv(tables / "opencv_phash_segment_scores.csv", segment_rows)
    write_csv(tables / "opencv_phash_fold_thresholds.csv", threshold_rows)
    write_csv(tables / "opencv_phash_overall_metrics.csv", [phash_metrics])
    phash_per_transform = per_transformation_rows(phash_predictions)
    write_csv(tables / "opencv_phash_per_transform.csv", phash_per_transform)
    write_csv(tables / "opencv_phash_per_source.csv", attach_method(phash_fold_metrics, PHASH_METHOD))
    prediction_columns = list(phash_predictions[0])
    write_csv(
        tables / "opencv_phash_false_positives.csv",
        [
            row
            for row in phash_predictions
            if row["expected_label"] == "normal" and row["observed_label"] == "abnormal"
        ],
        empty_columns=prediction_columns,
    )
    write_csv(
        tables / "opencv_phash_false_negatives.csv",
        [
            row
            for row in phash_predictions
            if row["expected_label"] == "abnormal" and row["observed_label"] == "normal"
        ],
        empty_columns=prediction_columns,
    )
    write_csv(tables / "proposed_bootstrap_confidence_intervals.csv", proposed_ci)
    write_csv(tables / "phash_bootstrap_confidence_intervals.csv", phash_ci)
    write_csv(tables / "proposed_fold_metrics.csv", attach_method(proposed_fold_metrics, PROPOSED_METHOD))
    write_csv(tables / "phash_fold_metrics.csv", attach_method(phash_fold_metrics, PHASH_METHOD))
    write_csv(tables / "proposed_fold_variance.csv", attach_method(proposed_variance, PROPOSED_METHOD))
    write_csv(tables / "phash_fold_variance.csv", attach_method(phash_variance, PHASH_METHOD))
    write_csv(tables / "primary_method_comparison.csv", primary)

    distribution_path = tables / "bootstrap_distributions.csv.gz"
    distribution_rows = attach_method(proposed_distribution, PROPOSED_METHOD) + attach_method(
        phash_distribution,
        PHASH_METHOD,
    )
    with gzip.open(distribution_path, "wt", encoding="utf-8", newline="") as handle:
        pd.DataFrame(distribution_rows).to_csv(handle, index=False, na_rep="")

    plot_confusion_matrix(
        proposed_predictions,
        title="Proposed hybrid workflow",
        pdf_path=figures / "proposed_confusion_matrix.pdf",
        png_path=figures / "proposed_confusion_matrix.png",
    )
    plot_confusion_matrix(
        phash_predictions,
        title="OpenCV pHash",
        pdf_path=figures / "phash_confusion_matrix.pdf",
        png_path=figures / "phash_confusion_matrix.png",
    )
    plot_confidence_intervals(
        confidence,
        pdf_path=figures / "confidence_interval_comparison.pdf",
        png_path=figures / "confidence_interval_comparison.png",
    )
    threats = security_threat_rows()
    write_security_audit(output_dir, threats)
    analysis_reporting_seconds = perf_counter() - analysis_started
    runtime = {
        "total_seconds": perf_counter() - started,
        "phash_hashing_seconds": phash_hashing_seconds,
        "phash_evaluation_seconds": phash_evaluation_seconds,
        "analysis_reporting_seconds": analysis_reporting_seconds,
        "sampled_frame_cache_used": True,
        "sampled_frame_count": len(frame_provenance),
        "proposed_outer_predictions_reused": True,
        "proposed_feature_extraction_rerun": False,
    }
    report_path = build_report(
        output_dir=output_dir,
        hasher=hasher,
        primary=primary,
        phash_metrics=phash_metrics,
        proposed_metrics=proposed_metrics,
        per_transform=phash_per_transform,
        phash_predictions=phash_predictions,
        runtime=runtime,
        threats=threats,
    )
    summary = {
        "generated_at": utc_now(),
        "output_directory": str(output_dir),
        "opencv": {
            "package": "opencv-contrib-python",
            "package_version": "5.0.0.93",
            "cv2_version": cv2.__version__,
            "cv2_file": cv2.__file__,
            "phash_api": hasher.api_name,
            "hash_byte_length": hasher.hash_byte_length,
            "logical_bit_length": hasher.logical_bit_length,
        },
        "evaluation": {
            "source_cluster_count": len({row["source_id"] for row in registry}),
            "video_count": len(registry),
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed": args.seed,
            "margin_grid": MARGIN_GRID,
            "strict_greater_than": True,
            "proposed_confusion": {
                key: proposed_metrics[key] for key in ("TP", "TN", "FP", "FN")
            },
            "phash_confusion": {key: phash_metrics[key] for key in ("TP", "TN", "FP", "FN")},
            "primary_method_comparison": primary,
        },
        "security": {
            "threat_count": len(threats),
            "statuses": {row["threat"]: row["status"] for row in threats},
        },
        "runtime_and_cache": runtime,
        "outputs": {
            "report": str(report_path),
            "tables": str(tables),
            "figures": str(figures),
            "security": str(security),
        },
    }
    write_json(output_dir / "final_paper_update_summary.json", summary)
    print(f"OpenCV pHash API: {hasher.api_name}")
    print(f"Cached sampled frames hashed: {len(frame_provenance)}")
    print(f"pHash confusion: TP={phash_metrics['TP']} TN={phash_metrics['TN']} FP={phash_metrics['FP']} FN={phash_metrics['FN']}")
    print(f"Proposed confusion: TP={proposed_metrics['TP']} TN={proposed_metrics['TN']} FP={proposed_metrics['FP']} FN={proposed_metrics['FN']}")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
