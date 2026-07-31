
from __future__ import annotations

import json
import math
from typing import Any, Iterable, Sequence

import numpy as np


METRIC_NAMES = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "balanced_accuracy",
    "FAR",
    "FRR",
)


def confusion_matrix_counts(rows: Iterable[dict[str, Any]]) -> np.ndarray:
    """Return [[TN, FP], [FN, TP]] for true rows and predicted columns."""

    metrics = binary_classification_metrics(rows)
    return np.asarray(
        [[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]],
        dtype=np.int64,
    )


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan


def binary_classification_metrics(
    rows: Iterable[dict[str, Any]],
    *,
    true_key: str = "expected_label",
    predicted_key: str = "observed_label",
) -> dict[str, float | int]:
    """Compute binary authentication metrics, retaining undefined values as NaN."""

    values = list(rows)
    allowed = {"normal", "abnormal"}
    for row in values:
        if row.get(true_key) not in allowed or row.get(predicted_key) not in allowed:
            raise ValueError(
                f"Labels must be 'normal' or 'abnormal'; got "
                f"{row.get(true_key)!r}/{row.get(predicted_key)!r}."
            )
    tp = sum(row[true_key] == "abnormal" and row[predicted_key] == "abnormal" for row in values)
    tn = sum(row[true_key] == "normal" and row[predicted_key] == "normal" for row in values)
    fp = sum(row[true_key] == "normal" and row[predicted_key] == "abnormal" for row in values)
    fn = sum(row[true_key] == "abnormal" and row[predicted_key] == "normal" for row in values)
    total = tp + tn + fp + fn
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if math.isfinite(precision) and math.isfinite(recall) and precision + recall
        else math.nan
    )
    balanced = (
        (recall + specificity) / 2.0
        if math.isfinite(recall) and math.isfinite(specificity)
        else math.nan
    )
    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "total": int(total),
        "accuracy": _ratio(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": balanced,
        "FAR": _ratio(fn, tp + fn),
        "FRR": _ratio(fp, tn + fp),
    }


def source_clustered_bootstrap(
    rows: Sequence[dict[str, Any]],
    *,
    source_key: str = "source_id",
    repetitions: int = 10_000,
    seed: int = 404,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resample complete source clusters with replacement and summarize metrics."""

    if repetitions <= 0:
        raise ValueError("Bootstrap repetitions must be positive.")
    clusters = sorted({str(row[source_key]) for row in rows})
    if not clusters:
        raise ValueError("Source-clustered bootstrap requires at least one cluster.")
    grouped = {cluster: [row for row in rows if str(row[source_key]) == cluster] for cluster in clusters}
    if any(not grouped[cluster] for cluster in clusters):
        raise ValueError("Every source cluster must contain at least one prediction.")

    rng = np.random.default_rng(seed)
    distributions: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        selected = rng.choice(clusters, size=len(clusters), replace=True).tolist()
        sampled_rows = [row for cluster in selected for row in grouped[str(cluster)]]
        metrics = binary_classification_metrics(sampled_rows)
        counts = {cluster: selected.count(cluster) for cluster in clusters}
        distributions.append(
            {
                "repetition": repetition,
                "sampled_source_multiplicity": json.dumps(counts, sort_keys=True, separators=(",", ":")),
                **metrics,
            }
        )

    point = binary_classification_metrics(rows)
    summaries: list[dict[str, Any]] = []
    for metric in METRIC_NAMES:
        all_values = np.asarray([float(row[metric]) for row in distributions], dtype=np.float64)
        valid = all_values[np.isfinite(all_values)]
        summaries.append(
            {
                "metric": metric,
                "point_estimate": float(point[metric]),
                "bootstrap_mean": float(np.mean(valid)) if valid.size else math.nan,
                "bootstrap_standard_deviation": (
                    float(np.std(valid, ddof=1)) if valid.size > 1 else math.nan
                ),
                "ci_2_5_percentile": float(np.percentile(valid, 2.5)) if valid.size else math.nan,
                "ci_97_5_percentile": float(np.percentile(valid, 97.5)) if valid.size else math.nan,
                "valid_repetition_count": int(valid.size),
                "undefined_repetition_count": int(repetitions - valid.size),
                "repetitions": int(repetitions),
                "seed": int(seed),
                "cluster_count": len(clusters),
            }
        )
    return summaries, distributions


def fold_metrics_and_variance(
    rows: Sequence[dict[str, Any]],
    *,
    fold_key: str = "held_out_source",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Calculate metrics by outer fold and summarize with sample variance."""

    folds = sorted({str(row[fold_key]) for row in rows})
    if not folds:
        raise ValueError("Outer-fold analysis requires at least one fold.")
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        metrics = binary_classification_metrics([row for row in rows if str(row[fold_key]) == fold])
        fold_rows.append({"held_out_source": fold, **metrics})

    variance_rows: list[dict[str, Any]] = []
    for metric in METRIC_NAMES:
        values = np.asarray([float(row[metric]) for row in fold_rows], dtype=np.float64)
        valid = values[np.isfinite(values)]
        variance_rows.append(
            {
                "metric": metric,
                "fold_mean": float(np.mean(valid)) if valid.size else math.nan,
                "fold_sample_standard_deviation": (
                    float(np.std(valid, ddof=1)) if valid.size > 1 else math.nan
                ),
                "fold_minimum": float(np.min(valid)) if valid.size else math.nan,
                "fold_maximum": float(np.max(valid)) if valid.size else math.nan,
                "fold_median": float(np.median(valid)) if valid.size else math.nan,
                "valid_fold_count": int(valid.size),
                "undefined_fold_count": int(len(folds) - valid.size),
            }
        )
    return fold_rows, variance_rows
