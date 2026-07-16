from __future__ import annotations

from typing import Any, Literal
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


ZeroDivision = Literal["warn", 0, 1, np.nan]


@dataclass(frozen=True)
class ClassificationReportTables:
    """Container for all generated report tables."""

    seed_level_results: pd.DataFrame
    per_seed_report: pd.DataFrame
    numeric_summary: pd.DataFrame
    formatted_summary: pd.DataFrame


def _as_1d_array(values: Any, *, name: str) -> np.ndarray:
    """Convert labels or predictions to a one-dimensional NumPy array."""
    array = np.asarray(values).ravel()

    if array.size == 0:
        raise ValueError(f"`{name}` must not be empty.")

    return array


def _is_scalar_like(value: Any) -> bool:
    """Return whether a value is suitable as metadata in a summary dataframe."""
    return value is None or isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
            np.integer,
            np.floating,
            np.bool_,
        ),
    )


def format_mean_std(mean: float, std: float, *, decimals: int = 3) -> str:
    """
    Format a mean/std pair as a compact string.

    Examples
    --------
    >>> format_mean_std(0.81234, 0.03456)
    '0.812 ± 0.035'

    >>> format_mean_std(0.81234, np.nan)
    '0.812'
    """
    if pd.isna(mean):
        return ""

    if pd.isna(std):
        return f"{mean:.{decimals}f}"

    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def make_seed_level_results_df(
    results: Sequence[Mapping[str, Any]],
    *,
    labels_key: str = "labels",
    predictions_key: str = "predictions",
) -> pd.DataFrame:
    """
    Create a seed-level result table excluding raw labels and predictions.

    Parameters
    ----------
    results:
        Sequence of dictionaries. Each dictionary represents one evaluated seed/model.
    labels_key:
        Key under which ground-truth labels are stored.
    predictions_key:
        Key under which predicted labels are stored.

    Returns
    -------
    pd.DataFrame
        One row per seed/model, excluding the raw label and prediction arrays.
    """
    rows = []

    for result in results:
        row = {
            key: value
            for key, value in result.items()
            if key not in {labels_key, predictions_key}
        }
        rows.append(row)

    return pd.DataFrame(rows)


def report_rows_from_result(
    result: Mapping[str, Any],
    *,
    class_order: Sequence[Any],
    class_names: Sequence[str],
    labels_key: str = "labels",
    predictions_key: str = "predictions",
    metadata_keys: Sequence[str] | None = None,
    zero_division: ZeroDivision = 0,
) -> list[dict[str, Any]]:
    """
    Create classification-report-style rows for one seed/model.

    Parameters
    ----------
    result:
        Dictionary containing at least ground-truth labels and predictions.
    class_order:
        Label values in the desired order, for example ``[0, 1]``.
    class_names:
        Human-readable class names in the same order as ``class_order``,
        for example ``["hc", "patho"]``.
    labels_key:
        Key in ``result`` containing the ground-truth labels.
    predictions_key:
        Key in ``result`` containing the predicted labels.
    metadata_keys:
        Optional metadata keys to carry into every row, for example
        ``["seed", "selected_fold", "cv_best_val_f1"]``.

        If ``None``, scalar-like metadata fields are inferred automatically.
    zero_division:
        Passed to ``sklearn.metrics.classification_report``.

    Returns
    -------
    list[dict[str, Any]]
        Rows containing precision, recall, f1-score, and support for each class,
        plus accuracy, macro average, and weighted average.
    """
    if len(class_order) != len(class_names):
        raise ValueError(
            "`class_order` and `class_names` must have the same length. "
            f"Got {len(class_order)} and {len(class_names)}."
        )

    labels = _as_1d_array(result[labels_key], name=labels_key)
    predictions = _as_1d_array(result[predictions_key], name=predictions_key)

    if labels.shape[0] != predictions.shape[0]:
        raise ValueError(
            "`labels` and `predictions` must have the same length. "
            f"Got {labels.shape[0]} and {predictions.shape[0]}."
        )

    if metadata_keys is None:
        metadata = {
            key: value
            for key, value in result.items()
            if key not in {labels_key, predictions_key} and _is_scalar_like(value)
        }
    else:
        metadata = {key: result.get(key) for key in metadata_keys}

    report = classification_report(
        labels,
        predictions,
        labels=list(class_order),
        target_names=list(class_names),
        output_dict=True,
        zero_division=zero_division,
    )

    rows: list[dict[str, Any]] = []

    for class_name in class_names:
        rows.append(
            {
                **metadata,
                "metric": class_name,
                "precision": report[class_name]["precision"],
                "recall": report[class_name]["recall"],
                "f1-score": report[class_name]["f1-score"],
                "support": report[class_name]["support"],
            }
        )

    rows.append(
        {
            **metadata,
            "metric": "accuracy",
            "precision": np.nan,
            "recall": np.nan,
            "f1-score": report["accuracy"],
            "support": len(labels),
        }
    )

    for average_name in ["macro avg", "weighted avg"]:
        rows.append(
            {
                **metadata,
                "metric": average_name,
                "precision": report[average_name]["precision"],
                "recall": report[average_name]["recall"],
                "f1-score": report[average_name]["f1-score"],
                "support": report[average_name]["support"],
            }
        )

    return rows


def make_per_seed_report_df(
    results: Sequence[Mapping[str, Any]],
    *,
    class_order: Sequence[Any],
    class_names: Sequence[str],
    labels_key: str = "labels",
    predictions_key: str = "predictions",
    metadata_keys: Sequence[str] | None = None,
    zero_division: ZeroDivision = 0,
) -> pd.DataFrame:
    """
    Create the long-form per-seed classification report table.

    Parameters
    ----------
    results:
        Sequence of dictionaries, one per seed/model.
    class_order:
        Label values in the desired report order.
    class_names:
        Human-readable class names in the same order as ``class_order``.
    labels_key:
        Key containing ground-truth labels.
    predictions_key:
        Key containing predicted labels.
    metadata_keys:
        Metadata columns to carry into the report table.
    zero_division:
        Passed to ``sklearn.metrics.classification_report``.

    Returns
    -------
    pd.DataFrame
        Long-form table with one row per seed and report metric.
    """
    rows: list[dict[str, Any]] = []

    for result in results:
        rows.extend(
            report_rows_from_result(
                result,
                class_order=class_order,
                class_names=class_names,
                labels_key=labels_key,
                predictions_key=predictions_key,
                metadata_keys=metadata_keys,
                zero_division=zero_division,
            )
        )

    return pd.DataFrame(rows)


def summarize_report_df(
    report_df: pd.DataFrame,
    *,
    metric_order: Sequence[str],
) -> pd.DataFrame:
    """
    Aggregate a long-form per-seed report into mean/std summary statistics.

    Parameters
    ----------
    report_df:
        Output of ``make_per_seed_report_df``.
    metric_order:
        Desired row order, usually class names followed by
        ``["accuracy", "macro avg", "weighted avg"]``.

    Returns
    -------
    pd.DataFrame
        Numeric summary with mean and std columns for precision, recall,
        f1-score, and support.
    """
    required_columns = {"metric", "precision", "recall", "f1-score", "support"}
    missing = required_columns.difference(report_df.columns)

    if missing:
        raise ValueError(f"`report_df` is missing required columns: {sorted(missing)}")

    summary = (
        report_df.groupby("metric", observed=False)[
            ["precision", "recall", "f1-score", "support"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        column[0]
        if isinstance(column, tuple) and column[1] == ""
        else f"{column[0]}_{column[1]}"
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]

    summary["metric"] = pd.Categorical(
        summary["metric"],
        categories=list(metric_order),
        ordered=True,
    )
    summary = summary.sort_values("metric").reset_index(drop=True)
    summary["metric"] = summary["metric"].astype(str)

    return summary


def format_report_summary_df(
    numeric_summary: pd.DataFrame,
    *,
    decimals: int = 3,
) -> pd.DataFrame:
    """
    Convert a numeric report summary into a sklearn-like formatted table.

    Parameters
    ----------
    numeric_summary:
        Output of ``summarize_report_df``.
    decimals:
        Number of decimals used for mean ± std formatting.

    Returns
    -------
    pd.DataFrame
        Formatted classification-report-style summary table.
    """
    formatted = pd.DataFrame(
        {
            "metric": numeric_summary["metric"],
            "precision": [
                format_mean_std(mean, std, decimals=decimals)
                for mean, std in zip(
                    numeric_summary["precision_mean"],
                    numeric_summary["precision_std"],
                )
            ],
            "recall": [
                format_mean_std(mean, std, decimals=decimals)
                for mean, std in zip(
                    numeric_summary["recall_mean"],
                    numeric_summary["recall_std"],
                )
            ],
            "f1-score": [
                format_mean_std(mean, std, decimals=decimals)
                for mean, std in zip(
                    numeric_summary["f1-score_mean"],
                    numeric_summary["f1-score_std"],
                )
            ],
            "support": numeric_summary["support_mean"].round().astype(int),
        }
    )

    return formatted


def make_classification_report_tables(
    results: Sequence[Mapping[str, Any]],
    *,
    class_order: Sequence[Any],
    class_names: Sequence[str],
    labels_key: str = "labels",
    predictions_key: str = "predictions",
    metadata_keys: Sequence[str] | None = None,
    decimals: int = 3,
    zero_division: ZeroDivision = 0,
) -> ClassificationReportTables:
    """
    Build seed-level, per-seed, numeric-summary, and formatted-summary tables.

    Parameters
    ----------
    results:
        Sequence of dictionaries, one dictionary per evaluated seed/model.
        Each dictionary must contain ground-truth labels and predictions.
    class_order:
        Label values in the desired report order.
    class_names:
        Human-readable class names in the same order as ``class_order``.
    labels_key:
        Key containing ground-truth labels.
    predictions_key:
        Key containing predicted labels.
    metadata_keys:
        Optional metadata fields to carry into the per-seed report table.
        Examples: ``["seed", "selected_fold", "cv_best_val_f1"]``.
    decimals:
        Number of decimals used for formatted mean ± std strings.
    zero_division:
        Passed to ``sklearn.metrics.classification_report``.

    Returns
    -------
    ClassificationReportTables
        Dataclass containing:
        - ``seed_level_results``
        - ``per_seed_report``
        - ``numeric_summary``
        - ``formatted_summary``
    """
    if len(results) == 0:
        raise ValueError("`results` must contain at least one result dictionary.")

    metric_order = list(class_names) + ["accuracy", "macro avg", "weighted avg"]

    seed_level_results = make_seed_level_results_df(
        results,
        labels_key=labels_key,
        predictions_key=predictions_key,
    )

    per_seed_report = make_per_seed_report_df(
        results,
        class_order=class_order,
        class_names=class_names,
        labels_key=labels_key,
        predictions_key=predictions_key,
        metadata_keys=metadata_keys,
        zero_division=zero_division,
    )

    numeric_summary = summarize_report_df(
        per_seed_report,
        metric_order=metric_order,
    )

    formatted_summary = format_report_summary_df(
        numeric_summary,
        decimals=decimals,
    )

    return ClassificationReportTables(
        seed_level_results=seed_level_results,
        per_seed_report=per_seed_report,
        numeric_summary=numeric_summary,
        formatted_summary=formatted_summary,
    )
