"""
evaluation.py
-------------
Run-history persistence for the Neon Nomads pipeline.

This module does NOT recompute any metrics. It only formats and saves
metrics that main.py already produces from the existing pipeline stages:
  - regression metrics (MAE / RMSE / R2) come from model.compare_models(),
    which cross-validates on data/training_data.csv (Valid-only rows).
  - classification metrics (accuracy / precision / recall / F1 / ROC-AUC /
    confusion matrix) come from anomaly_detection.build_anomaly_scores(),
    which cross-validates on data/training_data.csv against the historical
    Validity_Label.

Both are genuine out-of-fold / cross-validated INTERNAL VALIDATION metrics
computed only from the labelled training data. data/test_data.csv has no
ground truth (no Reference_Parameter, no Validity_Label), so it is never
used here to compute an "accuracy" - only to produce the final predictions
written to output/Neon Nomads.csv (and copied into the run folder).

Each call to save_run() creates a new runs/run_XXX/ folder (existing runs
are never overwritten) containing:
  - Neon Nomads.csv   (copy of this run's hackathon prediction output)
  - evaluation.txt    (human-readable internal validation report)
  - summary.json      (machine-readable run configuration + metrics)
"""

from __future__ import annotations
import glob
import json
import os
import re
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd


def next_run_dir(runs_root: str) -> str:
    """Return a fresh runs/run_XXX directory path (creates runs_root if
    needed) without creating the run folder itself. Existing run folders
    are never reused or overwritten - the number always increases."""
    os.makedirs(runs_root, exist_ok=True)
    existing = []
    for path in glob.glob(os.path.join(runs_root, "run_*")):
        m = re.fullmatch(r"run_(\d+)", os.path.basename(path))
        if m:
            existing.append(int(m.group(1)))
    next_n = (max(existing) + 1) if existing else 1
    return os.path.join(runs_root, f"run_{next_n:03d}")


def build_evaluation_text(
    run_id: str,
    generated_at: str,
    regression_metrics: Dict[str, float],
    regression_model_name: str,
    classification_report: Dict[str, Any],
    n_train: int,
    n_test: int,
    cv_folds_regression: int,
    cv_folds_classification: int,
    feature_set: Optional[str] = None,
    holdout_metrics: Optional[Dict[str, Any]] = None,
    robustness_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Formats the plain-text evaluation.txt content. Pure formatting -
    every value here is passed in from metrics the pipeline already
    computed (see module docstring)."""
    cm = classification_report.get("confusion_matrix")
    roc_auc = classification_report.get("roc_auc")

    lines = [
        "NEON NOMADS MODEL EVALUATION",
        "============================",
        "",
        f"Run: {run_id}",
        f"Date/Time: {generated_at}",
        "",
        "REFERENCE PARAMETER",
        "-------------------",
        f"Model: {regression_model_name}",
        f"Feature set used: {feature_set if feature_set else 'all'}",
        f"MAE  (lower is better):  {regression_metrics['MAE']:.4f}",
        f"RMSE (lower is better):  {regression_metrics['RMSE']:.4f}",
        f"R2   (higher is better): {regression_metrics['R2']:.4f}",
        "",
    ]
    if holdout_metrics:
        lines += [
            "UNTOUCHED HOLDOUT EVALUATION (never used for tuning/model/feature selection)",
            "-----------------------------------------------------------------",
            f"Dev rows: {holdout_metrics.get('dev_size')}   Holdout rows: {holdout_metrics.get('holdout_size')}"
            f"  (holdout ratio {holdout_metrics.get('holdout_ratio')})",
            f"Holdout MAE:  {holdout_metrics['MAE']:.4f}",
            f"Holdout RMSE: {holdout_metrics['RMSE']:.4f}",
            f"Holdout R2:   {holdout_metrics['R2']:.4f}",
            "",
        ]
    lines += [
        "VALID / INVALID CLASSIFICATION",
        "------------------------------",
        f"Model: {classification_report.get('selected_classifier', 'n/a')}",
        f"Accuracy:  {classification_report['accuracy']:.4f}",
        f"Precision: {classification_report['precision']:.4f}",
        f"Recall:    {classification_report['recall']:.4f}",
        f"F1 Score:  {classification_report['f1']:.4f}",
        f"ROC-AUC:   {roc_auc:.4f}" if roc_auc is not None else "ROC-AUC:   n/a",
        f"Confusion matrix (rows=true[Valid,Invalid], cols=pred): {cm}",
        "",
        "VALIDATION METHOD",
        "-----------------",
        f"Reference_Parameter: {cv_folds_regression}-fold cross-validation on Valid-only rows of data/training_data.csv",
        f"Valid/Invalid: {cv_folds_classification}-fold genuinely out-of-fold cross-validation on data/training_data.csv "
        "(consistency lines / scaler / Isolation Forest / classifier all refit per fold)",
        "",
        "NUMBER OF RECORDS",
        "------------------",
        f"Training records: {n_train}",
        f"Validation folds (regression): {cv_folds_regression}",
        f"Validation folds (classification): {cv_folds_classification}",
        f"Test records (final predictions, no ground truth): {n_test}",
        "",
    ]
    if robustness_summary:
        lines += ["ROBUSTNESS TESTS (perturbation trials on training data)", "-" * 56]
        for scenario, stats in robustness_summary.items():
            f1v = stats.get('f1_mean')
            f1s = f1v if f1v is not None else float('nan')
            lines.append(f"  {scenario:<20s}: F1={f1s:.4f} (n_trials={stats.get('n_trials')})")
        lines.append("")
    lines += [
        "IMPORTANT:",
        "These are INTERNAL VALIDATION metrics, computed only from",
        "data/training_data.csv (which has ground truth).",
        "data/test_data.csv has no Reference_Parameter / Validity_Label,",
        "so no accuracy is calculated against it - only final predictions",
        "are produced for the hackathon submission.",
        "The official hackathon test score will be determined using the",
        "hidden evaluation data.",
        "",
    ]
    return "\n".join(lines)


def build_summary_dict(
    run_id: str,
    generated_at: str,
    regression_metrics: Dict[str, float],
    regression_model_name: str,
    regression_model_params: Dict[str, Any],
    all_regression_models_compared: Any,
    classification_report: Dict[str, Any],
    classifier_comparison: Any,
    duplicate_check: Dict[str, Any],
    n_train: int,
    n_test: int,
    n_invalid_test: int,
    cv_folds_regression: int,
    cv_folds_classification: int,
    feature_set: Optional[str] = None,
    feature_set_comparison: Any = None,
    holdout_metrics: Optional[Dict[str, Any]] = None,
    robustness_results: Any = None,
) -> Dict[str, Any]:
    """Assembles the summary.json content. Pure aggregation of metrics the
    pipeline already computed - no new metric calculation."""
    return {
        "run_id": run_id,
        "generated_at_utc": generated_at,
        "reference_parameter_model": {
            "name": regression_model_name,
            "parameters": regression_model_params,
            "feature_set": feature_set,
            "feature_set_comparison": feature_set_comparison,
            "cv_folds": cv_folds_regression,
            "MAE": round(float(regression_metrics["MAE"]), 4),
            "RMSE": round(float(regression_metrics["RMSE"]), 4),
            "R2": round(float(regression_metrics["R2"]), 4),
            "all_models_compared": all_regression_models_compared,
            "holdout_evaluation": holdout_metrics,
        },
        "robustness_results": robustness_results,
        "validity_classification_model": {
            "name": classification_report.get("selected_classifier"),
            "cv_folds": cv_folds_classification,
            "accuracy": round(float(classification_report["accuracy"]), 4),
            "precision": round(float(classification_report["precision"]), 4),
            "recall": round(float(classification_report["recall"]), 4),
            "f1": round(float(classification_report["f1"]), 4),
            "roc_auc": (round(float(classification_report["roc_auc"]), 4)
                        if classification_report.get("roc_auc") is not None else None),
            "confusion_matrix": classification_report.get("confusion_matrix"),
            "chosen_threshold": classification_report.get("chosen_threshold"),
            "classifier_comparison": classifier_comparison,
        },
        "anomaly_detection_stats": {
            "override_z_threshold": classification_report.get("override_z_threshold"),
            "isolation_forest_params": classification_report.get("isolation_forest_params"),
            "duplicate_logic_verification_train": duplicate_check,
            "predicted_invalid_in_test": n_invalid_test,
        },
        "records": {
            "training_records": n_train,
            "test_records": n_test,
        },
        "note": (
            "All metrics above are INTERNAL VALIDATION metrics computed via "
            "cross-validation on data/training_data.csv. data/test_data.csv "
            "has no ground truth and is not scored here."
        ),
    }


def save_run(
    runs_root: str,
    pred_csv_path: str,
    regression_metrics: Dict[str, float],
    regression_model_name: str,
    regression_model_params: Dict[str, Any],
    all_regression_models_compared: Any,
    classification_report: Dict[str, Any],
    classifier_comparison: Optional[pd.DataFrame],
    duplicate_check: Dict[str, Any],
    n_train: int,
    n_test: int,
    n_invalid_test: int,
    cv_folds_regression: int = 5,
    cv_folds_classification: int = 5,
    feature_set: Optional[str] = None,
    feature_set_comparison: Any = None,
    holdout_metrics: Optional[Dict[str, Any]] = None,
    robustness_results: Any = None,
    robustness_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Creates a new runs/run_XXX/ folder and writes:
      - Neon Nomads.csv  (copy of the hackathon prediction output)
      - evaluation.txt
      - summary.json
    Never overwrites a previous run. Returns the run directory path."""
    run_dir = next_run_dir(runs_root)
    os.makedirs(run_dir, exist_ok=True)
    run_id = os.path.basename(run_dir)
    generated_at = datetime.now(timezone.utc).isoformat()

    shutil.copy2(pred_csv_path, os.path.join(run_dir, os.path.basename(pred_csv_path)))

    clf_comparison_records = (
        classifier_comparison.to_dict(orient="records")
        if isinstance(classifier_comparison, pd.DataFrame) else classifier_comparison
    )

    eval_text = build_evaluation_text(
        run_id=run_id,
        generated_at=generated_at,
        regression_metrics=regression_metrics,
        regression_model_name=regression_model_name,
        classification_report=classification_report,
        n_train=n_train,
        n_test=n_test,
        cv_folds_regression=cv_folds_regression,
        cv_folds_classification=cv_folds_classification,
        feature_set=feature_set,
        holdout_metrics=holdout_metrics,
        robustness_summary=robustness_summary,
    )
    with open(os.path.join(run_dir, "evaluation.txt"), "w") as f:
        f.write(eval_text)

    summary = build_summary_dict(
        run_id=run_id,
        generated_at=generated_at,
        regression_metrics=regression_metrics,
        regression_model_name=regression_model_name,
        regression_model_params=regression_model_params,
        all_regression_models_compared=all_regression_models_compared,
        classification_report=classification_report,
        classifier_comparison=clf_comparison_records,
        duplicate_check=duplicate_check,
        n_train=n_train,
        n_test=n_test,
        n_invalid_test=n_invalid_test,
        cv_folds_regression=cv_folds_regression,
        cv_folds_classification=cv_folds_classification,
        feature_set=feature_set,
        feature_set_comparison=feature_set_comparison,
        holdout_metrics=holdout_metrics,
        robustness_results=robustness_results,
    )
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return run_dir
