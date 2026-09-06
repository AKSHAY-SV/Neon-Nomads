"""
main.py
-------
End-to-end automated pipeline for the CPRI PowerNext-AI screening challenge
(corrected version - see methodology.md for the full list of fixes).

Runs, with NO manual editing of individual records:
  1. Load data (training_data.csv, test_data.csv)
  2. Preprocess: impossible-value handling + KNN imputation FIT ONLY ON
     TRAINING DATA, transformed (not refit) onto test data. Duplicate
     feature-row flag verified programmatically against Validity_Label.
  3. Detect abnormal / invalid test records: genuinely out-of-fold
     cross-validation (consistency lines / scaler / Isolation Forest /
     classifier all refit inside each fold) against historical
     Validity_Label, with a data-driven decision threshold, then a single
     final fit on the full training set applied to test data.
  4. Predict Reference_Parameter for every test record: cross-validated
     comparison of Ridge / RandomForest / ExtraTrees / GradientBoosting /
     HistGradientBoosting (+ XGBoost/LightGBM/CatBoost if installed),
     selected by CV MAE, with an explicit Valid-only vs all-rows training
     comparison to justify the training subset choice.
  5. Write <TeamName>.csv
  6. Write summary.json
  7. Print full evaluation/results to the terminal

Usage:
    python main.py [--team-name TEAM_NAME] [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import load_and_clean, verify_duplicate_logic, FEATURE_COLS
from anomaly_detection import build_anomaly_scores
from model import (compare_models, compare_training_subsets, select_and_fit_best,
                    feature_importance, predict, TARGET_COL, _candidate_models)

DEFAULT_TEAM_NAME = "CPRI_PowerNext_Team"


def parse_args():
    p = argparse.ArgumentParser(description="CPRI PowerNext-AI screening pipeline")
    p.add_argument("--team-name", default=DEFAULT_TEAM_NAME,
                    help="Team name used for the prediction CSV filename")
    p.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"),
                    help="Directory containing training_data.csv and test_data.csv")
    p.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "..", "output"),
                    help="Directory to write prediction CSV and summary.json into")
    return p.parse_args()


def safe_sanitize_team_name(name: str) -> str:
    keep = [c if (c.isalnum() or c in "_-") else "_" for c in name]
    out = "".join(keep).strip("_")
    return out or DEFAULT_TEAM_NAME


def main():
    t_start = time.time()
    args = parse_args()
    team_name = safe_sanitize_team_name(args.team_name)
    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "training_data.csv")
    test_path = os.path.join(data_dir, "test_data.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"ERROR: expected data files not found.\n"
              f"  training: {train_path} (exists={os.path.exists(train_path)})\n"
              f"  test:     {test_path} (exists={os.path.exists(test_path)})")
        sys.exit(1)

    print("=" * 72)
    print("CPRI PowerNext-AI Screening Round Pipeline (corrected)")
    print("=" * 72)

    # ---------------------------------------------------------------
    # Step 1-2: Load + preprocess (imputer fit on train, transform-only test)
    # ---------------------------------------------------------------
    print("\n[1/7] Loading and preprocessing data...")
    try:
        train, test = load_and_clean(train_path, test_path)
    except Exception:
        print("ERROR during data loading/preprocessing:")
        traceback.print_exc()
        sys.exit(1)

    n_train, n_test = len(train), len(test)
    print(f"  Training records: {n_train}   Test records: {n_test}")
    print(f"  Missing values remaining -> train: {train[FEATURE_COLS].isna().sum().sum()}, "
          f"test: {test[FEATURE_COLS].isna().sum().sum()}")
    print("  (Imputer fit ONCE on training data; test data transformed with the same fitted imputer.)")

    dup_check = verify_duplicate_logic(train)
    print(f"\n  Duplicate-logic verification (train): {dup_check}")
    print(f"  Exact duplicate feature-rows -> train: {int(train['Is_Duplicate_Feature_Row'].sum())}, "
          f"test: {int(test['Is_Duplicate_Feature_Row'].sum())}")

    # ---------------------------------------------------------------
    # Step 3: Anomaly / validity detection (genuine out-of-fold CV)
    # ---------------------------------------------------------------
    print("\n[2/7] Detecting abnormal / invalid records (nested, leakage-free CV)...")
    try:
        train_scored, test_scored, cv_report = build_anomaly_scores(train, test)
    except Exception:
        print("ERROR during anomaly detection:")
        traceback.print_exc()
        sys.exit(1)

    print("  Genuinely out-of-fold classifier performance vs historical Validity_Label")
    print("  (consistency lines / scaler / Isolation Forest / classifier all refit per fold):")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"    {k:>10s}: {cv_report[k]:.4f}")
    print(f"    confusion_matrix (rows=true[Valid,Invalid], cols=pred): "
          f"{cv_report['confusion_matrix']}")
    print(f"    decision threshold chosen on OOF predictions "
          f"(maximising F1): {cv_report['chosen_threshold']}")
    print(f"    hard-override residual-z threshold (data-driven): "
          f"{cv_report['override_z_threshold']:.2f}")

    n_invalid_test = int(test_scored["Predicted_Invalid"].sum())
    print(f"  Predicted Invalid in TEST set: {n_invalid_test} / {n_test} "
          f"({100*n_invalid_test/n_test:.1f}%)")

    # ---------------------------------------------------------------
    # Step 4: Reference_Parameter prediction
    # ---------------------------------------------------------------
    print("\n[3/7] Comparing regression models (5-fold CV, Valid-only training rows)...")
    train_valid = train_scored[train_scored["Validity_Label"] == "Valid"]
    try:
        results, fitted_models = compare_models(train_valid, n_splits=5)
    except Exception:
        print("ERROR during model comparison:")
        traceback.print_exc()
        sys.exit(1)

    print(results.to_string(index=False))

    best_name, best_model = select_and_fit_best(train_valid, results, fitted_models)
    print(f"\n  Selected model: {best_name} "
          f"(CV MAE={results.iloc[0]['MAE']:.4f}, RMSE={results.iloc[0]['RMSE']:.4f}, "
          f"R2={results.iloc[0]['R2']:.4f})")

    imp = feature_importance(best_name, best_model)
    if imp is not None:
        print("\n  Feature importance:")
        for feat, val in imp.items():
            print(f"    {feat:>25s}: {val:.4f}")

    print("\n[4/7] Verifying Valid-only vs all-rows training subset choice...")
    try:
        model_builder = lambda: _candidate_models()[best_name]
        subset_comparison = compare_training_subsets(train_scored, best_name, model_builder, n_splits=5)
        print("  Same model architecture, cross-validated on each subset "
              "(no test data involved):")
        for label, stats in subset_comparison.items():
            print(f"    {label:>10s}: n={stats['n_rows']:4d}  MAE={stats['MAE']:.4f}  "
                  f"RMSE={stats['RMSE']:.4f}  R2={stats['R2']:.4f}")
        better_subset = min(subset_comparison, key=lambda k: subset_comparison[k]["MAE"])
        print(f"  -> Lower CV MAE achieved by training on: '{better_subset}'")
    except Exception:
        print("WARNING: training-subset comparison failed (non-fatal):")
        traceback.print_exc()
        subset_comparison = None
        better_subset = "valid_only"

    print("\n[5/7] Predicting Reference_Parameter for all test records...")
    test_scored["Predicted_Reference_Parameter"] = predict(best_model, test_scored)

    n_nan_pred = int(pd.isna(test_scored["Predicted_Reference_Parameter"]).sum())
    if n_nan_pred > 0:
        print(f"  WARNING: {n_nan_pred} NaN predictions found - filling with training mean as a safeguard.")
        fallback = train_valid[TARGET_COL].mean()
        test_scored["Predicted_Reference_Parameter"] = test_scored["Predicted_Reference_Parameter"].fillna(fallback)

    # ---------------------------------------------------------------
    # Step 5: Write prediction CSV
    # ---------------------------------------------------------------
    print("\n[6/7] Writing deliverable files...")
    test_scored["Valid / Invalid"] = np.where(test_scored["Predicted_Invalid"], "Invalid", "Valid")

    pred_df = test_scored[["Test_ID", "Predicted_Reference_Parameter", "Valid / Invalid"]].copy()
    pred_df["Predicted_Reference_Parameter"] = pred_df["Predicted_Reference_Parameter"].round(4)

    # sanity checks before writing
    assert len(pred_df) == n_test, "Row count mismatch between predictions and test data!"
    assert pred_df["Test_ID"].is_unique, "Duplicate Test_IDs in prediction output!"
    assert pred_df["Predicted_Reference_Parameter"].notna().all(), "NaN predictions present!"
    assert pd.api.types.is_numeric_dtype(pred_df["Predicted_Reference_Parameter"]), \
        "Predicted_Reference_Parameter is not numeric!"
    assert set(pred_df["Valid / Invalid"].unique()) <= {"Valid", "Invalid"}, "Unexpected validity labels!"
    assert pred_df["Valid / Invalid"].notna().all(), "Missing Valid/Invalid label present!"

    csv_filename = f"{team_name}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    pred_df.to_csv(csv_path, index=False)
    print(f"  Wrote {csv_path}  ({len(pred_df)} rows)")

    # ---------------------------------------------------------------
    # Step 6: Automated summary
    # ---------------------------------------------------------------
    preds = pred_df["Predicted_Reference_Parameter"]

    # "3 Test IDs requiring highest attention": rank by the same evidence
    # the final Invalid decision is built from - classifier probability,
    # physical-inconsistency severity (max |resid z|), Isolation Forest
    # anomaly score, and duplicate-row status - combined as a single
    # z-scored composite so no single signal's scale dominates arbitrarily.
    resid_cols = [c for c in test_scored.columns if c.startswith("resid_z_")]

    def _z(s):
        s = s.astype(float)
        std = s.std()
        return (s - s.mean()) / std if std > 0 else s * 0.0

    attention_components = pd.DataFrame({
        "clf_proba_invalid_z": _z(test_scored["clf_proba_invalid"]),
        "max_resid_z_z": _z(test_scored[resid_cols].abs().max(axis=1)),
        "iso_forest_score_z": _z(test_scored["iso_forest_score"]),
    })
    test_scored["_attention_score"] = (
        attention_components.sum(axis=1)
        + test_scored["Is_Duplicate_Feature_Row"].astype(float) * 3.0  # duplicate = near-certain issue
    )
    top_attention = (
        test_scored.sort_values("_attention_score", ascending=False)
        .head(3)["Test_ID"].tolist()
    )

    methodology_summary = (
        "Imputer/consistency-lines/scaler/Isolation-Forest/classifier fit only "
        "on training folds (no leakage); test data transformed only. Invalid "
        "records flagged via physical-consistency checks, duplicate-row "
        "detection, Isolation Forest and a classifier, threshold chosen by "
        "out-of-fold F1. Reference_Parameter predicted with the best "
        f"cross-validated regressor ({best_name}) trained on verified rows."
    )
    n_words = len(methodology_summary.split())
    if n_words > 100:
        methodology_summary = " ".join(methodology_summary.split()[:100])

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "team_name": team_name,
        "records_analysed": int(n_test),
        "abnormal_invalid_count": n_invalid_test,
        "abnormal_invalid_percentage": round(100 * n_invalid_test / n_test, 2),
        "predicted_reference_parameter": {
            "minimum": round(float(preds.min()), 4),
            "maximum": round(float(preds.max()), 4),
            "average": round(float(preds.mean()), 4),
        },
        "test_ids_requiring_highest_attention": top_attention,
        "model_selection": {
            "selected_model": best_name,
            "cv_mae": round(float(results.iloc[0]["MAE"]), 4),
            "cv_rmse": round(float(results.iloc[0]["RMSE"]), 4),
            "cv_r2": round(float(results.iloc[0]["R2"]), 4),
            "all_models_compared": results.to_dict(orient="records"),
            "training_subset_used": "valid_only",
            "training_subset_comparison": subset_comparison,
        },
        "anomaly_detection_out_of_fold_performance_vs_historical_labels": {
            k: v for k, v in cv_report.items()
        },
        "duplicate_logic_verification_train": dup_check,
        "methodology_explanation": methodology_summary,
        "methodology_explanation_word_count": len(methodology_summary.split()),
        "pipeline_runtime_seconds": None,  # filled in just before writing, see below
    }

    summary["pipeline_runtime_seconds"] = round(time.time() - t_start, 2)

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote {summary_path}")

    # ---------------------------------------------------------------
    # Step 7: Print final terminal report
    # ---------------------------------------------------------------
    print("\n[7/7] FINAL SUMMARY")
    print("-" * 72)
    print(f"  Records analysed:            {summary['records_analysed']}")
    print(f"  Abnormal / Invalid detected: {summary['abnormal_invalid_count']} "
          f"({summary['abnormal_invalid_percentage']}%)")
    print(f"  Predicted Reference_Parameter -> "
          f"min={summary['predicted_reference_parameter']['minimum']}, "
          f"max={summary['predicted_reference_parameter']['maximum']}, "
          f"avg={summary['predicted_reference_parameter']['average']}")
    print(f"  Top-attention Test IDs:      {summary['test_ids_requiring_highest_attention']}")
    print(f"  Selected regression model:   {best_name}")
    print(f"  Methodology explanation word count: {summary['methodology_explanation_word_count']} (<=100 required)")
    print(f"  Total pipeline runtime: {summary['pipeline_runtime_seconds']}s")
    print("-" * 72)
    print("Pipeline completed successfully.")
    return summary, pred_df


if __name__ == "__main__":
    main()
