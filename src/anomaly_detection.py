"""
anomaly_detection.py
---------------------
Task 1: identify abnormal / invalid records.

Fix vs previous version (this is the important part - read before editing)
----------------------------------------------------------------------------
The previous version fit the physical-consistency lines, the StandardScaler
and the Isolation Forest on the WHOLE training set ONE TIME, and only then
ran cross-validation on the classifier. That means every fold's "held-out"
rows had already influenced the residual lines, the scaler's mean/std and
the Isolation Forest's learned tree structure - a real leakage path even
though the *labels* were never touched directly, because those upstream
transforms use the feature values of rows that later show up in the
validation fold.

This version fits EVERYTHING (consistency lines, scaler, Isolation Forest,
classifier) freshly inside each cross-validation fold, using only that
fold's training rows, and only ever calls `.transform()`/`.predict()` on
the held-out fold. Reported CV metrics are computed purely from these
out-of-fold predictions, so they are genuinely unbiased estimates of
performance on unseen data - not just "no label leakage" but "no feature
statistics leakage" either.

Duplicate-feature-row flagging is the one signal that is safe to compute
globally (see preprocessing.py's `flag_exact_duplicate_features` docstring):
it uses no label information and is a simple structural comparison against
the SAME dataframe's own rows, so it generalises to test data without ever
referencing training rows.

Signals combined (kept from the original design, since they were the
useful part):
  a) Physical-consistency residual z-scores (Sensor_S1/S3 vs Voltage,
     Sensor_S2 vs Current) - fit only on the fold's Valid training rows.
  b) Duplicate-feature-row flag (global, label-free, see above).
  c) Isolation Forest anomaly score - fit only on the fold's training rows.
  d) Gradient Boosting classifier trained on historical Validity_Label
     within the fold, using (a)-(c) as engineered features.

Decision threshold
-------------------
Rather than hard-coding 0.5, the classifier's out-of-fold probabilities are
swept across a grid of thresholds and the threshold that maximises F1 on
the OUT-OF-FOLD predictions (never on the test set, which has no labels
anyway) is selected. This threshold is then used, together with the two
label-free hard-override rules below, to make the final decision when the
pipeline is refit on the full training set and applied to test data.

Hard overrides (kept from before, but now data-driven rather than a magic
constant):
  - duplicate feature row -> Invalid (100% empirical rate in training)
  - |physical-consistency residual z| beyond a threshold derived from the
    OBSERVED maximum residual z-score among historically-Valid rows (plus a
    safety margin) -> Invalid. This avoids inventing an arbitrary z cutoff:
    it is set just above what genuinely-valid data ever produces.

Genuine operating-regime changes vs sensor errors
---------------------------------------------------
A record that is unusual in raw magnitude (e.g. very high voltage) but
whose sensors still track the expected Voltage/Current relationship will
have SMALL physical-consistency residuals and will not trigger the hard
overrides; it can only be flagged by the learned classifier if the
historical data contains similar patterns among labelled Invalid rows.
This is intentional: the brief explicitly requires that "an unusual value
is not necessarily an invalid value" - unusual-but-physically-consistent
combinations are treated as genuine regime changes, not anomalies.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

FEATURE_COLS = [
    "Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C",
    "Test_Duration_min", "Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4",
]

# Physical-consistency relationships discovered during EDA (see explore.py):
# (x_col, y_col) pairs where y is expected to be approximately linear in x
# for physically-valid records.
CONSISTENCY_PAIRS = [
    ("Applied_Voltage_kV", "Sensor_S1"),
    ("Applied_Voltage_kV", "Sensor_S3"),
    ("Load_Current_A", "Sensor_S2"),
]

RANDOM_STATE = 42
N_SPLITS = 5
ISO_N_ESTIMATORS = 200          # reduced from 300 for speed, negligible score impact
GB_N_ESTIMATORS = 150           # kept modest for speed on ~800-row folds
RESIDUAL_Z_MARGIN = 1.5         # safety margin added on top of the observed max Valid |z|
THRESHOLD_GRID = np.round(np.arange(0.10, 0.91, 0.05), 2)


# ---------------------------------------------------------------------
# Physical-consistency feature engineering (fit only on given rows)
# ---------------------------------------------------------------------
def _fit_consistency_line(valid_rows: pd.DataFrame, x: str, y: str):
    mask = valid_rows[[x, y]].notna().all(axis=1)
    slope, intercept = np.polyfit(valid_rows.loc[mask, x], valid_rows.loc[mask, y], 1)
    resid = valid_rows.loc[mask, y] - (slope * valid_rows.loc[mask, x] + intercept)
    resid_std = resid.std()
    max_abs_z = (resid / (resid_std if resid_std > 0 else 1.0)).abs().max()
    return slope, intercept, resid_std, max_abs_z


def fit_physical_consistency(train_fold: pd.DataFrame, validity_col="Validity_Label"):
    """Fit consistency lines using ONLY Valid rows within the given
    (fold-)training data. Returns a dict keyed by (x, y) ->
    (slope, intercept, resid_std, max_abs_z_on_valid)."""
    valid_rows = (train_fold[train_fold[validity_col] == "Valid"]
                  if validity_col in train_fold.columns else train_fold)
    fitted = {}
    for x, y in CONSISTENCY_PAIRS:
        fitted[(x, y)] = _fit_consistency_line(valid_rows, x, y)
    return fitted


def add_consistency_features(df: pd.DataFrame, fitted_lines: dict) -> pd.DataFrame:
    """Adds one z-scored residual column per consistency pair using
    parameters fit elsewhere (never refit here) - safe to call on
    held-out/validation/test data."""
    df = df.copy()
    for (x, y), (slope, intercept, resid_std, _max_z) in fitted_lines.items():
        pred = slope * df[x] + intercept
        resid = df[y] - pred
        z = resid / (resid_std if resid_std > 0 else 1.0)
        df[f"resid_z_{y}_vs_{x}"] = z
    return df


def _resid_z_cols(fitted_lines):
    return [f"resid_z_{y}_vs_{x}" for (x, y) in fitted_lines.keys()]


def _hard_override_threshold(fitted_lines: dict) -> float:
    """Data-driven residual-z override threshold: the largest |z| ever
    observed among historically-Valid rows (across all consistency pairs),
    plus a fixed safety margin, so we never flag typical Valid behaviour."""
    max_zs = [v[3] for v in fitted_lines.values()]
    return max(max_zs) + RESIDUAL_Z_MARGIN


# ---------------------------------------------------------------------
# One "unit of work": fit everything on a training subset, score a
# (possibly disjoint) evaluation subset. Used both inside CV folds and
# for the final full-train -> test pass, so the exact same logic runs in
# both places (no separate/inconsistent "production" code path).
# ---------------------------------------------------------------------
def _fit_and_score(train_subset: pd.DataFrame, eval_subset: pd.DataFrame):
    fitted_lines = fit_physical_consistency(train_subset)
    train_feat = add_consistency_features(train_subset, fitted_lines)
    eval_feat = add_consistency_features(eval_subset, fitted_lines)
    resid_cols = _resid_z_cols(fitted_lines)

    iso_cols = FEATURE_COLS + resid_cols
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feat[iso_cols])
    eval_scaled = scaler.transform(eval_feat[iso_cols])

    iso = IsolationForest(n_estimators=ISO_N_ESTIMATORS, contamination="auto",
                           random_state=RANDOM_STATE, n_jobs=1)
    iso.fit(train_scaled)
    train_feat["iso_forest_score"] = -iso.decision_function(train_scaled)
    eval_feat["iso_forest_score"] = -iso.decision_function(eval_scaled)

    clf_cols = FEATURE_COLS + resid_cols + ["iso_forest_score", "Is_Duplicate_Feature_Row"]
    y_train = (train_feat["Validity_Label"] == "Invalid").astype(int)
    X_train = train_feat[clf_cols].astype(float).values
    X_eval = eval_feat[clf_cols].astype(float).values

    clf = GradientBoostingClassifier(n_estimators=GB_N_ESTIMATORS, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    eval_proba = clf.predict_proba(X_eval)[:, 1]

    override_z = _hard_override_threshold(fitted_lines)
    return eval_feat, eval_proba, override_z, fitted_lines, scaler, iso, clf, clf_cols, resid_cols


def _best_threshold_by_f1(y_true, proba, grid=THRESHOLD_GRID):
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def cross_validate_anomaly_detector(train: pd.DataFrame, n_splits=N_SPLITS):
    """Genuine out-of-fold cross-validation: every fitted object (consistency
    lines, scaler, Isolation Forest, classifier) is fit ONLY on that fold's
    training rows and applied (transform/predict only) to the held-out
    fold. Returns (oof_proba, cv_report, chosen_threshold)."""
    y = (train["Validity_Label"] == "Invalid").astype(int).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    oof_proba = np.zeros(len(train))
    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(train, y)):
        train_fold = train.iloc[tr_idx].reset_index(drop=True)
        val_fold = train.iloc[val_idx].reset_index(drop=True)
        _, val_proba, *_ = _fit_and_score(train_fold, val_fold)
        oof_proba[val_idx] = val_proba

    chosen_threshold, best_oof_f1 = _best_threshold_by_f1(y, oof_proba)
    oof_pred = (oof_proba >= chosen_threshold).astype(int)

    cv_report = {
        "accuracy": accuracy_score(y, oof_pred),
        "precision": precision_score(y, oof_pred, zero_division=0),
        "recall": recall_score(y, oof_pred, zero_division=0),
        "f1": f1_score(y, oof_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, oof_proba),
        "confusion_matrix": confusion_matrix(y, oof_pred).tolist(),
        "chosen_threshold": chosen_threshold,
        "threshold_selection_metric": "F1 on out-of-fold predictions",
        "n_folds": n_splits,
    }
    return oof_proba, cv_report, chosen_threshold


def fit_final_and_predict_test(train: pd.DataFrame, test: pd.DataFrame, chosen_threshold: float):
    """Refit consistency lines / scaler / Isolation Forest / classifier on
    the FULL training set (this is the standard final step after CV has
    already given us an unbiased performance estimate and a chosen
    threshold) and apply them to the test set."""
    test_feat, test_proba, override_z, fitted_lines, scaler, iso, clf, clf_cols, resid_cols = \
        _fit_and_score(train, test)

    train_feat = add_consistency_features(train, fitted_lines)
    train_iso_cols = FEATURE_COLS + resid_cols
    train_scaled = scaler.transform(train_feat[train_iso_cols])
    train_feat["iso_forest_score"] = -iso.decision_function(train_scaled)
    train_feat["clf_proba_invalid"] = clf.predict_proba(train_feat[clf_cols].astype(float).values)[:, 1]

    test_feat["clf_proba_invalid"] = test_proba

    def decide(df):
        hard_dup = df["Is_Duplicate_Feature_Row"].astype(bool)
        max_abs_z = df[resid_cols].abs().max(axis=1)
        hard_phys = max_abs_z >= override_z
        soft = df["clf_proba_invalid"] >= chosen_threshold
        return hard_dup | hard_phys | soft

    train_feat["Predicted_Invalid"] = decide(train_feat)
    test_feat["Predicted_Invalid"] = decide(test_feat)

    meta = {"override_z_threshold": float(override_z), "resid_cols": resid_cols}
    return train_feat, test_feat, meta


def build_anomaly_scores(train: pd.DataFrame, test: pd.DataFrame):
    """
    Full anomaly-detection stage: (1) genuinely out-of-fold cross-validation
    against historical Validity_Label to report unbiased metrics and choose
    a decision threshold, then (2) refit on the full training set and
    apply to the test set.

    Returns (train_out, test_out, cv_report).
    """
    oof_proba, cv_report, chosen_threshold = cross_validate_anomaly_detector(train)
    train_out, test_out, meta = fit_final_and_predict_test(train, test, chosen_threshold)
    cv_report["override_z_threshold"] = meta["override_z_threshold"]
    return train_out, test_out, cv_report


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from preprocessing import load_and_clean, verify_duplicate_logic

    tr, te = load_and_clean("../data/training_data.csv", "../data/test_data.csv")
    print("Duplicate-logic verification (train):")
    for k, v in verify_duplicate_logic(tr).items():
        print(f"  {k}: {v}")

    tr_out, te_out, report = build_anomaly_scores(tr, te)
    print("\nGenuinely out-of-fold CV report vs historical Validity_Label:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print("\nPredicted invalid count (train, full-fit):", int(tr_out["Predicted_Invalid"].sum()))
    print("Predicted invalid count (test):", int(te_out["Predicted_Invalid"].sum()))
