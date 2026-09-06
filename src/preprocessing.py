"""
preprocessing.py
----------------
Loads the raw training/test data and performs cleaning:
- type coercion
- impossible-value handling (physically implausible readings -> NaN)
- duplicate-feature-row flagging (structural, uses no target/label information)
- missing-value imputation via a transformer that is FIT ONLY ON TRAINING
  DATA and then reused (transform-only) on the test data, to avoid any
  train/test leakage.

Fix vs previous version
------------------------
The previous version called `KNNImputer().fit_transform()` independently on
train and on test. That means the "neighbours" used to impute a test row's
missing sensor value could come from other test rows (and the imputer's
internal statistics were refit on test-only data) - not a target leak in
the strict sense (no label information was used), but a violation of
prevent-the-model-touching-test-data-during-fitting hygiene, and it makes
imputation behave inconsistently between the two datasets. This version
fits ONE imputer on the training feature distribution and calls
`.transform()` (never `.fit()`) on the test data, exactly as an
sklearn Pipeline would in production/inference.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

FEATURE_COLS = [
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
    "Sensor_S4",
]

# Physically implausible bounds (loose - domain sanity checks only, fixed
# constants derived from the parameter descriptions, not fit from data, so
# applying them independently to train/test introduces no leakage).
PLAUSIBLE_RANGES = {
    "Applied_Voltage_kV": (0, 100),
    "Load_Current_A": (0, 500),
    "Ambient_Temperature_C": (-20, 80),
    "Test_Duration_min": (0, 300),
    "Sensor_S1": (0, 200),
    "Sensor_S2": (0, 200),
    "Sensor_S3": (0, 200),
    "Sensor_S4": (0, 300),
}


class ImpossibleValueClipper(BaseEstimator, TransformerMixin):
    """Sets physically-impossible readings to NaN so they get imputed
    instead of silently trusted. Uses fixed domain bounds (not learned from
    data), so fit() is a no-op - this makes it safe to place inside a
    Pipeline alongside the imputer without any leakage risk."""

    def __init__(self, ranges=None, columns=None):
        self.ranges = ranges or PLAUSIBLE_RANGES
        self.columns = columns or FEATURE_COLS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns).copy()
        for col, (lo, hi) in self.ranges.items():
            if col in X.columns:
                bad = (X[col] < lo) | (X[col] > hi)
                X.loc[bad, col] = np.nan
        return X.values


def build_feature_pipeline(n_neighbors: int = 7) -> Pipeline:
    """Returns an UNFIT sklearn Pipeline: impossible-value clipping ->
    KNN imputation. Call `.fit(train[FEATURE_COLS])` once on the training
    data, then reuse the SAME fitted pipeline's `.transform()` on the test
    data (never refit on test)."""
    return Pipeline([
        ("clip", ImpossibleValueClipper()),
        ("impute", KNNImputer(n_neighbors=n_neighbors, weights="distance")),
    ])


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def flag_exact_duplicate_features(df: pd.DataFrame) -> pd.Series:
    """Rows whose full feature vector (Applied_Voltage ... Sensor_S4) is an
    exact duplicate of another row in the SAME dataframe. This uses only
    the feature columns (never Validity_Label or Reference_Parameter), so
    computing it independently on train and on test introduces no leakage
    - it is a structural property of the raw measurements, not something
    learned from labels.

    In the training data every such duplicate pair (12 pairs / 24 rows) was
    labelled Invalid (see verify_duplicate_logic below) - a documented
    "duplicate measurement pair" data-quality issue. The same pattern
    exists in the raw test data (4 pairs / 8 rows) and is detected purely
    from the test set's own feature vectors, so it generalises correctly to
    unseen data without ever referencing training records."""
    present_cols = [c for c in FEATURE_COLS if c in df.columns]
    return df.duplicated(subset=present_cols, keep=False)


def verify_duplicate_logic(train: pd.DataFrame) -> dict:
    """Programmatic verification that duplicate-feature-vector rows in the
    training data are (almost) always labelled Invalid. Returns counts so
    this can be asserted/reported rather than assumed."""
    dup_mask = flag_exact_duplicate_features(train)
    n_dup_rows = int(dup_mask.sum())
    n_dup_pairs = n_dup_rows // 2 if n_dup_rows else 0
    if n_dup_rows and "Validity_Label" in train.columns:
        dup_invalid = int((train.loc[dup_mask, "Validity_Label"] == "Invalid").sum())
    else:
        dup_invalid = None
    return {
        "n_duplicate_rows": n_dup_rows,
        "n_duplicate_pairs": n_dup_pairs,
        "n_duplicate_rows_labelled_invalid": dup_invalid,
        "all_duplicates_invalid": (dup_invalid == n_dup_rows) if dup_invalid is not None else None,
    }


def load_and_clean(train_path: str, test_path: str):
    """Load raw CSVs, clean both CONSISTENTLY (fit imputer on train only,
    transform test with the same fitted imputer), and return
    (train_df, test_df) with an added boolean column
    `Is_Duplicate_Feature_Row` and no missing values in the feature columns.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train = _coerce_types(train)
    test = _coerce_types(test)

    # Duplicate flag uses only feature columns -> safe to compute
    # independently on each dataset (no leakage, see docstring above).
    train["Is_Duplicate_Feature_Row"] = flag_exact_duplicate_features(train)
    test["Is_Duplicate_Feature_Row"] = flag_exact_duplicate_features(test)

    # Fit the impossible-value-clip + KNN-impute pipeline ONCE on training
    # features, then transform-only on test features.
    pipeline = build_feature_pipeline(n_neighbors=7)
    train_imputed = pipeline.fit_transform(train[FEATURE_COLS])
    test_imputed = pipeline.transform(test[FEATURE_COLS])

    train[FEATURE_COLS] = train_imputed
    test[FEATURE_COLS] = test_imputed

    return train, test


if __name__ == "__main__":
    tr, te = load_and_clean("../data/training_data.csv", "../data/test_data.csv")
    print("Train shape after cleaning:", tr.shape)
    print("Test shape after cleaning:", te.shape)
    print("Missing values remaining (train):", tr[FEATURE_COLS].isna().sum().sum())
    print("Missing values remaining (test):", te[FEATURE_COLS].isna().sum().sum())
    print("Duplicate feature rows (train):", tr["Is_Duplicate_Feature_Row"].sum())
    print("Duplicate feature rows (test):", te["Is_Duplicate_Feature_Row"].sum())
    print("\nDuplicate-logic verification (train):")
    for k, v in verify_duplicate_logic(tr).items():
        print(f"  {k}: {v}")
