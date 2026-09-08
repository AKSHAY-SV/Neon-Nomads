from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

OUTLIER_IQR_MULTIPLIER = 1.5


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
        n_total = len(X)
        cols = [c for c in self.ranges if c in X.columns]
        if not cols:
            return X.values

        lo = np.array([self.ranges[c][0] for c in cols], dtype=float)
        hi = np.array([self.ranges[c][1] for c in cols], dtype=float)
        sub = X[cols].to_numpy(dtype=float, copy=True)
        bad = (sub < lo) | (sub > hi)  # broadcasts (n_rows, n_cols) vs (n_cols,)

        n_bad_per_col = bad.sum(axis=0)
        for col, n_bad in zip(cols, n_bad_per_col):
            if n_bad > 0:
                lo_c, hi_c = self.ranges[col]
                logger.info(f"ImpossibleValueClipper: {col} - {int(n_bad)}/{n_total} values outside [{lo_c}, {hi_c}] set to NaN")

        sub[bad] = np.nan
        X[cols] = sub
        return X.values


class InfiniteValueHandler(BaseEstimator, TransformerMixin):
    """Replaces infinite values with NaN so they get imputed."""
    
    def __init__(self, columns=None):
        self.columns = columns or FEATURE_COLS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns).copy()
        n_total = len(X)
        cols = [c for c in self.columns if c in X.columns]
        if not cols:
            return X.values

        sub = X[cols].to_numpy(dtype=float, copy=True)
        bad = np.isinf(sub)
        n_bad_per_col = bad.sum(axis=0)
        for col, n_bad in zip(cols, n_bad_per_col):
            if n_bad > 0:
                logger.info(f"InfiniteValueHandler: {col} - {int(n_bad)}/{n_total} infinite values set to NaN")

        sub[bad] = np.nan
        X[cols] = sub
        return X.values



RAW_FEATURE_COLS = FEATURE_COLS  # alias: raw sensor/operating-condition columns
TARGET_COL = "Reference_Parameter"

RATIO_CLIP = 1e4


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive physics-informed features from the raw operating-condition
    and sensor columns. Purely row-wise algebra (no fitting, no dataset
    statistics), so this is safe to call independently on train and test
    without any leakage risk - unlike the KNN imputer above, it has no
    state to fit.
    """
    df_feat = df.copy()

    voltage = df_feat["Applied_Voltage_kV"].astype(float)
    current = df_feat["Load_Current_A"].astype(float)
    temp = df_feat["Ambient_Temperature_C"].astype(float)
    duration = df_feat["Test_Duration_min"].astype(float)

    eps = 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        df_feat["Power_kVA"] = voltage * current
        df_feat["Impedance_proxy"] = (voltage / (current + eps)).clip(-RATIO_CLIP, RATIO_CLIP)
        df_feat["Energy_proxy"] = df_feat["Power_kVA"] * (duration / 60.0)
        df_feat["Thermal_Load"] = current * temp
        df_feat["Temp_Duration"] = temp * duration

        sensor_cols = ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]
        sensors = df_feat[sensor_cols].astype(float)

        df_feat["Sensor_Mean"] = sensors.mean(axis=1)
        df_feat["Sensor_Std"] = sensors.std(axis=1).fillna(0.0)
        df_feat["Sensor_Min"] = sensors.min(axis=1)
        df_feat["Sensor_Max"] = sensors.max(axis=1)
        df_feat["Sensor_Spread"] = df_feat["Sensor_Max"] - df_feat["Sensor_Min"]

        df_feat["Sensor_Diff_12"] = df_feat["Sensor_S1"] - df_feat["Sensor_S2"]
        df_feat["Sensor_Diff_34"] = df_feat["Sensor_S3"] - df_feat["Sensor_S4"]
        df_feat["Sensor_Ratio_12"] = (df_feat["Sensor_S1"] / (df_feat["Sensor_S2"].abs() + eps)).clip(-RATIO_CLIP, RATIO_CLIP)
        df_feat["Sensor_Ratio_34"] = (df_feat["Sensor_S3"] / (df_feat["Sensor_S4"].abs() + eps)).clip(-RATIO_CLIP, RATIO_CLIP)

        df_feat["Power_Sensor_Ratio"] = (df_feat["Power_kVA"] / (df_feat["Sensor_Mean"].abs() + eps)).clip(-RATIO_CLIP, RATIO_CLIP)
        
        # Additional physics-informed features
        df_feat["Voltage_Current_Ratio"] = (voltage / (current + eps)).clip(-RATIO_CLIP, RATIO_CLIP)
        df_feat["Current_Density_proxy"] = current / (voltage + eps)
        df_feat["Thermal_Impedance"] = (temp + 273.15) / (current + eps)
        df_feat["Power_Duration"] = df_feat["Power_kVA"] * duration
        df_feat["Sensor_S1_S3_Ratio"] = (df_feat["Sensor_S1"] / (df_feat["Sensor_S3"].abs() + eps)).clip(-RATIO_CLIP, RATIO_CLIP)
        df_feat["Sensor_S2_S4_Ratio"] = (df_feat["Sensor_S2"] / (df_feat["Sensor_S4"].abs() + eps)).clip(-RATIO_CLIP, RATIO_CLIP)
        df_feat["Sensor_Geometric_Mean"] = (df_feat["Sensor_S1"] * df_feat["Sensor_S2"] * df_feat["Sensor_S3"] * df_feat["Sensor_S4"]).clip(lower=eps) ** 0.25
        df_feat["Sensor_Harmonic_Mean"] = 4 / (1/(df_feat["Sensor_S1"]+eps) + 1/(df_feat["Sensor_S2"]+eps) + 1/(df_feat["Sensor_S3"]+eps) + 1/(df_feat["Sensor_S4"]+eps))
        df_feat["Voltage_Squared"] = voltage ** 2
        df_feat["Current_Squared"] = current ** 2
        df_feat["Power_Density"] = df_feat["Power_kVA"] / (voltage + eps)

    return df_feat


_NON_FEATURE_COLS = {
    TARGET_COL, "Test_ID", "Record_ID", "Validity_Label", "Anomaly_Flag", "ID", "Split",
    "Is_Duplicate_Feature_Row", "Predicted_Invalid", "clf_proba_invalid",
    "iso_forest_score", "_attention_score",
}


def get_feature_names(df: pd.DataFrame) -> list:
    """All numeric, non-metadata/target columns currently in `df` - i.e.
    whatever raw + engineered features happen to be present."""
    return [col for col in df.columns
            if col not in _NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[col])]


def select_feature_set(df: pd.DataFrame, feature_set: str = "all") -> list:
    """Select a named feature subset for modelling/ablation.

    Options:
    - 'all': all raw + engineered features
    - 'no_s4': all features except anything derived from Sensor_S4
    - 'important_only': the handful of raw + engineered features most
      correlated with Reference_Parameter (see methodology.md section 2)
    - 'engineered_only': only the derived features, no raw columns
    - 'raw_only': only the raw sensor/operating-condition columns
    """
    all_features = get_feature_names(df)

    if feature_set == "all":
        return all_features
    elif feature_set == "no_s4":
        return [f for f in all_features if "Sensor_S4" not in f and "S4" not in f]
    elif feature_set == "important_only":
        important = ["Load_Current_A", "Sensor_S2", "Ambient_Temperature_C",
                     "Power_kVA", "Thermal_Load", "Sensor_Mean"]
        return [f for f in all_features if any(imp in f for imp in important)]
    elif feature_set == "engineered_only":
        raw_set = set(RAW_FEATURE_COLS)
        return [f for f in all_features if f not in raw_set]
    elif feature_set == "raw_only":
        return [f for f in all_features if f in RAW_FEATURE_COLS]
    else:
        return all_features


def build_feature_pipeline(n_neighbors: int = 7) -> Pipeline:
    """Returns an UNFIT sklearn Pipeline: infinite value handling ->
    impossible-value clipping -> KNN imputation. 
    Call `.fit(train[FEATURE_COLS])` once on the training data, 
    then transform-only on the test data (never refit on test)."""
    return Pipeline([
        ("inf_handler", InfiniteValueHandler()),
        ("clip", ImpossibleValueClipper()),
        ("impute", KNNImputer(n_neighbors=n_neighbors, weights="distance")),
    ])


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _log_missing_values(df: pd.DataFrame, name: str) -> dict:
    """Log missing value counts per column and return summary dict."""
    missing = df[FEATURE_COLS].isna().sum()
    total_missing = int(missing.sum())
    logger.info(f"{name}: Total missing values across features: {total_missing}")
    for col in FEATURE_COLS:
        n_missing = int(missing[col])
        if n_missing > 0:
            logger.info(f"  {col}: {n_missing}/{len(df)} missing ({100*n_missing/len(df):.1f}%)")
    return {"total_missing": total_missing, "per_column": missing.to_dict()}


def _log_duplicate_rows(df: pd.DataFrame, name: str) -> dict:
    """Log exact duplicate rows (all columns) and return summary."""
    n_exact_dup = int(df.duplicated().sum())
    logger.info(f"{name}: Exact duplicate rows (all columns): {n_exact_dup}")
    return {"exact_duplicate_rows": n_exact_dup}


def _log_infinite_values(df: pd.DataFrame, name: str) -> dict:
    """Log infinite values per column and return summary."""
    inf_counts = {}
    total_inf = 0
    for col in FEATURE_COLS:
        if col in df.columns:
            n_inf = int(np.isinf(df[col]).sum())
            inf_counts[col] = n_inf
            total_inf += n_inf
    logger.info(f"{name}: Total infinite values across features: {total_inf}")
    return {"total_infinite": total_inf, "per_column": inf_counts}


def _detect_outliers_iqr(df: pd.DataFrame, name: str) -> dict:
    """Detect outliers using IQR method (logged only, not removed)."""
    outlier_counts = {}
    for col in FEATURE_COLS:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower = Q1 - OUTLIER_IQR_MULTIPLIER * IQR
                upper = Q3 + OUTLIER_IQR_MULTIPLIER * IQR
                outliers = (df[col] < lower) | (df[col] > upper)
                n_outliers = int(outliers.sum())
                outlier_counts[col] = n_outliers
                if n_outliers > 0:
                    logger.info(f"{name}: {col} - {n_outliers}/{len(df)} outliers (IQR bounds: [{lower:.2f}, {upper:.2f}])")
    return {"per_column": outlier_counts}


def _align_columns(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Ensure train and test have the same columns. Add missing columns with NaN."""
    all_cols = set(train.columns) | set(test.columns)
    for col in all_cols:
        if col not in train.columns:
            train[col] = np.nan
            logger.warning(f"Column '{col}' missing in train, added with NaN")
        if col not in test.columns:
            test[col] = np.nan
            logger.warning(f"Column '{col}' missing in test, added with NaN")
    return train, test


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


def load_and_clean(train_path: str, test_path: str, log_stats: bool = True):
    """Load raw CSVs, clean both CONSISTENTLY (fit imputer on train only,
    transform test with the same fitted imputer), and return
    (train_df, test_df) with an added boolean column
    `Is_Duplicate_Feature_Row` and no missing values in the feature columns.
    
    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        log_stats: Whether to log preprocessing statistics
        
    Returns:
        Tuple of (train_df, test_df, preprocessing_stats)
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if log_stats:
        logger.info("=" * 60)
        logger.info("PREPROCESSING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Raw train shape: {train.shape}")
        logger.info(f"Raw test shape: {test.shape}")

    train = _coerce_types(train)
    test = _coerce_types(test)

    train, test = _align_columns(train, test)

    if log_stats:
        _log_missing_values(train, "Train (after type coercion)")
        _log_missing_values(test, "Test (after type coercion)")
        _log_duplicate_rows(train, "Train")
        _log_duplicate_rows(test, "Test")
        _log_infinite_values(train, "Train")
        _log_infinite_values(test, "Test")
        _detect_outliers_iqr(train, "Train")
        _detect_outliers_iqr(test, "Test")

    train["Is_Duplicate_Feature_Row"] = flag_exact_duplicate_features(train)
    test["Is_Duplicate_Feature_Row"] = flag_exact_duplicate_features(test)

    if log_stats:
        n_dup_train = int(train["Is_Duplicate_Feature_Row"].sum())
        n_dup_test = int(test["Is_Duplicate_Feature_Row"].sum())
        logger.info(f"Duplicate feature rows (train): {n_dup_train}")
        logger.info(f"Duplicate feature rows (test): {n_dup_test}")

    pipeline = build_feature_pipeline(n_neighbors=7)
    train_imputed = pipeline.fit_transform(train[FEATURE_COLS])
    test_imputed = pipeline.transform(test[FEATURE_COLS])

    train[FEATURE_COLS] = train_imputed
    test[FEATURE_COLS] = test_imputed

    if log_stats:
        logger.info(f"Missing values remaining (train): {pd.DataFrame(train_imputed, columns=FEATURE_COLS).isna().sum().sum()}")
        logger.info(f"Missing values remaining (test): {pd.DataFrame(test_imputed, columns=FEATURE_COLS).isna().sum().sum()}")
        logger.info("=" * 60)

    stats = {
        "n_train": len(train),
        "n_test": len(test),
        "n_duplicate_feature_rows_train": int(train["Is_Duplicate_Feature_Row"].sum()),
        "n_duplicate_feature_rows_test": int(test["Is_Duplicate_Feature_Row"].sum()),
    }

    return train, test, stats


def prepare_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Convenience wrapper: apply `engineer_features` to already-cleaned
    train/test frames (i.e. the output of `load_and_clean`). Row-wise only,
    so calling it independently on train and test is leakage-free (see
    `engineer_features` docstring)."""
    return engineer_features(train), engineer_features(test)


if __name__ == "__main__":
    tr, te, stats = load_and_clean("../data/training_data.csv", "../data/test_data.csv")
    print("Train shape after cleaning:", tr.shape)
    print("Test shape after cleaning:", te.shape)
    print("Missing values remaining (train):", tr[FEATURE_COLS].isna().sum().sum())
    print("Missing values remaining (test):", te[FEATURE_COLS].isna().sum().sum())
    print("Duplicate feature rows (train):", tr["Is_Duplicate_Feature_Row"].sum())
    print("Duplicate feature rows (test):", te["Is_Duplicate_Feature_Row"].sum())
    print("\nDuplicate-logic verification (train):")
    for k, v in verify_duplicate_logic(tr).items():
        print(f"  {k}: {v}")
