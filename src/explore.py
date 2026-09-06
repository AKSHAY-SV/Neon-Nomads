"""
explore.py
----------
Standalone exploratory-data-analysis script. Not required by main.py, but
documents / reproduces the analysis that informed the design choices in
preprocessing.py, anomaly_detection.py and model.py. Run directly:

    python src/explore.py

Prints: shapes, dtypes, missing-value counts, duplicate counts, descriptive
statistics, Valid-vs-Invalid comparisons, and correlation structure.
"""

import pandas as pd
import numpy as np

FEATURE_COLS = [
    "Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C",
    "Test_Duration_min", "Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4",
]


def main(train_path="../data/training_data.csv", test_path="../data/test_data.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print("=" * 70)
    print("SHAPES")
    print("=" * 70)
    print("Training:", train.shape, " Test:", test.shape)

    print("\n" + "=" * 70)
    print("DTYPES")
    print("=" * 70)
    print(train.dtypes)

    print("\n" + "=" * 70)
    print("MISSING VALUES")
    print("=" * 70)
    print("Train:\n", train.isna().sum())
    print("Test:\n", test.isna().sum())

    print("\n" + "=" * 70)
    print("DUPLICATES")
    print("=" * 70)
    print("Full-row duplicates (train):", train.duplicated().sum())
    print("Duplicate Test_IDs (train):", train["Test_ID"].duplicated().sum())
    print("Duplicate feature-vectors (train, excl. ID/target/label):",
          train.duplicated(subset=FEATURE_COLS).sum())
    print("Duplicate feature-vectors (test):",
          test.duplicated(subset=FEATURE_COLS).sum())

    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS (train)")
    print("=" * 70)
    print(train.describe().T)

    if "Validity_Label" in train.columns:
        print("\n" + "=" * 70)
        print("VALIDITY LABEL DISTRIBUTION")
        print("=" * 70)
        print(train["Validity_Label"].value_counts())

        print("\n" + "=" * 70)
        print("CORRELATION WITH Reference_Parameter (Valid rows only)")
        print("=" * 70)
        valid = train[train["Validity_Label"] == "Valid"]
        print(valid[FEATURE_COLS + ["Reference_Parameter"]].corr()["Reference_Parameter"])

        print("\n" + "=" * 70)
        print("PHYSICAL-CONSISTENCY RESIDUAL STDS (Valid vs Invalid)")
        print("=" * 70)
        for x, y in [("Applied_Voltage_kV", "Sensor_S1"),
                     ("Applied_Voltage_kV", "Sensor_S3"),
                     ("Load_Current_A", "Sensor_S2")]:
            mask = train[[x, y]].notna().all(axis=1)
            slope, intercept = np.polyfit(train.loc[mask, x], train.loc[mask, y], 1)
            resid = train.loc[mask, y] - (slope * train.loc[mask, x] + intercept)
            tmp = train.loc[mask].copy()
            tmp["_resid"] = resid
            print(f"\n{y} ~ {x} (slope={slope:.3f}, intercept={intercept:.3f})")
            print(tmp.groupby("Validity_Label")["_resid"].agg(["mean", "std", "min", "max"]))


if __name__ == "__main__":
    main()
