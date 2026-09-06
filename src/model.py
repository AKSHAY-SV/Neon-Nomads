from __future__ import annotations
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RAW_FEATURE_COLS = [
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
    "Sensor_S4",
]

FEATURE_COLS = RAW_FEATURE_COLS

TARGET_COL = "Reference_Parameter"
RANDOM_STATE = 42
N_ESTIMATORS = 300


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()

    voltage = df_feat["Applied_Voltage_kV"].astype(float)
    current = df_feat["Load_Current_A"].astype(float)
    temp = df_feat["Ambient_Temperature_C"].astype(float)
    duration = df_feat["Test_Duration_min"].astype(float)

    eps = 1e-6
    df_feat["Power_kVA"] = voltage * current
    df_feat["Impedance_proxy"] = voltage / (current + eps)
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
    df_feat["Sensor_Ratio_12"] = df_feat["Sensor_S1"] / (df_feat["Sensor_S2"].abs() + eps)
    df_feat["Sensor_Ratio_34"] = df_feat["Sensor_S3"] / (df_feat["Sensor_S4"].abs() + eps)

    df_feat["Power_Sensor_Ratio"] = df_feat["Power_kVA"] / (df_feat["Sensor_Mean"].abs() + eps)

    return df_feat


def get_feature_names(df: pd.DataFrame) -> List[str]:
    exclude = {TARGET_COL, "Record_ID", "Validity_Label", "Anomaly_Flag", "ID", "Split"}
    return [col for col in df.columns if col not in exclude and pd.api.types.is_numeric_dtype(df[col])]


def _candidate_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("m", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=12,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=14,
            min_samples_leaf=2,
            max_features=0.85,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.04,
            max_depth=4,
            min_samples_leaf=3,
            subsample=0.85,
            max_features=0.9,
            loss="huber",
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=N_ESTIMATORS,
            learning_rate=0.04,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }

    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.04,
            num_leaves=24,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = CatBoostRegressor(
            iterations=N_ESTIMATORS,
            depth=5,
            learning_rate=0.04,
            l2_leaf_reg=3.0,
            random_state=RANDOM_STATE,
            verbose=False,
        )
    except ImportError:
        pass

    ensemble_estimators = [
        ("gb", GradientBoostingRegressor(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.04,
            max_depth=4,
            min_samples_leaf=3,
            subsample=0.85,
            max_features=0.9,
            loss="huber",
            random_state=RANDOM_STATE,
        )),
        ("et", ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=14,
            min_samples_leaf=2,
            max_features=0.85,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
        ("hgb", HistGradientBoostingRegressor(
            max_iter=N_ESTIMATORS,
            learning_rate=0.04,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )),
    ]
    models["VotingEnsemble"] = VotingRegressor(
        estimators=ensemble_estimators,
        weights=[0.45, 0.30, 0.25],
    )

    return models


def compare_models(
    train_subset: pd.DataFrame,
    n_splits: int = 5,
    use_feature_engineering: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    if use_feature_engineering:
        df_processed = engineer_features(train_subset)
    else:
        df_processed = train_subset.copy()

    feature_cols = get_feature_names(df_processed)
    X = df_processed[feature_cols].astype(float).values
    y = df_processed[TARGET_COL].astype(float).values

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }

    models = _candidate_models()
    rows = []
    for name, model in models.items():
        cv_res = cross_validate(model, X, y, cv=kf, scoring=scoring, n_jobs=1)
        rows.append({
            "model": name,
            "MAE": -cv_res["test_MAE"].mean(),
            "MAE_std": cv_res["test_MAE"].std(),
            "RMSE": -cv_res["test_RMSE"].mean(),
            "RMSE_std": cv_res["test_RMSE"].std(),
            "R2": cv_res["test_R2"].mean(),
            "R2_std": cv_res["test_R2"].std(),
        })

    results = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    return results, models, feature_cols


def compare_training_subsets(
    train_scored: pd.DataFrame,
    model_name: str = "GradientBoosting",
    model_builder: Optional[Any] = None,
    n_splits: int = 5,
    use_feature_engineering: bool = True,
) -> Dict[str, Dict[str, float]]:
    if model_builder is None:
        model_builder = lambda: _candidate_models()[model_name]

    valid_only = train_scored[train_scored["Validity_Label"] == "Valid"].copy()
    all_rows = train_scored.copy()

    subsets = {
        "valid_only": valid_only,
        "all_rows": all_rows,
    }

    out = {}
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }

    for label, subset in subsets.items():
        if subset.empty or TARGET_COL not in subset.columns:
            continue

        if use_feature_engineering:
            subset_feat = engineer_features(subset)
        else:
            subset_feat = subset.copy()

        feature_cols = get_feature_names(subset_feat)
        X = subset_feat[feature_cols].astype(float).values
        y = subset_feat[TARGET_COL].astype(float).values

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        model = model_builder()
        cv_res = cross_validate(model, X, y, cv=kf, scoring=scoring, n_jobs=1)

        out[label] = {
            "n_rows": len(subset),
            "MAE": float(-cv_res["test_MAE"].mean()),
            "RMSE": float(-cv_res["test_RMSE"].mean()),
            "R2": float(cv_res["test_R2"].mean()),
        }

    return out


def select_and_fit_best(
    train_subset: pd.DataFrame,
    results: pd.DataFrame,
    fitted_models: Dict[str, Any],
    feature_cols: Optional[List[str]] = None,
    use_feature_engineering: bool = True,
) -> Tuple[str, Any, List[str]]:
    if use_feature_engineering:
        df_processed = engineer_features(train_subset)
    else:
        df_processed = train_subset.copy()

    if feature_cols is None:
        feature_cols = get_feature_names(df_processed)

    best_name = results.iloc[0]["model"]
    best_model = fitted_models[best_name]

    X = df_processed[feature_cols].astype(float).values
    y = df_processed[TARGET_COL].astype(float).values
    best_model.fit(X, y)

    return best_name, best_model, feature_cols


def feature_importance(
    best_name: str,
    best_model: Any,
    feature_cols: List[str],
) -> Optional[pd.Series]:
    try:
        if hasattr(best_model, "feature_importances_"):
            imp = best_model.feature_importances_
        elif hasattr(best_model, "named_steps") and hasattr(best_model.named_steps.get("m"), "coef_"):
            imp = np.abs(best_model.named_steps["m"].coef_)
        elif hasattr(best_model, "estimators_"):
            importances = []
            for est in best_model.estimators_:
                if hasattr(est, "feature_importances_"):
                    importances.append(est.feature_importances_)
            if importances:
                imp = np.mean(importances, axis=0)
            else:
                return None
        else:
            return None

        return pd.Series(imp, index=feature_cols).sort_values(ascending=False)
    except Exception:
        return None


def predict(
    model: Any,
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    use_feature_engineering: bool = True,
) -> np.ndarray:
    if use_feature_engineering and "Power_kVA" not in df.columns:
        df_processed = engineer_features(df)
    else:
        df_processed = df.copy()

    if feature_cols is None:
        feature_cols = [c for c in get_feature_names(df_processed) if c in df_processed.columns]

    X = df_processed[feature_cols].astype(float).values
    return model.predict(X)


def save_model(model: Any, path: Path | str, feature_cols: Optional[List[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_cols": feature_cols,
    }
    joblib.dump(payload, path)


def load_model(path: Path | str) -> Tuple[Any, Optional[List[str]]]:
    obj = joblib.load(Path(path))
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj.get("feature_cols")
    return obj, None


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from preprocessing import load_and_clean

    tr, te = load_and_clean("../data/training_data.csv", "../data/test_data.csv")
    tr_valid = tr[tr["Validity_Label"] == "Valid"]

    results, fitted, feature_cols = compare_models(tr_valid)
    print(results.to_string())

    best_name, best_model, feature_cols = select_and_fit_best(tr_valid, results, fitted, feature_cols=feature_cols)
    print("\nBest model:", best_name)

    imp = feature_importance(best_name, best_model, feature_cols)
    if imp is not None:
        print("\nFeature importance:\n", imp)
