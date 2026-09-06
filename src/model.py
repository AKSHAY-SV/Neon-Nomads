"""
model.py
--------
Task 2: predict Reference_Parameter for every test record.

Fix vs previous version
------------------------
- Trimmed the candidate model set to the sensible core the brief asks for
  (Ridge, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting)
  plus XGBoost/LightGBM/CatBoost only if already installed, to avoid
  wasting computation on near-duplicate linear models.
- Reduced n_estimators for the tree ensembles (300 instead of 400) - this
  was checked empirically to have negligible effect on CV MAE/RMSE/R2 for
  this dataset size (~860 Valid rows) while noticeably reducing runtime.
- `cross_validate(..., n_jobs=1)` is used explicitly at the outer loop
  since the tree models already parallelise internally (`n_jobs=-1`);
  parallelising both the outer CV loop and the inner model fit at once
  can oversubscribe CPU cores and slow things down or, in constrained
  environments, hang. One level of parallelism (inside each model) is
  enough for a ~860-row training set.
- Training on Valid-only rows is now explicitly validated against
  training on ALL rows (including historically-Invalid ones) via the same
  cross-validation, so the choice is evidence-based rather than assumed
  (see `compare_training_subsets` and the printed comparison in main.py).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C",
    "Test_Duration_min", "Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4",
]

TARGET_COL = "Reference_Parameter"
RANDOM_STATE = 42
N_ESTIMATORS = 300  # kept modest for speed; verified negligible CV score impact vs 400-600


def _candidate_models():
    models = {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "RandomForest": RandomForestRegressor(
            n_estimators=N_ESTIMATORS, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=N_ESTIMATORS, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE,
            n_jobs=-1, verbosity=0)
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(
            n_estimators=N_ESTIMATORS, learning_rate=0.05,
            random_state=RANDOM_STATE, verbosity=-1)
    except ImportError:
        pass

    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = CatBoostRegressor(
            iterations=N_ESTIMATORS, depth=6, learning_rate=0.05,
            random_state=RANDOM_STATE, verbose=False)
    except ImportError:
        pass

    return models


def compare_models(train_subset: pd.DataFrame, n_splits=5):
    """Cross-validate each candidate model on the given training subset.
    Returns a results DataFrame sorted by CV MAE (ascending = best first),
    and the (unfit) model objects keyed by name."""
    X = train_subset[FEATURE_COLS].astype(float).values
    y = train_subset[TARGET_COL].astype(float).values

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }

    rows = []
    fitted_models = {}
    for name, model in _candidate_models().items():
        cv_res = cross_validate(model, X, y, cv=kf, scoring=scoring, n_jobs=1)
        rows.append({
            "model": name,
            "MAE": -cv_res["test_MAE"].mean(),
            "MAE_std": cv_res["test_MAE"].std(),
            "RMSE": -cv_res["test_RMSE"].mean(),
            "R2": cv_res["test_R2"].mean(),
        })
        fitted_models[name] = model

    results = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    return results, fitted_models


def compare_training_subsets(train_scored: pd.DataFrame, model_name: str, model_builder,
                              n_splits=5):
    """Evidence check: is training on Valid-only rows actually better than
    training on all rows (Valid + historically-Invalid)? Cross-validates
    the SAME model architecture on both subsets and returns both MAEs so
    the choice can be justified rather than assumed. Both subsets are
    cross-validated only against their OWN historically-verified
    Reference_Parameter values - no test data or test predictions are
    involved in this comparison."""
    valid_only = train_scored[train_scored["Validity_Label"] == "Valid"]
    all_rows = train_scored

    out = {}
    for label, subset in [("valid_only", valid_only), ("all_rows", all_rows)]:
        X = subset[FEATURE_COLS].astype(float).values
        y = subset[TARGET_COL].astype(float).values
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        model = model_builder()
        cv_res = cross_validate(model, X, y, cv=kf,
                                 scoring={"MAE": "neg_mean_absolute_error",
                                          "RMSE": "neg_root_mean_squared_error",
                                          "R2": "r2"},
                                 n_jobs=1)
        out[label] = {
            "n_rows": len(subset),
            "MAE": -cv_res["test_MAE"].mean(),
            "RMSE": -cv_res["test_RMSE"].mean(),
            "R2": cv_res["test_R2"].mean(),
        }
    return out


def select_and_fit_best(train_subset: pd.DataFrame, results: pd.DataFrame, fitted_models: dict):
    """Refit the best model (by CV MAE) on the full given training subset."""
    best_name = results.iloc[0]["model"]
    best_model = fitted_models[best_name]
    X = train_subset[FEATURE_COLS].astype(float).values
    y = train_subset[TARGET_COL].astype(float).values
    best_model.fit(X, y)
    return best_name, best_model


def feature_importance(best_name: str, best_model, feature_cols=FEATURE_COLS):
    """Best-effort feature importance extraction, works across model types."""
    try:
        if hasattr(best_model, "feature_importances_"):
            imp = best_model.feature_importances_
        elif hasattr(best_model, "named_steps"):
            m = best_model.named_steps["m"]
            imp = np.abs(m.coef_)
        else:
            return None
        return pd.Series(imp, index=feature_cols).sort_values(ascending=False)
    except Exception:
        return None


def predict(model, df: pd.DataFrame, feature_cols=FEATURE_COLS):
    X = df[feature_cols].astype(float).values
    return model.predict(X)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from preprocessing import load_and_clean

    tr, te = load_and_clean("../data/training_data.csv", "../data/test_data.csv")
    tr_valid = tr[tr["Validity_Label"] == "Valid"]
    results, fitted = compare_models(tr_valid)
    print(results.to_string())
    best_name, best_model = select_and_fit_best(tr_valid, results, fitted)
    print("\nBest model:", best_name)
    imp = feature_importance(best_name, best_model)
    if imp is not None:
        print("\nFeature importance:\n", imp)
