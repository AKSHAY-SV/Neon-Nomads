from __future__ import annotations
from typing import Dict, Tuple, List, Optional, Any, Callable
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
from sklearn.model_selection import KFold, RepeatedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import (
    RAW_FEATURE_COLS,
    TARGET_COL,
    engineer_features,
    get_feature_names,
    select_feature_set,
)

FEATURE_COLS = RAW_FEATURE_COLS

RANDOM_STATE = 42
N_ESTIMATORS = 300


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
        ("xgb", XGBRegressor(
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
        )),
        ("lgb", LGBMRegressor(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.04,
            num_leaves=24,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )),
        ("cat", CatBoostRegressor(
            iterations=N_ESTIMATORS,
            depth=5,
            learning_rate=0.04,
            l2_leaf_reg=3.0,
            random_state=RANDOM_STATE,
            verbose=False,
        )),
    ]
    models["VotingEnsemble"] = VotingRegressor(
        estimators=ensemble_estimators,
        weights=[0.30, 0.20, 0.15, 0.15, 0.10, 0.10],
    )

    return models


def _get_repeated_kfold(n_splits: int = 5, n_repeats: int = 3, random_state: int = RANDOM_STATE) -> RepeatedKFold:
    """Get repeated K-Fold cross-validator."""
    return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)


def compare_models(
    train_subset: pd.DataFrame,
    n_splits: int = 5,
    n_repeats: int = 1,
    use_feature_engineering: bool = True,
    feature_set: str = "all",
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    if use_feature_engineering:
        df_processed = engineer_features(train_subset)
    else:
        df_processed = train_subset.copy()

    feature_cols = select_feature_set(df_processed, feature_set)
    X = df_processed[feature_cols].astype(float).values
    y = df_processed[TARGET_COL].astype(float).values

    if n_repeats > 1:
        cv = _get_repeated_kfold(n_splits=n_splits, n_repeats=n_repeats)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }

    models = _candidate_models()
    rows = []
    for name, model in models.items():
        cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        rows.append({
            "model": name,
            "MAE": -cv_res["test_MAE"].mean(),
            "MAE_std": cv_res["test_MAE"].std(),
            "RMSE": -cv_res["test_RMSE"].mean(),
            "RMSE_std": cv_res["test_RMSE"].std(),
            "R2": cv_res["test_R2"].mean(),
            "R2_std": cv_res["test_R2"].std(),
            "cv_folds": n_splits * n_repeats,
        })

    results = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    return results, models, feature_cols


def compare_feature_sets(
    train_subset: pd.DataFrame,
    model_name: str = "GradientBoosting",
    n_splits: int = 5,
    n_repeats: int = 3,
) -> pd.DataFrame:
    """Compare different feature set configurations."""
    feature_configs = ["all", "no_s4", "important_only", "engineered_only", "raw_only"]
    model_builder = lambda: _candidate_models()[model_name]
    
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }
    
    cv = _get_repeated_kfold(n_splits=n_splits, n_repeats=n_repeats)
    df_processed = engineer_features(train_subset)
    
    rows = []
    for feature_set in feature_configs:
        feature_cols = select_feature_set(df_processed, feature_set)
        X = df_processed[feature_cols].astype(float).values
        y = df_processed[TARGET_COL].astype(float).values
        
        model = model_builder()
        cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        
        rows.append({
            "feature_set": feature_set,
            "n_features": len(feature_cols),
            "MAE": -cv_res["test_MAE"].mean(),
            "MAE_std": cv_res["test_MAE"].std(),
            "RMSE": -cv_res["test_RMSE"].mean(),
            "RMSE_std": cv_res["test_RMSE"].std(),
            "R2": cv_res["test_R2"].mean(),
            "R2_std": cv_res["test_R2"].std(),
        })
    
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def compare_training_subsets(
    train_scored: pd.DataFrame,
    model_name: str = "GradientBoosting",
    model_builder: Optional[Any] = None,
    n_splits: int = 5,
    n_repeats: int = 1,
    use_feature_engineering: bool = True,
    feature_set: str = "all",
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

        feature_cols = select_feature_set(subset_feat, feature_set)
        X = subset_feat[feature_cols].astype(float).values
        y = subset_feat[TARGET_COL].astype(float).values

        if n_repeats > 1:
            cv = _get_repeated_kfold(n_splits=n_splits, n_repeats=n_repeats)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        
        model = model_builder()
        cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)

        out[label] = {
            "n_rows": len(subset),
            "MAE": float(-cv_res["test_MAE"].mean()),
            "RMSE": float(-cv_res["test_RMSE"].mean()),
            "R2": float(cv_res["test_R2"].mean()),
        }

    return out


def evaluate_on_holdout(
    train_subset: pd.DataFrame,
    holdout_subset: pd.DataFrame,
    model_builder: Callable,
    use_feature_engineering: bool = True,
    feature_set: str = "all",
) -> Dict[str, float]:
    """Evaluate model on untouched holdout set."""
    if use_feature_engineering:
        train_feat = engineer_features(train_subset)
        holdout_feat = engineer_features(holdout_subset)
    else:
        train_feat = train_subset.copy()
        holdout_feat = holdout_subset.copy()

    feature_cols = select_feature_set(train_feat, feature_set)
    feature_cols = [c for c in feature_cols if c in holdout_feat.columns]
    
    X_train = train_feat[feature_cols].astype(float).values
    y_train = train_feat[TARGET_COL].astype(float).values
    X_holdout = holdout_feat[feature_cols].astype(float).values
    y_holdout = holdout_feat[TARGET_COL].astype(float).values

    model = model_builder()
    model.fit(X_train, y_train)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_pred = model.predict(X_holdout)
    
    return {
        "MAE": float(mean_absolute_error(y_holdout, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_holdout, y_pred))),
        "R2": float(r2_score(y_holdout, y_pred)),
    }


def optimize_hyperparameters_optuna(
    train_subset: pd.DataFrame,
    model_name: str,
    n_trials: int = 100,
    n_splits: int = 5,
    n_repeats: int = 1,
    use_feature_engineering: bool = True,
    feature_set: str = "all",
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Optimize hyperparameters using Optuna."""
    try:
        import optuna
    except ImportError:
        raise ImportError("Optuna not installed. Run: pip install optuna")
    
    if use_feature_engineering:
        df_processed = engineer_features(train_subset)
    else:
        df_processed = train_subset.copy()

    feature_cols = select_feature_set(df_processed, feature_set)
    X = df_processed[feature_cols].astype(float).values
    y = df_processed[TARGET_COL].astype(float).values

    if n_repeats > 1:
        cv = _get_repeated_kfold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        if model_name == "GradientBoosting":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "max_features": trial.suggest_float("max_features", 0.5, 1.0),
                "loss": "huber",
                "random_state": random_state,
            }
            model = GradientBoostingRegressor(**params)
        elif model_name == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_float("max_features", 0.5, 1.0),
                "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
                "random_state": random_state,
                "n_jobs": -1,
            }
            model = RandomForestRegressor(**params)
        elif model_name == "ExtraTrees":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_float("max_features", 0.5, 1.0),
                "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
                "random_state": random_state,
                "n_jobs": -1,
            }
            model = ExtraTreesRegressor(**params)
        elif model_name == "HistGradientBoosting":
            params = {
                "max_iter": trial.suggest_int("max_iter", 100, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
                "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 5.0),
                "random_state": random_state,
            }
            model = HistGradientBoostingRegressor(**params)
        elif model_name == "XGBoost":
            try:
                from xgboost import XGBRegressor
            except ImportError:
                raise ImportError("XGBoost not installed")
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
                "random_state": random_state,
                "n_jobs": -1,
                "verbosity": 0,
            }
            model = XGBRegressor(**params)
        elif model_name == "LightGBM":
            try:
                from lightgbm import LGBMRegressor
            except ImportError:
                raise ImportError("LightGBM not installed")
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
                "random_state": random_state,
                "verbosity": -1,
            }
            model = LGBMRegressor(**params)
        else:
            raise ValueError(f"Optimization not implemented for {model_name}")

        cv_res = cross_validate(model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)
        return -cv_res["test_score"].mean()

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params


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


def get_model_selection_table(results: pd.DataFrame, include_classification: bool = True) -> str:
    """Generate a model selection table with all relevant metrics.
    
    Returns a formatted string table suitable for terminal output.
    """
    table_lines = []
    table_lines.append("=" * 78)
    table_lines.append("MODEL SELECTION TABLE")
    table_lines.append("=" * 78)
    table_lines.append(f"{'MODEL':<30} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Accuracy':>10} {'F1':>8}")
    table_lines.append("-" * 78)
    
    for _, row in results.iterrows():
        model_name = row["model"]
        mae = row["MAE"]
        rmse = row["RMSE"]
        r2 = row["R2"]
        accuracy = ""
        f1 = ""
        
        table_lines.append(f"{model_name:<30} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f} {accuracy:>10} {f1:>8}")
    
    table_lines.append("-" * 78)
    table_lines.append("")
    table_lines.append(f"{'BEST MODEL':<30} (lowest CV MAE)")
    table_lines.append("=" * 78)
    table_lines.append("")
    
    if include_classification:
        table_lines.append("CLASSIFICATION METRICS (on Valid/Invalid):")
        table_lines.append("-" * 78)
        table_lines.append(f"{'Metric':<15} {'Value':>15}")
        table_lines.append("-" * 78)
        table_lines.append(f"{'Accuracy':<15} (from CV report)")
        table_lines.append(f"{'Precision':<15} (from CV report)")
        table_lines.append(f"{'Recall':<15} (from CV report)")
        table_lines.append(f"{'F1':<15} (from CV report)")
        table_lines.append(f"{'ROC-AUC':<15} (from CV report)")
        table_lines.append("")
    
    return "\n".join(table_lines)


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
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocessing import load_and_clean

    _data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    tr, te, _ = load_and_clean(
        os.path.join(_data_dir, "training_data.csv"),
        os.path.join(_data_dir, "test_data.csv"),
    )
    tr_valid = tr[tr["Validity_Label"] == "Valid"]

    results, fitted, feature_cols = compare_models(tr_valid)
    print(results.to_string())

    best_name, best_model, feature_cols = select_and_fit_best(tr_valid, results, fitted, feature_cols=feature_cols)
    print("\nBest model:", best_name)

    imp = feature_importance(best_name, best_model, feature_cols)
    if imp is not None:
        print("\nFeature importance:\n", imp)
