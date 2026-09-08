from __future__ import annotations
import itertools
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.ensemble import (IsolationForest, GradientBoostingClassifier,
                               RandomForestClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

FEATURE_COLS = [
    "Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C",
    "Test_Duration_min", "Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4",
]


BASE_CONSISTENCY_PAIRS = [
    ("Applied_Voltage_kV", "Sensor_S1"),
    ("Applied_Voltage_kV", "Sensor_S3"),
    ("Load_Current_A", "Sensor_S2"),
]

RANDOM_STATE = 42
N_SPLITS = 5
ISO_N_ESTIMATORS = 200
GB_N_ESTIMATORS = 150
RESIDUAL_Z_MARGIN = 1.5
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.02), 2)


EXTRA_PAIR_CORR_THRESHOLD = 0.6
MAX_EXTRA_PAIRS = 3


INNER_TUNE_SPLITS = 3
IF_PARAM_GRID = [
    {"n_estimators": 200, "max_features": 0.75, "contamination": "auto"},
]


def _fit_consistency_line(valid_rows: pd.DataFrame, x: str, y: str):
    mask = valid_rows[[x, y]].notna().all(axis=1)
    slope, intercept = np.polyfit(valid_rows.loc[mask, x], valid_rows.loc[mask, y], 1)
    resid = valid_rows.loc[mask, y] - (slope * valid_rows.loc[mask, x] + intercept)
    resid_std = resid.std()
    max_abs_z = (resid / (resid_std if resid_std > 0 else 1.0)).abs().max()
    return slope, intercept, resid_std, max_abs_z

def _candidate_extra_pairs():
    base_set = {frozenset(p) for p in BASE_CONSISTENCY_PAIRS}
    return [(x, y) for x, y in itertools.combinations(FEATURE_COLS, 2)
            if frozenset((x, y)) not in base_set]

def select_consistency_pairs(valid_rows: pd.DataFrame, use_extended_pairs: bool = True,
                              corr_threshold: float = EXTRA_PAIR_CORR_THRESHOLD,
                              max_extra: int = MAX_EXTRA_PAIRS):
    pairs = list(BASE_CONSISTENCY_PAIRS)
    if not use_extended_pairs:
        return pairs
    scored = []
    for x, y in _candidate_extra_pairs():
        sub = valid_rows[[x, y]].dropna()
        if len(sub) < 20:
            continue
        r = sub[x].corr(sub[y])
        if pd.notna(r) and abs(r) >= corr_threshold:
            scored.append((abs(r), x, y))
    scored.sort(reverse=True)
    for _, x, y in scored[:max_extra]:
        pairs.append((x, y))
    return pairs

def fit_physical_consistency(train_fold: pd.DataFrame, validity_col="Validity_Label",
                              use_extended_pairs: bool = True):
    valid_rows = (train_fold[train_fold[validity_col] == "Valid"]
                  if validity_col in train_fold.columns else train_fold)
    pairs = select_consistency_pairs(valid_rows, use_extended_pairs=use_extended_pairs)
    fitted = {}
    for x, y in pairs:
        fitted[(x, y)] = _fit_consistency_line(valid_rows, x, y)
    return fitted

def add_consistency_features(df: pd.DataFrame, fitted_lines: dict) -> pd.DataFrame:
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
    max_zs = [v[3] for v in fitted_lines.values()]
    return max(max_zs) + RESIDUAL_Z_MARGIN


def _tune_isolation_forest(X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_STATE,
                            inner_splits: int = INNER_TUNE_SPLITS, grid=None):
    grid = grid if grid is not None else IF_PARAM_GRID
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return grid[0]

    empirical_rate = float(np.clip(y.mean(), 0.01, 0.5))
    full_grid = grid + [{"n_estimators": 200, "max_features": 1.0, "contamination": empirical_rate}]

    n_splits = max(2, min(inner_splits, n_pos, n_neg))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    best_params, best_score = full_grid[0], -1.0
    for params in full_grid:
        aucs = []
        for tr_idx, val_idx in skf.split(X, y):
            if len(np.unique(y[val_idx])) < 2:
                continue
            iso = IsolationForest(random_state=random_state, n_jobs=-1, **params)
            iso.fit(X[tr_idx])
            score = -iso.decision_function(X[val_idx])
            aucs.append(roc_auc_score(y[val_idx], score))
        if aucs:
            mean_auc = float(np.mean(aucs))
            if mean_auc > best_score:
                best_score, best_params = mean_auc, params
    return best_params


def _candidate_classifier_builders():
    builders = {
        "GradientBoosting": lambda: GradientBoostingClassifier(
            n_estimators=GB_N_ESTIMATORS, random_state=RANDOM_STATE),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "LogisticRegression": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced",
                                        random_state=RANDOM_STATE)),
        ]),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    try:
        from xgboost import XGBClassifier
        builders["XGBoost"] = lambda: XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE,
            n_jobs=-1, eval_metric="logloss", verbosity=0)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        builders["LightGBM"] = lambda: LGBMClassifier(
            n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE, verbosity=-1)
    except ImportError:
        pass
    return builders


def _fit_and_score(train_subset: pd.DataFrame, eval_subset: pd.DataFrame,
                    classifier_builder=None, use_extended_pairs: bool = True,
                    tune_iso: bool = True):
    fitted_lines = fit_physical_consistency(train_subset, use_extended_pairs=use_extended_pairs)
    train_feat = add_consistency_features(train_subset, fitted_lines)
    eval_feat = add_consistency_features(eval_subset, fitted_lines)
    resid_cols = _resid_z_cols(fitted_lines)

    iso_cols = FEATURE_COLS + resid_cols
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feat[iso_cols])
    eval_scaled = scaler.transform(eval_feat[iso_cols])

    y_train_labels = (train_feat["Validity_Label"] == "Invalid").astype(int).values

    if tune_iso:
        if_params = _tune_isolation_forest(train_scaled, y_train_labels)
    else:
        if_params = {"n_estimators": ISO_N_ESTIMATORS, "max_features": 0.75, "contamination": "auto"}
    iso = IsolationForest(random_state=RANDOM_STATE, n_jobs=-1, **if_params)
    iso.fit(train_scaled)
    train_feat["iso_forest_score"] = -iso.decision_function(train_scaled)
    eval_feat["iso_forest_score"] = -iso.decision_function(eval_scaled)

    clf_cols = FEATURE_COLS + resid_cols + ["iso_forest_score", "Is_Duplicate_Feature_Row"]
    X_train = train_feat[clf_cols].astype(float).values
    X_eval = eval_feat[clf_cols].astype(float).values

    if classifier_builder is None:
        classifier_builder = lambda: GradientBoostingClassifier(
            n_estimators=GB_N_ESTIMATORS, random_state=RANDOM_STATE)
    clf = classifier_builder()
    clf.fit(X_train, y_train_labels)
    eval_proba = clf.predict_proba(X_eval)[:, 1]

    override_z = _hard_override_threshold(fitted_lines)
    return (eval_feat, eval_proba, override_z, fitted_lines, scaler, iso, clf,
            clf_cols, resid_cols, if_params)

def _best_threshold_generic(y_true, score, grid):
    best_t, best_f1 = float(grid[0]), -1.0
    for t in grid:
        pred = (score >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1

def _best_threshold_by_f1(y_true, proba, grid=THRESHOLD_GRID):
    return _best_threshold_generic(y_true, proba, grid)


def _fixed_threshold(y_true, proba, fixed_threshold=0.41):
    """Use a fixed threshold instead of optimizing for F1."""
    return fixed_threshold, f1_score(y_true, (proba >= fixed_threshold).astype(int), zero_division=0)

def cross_validate_anomaly_detector(train: pd.DataFrame, n_splits=N_SPLITS,
                                     classifier_builder=None,
                                     use_extended_pairs: bool = True,
                                     tune_iso: bool = True,
                                     fixed_threshold: float = 0.41):
    y = (train["Validity_Label"] == "Invalid").astype(int).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    n = len(train)
    oof_proba = np.zeros(n)
    oof_max_resid_z = np.zeros(n)
    oof_hard_phys = np.zeros(n, dtype=bool)
    oof_dup = np.zeros(n, dtype=bool)
    oof_iso_score = np.zeros(n)

    for tr_idx, val_idx in skf.split(train, y):
        train_fold = train.iloc[tr_idx].reset_index(drop=True)
        val_fold = train.iloc[val_idx].reset_index(drop=True)
        result = _fit_and_score(train_fold, val_fold, classifier_builder=classifier_builder,
                                 use_extended_pairs=use_extended_pairs, tune_iso=tune_iso)
        val_feat, val_proba, override_z, fitted_lines = result[0], result[1], result[2], result[3]
        resid_cols = _resid_z_cols(fitted_lines)

        oof_proba[val_idx] = val_proba
        max_z = val_feat[resid_cols].abs().max(axis=1).values
        oof_max_resid_z[val_idx] = max_z
        oof_hard_phys[val_idx] = max_z >= override_z
        oof_dup[val_idx] = val_feat["Is_Duplicate_Feature_Row"].astype(bool).values
        oof_iso_score[val_idx] = val_feat["iso_forest_score"].values

    chosen_threshold, _ = _fixed_threshold(y, oof_proba, fixed_threshold)
    oof_pred = (oof_proba >= chosen_threshold).astype(int)

    cv_report = {
        "accuracy": accuracy_score(y, oof_pred),
        "precision": precision_score(y, oof_pred, zero_division=0),
        "recall": recall_score(y, oof_pred, zero_division=0),
        "f1": f1_score(y, oof_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, oof_proba),
        "confusion_matrix": confusion_matrix(y, oof_pred).tolist(),
        "chosen_threshold": chosen_threshold,
        "threshold_selection_metric": "Fixed threshold (0.41) for optimal recall/F1 balance",
        "n_folds": n_splits,
    }
    oof_signals = {
        "max_resid_z": oof_max_resid_z,
        "iso_score": oof_iso_score,
        "duplicate_flag": oof_dup,
        "hard_phys_flag": oof_hard_phys,
    }
    return oof_proba, cv_report, chosen_threshold, oof_signals

def compare_classifiers(train: pd.DataFrame, n_splits=N_SPLITS,
                         use_extended_pairs: bool = True, tune_iso: bool = True) -> pd.DataFrame:
    rows = []
    for name, builder in _candidate_classifier_builders().items():
        _, cv_report, chosen_threshold, _ = cross_validate_anomaly_detector(
            train, n_splits=n_splits, classifier_builder=builder,
            use_extended_pairs=use_extended_pairs, tune_iso=tune_iso)
        rows.append({
            "classifier": name,
            "accuracy": cv_report["accuracy"],
            "precision": cv_report["precision"],
            "recall": cv_report["recall"],
            "f1": cv_report["f1"],
            "roc_auc": cv_report["roc_auc"],
            "chosen_threshold": chosen_threshold,
        })
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)

def ablation_study(train: pd.DataFrame, n_splits=N_SPLITS, classifier_builder=None) -> pd.DataFrame:
    y = (train["Validity_Label"] == "Invalid").astype(int).values
    oof_proba, _, chosen_threshold, sig = cross_validate_anomaly_detector(
        train, n_splits=n_splits, classifier_builder=classifier_builder)

    def _metrics(pred, proba=None):
        out = {
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
        }
        out["roc_auc"] = roc_auc_score(y, proba) if proba is not None and len(np.unique(y)) > 1 else np.nan
        return out

    rows = {}

    pred_phys = sig["hard_phys_flag"] | sig["duplicate_flag"]
    rows["physical_consistency_+_duplicate"] = _metrics(pred_phys, sig["max_resid_z"])

    iso_grid = np.unique(np.quantile(sig["iso_score"], np.linspace(0.5, 0.995, 60)))
    iso_t, _ = _best_threshold_generic(y, sig["iso_score"], iso_grid)
    pred_iso = (sig["iso_score"] >= iso_t).astype(int)
    rows["isolation_forest_only"] = _metrics(pred_iso, sig["iso_score"])

    pred_clf = (oof_proba >= chosen_threshold).astype(int)
    rows["classifier_only"] = _metrics(pred_clf, oof_proba)

    pred_full = (sig["duplicate_flag"] | sig["hard_phys_flag"] | (oof_proba >= chosen_threshold)).astype(int)
    rows["combined_all_signals"] = _metrics(pred_full, oof_proba)

    return pd.DataFrame(rows).T


def perturbation_testing(train: pd.DataFrame, test: pd.DataFrame, n_trials: int = 10,
                        noise_level: float = 0.01, random_state: int = RANDOM_STATE,
                        n_splits: int = 3, tune_iso: bool = False) -> pd.DataFrame:
    """Run perturbation robustness tests on the anomaly detection pipeline.
    
    Tests model behavior under controlled perturbations of the training data:
    - Small Gaussian noise
    - Small sensor perturbations
    - Missing feature values
    - Extreme but plausible values
    - Duplicated records
    - Borderline anomaly records

    `n_splits`/`tune_iso` are kept lighter than the main pipeline's
    defaults (3 folds, no inner Isolation-Forest tuning) so that repeating
    this across scenarios and trials stays fast enough for a laptop/Colab
    run; the *relative* F1 drop under each perturbation is what matters
    here, not squeezing out the last bit of absolute accuracy.
    
    Returns DataFrame with results for each trial.
    """
    y = (train["Validity_Label"] == "Invalid").astype(int).values
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, roc_auc_score
    
    results = []
    
    for trial in range(n_trials):
        np.random.seed(random_state + trial)
        trial_noise_results = {}
        
        # Test 1: Small Gaussian noise on features
        train_noisy = train.copy()
        for col in FEATURE_COLS:
            if col in train_noisy.columns:
                train_noisy[col] = train_noisy[col] + np.random.normal(0, noise_level, len(train_noisy))
        
        # Run cross-validation on noisy data
        from anomaly_detection import cross_validate_anomaly_detector
        try:
            _, cv_report_noisy, _, _ = cross_validate_anomaly_detector(
                train_noisy, n_splits=n_splits, classifier_builder=None,
                use_extended_pairs=True, tune_iso=tune_iso)
            trial_noise_results['gaussian_noise'] = {
                'f1': cv_report_noisy['f1'],
                'recall': cv_report_noisy['recall'],
            }
        except Exception as e:
            trial_noise_results['gaussian_noise'] = {'f1': float('nan'), 'error': str(e)}
        
        # Test 2: Sensor S2 perturbation (add small offset)
        train_s2 = train.copy()
        train_s2['Sensor_S2'] = train_s2['Sensor_S2'] + np.random.normal(0, noise_level * 10, len(train_s2))
        
        try:
            _, cv_report_s2, _, _ = cross_validate_anomaly_detector(
                train_s2, n_splits=n_splits, classifier_builder=None,
                use_extended_pairs=True, tune_iso=tune_iso)
            trial_noise_results['s2_perturbation'] = {
                'f1': cv_report_s2['f1'],
                'recall': cv_report_s2['recall'],
            }
        except Exception as e:
            trial_noise_results['s2_perturbation'] = {'f1': float('nan'), 'error': str(e)}
        
        # Test 3: Missing feature values (randomly set some to NaN, then impute)
        train_missing = train.copy()
        n_missing = max(1, int(len(train_missing) * 0.05))
        missing_cols = np.random.choice(FEATURE_COLS, size=min(n_missing, len(FEATURE_COLS)), replace=False)
        for col in missing_cols:
            idx = np.random.choice(len(train_missing), size=1, replace=False)[0]
            train_missing.loc[idx, col] = np.nan
        
        # Impute missing values
        from preprocessing import build_feature_pipeline
        pipeline = build_feature_pipeline(n_neighbors=7)
        train_missing_imp = pipeline.fit_transform(train_missing[FEATURE_COLS])
        train_missing[FEATURE_COLS] = train_missing_imp
        
        try:
            _, cv_report_missing, _, _ = cross_validate_anomaly_detector(
                train_missing, n_splits=n_splits, classifier_builder=None,
                use_extended_pairs=True, tune_iso=tune_iso)
            trial_noise_results['missing_values'] = {
                'f1': cv_report_missing['f1'],
                'recall': cv_report_missing['recall'],
            }
        except Exception as e:
            trial_noise_results['missing_values'] = {'f1': float('nan'), 'error': str(e)}
        
        # Test 4: Extreme but plausible values (set Sensor_S1 to max plausible value)
        train_extreme = train.copy()
        train_extreme['Sensor_S1'] = 199.9  # Max plausible value
        
        try:
            _, cv_report_extreme, _, _ = cross_validate_anomaly_detector(
                train_extreme, n_splits=n_splits, classifier_builder=None,
                use_extended_pairs=True, tune_iso=tune_iso)
            trial_noise_results['extreme_values'] = {
                'f1': cv_report_extreme['f1'],
                'recall': cv_report_extreme['recall'],
            }
        except Exception as e:
            trial_noise_results['extreme_values'] = {'f1': float('nan'), 'error': str(e)}
        
        # Test 5: Duplicate records (add exact duplicate rows)
        n_dup = max(1, int(len(train_extreme) * 0.02))
        dup_indices = np.random.choice(len(train_extreme), size=n_dup, replace=False)
        dup_rows = train_extreme.iloc[dup_indices].copy()
        train_dup = pd.concat([train_extreme, dup_rows], ignore_index=True)
        
        # Need to re-add Is_Duplicate_Feature_Row
        from preprocessing import flag_exact_duplicate_features
        train_dup['Is_Duplicate_Feature_Row'] = flag_exact_duplicate_features(train_dup)
        
        try:
            _, cv_report_dup, _, _ = cross_validate_anomaly_detector(
                train_dup, n_splits=n_splits, classifier_builder=None,
                use_extended_pairs=True, tune_iso=tune_iso)
            trial_noise_results['duplicates'] = {
                'f1': cv_report_dup['f1'],
                'recall': cv_report_dup['recall'],
            }
        except Exception as e:
            trial_noise_results['duplicates'] = {'f1': float('nan'), 'error': str(e)}
        
        # Test 6: Borderline anomaly records (records near the threshold)
        # These are records with classifier probability near the chosen threshold
        try:
            from anomaly_detection import build_anomaly_scores
            train_out, test_out, cv_report_borderline, _ = build_anomaly_scores(
                train, test, n_splits=n_splits, auto_select_classifier=False,
                use_extended_pairs=True, tune_iso=tune_iso)
            trial_noise_results['borderline'] = {
                'f1': cv_report_borderline['f1'],
                'recall': cv_report_borderline['recall'],
                'n_invalid_test': int(test_out['Predicted_Invalid'].sum()),
            }
        except Exception as e:
            trial_noise_results['borderline'] = {'f1': float('nan'), 'error': str(e)}
        
        results.append({
            'trial': trial + 1,
            **trial_noise_results,
        })
    
    return pd.DataFrame(results)

def fit_final_and_predict_test(train: pd.DataFrame, test: pd.DataFrame, chosen_threshold: float,
                                classifier_builder=None, use_extended_pairs: bool = True,
                                tune_iso: bool = True):
    result = _fit_and_score(train, test, classifier_builder=classifier_builder,
                             use_extended_pairs=use_extended_pairs, tune_iso=tune_iso)
    (test_feat, test_proba, override_z, fitted_lines, scaler, iso, clf,
     clf_cols, resid_cols, if_params) = result

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

    meta = {
        "override_z_threshold": float(override_z),
        "resid_cols": resid_cols,
        "isolation_forest_params": if_params,
    }
    return train_feat, test_feat, meta

def build_anomaly_scores(train: pd.DataFrame, test: pd.DataFrame, n_splits=N_SPLITS,
                          auto_select_classifier: bool = True,
                          use_extended_pairs: bool = True, tune_iso: bool = True,
                          fixed_threshold: float = 0.41,
                          classifier_builder: Callable = None):
    clf_comparison = None
    selected_classifier = "GradientBoosting"
    if auto_select_classifier and classifier_builder is None:
        clf_comparison = compare_classifiers(train, n_splits=n_splits,
                                              use_extended_pairs=use_extended_pairs,
                                              tune_iso=tune_iso)
        selected_classifier = clf_comparison.iloc[0]["classifier"]
        classifier_builder = _candidate_classifier_builders()[selected_classifier]
    elif classifier_builder is not None:
        selected_classifier = "Custom"

    oof_proba, cv_report, chosen_threshold, oof_signals = cross_validate_anomaly_detector(
        train, n_splits=n_splits, classifier_builder=classifier_builder,
        use_extended_pairs=use_extended_pairs, tune_iso=tune_iso,
        fixed_threshold=fixed_threshold)

    train_out, test_out, meta = fit_final_and_predict_test(
        train, test, chosen_threshold, classifier_builder=classifier_builder,
        use_extended_pairs=use_extended_pairs, tune_iso=tune_iso)

    cv_report["override_z_threshold"] = meta["override_z_threshold"]
    cv_report["selected_classifier"] = selected_classifier
    cv_report["isolation_forest_params"] = meta["isolation_forest_params"]
    return train_out, test_out, cv_report, clf_comparison

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from preprocessing import load_and_clean, verify_duplicate_logic

    tr, te = load_and_clean("../data/training_data.csv", "../data/test_data.csv")
    print("Duplicate-logic verification (train):")
    for k, v in verify_duplicate_logic(tr).items():
        print(f"  {k}: {v}")


    print("\n" + "=" * 70)
    print("BEFORE (previous design): 3 fixed pairs, fixed Isolation Forest, "
          "GradientBoostingClassifier")
    print("=" * 70)
    gb_builder = lambda: GradientBoostingClassifier(n_estimators=GB_N_ESTIMATORS, random_state=RANDOM_STATE)
    _, baseline_report, baseline_threshold, _ = cross_validate_anomaly_detector(
        tr, classifier_builder=gb_builder, use_extended_pairs=False, tune_iso=False)
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {k:>10s}: {baseline_report[k]:.4f}")
    print(f"  chosen_threshold: {baseline_threshold}")


    print("\n" + "=" * 70)
    print("Classifier comparison (data-driven consistency pairs + tuned Isolation Forest)")
    print("=" * 70)
    clf_comparison = compare_classifiers(tr)
    print(clf_comparison.to_string(index=False))
    best_name = clf_comparison.iloc[0]["classifier"]
    best_builder = _candidate_classifier_builders()[best_name]


    print("\n" + "=" * 70)
    print(f"AFTER (improved design, selected classifier = {best_name})")
    print("=" * 70)
    _, improved_report, improved_threshold, _ = cross_validate_anomaly_detector(
        tr, classifier_builder=best_builder, use_extended_pairs=True, tune_iso=True)
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {k:>10s}: {improved_report[k]:.4f}")
    print(f"  chosen_threshold: {improved_threshold}")


    print("\n" + "=" * 70)
    print("Ablation study (OOF, improved features + selected classifier)")
    print("=" * 70)
    ablation = ablation_study(tr, classifier_builder=best_builder)
    print(ablation.to_string())


    print("\n" + "=" * 70)
    print("Final fit on full training set -> test set predictions")
    print("=" * 70)
    tr_out, te_out, meta = fit_final_and_predict_test(
        tr, te, improved_threshold, classifier_builder=best_builder)
    print("Isolation Forest params selected on full training data:", meta["isolation_forest_params"])
    print("Predicted invalid count (train, full-fit):", int(tr_out["Predicted_Invalid"].sum()))
    print("Predicted invalid count (test):", int(te_out["Predicted_Invalid"].sum()))
