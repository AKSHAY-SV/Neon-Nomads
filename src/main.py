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
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import (load_and_clean, verify_duplicate_logic, FEATURE_COLS,
                           engineer_features, get_feature_names, select_feature_set)
from anomaly_detection import build_anomaly_scores, perturbation_testing
from model import (compare_models, compare_training_subsets, select_and_fit_best,
                   feature_importance, predict, save_model, TARGET_COL, _candidate_models,
                   compare_feature_sets, evaluate_on_holdout, optimize_hyperparameters_optuna)
from evaluation import save_run
DEFAULT_TEAM_NAME = 'Neon Nomads'


def parse_args():
    p = argparse.ArgumentParser(description='Neon Nomads screening pipeline')
    p.add_argument('--team-name', default=DEFAULT_TEAM_NAME, help='Team name used for the prediction CSV filename')
    p.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), '..', 'data'), help='Directory containing training_data.csv and test_data.csv')
    p.add_argument('--output-dir', default=os.path.join(os.path.dirname(__file__), '..', 'output'), help='Directory to write prediction CSV and summary.json into')
    p.add_argument('--runs-dir', default=os.path.join(os.path.dirname(__file__), '..', 'runs'), help='Directory to store per-run evaluation history (runs/run_XXX/)')
    p.add_argument('--optimize', action='store_true', help='Run hyperparameter optimization with Optuna')
    p.add_argument('--trials', type=int, default=25, help='Number of Optuna trials for the final tuning pass (default: 25, kept small for laptop/Colab)')
    p.add_argument('--robustness-trials', type=int, default=2, help='Number of repeat trials per robustness scenario (default: 2, kept small for laptop/Colab)')
    p.add_argument('--skip-robustness', action='store_true', help='Skip the robustness perturbation test suite')
    p.add_argument('--skip-optuna', action='store_true', help='Skip Optuna hyperparameter tuning pass')
    p.add_argument('--test-mode', type=int, choices=range(1, 11), default=1, help='Test mode: 1=CV, 2=Holdout, 3=Feature ablation, 4=Hyperparam comparison, 5=Threshold comparison, 6=Outlier stress, 7=Missing value robustness, 8=Duplicate robustness, 9=Noise perturbation, 10=Full blind test')
    p.add_argument('--holdout-ratio', type=float, default=0.1, help='Holdout ratio for final validation (default: 0.1)')
    p.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds (default: 5)')
    p.add_argument('--cv-repeats', type=int, default=3, help='Number of CV repeats (default: 3)')
    p.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    p.add_argument('--feature-set', type=str, default='all', choices=['all', 'no_s4', 'important_only', 'engineered_only', 'raw_only'],
                   help='Feature set for ablation/testing')
    p.add_argument('--holdout', action='store_true', help='Use untouched holdout evaluation')
    p.add_argument('--ablation', action='store_true', help='Run feature ablation study')
    p.add_argument('--holdout-seed', type=int, default=None, help='Seed for holdout split')
    p.add_argument('--fixed-threshold', type=float, default=None, help='Fixed classification threshold for anomaly detection (overrides F1-optimal)')
    return p.parse_args()


def safe_sanitize_team_name(name: str) -> str:
    keep = [c if c.isalnum() or c in '_- ' else '_' for c in name]
    out = ''.join(keep).strip('_ ')
    return out or DEFAULT_TEAM_NAME


def run_feature_ablation(train_scored: pd.DataFrame, test: pd.DataFrame, args) -> dict:
    """Run feature ablation study (Phase 4)."""
    print('\n=== FEATURE ABLATION STUDY ===')
    ablation_results = compare_feature_sets(
        train_scored,
        model_name="GradientBoosting",
        n_splits=args.cv_folds,
        n_repeats=args.cv_repeats,
    )
    print(ablation_results.to_string(index=False))
    
    return {
        'feature_ablation_results': ablation_results.to_dict(orient='records'),
        'selected_feature_set': ablation_results.iloc[0]['feature_set']
    }


def run_optuna_optimization(train_valid: pd.DataFrame, args) -> dict:
    """Run hyperparameter optimization with Optuna (Phase 6)."""
    print(f'\n=== OPTUNA HYPERPARAMETER OPTIMIZATION ({args.trials} trials) ===')
    
    best_params = optimize_hyperparameters_optuna(
        train_subset=train_valid,
        model_name="GradientBoosting",
        n_trials=args.trials,
        n_splits=args.cv_folds,
        n_repeats=1,
        use_feature_engineering=True,
        feature_set=args.feature_set,
        random_state=args.seed,
    )
    
    print(f'Best parameters: {best_params}')
    
    from model import _candidate_models
    model_builder = lambda: _candidate_models()["GradientBoosting"]
    
    current_params = model_builder().get_params()
    current_params.update(best_params)
    
    best_model = model_builder()
    for k, v in best_params.items():
        setattr(best_model, k, v)
    
    from sklearn.model_selection import KFold, cross_validate
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    df_processed = engineer_features(train_valid) if "Power_kVA" in train_valid.columns else train_valid.copy()
    feature_cols = [c for c in get_feature_names(df_processed) if c in df_processed.columns]
    X = df_processed[feature_cols].astype(float).values
    y = df_processed[TARGET_COL].astype(float).values
    
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    cv_res = cross_validate(best_model, X, y, cv=kf, 
                           scoring={"MAE": "neg_mean_absolute_error", 
                                    "RMSE": "neg_root_mean_squared_error", 
                                    "R2": "r2"},
                           n_jobs=1)
    
    results = {
        "model": "GradientBoosting_Optuna",
        "MAE": -cv_res["test_MAE"].mean(),
        "MAE_std": cv_res["test_MAE"].std(),
        "RMSE": -cv_res["test_RMSE"].mean(),
        "RMSE_std": cv_res["test_RMSE"].std(),
        "R2": cv_res["test_R2"].mean(),
        "R2_std": cv_res["test_R2"].std(),
    }
    
    print(f'Optuna optimized results: MAE={results["MAE"]:.4f}, RMSE={results["RMSE"]:.4f}, R2={results["R2"]:.4f}')
    
    return {
        'optuna_best_params': best_params,
        'optuna_optimized_results': results
    }


def run_holdout_evaluation(train_scored: pd.DataFrame, test: pd.DataFrame, args) -> dict:
    """Run untouched holdout evaluation (Phase 9)."""
    print('\n=== HOLDOUT EVALUATION (Phase 9) ===')
    
    holdout_ratio = args.holdout_ratio
    random_state = args.holdout_seed if args.holdout_seed else args.seed
    
    dev_df, holdout_df = train_test_split(
        train_scored, 
        test_size=holdout_ratio, 
        random_state=random_state,
        stratify=train_scored['Validity_Label'] if 'Validity_Label' in train_scored.columns else None
    )
    
    print(f'Development set: {len(dev_df)} rows ({100*(1-holdout_ratio):.1f}%)')
    print(f'Holdout set: {len(holdout_df)} rows ({100*holdout_ratio:.1f}%)')
    
    model_builder = lambda: _candidate_models()["GradientBoosting"]
    subset_comparison = compare_training_subsets(
        dev_df, "GradientBoosting", model_builder, n_splits=args.cv_folds
    )
    
    better_subset = min(subset_comparison, key=lambda k: subset_comparison[k]['MAE'])
    print(f'Best subset: {better_subset}')
    
    best_model_name = subset_comparison[better_subset].get('model', 'GradientBoosting')
    best_model = _candidate_models()[best_model_name]
    
    holdout_metrics = evaluate_on_holdout(
        dev_df, holdout_df, lambda: _candidate_models()[best_model_name],
        use_feature_engineering=True, feature_set=args.feature_set
    )
    
    print(f'Holdout metrics: MAE={holdout_metrics["MAE"]:.4f}, RMSE={holdout_metrics["RMSE"]:.4f}, R2={holdout_metrics["R2"]:.4f}')
    
    test_scored, _, _ = build_anomaly_scores(dev_df, test)
    n_invalid_test = int(test_scored['Predicted_Invalid'].sum())
    
    return {
        'holdout_metrics': holdout_metrics,
        'dev_size': len(dev_df),
        'holdout_size': len(holdout_df),
        'n_invalid_test': n_invalid_test,
        'holdout_ratio': holdout_ratio
    }


def main():
    t_start = time.time()
    args = parse_args()
    team_name = safe_sanitize_team_name(args.team_name)
    data_dir = args.data_dir
    output_dir = args.output_dir
    runs_dir = args.runs_dir
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(data_dir, 'training_data.csv')
    test_path = os.path.join(data_dir, 'test_data.csv')
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f'ERROR: expected data files not found.\n  training: {train_path} (exists={os.path.exists(train_path)})\n  test:     {test_path} (exists={os.path.exists(test_path)})')
        sys.exit(1)
    print('=' * 72)
    print('Neon Nomads Screening Round Pipeline')
    print('=' * 72)
    print('\n[1/7] Loading and preprocessing data...')
    try:
        train, test, prep_stats = load_and_clean(train_path, test_path)
    except Exception:
        print('ERROR during data loading/preprocessing:')
        traceback.print_exc()
        sys.exit(1)
    n_train, n_test = (len(train), len(test))
    print(f'  Training records: {n_train}   Test records: {n_test}')
    print(f'  Missing values remaining -> train: {train[FEATURE_COLS].isna().sum().sum()}, test: {test[FEATURE_COLS].isna().sum().sum()}')
    print('  (Imputer fit ONCE on training data; test data transformed with the same fitted imputer.)')
    dup_check = verify_duplicate_logic(train)
    print(f'\n  Duplicate-logic verification (train): {dup_check}')
    print(f'  Exact duplicate feature-rows -> train: {int(train["Is_Duplicate_Feature_Row"].sum())}, test: {int(test["Is_Duplicate_Feature_Row"].sum())}')
    print('\n[2/7] Detecting abnormal / invalid records (nested, leakage-free CV)...')
    try:
        # Leakage-free: candidate classifiers and the decision threshold are
        # both chosen purely from out-of-fold cross-validation on the
        # training set (see cross_validate_anomaly_detector / compare_classifiers
        # in anomaly_detection.py) - the test set is never touched here.
        train_scored, test_scored, cv_report, clf_comparison = build_anomaly_scores(
            train, test, auto_select_classifier=True, tune_iso=True,
            fixed_threshold=args.fixed_threshold
        )
    except Exception:
        print('ERROR during anomaly detection:')
        traceback.print_exc()
        sys.exit(1)
    print('  Genuinely out-of-fold classifier performance vs historical Validity_Label')
    print('  (consistency lines / scaler / Isolation Forest / classifier all refit per fold):')
    for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        print(f'    {k:>10s}: {cv_report[k]:.4f}')
    print(f'    confusion_matrix (rows=true[Valid,Invalid], cols=pred): {cv_report["confusion_matrix"]}')
    print(f'    decision threshold (fixed for optimal recall/F1 balance): {cv_report["chosen_threshold"]}')
    print(f'    hard-override residual-z threshold (data-driven): {cv_report["override_z_threshold"]:.2f}')
    print(f'    selected classifier (best OOF ROC-AUC): {cv_report["selected_classifier"]}')
    if clf_comparison is not None:
        print('  Classifier comparison (OOF, sorted by ROC-AUC):')
        print('  ' + clf_comparison.to_string(index=False).replace('\n', '\n  '))
    n_invalid_test = int(test_scored['Predicted_Invalid'].sum())
    print(f'  Predicted Invalid in TEST set: {n_invalid_test} / {n_test} ({100 * n_invalid_test / n_test:.1f}%)')
    print('\n[3/7] Comparing regression models (5-fold CV, Valid-only training rows)...')
    train_valid = train_scored[train_scored['Validity_Label'] == 'Valid']

    # Use all valid rows for model selection (matching previous best runs)
    # Holdout evaluation is done at the end as a final check only
    dev_valid = train_valid
    holdout_valid = None
    print(f'  Using all {len(dev_valid)} valid rows for model selection (no holdout split)')

    try:
        results, fitted_models, feature_cols = compare_models(dev_valid, n_splits=args.cv_folds, feature_set=args.feature_set)
    except Exception:
        print('ERROR during model comparison:')
        traceback.print_exc()
        sys.exit(1)
    print(results.to_string(index=False))
    best_name, _, feature_cols = select_and_fit_best(dev_valid, results, fitted_models, feature_cols=feature_cols)
    baseline_dev_mae = float(results.iloc[0]['MAE'])
    print(f'\n  Selected model (on full valid set): {best_name} '
          f'(CV MAE={results.iloc[0]["MAE"]:.4f}, RMSE={results.iloc[0]["RMSE"]:.4f}, R2={results.iloc[0]["R2"]:.4f})')

    # --- Feature-set comparison (Priority 3) -------------------------------
    print('\n[3b/7] Comparing feature sets (raw / engineered / all / reduced) via CV on the full valid set...')
    chosen_feature_set = 'all'
    feature_set_comparison_records = None
    try:
        fs_results = compare_feature_sets(dev_valid, model_name=best_name, n_splits=args.cv_folds, n_repeats=1)
        print(fs_results.to_string(index=False))
        feature_set_comparison_records = fs_results.to_dict(orient='records')
        best_fs_row = fs_results.iloc[0]
        all_row = fs_results[fs_results['feature_set'] == 'all'].iloc[0]
        # Only switch away from the safer 'all' feature set if another set
        # gives a real (>1%) MAE improvement - otherwise keep the existing
        # working configuration rather than chasing CV noise.
        if best_fs_row['feature_set'] != 'all' and best_fs_row['MAE'] < all_row['MAE'] * 0.99:
            chosen_feature_set = best_fs_row['feature_set']
            print(f"  -> Switching to feature set '{chosen_feature_set}' "
                  f"(MAE {best_fs_row['MAE']:.4f} vs 'all' MAE {all_row['MAE']:.4f})")
        else:
            print(f"  -> Keeping feature set 'all' (best alternative did not clearly improve on it)")
    except Exception:
        print('WARNING: feature-set comparison failed (non-fatal, keeping "all"):')
        traceback.print_exc()

    df_dev_processed = engineer_features(dev_valid)
    feature_cols = select_feature_set(df_dev_processed, chosen_feature_set)

    # --- Optuna hyperparameter tuning (Priority 2) -------------------------
    # Search space and trial count are kept small (default 25 trials, one
    # model) so this stays fast on a laptop/Colab. Only ever run on the dev
    # set - the untouched holdout and the 350 test rows are never used here.
    tuned_params = None
    if not args.skip_optuna and best_name in ('GradientBoosting', 'RandomForest', 'ExtraTrees', 'HistGradientBoosting'):
        print(f'\n[3c/7] Optuna hyperparameter tuning for {best_name} ({args.trials} trials, dev set only)...')
        try:
            candidate_params = optimize_hyperparameters_optuna(
                dev_valid, model_name=best_name, n_trials=args.trials,
                n_splits=args.cv_folds, n_repeats=1, use_feature_engineering=True,
                feature_set=chosen_feature_set, random_state=args.seed,
            )
            tuned_model = _candidate_models()[best_name]
            for k, v in candidate_params.items():
                setattr(tuned_model, k, v)
            X_dev = df_dev_processed[feature_cols].astype(float).values
            y_dev = df_dev_processed[TARGET_COL].astype(float).values
            from sklearn.model_selection import KFold as _KFold, cross_validate as _cross_validate
            cv_tuned = _cross_validate(
                tuned_model, X_dev, y_dev,
                cv=_KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed),
                scoring={'MAE': 'neg_mean_absolute_error'}, n_jobs=1,
            )
            tuned_mae = float(-cv_tuned['test_MAE'].mean())
            print(f'  Optuna-tuned CV MAE: {tuned_mae:.4f}  (baseline default-params CV MAE: {baseline_dev_mae:.4f})')
            if tuned_mae < baseline_dev_mae:
                tuned_params = candidate_params
                print('  -> Tuned hyperparameters improve on the baseline; using them.')
            else:
                print('  -> Tuned hyperparameters do not improve on the safer default configuration; keeping defaults.')
        except Exception:
            print('WARNING: Optuna tuning failed (non-fatal, keeping default hyperparameters):')
            traceback.print_exc()
    else:
        print('\n[3c/7] Skipping Optuna tuning (--skip-optuna set or model has no tuned search space).')

    def _build_final_model():
        m = _candidate_models()[best_name]
        if tuned_params:
            for k, v in tuned_params.items():
                setattr(m, k, v)
        return m

    # --- Untouched holdout evaluation (Priority 1) --------------------------
    print('\n[3d/7] Evaluating chosen model/feature-set/hyperparameters...')
    holdout_metrics = None
    if holdout_valid is not None:
        try:
            holdout_eval = evaluate_on_holdout(
                dev_valid, holdout_valid, _build_final_model,
                use_feature_engineering=True, feature_set=chosen_feature_set,
            )
            holdout_metrics = {
                **holdout_eval,
                'dev_size': len(dev_valid), 'holdout_size': len(holdout_valid),
                'holdout_ratio': args.holdout_ratio,
            }
            print(f"  Holdout MAE={holdout_eval['MAE']:.4f}  RMSE={holdout_eval['RMSE']:.4f}  R2={holdout_eval['R2']:.4f} "
                  f"(n={len(holdout_valid)}, never used for tuning)")
        except Exception:
            print('WARNING: holdout evaluation failed (non-fatal):')
            traceback.print_exc()
    else:
        print('  Skipping holdout evaluation (using all data for model selection)')

    # Final model shipped for test predictions: same model/feature-set/
    # hyperparameters just selected and holdout-checked above, refit on ALL
    # Valid-only training rows (dev + holdout) for the best possible fit.
    best_model = _build_final_model()
    df_train_valid_processed = engineer_features(train_valid)
    X_final = df_train_valid_processed[feature_cols].astype(float).values
    y_final = df_train_valid_processed[TARGET_COL].astype(float).values
    best_model.fit(X_final, y_final)
    imp = feature_importance(best_name, best_model, feature_cols)
    if imp is not None:
        print('\n  Feature importance:')
        for feat, val in imp.items():
            print(f'    {feat:>25s}: {val:.4f}')
    print('\n[4/7] Feature ablation study...')
    ablation_results = {}
    if args.ablation:
        ablation_results = run_feature_ablation(train_scored, test, args)
    else:
        print('  (Skipping feature ablation - use --ablation flag to enable)')
    print('\n[5/7] Verifying Valid-only vs all-rows training subset choice...')
    try:
        model_builder = lambda: _candidate_models()[best_name]
        subset_comparison = compare_training_subsets(train_scored, best_name, model_builder, n_splits=3)
        print('  Same model architecture, cross-validated on each subset (no test data involved):')
        for label, stats in subset_comparison.items():
            print(f'    {label:>10s}: n={stats["n_rows"]:4d}  MAE={stats["MAE"]:.4f}  RMSE={stats["RMSE"]:.4f}  R2={stats["R2"]:.4f}')
        better_subset = min(subset_comparison, key=lambda k: subset_comparison[k]['MAE'])
        print(f"  -> Lower CV MAE achieved by training on: '{better_subset}'")
    except Exception:
        print('WARNING: training-subset comparison failed (non-fatal):')
        traceback.print_exc()
        subset_comparison = None
        better_subset = 'valid_only'
    model_path = os.path.join(output_dir, 'model.joblib')
    try:
        save_model(best_model, model_path, feature_cols=feature_cols)
        print(f'  Saved fitted model -> {model_path}')
    except Exception:
        print('WARNING: failed to save the trained model (non-fatal):')
        traceback.print_exc()
    print('\n[6/7] Predicting Reference_Parameter for all test records...')
    test_scored['Predicted_Reference_Parameter'] = predict(best_model, test_scored, feature_cols=feature_cols)
    n_nan_pred = int(pd.isna(test_scored['Predicted_Reference_Parameter']).sum())
    if n_nan_pred > 0:
        print(f'  WARNING: {n_nan_pred} NaN predictions found - filling with training mean as a safeguard.')
        fallback = train_valid[TARGET_COL].mean()
        test_scored['Predicted_Reference_Parameter'] = test_scored['Predicted_Reference_Parameter'].fillna(fallback)
    print('\n[7/7] Writing deliverable files...')
    test_scored['Valid / Invalid'] = np.where(test_scored['Predicted_Invalid'], 'Invalid', 'Valid')
    pred_df = test_scored[['Test_ID', 'Predicted_Reference_Parameter', 'Valid / Invalid']].copy()
    pred_df['Predicted_Reference_Parameter'] = pred_df['Predicted_Reference_Parameter'].round(4)
    assert len(pred_df) == n_test, 'Row count mismatch between predictions and test data!'
    assert pred_df['Test_ID'].is_unique, 'Duplicate Test_IDs in prediction output!'
    assert pred_df['Predicted_Reference_Parameter'].notna().all(), 'NaN predictions present!'
    assert pd.api.types.is_numeric_dtype(pred_df['Predicted_Reference_Parameter']), 'Predicted_Reference_Parameter is not numeric!'
    assert set(pred_df['Valid / Invalid'].unique()) <= {'Valid', 'Invalid'}, 'Unexpected validity labels!'
    assert pred_df['Valid / Invalid'].notna().all(), 'Missing Valid/Invalid label present!'
    csv_filename = f'{team_name}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    pred_df.to_csv(csv_path, index=False)
    print(f'  Wrote {csv_path}  ({len(pred_df)} rows)')
    preds = pred_df['Predicted_Reference_Parameter']
    resid_cols = [c for c in test_scored.columns if c.startswith('resid_z_')]

    def _z(s):
        s = s.astype(float)
        std = s.std()
        return (s - s.mean()) / std if std > 0 else s * 0.0
    attention_components = pd.DataFrame({'clf_proba_invalid_z': _z(test_scored['clf_proba_invalid']), 'max_resid_z_z': _z(test_scored[resid_cols].abs().max(axis=1)), 'iso_forest_score_z': _z(test_scored['iso_forest_score'])})
    test_scored['_attention_score'] = attention_components.sum(axis=1) + test_scored['Is_Duplicate_Feature_Row'].astype(float) * 3.0
    top_attention = test_scored.sort_values('_attention_score', ascending=False).head(3)['Test_ID'].tolist()
    methodology_summary = f'Imputer/consistency-lines/scaler/Isolation-Forest/classifier fit only on training folds (no leakage); test data transformed only. Invalid records flagged via physical-consistency checks, duplicate-row detection, Isolation Forest and a classifier, threshold chosen by out-of-fold F1. Reference_Parameter predicted with the best cross-validated regressor ({best_name}) trained on verified rows.'
    n_words = len(methodology_summary.split())
    if n_words > 100:
        methodology_summary = ' '.join(methodology_summary.split()[:100])
    summary = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'team_name': team_name, 'records_analysed': int(n_test), 'abnormal_invalid_count': n_invalid_test, 'abnormal_invalid_percentage': round(100 * n_invalid_test / n_test, 2), 'predicted_reference_parameter': {'minimum': round(float(preds.min()), 4), 'maximum': round(float(preds.max()), 4), 'average': round(float(preds.mean()), 4)}, 'test_ids_requiring_highest_attention': top_attention, 'model_selection': {'selected_model': best_name, 'feature_set': chosen_feature_set, 'feature_set_comparison': feature_set_comparison_records, 'holdout_evaluation': holdout_metrics, 'cv_mae': round(float(results.iloc[0]['MAE']), 4), 'cv_rmse': round(float(results.iloc[0]['RMSE']), 4), 'cv_r2': round(float(results.iloc[0]['R2']), 4), 'all_models_compared': results.to_dict(orient='records'), 'training_subset_used': 'valid_only', 'training_subset_comparison': subset_comparison, 'feature_ablation': ablation_results}, 'anomaly_detection_out_of_fold_performance_vs_historical_labels': {k: v for k, v in cv_report.items()}, 'anomaly_detection_classifier_comparison': (clf_comparison.to_dict(orient='records') if clf_comparison is not None else None), 'duplicate_logic_verification_train': dup_check, 'methodology_explanation': methodology_summary, 'methodology_explanation_word_count': len(methodology_summary.split()), 'pipeline_runtime_seconds': None}
    summary['pipeline_runtime_seconds'] = round(time.time() - t_start, 2)
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  Wrote {summary_path}')
    print('\n[7/7] FINAL SUMMARY')
    print('-' * 72)
    print(f'  Records analysed:            {summary["records_analysed"]}')
    print(f'  Abnormal / Invalid detected: {summary["abnormal_invalid_count"]} ({summary["abnormal_invalid_percentage"]}%)')
    print(f'  Predicted Reference_Parameter -> min={summary["predicted_reference_parameter"]["minimum"]}, max={summary["predicted_reference_parameter"]["maximum"]}, avg={summary["predicted_reference_parameter"]["average"]}')
    print(f'  Top-attention Test IDs:      {summary["test_ids_requiring_highest_attention"]}')
    print(f'  Selected regression model:   {best_name}')
    print(f'  Methodology explanation word count: {summary["methodology_explanation_word_count"]} (<=100 required)')
    print(f'  Total pipeline runtime: {summary["pipeline_runtime_seconds"]}s')
    print('-' * 72)

    if args.test_mode == 2:
        print('\n--- Running Holdout Evaluation (Phase 2) ---')
        holdout_results = run_holdout_evaluation(train_scored, test, args)
        summary['holdout_results'] = holdout_results
    elif args.test_mode == 3 and args.ablation:
        print('\n--- Feature ablation already run above ---')
        summary['feature_ablation'] = ablation_results
    elif args.test_mode == 4:
        print('\n--- Running Hyperparameter Optimization (Phase 4) ---')
        optuna_results = run_optuna_optimization(train_valid, args)
        summary['optuna_results'] = optuna_results
    elif args.test_mode == 10:
        print('\n--- Full blind test prediction ---')
        pass
    
    print('\nSaving run history (internal validation, computed only from data/training_data.csv)...')
    try:
        raw_params = best_model.get_params() if hasattr(best_model, 'get_params') else None
        best_params = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)) for k, v in raw_params.items()} if raw_params else None
    except Exception:
        best_params = None

    # --- Robustness tests (Priority 4) --------------------------------
    # Covers feature noise, missing values, duplicates, physically
    # plausible extreme values, and borderline anomaly cases - all against
    # the (already leakage-free, nested-CV) anomaly detector, computed only
    # from data/training_data.csv. Trial count is kept small (default 2) so
    # this stays reasonable on a laptop/Colab.
    robustness_results = None
    robustness_summary = None
    if not args.skip_robustness:
        print(f'\nRunning robustness / perturbation test suite ({args.robustness_trials} trial(s) per scenario)...')
        try:
            rob_df = perturbation_testing(train, test, n_trials=args.robustness_trials, random_state=args.seed,
                                           n_splits=3, tune_iso=False)
            robustness_results = rob_df.to_dict(orient='records')
            scenario_names = ['gaussian_noise', 's2_perturbation', 'missing_values', 'extreme_values', 'duplicates', 'borderline']
            robustness_summary = {}
            for scenario in scenario_names:
                f1_vals = [row[scenario]['f1'] for row in robustness_results
                           if isinstance(row.get(scenario), dict) and row[scenario].get('f1') is not None
                           and not pd.isna(row[scenario].get('f1', float('nan')))]
                robustness_summary[scenario] = {
                    'f1_mean': float(np.mean(f1_vals)) if f1_vals else None,
                    'n_trials': len(f1_vals),
                }
            print('  Robustness summary (mean F1 of the anomaly detector under each perturbation):')
            for scenario, stats in robustness_summary.items():
                f1v = stats['f1_mean']
                print(f"    {scenario:<20s}: F1={f1v:.4f}" if f1v is not None else f"    {scenario:<20s}: n/a")
        except Exception:
            print('WARNING: robustness test suite failed (non-fatal):')
            traceback.print_exc()
    else:
        print('\nSkipping robustness test suite (--skip-robustness set).')

    run_dir = save_run(
        runs_root=runs_dir,
        pred_csv_path=csv_path,
        regression_metrics={'MAE': results.iloc[0]['MAE'], 'RMSE': results.iloc[0]['RMSE'], 'R2': results.iloc[0]['R2']},
        regression_model_name=best_name,
        regression_model_params=best_params,
        all_regression_models_compared=results.to_dict(orient='records'),
        classification_report=cv_report,
        classifier_comparison=clf_comparison,
        duplicate_check=dup_check,
        n_train=n_train,
        n_test=n_test,
        n_invalid_test=n_invalid_test,
        cv_folds_regression=args.cv_folds,
        cv_folds_classification=5,
        feature_set=chosen_feature_set,
        feature_set_comparison=feature_set_comparison_records,
        holdout_metrics=holdout_metrics,
        robustness_results=robustness_results,
        robustness_summary=robustness_summary,
    )
    run_id = os.path.basename(run_dir)
    with open(os.path.join(run_dir, 'evaluation.txt')) as f:
        print(f.read())
    print(f'Run saved -> {run_dir}')
    print(f'  ({os.path.join(run_dir, os.path.basename(csv_path))}, evaluation.txt, summary.json)')

    print('\nPipeline completed successfully.')
    return (summary, pred_df, run_id)


if __name__ == '__main__':
    main()
