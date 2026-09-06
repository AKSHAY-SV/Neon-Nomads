# Methodology — CPRI PowerNext-AI Screening Round (Corrected)

**Team:** CPRI_ClaudeSquad

This document describes the pipeline as it actually runs in `src/main.py`
after fixing the cross-validation leakage and train/test-consistency issues
present in the first draft. All numbers below are taken directly from the
latest run's terminal output / `summary.json`, not assumed.

## 1. Preprocessing

Physically impossible readings (negative values, or values outside a loose
domain-plausibility band — e.g. temperature outside [-20, 80]°C) are set to
NaN. Missing sensor values are then imputed with a `KNNImputer`
(`n_neighbors=7`, distance-weighted) wrapped in an sklearn `Pipeline`. This
pipeline is **fit once on the training data only**; the test data is
transformed with that same fitted pipeline (never refit) — this was the
first correctness fix, since the original version fit a separate imputer
on test data, which is inconsistent even though it does not touch labels.

Duplicate-feature-vector rows (identical Voltage/Current/Ambient/Duration/
S1–S4 but a different Test_ID) are flagged using only the feature columns,
computed independently on each dataset — this uses no label information so
computing it per-dataset is not leakage. Programmatic verification on the
current training data: **12 duplicate pairs (24 rows), 24/24 (100%)
labelled Invalid**. The same structural check applied to test data finds
8 rows (4 pairs), flagged identically without ever referencing training
rows.

## 2. Important parameters / features

Correlation analysis (Valid rows only) and the final regression model's
feature importances agree: **Load_Current_A dominates** (~88% importance,
r≈0.88 with Reference_Parameter), followed by **Sensor_S2** (~8%) and
**Ambient_Temperature_C** (~4%). `Applied_Voltage_kV`, `Test_Duration_min`,
`Sensor_S1`, `Sensor_S3` and especially **Sensor_S4** contribute
negligibly once Load_Current and Sensor_S2 are known. All features are
still passed to the tree-based model rather than hand-pruned, since trees
are robust to low-importance inputs and this avoids a subjective
feature-removal call.

## 3. Reference_Parameter prediction

Five regressors (Ridge, RandomForest, ExtraTrees, GradientBoosting,
HistGradientBoosting — plus XGBoost/LightGBM/CatBoost automatically if
installed) are compared with 5-fold cross-validation on the
historically-Valid subset. **GradientBoosting was selected** by lowest CV
MAE: **MAE 0.536, RMSE 0.848, R² 0.994** — consistent with (in fact
slightly better than) the original draft's result, confirming the leakage
fixes elsewhere did not depend on any accidental advantage in the
regression stage (the regression stage was never a source of leakage,
since it always trained/validated only on historically-Valid rows with
plain K-fold CV over disjoint folds).

**Valid-only vs. all-rows training was verified, not assumed**: the same
GradientBoosting architecture was cross-validated on (a) Valid-only rows
and (b) all rows including historically-Invalid ones. Result: Valid-only
achieves MAE 0.536 vs. 0.884 for all-rows — confirming the original design
choice is empirically justified, not just a reasonable-sounding assumption.
No test data or test predictions were used in this comparison.

## 4. Anomaly / invalid-record detection

**This is where the main correctness fix applies.** In the original
version, the physical-consistency regression lines, the `StandardScaler`,
and the `IsolationForest` were fit once on the *entire* training set before
cross-validating only the classifier — meaning every fold's "held-out" rows
had already influenced the very features being classified. The corrected
version refits **all four** components (consistency lines, scaler,
Isolation Forest, classifier) **inside each of the 5 cross-validation
folds**, using only that fold's training rows; validation rows are only
ever transformed/scored, never used to fit anything.

Signals combined (unchanged in spirit from the original design):
1. **Physical-consistency residuals** — Sensor_S1/S3 vs Applied_Voltage,
   Sensor_S2 vs Load_Current, fit per-fold on that fold's Valid rows only.
2. **Duplicate-feature-row flag** (label-free, safe to compute globally).
3. **Isolation Forest** anomaly score, fit per-fold.
4. **Gradient Boosting classifier**, trained per-fold on the above features
   plus the raw parameters.

Genuinely out-of-fold performance against historical `Validity_Label`:
**accuracy 0.966, precision 0.917, recall 0.821, F1 0.866, ROC-AUC 0.911**
(confusion matrix `[[856,10],[24,110]]`, true rows = [Valid, Invalid]).
These numbers are very close to the original (non-leakage-free) run's
reported metrics, which indicates the previous leakage was not materially
inflating the headline scores — but they are now honestly obtained.

**Decision threshold**: rather than a fixed 0.5, the out-of-fold
probabilities from the CV above are swept over a grid (0.10–0.90) and the
threshold maximising **out-of-fold F1** is selected (**0.45** on the
current data) — this is tuned only against historical labels, never
against the test set, which has none.

**Hard-override rules** (near-certain data-quality issues, applied on top
of the classifier): a record is forced Invalid if it is a duplicate
feature-row, or if any physical-consistency residual z-score exceeds a
**data-driven threshold** — the maximum |z| ever observed among
historically-Valid rows, plus a small safety margin — rather than an
arbitrary fixed constant.

On the 350-record test set this pipeline flags **41 records (11.7%)** as
Invalid, close to training's 13.4% Invalid rate.

### Distinguishing genuine regime changes from sensor errors

A record that is unusual in raw magnitude but whose Sensor_S1/S2/S3
readings still track Applied_Voltage/Load_Current as physically expected
will have **small** consistency-residual z-scores and will not trigger the
hard-override rules; the multivariate Isolation Forest also only flags
unusual *combinations*, not unusual single values, and includes the
residual features so a physically-consistent-but-unusual point is not
penalised for being unusual on one raw axis alone. Only the learned
classifier — trained on the historical label distribution — can flag such
a point as Invalid, and only if the historical data contains a similar
pattern among labelled-Invalid rows. This matches the brief's requirement
that "an unusual value is not necessarily an invalid value."

## 5. Assumptions

- Reference_Parameter values attached to historically-Invalid rows may
  themselves be unreliable, so regression training uses Valid rows only
  (empirically justified, see §3).
- Physically impossible values are treated as missing, not as evidence of
  a genuine regime change.
- A record's validity is assessed using only information available at
  test time (its own features) — no test-set statistics are used to fit
  any transformer or model.

## 6. Digital-twin automation extension

To run this continuously rather than as a one-off batch job: (1) persist
the fitted preprocessing pipeline, consistency lines, scaler, Isolation
Forest, classifier and regressor as versioned artifacts (`joblib`) instead
of refitting on every invocation; (2) stream new test records through the
same `transform`/`predict` calls used here; (3) monitor the out-of-fold CV
metrics and the physical-consistency residual distribution over time —
a sustained shift signals either genuine equipment drift (retrain) or a
new sensor-fault mode (investigate); (4) surface `Predicted_Invalid`,
`clf_proba_invalid`, and the attention-score ranking through a dashboard so
engineers review borderline cases rather than trusting the automated label
blindly; (5) periodically retrain on newly engineer-verified records to
keep the historical-label-dependent components (classifier, threshold)
current.
