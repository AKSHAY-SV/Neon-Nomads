# CPRI PowerNext-AI — Screening Round Submission (Corrected)

**Team:** CPRI_ClaudeSquad
**Challenge:** The Black-Box Test Bench Challenge

This is the corrected version of the pipeline. See `methodology.md` for the
full list of fixes applied (cross-validation leakage in the anomaly
detector, train/test-inconsistent imputation, threshold tuning, etc.) and
the reasoning behind every remaining design choice.

## What this does

Given `training_data.csv` (has `Reference_Parameter` and `Validity_Label`)
and `test_data.csv` (does not), this pipeline automatically:

1. Cleans both datasets — fits impossible-value clipping + KNN imputation
   **once on training data** and transforms test data with the **same**
   fitted pipeline (no separate test-side fitting).
2. Detects abnormal / invalid test records using an ensemble of physical
   consistency checks, duplicate-row detection, Isolation Forest, and a
   Gradient Boosting classifier — evaluated with **genuinely out-of-fold**
   cross-validation (every fitted component is refit inside each fold) and
   a decision threshold chosen by out-of-fold F1, not a fixed 0.5.
3. Predicts `Reference_Parameter` for every test record using the best of
   five cross-validated regression models (Ridge, RandomForest, ExtraTrees,
   GradientBoosting, HistGradientBoosting — plus XGBoost/LightGBM/CatBoost
   automatically if installed).
4. Writes `<TeamName>.csv` and `summary.json`, and prints a full evaluation
   report to the terminal.

## Project structure

```
project/
├── data/
│   ├── training_data.csv
│   └── test_data.csv
├── src/
│   ├── explore.py             # standalone EDA script
│   ├── preprocessing.py       # cleaning / imputation (fit-on-train only) / duplicate flagging
│   ├── anomaly_detection.py   # Task 1: leakage-free nested-CV anomaly detection
│   ├── model.py                # Task 2: Reference_Parameter regression
│   └── main.py                 # orchestrates the full pipeline (run this)
├── output/
│   ├── CPRI_ClaudeSquad.csv
│   └── summary.json
├── requirements.txt
├── README.md
└── methodology.md
```

## How to run (from a clean environment)

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/main.py --team-name "CPRI_ClaudeSquad"
```

Run this from the project root (the folder containing `src/`, `data/`,
`requirements.txt`). `src/main.py` resolves `../data` and `../output`
relative to its own file location, so it also works if you `cd src` first:

```bash
cd src
python3 main.py --team-name "CPRI_ClaudeSquad"
```

Optional arguments:

```
--team-name   Name used for the output CSV filename (default: CPRI_PowerNext_Team)
--data-dir    Directory containing training_data.csv / test_data.csv (default: ../data)
--output-dir  Directory to write outputs into (default: ../output)
```

Typical runtime on a normal laptop: **under 25 seconds** end-to-end
(measured on this machine: ~18–23s), including both cross-validation
stages.

## Verified before submission

- ✅ Full pipeline runs end-to-end with a single command, no manual editing
  of individual records, no hard-coded Test_IDs anywhere in the source.
- ✅ Tested from a clean virtual environment using only `requirements.txt`.
- ✅ All five source files (`preprocessing.py`, `anomaly_detection.py`,
  `model.py`, `main.py`, `explore.py`) compile cleanly
  (`python -m py_compile`).
- ✅ Output CSV has exactly the required columns: `Test_ID`,
  `Predicted_Reference_Parameter`, `Valid / Invalid`.
- ✅ Output row count matches `test_data.csv` row count (350 rows),
  `Test_ID` values are unique.
- ✅ `Predicted_Reference_Parameter` is fully numeric, no NaNs (asserted
  programmatically before writing the CSV, and re-checked independently
  after).
- ✅ Every test record receives a Valid/Invalid classification.
- ✅ `summary.json` is valid JSON and contains all required fields; the
  methodology explanation is verified ≤100 words programmatically.
- ✅ Anomaly-detector cross-validation is genuinely out-of-fold: every
  fitted transformer/model is refit inside each fold on that fold's
  training rows only.
- ✅ Imputation is fit once on training data and only ever `.transform()`-ed
  on test data.

## Key results (latest run — see `output/summary.json` for the exact numbers)

Regression, 5-fold CV on historically-Valid rows:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **GradientBoosting (selected)** | **0.536** | **0.848** | **0.994** |
| HistGradientBoosting | 0.636–0.663 | 1.07–1.11 | 0.989–0.990 |
| ExtraTrees | 0.644 | 1.27 | 0.985 |
| RandomForest | 0.683 | 1.22 | 0.987 |
| Ridge | 3.361 | 3.963 | 0.863 |

Valid-only vs. all-rows training (same model, cross-validated on each):
Valid-only MAE 0.536 vs. all-rows MAE 0.884 — confirms Valid-only training
is the better choice on this data.

Anomaly/validity classifier, **genuinely out-of-fold** against historical
`Validity_Label`: **accuracy 0.966, precision 0.917, recall 0.821, F1
0.866, ROC-AUC 0.911**, decision threshold 0.45 (chosen by out-of-fold F1).

Test set: **41 / 350 records (11.7%)** flagged Invalid.

(Exact figures vary by a few thousandths between runs due to
BLAS/floating-point non-determinism in some sklearn internals; all
explicit random seeds are fixed, so this variation is not sampling noise
in the pipeline logic.)
