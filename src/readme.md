# Source Code Overview

This directory contains the core source code for the CPRI PowerNext-AI screening solution. The implementation is organized as a modular machine-learning pipeline that preprocesses the test-bench data, predicts the `Reference_Parameter`, identifies potentially abnormal or invalid tests, and generates the required output files automatically.

## Source Files

### `main.py` — Main Pipeline Controller

The main entry point of the project. It coordinates the complete workflow from loading the training and test datasets to generating the final prediction CSV and automated summary.

The pipeline broadly performs:

* Loading the datasets
* Data preprocessing
* Training and selecting the Reference Parameter prediction model
* Training and applying the anomaly/validity detection system
* Generating predictions for the test dataset
* Identifying records requiring attention
* Creating the final CSV and summary output

Run the complete solution through this file.

### `preprocessing.py` — Data Preprocessing

Handles preparation of the raw CPRI test-bench data before it is passed to the machine-learning models.

The module deals with data-quality issues such as missing measurements, numerical features, and consistent transformation of training and test data. The preprocessing pipeline is designed so that transformations learned from the training data can be applied consistently to unseen test data.

Its main purpose is to convert the raw experimental measurements into reliable, model-ready features.

### `model.py` — Reference Parameter Prediction

Contains the regression modelling component used to predict the `Reference_Parameter`.

The model uses the available operating conditions and sensor measurements to learn the relationship between the recorded test-bench parameters and the verified reference temperature.

The implementation evaluates suitable regression models using validation data and selects the best-performing approach. The current baseline uses a Gradient Boosting regression model, which captures nonlinear relationships between the electrical operating conditions and sensor measurements.

The module is responsible for:

* Preparing prediction features
* Training candidate regression models
* Evaluating model performance
* Selecting the best model
* Training the final predictor
* Generating `Reference_Parameter` predictions for new tests

### `anomaly_detection.py` — Abnormal / Invalid Test Detection

Contains the logic for determining whether a test record should be considered `Valid` or `Invalid`.

Instead of treating every unusual measurement as an error, the system analyses multiple signals from the data, including sensor relationships, physical consistency, anomaly scores, and patterns learned from historically labelled `Validity_Label` records.

The implementation incorporates approaches such as:

* Physical-consistency analysis
* Residual-based anomaly signals
* Multivariate anomaly detection
* Isolation Forest
* Supervised classification
* Out-of-fold validation and threshold selection

The final result is a `Valid` or `Invalid` classification for each test record, along with information that can be used to identify tests requiring higher attention.

### `explore.py` — Dataset Exploration

Provides utilities for initially analysing and understanding the CPRI dataset.

It is primarily used during development and investigation rather than as the main production pipeline.

The exploration process examines:

* Dataset structure
* Available parameters
* Data types
* Missing values
* Duplicate records
* Numerical statistics
* Feature relationships
* Sensor behaviour
* Relationships between inputs and the `Reference_Parameter`
* Differences between valid and invalid historical tests

This analysis helps determine which features and modelling strategies are appropriate for the final solution.

## How the Source Files Work Together

The modules form a single automated pipeline:

```text
Training Data
     │
     ▼
preprocessing.py
     │
     ├───────────────┐
     ▼               ▼
model.py      anomaly_detection.py
     │               │
     ▼               ▼
Reference       Valid / Invalid
Prediction       Detection
     │               │
     └───────┬───────┘
             ▼
          main.py
             │
             ▼
        Test Dataset
             │
             ▼
       Final Predictions
             │
       ┌─────┴─────┐
       ▼           ▼
   TeamName.csv  summary.json
```

`explore.py` supports the development and analysis stage, while the other modules form the core automated prediction and anomaly-detection pipeline.

## Overall Purpose

The `src` directory implements the core intelligence of the hackathon solution:

**Understand the historical test data → prepare reliable features → learn the Reference Parameter → identify abnormal behaviour → process new tests automatically → generate the required submission outputs.**

