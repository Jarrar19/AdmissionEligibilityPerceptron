# System Design & Architecture Specification

**Project:** Perceptron-Based Admission Eligibility Prediction System (`admission-eligibility-perceptron`)  
**Specification Version:** 1.0  
**Status:** Implemented & Verified  

---

## 1. System Architecture Overview

The system is structured as two distinct, decoupled pipelines:
1. **Machine Learning Pipeline (Offline/Batch)**: Data Generation $\to$ Validation $\to$ Cleaning $\to$ EDA $\to$ Feature Selection $\to$ Stratified Split $\to$ Leakage-Safe Scaling $\to$ Perceptron Training $\to$ Model Evaluation $\to$ Joblib Persistence.
2. **Serving & Prediction Pipeline (Online/Interactive)**: User Profile $\to$ Bounded Input Validation $\to$ Pre-trained Pipeline Loading $\to$ Inference Calculation $\to$ UI Result Card & Explanations.

```
+-----------------------------------------------------------------------------------------------+
|                                      ML PIPELINE (OFFLINE)                                     |
+-----------------------------------------------------------------------------------------------+
|  data/raw/admission_data.csv (1000 records)                                                   |
|           |                                                                                   |
|           v                                                                                   |
|  src/data_loader.py (schema check, min records >= 500, zero nulls)                            |
|           |                                                                                   |
|           v                                                                                   |
|  src/preprocessing.py (clean duplicates, separate X/y, drop Student_ID, stratified split)     |
|           |                                                                                   |
|           v                                                                                   |
|  src/train_model.py (Pipeline[StandardScaler, Perceptron(max_iter=1000, tol=1e-3)])           |
|           |                                                                                   |
|           +-----------------------------------+-----------------------------------+           |
|           |                                                                       |           |
|           v                                                                       v           |
|  models/admission_perceptron_pipeline.joblib                   src/evaluate_model.py          |
|                                                                                   |           |
|                                                                                   v           |
|                                                                  outputs/reports/metrics.json |
|                                                                  outputs/figures/*.png        |
+-----------------------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------------------+
|                                  INFERENCE & WEB APPLICATION                                  |
+-----------------------------------------------------------------------------------------------+
|  app/app.py (Streamlit Web UI)                                                                |
|           |                                                                                   |
|           v                                                                                   |
|  src/predict.py (validate_student_input against domain bounds)                                |
|           |                                                                                   |
|           v                                                                                   |
|  models/admission_perceptron_pipeline.joblib (zero retraining)                                 |
|           |                                                                                   |
|           v                                                                                   |
|  Compute: z = sum(w_i * x_scaled_i) + b                                                        |
|           |                                                                                   |
|           +-----------------------------------+-----------------------------------+           |
|           |                                                                       |           |
|           v (z >= 0)                                                              v (z < 0)   |
|  Status: ELIGIBLE (Green Card)                                Status: NOT ELIGIBLE (Red Card) |
+-----------------------------------------------------------------------------------------------+
```

---

## 2. Requirements Traceability Matrix (§13)

This matrix maps every Functional Requirement (FR) from `docs/PROJECT_REQUIREMENTS.md` to its implementing source code module and the test case that verifies it:

| Requirement ID | Requirement Summary | Implementing Source File | Verification Mechanism / Test Case | Status |
| :--- | :--- | :--- | :--- | :--- |
| **FR-01** | Dataset generation, schema verification, $\ge 500$ rows | `scripts/generate_dataset.py`<br>`src/data_loader.py` | `tests/test_preprocessing.py::test_dataset_exists_and_loads`<br>`tests/test_prediction.py::test_case_1_dataset_validation` | **VERIFIED** |
| **FR-02** | Leakage-free preprocessing, drop identifier, stratified split | `src/preprocessing.py` | `tests/test_preprocessing.py::test_feature_target_separation_excludes_identifier`<br>`tests/test_preprocessing.py::test_stratified_split_preserves_class_ratio` | **VERIFIED** |
| **FR-03** | Exploratory Data Analysis, distributions, correlation plots | `scripts/run_eda.py`<br>`notebooks/01_eda.ipynb` | Execution of `run_eda.py` generating figures in `outputs/figures/` | **VERIFIED** |
| **FR-04** | Perceptron model training on training partition only | `src/train_model.py` | `tests/test_model.py::test_perceptron_weights_and_bias_dimensions` | **VERIFIED** |
| **FR-05** | Pipeline persistence to joblib format | `src/train_model.py` | `tests/test_model.py::test_model_pipeline_artifact_exists`<br>`tests/test_model.py::test_model_artifact_is_valid_pipeline` | **VERIFIED** |
| **FR-06** | Test metrics, confusion matrix, weights, secondary comparison | `src/evaluate_model.py` | `tests/test_model.py::test_evaluate_pipeline_generates_valid_metrics`<br>`outputs/reports/metrics.json` | **VERIFIED** |
| **FR-07** | Input validation and standalone prediction inference | `src/predict.py` | `tests/test_prediction.py::test_case_2_valid_input`<br>`tests/test_prediction.py::test_case_3_invalid_cgpa` through `test_case_7` | **VERIFIED** |
| **FR-08** | Streamlit web application with academic disclaimer | `app/app.py` | `tests/test_prediction.py::test_case_8_saved_model_reload`<br>Manual UI verification | **VERIFIED** |

---

## 3. Risk Register (§13)

Per Section 13 of the PBL Master Specification, the risk register covers critical failure modes across the ML lifecycle:

| Risk ID | Risk Description | Likelihood | Impact | Mitigation Strategy Implemented |
| :--- | :--- | :--- | :--- | :--- |
| **RSK-01** | **Dataset Quality & Small Sample Size**<br>Dataset contains fewer than 500 records or noisy invalid entries. | Low | Critical | Controlled dataset generation script enforces $N = 1,000$ rows (above §6.1 floor). `src/data_loader.py` enforces a programmatic assertion `len(df) >= 500`. |
| **RSK-02** | **Class Imbalance**<br>Extreme skew in target distribution causing trivial majority-class predictions. | Medium | High | Dataset generation applies a calibrated threshold (0.730) yielding 55.5% Eligible and 44.5% Not Eligible. Stratified splitting preserves this balance across partitions. |
| **RSK-03** | **Data Leakage**<br>Test set statistics inadvertently inform training transforms. | High | Critical | Single scikit-learn `Pipeline` encapsulates `StandardScaler` and `Perceptron`. `scaler.fit` executes strictly on $X_{\text{train}}$ inside `train_model.py`. |
| **RSK-04** | **Low Perceptron Linear Separability**<br>Perceptron fails to converge due to non-linearly separable academic boundary. | Medium | Moderate | Academic honesty contract (§1 & §8.3): report the true accuracy achieved (73.00%). The model weights and decision score are fully explained without artificial manipulation. |
| **RSK-05** | **Invalid / Out-of-Bounds User Input**<br>Users input impossible values (e.g. CGPA 15, negative backlogs) causing crashes or absurd outputs. | High | High | `src/predict.py` implements `validate_student_input()` checking all feature bounds before inference. Streamlit UI wraps all inputs with bounded constraints and graceful error alerts. |
| **RSK-06** | **Missing Model Artifact on Launch**<br>App fails with fatal exception if `joblib` file is absent. | Low | High | Clear, human-readable error messages (§9 error handling): checks `MODEL_PIPELINE_PATH.exists()` and instructs user to execute `python src/train_model.py`. |
