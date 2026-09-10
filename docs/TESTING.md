# Test Execution Log & Validation Matrix (§11)

**Project:** Perceptron-Based Admission Eligibility Prediction System (`admission-eligibility-perceptron`)  
**Test Suite:** `pytest -v tests/`  
**Runtime:** Python 3.12.10 (pytest 8.4.2)  
**Execution Date:** 2026-09-04  
**Summary:** 20 passed, 0 failed (100% pass rate)

---

## 1. Section 11 Test Cases Execution Table

Per Section 11 of the PBL Master Specification, all required verification cases were programmatically evaluated and verified:

| Test ID | Case Description | Test Input | Expected Behavior | Actual Behavior Observed | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TC-01** | Dataset validation | `data/raw/admission_data.csv` | Dataset loads; $\ge 500$ rows; required columns present; target binary {0, 1}. | Loaded 1,000 rows, 11 columns, zero missing values, target values strictly in {0, 1}. | **PASS** |
| **TC-02** | Valid input prediction | Realistic profile (CGPA: 8.5, Entrance: 82, 10th: 86.5%, 12th: 84%, Backlogs: 0) | Pipeline runs; returns human-readable label (`ELIGIBLE`), score $z = +12.85$, and disclaimer. | Returned `status_code: 1`, `status_label: "ELIGIBLE"`, $z = +12.8486$, with linear explanation. | **PASS** |
| **TC-03** | Invalid CGPA | `CGPA = 15.0` (out of [0.0, 10.0] scale) | Validation error raised; inference blocked. | `validate_student_input` flagged range violation; `predict_eligibility` raised `ValueError("Validation Error for 'CGPA': Value 15.0 is out of allowable range [0.0, 10.0]")`. | **PASS** |
| **TC-04** | Invalid percentage | `12th_Percentage = 120.0` (out of [0.0, 100.0] scale) | Validation error raised; inference blocked. | Validation error raised; `predict_eligibility` raised `ValueError("Validation Error for '12th_Percentage': Value 120.0 is out of allowable range [0.0, 100.0]")`. | **PASS** |
| **TC-05** | Negative backlogs | `Previous_Backlogs = -2` | Validation error raised; negative count rejected. | Validation error raised; `predict_eligibility` raised `ValueError("Invalid value for 'Previous_Backlogs': backlogs cannot be negative (received -2)")`. | **PASS** |
| **TC-06** | Missing input | `Entrance_Score = ""` (blank field) | Clear validation error returned; no crash or stack trace. | Validation caught empty field; raised `ValueError("Missing required input: 'Entrance_Score' cannot be empty (§11)")`. | **PASS** |
| **TC-07A**| Boundary value: Minimum | All percentages: 0.0, CGPA: 0.0, Entrance: 0.0, Backlogs: 0 | Accepted as valid boundary limits; executes without error. | Pipeline executed; returned `NOT ELIGIBLE` with negative linear decision score ($z = -14.28$). | **PASS** |
| **TC-07B**| Boundary value: Maximum | All percentages: 100.0, CGPA: 10.0, Entrance: 100.0, Backlogs: 20 | Accepted as valid boundary limits; executes without error. | Pipeline executed smoothly; extreme backlogs penalty correctly applied. | **PASS** |
| **TC-08** | Saved model persistence & reload | Reload `models/admission_perceptron_pipeline.joblib` | Loads pipeline artifact without retraining; produces identical deterministic predictions. | Pipeline reloaded cleanly via `joblib.load`; verified identical deterministic output ($z_1 = z_2$). | **PASS** |

---

## 2. Preprocessing & Architecture Unit Tests

| Test Function | Target Module | Scope | Status |
| :--- | :--- | :--- | :--- |
| `test_dataset_exists_and_loads` | `src/data_loader.py` | Validates file existence and DataFrame loading | **PASS** |
| `test_dataset_schema_and_columns` | `src/data_loader.py` | Confirms all 11 required columns are present | **PASS** |
| `test_target_is_binary` | `src/data_loader.py` | Verifies target values are strictly 0 and 1 | **PASS** |
| `test_zero_missing_values` | `src/data_loader.py` | Ensures dataset is 100% complete with no nulls | **PASS** |
| `test_feature_target_separation_excludes_identifier` | `src/preprocessing.py` | Asserts `Student_ID` is removed from $X$ | **PASS** |
| `test_stratified_split_preserves_class_ratio` | `src/preprocessing.py` | Verifies train and test splits have identical class proportions | **PASS** |
| `test_pipeline_structure` | `src/preprocessing.py` | Verifies Pipeline contains `StandardScaler` then `Perceptron` | **PASS** |

---

## 3. Model Architecture & Persistence Tests

| Test Function | Target Module | Scope | Status |
| :--- | :--- | :--- | :--- |
| `test_model_pipeline_artifact_exists` | `src/train_model.py` | Asserts `.joblib` file exists on disk | **PASS** |
| `test_model_artifact_is_valid_pipeline` | `src/train_model.py` | Asserts artifact is an instantiated scikit-learn Pipeline | **PASS** |
| `test_perceptron_weights_and_bias_dimensions` | `src/evaluate_model.py`| Verifies 9 weights match 9 candidate features and 1 bias scalar | **PASS** |
| `test_evaluate_pipeline_generates_valid_metrics` | `src/evaluate_model.py`| Asserts all test metrics lie within $[0, 1]$ and CM sums to 200 | **PASS** |

---

## 4. Test Execution Command & Verification Output

```bash
$ pytest -v tests/
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-8.4.2, pluggy-1.6.0
rootdir: C:\Users\DELL\OneDrive\Desktop\NNDL
collected 20 items

tests/test_model.py::test_model_pipeline_artifact_exists PASSED          [  5%]
tests/test_model.py::test_model_artifact_is_valid_pipeline PASSED        [ 10%]
tests/test_model.py::test_perceptron_weights_and_bias_dimensions PASSED  [ 15%]
tests/test_model.py::test_evaluate_pipeline_generates_valid_metrics PASSED [ 20%]
tests/test_prediction.py::test_case_1_dataset_validation PASSED          [ 25%]
tests/test_prediction.py::test_case_2_valid_input PASSED                 [ 30%]
tests/test_prediction.py::test_case_3_invalid_cgpa PASSED                [ 35%]
tests/test_prediction.py::test_case_4_invalid_percentage PASSED          [ 40%]
tests/test_prediction.py::test_case_5_negative_backlogs PASSED           [ 45%]
tests/test_prediction.py::test_case_6_missing_input PASSED               [ 50%]
tests/test_prediction.py::test_case_7_boundary_values_minimum PASSED     [ 55%]
tests/test_prediction.py::test_case_7_boundary_values_maximum PASSED     [ 60%]
tests/test_prediction.py::test_case_8_saved_model_reload PASSED          [ 65%]
tests/test_preprocessing.py::test_dataset_exists_and_loads PASSED        [ 70%]
tests/test_preprocessing.py::test_dataset_schema_and_columns PASSED      [ 75%]
tests/test_preprocessing.py::test_target_is_binary PASSED                [ 80%]
tests/test_preprocessing.py::test_zero_missing_values PASSED             [ 85%]
tests/test_preprocessing.py::test_feature_target_separation_excludes_identifier PASSED [ 90%]
tests/test_preprocessing.py::test_stratified_split_preserves_class_ratio PASSED [ 95%]
tests/test_preprocessing.py::test_pipeline_structure PASSED              [100%]

======================== 20 passed, 1 warning in 7.45s ========================
```
