"""Unit tests executing the exact test matrix defined in Section 11 of the PBL Master Specification."""

import sys
from pathlib import Path
import joblib
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_PIPELINE_PATH
from src.data_loader import load_data, validate_raw_data
from src.predict import get_pipeline, predict_eligibility, validate_student_input


@pytest.fixture
def valid_student_input():
    """Realistic student profile for inference testing (§11 Case 2)."""
    return {
        "10th_Percentage": 86.5,
        "12th_Percentage": 84.0,
        "CGPA": 8.5,
        "Entrance_Score": 82.0,
        "Interview_Score": 78.0,
        "Aptitude_Score": 80.0,
        "Extracurricular_Score": 75.0,
        "Attendance": 92.0,
        "Previous_Backlogs": 0,
    }


# ---------------------------------------------------------
# Test Case 1: Dataset validation (§11)
# Input: Dataset CSV
# Expected: Loads; required columns present; target valid
# ---------------------------------------------------------
def test_case_1_dataset_validation():
    df = load_data()
    is_valid, msg = validate_raw_data(df)
    assert is_valid is True, f"Dataset validation failed: {msg}"
    assert len(df) >= 500
    assert "Eligible" in df.columns
    assert set(df["Eligible"].unique()).issubset({0, 1})


# ---------------------------------------------------------
# Test Case 2: Valid input (§11)
# Input: Realistic student values
# Expected: Pipeline runs; returns Eligible / Not Eligible with score
# ---------------------------------------------------------
def test_case_2_valid_input(valid_student_input):
    result = predict_eligibility(valid_student_input)
    assert "status_code" in result
    assert result["status_code"] in [0, 1]
    assert result["status_label"] in ["ELIGIBLE", "NOT ELIGIBLE"]
    assert "decision_score_z" in result
    assert isinstance(result["decision_score_z"], float)
    assert "disclaimer" in result


# ---------------------------------------------------------
# Test Case 3: Invalid CGPA (§11)
# Input: CGPA = 15
# Expected: Validation error returned; model inference blocked
# ---------------------------------------------------------
def test_case_3_invalid_cgpa(valid_student_input):
    bad_input = valid_student_input.copy()
    bad_input["CGPA"] = 15.0
    is_valid, err_msg = validate_student_input(bad_input)
    assert is_valid is False
    assert "CGPA" in err_msg
    with pytest.raises(ValueError, match="CGPA"):
        predict_eligibility(bad_input)


# ---------------------------------------------------------
# Test Case 4: Invalid percentage (§11)
# Input: 12th_Percentage = 120
# Expected: Validation error returned; model inference blocked
# ---------------------------------------------------------
def test_case_4_invalid_percentage(valid_student_input):
    bad_input = valid_student_input.copy()
    bad_input["12th_Percentage"] = 120.0
    is_valid, err_msg = validate_student_input(bad_input)
    assert is_valid is False
    assert "12th_Percentage" in err_msg
    with pytest.raises(ValueError, match="12th_Percentage"):
        predict_eligibility(bad_input)


# ---------------------------------------------------------
# Test Case 5: Negative backlogs (§11)
# Input: Previous_Backlogs = -2
# Expected: Validation error returned; negative value rejected
# ---------------------------------------------------------
def test_case_5_negative_backlogs(valid_student_input):
    bad_input = valid_student_input.copy()
    bad_input["Previous_Backlogs"] = -2
    is_valid, err_msg = validate_student_input(bad_input)
    assert is_valid is False
    assert "Previous_Backlogs" in err_msg or "backlogs cannot be negative" in err_msg
    with pytest.raises(ValueError, match="backlogs"):
        predict_eligibility(bad_input)


# ---------------------------------------------------------
# Test Case 6: Missing input (§11)
# Input: Blank / omitted field
# Expected: Clear validation message returned
# ---------------------------------------------------------
def test_case_6_missing_input(valid_student_input):
    bad_input = valid_student_input.copy()
    bad_input["Entrance_Score"] = ""  # blank field
    is_valid, err_msg = validate_student_input(bad_input)
    assert is_valid is False
    assert "Entrance_Score" in err_msg
    with pytest.raises(ValueError, match="Entrance_Score"):
        predict_eligibility(bad_input)


# ---------------------------------------------------------
# Test Case 7: Boundary values (§11)
# Input: CGPA 0/10, % 0/100, attendance 0/100
# Expected: Accepted (valid boundary limits)
# ---------------------------------------------------------
def test_case_7_boundary_values_minimum():
    min_input = {
        "10th_Percentage": 0.0,
        "12th_Percentage": 0.0,
        "CGPA": 0.0,
        "Entrance_Score": 0.0,
        "Interview_Score": 0.0,
        "Aptitude_Score": 0.0,
        "Extracurricular_Score": 0.0,
        "Attendance": 0.0,
        "Previous_Backlogs": 0,
    }
    is_valid, err = validate_student_input(min_input)
    assert is_valid is True, f"Min boundary rejected: {err}"
    res = predict_eligibility(min_input)
    assert res["status_label"] == "NOT ELIGIBLE"


def test_case_7_boundary_values_maximum():
    max_input = {
        "10th_Percentage": 100.0,
        "12th_Percentage": 100.0,
        "CGPA": 10.0,
        "Entrance_Score": 100.0,
        "Interview_Score": 100.0,
        "Aptitude_Score": 100.0,
        "Extracurricular_Score": 100.0,
        "Attendance": 100.0,
        "Previous_Backlogs": 20,
    }
    is_valid, err = validate_student_input(max_input)
    assert is_valid is True, f"Max boundary rejected: {err}"
    res = predict_eligibility(max_input)
    assert res["status_label"] in ["ELIGIBLE", "NOT ELIGIBLE"]


# ---------------------------------------------------------
# Test Case 8: Saved model reload without retraining (§11)
# Input: Fresh pipeline reload from disk
# Expected: Loads pipeline cleanly; returns identical deterministic predictions
# ---------------------------------------------------------
def test_case_8_saved_model_reload(valid_student_input):
    # Fresh reload directly via joblib
    pipe1 = joblib.load(MODEL_PIPELINE_PATH)
    pipe2 = joblib.load(MODEL_PIPELINE_PATH)

    assert pipe1 is not None
    assert pipe2 is not None

    # Verify identical deterministic predictions
    pred1 = predict_eligibility(valid_student_input, model_path=MODEL_PIPELINE_PATH)
    pred2 = predict_eligibility(valid_student_input, model_path=MODEL_PIPELINE_PATH)
    assert pred1["status_code"] == pred2["status_code"]
    assert pred1["decision_score_z"] == pred2["decision_score_z"]
