"""Prediction & Input Validation module for Admission Eligibility Perceptron System.

Provides standalone prediction functionality, strict input validation against
defined feature boundaries, and human-readable decision outputs
per Section 5, Section 10, and Section 11 of the PBL Master Specification.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FEATURE_COLS,
    MODEL_PIPELINE_PATH,
    TARGET_LABELS,
    VALIDATION_RANGES,
)

# Global cached pipeline to prevent reloading on repeated calls
_CACHED_PIPELINE: Optional[Pipeline] = None


def get_pipeline(model_path: Optional[Path] = None) -> Pipeline:
    """Load and cache the trained Perceptron pipeline artifact (§8.2 & §10)."""
    global _CACHED_PIPELINE
    path = model_path or MODEL_PIPELINE_PATH
    if _CACHED_PIPELINE is None or model_path is not None:
        if not path.exists():
            raise FileNotFoundError(
                f"Trained model pipeline not found at '{path}'. "
                f"Please run 'python src/train_model.py' first (§9 error handling)."
            )
        _CACHED_PIPELINE = joblib.load(path)
    return _CACHED_PIPELINE


def validate_student_input(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate student feature inputs against physical and domain boundaries (§10 & §11).

    Checks:
    - All required features are present and non-empty.
    - Numeric types are valid.
    - Feature values lie within strict valid bounds (e.g. CGPA 0-10, % 0-100, backlogs >= 0).

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    for col in FEATURE_COLS:
        if col not in data or data[col] is None or data[col] == "":
            return False, f"Missing required input: '{col}' cannot be empty (§11)."

        val = data[col]
        bounds = VALIDATION_RANGES[col]
        target_type = bounds["type"]

        # Type conversion check
        try:
            numeric_val = target_type(val)
        except (ValueError, TypeError):
            return False, f"Invalid format for '{col}': expected a {target_type.__name__}, received '{val}'."

        # Range bounds check
        min_v, max_v = bounds["min"], bounds["max"]
        if numeric_val < min_v or numeric_val > max_v:
            if col == "Previous_Backlogs" and numeric_val < 0:
                return False, f"Invalid value for 'Previous_Backlogs': backlogs cannot be negative (received {numeric_val})."
            return (
                False,
                f"Validation Error for '{col}': Value {numeric_val} is out of allowable range [{min_v}, {max_v}] {bounds['unit']}.",
            )

    return True, None


def predict_eligibility(
    data: Dict[str, Any], model_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Execute prediction inference on a validated student record.

    Calculates:
    1. Pipeline transformation (StandardScaler).
    2. Linear decision score: z = sum(w_i * x_scaled_i) + b.
    3. Step activation: Eligible (1) if z >= 0 else Not Eligible (0).

    Args:
        data: Dictionary of student features.
        model_path: Optional path to joblib model.

    Returns:
        Dictionary with status, human-readable label, linear score, and disclaimer.

    Raises:
        ValueError: If input validation fails.
    """
    is_valid, error_msg = validate_student_input(data)
    if not is_valid:
        raise ValueError(error_msg)

    pipeline = get_pipeline(model_path)

    # Format into 1-row DataFrame with strict column ordering
    input_df = pd.DataFrame([{col: VALIDATION_RANGES[col]["type"](data[col]) for col in FEATURE_COLS}])

    # Extract pipeline components for interpretability
    scaler = pipeline.named_steps["scaler"]
    perceptron = pipeline.named_steps["perceptron"]

    # Transform input
    scaled_features = scaler.transform(input_df)[0]
    weights = perceptron.coef_[0]
    bias = float(perceptron.intercept_[0])

    # Compute raw linear decision function z
    z = float(np.dot(weights, scaled_features) + bias)
    pred_code = int(1 if z >= 0 else 0)
    pred_label = TARGET_LABELS[pred_code]

    # Detailed feature contribution breakdown
    contributions = {}
    for feat, w, x_s in zip(FEATURE_COLS, weights, scaled_features):
        contributions[feat] = {
            "raw_value": input_df[feat].iloc[0],
            "scaled_value": round(float(x_s), 4),
            "weight": round(float(w), 4),
            "linear_contribution": round(float(w * x_s), 4),
        }

    return {
        "status_code": pred_code,
        "status_label": pred_label,
        "decision_score_z": round(z, 4),
        "bias": round(bias, 4),
        "feature_contributions": contributions,
        "is_eligible": bool(pred_code == 1),
        "disclaimer": (
            "This prediction is generated by an academic machine learning demonstration "
            "(single-layer Perceptron) and does NOT constitute an official university admission decision (§10)."
        ),
    }


if __name__ == "__main__":
    sample_student = {
        "10th_Percentage": 88.5,
        "12th_Percentage": 86.0,
        "CGPA": 8.7,
        "Entrance_Score": 82.0,
        "Interview_Score": 80.0,
        "Aptitude_Score": 85.0,
        "Extracurricular_Score": 75.0,
        "Attendance": 92.0,
        "Previous_Backlogs": 0,
    }
    result = predict_eligibility(sample_student)
    print("Sample Prediction Result:")
    print(f"Status: {result['status_label']} (Score z: {result['decision_score_z']})")
