"""Central configuration module for Admission Eligibility Perceptron System.

Centralizes all constants, hyperparameters, paths, and validation ranges
to eliminate magic numbers across modules per Section 9 of the specification.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "admission_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cleaned_data.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PIPELINE_PATH = MODELS_DIR / "admission_perceptron_pipeline.joblib"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
METRICS_JSON_PATH = REPORTS_DIR / "metrics.json"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"

# Reproducibility & Splitting Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_RECORDS_REQUIRED = 500

# Perceptron Hyperparameters (§8.2)
PERCEPTRON_PARAMS = {
    "max_iter": 1000,
    "tol": 1e-3,
    "random_state": RANDOM_STATE,
    "eta0": 1.0,
    "penalty": None,
    "fit_intercept": True,
}

# Column Definitions (§6.4 & §6.5)
IDENTIFIER_COL = "Student_ID"
TARGET_COL = "Eligible"

FEATURE_COLS = [
    "10th_Percentage",
    "12th_Percentage",
    "CGPA",
    "Entrance_Score",
    "Interview_Score",
    "Aptitude_Score",
    "Extracurricular_Score",
    "Attendance",
    "Previous_Backlogs",
]

# Validation Boundaries for Inference and App (§10 & §11)
VALIDATION_RANGES = {
    "10th_Percentage": {"min": 0.0, "max": 100.0, "type": float, "unit": "%"},
    "12th_Percentage": {"min": 0.0, "max": 100.0, "type": float, "unit": "%"},
    "CGPA": {"min": 0.0, "max": 10.0, "type": float, "unit": "/10.0"},
    "Entrance_Score": {"min": 0.0, "max": 100.0, "type": float, "unit": "/100"},
    "Interview_Score": {"min": 0.0, "max": 100.0, "type": float, "unit": "/100"},
    "Aptitude_Score": {"min": 0.0, "max": 100.0, "type": float, "unit": "/100"},
    "Extracurricular_Score": {"min": 0.0, "max": 100.0, "type": float, "unit": "/100"},
    "Attendance": {"min": 0.0, "max": 100.0, "type": float, "unit": "%"},
    "Previous_Backlogs": {"min": 0, "max": 20, "type": int, "unit": "count"},
}

# Human-Readable Target Labels (§5)
TARGET_LABELS = {
    0: "NOT ELIGIBLE",
    1: "ELIGIBLE",
}
