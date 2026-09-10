"""Data Loader module for Admission Eligibility Perceptron System.

Handles locating, loading, and structural verification of the raw admission dataset
per Section 8.1 and Section 12 of the PBL Master Specification.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FEATURE_COLS,
    IDENTIFIER_COL,
    MIN_RECORDS_REQUIRED,
    RAW_DATA_PATH,
    TARGET_COL,
)


def validate_raw_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate that the dataset satisfies structural and integrity acceptance criteria (§6.3).

    Checks:
    1. Minimum record count (>= 500 records).
    2. All expected columns (Identifier, Features, Target) are present.
    3. Target column is binary with values {0, 1}.
    4. Missing values count.

    Returns:
        Tuple of (is_valid, validation_message).
    """
    if len(df) < MIN_RECORDS_REQUIRED:
        return (
            False,
            f"Dataset size violation: contains {len(df)} records, required at least {MIN_RECORDS_REQUIRED} (§6.1).",
        )

    expected_cols = {IDENTIFIER_COL} | set(FEATURE_COLS) | {TARGET_COL}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        return False, f"Missing required columns in dataset: {sorted(list(missing_cols))}"

    # Target binary check
    unique_targets = set(df[TARGET_COL].dropna().unique())
    if not unique_targets.issubset({0, 1}):
        return False, f"Target column '{TARGET_COL}' contains invalid values: {unique_targets}. Expected {{0, 1}}."

    # Check for missing values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        return (
            False,
            f"Dataset contains {total_nulls} missing values across columns: {null_counts[null_counts > 0].to_dict()}",
        )

    return True, f"Dataset validated successfully: {len(df)} records, all required columns present, zero missing values."


def load_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load admission dataset from CSV and verify its integrity.

    Args:
        file_path: Optional custom path. Defaults to RAW_DATA_PATH from config.

    Returns:
        Loaded and verified pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If dataset validation fails.
    """
    path = file_path or RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Admission dataset was not found at '{path}'. "
            f"Please run 'python scripts/generate_dataset.py' first (§9 error handling)."
        )

    df = pd.read_csv(path)
    is_valid, message = validate_raw_data(df)
    if not is_valid:
        raise ValueError(f"Dataset integrity error: {message}")

    return df


if __name__ == "__main__":
    try:
        data = load_data()
        print(f"Loaded dataset successfully with shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
    except Exception as exc:
        print(f"Data loading error: {exc}")
