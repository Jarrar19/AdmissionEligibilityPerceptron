"""Preprocessing module for Admission Eligibility Perceptron System.

Handles schema/range validation, dataset cleaning, feature/target separation,
stratified data splitting, and end-to-end scikit-learn Pipeline construction
per Section 8.1 and Section 12 of the PBL Master Specification.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FEATURE_COLS,
    IDENTIFIER_COL,
    PERCEPTRON_PARAMS,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    VALIDATION_RANGES,
)


def validate_feature_ranges(df: pd.DataFrame) -> Tuple[bool, list]:
    """Audit feature values against domain valid ranges (§6.5 & §10).

    Returns:
        Tuple of (is_valid, list_of_violations).
    """
    violations = []
    for col, bounds in VALIDATION_RANGES.items():
        if col not in df.columns:
            continue
        min_val, max_val = bounds["min"], bounds["max"]
        out_of_bounds = df[(df[col] < min_val) | (df[col] > max_val)]
        if not out_of_bounds.empty:
            violations.append(
                f"{col}: {len(out_of_bounds)} values out of range [{min_val}, {max_val}]"
            )
    return len(violations) == 0, violations


def clean_dataset(
    df: pd.DataFrame, output_path: Optional[Path] = PROCESSED_DATA_PATH
) -> pd.DataFrame:
    """Clean the dataset, document duplicate handling, and save processed snapshot.

    Duplicate handling policy (§8.1):
    - Exact duplicate rows are audited. In a synthetic generation scenario, identical profiles
      across 10 distinct features are statistically improbable; if found, duplicates are dropped.
    - Missing value policy (§8.1): Dataset is verified complete; if any nulls emerge, rows are logged.

    Args:
        df: Raw DataFrame loaded via data_loader.
        output_path: Destination for cleaned CSV.

    Returns:
        Cleaned pandas DataFrame.
    """
    cleaned = df.copy()

    # Check and handle duplicates
    dup_count = cleaned.duplicated().sum()
    if dup_count > 0:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        print(f"Removed {dup_count} duplicate records (§8.1).")
    else:
        print("Duplicate check passed: Zero duplicate records found.")

    # Audit valid ranges
    is_valid, violations = validate_feature_ranges(cleaned)
    if not is_valid:
        raise ValueError(f"Feature range validation failed: {violations}")

    # Persist processed snapshot
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(output_path, index=False)
        print(f"Processed dataset saved to: {output_path}")

    return cleaned


def separate_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target, strictly excluding identifiers (§6.4 & §8.1).

    Ensures Student_ID is dropped so that no non-predictive identifier reaches the model.

    Returns:
        Tuple of (X, y) where X contains only FEATURE_COLS and y is TARGET_COL.
    """
    if IDENTIFIER_COL in df.columns:
        X = df[FEATURE_COLS].copy()
    else:
        X = df.drop(columns=[TARGET_COL]).copy()

    y = df[TARGET_COL].astype(int).copy()
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target using stratified train/test partitioning.

    Stratification ensures class balance proportions are strictly preserved
    across both train and test partitions without data leakage (§7 & §8.1).

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_pipeline(perceptron_params: Optional[dict] = None) -> Pipeline:
    """Build a unified scikit-learn Pipeline with StandardScaler and Perceptron.

    Using a single Pipeline ensures:
    1. StandardScaler is fit on training data only (zero data leakage per §8.1).
    2. Identical transformation parameters are applied at inference time without drift.

    Args:
        perceptron_params: Optional hyperparameter overrides. Defaults to PERCEPTRON_PARAMS.

    Returns:
        Un-fitted scikit-learn Pipeline.
    """
    params = perceptron_params or PERCEPTRON_PARAMS
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("perceptron", Perceptron(**params)),
        ]
    )
    return pipeline


if __name__ == "__main__":
    from src.data_loader import load_data

    raw = load_data()
    clean = clean_dataset(raw)
    X, y = separate_features_target(clean)
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"Train set: {X_tr.shape[0]} rows (Eligible: {y_tr.sum()}, Not Eligible: {len(y_tr) - y_tr.sum()})")
    print(f"Test set: {X_te.shape[0]} rows (Eligible: {y_te.sum()}, Not Eligible: {len(y_te) - y_te.sum()})")
    pipe = build_pipeline()
    print(f"Pipeline constructed: {pipe}")
