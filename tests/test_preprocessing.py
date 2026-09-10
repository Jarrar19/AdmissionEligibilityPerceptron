"""Unit tests for data loading and preprocessing modules (§11 & §12)."""

import sys
from pathlib import Path
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FEATURE_COLS, IDENTIFIER_COL, MIN_RECORDS_REQUIRED, RAW_DATA_PATH, TARGET_COL
from src.data_loader import load_data, validate_raw_data
from src.preprocessing import build_pipeline, clean_dataset, separate_features_target, split_data


def test_dataset_exists_and_loads():
    """Verify raw dataset file exists and loads into pandas DataFrame."""
    assert RAW_DATA_PATH.exists(), f"Raw data CSV missing at {RAW_DATA_PATH}"
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= MIN_RECORDS_REQUIRED, f"Expected >= {MIN_RECORDS_REQUIRED} rows, found {len(df)}"


def test_dataset_schema_and_columns():
    """Verify all expected columns (Identifier, Features, Target) are present."""
    df = load_data()
    expected_cols = {IDENTIFIER_COL} | set(FEATURE_COLS) | {TARGET_COL}
    assert expected_cols.issubset(set(df.columns))


def test_target_is_binary():
    """Verify target column contains strictly {0, 1}."""
    df = load_data()
    unique_vals = set(df[TARGET_COL].unique())
    assert unique_vals == {0, 1}, f"Target values must be exactly {{0, 1}}, got {unique_vals}"


def test_zero_missing_values():
    """Verify raw dataset contains no null or missing values."""
    df = load_data()
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected null values"


def test_feature_target_separation_excludes_identifier():
    """Verify Student_ID identifier is excluded from model feature matrix X (§6.4 & §8.1)."""
    df = load_data()
    X, y = separate_features_target(df)
    assert IDENTIFIER_COL not in X.columns, f"Identifier {IDENTIFIER_COL} was not removed from features"
    assert list(X.columns) == FEATURE_COLS
    assert len(y) == len(df)


def test_stratified_split_preserves_class_ratio():
    """Verify train_test_split preserves identical class proportions without leakage."""
    df = load_data()
    clean = clean_dataset(df)
    X, y = separate_features_target(clean)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42)

    total_ratio = y.mean()
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()

    assert abs(total_ratio - train_ratio) < 0.01, "Train ratio drifted from population"
    assert abs(total_ratio - test_ratio) < 0.01, "Test ratio drifted from population"
    assert len(X_train) == 800
    assert len(X_test) == 200


def test_pipeline_structure():
    """Verify Pipeline has StandardScaler as step 1 and Perceptron as step 2."""
    pipeline = build_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert "scaler" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert "perceptron" in pipeline.named_steps
