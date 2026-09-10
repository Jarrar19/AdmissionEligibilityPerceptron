"""Model Training module for Admission Eligibility Perceptron System.

Fits a scikit-learn Pipeline (StandardScaler + Perceptron) on the training partition
and persists the serialized model pipeline to models/admission_perceptron_pipeline.joblib
per Section 8.1, 8.2, and 12 of the PBL Master Specification.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODEL_PIPELINE_PATH,
    MODELS_DIR,
    PERCEPTRON_PARAMS,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_loader import load_data
from src.preprocessing import build_pipeline, clean_dataset, separate_features_target, split_data


def train_model(
    save_path: Optional[Path] = MODEL_PIPELINE_PATH,
    perceptron_params: Optional[dict] = None,
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Execute complete end-to-end training pipeline.

    1. Loads dataset and validates schema.
    2. Cleans data and separates features/target (excluding Student_ID).
    3. Performs stratified 80/20 train/test split.
    4. Builds and fits Pipeline(StandardScaler, Perceptron) on training set only.
    5. Saves pipeline to models/admission_perceptron_pipeline.joblib.

    Returns:
        Tuple of (fitted_pipeline, X_train, X_test, y_train, y_test)
    """
    print("=" * 60)
    print("STARTING MODEL TRAINING PIPELINE (§8)")
    print("=" * 60)

    # 1. Load & Clean
    raw_df = load_data()
    clean_df = clean_dataset(raw_df)
    X, y = separate_features_target(clean_df)

    # 2. Stratified Train/Test Split (Zero Data Leakage)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train split: {len(X_train)} samples (Eligible: {y_train.sum()}, Not Eligible: {len(y_train) - y_train.sum()})")
    print(f"Test split:  {len(X_test)} samples (Eligible: {y_test.sum()}, Not Eligible: {len(y_test) - y_test.sum()})")

    # 3. Build and Fit Pipeline
    pipeline = build_pipeline(perceptron_params or PERCEPTRON_PARAMS)
    print(f"\nFitting Pipeline on training data only...")
    pipeline.fit(X_train, y_train)

    train_acc = pipeline.score(X_train, y_train)
    print(f"Training Set Accuracy: {train_acc * 100:.2f}%")

    # 4. Persist Pipeline Artifact (§8.2)
    destination = save_path or MODEL_PIPELINE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination)
    print(f"Trained pipeline persisted successfully to: {destination}")

    return pipeline, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    train_model()
