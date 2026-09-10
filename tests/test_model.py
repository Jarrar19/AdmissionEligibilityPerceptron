"""Unit tests for Perceptron model training, evaluation, and persistence (§11 & §12)."""

import sys
from pathlib import Path
import joblib
import pytest
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FEATURE_COLS, METRICS_JSON_PATH, MODEL_PIPELINE_PATH
from src.evaluate_model import evaluate_pipeline
from src.train_model import train_model


def test_model_pipeline_artifact_exists():
    """Verify saved joblib pipeline exists at designated model path (§8.2)."""
    assert MODEL_PIPELINE_PATH.exists(), f"Model pipeline missing at {MODEL_PIPELINE_PATH}"


def test_model_artifact_is_valid_pipeline():
    """Verify loaded artifact is a fitted scikit-learn Pipeline."""
    pipeline = joblib.load(MODEL_PIPELINE_PATH)
    assert isinstance(pipeline, Pipeline)
    assert hasattr(pipeline, "predict"), "Pipeline missing predict method"


def test_perceptron_weights_and_bias_dimensions():
    """Verify Perceptron coefficient vector matches exact feature count (9 features)."""
    pipeline = joblib.load(MODEL_PIPELINE_PATH)
    perceptron = pipeline.named_steps["perceptron"]
    assert perceptron.coef_.shape == (1, len(FEATURE_COLS)), "Coefficient dimension mismatch"
    assert hasattr(perceptron, "intercept_"), "Intercept missing"
    assert len(perceptron.intercept_) == 1, "Intercept must be 1-dimensional for binary classification"


def test_evaluate_pipeline_generates_valid_metrics():
    """Verify programmatic evaluation produces valid metrics within [0, 1]."""
    summary = evaluate_pipeline(save_reports=False)
    m = summary["metrics"]
    for metric_name in ["accuracy", "precision", "recall", "f1_score"]:
        val = m[metric_name]
        assert 0.0 <= val <= 1.0, f"Metric {metric_name} ({val}) out of probability bounds [0, 1]"

    cm = summary["confusion_matrix"]
    assert cm["true_negatives"] + cm["false_positives"] + cm["false_negatives"] + cm["true_positives"] == 200
