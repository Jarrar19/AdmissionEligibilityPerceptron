"""Model Evaluation module for Admission Eligibility Perceptron System.

Evaluates the trained Perceptron pipeline on the held-out test set,
computes true classification metrics (Accuracy, Precision, Recall, F1, Confusion Matrix),
extracts decision weights and bias for model interpretability, and exports reports
per Section 8.4 and Section 12 of the PBL Master Specification.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASSIFICATION_REPORT_PATH,
    FEATURE_COLS,
    FIGURES_DIR,
    METRICS_JSON_PATH,
    MODEL_PIPELINE_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_LABELS,
    TEST_SIZE,
)
from src.data_loader import load_data
from src.preprocessing import clean_dataset, separate_features_target, split_data


def evaluate_pipeline(
    pipeline_path: Optional[Path] = MODEL_PIPELINE_PATH,
    save_reports: bool = True,
) -> Dict:
    """Evaluate trained pipeline on held-out test split programmatically.

    Never hard-codes results. All metrics are computed strictly from real predictions.

    Returns:
        Dictionary containing test metrics, confusion matrix, weights, and secondary comparison.
    """
    path = pipeline_path or MODEL_PIPELINE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{path}'. "
            f"Please run 'python src/train_model.py' before running evaluation (§9 & §10)."
        )

    pipeline: Pipeline = joblib.load(path)
    print(f"Loaded trained pipeline from: {path}")

    # Load dataset and recreate identical held-out test split
    raw_df = load_data()
    clean_df = clean_dataset(raw_df)
    X, y = separate_features_target(clean_df)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Programmatic predictions on held-out test set (§8.4)
    y_pred = pipeline.predict(X_test)

    # Compute classification metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
    clf_report = classification_report(y_test, y_pred, target_names=["Not Eligible", "Eligible"])

    # Model Interpretation (§8.4): Extract weights and bias
    perceptron_step: Perceptron = pipeline.named_steps["perceptron"]
    weights = perceptron_step.coef_[0].tolist()
    bias = float(perceptron_step.intercept_[0])
    feature_weight_map = {feat: round(w, 4) for feat, w in zip(FEATURE_COLS, weights)}

    # Secondary Baseline Comparison (§8.4): Logistic Regression comparison
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)
    lr_metrics = {
        "model": "Logistic Regression (Secondary Baseline)",
        "accuracy": float(round(accuracy_score(y_test, lr_pred), 4)),
        "precision": float(round(precision_score(y_test, lr_pred, zero_division=0), 4)),
        "recall": float(round(recall_score(y_test, lr_pred, zero_division=0), 4)),
        "f1_score": float(round(f1_score(y_test, lr_pred, zero_division=0), 4)),
    }

    metrics_summary = {
        "primary_model": "Perceptron Binary Classifier (sklearn.linear_model.Perceptron)",
        "hyperparameters": perceptron_step.get_params(),
        "test_sample_count": len(y_test),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
        },
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "matrix": cm,
        },
        "model_parameters": {
            "weights": feature_weight_map,
            "bias_intercept": round(bias, 4),
            "decision_rule": "z = sum(w_i * x_i) + b; Predict Eligible (1) if z >= 0 else Not Eligible (0)",
        },
        "secondary_comparison": lr_metrics,
    }

    print("\n" + "=" * 60)
    print("PROGRAMMATIC MODEL EVALUATION RESULTS (§8.4)")
    print("=" * 60)
    print(f"Test Set Accuracy:  {acc * 100:.2f}%")
    print(f"Test Set Precision: {prec * 100:.2f}%")
    print(f"Test Set Recall:    {rec * 100:.2f}%")
    print(f"Test Set F1-Score:  {f1 * 100:.2f}%")
    print(f"\nConfusion Matrix (Test Split = {len(y_test)}):")
    print(f"  TN (Correct Not Eligible): {tn}  |  FP (False Alarm):    {fp}")
    print(f"  FN (Missed Eligible):     {fn}  |  TP (Correct Eligible): {tp}")

    print("\nPerceptron Learned Parameters:")
    print(f"  Bias (b): {bias:.4f}")
    for feat, w in feature_weight_map.items():
        print(f"  {feat:<22}: {w:+.4f}")

    print("\nClassification Report:\n" + clf_report)

    print("\nSecondary Model Comparison (§8.4):")
    print(f"  Perceptron Accuracy:          {acc*100:.2f}% (F1: {f1*100:.2f}%)")
    print(f"  Logistic Regression Accuracy: {lr_metrics['accuracy']*100:.2f}% (F1: {lr_metrics['f1_score']*100:.2f}%)")
    print("  *Perceptron remains the designated primary model per §1 Rule 1.*")

    if save_reports:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        # Save metrics.json
        with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"\nSaved metrics summary: {METRICS_JSON_PATH}")

        # Save classification_report.txt
        with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("ADMISSION ELIGIBILITY PERCEPTRON CLASSIFICATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Test Set Size: {len(y_test)} samples\n")
            f.write(f"Accuracy:  {acc:.4f}\n")
            f.write(f"Precision: {prec:.4f}\n")
            f.write(f"Recall:    {rec:.4f}\n")
            f.write(f"F1-Score:  {f1:.4f}\n\n")
            f.write("Detailed Report:\n")
            f.write(clf_report + "\n")
            f.write("Confusion Matrix:\n")
            f.write(f"  TN={tn}, FP={fp}\n  FN={fn}, TP={tp}\n\n")
            f.write("Learned Feature Weights:\n")
            for feat, w in feature_weight_map.items():
                f.write(f"  {feat}: {w:+.4f}\n")
            f.write(f"Bias (b): {bias:+.4f}\n\n")
            f.write("Secondary Comparison Table:\n")
            f.write(f"  Perceptron:          Acc={acc:.4f}, F1={f1:.4f}\n")
            f.write(f"  Logistic Regression: Acc={lr_metrics['accuracy']:.4f}, F1={lr_metrics['f1_score']:.4f}\n")
        print(f"Saved text report: {CLASSIFICATION_REPORT_PATH}")

        # 1. Confusion Matrix Figure
        fig, ax = plt.subplots(figsize=(5, 4.2), dpi=300)
        cax = ax.matshow([[tn, fp], [fn, tp]], cmap="Blues")
        fig.colorbar(cax, fraction=0.046, pad=0.04)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Not Eligible (0)", "Eligible (1)"], fontsize=9, fontweight="bold")
        ax.set_yticklabels(["Not Eligible (0)", "Eligible (1)"], fontsize=9, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold", labelpad=8)
        ax.set_ylabel("True Ground Truth", fontsize=10, fontweight="bold", labelpad=8)
        ax.set_title("Perceptron Test Confusion Matrix (§8.4)", fontsize=11, fontweight="bold", pad=16)

        matrix_cells = [[(tn, "TN"), (fp, "FP")], [(fn, "FN"), (tp, "TP")]]
        for i in range(2):
            for j in range(2):
                val, tag = matrix_cells[i][j]
                color = "white" if val > (len(y_test) / 3) else "black"
                ax.text(j, i, f"{tag}\n{val}", ha="center", va="center", color=color, fontweight="bold", fontsize=11)

        plt.tight_layout()
        cm_path = FIGURES_DIR / "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"Saved confusion matrix plot: {cm_path}")

        # 2. Feature Weights Bar Chart
        sorted_weights = sorted(feature_weight_map.items(), key=lambda item: item[1])
        feats, w_vals = zip(*sorted_weights)
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
        bar_colors = ["#2ecc71" if w >= 0 else "#e74c3c" for w in w_vals]
        ax.barh(feats, w_vals, color=bar_colors, edgecolor="#333333", linewidth=0.8, height=0.6)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title("Perceptron Learned Feature Weights ($w_i$)", fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("Weight Magnitude (Sign determines positive/negative contribution)", fontsize=9)
        ax.grid(axis="x", linestyle="--", alpha=0.6)
        for i, val in enumerate(w_vals):
            offset = 0.05 if val >= 0 else -0.05
            ha = "left" if val >= 0 else "right"
            ax.text(val + offset, i, f"{val:+.2f}", va="center", ha=ha, fontsize=8, fontweight="bold")

        plt.tight_layout()
        weights_path = FIGURES_DIR / "feature_weights.png"
        plt.savefig(weights_path)
        plt.close()
        print(f"Saved feature weights plot: {weights_path}")

    return metrics_summary


if __name__ == "__main__":
    evaluate_pipeline()
