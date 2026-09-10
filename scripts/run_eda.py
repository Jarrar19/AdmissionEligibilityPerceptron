"""Exploratory Data Analysis (EDA) Script for Admission Eligibility Perceptron System.

Generates descriptive statistics, class balance metrics, correlation matrices,
and clean figures saved to outputs/figures/ per Section 7 of the PBL Spec.
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FEATURE_COLS, FIGURES_DIR, TARGET_COL
from src.data_loader import load_data


def run_eda():
    """Execute complete EDA pipeline and export figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA) REPORT (§7)")
    print("=" * 60)
    print(f"Total Records: {len(df)}")
    print(f"Total Features: {len(FEATURE_COLS)}")
    print("\nMissing Values Count:")
    print(df.isnull().sum().to_dict())

    print("\nDescriptive Statistics Summary:")
    desc = df[FEATURE_COLS].describe().round(2)
    print(desc)

    # 1. Target Class Distribution (§7)
    counts = df[TARGET_COL].value_counts().sort_index()
    pcts = df[TARGET_COL].value_counts(normalize=True).sort_index() * 100

    print("\nTarget Class Distribution:")
    for cls_val in [0, 1]:
        label = "Eligible (1)" if cls_val == 1 else "Not Eligible (0)"
        print(f"  {label}: {counts[cls_val]} ({pcts[cls_val]:.1f}%)")

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    colors = ["#e74c3c", "#2ecc71"]
    bars = ax.bar(["Not Eligible (0)", "Eligible (1)"], counts.values, color=colors, width=0.5, edgecolor="#333333", linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 12,
            f"{int(h)} ({h/len(df)*100:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    ax.set_title("Admission Eligibility Class Distribution (§7)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Students", fontsize=10)
    ax.set_ylim(0, max(counts.values) + 80)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    class_fig_path = FIGURES_DIR / "class_distribution.png"
    plt.tight_layout()
    plt.savefig(class_fig_path)
    plt.close()
    print(f"Saved: {class_fig_path}")

    # 2. Correlation Matrix Heatmap
    corr_df = df[FEATURE_COLS + [TARGET_COL]].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
    cax = ax.matshow(corr_df, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    cols = FEATURE_COLS + [TARGET_COL]
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="left", fontsize=8, fontweight="bold")
    ax.set_yticklabels(cols, fontsize=8, fontweight="bold")

    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr_df.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.4 else "black", fontsize=7.5)

    ax.set_title("Feature Correlation Matrix with Admission Eligibility", fontsize=11, fontweight="bold", pad=20)
    corr_fig_path = FIGURES_DIR / "correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(corr_fig_path)
    plt.close()
    print(f"Saved: {corr_fig_path}")

    # 3. Feature Distributions Grid (3x3)
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), dpi=300)
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        ax = axes[i]
        val_0 = df[df[TARGET_COL] == 0][col]
        val_1 = df[df[TARGET_COL] == 1][col]
        ax.hist(val_0, bins=15, alpha=0.55, color="#e74c3c", label="Not Eligible (0)", density=True)
        ax.hist(val_1, bins=15, alpha=0.55, color="#2ecc71", label="Eligible (1)", density=True)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Feature Distributions by Admission Status", fontsize=12, fontweight="bold", y=0.99)
    dist_fig_path = FIGURES_DIR / "feature_distributions.png"
    plt.tight_layout()
    plt.savefig(dist_fig_path)
    plt.close()
    print(f"Saved: {dist_fig_path}")

    # 4. Scatter Plot: CGPA vs Entrance Score
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    df_0 = df[df[TARGET_COL] == 0]
    df_1 = df[df[TARGET_COL] == 1]
    ax.scatter(df_0["CGPA"], df_0["Entrance_Score"], color="#e74c3c", alpha=0.6, s=35, label="Not Eligible (0)", edgecolors="none")
    ax.scatter(df_1["CGPA"], df_1["Entrance_Score"], color="#27ae60", alpha=0.6, s=35, label="Eligible (1)", edgecolors="none")
    ax.set_title("CGPA vs. Entrance Score by Admission Eligibility", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("CGPA (0 - 10.0 scale)", fontsize=9)
    ax.set_ylabel("Entrance Score (0 - 100 scale)", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Status", fontsize=8, loc="upper left")
    scatter_fig_path = FIGURES_DIR / "cgpa_vs_entrance_by_eligibility.png"
    plt.tight_layout()
    plt.savefig(scatter_fig_path)
    plt.close()
    print(f"Saved: {scatter_fig_path}")

    print("\nEDA Completed Successfully! All figures saved to outputs/figures/.")


if __name__ == "__main__":
    run_eda()
