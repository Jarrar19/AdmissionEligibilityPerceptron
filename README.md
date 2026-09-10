# 🎓 Admission Eligibility Prediction System (Perceptron PBL)

[![Python 3.12.10](https://img.shields.io/badge/Python-3.12.10-blue.svg)](https://www.python.org/)
[![Model: Perceptron](https://img.shields.io/badge/Model-Perceptron%20(sklearn)-success.svg)](https://scikit-learn.org/)
[![Framework: Streamlit](https://img.shields.io/badge/UI-Streamlit%201.50-FF4B4B.svg)](https://streamlit.io/)
[![Tests: 20 Passed](https://img.shields.io/badge/Tests-20%20Passed%20(100%25)-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Problem-Based Learning (PBL) machine learning system for the **Neural Networks & Deep Learning (NNDL)** course. The system implements, evaluates, and deploys a single-layer binary **Perceptron** classifier to predict university admission eligibility from academic applicant profiles.

---

## 📌 Table of Contents
- [1. Overview & Problem Statement](#1-overview--problem-statement)
- [2. System Architecture](#2-system-architecture)
- [3. Dataset & Data Dictionary](#3-dataset--data-dictionary)
- [4. Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)
- [5. Machine Learning Pipeline](#5-machine-learning-pipeline)
- [6. Real Evaluation Results](#6-real-evaluation-results)
- [7. Streamlit Web Application](#7-streamlit-web-application)
- [8. Installation & Execution](#8-installation--execution)
- [9. Automated Testing](#9-automated-testing)
- [10. Limitations & Ethics](#10-limitations--ethics)
- [11. PBL Viva Preparation](#11-pbl-viva-preparation)

---

## 1. Overview & Problem Statement

### Academic Problem Statement
Develop and evaluate a supervised binary classification system utilizing Frank Rosenblatt's **Perceptron** algorithm (`sklearn.linear_model.Perceptron`) to predict whether an applicant is **Eligible (1)** or **Not Eligible (0)** for admission based on standardized academic features.

> [!WARNING]
> **Academic Demonstration Disclaimer (§10)**: This application is strictly an academic machine-learning demonstration and is **NOT** a substitute for an official university admission decision. Real admissions require institutional policy, human committee review, and non-linear holistic evaluations.

---

## 2. System Architecture

The project maintains a clean decoupling between the batch ML engineering pipeline and the real-time inference web application:

```
admission-eligibility-perceptron/
├── data/
│   ├── raw/admission_data.csv             # 1,000 synthetic student records
│   └── processed/cleaned_data.csv         # Verified & cleaned dataset snapshot
├── notebooks/
│   └── 01_eda.ipynb                       # Interactive EDA notebook
├── scripts/
│   ├── generate_dataset.py                # Controlled dataset generator
│   └── run_eda.py                         # Automated EDA & visualization script
├── src/
│   ├── config.py                          # Centralized paths, bounds, and hyperparameters
│   ├── data_loader.py                     # Schema validation & CSV loading
│   ├── preprocessing.py                   # Leakage-safe scaling & Pipeline builder
│   ├── train_model.py                     # Stratified training & pipeline persistence
│   ├── evaluate_model.py                  # Held-out metrics, weights, and comparison
│   └── predict.py                         # Range validation & standalone inference
├── models/
│   └── admission_perceptron_pipeline.joblib # Serialized Pipeline(StandardScaler, Perceptron)
├── outputs/
│   ├── figures/                           # High-res diagnostic plots
│   └── reports/                           # metrics.json & classification_report.txt
├── app/
│   └── app.py                             # Streamlit interactive web interface
├── tests/
│   ├── test_preprocessing.py              # Preprocessing unit tests
│   ├── test_model.py                      # Model persistence & weight tests
│   └── test_prediction.py                 # Full §11 test matrix
├── docs/                                  # Comprehensive specifications & reports
├── requirements.txt                       # Pinned dependencies
└── README.md                              # Master project documentation
```

---

## 3. Dataset & Data Dictionary

The project uses a documented, realistic synthetic dataset containing **1,000 records** (exceeding the §6.1 minimum floor of 500 records). 

### Public Dataset Evaluation & Rejection Rationale (§6.3)
Before generating data, public admission repositories were evaluated against §6.3 criteria. **UCI Dataset 582** (*Student Performance on an Entrance Examination*, 666 rows) was audited and rejected because:
1. **Target Mismatch**: Target was a 4-tier categorization (`Good`, `Vg`, `Average`, `Excellent`), not binary eligibility.
2. **Rule 6 Violation**: Heavily relied on sensitive demographic attributes (`Caste`, `Gender`, parental occupation) which are strictly prohibited.
3. **Feature Deficiencies**: Lacked continuous entrance scores, CGPA, interview ratings, and backlogs.
*(See [DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) for the full rejection audit).*

### Feature Dictionary Summary
- **Target**: `Eligible` (1 = Eligible, 0 = Not Eligible) | Class Balance: 555 Eligible (55.5%), 445 Not Eligible (44.5%).
- **Features (9 Inputs)**: `10th_Percentage`, `12th_Percentage`, `CGPA`, `Entrance_Score`, `Interview_Score`, `Aptitude_Score`, `Extracurricular_Score`, `Attendance`, `Previous_Backlogs`.
- **Identifier**: `Student_ID` (strictly excluded from predictive modeling).

---

## 4. Exploratory Data Analysis (EDA)

Key observations derived programmatically via `scripts/run_eda.py`:
- **Class Balance**: 55.5% Eligible vs 44.5% Not Eligible. Balanced distribution eliminates the need for artificial oversampling (SMOTE).
- **Strongest Positive Associations**: `Entrance_Score` ($r = 0.58$) and `CGPA` ($r = 0.55$) demonstrate the strongest linear correlation with admission eligibility.
- **Critical Disqualification Factor**: `Previous_Backlogs` ($r = -0.48$) strongly correlates with rejection.
- **Non-Linear Overlap**: Scatter plots show non-trivial overlap in borderline ranges (CGPA 7.2–8.0, Entrance 65–75). Because the Perceptron is a linear classifier, it cannot achieve 100% accuracy on this boundary.

---

## 5. Machine Learning Pipeline

1. **Zero Data Leakage**: An 80/20 stratified split divides the 1,000 records into 800 training samples and 200 held-out test samples.
2. **Unified Pipeline**: Feature scaling (`StandardScaler`) is fitted strictly on the 800 training records inside an end-to-end scikit-learn `Pipeline`:
   ```python
   Pipeline([
       ('scaler', StandardScaler()),
       ('perceptron', Perceptron(max_iter=1000, tol=1e-3, random_state=42))
   ])
   ```
3. **Perceptron Decision Rule**:
   $$z = \mathbf{w}^T \mathbf{x}_{\text{scaled}} + b = \sum_{i=1}^9 w_i x_i + b$$
   $$\hat{y} = 1 \quad \text{if } z \geq 0 \quad \text{else } 0$$
4. **Model Artifact**: Serialized to `models/admission_perceptron_pipeline.joblib`.

---

## 6. Real Evaluation Results

Every metric below is computed programmatically from the held-out test set of 200 samples (zero hard-coding):

| Metric | Perceptron (Primary Model) | Logistic Regression (Secondary Baseline) |
| :--- | :--- | :--- |
| **Accuracy** | **73.00%** | **76.00%** |
| **Precision** | **72.44%** | **76.47%** |
| **Recall** | **82.88%** | **81.98%** |
| **F1-Score** | **77.31%** | **79.13%** |

### Confusion Matrix (Test Split = 200 Samples)
- **True Negatives (TN)**: 54 (Correctly classified Not Eligible)
- **False Positives (FP)**: 35 (Incorrectly classified Eligible)
- **False Negatives (FN)**: 19 (Qualified applicants missed)
- **True Positives (TP)**: 92 (Correctly classified Eligible)

### Perceptron Learned Weights & Bias
- **Bias ($b$)**: `+2.0000`
- **`Entrance_Score`**: `+3.5826` (Strongest positive driver)
- **`Extracurricular_Score`**: `+2.4881`
- **`CGPA`**: `+1.5494`
- **`12th_Percentage`**: `+0.7692`
- **`Interview_Score`**: `+0.7587`
- **`Aptitude_Score`**: `-0.2341`
- **`10th_Percentage`**: `-0.6845`
- **`Attendance`**: `-2.3495`
- **`Previous_Backlogs`**: `-8.1285` (Heaviest rejection penalty)

---

## 7. Streamlit Web Application

The Streamlit app (`app/app.py`) provides an interactive interface for evaluating applicants:
- **Instant Pipeline Loading**: Loads `admission_perceptron_pipeline.joblib` without retraining.
- **Strict Input Validation**: Rejects invalid values (e.g., negative backlogs, CGPA > 10) with descriptive messages.
- **Prominent Decision Cards**: Displays green for `ELIGIBLE` and red for `NOT ELIGIBLE`.
- **Decision Transparency**: Displays the linear score $z$ and a full feature contribution table ($w_i \cdot x_i$).
- **Academic Disclaimer Callout**: Prominently displayed across the header.

---

## 8. Installation & Execution

### Prerequisites
- Python 3.12.10 (Host environment runtime)
- Git

### Step-by-Step Setup

1. **Clone the Repository**:
   ```bash
   git clone <repo-url>
   cd NNDL
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate Dataset & Run EDA**:
   ```bash
   python scripts/generate_dataset.py
   python scripts/run_eda.py
   ```

4. **Train & Evaluate the Perceptron**:
   ```bash
   python src/train_model.py
   python src/evaluate_model.py
   ```

5. **Launch the Streamlit Web Application**:
   ```bash
   streamlit run app/app.py
   ```

---

## 9. Automated Testing

The project includes an automated test suite verifying all 8 cases in the §11 verification matrix:
```bash
pytest -v tests/
```
**Test Results**: 20 tests passed out of 20 (100% pass rate). Detailed results documented in [docs/TESTING.md](docs/TESTING.md).

---

## 10. Limitations & Ethics

- **Linearity Bound**: The single-layer Perceptron can only construct planar linear decision boundaries. It cannot model complex non-linear combinations or qualitative achievements.
- **Synthetic Distribution**: Synthetic data cannot capture the sociological diversity of real-world applicant pools.
- **Fairness & Privacy**: No demographic or sensitive attributes (caste, gender, religion, address) are used in modeling (§1 Rule 6).
- **Human Oversight**: Algorithmic predictions must serve only as screening aids; human admissions committees retain ultimate decision authority.
*(See [docs/ETHICS_AND_LIMITATIONS.md](docs/ETHICS_AND_LIMITATIONS.md) for the full review).*

---

## 11. PBL Viva Preparation

### Key Viva Concepts & Answers
1. **Q: What is a Perceptron?**  
   *A:* A single-layer artificial neural computing unit for binary classification. It computes a linear weighted sum $z = \mathbf{w}^T \mathbf{x} + b$ and passes it through a step function ($\hat{y} = 1$ if $z \ge 0$, else 0).
2. **Q: How does the Perceptron update its weights?**  
   *A:* On misclassification: $w_i \leftarrow w_i + \eta(y - \hat{y})x_i$ and $b \leftarrow b + \eta(y - \hat{y})$. If classification is correct, no update occurs.
3. **Q: Why is feature scaling mandatory for Perceptron?**  
   *A:* Because weight updates are directly proportional to raw feature values $x_i$. Without `StandardScaler`, large-magnitude features (like Entrance Score 0–100) dominate updates while small-magnitude features (like CGPA 0–10) are ignored.
4. **Q: Why did your model achieve 73% test accuracy rather than 95%+?**  
   *A:* Real admission boundaries feature overlapping borderline cases that are not linearly separable. By Novikoff's theorem, a Perceptron cannot achieve 100% on non-linearly separable data. Per §1 Rule 3, reporting 73.00% is academically honest and authentic.
5. **Q: Why was Perceptron kept when Logistic Regression scored higher (76%)?**  
   *A:* Per §1 Rule 1, the core educational objective of this NNDL PBL project is to study the foundational Perceptron unit. Secondary models serve solely as educational benchmarks.
