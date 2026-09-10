# Project Requirements Specification

**Project:** Perceptron-Based Admission Eligibility Prediction System (`admission-eligibility-perceptron`)  
**Specification Version:** 1.0 (Aligned with Master PBL Specification)  
**Target Environment:** Python 3.12.10 (see reproducibility note)

---

## 1. Functional Requirements (FR)

### FR-01: Dataset Acquisition & Loading
- **FR-01.1**: The system shall acquire or generate a structured dataset containing at least 500 records (measured size: 1,000 records).
- **FR-01.2**: The dataset must include 9 candidate academic features (`10th_Percentage`, `12th_Percentage`, `CGPA`, `Entrance_Score`, `Interview_Score`, `Aptitude_Score`, `Extracurricular_Score`, `Attendance`, `Previous_Backlogs`) and one binary target (`Eligible`).
- **FR-01.3**: The data loader module (`src/data_loader.py`) shall verify file existence, schema completeness, zero null values, and target validity.

### FR-02: Preprocessing & Data Leakage Prevention
- **FR-02.1**: The preprocessing module (`src/preprocessing.py`) shall strictly separate non-predictive identifiers (`Student_ID`) from the feature matrix $X$.
- **FR-02.2**: The data split must follow an 80/20 train/test ratio with stratification (`stratify=y`) and fixed random state (`random_state=42`).
- **FR-02.3**: Feature normalization (`StandardScaler`) must be fitted solely on the training partition within a unified scikit-learn `Pipeline` to prevent data leakage.

### FR-03: Exploratory Data Analysis (EDA)
- **FR-03.1**: The system shall compute descriptive summary statistics and report exact target class distributions (counts and percentages).
- **FR-03.2**: Automated visualization scripts (`scripts/run_eda.py`) and notebook (`notebooks/01_eda.ipynb`) shall generate correlation heatmaps, feature distribution grids, and class overlap scatter plots.
- **FR-03.3**: All observations must be reported in plain language without claiming causal relationships from statistical correlations (§7).

### FR-04: Primary Model Training
- **FR-04.1**: The system must train a single-layer binary Perceptron (`sklearn.linear_model.Perceptron`) as the primary classifier.
- **FR-04.2**: Hyperparameters must be centralized: `max_iter=1000`, `tol=1e-3`, `random_state=42`, `eta0=1.0`, `fit_intercept=True`.
- **FR-04.3**: Model training shall never use the test set or alter training data to artificially manipulate scores.

### FR-05: Model Persistence
- **FR-05.1**: The trained pipeline (StandardScaler + Perceptron) must be serialized to disk as `models/admission_perceptron_pipeline.joblib`.
- **FR-05.2**: The persisted pipeline must be self-contained and reproducible without requiring retraining upon application startup.

### FR-06: Programmatic Evaluation & Metrics
- **FR-06.1**: The evaluation module (`src/evaluate_model.py`) shall compute Accuracy, Precision, Recall, F1-Score, and a $2 \times 2$ Confusion Matrix from the held-out test split.
- **FR-06.2**: Model interpretation must extract the Perceptron weight vector $\mathbf{w}$ and scalar bias $b$, visualizing relative feature contributions.
- **FR-06.3**: An optional secondary benchmark (Logistic Regression) shall be trained on the identical split for comparison, keeping Perceptron as primary.

### FR-07: Standalone Prediction & Input Validation
- **FR-07.1**: The prediction module (`src/predict.py`) must strictly validate candidate inputs against domain boundaries (e.g. CGPA $\in [0, 10]$, % $\in [0, 100]$, backlogs $\ge 0$).
- **FR-07.2**: Out-of-range or missing inputs must be blocked with informative error messages.
- **FR-07.3**: The inference output must return human-readable status (`ELIGIBLE` or `NOT ELIGIBLE`), raw decision score $z$, and feature contribution breakdown.

### FR-08: Streamlit Web Interface
- **FR-08.1**: The web application (`app/app.py`) shall provide an intuitive, responsive interface with bounded numeric inputs.
- **FR-08.2**: A prominent, non-dismissible **Academic Disclaimer** (§10) must be displayed on screen.
- **FR-08.3**: Predictions must be displayed as prominent status cards with a collapsible technical diagnostics section showing weights and test metrics.

---

## 2. Non-Functional Requirements (NFR)

- **NFR-01: Reproducibility**: All random operations, splits, and algorithms must use `RANDOM_STATE = 42`. The runtime and library versions must be pinned.
- **NFR-02: Zero Data Leakage**: No preprocessing or scaling parameters from test data may inform training.
- **NFR-03: Efficiency & Modularity**: The web application must load pre-trained model weights instantaneously with zero retraining on page reloads.
- **NFR-04: Ethical Integrity**: No sensitive personal demographic variables (caste, gender, religion, race, addresses) may be collected, stored, or modeled.
- **NFR-05: Test Coverage**: 100% of the §11 verification cases must be covered by automated unit tests and pass cleanly.
