# TAE PBL — Master Project Specification
## Perceptron-Based Admission Eligibility Prediction System

**Primary Model:** Perceptron Binary Classifier · **App:** Streamlit · **Language:** Python 3.12.10 (see reproducibility note)
**Repo name:** `admission-eligibility-perceptron`

---

## 1. Agent Operating Contract

This document is the single source of truth. Read it fully before writing code. Treat every requirement as authoritative; ask for clarification only when something is genuinely impossible or contradictory. At the end, provide an implementation summary (what was built, final dataset, final feature list, target definition, Perceptron config, actual evaluation results, files created, tests run, how to run it, remaining limitations). Do not declare the project complete until §14 (Definition of Done) is verified end-to-end.

**Non-negotiable rules, in force everywhere in this document:**

1. Perceptron (`sklearn.linear_model.Perceptron`) is the primary model. Never replace it with Logistic Regression, a Decision Tree, Random Forest, XGBoost, SVM, or a neural network — even if one scores higher. Those may appear only as an optional, clearly secondary comparison (§8.4).
2. Never fabricate data, metrics, screenshots, or results. Every number in the app/report must come from an actual run.
3. Never set a target accuracy in advance (not 75%, not 90%, not 99%) and never manipulate data, labels, the test set, or the algorithm to hit a number. Report whatever the model actually achieves.
4. Never train on the test set, remove hard test samples after seeing predictions, or repeatedly tune on the final test set and report it as unbiased.
5. No data leakage: split before fitting any transform; fit scalers on training data only.
6. Never use sensitive personal attributes (name, phone, email, address, religion, caste, race/ethnicity, political affiliation) or identifiers (`Student_ID`) as predictive features.
7. If synthetic data is used, label it as synthetic everywhere it appears — README, app, docs — and document how it was generated.
8. Save the trained pipeline; never retrain on every Streamlit launch.
9. Don't overengineer or add technologies outside §9's stack without a documented reason. Keep code modular, readable, and explainable by a student in a viva.
10. Document non-obvious design decisions as you make them; don't mark a requirement done until it's actually verified.

---

## 2. Overview, Positioning & Problem Statement

The project predicts whether a student is **eligible (1)** or **not eligible (0)** for admission, from academic/admission-related features, using a Perceptron trained on historical or synthetic records, served through a Streamlit UI with real evaluation metrics attached.

**Formal problem statement:** develop and evaluate a Perceptron-based binary classifier that predicts admission eligibility from relevant student features.

**This is an academic demonstration — not**: a real university admission authority, a legally binding decision system, a replacement for an admission committee, a guaranteed predictor, or a production platform. Real admissions involve institutional policy, human review, additional criteria, and legal/fairness constraints a small academic dataset can't capture. State this in the docs *and* in the running app (§10).

**Expected result:** a Perceptron is a linear classifier — it may not perfectly separate the classes, and that's fine. Good test accuracy on a small/synthetic dataset also doesn't imply real-world validity; say so in the limitations section (§8.3, §13).

Two parts: **(A) ML pipeline** — dataset → validation → cleaning → EDA → feature selection → preprocessing → Perceptron training → evaluation → persistence. **(B) prediction app** — student input → validation → saved pipeline → Eligible/Not Eligible.

---

## 3. Objectives

Understand supervised binary classification and the Perceptron algorithm; obtain and validate a suitable dataset; run EDA; select and preprocess features correctly; split data without leakage; train and evaluate the Perceptron; persist the model+pipeline; build and validate a Streamlit predictor; test the full system; document and present it for viva.

---

## 4. Scope

**In scope:** dataset acquisition/validation/cleaning, EDA, feature selection, preprocessing/scaling, train/test split, Perceptron training, evaluation + confusion matrix, model persistence, prediction + input validation, Streamlit UI, basic model interpretation, testing, Git/GitHub, documentation, PBL presentation material.

**Out of scope:** a real admissions management system, payment processing, student registration/auth, production databases, cloud infra, microservices, Docker/Kubernetes, deep learning, LLMs, enterprise architecture, or automated real-world admission decisions. This is an ML PBL prototype, not an enterprise app.

---

## 5. Target Definition

```text
1 = Eligible        0 = Not Eligible
```
Always show the human-readable label to end users — never a bare 0/1.

---

## 6. Dataset

### 6.1 Size
**The final dataset must contain at least 500 records — no exceptions.** Preferably 1,000+. This is a hard floor, not a target: if a public dataset falls short, use the synthetic fallback (§6.2) specifically so this is always achievable. The size must be *measured* from the actual dataset, never claimed.

### 6.2 Selection strategy
1. **Search first** for a suitable public dataset (student/graduate admission, college admission prediction, student eligibility). **A public dataset with fewer than 500 records must not be used as the final dataset.**
2. **Check it against the acceptance criteria** below.
3. **If no suitable public dataset with at least 500 records is found, generate a clearly documented synthetic dataset containing at least 500 records** — don't let searching block development indefinitely.

### 6.3 Acceptance criteria (a dataset — real or synthetic — must satisfy all of these)
- At least 500 records (see §6.1); a meaningful binary target (or a documented method to define one); relevant academic/admission features; sufficient variation; no required sensitive attributes; identifiers not used as features; target definition and source/generation method documented; missing values and class distribution analyzable.
- **The target must not be trivially leaked.** If a synthetic target is rule-based (e.g. `Eligible = Entrance_Score > 60`), that rule must be disclosed — don't let the model "discover" a relationship you built in without saying so. Include realistic variation and some borderline/noisy cases so performance is meaningful, not a rounding artifact.

### 6.4 Candidate features (final set doesn't need every row — inspect the actual data and document what's used)

| Feature | Type | Role |
|---|---|---|
| Student_ID | String | Identifier only — never a model input |
| 10th_Percentage, 12th_Percentage | Float | Feature |
| CGPA | Float | Feature |
| Entrance_Score, Interview_Score, Aptitude_Score, Extracurricular_Score | Float | Feature |
| Attendance | Float | Feature |
| Previous_Backlogs | Integer | Feature |
| Eligible | Integer | Target |

### 6.5 Data dictionary (`docs/DATA_DICTIONARY.md`)
For every final column: name, dtype, description, valid range, role, whether it's a model input, and reason for inclusion/exclusion.

| Column | Type | Range | Role | Model Input |
|---|---|---|---|---|
| CGPA | Float | 0–10 | Feature | Yes |
| Entrance_Score | Float | 0–100 | Feature | Yes |
| Student_ID | String | — | Identifier | No |
| Eligible | Integer | 0/1 | Target | No |

### 6.6 Privacy
Never use real names, phone numbers, emails, addresses, passwords, or other sensitive attributes. If a public dataset contains them, strip them from the modeling workflow and never commit them to GitHub.

---

## 7. EDA & Class Imbalance

**EDA must cover:** dataset overview (rows, columns, dtypes, missing values, duplicates); target analysis (class counts + percentages); feature analysis (CGPA, entrance score, percentages, interview score, attendance, backlogs); relationship analysis (plots/correlation). Write findings in plain language (e.g. *"eligible students generally have higher entrance scores, with overlap between classes"*) — never claim causation from correlation.

**Class imbalance:** measure and report the class distribution (count + %) before training. Use **stratified** train/test splitting to preserve proportions — but stratification is *not* a balancing method; don't conflate the two. Don't automatically apply SMOTE or over/undersampling; that's an optional, separately-justified extension (§15), not baseline scope.

---

## 8. Preprocessing, Model & Evaluation

### 8.1 Pipeline
```text
Raw Dataset → Validation → Separate Features/Target → Train/Test Split (80/20, random_state=42, stratified)
   → Fit preprocessing (StandardScaler) on TRAINING data only → Transform train and test
   → Train Perceptron
```
Use one scikit-learn `Pipeline` object end to end so training and inference preprocessing can never drift apart. Document missing-value strategy (only if missingness exists) and the duplicate-handling decision — don't auto-delete without checking whether duplicates are legitimate repeats.

### 8.2 Model
```python
Perceptron(max_iter=1000, tol=1e-3, random_state=42)
```
Starting point only — final hyperparameters must be documented after experimentation. Save the complete pipeline to `models/admission_perceptron_pipeline.joblib`.

### 8.3 Methodology (must be documented and viva-ready)
Weighted sum: \( z = \sum w_i x_i + b \). Step decision: \(\hat y = 1\) if \(z \geq 0\) else \(0\). Update on misclassification: \(w_i \mathrel{+}= \eta(y-\hat y)x_i\), \(b \mathrel{+}= \eta(y-\hat y)\). Explain that this is a **linear** classifier: it may struggle with nonlinear relationships, dataset quality strongly affects it, and a good test score is not a guarantee of real-world validity.

### 8.4 Evaluation
From the held-out test set, computed programmatically — never hard-coded:
\[
Accuracy=\tfrac{TP+TN}{TP+TN+FP+FN},\quad Precision=\tfrac{TP}{TP+FP},\quad Recall=\tfrac{TP}{TP+FN},\quad F1=2\cdot\tfrac{P\cdot R}{P+R}
\]
plus the confusion matrix. Explain what each value means in the admission context.

**Model interpretation:** show feature names, learned weights, and bias — and state plainly that a weight shows contribution to the model's decision function, not causation.

**Optional cross-validation:** only during development; the held-out test set stays untouched until final evaluation; CV results never replace it.

**Optional model comparison** (e.g. vs. Logistic Regression / Decision Tree): allowed as a secondary table with real results, but Perceptron stays the reported/final model regardless of which scores higher — explain *why* Perceptron fits the educational goal.

---

## 9. Technology Stack & Config

- **Python 3.12.10 (see reproducibility note)** (pinned for host environment reproducibility; see §13). **ML:** scikit-learn (Perceptron, Pipeline, StandardScaler, split, metrics). **Data:** pandas, numpy. **Viz:** matplotlib (+ seaborn, optional). **UI:** Streamlit. **Persistence:** joblib. **Dev:** VS Code, venv, Jupyter (optional for EDA). **VCS:** Git + GitHub.
- `requirements.txt`: only what's used — typically pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, joblib. Pin versions once tested. Do not add TensorFlow, PyTorch, XGBoost, Flask, React, Node.js unless scope is explicitly changed.
- Avoid magic numbers — centralize `RANDOM_STATE = 42`, `TEST_SIZE = 0.20`, and any UI validation ranges in one config source, not duplicated across files.
- **Error handling:** clear user-facing messages ("Admission dataset was not found…", "Trained model not found…", "Please enter valid values…"). Never use bare `except: pass`.

---

## 10. Streamlit Application

**Page structure:** header ("Admission Eligibility Prediction — Perceptron-Based ML System") → brief intro (what it does, what a Perceptron is, what the prediction means) → **academic disclaimer** ("This application is an academic machine-learning demonstration and is not a substitute for an official admission decision.") → input controls for only the features the trained model actually uses → "Predict Admission Eligibility" button → prominent result ("Admission Status: ELIGIBLE / NOT ELIGIBLE") → optional collapsible model info (model type, dataset size, test accuracy/metrics) — don't clutter the main view with technical detail.

**Input validation:** use the dataset's actual final ranges (e.g. CGPA 0–10, percentages 0–100, backlogs ≥ 0); reject invalid values with a specific message ("Please enter a CGPA between 0 and 10.") — never let invalid input reach the model.

**App is correct when:** it starts without errors; loads the saved pipeline without retraining; shows purpose + disclaimer + correct input fields; validates input; predicts via the saved pipeline; displays a clear label; shows a clean message (not a stack trace) on invalid input.

---

## 11. Testing

| Case | Input | Expected |
|---|---|---|
| Dataset validation | — | Loads; required columns present; target valid |
| Valid input | realistic values | Pipeline runs; returns Eligible/Not Eligible |
| Invalid CGPA | 15 | Validation error |
| Invalid percentage | 120 | Validation error |
| Negative backlogs | -2 | Validation error |
| Missing input | blank field | Clear validation message |
| Boundary values | CGPA 0/10, % 0/100, attendance 0/100 | Accepted (valid boundaries) |
| Saved model | restart app | Loads pipeline without retraining |

Log results in `docs/TESTING.md` as a table (ID, description, input, expected, actual, pass/fail) — mark "pass" only after actually running it.

---

## 12. Architecture, Modules & Folder Structure

```text
Dataset → Validation → Cleaning → EDA → Feature Selection → Train/Test Split (80/20)
   → Preprocessing (StandardScaler) → Perceptron → { Evaluation (metrics+matrix) , Saved Pipeline (.joblib) }
Saved Pipeline → Streamlit App → Student Input → Input Validation → Saved Pipeline → Eligible / Not Eligible
```

| Module | File | Responsibility |
|---|---|---|
| Data Loader | `src/data_loader.py` | Locate, load, verify the CSV |
| Preprocessing | `src/preprocessing.py` | Schema/range validation, cleaning, leakage-safe pipeline build |
| EDA | `notebooks/01_eda.ipynb` | Stats, distributions, correlation, written observations |
| Training | `src/train_model.py` | Split, build pipeline, train, save, log config |
| Evaluation | `src/evaluate_model.py` | Test-set metrics, confusion matrix, saved reports |
| Prediction | `src/predict.py` | Load pipeline, validate/prepare input, predict, human-readable output |
| App | `app/app.py` | UI, validation, prediction, disclaimer, result display |

```text
admission-eligibility-perceptron/
├── data/{raw/admission_data.csv, processed/cleaned_data.csv}
├── notebooks/01_eda.ipynb
├── src/{data_loader,preprocessing,train_model,evaluate_model,predict}.py
├── models/admission_perceptron_pipeline.joblib
├── outputs/{figures/*.png, reports/{classification_report.txt, metrics.json}}
├── app/app.py
├── tests/{test_preprocessing,test_model,test_prediction}.py
├── docs/{PROJECT_REQUIREMENTS,SYSTEM_DESIGN,DATA_DICTIONARY,ML_METHODOLOGY,TESTING,ETHICS_AND_LIMITATIONS}.md
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE
```
May be simplified, but keep responsibilities separated.

---

## 13. Documentation, Ethics & Limitations

**`README.md`** covers: title, overview, problem statement, objectives, features, tech stack, dataset source & size, target/feature definitions, architecture, folder structure, install/setup, training/evaluation/app instructions, example prediction, actual results, limitations, ethics, future scope, team info. Only verified results.

**`docs/`** files (`PROJECT_REQUIREMENTS`, `SYSTEM_DESIGN`, `DATA_DICTIONARY`, `ML_METHODOLOGY`, `TESTING`, `ETHICS_AND_LIMITATIONS`) must match the actual implementation — never describe a feature that doesn't exist. `SYSTEM_DESIGN.md` must additionally include: **(a)** a requirements traceability table mapping each FR to its implementing file and how it's verified (e.g. FR-01 Dataset loading → `data_loader.py` → test), and **(b)** a short risk register (risk / impact / mitigation) covering at least dataset quality, class imbalance, data leakage, low Perceptron accuracy, invalid input, and a missing model file.

**Ethics (`ETHICS_AND_LIMITATIONS.md`)**: privacy (data anonymous/synthetic), bias (the model may learn dataset biases), fairness (no sensitive attributes used to boost accuracy), transparency (limitations stated clearly), human oversight (real admissions need institutional policy + human review).

**Limitations to state explicitly:** Perceptron is linear and may miss nonlinear relationships; dataset quality bounds model quality; synthetic data (if used) doesn't represent a real population; test-set performance ≠ real-world guarantee; no institutional validation or formal fairness audit performed unless separately added; predictions are not guaranteed outcomes.

**Reproducibility:** document Python/package versions, dataset source/generation method, random state, test size, feature list, target definition, preprocessing, and hyperparameters — enough for someone else to reproduce the project from the README alone.

> [!NOTE]
> **Reproducibility Note on Python Version:** The reference specification originally cited Python 3.11. The project runtime is executed and pinned on Python 3.12.10 (64-bit) to align with the active host environment. The core machine learning stack (scikit-learn 1.5.2, numpy, pandas) maintains full deterministic reproducibility and API parity across Python 3.11 and 3.12 with `random_state=42`.

**Git:** meaningful, incremental commit messages (not "update"/"final"/"test123"); `.gitignore` covers `.venv/`, `venv/`, `__pycache__/`, `*.pyc`, `.env`, `.DS_Store`; never commit secrets, keys, or personal data.

---

## 14. Definition of Done

- [ ] Dataset: exists, source/size documented, data dictionary complete, target + feature list documented, no sensitive/identifier features.
- [ ] Processing: validation, cleaning, feature selection, and a leakage-safe preprocessing pipeline all work; class distribution reported.
- [ ] EDA: overview, target/feature analysis, correlation analysis, and written findings completed.
- [ ] Model: Perceptron trained, hyperparameters documented, pipeline saved and loadable.
- [ ] Evaluation: accuracy/precision/recall/F1/confusion matrix computed from real held-out predictions, not hard-coded.
- [ ] App: starts, loads saved pipeline without retraining, validates input, predicts, displays a clear labeled result + disclaimer, handles errors gracefully.
- [ ] Testing: all cases in §11 run and logged in `docs/TESTING.md`.
- [ ] Documentation: README + all `docs/` files complete and accurate, including ethics/limitations/future scope.
- [ ] Repository: `.gitignore` set, no secrets/personal data, meaningful commit history, project reproducible from README.

**Before declaring completion, run the full workflow once end to end:** environment → dataset → validation → preprocessing → EDA → training → save pipeline → evaluate on test set → generate metrics/confusion matrix → run automated tests → start Streamlit → valid prediction → invalid-input handling → restart app and confirm no retraining → review this checklist.

---

## 15. PBL Demonstration, Viva & Future Scope

**Demonstration should show:** problem statement, dataset, data dictionary, EDA, preprocessing, Perceptron theory, training, evaluation metrics + confusion matrix, saved model, the Streamlit app (valid + invalid input), architecture, GitHub repo, limitations and future scope. A 15-slide deck works well: title → problem → objectives → existing-vs-proposed → dataset → preprocessing → EDA → Perceptron theory → architecture → implementation → results → app screenshots → limitations/ethics → future scope → conclusion.

**Every team member should be able to explain:** supervised learning, classification, features/target, train/test data, data leakage, overfitting; what a Perceptron is, weights, bias, step function, learning rate, the update rule, why scaling matters, and why it can fail on nonlinear data; accuracy/precision/recall/F1/confusion matrix and TP/TN/FP/FN; and this project's specific dataset, size, features, preprocessing, split, hyperparameters, actual final metrics, UI workflow, limitations, and future scope.

**Future scope** (explicitly *not* required for the baseline): larger real-world anonymized datasets, cross-validation, additional model comparisons, class-imbalance techniques (SMOTE etc.), fairness analysis, deeper explainability, model monitoring, institutional validation, deployment.

---

## 16. Priority & Philosophy

```text
Correctness > Academic validity > Explainability > Reproducibility > Data integrity > Clean architecture > UX > Performance > Extra features > Complexity
```
Never sacrifice correctness for looks, explainability for complexity, or data integrity for a higher score.

> The goal is not the most sophisticated admission predictor possible — it's to demonstrate how a Perceptron can be used **responsibly and correctly** for binary classification on structured data, from dataset preparation through a working, honestly-evaluated prediction interface. **Never make the project look better than it actually is.**
