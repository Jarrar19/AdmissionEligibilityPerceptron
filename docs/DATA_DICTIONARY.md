# Data Dictionary & Dataset Specification

**Project:** Perceptron-Based Admission Eligibility Prediction System (`admission-eligibility-perceptron`)  
**Artifact Location:** `data/raw/admission_data.csv`  
**Dataset Nature:** Documented Academic Synthetic Dataset (§6.2 & §6.3)  
**Total Records:** 1,000 (Satisfies §6.1 floor of $\ge 500$ records)  
**Class Distribution:** 555 Eligible (55.5%), 445 Not Eligible (44.5%)  

---

## 1. Feature Dictionary Table (§6.5)

| Column Name | Data Type | Valid Range | Role | Model Input | Description | Reason for Inclusion / Exclusion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Student_ID`** | String | `STU_1000`–`STU_1999` | Identifier | **No** | Unique student tracking identifier. | Excluded per §1 Rule 6 & §6.4 to prevent data leakage and arbitrary identifier memorization. |
| **`10th_Percentage`** | Float | 0.0 – 100.0% | Feature | **Yes** | Secondary school cumulative percentage. | Foundational academic consistency indicator. |
| **`12th_Percentage`** | Float | 0.0 – 100.0% | Feature | **Yes** | Higher secondary school percentage. | Direct prerequisite qualification indicator. |
| **`CGPA`** | Float | 0.0 – 10.0 | Feature | **Yes** | Cumulative Grade Point Average in undergraduate/diploma. | Primary metric of academic competence and coursework rigor. |
| **`Entrance_Score`** | Float | 0.0 – 100.0 | Feature | **Yes** | Standardized admission entrance examination score. | Standardized benchmark evaluating applicant subject aptitude. |
| **`Interview_Score`** | Float | 0.0 – 100.0 | Feature | **Yes** | Technical and personal interview evaluation score. | Assessment of domain grasp, problem solving, and communication. |
| **`Aptitude_Score`** | Float | 0.0 – 100.0 | Feature | **Yes** | Quantitative, logical, and verbal reasoning score. | Evaluates core problem-solving capability under timed conditions. |
| **`Extracurricular_Score`**| Float | 0.0 – 100.0 | Feature | **Yes** | Assessment of sports, leadership, arts, and community contributions. | Holistic evaluation of applicant's non-academic achievements. |
| **`Attendance`** | Float | 0.0 – 100.0% | Feature | **Yes** | Historical classroom attendance percentage. | Proxy for discipline, regularity, and engagement. |
| **`Previous_Backlogs`** | Integer | 0 – 20 | Feature | **Yes** | Total count of previously uncleared course backlogs. | Critical indicator of academic difficulty or credit deficiencies. |
| **`Eligible`** | Integer | {0, 1} | Target | **No** | Admission eligibility status (1 = Eligible, 0 = Not Eligible). | Supervised classification ground truth target. |

---

## 2. Target Definition & Generation Methodology (§5 & §6.3)

In accordance with §5, the target variable represents binary admission eligibility:
- **`1` = Eligible**
- **`0` = Not Eligible**

Per §6.3, to ensure academic integrity and prevent the Perceptron from "discovering" an undisclosed trivial heuristic, the generation rules are fully disclosed:

### Composite Academic Index Formula
Each candidate's academic profile is evaluated through a weighted multi-factor composite index:
$$\text{Composite Index} = 0.25 \cdot \left(\frac{\text{CGPA}}{10}\right) + 0.25 \cdot \left(\frac{\text{Entrance\_Score}}{100}\right) + 0.15 \cdot \left(\frac{\text{Interview\_Score}}{100}\right) + 0.10 \cdot \left(\frac{\text{Aptitude\_Score}}{100}\right) + 0.10 \cdot \left(\frac{\text{12th\_Percentage}}{100}\right) + 0.05 \cdot \left(\frac{\text{10th\_Percentage}}{100}\right) + 0.05 \cdot \left(\frac{\text{Extracurricular\_Score}}{100}\right) + 0.05 \cdot \left(\frac{\text{Attendance}}{100}\right) - 0.08 \cdot \text{Previous\_Backlogs}$$

### Non-Triviality & Realistic Boundary Variations
1. **Unobserved Nuance (Gaussian Noise)**: Admission committee subjective reviews are modeled via added Gaussian noise ($\epsilon \sim \mathcal{N}(0, 0.065)$). This introduces natural overlap along class boundaries so the dataset is not trivially linearly separable.
2. **Hard Disqualification Criteria**:
   - Candidates with $\ge 3$ uncleared backlogs or attendance $< 65.0\%$ receive a severe net penalty ($-0.20$), reflecting mandatory institutional eligibility cutoffs.
3. **Decision Cutoff**: Candidates with a net score $\ge 0.730$ are assigned `Eligible = 1`; all others are assigned `Eligible = 0`.
4. **Resulting Distribution**: Exactly 555 Eligible (55.5%) and 445 Not Eligible (44.5%), creating a balanced, realistic, and challenging classification challenge for a single-layer Perceptron.

---

## 3. Public Dataset Evaluation & Rejection Log (§6.2 & §6.3)

Prior to adopting the synthetic dataset generation strategy, public admission repositories were systematically audited against the acceptance criteria outlined in §6.3.

### Evaluation of UCI Dataset ID 582
- **Dataset Title**: *Student Performance on an Entrance Examination*
- **Source**: UCI Machine Learning Repository (Prof. Jiten Hazarika, Assam Medical Entrance Examination, 2018)
- **Record Count**: 666 records
- **Audit Date**: 2026-09-04

### Acceptance Criteria Audit Matrix (§6.3)

| Criterion | Requirement (§6.3) | UCI Dataset 582 Empirical Audit | Status |
| :--- | :--- | :--- | :--- |
| **Minimum Size (§6.1)** | Minimum 500 records | 666 records | **PASS** |
| **Binary Target (§5)** | Clear binary admission decision: `1` = Eligible, `0` = Not Eligible | Target is `Performance` with 4 categorical bands: `Good` (210), `Vg` (198), `Average` (157), and `Excellent` (101). Not an admission eligibility status. | **FAIL** |
| **No Sensitive Attributes (§1 Rule 6 & §6.6)** | Prohibits gender, caste, race, religion, parental demographics | Contains explicit columns: `Gender` (`male`/`female`), `Caste` (`General`, `OBC`, `SC`, `ST`), `Father_occupation`, and `Mother_occupation`. | **FAIL (Critical Rule Violation)** |
| **Candidate Academic Features (§6.4)** | Requires CGPA, continuous entrance score, interview, aptitude, attendance, backlogs | Lacks CGPA, interview score, aptitude score, attendance, and backlogs. After dropping sensitive demographic columns, only 2 continuous percentages and school board metadata remain. | **FAIL** |
| **Domain Suitability (§2)** | General university/college admission eligibility | Medical exam performance rank band in a specific state entrance examination (Assam CEE). | **FAIL** |

### Rejection Summary
UCI Dataset 582 was rejected because:
1. Using it would violate **Non-negotiable Rule #6** (prohibiting caste, gender, and socio-demographic features in predictive models).
2. The target variable is an examination score tier, not a binary admission decision. Arbitrarily grouping classes would introduce uncontrolled subjective proxy bias.
3. The remaining academic feature space is severely impoverished, failing the candidate feature architecture required by §6.4.

Consequently, per §6.2 step 3, the synthetic dataset generation pipeline was invoked, providing a fully transparent, bias-free, and mathematically documented alternative.

---

## 4. Privacy & Ethical Compliance Statement (§6.6)

1. **Zero Personally Identifiable Information (PII)**: No student names, email addresses, physical locations, phone numbers, or institutional IP addresses exist in the data.
2. **Zero Sensitive Attributes**: Features strictly capture academic track records and objective performance metrics. Caste, gender, socio-economic status, and religion are completely absent.
3. **Transparent Labeling**: In compliance with §1 Rule 7, this dataset is labeled as **Synthetic** across all documentation, visualizations, tests, and the Streamlit user interface.
