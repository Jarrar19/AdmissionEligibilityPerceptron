# Ethics, Fairness & Limitations

**Project:** Perceptron-Based Admission Eligibility Prediction System (`admission-eligibility-perceptron`)  
**Context:** TAE Problem-Based Learning (PBL) — Neural Networks & Deep Learning  

---

## 1. Ethical Governance & Compliance

Machine learning applications in high-stakes domains like university admissions carry profound ethical responsibilities. An erroneous prediction can unfairly impact a student's educational trajectory. In adherence to Section 1 (Operating Contract) and Section 13 (Ethics & Limitations), this project enforces strict safeguards:

### 1.1 Strict Exclusion of Sensitive Personal Attributes (§1 Rule 6 & §6.6)
Under no circumstances are protected demographic characteristics incorporated into feature selection or training. Specifically, the model contains **zero features** representing:
- Name, Student Identifier, or institutional roll numbers
- Gender, sex, or sexual orientation
- Caste, tribe, race, or ethnic background
- Religion, belief systems, or political affiliation
- Residential address, postal code, or geographical origin
- Parental income, occupation, or wealth status

The model is strictly restricted to verifiable, academic performance indicators (`CGPA`, `Entrance_Score`, `Percentages`, `Interview_Score`, `Aptitude_Score`, `Attendance`, and `Backlogs`).

### 1.2 Data Privacy & Confidentiality (§6.6)
- **Zero Personally Identifiable Information (PII)**: The dataset is generated synthetically and contains no records corresponding to real living individuals.
- **Audit Compliance**: If applied to real institutional archives, all records must be fully anonymized, hashed, and stripped of tracking identifiers prior to modeling.

### 1.3 Algorithmic Transparency & Model Explainability
- The system rejects "black-box" predictions. The exact decision score $z = \sum w_i x_i + b$, along with feature-by-feature standardized contributions, is rendered in the user interface.
- Prediction thresholds and decision logic are fully disclosed.

---

## 2. Institutional Limitations & Academic Positioning (§2 & §13)

> [!WARNING]
> **Academic Demonstration Notice**: This software is developed purely as an educational demonstration of the single-layer Perceptron algorithm for an NNDL PBL course. It is **NOT** an automated admissions engine and must never be utilized to make binding admission decisions.

### 2.1 The Linearity Limitation
- The single-layer Perceptron is constrained to planar linear hyperplanes ($\mathbf{w}^T \mathbf{x} + b = 0$).
- Real university admissions involve intricate, non-linear trade-offs. For example, a student with exceptional national-level sports achievements or research publications might be admitted despite a lower entrance examination score. A single linear boundary cannot model such non-linear compensatory interactions.

### 2.2 Synthetic Data Constraints
- While carefully calibrated with Gaussian noise and realistic academic distributions, synthetic data cannot replicate the full sociological variance of real-world applicant populations.
- A test accuracy of **73.00%** on synthetic data does not imply identical performance on real institutional applicant pools.

### 2.3 Qualitative Holistic Factors
University admissions committees review holistic components that quantitative models cannot evaluate:
- Statements of Purpose (SOP) and personal motivation essays
- Letters of Recommendation (LOR) from research advisors and educators
- Demonstration of resilience, character, and overcoming personal adversity
- Diversity contributions to institutional culture and specialized programmatic quotas

### 2.4 Error Analysis & Impact (False Negatives vs. False Positives)
In our held-out test evaluation of 200 applicants:
- **False Positives ($FP = 35$)**: 35 applicants who were actually Not Eligible were predicted Eligible. In a real scenario, this would misdirect institutional resources toward unqualified candidates.
- **False Negatives ($FN = 19$)**: 19 eligible candidates were incorrectly predicted Not Eligible. In real admissions, false negatives represent the most harmful error—wrongfully denying qualified students access to higher education.

---

## 3. Human Oversight & Governance Principles

1. **Human-in-the-Loop**: Algorithmic predictions must only serve as preliminary screening aids; final admission decisions must always rest with qualified human admissions committees.
2. **Right to Explanation & Appeal**: Applicants have an ethical right to understand why an admission decision was reached and a mechanism to contest automated flags.
3. **Continuous Auditing**: Any production system must undergo recurring disparate impact audits to verify that automated thresholds do not systematically disadvantage specific student demographics.
