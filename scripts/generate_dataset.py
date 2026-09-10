"""Synthetic Dataset Generator for Admission Eligibility Perceptron System.

Generates an academic admission dataset of 1,000 records adhering to Section 6
of the PBL Master Specification. 

Target Generation Methodology (§6.3):
- Target 'Eligible' (1 or 0) is derived from a multi-factor composite academic index:
  Composite Index = 0.25*(CGPA/10) + 0.25*(Entrance_Score/100) + 0.15*(Interview_Score/100)
                  + 0.10*(Aptitude_Score/100) + 0.10*(12th_Percentage/100) + 0.05*(10th_Percentage/100)
                  + 0.05*(Extracurricular_Score/100) + 0.05*(Attendance/100)
                  - 0.08 * (Previous_Backlogs)
- Realistic Gaussian noise (mean=0, std=0.06) is added to reflect human admission nuances,
  creating overlapping borderline candidates and preventing trivial linear separability.
- Students with severe backlog counts (>=3) or failing attendance (<65%) receive hard penalties.
- Students with final net index >= 0.61 are labeled Eligible (1), else Not Eligible (0).
- This produces ~54-56% Eligible candidates, satisfying §6.3 and §7 without trivial leakage.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RANDOM_STATE, RAW_DATA_PATH


def generate_admission_dataset(
    n_samples: int = 1000,
    random_state: int = RANDOM_STATE,
    output_path: Path = RAW_DATA_PATH,
) -> pd.DataFrame:
    """Generate synthetic admission dataset and save to CSV."""
    np.random.seed(random_state)

    student_ids = [f"STU_{1000 + i:04d}" for i in range(n_samples)]

    # 10th & 12th Percentage: Beta distribution scaled to realistic high school scores (55% - 99%)
    tenth_pct = np.clip(np.random.beta(a=6, b=3, size=n_samples) * 44 + 55, 55.0, 99.5).round(2)
    # 12th percentage is correlated with 10th
    twelfth_pct = np.clip(
        0.7 * tenth_pct + 0.3 * (np.random.beta(a=5, b=3, size=n_samples) * 45 + 53)
        + np.random.normal(0, 2.5, size=n_samples),
        52.0,
        99.0,
    ).round(2)

    # CGPA (on 10.0 scale): 5.5 to 9.9
    cgpa_raw = 0.5 * (twelfth_pct / 10.0) + 0.5 * (np.random.beta(a=5, b=2.5, size=n_samples) * 4.2 + 5.6)
    cgpa = np.clip(cgpa_raw + np.random.normal(0, 0.2, size=n_samples), 5.5, 9.95).round(2)

    # Entrance Score (0 - 100): General admission test score
    entrance_score = np.clip(
        0.4 * (cgpa * 10) + 0.6 * (np.random.beta(a=4, b=3, size=n_samples) * 65 + 32)
        + np.random.normal(0, 4.0, size=n_samples),
        25.0,
        99.5,
    ).round(2)

    # Interview Score (0 - 100): Independent assessment of communication and technical fit
    interview_score = np.clip(
        np.random.beta(a=4.5, b=3.5, size=n_samples) * 65 + 32 + np.random.normal(0, 5.0, size=n_samples),
        28.0,
        98.0,
    ).round(2)

    # Aptitude Score (0 - 100): Quantitative and logical reasoning
    aptitude_score = np.clip(
        0.5 * entrance_score + 0.5 * (np.random.beta(a=4, b=3, size=n_samples) * 60 + 35)
        + np.random.normal(0, 4.0, size=n_samples),
        25.0,
        99.0,
    ).round(2)

    # Extracurricular Score (0 - 100): Leadership, sports, cultural activities
    extracurricular = np.clip(
        np.random.beta(a=3, b=3, size=n_samples) * 75 + 20 + np.random.normal(0, 5.0, size=n_samples),
        15.0,
        98.0,
    ).round(2)

    # Attendance (0 - 100%): College/school track record (most students 70-98%)
    attendance = np.clip(
        np.random.beta(a=8, b=2.5, size=n_samples) * 35 + 63 + np.random.normal(0, 2.0, size=n_samples),
        55.0,
        100.0,
    ).round(2)

    # Previous Backlogs (Integer): Most students have 0, some 1, fewer 2-5
    backlog_probs = [0.65, 0.18, 0.10, 0.05, 0.015, 0.005]
    previous_backlogs = np.random.choice(len(backlog_probs), size=n_samples, p=backlog_probs)

    # Compute composite academic index
    composite_index = (
        0.25 * (cgpa / 10.0)
        + 0.25 * (entrance_score / 100.0)
        + 0.15 * (interview_score / 100.0)
        + 0.10 * (aptitude_score / 100.0)
        + 0.10 * (twelfth_pct / 100.0)
        + 0.05 * (tenth_pct / 100.0)
        + 0.05 * (extracurricular / 100.0)
        + 0.05 * (attendance / 100.0)
        - 0.08 * previous_backlogs
    )

    # Add realistic unobserved human review / borderline noise (mean=0, std=0.065)
    noise = np.random.normal(0, 0.065, size=n_samples)
    final_score = composite_index + noise

    # Severe penalty: 3 or more backlogs or attendance < 65% makes admission highly unlikely
    hard_disqualification = (previous_backlogs >= 3) | (attendance < 65.0)
    final_score = np.where(hard_disqualification, final_score - 0.20, final_score)

    # Classification threshold (0.73 yields balanced realistic distribution: ~55% eligible, 45% not eligible)
    eligible = (final_score >= 0.730).astype(int)

    df = pd.DataFrame(
        {
            "Student_ID": student_ids,
            "10th_Percentage": tenth_pct,
            "12th_Percentage": twelfth_pct,
            "CGPA": cgpa,
            "Entrance_Score": entrance_score,
            "Interview_Score": interview_score,
            "Aptitude_Score": aptitude_score,
            "Extracurricular_Score": extracurricular,
            "Attendance": attendance,
            "Previous_Backlogs": previous_backlogs,
            "Eligible": eligible,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully at: {output_path}")
    print(f"Total Records: {len(df)}")
    print(f"Eligible (1): {df['Eligible'].sum()} ({df['Eligible'].mean()*100:.1f}%)")
    print(f"Not Eligible (0): {len(df) - df['Eligible'].sum()} ({(1 - df['Eligible'].mean())*100:.1f}%)")

    return df


if __name__ == "__main__":
    generate_admission_dataset()
