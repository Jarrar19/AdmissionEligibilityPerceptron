"""Streamlit Web Application for Admission Eligibility Prediction.

Serves the trained Perceptron binary classifier pipeline with strict input validation,
prominent academic disclaimers, clear decision outcomes, and model transparency diagnostics
per Section 10 of the PBL Master Specification.
"""

import json
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FEATURE_COLS,
    FIGURES_DIR,
    METRICS_JSON_PATH,
    MODEL_PIPELINE_PATH,
    VALIDATION_RANGES,
)
from src.predict import get_pipeline, predict_eligibility, validate_student_input

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Admission Eligibility Predictor | Perceptron PBL",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished academic UI
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Force Light Background for the app */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], section.main {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    /* Force ALL text, headers, markdown, and labels to high-contrast dark colors */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
    [data-testid="stMarkdownContainer"] *,
    [data-testid="stHeading"] *,
    .stHeadingWithActionElements * {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }

    /* Widget Labels (Form fields) - ensure 100% visibility */
    label,
    .stWidgetLabel,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    label[data-testid="stWidgetLabel"] p {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }

    /* Main header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        letter-spacing: -0.025em;
        margin-bottom: 0.25rem;
    }

    .sub-header {
        font-size: 1.05rem;
        color: #475569 !important;
        -webkit-text-fill-color: #475569 !important;
        font-weight: 500;
        margin-bottom: 1.25rem;
    }

    /* Info card */
    .info-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1.5rem;
        color: #334155 !important;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }

    .code-pill {
        background-color: #f1f5f9;
        color: #1e293b !important;
        -webkit-text-fill-color: #1e293b !important;
        padding: 2px 7px;
        border-radius: 5px;
        border: 1px solid #cbd5e1;
        font-family: monospace;
        font-size: 0.88rem;
        font-weight: 600;
    }

    /* Form card container */
    [data-testid="stForm"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.04) !important;
    }

    /* Input Fields styling */
    div[data-baseweb="input"],
    div[data-baseweb="input"] > div,
    div[data-testid="stNumberInput"] div[data-baseweb="input"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="input"] input,
    div[data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Stepper buttons (+ and -) */
    div[data-testid="stNumberInput"] button {
        background-color: #f1f5f9 !important;
        color: #1e293b !important;
        border-left: 1px solid #cbd5e1 !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background-color: #e2e8f0 !important;
    }

    div[data-testid="stNumberInput"] button svg {
        fill: #1e293b !important;
    }

    /* Decision status cards */
    .status-eligible {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: #ffffff !important;
        padding: 1.6rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(5, 150, 105, 0.25), 0 4px 6px -4px rgba(5, 150, 105, 0.2);
        margin: 1.4rem 0;
    }

    .status-not-eligible {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: #ffffff !important;
        padding: 1.6rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(220, 38, 38, 0.25), 0 4px 6px -4px rgba(220, 38, 38, 0.2);
        margin: 1.4rem 0;
    }

    /* Metric Badges & Sidebar */
    .stMetric {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 0.6rem 0.8rem !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04) !important;
    }

    .stMetric label, .stMetric [data-testid="stMetricValue"] {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }

    /* Form submit button styling */
    div.stButton > button, div.stFormSubmitButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        padding: 0.7rem 1.5rem !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.2s ease-in-out !important;
    }

    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        box-shadow: 0 6px 12px -2px rgba(37, 99, 235, 0.4) !important;
        transform: translateY(-1px);
    }

    /* Sidebar light background and text */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    [data-testid="stSidebar"] * {
        color: #1e293b !important;
    }

    /* Expander & Tabs styling */
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
    }

    [data-testid="stExpander"] summary span {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar: System Status & Quick Info
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.shields.io/badge/Model-Perceptron%20(sklearn)-blue.svg", use_container_width=True)
    st.markdown("### 📌 System Metadata")
    st.markdown("**Repo:** `admission-eligibility-perceptron`")
    st.markdown("**Course:** TAE PBL — Neural Networks & Deep Learning")
    st.markdown("**Algorithm:** Single-Layer Perceptron")
    st.markdown("**Runtime:** Python 3.12.10 (pinned)")
    st.markdown("---")

    # Load persisted pipeline status without retraining (§10)
    pipeline_loaded = False
    try:
        pipeline = get_pipeline(MODEL_PIPELINE_PATH)
        pipeline_loaded = True
        st.success("✅ Pre-trained Pipeline Loaded\n(Zero retraining on launch)")
    except Exception as exc:
        st.error(f"❌ Pipeline Load Failed: {exc}")

    # Load offline evaluated test metrics
    test_metrics = None
    if METRICS_JSON_PATH.exists():
        try:
            with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
                test_metrics = json.load(f)
            st.markdown("### 📊 Held-Out Test Metrics")
            m = test_metrics["metrics"]
            st.metric("Test Accuracy", f"{m['accuracy']*100:.2f}%")
            st.metric("Precision", f"{m['precision']*100:.2f}%")
            st.metric("Recall", f"{m['recall']*100:.2f}%")
            st.metric("F1-Score", f"{m['f1_score']*100:.2f}%")
        except Exception:
            pass

# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">🎓 Admission Eligibility Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Supervised Binary Classification Using a Single-Layer Perceptron Pipeline</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-card">
        <strong>💡 System Overview:</strong> The applicant profile below is evaluated in real-time by a serialized 
        <span class="code-pill">scikit-learn</span> Perceptron pipeline. Inputs are first standardized via 
        <span class="code-pill">StandardScaler</span>, and the decision score <strong>z = &sum; w<sub>i</sub> x<sub>i</sub> + b</strong> 
        is computed. If <strong>z &ge; 0</strong>, the applicant meets the eligibility threshold (<strong>Eligible</strong>); 
        otherwise, the profile falls below the decision boundary (<strong>Not Eligible</strong>).
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Student Input Form (§10 & §11)
# ---------------------------------------------------------
st.markdown('<h3 style="color: #0f172a; font-weight: 700; margin-top: 0.5rem; margin-bottom: 0.85rem;">📝 Enter Applicant Academic Profile</h3>', unsafe_allow_html=True)

with st.form("admission_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<h5 style="color: #1e293b; font-weight: 700; margin-bottom: 0.8rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.4rem;">📚 Academic Fundamentals</h5>', unsafe_allow_html=True)
        tenth_pct = st.number_input(
            "10th Grade Percentage (%)",
            min_value=0.0,
            max_value=100.0,
            value=84.5,
            step=0.5,
            help="Cumulative secondary school percentage [0.0 - 100.0%]",
        )
        twelfth_pct = st.number_input(
            "12th Grade Percentage (%)",
            min_value=0.0,
            max_value=100.0,
            value=83.0,
            step=0.5,
            help="Higher secondary aggregate percentage [0.0 - 100.0%]",
        )
        cgpa = st.number_input(
            "Undergraduate CGPA (Scale 0-10)",
            min_value=0.0,
            max_value=10.0,
            value=8.2,
            step=0.05,
            help="Undergraduate Cumulative GPA [0.0 - 10.0]",
        )

    with col2:
        st.markdown('<h5 style="color: #1e293b; font-weight: 700; margin-bottom: 0.8rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.4rem;">🎯 Standardized Scores</h5>', unsafe_allow_html=True)
        entrance_score = st.number_input(
            "Entrance Examination Score",
            min_value=0.0,
            max_value=100.0,
            value=78.0,
            step=1.0,
            help="Admission entrance examination score [0.0 - 100.0]",
        )
        interview_score = st.number_input(
            "Technical Interview Score",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0,
            help="Panel interview evaluation [0.0 - 100.0]",
        )
        aptitude_score = st.number_input(
            "Aptitude & Reasoning Score",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0,
            help="Quantitative and analytical reasoning [0.0 - 100.0]",
        )

    with col3:
        st.markdown('<h5 style="color: #1e293b; font-weight: 700; margin-bottom: 0.8rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.4rem;">🏛️ Discipline & Activities</h5>', unsafe_allow_html=True)
        extracurricular = st.number_input(
            "Extracurricular & Leadership Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0,
            help="Sports, leadership, and community activities [0.0 - 100.0]",
        )
        attendance = st.number_input(
            "Classroom Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=90.0,
            step=0.5,
            help="Attendance record percentage [0.0 - 100.0%]",
        )
        backlogs = st.number_input(
            "Previous Uncleared Backlogs",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help="Number of active or previous uncleared backlogs (integer >= 0)",
        )

    submit_button = st.form_submit_button("🔍 Predict Admission Eligibility", use_container_width=True)

# ---------------------------------------------------------
# Prediction Execution & Result Display (§10)
# ---------------------------------------------------------
if submit_button:
    input_data = {
        "10th_Percentage": tenth_pct,
        "12th_Percentage": twelfth_pct,
        "CGPA": cgpa,
        "Entrance_Score": entrance_score,
        "Interview_Score": interview_score,
        "Aptitude_Score": aptitude_score,
        "Extracurricular_Score": extracurricular,
        "Attendance": attendance,
        "Previous_Backlogs": backlogs,
    }

    # Strict Validation Check
    is_valid, err = validate_student_input(input_data)
    if not is_valid:
        st.error(f"❌ **Validation Error**: {err}")
    else:
        try:
            res = predict_eligibility(input_data, MODEL_PIPELINE_PATH)
            status_label = res["status_label"]
            z_score = res["decision_score_z"]
            is_eligible = res["is_eligible"]

            if is_eligible:
                st.markdown(
                    f"""
                    <div class="status-eligible">
                        <h2 style="margin: 0; color: white;">🎉 Admission Status: ELIGIBLE</h2>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.95;">
                            Decision Boundary Score: <strong>z = {z_score:+.4f}</strong> (Threshold: z ≥ 0)
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="status-not-eligible">
                        <h2 style="margin: 0; color: white;">⚠️ Admission Status: NOT ELIGIBLE</h2>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.95;">
                            Decision Boundary Score: <strong>z = {z_score:+.4f}</strong> (Threshold: z < 0)
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Feature contribution breakdown
            st.markdown('<h4 style="color: #0f172a; font-weight: 700; margin-top: 1.2rem; margin-bottom: 0.5rem;">🔬 Perceptron Linear Decision Breakdown</h4>', unsafe_allow_html=True)
            st.write(
                f"Equation: $z = \\sum (w_i \\cdot x_i) + b = ({z_score - res['bias']:+.4f}) + ({res['bias']:+.4f}) = {z_score:+.4f}$"
            )

            contrib_rows = []
            for feat, info in res["feature_contributions"].items():
                contrib_rows.append(
                    {
                        "Feature": feat,
                        "Applicant Raw Value": info["raw_value"],
                        "Standardized (z-score)": info["scaled_value"],
                        "Learned Weight (w_i)": info["weight"],
                        "Net Contribution (w_i * x_i)": info["linear_contribution"],
                    }
                )
            contrib_df = pd.DataFrame(contrib_rows)
            st.dataframe(contrib_df, use_container_width=True)

        except Exception as exc:
            st.error(f"An error occurred during prediction: {exc}")

# ---------------------------------------------------------
# Collapsible Model Transparency & Diagnostics (§10)
# ---------------------------------------------------------
with st.expander("🛠️ Model Architecture & Technical Diagnostics (PBL Viva Reference)"):
    tab1, tab2, tab3 = st.tabs(["Algorithm Details", "Evaluation Metrics", "Figures & Plots"])

    with tab1:
        st.markdown("#### Perceptron Binary Classifier Formulation (§8.3)")
        st.latex(r"z = \sum_{i=1}^n w_i x_i + b = \mathbf{w}^T \mathbf{x} + b")
        st.latex(r"\hat{y} = \begin{cases} 1 & \text{if } z \geq 0 \text{ (Eligible)} \\ 0 & \text{if } z < 0 \text{ (Not Eligible)} \end{cases}")
        st.markdown(
            """
            - **Update Rule on Misclassification**:
              $$w_i \\leftarrow w_i + \\eta (y - \\hat{y}) x_i, \\quad b \\leftarrow b + \\eta (y - \\hat{y})$$
            - **Standardization Mandatory**: Because Perceptron weights update proportionally to input magnitude $x_i$, 
              `StandardScaler` is required to ensure entrance scores (0-100) do not artificially dwarf CGPA (0-10).
            - **Linearity Limitation**: The Perceptron defines a flat hyperplane. Borderline students with conflicting strengths 
              cannot be separated non-linearly.
            """
        )

    with tab2:
        if test_metrics:
            st.markdown("#### Verified Held-Out Test Evaluation (§8.4)")
            colA, colB, colC, colD = st.columns(4)
            m = test_metrics["metrics"]
            colA.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
            colB.metric("Precision", f"{m['precision']*100:.2f}%")
            colC.metric("Recall", f"{m['recall']*100:.2f}%")
            colD.metric("F1-Score", f"{m['f1_score']*100:.2f}%")

            st.markdown("##### Secondary Baseline Comparison (§8.4)")
            sec = test_metrics["secondary_comparison"]
            comp_df = pd.DataFrame(
                [
                    {
                        "Model": "Perceptron (Designated Primary)",
                        "Accuracy": f"{m['accuracy']*100:.2f}%",
                        "F1-Score": f"{m['f1_score']*100:.2f}%",
                        "Decision Boundary": "Linear Hyperplane (Step function)",
                    },
                    {
                        "Model": sec["model"],
                        "Accuracy": f"{sec['accuracy']*100:.2f}%",
                        "F1-Score": f"{sec['f1_score']*100:.2f}%",
                        "Decision Boundary": "Linear Hyperplane (Logistic Sigmoid)",
                    },
                ]
            )
            st.table(comp_df)

    with tab3:
        st.markdown("#### Diagnostic Figures Generated During Training & EDA")
        cm_fig = FIGURES_DIR / "confusion_matrix.png"
        weights_fig = FIGURES_DIR / "feature_weights.png"
        corr_fig = FIGURES_DIR / "correlation_heatmap.png"

        col_img1, col_img2 = st.columns(2)
        if cm_fig.exists():
            col_img1.image(str(cm_fig), caption="Confusion Matrix on Held-Out Test Partition", use_container_width=True)
        if weights_fig.exists():
            col_img2.image(str(weights_fig), caption="Learned Perceptron Feature Weights", use_container_width=True)
        if corr_fig.exists():
            st.image(str(corr_fig), caption="Feature Correlation Heatmap", use_container_width=True)
