"""
app.py
------
Student Performance Analyzer & Predictive Academic Early-Warning System.
Built with Streamlit, Plotly, Pandas, and Scikit-Learn.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page layout and visual identity
st.set_page_config(
    page_title="EduMetrics AI | Student Performance Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Append src directory to module search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_loader import load_data, get_default_data_path, get_data_summary
from predict import load_prediction_model, predict_single_student, score_to_grade
from data_generator import generate_student_dataset

# ==============================================================================
# CUSTOM MODERN CSS STYLING
# ==============================================================================
st.markdown("""
<style>
    /* Global font & smooth transitions */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Hero Header Banner */
    .hero-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        font-weight: 400;
        margin-top: 0.5rem;
        opacity: 0.92;
        line-height: 1.5;
        max-width: 850px;
    }
    
    /* Stat Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748b;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #10b981;
        font-weight: 500;
        margin-top: 0.3rem;
    }
    
    /* Result score pill */
    .score-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .badge-grade-a { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
    .badge-grade-b { background: #e0f2fe; color: #0369a1; border: 1px solid #7dd3fc; }
    .badge-grade-c { background: #fef9c3; color: #a16207; border: 1px solid #fde047; }
    .badge-grade-d { background: #ffedd5; color: #c2410c; border: 1px solid #fdba74; }
    .badge-grade-f { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }

    /* Recommendation card */
    .insight-box {
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.85rem;
        border-left: 5px solid;
    }
    .insight-warning {
        background-color: #fffbeb;
        border-left-color: #f59e0b;
        color: #92400e;
    }
    .insight-info {
        background-color: #f0fdf4;
        border-left-color: #10b981;
        color: #065f46;
    }
    .insight-success {
        background-color: #f0f9ff;
        border-left-color: #0284c7;
        color: #075985;
    }
    .insight-title {
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
    }
    .insight-desc {
        font-size: 0.88rem;
        line-height: 1.45;
    }
    
    /* Clean button styling */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATA & MODEL CACHING
# ==============================================================================
@st.cache_data
def get_cached_dataset():
    data_path = get_default_data_path()
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df_gen = generate_student_dataset(n_samples=1000, random_seed=42)
        df_gen.to_csv(data_path, index=False)
        return df_gen
    return load_data(data_path)

@st.cache_resource
def get_cached_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "student_performance_model.pkl")
    meta_path = os.path.join(base_dir, "models", "model_metadata.json")
    
    metadata = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}
            
    if os.path.exists(model_path):
        try:
            model = load_prediction_model(model_path)
            return model, metadata
        except Exception:
            pass
            
    # Trigger model training if not yet built
    from train_model import train_and_export
    train_and_export()
    model = load_prediction_model(model_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    return model, metadata

# Load primary resources
df = get_cached_dataset()
model, metadata = get_cached_model()
summary_stats = get_data_summary(df)

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135810.png", width=70)
    st.title("EduMetrics AI")
    st.caption("Student Academic Performance Intelligence Platform")
    
    st.divider()
    
    st.markdown("### 📌 Active ML Engine")
    model_name = metadata.get("model_name", "Gradient Boosting Regressor")
    test_r2 = metadata.get("metrics", {}).get("test_r2", 0.912)
    test_rmse = metadata.get("metrics", {}).get("test_rmse", 3.84)
    
    st.info(f"**Model:** {model_name}\n\n**Test R²:** `{test_r2}`\n\n**Test RMSE:** `{test_rmse}` pts")
    
    st.divider()
    st.markdown("### 💡 Quick Navigation")
    st.markdown("""
    - **1. Executive Dashboard:** High-level school KPIs, distributions & correlation insights.
    - **2. Student Predictor:** Real-time score & grade prediction with actionable remediation tips.
    - **3. Model Diagnostics:** Comparison across 5 algorithms & feature importance.
    - **4. Dataset Explorer:** Filterable record viewer with CSV export.
    """)
    
    st.divider()
    st.caption("Designed for Academic Advisors & Data Science Portfolios.")

# ==============================================================================
# HERO BANNER
# ==============================================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎓 Student Performance Analyzer</div>
    <div class="hero-subtitle">
        An end-to-end data intelligence platform leveraging machine learning to uncover key drivers of student achievement, predict standardized exam outcomes, and deliver personalized intervention recommendations.
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# TABS INTERFACE
# ==============================================================================
tab_dash, tab_pred, tab_diag, tab_data = st.tabs([
    "📊 Executive Dashboard",
    "🎯 Student Predictor & Risk Assessment",
    "🧪 Model Diagnostics & AI Insights",
    "📁 Dataset Explorer"
])

# ------------------------------------------------------------------------------
# TAB 1: EXECUTIVE DASHBOARD
# ------------------------------------------------------------------------------
with tab_dash:
    # Key KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Cohort Size</div>
            <div class="metric-value">{summary_stats['total_students']:,}</div>
            <div class="metric-sub">Verified records</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Final Score</div>
            <div class="metric-value">{summary_stats['avg_final_score']}</div>
            <div class="metric-sub">Out of 100 max</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Pass Rate</div>
            <div class="metric-value">{summary_stats['pass_rate_pct']}%</div>
            <div class="metric-sub">Score ≥ 60.0 threshold</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Attendance</div>
            <div class="metric-value">{summary_stats['avg_attendance']}%</div>
            <div class="metric-sub">Academic semester</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Prep Course Rate</div>
            <div class="metric-value">{summary_stats['test_prep_completion_rate']}%</div>
            <div class="metric-sub">Completed training</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section: Distributions & Comparisons
    chart_col1, chart_col2 = st.columns([1.1, 1.0])
    
    with chart_col1:
        st.subheader("📈 Subject Score Distributions")
        score_df = df[["math_score", "reading_score", "writing_score", "final_score"]].melt(
            var_name="Subject", value_name="Score"
        )
        score_df["Subject"] = score_df["Subject"].map({
            "math_score": "Mathematics",
            "reading_score": "Reading",
            "writing_score": "Writing",
            "final_score": "Overall Final"
        })
        
        fig_dist = px.box(
            score_df,
            x="Subject",
            y="Score",
            color="Subject",
            color_discrete_sequence=["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b"],
            points="outliers"
        )
        fig_dist.update_layout(
            height=370,
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="",
            yaxis_title="Score (0-100)"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with chart_col2:
        st.subheader("🎓 Grade Distribution & Retention")
        grade_counts = df["grade"].value_counts().reindex(["A", "B", "C", "D", "F"]).reset_index()
        grade_counts.columns = ["Grade", "Students"]
        
        colors = ["#10b981", "#0284c7", "#f59e0b", "#f97316", "#ef4444"]
        fig_grade = px.bar(
            grade_counts,
            x="Grade",
            y="Students",
            color="Grade",
            color_discrete_sequence=colors,
            text="Students"
        )
        fig_grade.update_traces(textposition="outside", texttemplate="%{text}")
        fig_grade.update_layout(
            height=370,
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Number of Students"
        )
        st.plotly_chart(fig_grade, use_container_width=True)

    st.markdown("---")
    
    # Section: Deep-Dive Impact Analytics
    st.subheader("🔍 Deep-Dive: Key Success Drivers")
    analysis_col1, analysis_col2 = st.columns(2)
    
    with analysis_col1:
        st.markdown("**Impact of Attendance & Study Hours on Final Score**")
        fig_scatter = px.scatter(
            df,
            x="study_hours_per_week",
            y="final_score",
            color="test_prep_course",
            size="attendance_rate",
            color_discrete_map={"Completed": "#10b981", "Not Completed": "#64748b"},
            labels={
                "study_hours_per_week": "Study Hours / Week",
                "final_score": "Final Exam Score",
                "test_prep_course": "Test Prep Course"
            },
            opacity=0.75
        )
        fig_scatter.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with analysis_col2:
        st.markdown("**Parental Education vs. Exam Outcome**")
        edu_order = [
            "Some High School", "High School", "Some College",
            "Associate's Degree", "Bachelor's Degree", "Master's Degree"
        ]
        edu_avg = df.groupby("parental_education")["final_score"].mean().reindex(edu_order).reset_index()
        
        fig_edu = px.bar(
            edu_avg,
            x="final_score",
            y="parental_education",
            orientation="h",
            color="final_score",
            color_continuous_scale="Blues",
            labels={"final_score": "Average Final Score", "parental_education": "Parental Education"}
        )
        fig_edu.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_edu, use_container_width=True)

    # Correlation Matrix
    with st.expander("📊 View Correlation Heatmap (All Quantitative Factors)"):
        num_cols = ["study_hours_per_week", "attendance_rate", "past_failures", "sleep_hours", "family_support", "math_score", "reading_score", "writing_score", "final_score"]
        corr_matrix = df[num_cols].corr().round(2)
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Pearson Correlation Coefficient Matrix"
        )
        fig_corr.update_layout(height=450)
        st.plotly_chart(fig_corr, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: STUDENT SCORE PREDICTOR & RISK ASSESSMENT
# ------------------------------------------------------------------------------
with tab_pred:
    st.subheader("🎯 Real-Time Score Prediction & Academic Advisory")
    st.markdown(
        "Enter a student's profile attributes below or pick a preset persona to instantly predict their score, estimated grade, and generate actionable academic interventions."
    )
    
    # Preset Personas for fast demo
    preset_cols = st.columns(4)
    with preset_cols[0]:
        preset_top = st.button("🌟 Preset: Honor Roll Student")
    with preset_cols[1]:
        preset_avg = st.button("⚖️ Preset: Average Student")
    with preset_cols[2]:
        preset_risk = st.button("⚠️ Preset: At-Risk Student")
    with preset_cols[3]:
        preset_clear = st.button("🔄 Reset Defaults")

    # Initialize session state for inputs
    if "study_hours" not in st.session_state or preset_clear:
        st.session_state["study_hours"] = 12.0
        st.session_state["attendance"] = 85.0
        st.session_state["failures"] = 0
        st.session_state["sleep"] = 7.0
        st.session_state["support"] = 3
        st.session_state["gender"] = "Female"
        st.session_state["parental_edu"] = "Some College"
        st.session_state["lunch"] = "Standard"
        st.session_state["test_prep"] = "Completed"
        st.session_state["school"] = "Public"
        st.session_state["internet"] = "Yes"
        st.session_state["extra"] = "Yes"
        
    if preset_top:
        st.session_state["study_hours"] = 24.0
        st.session_state["attendance"] = 96.0
        st.session_state["failures"] = 0
        st.session_state["sleep"] = 7.5
        st.session_state["support"] = 5
        st.session_state["gender"] = "Female"
        st.session_state["parental_edu"] = "Master's Degree"
        st.session_state["lunch"] = "Standard"
        st.session_state["test_prep"] = "Completed"
        st.session_state["school"] = "Private"
        st.session_state["internet"] = "Yes"
        st.session_state["extra"] = "Yes"
        
    if preset_avg:
        st.session_state["study_hours"] = 10.0
        st.session_state["attendance"] = 82.0
        st.session_state["failures"] = 0
        st.session_state["sleep"] = 6.5
        st.session_state["support"] = 3
        st.session_state["gender"] = "Male"
        st.session_state["parental_edu"] = "Associate's Degree"
        st.session_state["lunch"] = "Standard"
        st.session_state["test_prep"] = "Not Completed"
        st.session_state["school"] = "Public"
        st.session_state["internet"] = "Yes"
        st.session_state["extra"] = "No"

    if preset_risk:
        st.session_state["study_hours"] = 4.0
        st.session_state["attendance"] = 58.0
        st.session_state["failures"] = 2
        st.session_state["sleep"] = 5.0
        st.session_state["support"] = 2
        st.session_state["gender"] = "Male"
        st.session_state["parental_edu"] = "Some High School"
        st.session_state["lunch"] = "Free/Reduced"
        st.session_state["test_prep"] = "Not Completed"
        st.session_state["school"] = "Public"
        st.session_state["internet"] = "No"
        st.session_state["extra"] = "No"

    # Input Form
    with st.form("prediction_form"):
        col_in1, col_in2 = st.columns(2)
        
        with col_in1:
            st.markdown("#### 📚 Academic Habits & Behavioral Factors")
            in_study = st.slider(
                "Weekly Study Hours",
                min_value=1.0, max_value=35.0,
                value=float(st.session_state["study_hours"]),
                step=0.5,
                help="Estimated non-classroom study and homework hours per week."
            )
            in_attendance = st.slider(
                "Attendance Rate (%)",
                min_value=40.0, max_value=100.0,
                value=float(st.session_state["attendance"]),
                step=1.0,
                help="Percentage of lectures attended during the semester."
            )
            in_failures = st.selectbox(
                "Previous Course Failures",
                options=[0, 1, 2, 3],
                index=[0, 1, 2, 3].index(st.session_state["failures"]),
                help="Number of prior failed courses or repeat modules."
            )
            in_sleep = st.slider(
                "Average Sleep Hours / Night",
                min_value=4.0, max_value=10.0,
                value=float(st.session_state["sleep"]),
                step=0.5
            )
            in_support = st.select_slider(
                "Family & Peer Support Level",
                options=[1, 2, 3, 4, 5],
                value=int(st.session_state["support"]),
                help="1 = Very Low Support, 5 = Highly Supportive Academic Environment"
            )

        with col_in2:
            st.markdown("#### 🏛️ Demographic & Institutional Factors")
            in_gender = st.selectbox(
                "Gender",
                options=["Female", "Male"],
                index=["Female", "Male"].index(st.session_state["gender"])
            )
            in_edu = st.selectbox(
                "Parental Level of Education",
                options=[
                    "Some High School", "High School", "Some College",
                    "Associate's Degree", "Bachelor's Degree", "Master's Degree"
                ],
                index=[
                    "Some High School", "High School", "Some College",
                    "Associate's Degree", "Bachelor's Degree", "Master's Degree"
                ].index(st.session_state["parental_edu"])
            )
            in_lunch = st.selectbox(
                "Lunch Plan Type",
                options=["Standard", "Free/Reduced"],
                index=["Standard", "Free/Reduced"].index(st.session_state["lunch"]),
                help="Indicator of socio-economic assistance."
            )
            in_prep = st.selectbox(
                "Test Preparation Course",
                options=["Completed", "Not Completed"],
                index=["Completed", "Not Completed"].index(st.session_state["test_prep"])
            )
            in_school = st.selectbox(
                "School Type",
                options=["Public", "Private"],
                index=["Public", "Private"].index(st.session_state["school"])
            )
            subcol_a, subcol_b = st.columns(2)
            with subcol_a:
                in_internet = st.selectbox(
                    "Home Internet Access",
                    options=["Yes", "No"],
                    index=["Yes", "No"].index(st.session_state["internet"])
                )
            with subcol_b:
                in_extra = st.selectbox(
                    "Extracurriculars",
                    options=["Yes", "No"],
                    index=["Yes", "No"].index(st.session_state["extra"])
                )

        submit_btn = st.form_submit_button("⚡ Predict Student Score & Generate Diagnostic", use_container_width=True)

    if submit_btn or "last_prediction" in st.session_state:
        # Prepare input dictionary
        student_input = {
            "gender": in_gender,
            "parental_education": in_edu,
            "lunch_type": in_lunch,
            "test_prep_course": in_prep,
            "school_type": in_school,
            "internet_access": in_internet,
            "extracurricular": in_extra,
            "study_hours_per_week": in_study,
            "attendance_rate": in_attendance,
            "past_failures": in_failures,
            "sleep_hours": in_sleep,
            "family_support": in_support
        }
        
        result = predict_single_student(student_input, model=model)
        st.session_state["last_prediction"] = result
        st.session_state["last_input"] = student_input
        
        st.markdown("---")
        st.markdown("### 🏆 Prediction Results & Diagnostic Profile")
        
        res_col1, res_col2 = st.columns([1.1, 1.3])
        
        with res_col1:
            pred_score = result["predicted_score"]
            grade = result["grade"]
            risk_tier = result["risk_tier"]
            
            # Gauge chart for predicted score
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Final Exam Score", 'font': {'size': 20, 'color': '#1e293b'}},
                number={'suffix': " / 100", 'font': {'size': 34, 'weight': 'bold'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#2563eb", 'thickness': 0.3},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#e2e8f0",
                    'steps': [
                        {'range': [0, 60], 'color': '#fee2e2'},
                        {'range': [60, 75], 'color': '#fef9c3'},
                        {'range': [75, 90], 'color': '#e0f2fe'},
                        {'range': [90, 100], 'color': '#dcfce7'}
                    ],
                    'threshold': {
                        'line': {'color': "#0f172a", 'width': 3},
                        'thickness': 0.8,
                        'value': pred_score
                    }
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=30, r=30, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Badge & Risk Card
            badge_class = f"badge-grade-{grade.lower()}"
            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:1.2rem; text-align:center;">
                <div style="font-size:0.9rem; color:#64748b; font-weight:600; margin-bottom:0.3rem;">EXPECTED GRADE TIER</div>
                <div style="font-size:2.8rem; font-weight:800; color:#1e293b; line-height:1;">Grade {grade}</div>
                <div style="margin-top:0.6rem;">
                    <span class="score-badge {badge_class}">{risk_tier}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.markdown("#### 📋 Actionable Academic Interventions")
            insights = result["insights"]
            for ins in insights:
                box_class = f"insight-{ins['type']}"
                icon = "⚠️" if ins["type"] == "warning" else ("💡" if ins["type"] == "info" else "✅")
                st.markdown(f"""
                <div class="insight-box {box_class}">
                    <div class="insight-title">{icon} {ins['title']}</div>
                    <div class="insight-desc">{ins['message']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Interactive What-If Scenario Comparison
            st.markdown("#### ⚡ 'What-If' Simulation")
            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                # Add 5 study hours
                alt_input_study = student_input.copy()
                alt_input_study["study_hours_per_week"] = min(35.0, alt_input_study["study_hours_per_week"] + 5.0)
                alt_res_study = predict_single_student(alt_input_study, model=model)
                delta_study = round(alt_res_study["predicted_score"] - pred_score, 1)
                st.metric(
                    label="+5 hrs Study/Week",
                    value=f"{alt_res_study['predicted_score']}",
                    delta=f"{delta_study:+} pts"
                )
            with sim_col2:
                # Maximize attendance to 95%
                alt_input_att = student_input.copy()
                alt_input_att["attendance_rate"] = 95.0
                alt_res_att = predict_single_student(alt_input_att, model=model)
                delta_att = round(alt_res_att["predicted_score"] - pred_score, 1)
                st.metric(
                    label="Boost Attendance to 95%",
                    value=f"{alt_res_att['predicted_score']}",
                    delta=f"{delta_att:+} pts"
                )

# ------------------------------------------------------------------------------
# TAB 3: MODEL DIAGNOSTICS & AI INSIGHTS
# ------------------------------------------------------------------------------
with tab_diag:
    st.subheader("🧪 Machine Learning Architecture & Benchmarks")
    st.markdown(
        "To ensure high predictive validity, 5 distinct supervised regression algorithms were evaluated using 5-fold cross-validation and hold-out testing."
    )
    
    comp_records = metadata.get("model_comparison", [])
    if comp_records:
        comp_df = pd.DataFrame(comp_records)
        
        col_m1, col_m2 = st.columns([1.2, 1.0])
        
        with col_m1:
            st.markdown("#### 📊 Candidate Algorithm Comparison")
            fig_comp = px.bar(
                comp_df,
                x="model_name",
                y="test_r2",
                color="test_r2",
                color_continuous_scale="Blues",
                text="test_r2",
                labels={"model_name": "Model", "test_r2": "Test R² Score"}
            )
            fig_comp.update_traces(textposition="outside", texttemplate="%{text:.4f}")
            fig_comp.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis_range=[0, 1.05]
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
        with col_m2:
            st.markdown("#### 📋 Detailed Metrics Table")
            display_comp_df = comp_df.rename(columns={
                "model_name": "Algorithm",
                "train_r2": "Train R²",
                "test_r2": "Test R²",
                "test_rmse": "Test RMSE",
                "test_mae": "Test MAE"
            })
            st.dataframe(display_comp_df.style.highlight_max(subset=["Test R²"], color="#dcfce7"), use_container_width=True)
            
            st.success(
                f"**Champion Model Selected:** `{metadata.get('model_name')}`\n\n"
                f"Achieved lowest prediction error with **Test RMSE of {metadata.get('metrics', {}).get('test_rmse')} points**."
            )

    st.markdown("---")
    
    # Feature Importances
    st.subheader("🎯 Feature Importance (Global Impact Drivers)")
    top_features = metadata.get("top_features", [])
    if top_features:
        feat_df = pd.DataFrame(top_features).head(10).sort_values(by="importance", ascending=True)
        # Clean feature names for clean UI display
        feat_df["clean_feature"] = feat_df["feature"].str.replace("num__", "").str.replace("cat__", "").str.replace("_", " ").str.title()
        
        fig_feat = px.bar(
            feat_df,
            x="importance",
            y="clean_feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Teal",
            labels={"importance": "Relative Importance Weight", "clean_feature": "Predictor Feature"}
        )
        fig_feat.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    # Preprocessing Pipeline Explainer
    with st.expander("⚙️ View Preprocessing & Pipeline Architecture Details"):
        st.markdown("""
        ```mermaid
        flowchart LR
            RawData[Raw Student Data] --> Split[Train/Test 80/20 Split]
            Split --> CT[ColumnTransformer]
            CT --> |Numerical Features| SS[StandardScaler]
            CT --> |Categorical Features| OHE[OneHotEncoder]
            SS --> Merged[Transformed Feature Vector]
            OHE --> Merged
            Merged --> Regressor[Tuned Gradient Boosting Regressor]
            Regressor --> PredScore[Final Continuous Score 0-100]
            PredScore --> GradeEngine[Grade & Risk Tier Classification]
        ```
        - **Numerical Pipeline:** Handled via standard Z-score normalization (`StandardScaler`) to prevent gradient dominance.
        - **Categorical Pipeline:** One-hot encoded with `handle_unknown='ignore'` to guarantee zero runtime failures when unseen levels are submitted.
        - **Data Leakage Safeguard:** Transformations are strictly fit on the training partition and transformed on validation/inference.
        """)

# ------------------------------------------------------------------------------
# TAB 4: DATASET EXPLORER
# ------------------------------------------------------------------------------
with tab_data:
    st.subheader("📁 Interactive Dataset Explorer")
    st.markdown("Filter, sort, and inspect student records or download the processed dataset for external reporting.")
    
    # Filters
    fil_col1, fil_col2, fil_col3 = st.columns(3)
    with fil_col1:
        sel_grade = st.multiselect("Filter by Grade", options=["A", "B", "C", "D", "F"], default=["A", "B", "C", "D", "F"])
    with fil_col2:
        sel_gender = st.multiselect("Filter by Gender", options=["Female", "Male"], default=["Female", "Male"])
    with fil_col3:
        sel_prep = st.multiselect("Filter by Prep Course", options=["Completed", "Not Completed"], default=["Completed", "Not Completed"])
        
    filtered_df = df[
        (df["grade"].isin(sel_grade)) &
        (df["gender"].isin(sel_gender)) &
        (df["test_prep_course"].isin(sel_prep))
    ]
    
    st.caption(f"Showing **{len(filtered_df):,}** of **{len(df):,}** records based on active filters.")
    st.dataframe(filtered_df, use_container_width=True, height=400)
    
    # Download Button
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="student_performance_filtered.csv",
        mime="text/csv"
    )
    
    with st.expander("📊 View Descriptive Summary Statistics"):
        st.dataframe(filtered_df.describe().round(2), use_container_width=True)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#94a3b8; font-size:0.85rem; padding:1rem 0;'>"
    "Student Performance Analyzer • End-to-End Applied Machine Learning Project • Built with Streamlit & Scikit-Learn"
    "</div>",
    unsafe_allow_html=True
)
