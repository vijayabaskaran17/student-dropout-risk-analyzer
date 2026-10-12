"""
Configuration and constants for Student Dropout Risk Analyzer.
Defines paths, feature schemas, cost matrices, and styling parameters.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_dropout_dataset_v3.csv"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Ensure directories exist
for p in [PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, METRICS_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Target and Identifier
ID_COL = "Student_ID"
TARGET_COL = "Dropout"

# Raw Feature Definitions
NUMERICAL_FEATURES = [
    "Age",
    "Family_Income",
    "Study_Hours_per_Day",
    "Attendance_Rate",
    "Assignment_Delay_Days",
    "Travel_Time_Minutes",
    "Stress_Index",
    "GPA",
    "Semester_GPA",
    "CGPA"
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Internet_Access",
    "Part_Time_Job",
    "Scholarship",
    "Semester",
    "Department",
    "Parental_Education"
]

# Engineered Features
ENGINEERED_NUMERICAL_FEATURES = [
    "Academic_Average",
    "Academic_Trend",
    "Attendance_Risk",
    "Assignment_Delay_Risk",
    "Study_Engagement",
    "Stress_Academic_Vulnerability",
    "Accessibility_Travel_Risk",
    "Socioeconomic_Risk",
    "GPA_Stability",
    "Attendance_Study_Ratio",
    "Delay_Attendance_Interaction",
    "Stress_Attendance_Interaction",
    "Stress_Study_Interaction",
    "Travel_Study_Burden",
    "Financial_Academic_Risk",
    "Overall_Academic_Risk",
    "Overall_Engagement_Risk"
]

ALL_NUMERICAL_FEATURES = NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES
ALL_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES

# Cost Matrix Weights
# A False Negative (missed dropout student who receives no help) is 5x more severe
# than a False Positive (student mistakenly offered academic support)
COST_FN = 5.0
COST_FP = 1.0

# Risk Threshold Categories
RISK_CATEGORIES = {
    "LOW": (0.00, 0.30, "Low Risk", "#2ecc71"),
    "MODERATE": (0.30, 0.60, "Moderate Risk", "#f39c12"),
    "HIGH": (0.60, 0.80, "High Risk", "#e67e22"),
    "CRITICAL": (0.80, 1.01, "Critical Risk", "#e74c3c")
}

# Artifact file paths
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.joblib"
FEATURE_RANKING_PATH = MODELS_DIR / "feature_ranking.joblib"
SHAP_DATA_PATH = MODELS_DIR / "shap_data.joblib"
