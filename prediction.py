"""
Student Dropout Risk Analyzer - Inference & Explainability Engine
==================================================================
Provides direct inference pipeline for student risk evaluation,
feature impact attribution (risk vs protective factors), multi-domain
risk ratings, and institutional intervention recommendations.
Adheres to zero data leakage and non-causal statistical interpretation.
"""

import os
import joblib
import numpy as np
import pandas as pd

from config import (
    FINAL_MODEL_PATH, PREPROCESSOR_PATH, MODEL_METADATA_PATH,
    RISK_CATEGORIES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
)
from pipeline_preprocessing import StudentFeatureEngineer

# Global cache: Model and preprocessor loaded once at process startup
_model = None
_preprocessor = None
_metadata = None
_feature_engineer = StudentFeatureEngineer()


def load_artifacts():
    """Loads serialised model, ColumnTransformer, and metadata once."""
    global _model, _preprocessor, _metadata
    if _model is None or _preprocessor is None:
        if not FINAL_MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError("Model artifacts not found. Please execute train.py first.")
        _model = joblib.load(FINAL_MODEL_PATH)
        _preprocessor = joblib.load(PREPROCESSOR_PATH)
        _metadata = joblib.load(MODEL_METADATA_PATH)
    return _model, _preprocessor, _metadata


def predict_student(student_data: dict) -> dict:
    """
    Accepts raw student features, performs identical feature engineering
    and ColumnTransformer transformations, and evaluates probability,
    threshold decision, risk level, and explainability factors.
    
    Parameters:
    -----------
    student_data : dict
        Contains the 17 raw features (Age, Gender, GPA, Attendance_Rate, etc.)
        
    Returns:
    --------
    dict with prediction (0/1), probability, risk_level, threshold,
    risk_factors, protective_factors, domain_risks, and interventions.
    """
    model, preprocessor, metadata = load_artifacts()

    # Convert dictionary to single-row DataFrame
    raw_df = pd.DataFrame([student_data])

    # Enforce correct numerical casting
    for col in NUMERICAL_FEATURES:
        if col in raw_df.columns:
            raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')

    # Apply identical Feature Engineering
    engineered_df = _feature_engineer.transform(raw_df)

    # Apply identical Preprocessing pipeline
    processed_matrix = preprocessor.transform(engineered_df).astype(np.float32)

    # Generate Probability
    prob_dropout = float(model.predict_proba(processed_matrix)[0, 1])
    prob_retained = 1.0 - prob_dropout

    # Apply cost-optimized threshold
    threshold = float(metadata.get('selected_threshold', 0.50))
    prediction = 1 if prob_dropout >= threshold else 0
    prediction_label = "Likely to Drop Out" if prediction == 1 else "Likely to Persist / Graduate"

    # Determine Risk Category
    risk_level = "Low Risk"
    risk_color = "#2ecc71"
    for cat, (low, high, label, color) in RISK_CATEGORIES.items():
        if low <= prob_dropout < high or (cat == "CRITICAL" and prob_dropout >= 0.80):
            risk_level = label
            risk_color = color
            break

    # Extract student values for factor analysis
    gpa = float(student_data.get('GPA', 2.5))
    sem_gpa = float(student_data.get('Semester_GPA', 2.5))
    cgpa = float(student_data.get('CGPA', 2.5))
    attendance = float(student_data.get('Attendance_Rate', 80.0))
    study_hrs = float(student_data.get('Study_Hours_per_Day', 3.0))
    delay_days = float(student_data.get('Assignment_Delay_Days', 1.0))
    stress = float(student_data.get('Stress_Index', 5.0))
    travel_time = float(student_data.get('Travel_Time_Minutes', 25.0))
    income = float(student_data.get('Family_Income', 30000.0) or 30000.0)
    scholarship = str(student_data.get('Scholarship', 'No')).strip().lower()
    part_time = str(student_data.get('Part_Time_Job', 'No')).strip().lower()
    internet = str(student_data.get('Internet_Access', 'Yes')).strip().lower()

    academic_trend = sem_gpa - cgpa
    acad_avg = (gpa + sem_gpa + cgpa) / 3.0

    # Risk Factors and Protective Factors
    risk_factors = []
    protective_factors = []

    # 1. Academic factors
    if acad_avg < 2.0:
        risk_factors.append(f"Severely low academic average (Avg: {acad_avg:.2f} / 4.0)")
    elif acad_avg >= 3.0:
        protective_factors.append(f"Strong academic performance cushion (Avg: {acad_avg:.2f} / 4.0)")

    if academic_trend < -0.3:
        risk_factors.append(f"Declining academic momentum (Semester GPA is {abs(academic_trend):.2f} pts lower than CGPA)")
    elif academic_trend > 0.2:
        protective_factors.append(f"Positive upward grade trajectory (+{academic_trend:.2f} this semester)")

    # 2. Attendance & Engagement
    if attendance < 75.0:
        risk_factors.append(f"Attendance rate ({attendance:.1f}%) breaches critical 75% institutional threshold")
    elif attendance >= 85.0:
        protective_factors.append(f"High classroom engagement and attendance ({attendance:.1f}%)")

    if delay_days >= 3.0:
        risk_factors.append(f"Chronic assignment delay pattern ({int(delay_days)} days average lag)")
    elif delay_days <= 1.0:
        protective_factors.append("Timely academic assignment submissions (≤ 1 day lag)")

    if study_hrs < 2.0:
        risk_factors.append(f"Low dedicated daily study time ({study_hrs:.1f} hrs/day)")
    elif study_hrs >= 4.0:
        protective_factors.append(f"Substantial independent study investment ({study_hrs:.1f} hrs/day)")

    # 3. Psychological & Well-being
    if stress >= 7.0:
        risk_factors.append(f"Elevated psychological stress index ({stress:.1f} / 10.0)")
    elif stress <= 4.0:
        protective_factors.append(f"Healthy manageable stress level ({stress:.1f} / 10.0)")

    # 4. Socioeconomic & Commute
    if travel_time >= 45.0:
        risk_factors.append(f"High daily commute friction ({travel_time:.0f} mins each way)")
    elif travel_time <= 20.0:
        protective_factors.append("Proximity to campus with negligible travel burden")

    if scholarship == 'yes':
        protective_factors.append("Institutional scholarship recipient providing financial support")
    else:
        if income < 25000:
            risk_factors.append("Low family income without scholarship assistance")

    if part_time == 'yes':
        risk_factors.append("Part-time employment commitments competing with academic schedule")

    if internet == 'no':
        risk_factors.append("Lack of reliable home internet access hindering remote study")
    else:
        protective_factors.append("Reliable high-speed home internet connectivity")

    # Domain Risk Severity Ratings (Academic, Engagement, Socioeconomic, Commute, Stress)
    acad_score = 0
    if acad_avg < 2.2: acad_score += 2
    if academic_trend < -0.2: acad_score += 1
    acad_rating = "High" if acad_score >= 2 else ("Moderate" if acad_score == 1 else "Low")

    eng_score = 0
    if attendance < 75.0: eng_score += 2
    if delay_days >= 3.0: eng_score += 1
    if study_hrs < 2.5: eng_score += 1
    eng_rating = "High" if eng_score >= 3 else ("Moderate" if eng_score >= 1 else "Low")

    fin_score = 0
    if scholarship == 'no': fin_score += 1
    if part_time == 'yes': fin_score += 1
    if income < 25000: fin_score += 1
    fin_rating = "High" if fin_score >= 2 else ("Moderate" if fin_score == 1 else "Low")

    acc_score = 1 if travel_time >= 45.0 else (1 if internet == 'no' else 0)
    acc_rating = "High" if (travel_time >= 50 and internet == 'no') else ("Moderate" if acc_score > 0 else "Low")

    stress_rating = "High" if stress >= 7.0 else ("Moderate" if stress >= 5.0 else "Low")

    domain_risks = {
        "Academic_Risk": acad_rating,
        "Engagement_Risk": eng_rating,
        "Financial_Social_Risk": fin_rating,
        "Accessibility_Risk": acc_rating,
        "Stress_Risk": stress_rating
    }

    # Actionable Interventions
    interventions = []
    if prob_dropout >= 0.60:
        interventions.append({
            "category": "Immediate Priority",
            "action": "Assign designated faculty retention mentor within 48 hours.",
            "rationale": "High estimated risk warrants immediate personal outreach."
        })
    if acad_rating == "High":
        interventions.append({
            "category": "Academic Mentoring",
            "action": "Enroll in subject-specific tutoring labs and midterm grade monitoring.",
            "rationale": "Compensate for low cumulative GPA and declining term trajectory."
        })
    if eng_rating == "High":
        interventions.append({
            "category": "Attendance & Habit Support",
            "action": "Implement bi-weekly check-ins with academic coach and automated attendance alerts.",
            "rationale": "Chronic absenteeism (<75%) and submission lags indicate disengagement."
        })
    if stress_rating == "High":
        interventions.append({
            "category": "Counseling & Well-Being",
            "action": "Refer to campus counseling center for time management and stress mitigation.",
            "rationale": "High psychological stress index impairs academic retention."
        })
    if fin_rating == "High":
        interventions.append({
            "category": "Financial Advisory",
            "action": "Initiate emergency student aid audit and campus work-study review.",
            "rationale": "Workforce commitments and financial strain compete with study time."
        })
    if acc_rating == "High":
        interventions.append({
            "category": "Commute & Logistics Support",
            "action": "Provide campus study lounge access and portable Wi-Fi hotspot support.",
            "rationale": "Long daily commute and internet gaps create substantial academic friction."
        })
    if not interventions:
        interventions.append({
            "category": "Routine Advisory",
            "action": "Maintain scheduled semester advising and degree milestone audit.",
            "rationale": "Student maintains robust academic indicators and low attrition likelihood."
        })

    return {
        "prediction": prediction,
        "prediction_label": prediction_label,
        "probability": round(prob_dropout, 4),
        "probability_percent": round(prob_dropout * 100, 1),
        "retained_percent": round(prob_retained * 100, 1),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "threshold": threshold,
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "domain_risks": domain_risks,
        "interventions": interventions,
        "student_summary": {
            "Academic_Average": round(acad_avg, 2),
            "Academic_Trend": round(academic_trend, 2),
            "Attendance_Rate": attendance,
            "Stress_Index": stress,
            "Study_Hours": study_hrs
        }
    }


if __name__ == "__main__":
    # Test sample profile
    test_student = {
        "Age": 21.0,
        "Gender": "Male",
        "Family_Income": 22000.0,
        "Internet_Access": "No",
        "Study_Hours_per_Day": 1.8,
        "Attendance_Rate": 64.0,
        "Assignment_Delay_Days": 4.0,
        "Travel_Time_Minutes": 55.0,
        "Part_Time_Job": "Yes",
        "Scholarship": "No",
        "Stress_Index": 8.5,
        "GPA": 1.45,
        "Semester_GPA": 1.10,
        "CGPA": 1.40,
        "Semester": "Year 2",
        "Department": "CS",
        "Parental_Education": "High School"
    }
    result = predict_student(test_student)
    print("Inference Test Result:")
    print(f"Prediction: {result['prediction_label']} ({result['probability_percent']}%)")
    print(f"Risk Tier:  {result['risk_level']}")
    print(f"Threshold:  {result['threshold']}")
    print("Risk Factors:", result['risk_factors'])
    print("Protective Factors:", result['protective_factors'])
    print("Domain Risks:", result['domain_risks'])
