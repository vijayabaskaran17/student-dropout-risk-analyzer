"""
Student Dropout Risk Analyzer - Data Preprocessing & Feature Engineering Module
Contains Scikit-learn compliant transformers for domain feature engineering and ColumnTransformer creation.
Ensures ZERO DATA LEAKAGE by fitting transformations strictly on training distributions.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ENGINEERED_NUMERICAL_FEATURES,
    ALL_NUMERICAL_FEATURES,
    ALL_CATEGORICAL_FEATURES
)


class StudentFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible Feature Engineering Transformer.
    Generates domain-grounded indicators for academic momentum, engagement,
    stress vulnerability, commute friction, and compound disengagement.
    
    Zero-Leakage Assurance:
    Every calculation is strictly row-wise and relies solely on individual student
    attributes without cross-sample statistics (mean, variance, quantiles).
    """

    def __init__(self):
        self.engineered_features_ = ENGINEERED_NUMERICAL_FEATURES

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Work on a copy to prevent in-place mutation of caller DataFrame
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise TypeError("Expected pandas DataFrame as input to StudentFeatureEngineer.")

        # Ensure numeric columns are cast to float
        numeric_cols_needed = [
            'GPA', 'Semester_GPA', 'CGPA', 'Attendance_Rate',
            'Assignment_Delay_Days', 'Study_Hours_per_Day',
            'Stress_Index', 'Travel_Time_Minutes'
        ]
        for col in numeric_cols_needed:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 1. Academic_Average: Unweighted mean of GPA, Semester_GPA, and CGPA
        df['Academic_Average'] = (df['GPA'] + df['Semester_GPA'] + df['CGPA']) / 3.0

        # 2. Academic_Trend: Recent term momentum relative to cumulative history
        df['Academic_Trend'] = df['Semester_GPA'] - df['CGPA']

        # 3. Attendance_Risk: Binary indicator for critical institutional threshold (< 75%)
        df['Attendance_Risk'] = (df['Attendance_Rate'] < 75.0).astype(float)

        # 4. Assignment_Delay_Risk: Indicator for chronic assignment submission delay (>= 3 days)
        df['Assignment_Delay_Risk'] = (df['Assignment_Delay_Days'] >= 3.0).astype(float)

        # 5. Study_Engagement: Compound interaction of daily study hours and attendance
        df['Study_Engagement'] = df['Study_Hours_per_Day'] * (df['Attendance_Rate'] / 100.0)

        # 6. Stress_Academic_Vulnerability: Stress level scaled inversely by academic cushion (eps=0.1)
        df['Stress_Academic_Vulnerability'] = df['Stress_Index'] / (df['Academic_Average'].clip(lower=0.1) + 0.1)

        # 7. Accessibility_Travel_Risk: Indicator for excessive daily travel commute (>= 45 mins)
        df['Accessibility_Travel_Risk'] = (df['Travel_Time_Minutes'] >= 45.0).astype(float)

        # 8. Socioeconomic_Risk: Composite index from scholarship lack, part-time work, and no internet
        no_sch = (df['Scholarship'].astype(str).str.strip().str.lower() == 'no').astype(float)
        part_time = (df['Part_Time_Job'].astype(str).str.strip().str.lower() == 'yes').astype(float)
        no_net = (df['Internet_Access'].astype(str).str.strip().str.lower() == 'no').astype(float)
        df['Socioeconomic_Risk'] = no_sch * 0.35 + part_time * 0.35 + no_net * 0.30

        # --- Additional Meaningful Domain Features ---
        # 9. GPA_Stability: Absolute deviation between current semester and baseline GPA
        df['GPA_Stability'] = (df['Semester_GPA'] - df['GPA']).abs()

        # 10. Attendance_Study_Ratio: Ratio of classroom attendance to daily study hours
        df['Attendance_Study_Ratio'] = df['Attendance_Rate'] / (df['Study_Hours_per_Day'].clip(lower=0.1) + 0.5)

        # 11. Delay_Attendance_Interaction: Compound disengagement (high delay combined with low attendance)
        df['Delay_Attendance_Interaction'] = df['Assignment_Delay_Days'] * ((100.0 - df['Attendance_Rate']).clip(lower=0.0) / 100.0)

        # 12. Stress_Attendance_Interaction: Elevated stress manifested as classroom avoidance
        df['Stress_Attendance_Interaction'] = df['Stress_Index'] * ((100.0 - df['Attendance_Rate']).clip(lower=0.0) / 100.0)

        # 13. Stress_Study_Interaction: High psychological pressure relative to self-study coping capacity
        df['Stress_Study_Interaction'] = df['Stress_Index'] / (df['Study_Hours_per_Day'].clip(lower=0.1) + 0.5)

        # 14. Travel_Study_Burden: Ratio of daily travel minutes to study minutes
        df['Travel_Study_Burden'] = df['Travel_Time_Minutes'] / ((df['Study_Hours_per_Day'].clip(lower=0.1) * 60.0) + 15.0)

        # 15. Financial_Academic_Risk: Socioeconomic strain weighted against low academic cushion
        df['Financial_Academic_Risk'] = df['Socioeconomic_Risk'] / (df['Academic_Average'].clip(lower=0.1) + 0.1)

        # 16. Overall_Academic_Risk: Composite multi-factor academic vulnerability
        low_gpa_term = (4.0 - df['Academic_Average']).clip(lower=0.0) / 4.0
        neg_trend_term = (df['Academic_Trend'] < 0.0).astype(float) * 0.3
        vuln_term = (df['Stress_Academic_Vulnerability'] > 2.0).astype(float) * 0.3
        df['Overall_Academic_Risk'] = low_gpa_term * 0.4 + neg_trend_term + vuln_term

        # 17. Overall_Engagement_Risk: Composite index from attendance risk, delay risk, and low study hours
        low_study_term = (df['Study_Hours_per_Day'] < 2.5).astype(float) * 0.25
        df['Overall_Engagement_Risk'] = df['Attendance_Risk'] * 0.40 + df['Assignment_Delay_Risk'] * 0.35 + low_study_term

        return df


def build_preprocessor():
    """
    Constructs the Scikit-learn ColumnTransformer for full dataset preprocessing.
    - Numerical Pipeline: Median Imputer (robust to outliers) -> StandardScaler (mean=0, std=1)
    - Categorical Pipeline: Most Frequent Imputer -> OneHotEncoder(handle_unknown='ignore')
    """
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, ALL_NUMERICAL_FEATURES),
            ('cat', categorical_pipeline, ALL_CATEGORICAL_FEATURES)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    return preprocessor
