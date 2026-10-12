# Academic Machine Learning Project Report: Student Dropout Risk Analyzer

## 1. Executive Summary & Problem Formulation
The **Student Dropout Risk Analyzer** is an end-to-end predictive modeling system developed to detect undergraduate students at risk of premature academic attrition. Operating on traditional Machine Learning paradigms, the project strictly excludes deep neural networks and cloud orchestration, prioritizing statistical rigor, model interpretability, and business-grounded cost-sensitive decision rules.

- **Total Student Population Analyzed**: 10,000
- **Baseline Retention**: 76.46% (Class 0: Not Dropout)
- **Baseline Attrition Rate**: 23.54% (Class 1: Dropout)
- **Empirical Class Imbalance**: 3.25:1
- **Optimal Decision Threshold**: 0.49 (Cost-optimized vs. default 0.50)
- **Top Selected Classifier**: Logistic Regression

---

## 2. Rigorous Preprocessing & Zero-Leakage Architecture
To ensure zero methodological data leakage:
1. `Student_ID` was isolated immediately prior to model ingestion.
2. Domain feature engineering functions were strictly calculated on row-level instances.
3. Stratified splitting allocated **70% Training (7,000)**, **15% Validation (1,500)**, and **15% Test (1,500)**.
4. Feature imputers (Median for numericals, Most-Frequent for categoricals) and Scalers (StandardScaler) were fitted solely on the training partition.
5. Imbalance oversampling (SMOTE) was applied exclusively to training folds.

---

## 3. Domain Feature Engineering
Seventeen domain-engineered attributes were constructed to represent complex student behaviors:
- **Academic Momentum**: `Academic_Average`, `Academic_Trend`, `GPA_Stability`
- **Institutional Compliance**: `Attendance_Risk` (<75%), `Assignment_Delay_Risk` (>=3 days)
- **Engagement Dynamics**: `Study_Engagement`, `Attendance_Study_Ratio`, `Delay_Attendance_Interaction`
- **Psychological & Environmental Stress**: `Stress_Academic_Vulnerability`, `Stress_Attendance_Interaction`, `Stress_Study_Interaction`
- **Commute & Socioeconomic Friction**: `Accessibility_Travel_Risk`, `Travel_Study_Burden`, `Socioeconomic_Risk`, `Financial_Academic_Risk`
- **Composite Risk Indices**: `Overall_Academic_Risk`, `Overall_Engagement_Risk`

---

## 4. Multi-Method Feature Selection Hierarchy
Top predictors ranked across Mutual Information, ANOVA F-Statistic, and Random Forest Gini Importance:
| Overall Rank | Feature | Mutual Information | ANOVA F-Score | RF Importance |
|:---:|:---|:---:|:---:|:---:|
| 1 | Academic_Average | 0.1105 | 1856.3 | 0.0803 |
| 2 | GPA | 0.1104 | 1916.0 | 0.0762 |
| 3 | CGPA | 0.1121 | 1770.8 | 0.0616 |
| 4 | Semester_GPA | 0.1062 | 1773.4 | 0.0547 |
| 5 | Stress_Academic_Vulnerability | 0.1047 | 1053.4 | 0.0614 |
| 6 | Overall_Academic_Risk | 0.1048 | 1040.0 | 0.0515 |
| 7 | Financial_Academic_Risk | 0.0735 | 691.3 | 0.0401 |
| 8 | Stress_Attendance_Interaction | 0.0340 | 639.8 | 0.0447 |
| 9 | Stress_Study_Interaction | 0.0375 | 291.2 | 0.0347 |
| 10 | Stress_Index | 0.0291 | 442.5 | 0.0315 |

---

## 5. Cost-Sensitive Model Performance Comparison (Untouched Test Partition)
Intervention cost is parameterized with **False Negative = 5** (unassisted dropout) and **False Positive = 1** (precautionary mentoring):

| Model                                |   Threshold |   Accuracy |   Precision |   Recall |     F1 |   ROC_AUC |   PR_AUC |   Specificity |   Cost | Is_Final   |
|:-------------------------------------|------------:|-----------:|------------:|---------:|-------:|----------:|---------:|--------------:|-------:|:-----------|
| Logistic Regression                  |        0.5  |     0.7447 |      0.4733 |   0.7535 | 0.5814 |    0.8222 |   0.5928 |        0.7419 |    731 | False      |
| Decision Tree                        |        0.5  |     0.7093 |      0.4281 |   0.6997 | 0.5312 |    0.7823 |   0.5443 |        0.7123 |    860 | False      |
| Random Forest                        |        0.5  |     0.7707 |      0.5097 |   0.6686 | 0.5784 |    0.8105 |   0.5654 |        0.8021 |    812 | False      |
| Gradient Boosting                    |        0.5  |     0.7547 |      0.4844 |   0.6601 | 0.5588 |    0.7967 |   0.5652 |        0.7838 |    848 | False      |
| Support Vector Machine               |        0.5  |     0.7887 |      0.5833 |   0.3569 | 0.4429 |    0.7929 |   0.5155 |        0.9215 |   1225 | False      |
| Stacking Ensemble                    |        0.5  |     0.7467 |      0.476  |   0.7592 | 0.5852 |    0.8222 |   0.5909 |        0.7428 |    720 | False      |
| Logistic Regression (Threshold 0.49) |        0.49 |     0.7427 |      0.471  |   0.7592 | 0.5813 |    0.8222 |   0.5928 |        0.7376 |    726 | True       |

---

## 6. Actionable Institutional Interventions
Predictions are partitioned into actionable intervention tiers:
- **Low Risk (0 - 30%)**: Standard advising and semester check-ins.
- **Moderate Risk (30 - 60%)**: Faculty advisor conference; peer tutoring encouragement.
- **High Risk (60 - 80%)**: Structured academic recovery plan, attendance tracking, and study hall requirement.
- **Critical Risk (80 - 100%)**: Immediate case worker assignment, student wellness counseling, and financial aid hardship review.
