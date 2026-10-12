# Student Dropout Risk Analyzer & Academic ML Framework

> **An Interactive Machine Learning Decision Support System with Real-Time Inference, Explainable AI (Tree SHAP), and MongoDB Enterprise Backend Connectivity.**

---

## 📌 Executive Summary

Higher education institutions face substantial retention challenges where student attrition occurs silently prior to formal withdrawal. **Student Dropout Risk Analyzer** is an end-to-end Machine Learning web application designed to detect early warning signs of student attrition, predict dropout propensity before formal disenrollment, and recommend actionable institutional mentoring directives.

### Key Highlights
- **Strict Academic Machine Learning Constraints:** Zero Deep Learning / Neural Networks; built on tree-based ensembles, penalized linear models, and probabilistic classifiers.
- **Zero Data Leakage Protocol:** Feature engineering, imputations, scalers, and encoding fitted exclusively on the training distribution.
- **Cost-Sensitive Threshold Tuning:** Optimization prioritizing sensitivity to minimize false negatives ($C_{FN} = \$5,000$ vs $C_{FP} = \$500$).
- **Explainable AI (XAI):** Integrated Tree SHAP (SHapley Additive exPlanations) for global feature attribution and individual student risk profiles.
- **Full Backend Connectivity:** Native MongoDB integration with automatic audit logging, student records CRUD, and real-time aggregation analytics.
- **REST API Suite:** Built-in Flask REST endpoints for headless programmatic inference, health checks, and data queries.

---

## 🏗️ Architecture & Technology Stack

| Layer | Technologies & Implementations |
| :--- | :--- |
| **Frontend / Dashboard** | Dash 2.x, Dash Bootstrap Components (Darkly theme), Plotly Interactive Graphics, Custom Glassmorphism CSS |
| **Backend & REST API** | Flask (in-process microframework), PyMongo native driver, RESTful endpoints (`/api/v1/*`) |
| **Database** | MongoDB Community Server (Database: `student_dropout_db`), collections for `predictions`, `students`, `model_metadata` |
| **Machine Learning Engine** | Scikit-learn (Random Forest, Gradient Boosting, Logistic Regression, etc.), SHAP |
| **Data & Preprocessing** | Pandas, NumPy, Scikit-learn ColumnTransformer & Pipeline |

---

## 🗄️ Backend Connectivity: MongoDB & REST API

### MongoDB Collections
1. **`predictions`**: Audit trail storing every inference execution with timestamp, full feature payload, estimated probability %, risk classification, and recommended action directives.
2. **`students`**: Student roster data with unique indexing and CRUD capabilities.
3. **`model_metadata`**: Versioned snapshots of trained model hyperparameters, validation metrics, and threshold cutoffs.

### REST API Endpoints
All endpoints run alongside the web dashboard on port `8050`:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service and MongoDB cluster health check |
| `GET` | `/api/v1/predictions` | Fetches recent prediction audit logs from MongoDB |
| `GET` | `/api/v1/stats` | Aggregated metrics computed via MongoDB `$group` pipeline |
| `POST` | `/api/v1/predict` | Headless JSON inference; auto-saves record to MongoDB |

#### Sample Prediction Request (`POST /api/v1/predict`)
```json
{
  "Age": 20,
  "Gender": "Male",
  "Family_Income": 25000,
  "Internet_Access": "No",
  "Study_Hours_per_Day": 2.0,
  "Attendance_Rate": 60.0,
  "Assignment_Delay_Days": 3.0,
  "Travel_Time_Minutes": 40.0,
  "Part_Time_Job": "Yes",
  "Scholarship": "No",
  "Stress_Index": 7.0,
  "GPA": 2.0,
  "Semester_GPA": 1.8,
  "CGPA": 2.0,
  "Semester": "Year 2",
  "Department": "CS",
  "Parental_Education": "High School"
}
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10+
- MongoDB Community Server installed and running locally on `localhost:27017`

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/student-dropout-risk-analyzer.git
cd student-dropout-risk-analyzer
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
Open your browser at: **`http://127.0.0.1:8050/`**

---

## 📊 Dashboard Modules

1. **Overview & KPIs:** Executive retention indicators, baseline attrition rate, and top classifier metrics.
2. **EDA Dashboard:** Univariate feature distributions, correlation matrices, and department-level demographic cross-tabs.
3. **Model Performance:** Side-by-side ROC-AUC curves, Precision-Recall curves, confusion matrices, and model comparison rankings.
4. **Threshold Tuning:** Cost curve analysis and interactive threshold slider demonstrating sensitivity vs specificity tradeoffs.
5. **Risk Predictor:** Real-time student profile evaluation with dynamic risk gauge, automatic MongoDB logging, and instant risk factor diagnostic flags.
6. **Profile & Insights:** Multi-domain radar diagram (Academic, Engagement, Financial, Well-being) and tailored institutional action directives.
7. **SHAP Explainability:** Tree SHAP global summary plot and multi-method feature selection ranking.
8. **MongoDB Backend:** Live connection status, document collection counters, interactive audit history table with live Seed/Clear controls, and MongoDB aggregation charts.

---

## 📄 License
This project is licensed under the MIT License.
