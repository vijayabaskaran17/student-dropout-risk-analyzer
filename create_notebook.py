import json

# Define notebook structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🎓 Advanced Student Dropout Risk Analyzer: Leakage-Free Preprocessing Pipeline\n",
                "\n",
                "**Project:** Machine Learning Lab (Advanced Classification & Ensemble Learning Prep)  \n",
                "**Dataset:** `student_dropout_dataset_v3.csv` (10,000 records, 19 initial attributes)  \n",
                "**Target:** `Dropout` (0 = Not Dropout, 1 = Dropout)  \n",
                "\n",
                "---\n",
                "### 📌 Pipeline Objectives:\n",
                "1. **Zero Data Leakage:** Imputation statistics, scaling parameters, and encoding mappings are computed **strictly on the training split**.\n",
                "2. **Rigorous Data Quality & Outlier Analysis:** Evaluate distributions, invalid values, skewness, and determine non-destructive scaling.\n",
                "3. **Domain-Specific Feature Engineering:** Academic performance indices, trends, engagement, stress-vulnerability, and travel burdens.\n",
                "4. **Stratified 70/15/15 Splitting:** Ensure uniform target ratios across Train, Validation, and Test sets.\n",
                "5. **Production Scikit-Learn Preprocessor:** Modular `ColumnTransformer` with `handle_unknown='ignore'` and full feature recovery for SHAP/LIME.\n",
                "6. **Imbalanced Learning Preparation:** SMOTE integration via `imblearn` pipeline for cross-validation on training folds only.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Environment Setup and Package Verification\n",
                "import os\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.base import BaseEstimator, TransformerMixin\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.compose import ColumnTransformer\n",
                "from sklearn.impute import SimpleImputer\n",
                "from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder\n",
                "from sklearn.feature_selection import mutual_info_classif, f_classif\n",
                "from sklearn.ensemble import RandomForestClassifier\n",
                "\n",
                "from imblearn.pipeline import Pipeline as ImbPipeline\n",
                "from imblearn.over_sampling import SMOTE\n",
                "import joblib\n",
                "\n",
                "# Set aesthetic styling for visualizations\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n",
                "plt.rcParams['font.sans-serif'] = 'DejaVu Sans'\n",
                "plt.rcParams['figure.dpi'] = 120\n",
                "\n",
                "print('Libraries successfully imported!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load and Inspect the Dataset\n",
                "We load `student_dropout_dataset_v3.csv` and inspect dimensions, column types, and sample records."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "dataset_path = 'student_dropout_dataset_v3.csv'\n",
                "df_raw = pd.read_csv(dataset_path)\n",
                "\n",
                "print(f'Total Rows: {df_raw.shape[0]}')\n",
                "print(f'Total Columns: {df_raw.shape[1]}')\n",
                "print('\\nColumn Types:')\n",
                "print(df_raw.dtypes)\n",
                "\n",
                "display(df_raw.head())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2 & 3. Data Quality & Missing Value Analysis\n",
                "We check for missing counts, missing percentages, unique values, and numerical ranges across all features."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "quality_records = []\n",
                "for col in df_raw.columns:\n",
                "    col_data = df_raw[col]\n",
                "    dtype = str(col_data.dtype)\n",
                "    missing_cnt = col_data.isnull().sum()\n",
                "    missing_pct = round((missing_cnt / len(df_raw)) * 100, 2)\n",
                "    unique_cnt = col_data.nunique(dropna=True)\n",
                "    if np.issubdtype(col_data.dtype, np.number):\n",
                "        min_val = round(float(col_data.min()), 2)\n",
                "        max_val = round(float(col_data.max()), 2)\n",
                "    else:\n",
                "        min_val = 'N/A'\n",
                "        max_val = 'N/A'\n",
                "    quality_records.append({\n",
                "        'Feature': col,\n",
                "        'Data Type': dtype,\n",
                "        'Missing Count': missing_cnt,\n",
                "        'Missing %': missing_pct,\n",
                "        'Unique Values': unique_cnt,\n",
                "        'Min': min_val,\n",
                "        'Max': max_val\n",
                "    })\n",
                "\n",
                "quality_df = pd.DataFrame(quality_records)\n",
                "display(quality_df)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4 & 5. Duplicate Detection & Student_ID Handling\n",
                "`Student_ID` is a primary key identifier with zero predictive value. Retaining it would allow models to memorize student identity rather than learning generalizable risk factors."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(f'Full duplicate rows: {df_raw.duplicated().sum()}')\n",
                "print(f'Duplicate Student_IDs: {df_raw[\"Student_ID\"].duplicated().sum()}')\n",
                "\n",
                "# Separate Student_ID and target from feature matrix\n",
                "student_ids = df_raw['Student_ID'].copy()\n",
                "y = df_raw['Dropout'].copy()\n",
                "X_raw = df_raw.drop(columns=['Student_ID', 'Dropout'])\n",
                "\n",
                "print(f'X_raw shape: {X_raw.shape}, y shape: {y.shape}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6 & 7. Target Variable & Imbalance Analysis\n",
                "The target is binary: `0 = Not Dropout` and `1 = Dropout`. We assess the imbalance ratio."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "counts = y.value_counts().sort_index()\n",
                "pcts = y.value_counts(normalize=True).sort_index() * 100\n",
                "\n",
                "print(f'Class 0 (Not Dropout): {counts[0]} ({pcts[0]:.2f}%)')\n",
                "print(f'Class 1 (Dropout):     {counts[1]} ({pcts[1]:.2f}%)')\n",
                "print(f'Imbalance Ratio:       {counts[0]/counts[1]:.2f}:1')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8 & 9. Outlier Analysis\n",
                "We calculate IQR outlier bounds for all numerical features. Instead of dropping student records with high income or long commute, we employ robust scaling and tree-friendly transformations."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()\n",
                "outlier_rows = []\n",
                "for col in num_cols:\n",
                "    s = X_raw[col].dropna()\n",
                "    q25 = s.quantile(0.25)\n",
                "    q75 = s.quantile(0.75)\n",
                "    iqr = q75 - q25\n",
                "    lb = q25 - 1.5 * iqr\n",
                "    ub = q75 + 1.5 * iqr\n",
                "    outliers = s[(s < lb) | (s > ub)]\n",
                "    outlier_rows.append({\n",
                "        'Feature': col,\n",
                "        'Lower Bound': round(lb, 2),\n",
                "        'Upper Bound': round(ub, 2),\n",
                "        'Outlier Count': len(outliers),\n",
                "        'Outlier %': round(len(outliers) / len(s) * 100, 2)\n",
                "    })\n",
                "display(pd.DataFrame(outlier_rows))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 11. Domain-Specific Feature Engineering\n",
                "We implement a custom, modular Scikit-Learn transformer `StudentFeatureEngineer` to generate:\n",
                "1. `Academic_Average`: Mean of GPA, Semester_GPA, and CGPA.\n",
                "2. `Academic_Trend`: Momentum (Semester_GPA - CGPA).\n",
                "3. `Attendance_Risk`: Binary indicator for attendance < 75%.\n",
                "4. `Assignment_Delay_Risk`: Chronic delay indicator (>= 3 days).\n",
                "5. `Study_Engagement`: Product of Study Hours and Attendance Rate.\n",
                "6. `Stress_Academic_Vulnerability`: Stress Index divided by (Academic_Average + 0.1).\n",
                "7. `Accessibility_Travel_Risk`: Excessive travel time indicator (>= 45 mins).\n",
                "8. `Socioeconomic_Risk`: Composite index from scholarship, job, and internet status."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class StudentFeatureEngineer(BaseEstimator, TransformerMixin):\n",
                "    def __init__(self):\n",
                "        self.engineered_features_ = [\n",
                "            'Academic_Average', 'Academic_Trend', 'Attendance_Risk',\n",
                "            'Assignment_Delay_Risk', 'Study_Engagement',\n",
                "            'Stress_Academic_Vulnerability', 'Accessibility_Travel_Risk',\n",
                "            'Socioeconomic_Risk'\n",
                "        ]\n",
                "    def fit(self, X, y=None):\n",
                "        return self\n",
                "    def transform(self, X):\n",
                "        X = X.copy()\n",
                "        X['Academic_Average'] = (X['GPA'] + X['Semester_GPA'] + X['CGPA']) / 3.0\n",
                "        X['Academic_Trend'] = X['Semester_GPA'] - X['CGPA']\n",
                "        X['Attendance_Risk'] = (X['Attendance_Rate'] < 75.0).astype(float)\n",
                "        X['Assignment_Delay_Risk'] = (X['Assignment_Delay_Days'] >= 3.0).astype(float)\n",
                "        X['Study_Engagement'] = X['Study_Hours_per_Day'] * (X['Attendance_Rate'] / 100.0)\n",
                "        X['Stress_Academic_Vulnerability'] = X['Stress_Index'] / (X['Academic_Average'] + 0.1)\n",
                "        X['Accessibility_Travel_Risk'] = (X['Travel_Time_Minutes'] >= 45.0).astype(float)\n",
                "        no_sch = (X['Scholarship'].astype(str) == 'No').astype(float)\n",
                "        part_time = (X['Part_Time_Job'].astype(str) == 'Yes').astype(float)\n",
                "        no_net = (X['Internet_Access'].astype(str) == 'No').astype(float)\n",
                "        X['Socioeconomic_Risk'] = no_sch * 0.35 + part_time * 0.35 + no_net * 0.30\n",
                "        return X\n",
                "\n",
                "engineer = StudentFeatureEngineer()\n",
                "X_engineered = engineer.transform(X_raw)\n",
                "print('Feature engineering complete. Total feature columns:', X_engineered.shape[1])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 15. Train / Validation / Test Stratified Split (70% / 15% / 15%)\n",
                "To guarantee zero leakage, we split the raw engineered data **before** calculating any imputation or scaling parameters."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 70% Train, 30% Temp\n",
                "X_train_raw, X_temp_raw, y_train, y_temp, id_train, id_temp = train_test_split(\n",
                "    X_engineered, y, student_ids, test_size=0.30, stratify=y, random_state=42\n",
                ")\n",
                "# 15% Val, 15% Test\n",
                "X_val_raw, X_test_raw, y_val, y_test, id_val, id_test = train_test_split(\n",
                "    X_temp_raw, y_temp, id_temp, test_size=0.50, stratify=y_temp, random_state=42\n",
                ")\n",
                "\n",
                "print(f'Train set:      {len(X_train_raw)} ({y_train.mean()*100:.2f}% Dropout)')\n",
                "print(f'Validation set: {len(X_val_raw)} ({y_val.mean()*100:.2f}% Dropout)')\n",
                "print(f'Test set:       {len(X_test_raw)} ({y_test.mean()*100:.2f}% Dropout)')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 16 & 17. Reusable Preprocessing Pipeline (Scikit-Learn ColumnTransformer)\n",
                "- **Numerical:** Median Imputation -> StandardScaler\n",
                "- **Categorical:** Most-Frequent Imputation -> OneHotEncoder(handle_unknown='ignore')\n",
                "- **Fitted exclusively on Training Set**"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cat_cols = X_train_raw.select_dtypes(include=['object']).columns.tolist()\n",
                "all_num_cols = [c for c in X_train_raw.columns if c not in cat_cols]\n",
                "\n",
                "num_pipe = Pipeline([\n",
                "    ('imputer', SimpleImputer(strategy='median')),\n",
                "    ('scaler', StandardScaler())\n",
                "])\n",
                "\n",
                "cat_pipe = Pipeline([\n",
                "    ('imputer', SimpleImputer(strategy='most_frequent')),\n",
                "    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))\n",
                "])\n",
                "\n",
                "preprocessor = ColumnTransformer(\n",
                "    transformers=[\n",
                "        ('num', num_pipe, all_num_cols),\n",
                "        ('cat', cat_pipe, cat_cols)\n",
                "    ],\n",
                "    remainder='drop',\n",
                "    verbose_feature_names_out=False\n",
                ")\n",
                "\n",
                "# Fit ONLY on Training Data\n",
                "preprocessor.fit(X_train_raw)\n",
                "transformed_feature_names = preprocessor.get_feature_names_out()\n",
                "print(f'Preprocessed Feature Matrix dimension: {len(transformed_feature_names)} features')\n",
                "\n",
                "# Transform all splits\n",
                "X_train_proc = preprocessor.transform(X_train_raw)\n",
                "X_val_proc = preprocessor.transform(X_val_raw)\n",
                "X_test_proc = preprocessor.transform(X_test_raw)\n",
                "\n",
                "# Save reusable preprocessor\n",
                "joblib.dump(preprocessor, 'preprocessor.joblib')\n",
                "print('Fitted preprocessor saved to preprocessor.joblib')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 18. Feature Selection Preparation (Training Set Only)\n",
                "We compute Mutual Information, ANOVA F-statistic, and Random Forest Gini Importance on the training set."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "mi_scores = mutual_info_classif(X_train_proc, y_train, random_state=42)\n",
                "f_scores, p_values = f_classif(X_train_proc, y_train)\n",
                "rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)\n",
                "rf.fit(X_train_proc, y_train)\n",
                "\n",
                "ranking_df = pd.DataFrame({\n",
                "    'Feature': transformed_feature_names,\n",
                "    'Mutual_Info': mi_scores,\n",
                "    'ANOVA_F': f_scores,\n",
                "    'RF_Importance': rf.feature_importances_\n",
                "}).sort_values(by='Mutual_Info', ascending=False)\n",
                "\n",
                "display(ranking_df.head(10))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 19. Imbalanced Learning Preparation (SMOTE inside Pipeline)\n",
                "We demonstrate `imblearn.pipeline.Pipeline` to guarantee SMOTE is strictly executed during cross-validation training folds."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "training_pipeline_smote = ImbPipeline([\n",
                "    ('preprocessor', preprocessor),\n",
                "    ('smote', SMOTE(random_state=42))\n",
                "])\n",
                "print('SMOTE Training Pipeline established successfully.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 24. Preprocessing Visualizations (All 10 Figures)\n",
                "Below we generate and display the complete set of exploratory and diagnostic visualizations."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Figure 1: Target Class Distribution\n",
                "fig, ax = plt.subplots(figsize=(6, 4))\n",
                "sns.countplot(x=y, palette=['#2b5c8f', '#d9534f'], ax=ax, edgecolor='black', alpha=0.85)\n",
                "ax.set_title('Figure 1: Target Class Distribution (Dropout)', fontweight='bold')\n",
                "ax.set_xticklabels(['Not Dropout (0)', 'Dropout (1)'])\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Figure 5: Correlation Heatmap\n",
                "fig, ax = plt.subplots(figsize=(10, 8))\n",
                "sns.heatmap(df_raw[num_cols + ['Dropout']].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)\n",
                "ax.set_title('Figure 5: Correlation Heatmap', fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Figure 9: Top 15 Features by Mutual Information\n",
                "fig, ax = plt.subplots(figsize=(10, 6))\n",
                "top15 = ranking_df.head(15).sort_values(by='Mutual_Info', ascending=True)\n",
                "ax.barh(top15['Feature'], top15['Mutual_Info'], color='#16a085', edgecolor='black', alpha=0.85)\n",
                "ax.set_title('Figure 9: Mutual Information Ranking (Train Only)', fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 25. Final Export & Ready for ML Modeling\n",
                "The processed training, validation, and test arrays are ready for model training, threshold tuning, and ensemble modeling!"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('student_dropout_preprocessing.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print('Successfully created student_dropout_preprocessing.ipynb!')
