"""
Student Dropout Risk Analyzer - Master Model Training & Evaluation Pipeline
=============================================================================
Academic Machine Learning Pipeline adhering to strict Zero-Leakage protocols:
1. Data Quality Analysis
2. Target Separation & Student_ID Exclusion
3. Domain Feature Engineering (17 domain risk indicators)
4. Stratified Split (70% Train, 15% Validation, 15% Test)
5. Preprocessing fit STRICTLY on Training Distribution
6. Imbalanced Learning Experiments (Baseline vs SMOTE vs Cost-Sensitive)
7. Multi-Method Feature Selection (Mutual Info, ANOVA F-Score, Tree Importance)
8. Multi-Model Training (Logistic Regression, Decision Tree, Random Forest,
   HistGradientBoosting, Support Vector Classifier, Stacking Classifier)
9. Stratified K-Fold Hyperparameter Tuning
10. Cost-Sensitive Threshold Optimization (FN=5, FP=1) on Validation Set
11. Final Evaluation on Untouched Test Set
12. Explainable AI with SHAP
13. Memory-Safe Publication Quality Plots Generation
14. Model Artifact Serialization for Dash UI Integration
"""

import os
import sys
import gc
import time
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE

import shap

from config import (
    DATA_PATH, PROCESSED_DATA_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR, REPORTS_DIR,
    ID_COL, TARGET_COL, COST_FN, COST_FP,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ENGINEERED_NUMERICAL_FEATURES,
    FINAL_MODEL_PATH, PREPROCESSOR_PATH, MODEL_METADATA_PATH,
    FEATURE_RANKING_PATH, SHAP_DATA_PATH
)
from pipeline_preprocessing import StudentFeatureEngineer, build_preprocessor


def run_training_pipeline():
    print("=" * 80, flush=True)
    print("STUDENT DROPOUT RISK ANALYZER - ACADEMIC ML PIPELINE", flush=True)
    print("=" * 80, flush=True)

    # --------------------------------------------------------------------------
    # 1. LOAD DATASET & QUALITY CHECK
    # --------------------------------------------------------------------------
    print("\n[Step 1/13] Loading Raw Dataset & Executing Data Quality Audits...", flush=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df_raw = pd.read_csv(DATA_PATH)
    n_rows, n_cols = df_raw.shape
    print(f"-> Loaded {n_rows:,} records with {n_cols} attributes.", flush=True)

    missing_summary = df_raw.isnull().sum()
    missing_pct = (missing_summary / n_rows) * 100
    dup_rows = df_raw.duplicated().sum()
    dup_ids = df_raw[ID_COL].duplicated().sum()

    print(f"-> Duplicate records: {dup_rows} | Duplicate IDs: {dup_ids}", flush=True)
    cols_with_na = missing_summary[missing_summary > 0]
    print(f"-> Columns with missing values: {len(cols_with_na)} ({dict(cols_with_na)})", flush=True)

    quality_df = pd.DataFrame({
        'Feature': df_raw.columns,
        'Dtype': df_raw.dtypes.astype(str),
        'Missing_Count': missing_summary.values,
        'Missing_Pct': missing_pct.round(2).values,
        'Unique_Count': [df_raw[c].nunique() for c in df_raw.columns]
    })
    quality_df.to_csv(METRICS_DIR / "data_quality_report.csv", index=False)
    gc.collect()

    # --------------------------------------------------------------------------
    # 2. TARGET SEPARATION & LEAKAGE PREVENTION
    # --------------------------------------------------------------------------
    print("\n[Step 2/13] Target Separation & Zero-Leakage Enforcement...", flush=True)
    student_ids = df_raw[ID_COL].copy()
    y = df_raw[TARGET_COL].copy().astype(int)
    X_raw = df_raw.drop(columns=[ID_COL, TARGET_COL])

    class_counts = y.value_counts().to_dict()
    minority_pct = (class_counts.get(1, 0) / n_rows) * 100
    imbalance_ratio = round(class_counts.get(0, 1) / max(class_counts.get(1, 1), 1), 2)
    print(f"-> Class 0 (Retained): {class_counts.get(0, 0):,} ({100 - minority_pct:.2f}%)", flush=True)
    print(f"-> Class 1 (Dropout):  {class_counts.get(1, 0):,} ({minority_pct:.2f}%)", flush=True)
    print(f"-> Imbalance Ratio: {imbalance_ratio}:1", flush=True)

    # --------------------------------------------------------------------------
    # 3. FEATURE ENGINEERING
    # --------------------------------------------------------------------------
    print("\n[Step 3/13] Executing Domain Feature Engineering (17 Risk Indicators)...", flush=True)
    feature_engineer = StudentFeatureEngineer()
    X_engineered = feature_engineer.transform(X_raw)
    print(f"-> Engineered features generated successfully. Total features: {X_engineered.shape[1]}", flush=True)

    # --------------------------------------------------------------------------
    # 4. STRATIFIED TRAIN / VALIDATION / TEST SPLIT (70% / 15% / 15%)
    # --------------------------------------------------------------------------
    print("\n[Step 4/13] Performing Stratified Partitioning (70% Train, 15% Val, 15% Test)...", flush=True)
    X_train_raw, X_temp_raw, y_train, y_temp, id_train, id_temp = train_test_split(
        X_engineered, y, student_ids, test_size=0.30, stratify=y, random_state=42
    )
    X_val_raw, X_test_raw, y_val, y_test, id_val, id_test = train_test_split(
        X_temp_raw, y_temp, id_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    del X_temp_raw, y_temp, id_temp
    gc.collect()

    print(f"-> Train Set:      {len(X_train_raw):,} samples (Dropout rate: {y_train.mean()*100:.2f}%)", flush=True)
    print(f"-> Validation Set: {len(X_val_raw):,} samples (Dropout rate: {y_val.mean()*100:.2f}%)", flush=True)
    print(f"-> Test Set:       {len(X_test_raw):,} samples (Dropout rate: {y_test.mean()*100:.2f}%)", flush=True)

    # --------------------------------------------------------------------------
    # 5. FIT PREPROCESSING STRICTLY ON TRAINING DATA (ZERO DATA LEAKAGE)
    # --------------------------------------------------------------------------
    print("\n[Step 5/13] Fitting ColumnTransformer on Training Data ONLY...", flush=True)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train_raw)
    feature_names = preprocessor.get_feature_names_out()
    print(f"-> Encoded/Scaled Feature Matrix Columns: {len(feature_names)}", flush=True)

    X_train_proc = preprocessor.transform(X_train_raw).astype(np.float32)
    X_val_proc = preprocessor.transform(X_val_raw).astype(np.float32)
    X_test_proc = preprocessor.transform(X_test_raw).astype(np.float32)

    # Memory-safe batch CSV saving
    def save_processed_partition(X_mat, y_series, id_series, path):
        df_p = pd.DataFrame(X_mat, columns=feature_names)
        df_p[TARGET_COL] = y_series.values
        df_p[ID_COL] = id_series.values
        df_p.to_csv(path, float_format='%.3f', chunksize=1000, index=False)
        del df_p
        gc.collect()

    save_processed_partition(X_train_proc, y_train, id_train, PROCESSED_DATA_DIR / "train_processed.csv")
    save_processed_partition(X_val_proc, y_val, id_val, PROCESSED_DATA_DIR / "val_processed.csv")
    save_processed_partition(X_test_proc, y_test, id_test, PROCESSED_DATA_DIR / "test_processed.csv")

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"-> Preprocessor and CSV partitions successfully persisted.", flush=True)

    # --------------------------------------------------------------------------
    # 6. FEATURE SELECTION (TRAINING DATA ONLY)
    # --------------------------------------------------------------------------
    print("\n[Step 6/13] Running Multi-Method Feature Selection (MI, ANOVA F, Tree Importance)...", flush=True)
    mi_scores = mutual_info_classif(X_train_proc, y_train, random_state=42)
    f_scores, p_values = f_classif(X_train_proc, y_train)

    rf_selector = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=1)
    rf_selector.fit(X_train_proc, y_train)
    rf_importances = rf_selector.feature_importances_

    feature_ranking_df = pd.DataFrame({
        'Feature': feature_names,
        'Mutual_Information': mi_scores,
        'ANOVA_F_Score': f_scores,
        'ANOVA_p_value': p_values,
        'RF_Importance': rf_importances
    })

    feature_ranking_df['Rank_MI'] = feature_ranking_df['Mutual_Information'].rank(ascending=False)
    feature_ranking_df['Rank_F'] = feature_ranking_df['ANOVA_F_Score'].rank(ascending=False)
    feature_ranking_df['Rank_RF'] = feature_ranking_df['RF_Importance'].rank(ascending=False)
    feature_ranking_df['Overall_Rank'] = (
        feature_ranking_df['Rank_MI'] + feature_ranking_df['Rank_F'] + feature_ranking_df['Rank_RF']
    ) / 3.0
    feature_ranking_df = feature_ranking_df.sort_values(by='Overall_Rank', ascending=True).reset_index(drop=True)
    feature_ranking_df.to_csv(METRICS_DIR / "feature_ranking.csv", index=False)
    joblib.dump(feature_ranking_df, FEATURE_RANKING_PATH)

    print("-> Top 5 Selected Features:", flush=True)
    for _, row in feature_ranking_df.head(5).iterrows():
        print(f"   * {row['Feature']}: MI={row['Mutual_Information']:.4f}, F={row['ANOVA_F_Score']:.1f}, RF_Imp={row['RF_Importance']:.4f}", flush=True)
    gc.collect()

    # --------------------------------------------------------------------------
    # 7. IMBALANCED DATA EXPERIMENTATION (TRAINING ONLY)
    # --------------------------------------------------------------------------
    print("\n[Step 7/13] Evaluating Imbalanced Handling Techniques on Validation Set...", flush=True)
    # Baseline
    rf_base = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=1)
    rf_base.fit(X_train_proc, y_train)
    p_base = rf_base.predict_proba(X_val_proc)[:, 1]
    y_pred_base = (p_base >= 0.5).astype(int)
    cm_base = confusion_matrix(y_val, y_pred_base)
    cost_base = COST_FN * cm_base[1, 0] + COST_FP * cm_base[0, 1]

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_proc, y_train)
    rf_smote = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=1)
    rf_smote.fit(X_train_smote, y_train_smote)
    p_smote = rf_smote.predict_proba(X_val_proc)[:, 1]
    y_pred_smote = (p_smote >= 0.5).astype(int)
    cm_smote = confusion_matrix(y_val, y_pred_smote)
    cost_smote = COST_FN * cm_smote[1, 0] + COST_FP * cm_smote[0, 1]

    # Cost-Sensitive
    rf_weighted = RandomForestClassifier(n_estimators=80, class_weight='balanced', random_state=42, n_jobs=1)
    rf_weighted.fit(X_train_proc, y_train)
    p_weight = rf_weighted.predict_proba(X_val_proc)[:, 1]
    y_pred_weight = (p_weight >= 0.5).astype(int)
    cm_weight = confusion_matrix(y_val, y_pred_weight)
    cost_weight = COST_FN * cm_weight[1, 0] + COST_FP * cm_weight[0, 1]

    imbalance_comp_df = pd.DataFrame([
        {
            'Strategy': '1. Baseline (No Balancing)',
            'Recall': round(recall_score(y_val, y_pred_base), 4),
            'Precision': round(precision_score(y_val, y_pred_base), 4),
            'F1': round(f1_score(y_val, y_pred_base), 4),
            'ROC_AUC': round(roc_auc_score(y_val, p_base), 4),
            'Cost (FN=5, FP=1)': float(cost_base)
        },
        {
            'Strategy': '2. SMOTE (Train Set Only)',
            'Recall': round(recall_score(y_val, y_pred_smote), 4),
            'Precision': round(precision_score(y_val, y_pred_smote), 4),
            'F1': round(f1_score(y_val, y_pred_smote), 4),
            'ROC_AUC': round(roc_auc_score(y_val, p_smote), 4),
            'Cost (FN=5, FP=1)': float(cost_smote)
        },
        {
            'Strategy': '3. Cost-Sensitive (class_weight="balanced")',
            'Recall': round(recall_score(y_val, y_pred_weight), 4),
            'Precision': round(precision_score(y_val, y_pred_weight), 4),
            'F1': round(f1_score(y_val, y_pred_weight), 4),
            'ROC_AUC': round(roc_auc_score(y_val, p_weight), 4),
            'Cost (FN=5, FP=1)': float(cost_weight)
        }
    ])
    imbalance_comp_df.to_csv(METRICS_DIR / "imbalance_comparison.csv", index=False)
    print(imbalance_comp_df.to_string(index=False), flush=True)

    del rf_base, rf_smote, rf_weighted
    gc.collect()

    # --------------------------------------------------------------------------
    # 8. MULTI-MODEL CANDIDATE TRAINING & TUNING
    # --------------------------------------------------------------------------
    print("\n[Step 8/13] Training and Comparing 6 Diverse Machine Learning Models...", flush=True)
    models_dict = {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, class_weight='balanced', random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=10, class_weight='balanced', random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=4,
            class_weight='balanced', random_state=42, n_jobs=1
        ),
        'Gradient Boosting': HistGradientBoostingClassifier(
            max_iter=100, max_depth=6, min_samples_leaf=10,
            class_weight='balanced', random_state=42
        ),
        'Support Vector Machine': SVC(
            C=1.0, kernel='rbf', probability=True,
            class_weight='balanced', random_state=42, cache_size=50, max_iter=2000
        )
    }

    # Stacking Ensemble
    estimators_for_stack = [
        ('rf', RandomForestClassifier(n_estimators=50, max_depth=8, class_weight='balanced', random_state=42, n_jobs=1)),
        ('gb', HistGradientBoostingClassifier(max_iter=50, max_depth=5, class_weight='balanced', random_state=42)),
        ('lr', LogisticRegression(C=0.5, max_iter=500, class_weight='balanced', random_state=42))
    ]
    models_dict['Stacking Ensemble'] = StackingClassifier(
        estimators=estimators_for_stack,
        final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
        cv=3,
        n_jobs=1
    )

    trained_models = {}
    val_predictions = {}
    val_probs = {}

    for name, model in models_dict.items():
        t0 = time.time()
        # For SVC, fit on stratified 3000 subset to ensure high speed and low memory
        if name == 'Support Vector Machine':
            sub_idx, _ = train_test_split(np.arange(len(y_train)), train_size=3000, stratify=y_train, random_state=42)
            model.fit(X_train_proc[sub_idx], y_train.iloc[sub_idx])
        else:
            model.fit(X_train_proc, y_train)

        fit_time = round(time.time() - t0, 2)
        trained_models[name] = model

        probs = model.predict_proba(X_val_proc)[:, 1]
        preds = (probs >= 0.5).astype(int)
        val_probs[name] = probs
        val_predictions[name] = preds

        acc = accuracy_score(y_val, preds)
        rec = recall_score(y_val, preds)
        f1 = f1_score(y_val, preds)
        roc = roc_auc_score(y_val, probs)
        print(f"-> {name:<22} (Fit: {fit_time}s) | Acc: {acc:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | ROC-AUC: {roc:.3f}", flush=True)

    # Hyperparameter Tuning on Random Forest
    print("\n-> Running Stratified K-Fold CV Tuning on Random Forest...", flush=True)
    rf_grid_params = {
        'n_estimators': [80, 120],
        'max_depth': [8, 12]
    }
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1),
        rf_grid_params,
        cv=cv_strategy,
        scoring='f1',
        n_jobs=1
    )
    grid_search.fit(X_train_proc, y_train)
    print(f"   * Best RF Parameters: {grid_search.best_params_} (Mean CV F1: {grid_search.best_score_:.4f})", flush=True)
    trained_models['Random Forest'] = grid_search.best_estimator_
    val_probs['Random Forest'] = grid_search.best_estimator_.predict_proba(X_val_proc)[:, 1]
    val_predictions['Random Forest'] = (val_probs['Random Forest'] >= 0.5).astype(int)
    gc.collect()

    # --------------------------------------------------------------------------
    # 9. COST-SENSITIVE THRESHOLD OPTIMIZATION (VALIDATION SET ONLY)
    # --------------------------------------------------------------------------
    print("\n[Step 9/13] Conducting Cost-Sensitive Threshold Optimization (0.10 to 0.90)...", flush=True)
    model_val_scores = {
        name: f1_score(y_val, val_predictions[name]) + roc_auc_score(y_val, val_probs[name])
        for name in trained_models
    }
    best_model_name = max(model_val_scores, key=model_val_scores.get)
    print(f"-> Primary candidate for threshold optimization: [{best_model_name}]", flush=True)

    candidate_probs = val_probs[best_model_name]
    threshold_steps = np.linspace(0.10, 0.90, 81)
    threshold_records = []

    best_thresh = 0.50
    min_cost = float('inf')

    for t in threshold_steps:
        preds_t = (candidate_probs >= t).astype(int)
        cm = confusion_matrix(y_val, preds_t)
        tn, fp, fn, tp = cm.ravel()
        prec = precision_score(y_val, preds_t, zero_division=0)
        rec = recall_score(y_val, preds_t, zero_division=0)
        f1 = f1_score(y_val, preds_t, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        total_cost = COST_FN * fn + COST_FP * fp

        record = {
            'Threshold': round(float(t), 3),
            'Precision': round(float(prec), 4),
            'Recall': round(float(rec), 4),
            'F1': round(float(f1), 4),
            'Specificity': round(float(spec), 4),
            'TP': int(tp),
            'FP': int(fp),
            'TN': int(tn),
            'FN': int(fn),
            'Total_Cost': float(total_cost)
        }
        threshold_records.append(record)

        if total_cost < min_cost:
            min_cost = total_cost
            best_thresh = round(float(t), 3)

    threshold_df = pd.DataFrame(threshold_records)
    threshold_df.to_csv(METRICS_DIR / "threshold_optimization_curve.csv", index=False)
    std_cost_val = threshold_df.loc[threshold_df['Threshold'] == 0.50, 'Total_Cost'].values[0]
    print(f"-> Standard Threshold (0.50) Cost on Validation: {std_cost_val}", flush=True)
    print(f"-> Optimal Cost-Sensitive Threshold: [{best_thresh}] with Total Cost: {min_cost} (Cost reduction: {std_cost_val - min_cost:.1f})", flush=True)

    # --------------------------------------------------------------------------
    # 10. FINAL EVALUATION ON UNTOUCHED TEST SET
    # --------------------------------------------------------------------------
    print("\n[Step 10/13] Evaluating All Models on Untouched Test Set...", flush=True)
    test_results = []
    roc_curves_data = {}
    pr_curves_data = {}
    confusion_matrices = {}

    for name, model in trained_models.items():
        test_prob = model.predict_proba(X_test_proc)[:, 1]
        test_pred_std = (test_prob >= 0.50).astype(int)

        cm = confusion_matrix(y_test, test_pred_std)
        tn, fp, fn, tp = cm.ravel()
        cost = COST_FN * fn + COST_FP * fp

        acc = accuracy_score(y_test, test_pred_std)
        prec = precision_score(y_test, test_pred_std)
        rec = recall_score(y_test, test_pred_std)
        f1 = f1_score(y_test, test_pred_std)
        roc_auc = roc_auc_score(y_test, test_prob)
        pr_auc = average_precision_score(y_test, test_prob)
        spec = tn / (tn + fp)

        test_results.append({
            'Model': name,
            'Threshold': 0.50,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'ROC_AUC': round(roc_auc, 4),
            'PR_AUC': round(pr_auc, 4),
            'Specificity': round(spec, 4),
            'Cost': float(cost),
            'Is_Final': False
        })

        fpr, tpr, _ = roc_curve(y_test, test_prob)
        roc_curves_data[name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': round(roc_auc, 4)}

        p_prec, p_rec, _ = precision_recall_curve(y_test, test_prob)
        pr_curves_data[name] = {'precision': p_prec.tolist(), 'recall': p_rec.tolist(), 'pr_auc': round(pr_auc, 4)}

        confusion_matrices[name] = cm.tolist()

    # Optimal Threshold Evaluation
    best_model = trained_models[best_model_name]
    best_test_prob = best_model.predict_proba(X_test_proc)[:, 1]
    best_test_pred_opt = (best_test_prob >= best_thresh).astype(int)
    cm_opt = confusion_matrix(y_test, best_test_pred_opt)
    tn_o, fp_o, fn_o, tp_o = cm_opt.ravel()
    cost_opt = COST_FN * fn_o + COST_FP * fp_o

    test_results.append({
        'Model': f"{best_model_name} (Threshold {best_thresh})",
        'Threshold': best_thresh,
        'Accuracy': round(accuracy_score(y_test, best_test_pred_opt), 4),
        'Precision': round(precision_score(y_test, best_test_pred_opt), 4),
        'Recall': round(recall_score(y_test, best_test_pred_opt), 4),
        'F1': round(f1_score(y_test, best_test_pred_opt), 4),
        'ROC_AUC': round(roc_auc_score(y_test, best_test_prob), 4),
        'PR_AUC': round(average_precision_score(y_test, best_test_prob), 4),
        'Specificity': round(tn_o / (tn_o + fp_o), 4),
        'Cost': float(cost_opt),
        'Is_Final': True
    })
    confusion_matrices[f"{best_model_name} (Optimized)"] = cm_opt.tolist()

    model_comp_df = pd.DataFrame(test_results)
    model_comp_df.to_csv(METRICS_DIR / "model_comparison_table.csv", index=False)
    print("\n--- Model Test Performance Comparison ---", flush=True)
    print(model_comp_df[['Model', 'Threshold', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'PR_AUC', 'Cost']].to_string(index=False), flush=True)

    # --------------------------------------------------------------------------
    # 11. EXPLAINABLE MACHINE LEARNING (SHAP)
    # --------------------------------------------------------------------------
    print("\n[Step 11/13] Generating Explainable AI Artifacts with SHAP...", flush=True)
    shap_model_ref = trained_models.get('Random Forest', best_model)
    bg_sample_idx = np.random.choice(X_train_proc.shape[0], size=min(80, X_train_proc.shape[0]), replace=False)
    X_background = X_train_proc[bg_sample_idx]
    val_sample_idx = np.random.choice(X_val_proc.shape[0], size=min(150, X_val_proc.shape[0]), replace=False)
    X_val_sample = X_val_proc[val_sample_idx]

    try:
        explainer = shap.TreeExplainer(shap_model_ref, data=X_background)
        shap_values = explainer.shap_values(X_val_sample)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_matrix = np.array(shap_values[1])
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_values_matrix = shap_values[:, :, 1]
        else:
            shap_values_matrix = np.array(shap_values)

        mean_abs_shap = np.mean(np.abs(shap_values_matrix), axis=0)
        shap_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean_Abs_SHAP': mean_abs_shap
        }).sort_values(by='Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

        print("-> Top 5 Features by SHAP Impact:", flush=True)
        for _, r in shap_importance_df.head(5).iterrows():
            print(f"   * {r['Feature']}: {r['Mean_Abs_SHAP']:.4f}", flush=True)

        shap_data_payload = {
            'feature_names': feature_names.tolist(),
            'shap_importance': shap_importance_df.to_dict(orient='records'),
            'sample_shap_values': shap_values_matrix[:40].tolist(),
            'sample_data': X_val_sample[:40].tolist()
        }
        joblib.dump(shap_data_payload, SHAP_DATA_PATH)
        print("-> SHAP artifact saved to models/shap_data.joblib.", flush=True)
    except Exception as e:
        print(f"-> Note on SHAP generation: {e}. Falling back to Random Forest Importances.", flush=True)
        mean_abs_shap = rf_selector.feature_importances_
        shap_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean_Abs_SHAP': mean_abs_shap
        }).sort_values(by='Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
        joblib.dump({'feature_names': feature_names.tolist(), 'shap_importance': shap_importance_df.to_dict(orient='records')}, SHAP_DATA_PATH)

    # --------------------------------------------------------------------------
    # 12. GENERATE ALL 17 PUBLICATION QUALITY VISUALIZATIONS
    # --------------------------------------------------------------------------
    print("\n[Step 12/13] Generating All Required Static Publication Plots...", flush=True)
    generate_all_plots(
        df_raw, X_engineered, y_train, y_train_smote,
        feature_ranking_df, threshold_df, best_thresh,
        roc_curves_data, pr_curves_data, confusion_matrices,
        model_comp_df, shap_importance_df, PLOTS_DIR
    )

    # --------------------------------------------------------------------------
    # 13. SERIALIZATION OF MASTER ARTIFACTS
    # --------------------------------------------------------------------------
    print("\n[Step 13/13] Serializing Final Model and Comprehensive Metadata...", flush=True)
    joblib.dump(best_model, FINAL_MODEL_PATH)

    metadata = {
        'best_model_name': best_model_name,
        'selected_threshold': float(best_thresh),
        'default_threshold': 0.50,
        'cost_fn': COST_FN,
        'cost_fp': COST_FP,
        'total_dataset_size': int(n_rows),
        'dropout_rate': float(minority_pct),
        'imbalance_ratio': float(imbalance_ratio),
        'train_samples': int(len(X_train_raw)),
        'val_samples': int(len(X_val_raw)),
        'test_samples': int(len(X_test_raw)),
        'feature_names': feature_names.tolist(),
        'model_comparison': model_comp_df.to_dict(orient='records'),
        'imbalance_comparison': imbalance_comp_df.to_dict(orient='records'),
        'threshold_curve': threshold_df.to_dict(orient='records'),
        'roc_curves': roc_curves_data,
        'pr_curves': pr_curves_data,
        'confusion_matrices': confusion_matrices,
        'feature_ranking': feature_ranking_df.to_dict(orient='records')
    }
    joblib.dump(metadata, MODEL_METADATA_PATH)

    generate_academic_report(metadata, quality_df, model_comp_df, feature_ranking_df, REPORTS_DIR)

    print("\n" + "=" * 80, flush=True)
    print("ALL TRAINING, EVALUATION, AND SERIALIZATION TASKS COMPLETED SUCCESSFULLY!", flush=True)
    print(f"Artifacts stored in: {MODELS_DIR}", flush=True)
    print(f"Plots stored in:     {PLOTS_DIR}", flush=True)
    print(f"Reports stored in:   {REPORTS_DIR}", flush=True)
    print("=" * 80, flush=True)
    return metadata


def generate_all_plots(df_raw, X_engineered, y_train, y_train_smote,
                       feature_ranking_df, threshold_df, best_thresh,
                       roc_curves_data, pr_curves_data, confusion_matrices,
                       model_comp_df, shap_importance_df, plots_dir):
    """Generates all 17 distinct visual charts for the project."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['figure.dpi'] = 120

    # 1. Target Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df_raw[TARGET_COL].value_counts().sort_index()
    bars = ax.bar(['Retained (0)', 'Dropout (1)'], counts.values, color=['#2b5c8f', '#d9534f'], edgecolor='black')
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 80, f"{b.get_height():,} ({b.get_height()/len(df_raw)*100:.1f}%)", ha='center', fontweight='bold')
    ax.set_title('Target Distribution: Student Dropout', fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "01_target_distribution.png")
    plt.close('all')

    # 2. Missing Values
    fig, ax = plt.subplots(figsize=(8, 4))
    mv = df_raw.isnull().sum()
    mv = mv[mv > 0]
    if len(mv) > 0:
        ax.barh(mv.index, mv.values, color='#e67e22', edgecolor='black')
        for i, (k, v) in enumerate(mv.items()):
            ax.text(v + 5, i, f"{v:,} ({v/len(df_raw)*100:.1f}%)", va='center')
    ax.set_title('Missing Values by Feature', fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "02_missing_values.png")
    plt.close('all')

    # 3. Numerical Distributions
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    axes = axes.flatten()
    for i, col in enumerate(NUMERICAL_FEATURES):
        sns.histplot(df_raw[col].dropna(), kde=True, ax=axes[i], color='#3498db', edgecolor='black')
        axes[i].set_title(col, fontweight='bold', fontsize=8)
    plt.suptitle('Distributions of Raw Numerical Features', fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "03_numerical_distributions.png")
    plt.close('all')

    # 4. Numerical Boxplots (Outliers)
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    axes = axes.flatten()
    for i, col in enumerate(NUMERICAL_FEATURES):
        sns.boxplot(y=df_raw[col].dropna(), ax=axes[i], color='#9b59b6')
        axes[i].set_title(col, fontweight='bold', fontsize=8)
    plt.suptitle('Boxplots for Outlier Inspection', fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "04_numerical_boxplots.png")
    plt.close('all')

    # 5. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df_raw[NUMERICAL_FEATURES + [TARGET_COL]].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title('Correlation Heatmap: Numerical Features & Target', fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "05_correlation_heatmap.png")
    plt.close('all')

    # 6. Categorical Distributions
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()
    for i, col in enumerate(CATEGORICAL_FEATURES):
        counts = df_raw[col].value_counts()
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=axes[i], palette='viridis', edgecolor='black')
        axes[i].set_title(col, fontweight='bold', fontsize=9)
        axes[i].tick_params(axis='x', rotation=30)
    axes[7].axis('off')
    plt.suptitle('Categorical Feature Distributions', fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "06_categorical_distributions.png")
    plt.close('all')

    # 7. Class Distribution Before Balancing
    fig, ax = plt.subplots(figsize=(6, 4))
    tb = pd.Series(y_train).value_counts().sort_index()
    ax.bar(['Retained (0)', 'Dropout (1)'], tb.values, color=['#2980b9', '#c0392b'], edgecolor='black')
    ax.set_title('Train Class Distribution (Before Balancing)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "07_class_dist_before_balancing.png")
    plt.close('all')

    # 8. Class Distribution After SMOTE
    fig, ax = plt.subplots(figsize=(6, 4))
    ta = pd.Series(y_train_smote).value_counts().sort_index()
    ax.bar(['Retained (0)', 'Dropout (1)'], ta.values, color=['#27ae60', '#27ae60'], edgecolor='black')
    ax.set_title('Train Class Distribution (After SMOTE Oversampling)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "08_class_dist_after_smote.png")
    plt.close('all')

    # 9. Feature Importance (Mutual Info & ANOVA)
    fig, ax = plt.subplots(figsize=(9, 5))
    top_feats = feature_ranking_df.head(12).sort_values(by='Mutual_Information', ascending=True)
    ax.barh(top_feats['Feature'], top_feats['Mutual_Information'], color='#16a085', edgecolor='black')
    ax.set_title('Top 12 Features Ranked by Mutual Information', fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "09_feature_importance.png")
    plt.close('all')

    # 10. Engineered Feature Distributions
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    axes = axes.flatten()
    eng_subset = ENGINEERED_NUMERICAL_FEATURES[:12]
    for i, col in enumerate(eng_subset):
        sns.histplot(X_engineered[col].dropna(), kde=True, ax=axes[i], color='#e74c3c', edgecolor='black')
        axes[i].set_title(col, fontweight='bold', fontsize=8)
    plt.suptitle('Key Engineered Risk Indicators Distributions', fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "10_engineered_features_dist.png")
    plt.close('all')

    # 11. ROC Curves
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, data in roc_curves_data.items():
        ax.plot(data['fpr'], data['tpr'], label=f"{name} (AUC={data['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', label='Chance (AUC=0.500)')
    ax.set_title('Test Set ROC Curves: Candidate Model Comparison', fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "11_roc_curves.png")
    plt.close('all')

    # 12. PR Curves
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, data in pr_curves_data.items():
        ax.plot(data['recall'], data['precision'], label=f"{name} (PR-AUC={data['pr_auc']:.3f})")
    ax.set_title('Test Set Precision-Recall Curves', fontweight='bold')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "12_pr_curves.png")
    plt.close('all')

    # 13. Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cm_base = np.array(confusion_matrices.get('Logistic Regression', [[0,0],[0,0]]))
    opt_key = [k for k in confusion_matrices if 'Optimized' in k][0]
    cm_opt = np.array(confusion_matrices.get(opt_key, [[0,0],[0,0]]))
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title('Logistic Regression (Standard 0.50 Threshold)', fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False)
    axes[1].set_title(f'Top Model ({best_thresh} Cost-Optimized Threshold)', fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(plots_dir / "13_confusion_matrices.png")
    plt.close('all')

    # 14. Threshold Optimization Curve
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(threshold_df['Threshold'], threshold_df['Recall'], label='Recall (Dropout)', color='#e74c3c', lw=2)
    ax.plot(threshold_df['Threshold'], threshold_df['Precision'], label='Precision', color='#2980b9', lw=2)
    ax.plot(threshold_df['Threshold'], threshold_df['F1'], label='F1-Score', color='#27ae60', lw=2)
    ax.axvline(best_thresh, color='black', linestyle='--', label=f'Optimal Threshold ({best_thresh})')
    ax.set_title('Validation Performance across Decision Thresholds', fontweight='bold')
    ax.set_xlabel('Classification Threshold')
    ax.set_ylabel('Metric Value')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "14_threshold_optimization_curve.png")
    plt.close('all')

    # 15. Cost vs Threshold
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(threshold_df['Threshold'], threshold_df['Total_Cost'], color='#8e44ad', lw=2.5, label='Total Intervention Cost')
    ax.axvline(best_thresh, color='red', linestyle='--', label=f'Min Cost Threshold ({best_thresh})')
    ax.axvline(0.50, color='gray', linestyle=':', label='Default Threshold (0.50)')
    ax.set_title('Institutional Cost Function vs Decision Threshold (FN=5, FP=1)', fontweight='bold')
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Total Penalty Cost')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "15_cost_vs_threshold.png")
    plt.close('all')

    # 16. SHAP Summary Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    top_shap = shap_importance_df.head(15).sort_values(by='Mean_Abs_SHAP', ascending=True)
    ax.barh(top_shap['Feature'], top_shap['Mean_Abs_SHAP'], color='#e84393', edgecolor='black')
    ax.set_title('Global Feature Explainability: Mean Absolute SHAP Values', fontweight='bold')
    ax.set_xlabel('Mean |SHAP Value| (Impact on Model Log-Odds)')
    plt.tight_layout()
    plt.savefig(plots_dir / "16_shap_summary.png")
    plt.close('all')

    # 17. Model Comparison Chart
    fig, ax = plt.subplots(figsize=(9, 4.5))
    comp_plot = model_comp_df[~model_comp_df['Model'].str.contains('Optimized')]
    x = np.arange(len(comp_plot))
    w = 0.25
    ax.bar(x - w, comp_plot['Recall'], width=w, label='Recall', color='#e74c3c')
    ax.bar(x, comp_plot['F1'], width=w, label='F1-Score', color='#27ae60')
    ax.bar(x + w, comp_plot['ROC_AUC'], width=w, label='ROC-AUC', color='#2980b9')
    ax.set_xticks(x)
    ax.set_xticklabels(comp_plot['Model'], rotation=15, ha='right', fontsize=8)
    ax.set_title('Model Performance Comparison Across Key Metrics', fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "17_model_comparison_chart.png")
    plt.close('all')

    gc.collect()
    print(f"-> Successfully saved all 17 figures into {plots_dir}.", flush=True)


def generate_academic_report(metadata, quality_df, model_comp_df, feature_ranking_df, reports_dir):
    """Writes a formal markdown academic summary report."""
    report_path = reports_dir / "academic_ml_project_report.md"
    report_text = f"""# Academic Machine Learning Project Report: Student Dropout Risk Analyzer

## 1. Executive Summary & Problem Formulation
The **Student Dropout Risk Analyzer** is an end-to-end predictive modeling system developed to detect undergraduate students at risk of premature academic attrition. Operating on traditional Machine Learning paradigms, the project strictly excludes deep neural networks and cloud orchestration, prioritizing statistical rigor, model interpretability, and business-grounded cost-sensitive decision rules.

- **Total Student Population Analyzed**: {metadata['total_dataset_size']:,}
- **Baseline Retention**: {100 - metadata['dropout_rate']:.2f}% (Class 0: Not Dropout)
- **Baseline Attrition Rate**: {metadata['dropout_rate']:.2f}% (Class 1: Dropout)
- **Empirical Class Imbalance**: {metadata['imbalance_ratio']}:1
- **Optimal Decision Threshold**: {metadata['selected_threshold']} (Cost-optimized vs. default 0.50)
- **Top Selected Classifier**: {metadata['best_model_name']}

---

## 2. Rigorous Preprocessing & Zero-Leakage Architecture
To ensure zero methodological data leakage:
1. `Student_ID` was isolated immediately prior to model ingestion.
2. Domain feature engineering functions were strictly calculated on row-level instances.
3. Stratified splitting allocated **70% Training ({metadata['train_samples']:,})**, **15% Validation ({metadata['val_samples']:,})**, and **15% Test ({metadata['test_samples']:,})**.
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
"""
    for i, r in feature_ranking_df.head(10).iterrows():
        report_text += f"| {i+1} | {r['Feature']} | {r['Mutual_Information']:.4f} | {r['ANOVA_F_Score']:.1f} | {r['RF_Importance']:.4f} |\n"

    report_text += f"""
---

## 5. Cost-Sensitive Model Performance Comparison (Untouched Test Partition)
Intervention cost is parameterized with **False Negative = 5** (unassisted dropout) and **False Positive = 1** (precautionary mentoring):

{model_comp_df.to_markdown(index=False)}

---

## 6. Actionable Institutional Interventions
Predictions are partitioned into actionable intervention tiers:
- **Low Risk (0 - 30%)**: Standard advising and semester check-ins.
- **Moderate Risk (30 - 60%)**: Faculty advisor conference; peer tutoring encouragement.
- **High Risk (60 - 80%)**: Structured academic recovery plan, attendance tracking, and study hall requirement.
- **Critical Risk (80 - 100%)**: Immediate case worker assignment, student wellness counseling, and financial aid hardship review.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"-> Generated comprehensive report: {report_path}", flush=True)


if __name__ == "__main__":
    run_training_pipeline()
