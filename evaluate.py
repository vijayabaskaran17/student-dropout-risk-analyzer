"""
Student Dropout Risk Analyzer - Model Evaluation Script
========================================================
Loads the untouched test partition and evaluates the final deployed model
under both the default (0.50) and cost-optimized threshold (0.49).
Calculates complete confusion metrics, precision-recall, and cost profiles.
"""

import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

from config import (
    PROCESSED_DATA_DIR, FINAL_MODEL_PATH, MODEL_METADATA_PATH,
    TARGET_COL, ID_COL, COST_FN, COST_FP
)


def evaluate_final_model():
    print("=" * 80)
    print("STANDALONE TEST SET EVALUATION - STUDENT DROPOUT RISK ANALYZER")
    print("=" * 80)

    test_path = PROCESSED_DATA_DIR / "test_processed.csv"
    if not test_path.exists():
        print(f"Error: Processed test set not found at {test_path}. Run train.py first.")
        sys.exit(1)

    df_test = pd.read_csv(test_path)
    y_test = df_test[TARGET_COL].values
    X_test = df_test.drop(columns=[TARGET_COL, ID_COL]).values

    model = joblib.load(FINAL_MODEL_PATH)
    metadata = joblib.load(MODEL_METADATA_PATH)

    opt_thresh = metadata.get('selected_threshold', 0.49)
    best_name = metadata.get('best_model_name', 'Selected Classifier')

    # Probability estimation
    probs = model.predict_proba(X_test)[:, 1]

    # Evaluation at standard 0.50 threshold
    preds_50 = (probs >= 0.50).astype(int)
    cm_50 = confusion_matrix(y_test, preds_50)
    tn_50, fp_50, fn_50, tp_50 = cm_50.ravel()
    cost_50 = COST_FN * fn_50 + COST_FP * fp_50

    # Evaluation at cost-optimized threshold
    preds_opt = (probs >= opt_thresh).astype(int)
    cm_opt = confusion_matrix(y_test, preds_opt)
    tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel()
    cost_opt = COST_FN * fn_opt + COST_FP * fp_opt

    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)

    print(f"\nModel Selected: {best_name}")
    print(f"Test Partition Size: {len(y_test):,} students (Dropout Prevalence: {y_test.mean()*100:.2f}%)")
    print(f"Global Discriminative Ability: ROC-AUC = {roc_auc:.4f} | PR-AUC = {pr_auc:.4f}")

    print("\n" + "-" * 80)
    print("THRESHOLD COMPARISON ANALYSIS (FN Cost = 5, FP Cost = 1)")
    print("-" * 80)
    comp_records = [
        {
            'Decision Threshold': 'Standard (0.50)',
            'Accuracy': f"{accuracy_score(y_test, preds_50):.4f}",
            'Precision': f"{precision_score(y_test, preds_50):.4f}",
            'Recall (Sensitivity)': f"{recall_score(y_test, preds_50):.4f}",
            'F1-Score': f"{f1_score(y_test, preds_50):.4f}",
            'Specificity': f"{tn_50 / (tn_50 + fp_50):.4f}",
            'False Negatives (Missed)': fn_50,
            'False Positives (False Alarm)': fp_50,
            'Total Cost Penalty': f"{cost_50:.1f}"
        },
        {
            'Decision Threshold': f'Cost-Optimized ({opt_thresh:.2f})',
            'Accuracy': f"{accuracy_score(y_test, preds_opt):.4f}",
            'Precision': f"{precision_score(y_test, preds_opt):.4f}",
            'Recall (Sensitivity)': f"{recall_score(y_test, preds_opt):.4f}",
            'F1-Score': f"{f1_score(y_test, preds_opt):.4f}",
            'Specificity': f"{tn_opt / (tn_opt + fp_opt):.4f}",
            'False Negatives (Missed)': fn_opt,
            'False Positives (False Alarm)': fp_opt,
            'Total Cost Penalty': f"{cost_opt:.1f}"
        }
    ]
    comp_df = pd.DataFrame(comp_records)
    print(comp_df.to_string(index=False))

    print("\n" + "-" * 80)
    print(f"DETAILED CLASSIFICATION REPORT (Threshold = {opt_thresh:.2f})")
    print("-" * 80)
    print(classification_report(y_test, preds_opt, target_names=['Retained (0)', 'Dropout (1)']))

    print("Confusion Matrix:")
    print(f"                 Predicted 0   Predicted 1")
    print(f"Actual Retained (0):   {tn_opt:<12} {fp_opt} (FP)")
    print(f"Actual Dropout  (1):   {fn_opt:<12} {tp_opt} (TP)")
    print(f"\nCost Reduction Achieved: {cost_50 - cost_opt:.1f} penalty points saved by shifting threshold!")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_final_model()
