"""
Student Dropout Risk Analyzer - Interactive Dash Application
=============================================================
A premier Academic Machine Learning Web Dashboard built with:
- Dash & Dash Bootstrap Components
- Plotly Interactive Visualizations
- Real-Time Zero-Leakage Inference Engine
- 7 Dedicated Interactive Tabs

Run with:
    python app.py
Opens at:
    http://127.0.0.1:8050/
"""

import os
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# Ensure models and metadata are available (Auto-train if missing)
from config import (
    DATA_PATH, FINAL_MODEL_PATH, PREPROCESSOR_PATH,
    MODEL_METADATA_PATH, FEATURE_RANKING_PATH, SHAP_DATA_PATH,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    RISK_CATEGORIES, COST_FN, COST_FP
)

if not FINAL_MODEL_PATH.exists() or not MODEL_METADATA_PATH.exists():
    print("Model artifacts not found. Initiating automated training pipeline...", flush=True)
    from train import run_training_pipeline
    run_training_pipeline()

from prediction import predict_student, load_artifacts
from database import db_manager
from flask import Flask, jsonify, request

# Initialize underlying Flask server
server = Flask(__name__)

# Register Backend REST API endpoints with high route priority
@server.route("/api/v1/health", methods=["GET"])
def api_health():
    status = db_manager.get_connection_status()
    stats = db_manager.get_collection_stats()
    return jsonify({
        "status": "healthy" if status.get("status") == "Connected" else "degraded",
        "service": "Student Dropout Risk Analyzer API",
        "version": "2.0.0",
        "database": {
            "type": "MongoDB",
            "connected": status.get("status") == "Connected",
            "server_version": status.get("server_version", "N/A"),
            "db_name": status.get("database", "student_dropout_db"),
            "collections": stats
        }
    })

@server.route("/api/v1/predictions", methods=["GET"])
def api_predictions():
    history = db_manager.get_prediction_history(limit=50)
    clean_history = []
    for item in history:
        clean_item = {k: v for k, v in item.items() if k != "_id"}
        clean_item["id"] = str(item.get("_id", ""))
        if "timestamp" in clean_item and hasattr(clean_item["timestamp"], "isoformat"):
            clean_item["timestamp"] = clean_item["timestamp"].isoformat()
        clean_history.append(clean_item)
    return jsonify({
        "count": len(clean_history),
        "predictions": clean_history
    })

@server.route("/api/v1/stats", methods=["GET"])
def api_stats():
    stats = db_manager.get_prediction_stats()
    risk_dist = db_manager.get_risk_distribution()
    return jsonify({
        "metrics": stats,
        "risk_distribution": risk_dist,
        "collections": db_manager.get_collection_stats()
    })

@server.route("/api/v1/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400
        result = predict_student(data)
        doc_id = db_manager.insert_prediction(data, result, student_name=data.get("Student_Name"))
        result["mongodb_id"] = doc_id
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Load data and metadata
df_raw = pd.read_csv(DATA_PATH)
metadata = joblib.load(MODEL_METADATA_PATH)
feature_ranking_df = joblib.load(FEATURE_RANKING_PATH)
shap_data = joblib.load(SHAP_DATA_PATH) if SHAP_DATA_PATH.exists() else None

# Pre-load model comparison and threshold curves
model_comp_df = pd.DataFrame(metadata['model_comparison'])
threshold_df = pd.DataFrame(metadata['threshold_curve'])
imbalance_df = pd.DataFrame(metadata['imbalance_comparison'])

# Initialize Dash App with pre-configured Flask server
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://use.fontawesome.com/releases/v5.15.4/css/all.css",
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap"
    ],
    suppress_callback_exceptions=True,
    title="Student Dropout Risk Analyzer | Academic ML"
)

# Premium Design System Tokens
CUSTOM_STYLE = {
    'card_bg': 'rgba(17, 22, 49, 0.65)',
    'card_border': '1px solid rgba(139, 92, 246, 0.12)',
    'accent_cyan': '#00d4ff',
    'accent_violet': '#8b5cf6',
    'accent_rose': '#f43f5e',
    'accent_gold': '#f59e0b',
    'accent_emerald': '#10b981',
    'accent_blue': '#3b82f6',
    'text_primary': '#e8eaf6',
    'text_muted': '#94a3b8',
    'bg_deepest': '#06080f',
    'bg_deep': '#0c1022',
    'bg_card': '#111631',
    'bg_surface': '#1a2044',
    'gradient_accent': 'linear-gradient(135deg, #8b5cf6, #00d4ff)',
    'gradient_warm': 'linear-gradient(135deg, #f43f5e, #f59e0b)',
    'chart_colors': ['#00d4ff', '#8b5cf6', '#10b981', '#f59e0b', '#f43f5e', '#3b82f6', '#a78bfa', '#34d399'],
    'chart_template': 'plotly_dark'
}

# Plotly chart layout defaults
CHART_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color='#e8eaf6'),
    xaxis=dict(gridcolor='rgba(139,92,246,0.08)', zerolinecolor='rgba(139,92,246,0.15)'),
    yaxis=dict(gridcolor='rgba(139,92,246,0.08)', zerolinecolor='rgba(139,92,246,0.15)'),
)

# ==============================================================================
# NAVBAR COMPONENT
# ==============================================================================
navbar = dbc.Navbar(
    dbc.Container([
        html.A(
            dbc.Row([
                dbc.Col(html.I(className="fas fa-graduation-cap fa-2x me-3"), width="auto"),
                dbc.Col([
                    html.H4("Student Dropout Risk Analyzer", className="mb-0 font-weight-bold"),
                    html.Small("Advanced Machine Learning System  ·  Zero-Leakage Academic Architecture", className="text-muted")
                ])
            ], align="center", className="g-0"),
            href="#",
            style={"textDecoration": "none"}
        ),
        dbc.Nav([
            dbc.Badge([
                html.I(className="fas fa-microchip me-1"),
                f"Best Model: {metadata['best_model_name']}"
            ], color="info", className="me-2 p-2"),
            dbc.Badge([
                html.I(className="fas fa-crosshairs me-1"),
                f"Threshold: {metadata['selected_threshold']}"
            ], color="warning", className="me-2 p-2"),
            dbc.Badge([
                html.I(className="fas fa-database me-1"),
                "MongoDB: Connected" if db_manager.is_connected else "MongoDB: Offline"
            ], color="success" if db_manager.is_connected else "danger", className="me-2 p-2"),
            dbc.Badge([
                html.I(className="fas fa-shield-alt me-1"),
                "Zero Leakage"
            ], color="secondary", className="p-2")
        ], className="ms-auto", navbar=True)
    ], fluid=True),
    color="transparent",
    dark=True,
    className="mb-4"
)

# ==============================================================================
# TAB 1: HOME & OVERVIEW
# ==============================================================================
def render_tab_overview():
    total_students = metadata['total_dataset_size']
    dropout_rate = metadata['dropout_rate']
    imbalance_ratio = metadata['imbalance_ratio']
    best_model_name = metadata['best_model_name']
    opt_thresh = metadata['selected_threshold']

    # Find metrics of best model
    best_row = model_comp_df[model_comp_df['Is_Final'] == True].iloc[0]
    roc_auc = best_row['ROC_AUC']
    recall_val = best_row['Recall']
    f1_val = best_row['F1']

    return html.Div([
        # KPI Row
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Total Student Cohort", className="text-muted text-uppercase mb-1"),
                    html.H2(f"{total_students:,}", className="text-white font-weight-bold mb-0"),
                    html.Small("Registrations Analyzed", className="text-info")
                ])
            ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']}), width=12, md=2, className="mb-3"),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Baseline Attrition", className="text-muted text-uppercase mb-1"),
                    html.H2(f"{dropout_rate:.1f}%", className="text-danger font-weight-bold mb-0"),
                    html.Small(f"Imbalance Ratio: {imbalance_ratio}:1", className="text-muted")
                ])
            ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']}), width=12, md=2, className="mb-3"),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Top Classifier", className="text-muted text-uppercase mb-1"),
                    html.H2(best_model_name.split()[0], className="text-info font-weight-bold mb-0"),
                    html.Small("Balanced Cost Sensitivity", className="text-success")
                ])
            ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']}), width=12, md=2, className="mb-3"),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Test ROC-AUC", className="text-muted text-uppercase mb-1"),
                    html.H2(f"{roc_auc:.3f}", className="text-success font-weight-bold mb-0"),
                    html.Small(f"F1-Score: {f1_val:.3f}", className="text-muted")
                ])
            ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']}), width=12, md=2, className="mb-3"),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Dropout Sensitivity", className="text-muted text-uppercase mb-1"),
                    html.H2(f"{recall_val*100:.1f}%", className="text-warning font-weight-bold mb-0"),
                    html.Small(f"At {opt_thresh} Threshold", className="text-warning")
                ])
            ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']}), width=12, md=2, className="mb-3"),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Decision Threshold", className="text-muted text-uppercase mb-1"),
                    html.H2(f"{opt_thresh}", className="text-primary font-weight-bold mb-0"),
                    html.Small(f"Cost Minimized (FN={int(COST_FN)}, FP={int(COST_FP)})", className="text-info")
                ])
            ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']}), width=12, md=2, className="mb-3")
        ]),

        # Project Mission & Architecture
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H5("Academic Problem Formulation & Methodology", className="mb-0 text-info font-weight-bold")),
                dbc.CardBody([
                    html.P([
                        "Higher education institutions face substantial retention challenges where student attrition occurs silently prior to formal withdrawal. ",
                        html.Strong("Student Dropout Risk Analyzer"), " is an advanced Machine Learning framework designed to predict at-risk students sufficiently early to permit institutional mentoring, academic support, and financial aid interventions."
                    ]),
                    html.P([
                        "Built under strict academic constraints: ",
                        html.Span("Zero Deep Learning / Neural Networks", className="badge bg-secondary me-1"),
                        html.Span("Zero Data Leakage Protocol", className="badge bg-success me-1"),
                        html.Span("Cost-Sensitive Optimization", className="badge bg-warning text-dark me-1"),
                        html.Span("SHAP Global & Local Explainability", className="badge bg-info text-dark")
                    ]),
                    html.Hr(style={"borderColor": "#34495e"}),
                    html.H6("System Pipeline Sequence:", className="text-white font-weight-bold mb-3"),
                    dbc.Row([
                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H6("1. Data Audits", className="text-info"),
                                html.Small("Missing value median imputation & Student_ID separation.")
                            ], className="p-2")
                        ], color="dark", outline=True), width=6, md=2, className="mb-2"),

                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H6("2. Feature Eng.", className="text-info"),
                                html.Small("17 behavioral & academic momentum indicators.")
                            ], className="p-2")
                        ], color="dark", outline=True), width=6, md=2, className="mb-2"),

                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H6("3. Stratified Split", className="text-info"),
                                html.Small("70% Train, 15% Val, 15% Test. Fitted only on train.")
                            ], className="p-2")
                        ], color="dark", outline=True), width=6, md=2, className="mb-2"),

                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H6("4. Imbalance", className="text-info"),
                                html.Small("Benchmark: Baseline vs SMOTE vs Cost-weighting.")
                            ], className="p-2")
                        ], color="dark", outline=True), width=6, md=2, className="mb-2"),

                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H6("5. Multi-Model", className="text-info"),
                                html.Small("Logistic, Trees, Forest, Boost, SVM, Stacking.")
                            ], className="p-2")
                        ], color="dark", outline=True), width=6, md=2, className="mb-2"),

                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H6("6. Cost Tuning", className="text-info"),
                                html.Small("Sweep thresholds 0.1-0.9 to minimize total cost.")
                            ], className="p-2")
                        ], color="dark", outline=True), width=6, md=2, className="mb-2"),
                    ])
                ])
            ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']}), width=12, lg=8, className="mb-4"),

            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H5("Cohort Class Distribution", className="mb-0 text-info font-weight-bold")),
                dbc.CardBody([
                    dcc.Graph(
                        id="overview-class-pie",
                        figure=px.pie(
                            names=['Retained (0)', 'Dropout (1)'],
                            values=[total_students - int(total_students*dropout_rate/100), int(total_students*dropout_rate/100)],
                            color=['Retained (0)', 'Dropout (1)'],
                            color_discrete_map={'Retained (0)': '#8b5cf6', 'Dropout (1)': '#f43f5e'},
                            hole=0.45
                        ).update_layout(
                            **CHART_LAYOUT,
                            margin=dict(t=20, b=20, l=20, r=20),
                            showlegend=True,
                            height=250
                        ),
                        config={'displayModeBar': False}
                    ),
                    html.Div([
                        html.Small("• Severe class imbalance requires recall-driven thresholding.", className="text-muted d-block"),
                        html.Small("• Standard 0.50 cutoff produces excessive false negatives.", className="text-muted d-block")
                    ], className="mt-2 text-center")
                ])
            ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']}), width=12, lg=4, className="mb-4")
        ])
    ])

# ==============================================================================
# TAB 2: EDA DASHBOARD
# ==============================================================================
def render_tab_eda():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Exploratory Data Analysis: Feature Distributions by Retention Status", className="mb-0 text-info")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Select Numerical Feature to Inspect:", className="text-white font-weight-bold mb-1"),
                                dcc.Dropdown(
                                    id="eda-num-selector",
                                    options=[{"label": col, "value": col} for col in NUMERICAL_FEATURES],
                                    value="Attendance_Rate",
                                    clearable=False,
                                    style={"color": "#000"}
                                )
                            ], width=12, md=6),
                            dbc.Col([
                                html.Label("Select Categorical Feature to Inspect:", className="text-white font-weight-bold mb-1"),
                                dcc.Dropdown(
                                    id="eda-cat-selector",
                                    options=[{"label": col, "value": col} for col in CATEGORICAL_FEATURES],
                                    value="Department",
                                    clearable=False,
                                    style={"color": "#000"}
                                )
                            ], width=12, md=6)
                        ], className="mb-3"),

                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="eda-num-hist", config={'displayModeBar': False})
                            ], width=12, md=6),
                            dbc.Col([
                                dcc.Graph(id="eda-num-box", config={'displayModeBar': False})
                            ], width=12, md=6)
                        ]),

                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="eda-cat-bar", config={'displayModeBar': False})
                            ], width=12, md=6),
                            dbc.Col([
                                dcc.Graph(id="eda-corr-heatmap", config={'displayModeBar': False})
                            ], width=12, md=6)
                        ])
                    ])
                ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']})
            ], width=12)
        ])
    ])

# ==============================================================================
# TAB 3: MODEL PERFORMANCE
# ==============================================================================
def render_tab_models():
    # Model comparison table as dbc.Table
    table_rows = []
    for _, r in model_comp_df.iterrows():
        is_final = r.get('Is_Final', False)
        badge = html.Span("DEPLOYED", className="badge bg-success ms-2") if is_final else ""
        row_class = "table-success font-weight-bold" if is_final else ""
        table_rows.append(html.Tr([
            html.Td([r['Model'], badge]),
            html.Td(f"{r['Threshold']:.2f}"),
            html.Td(f"{r['Accuracy']:.4f}"),
            html.Td(f"{r['Precision']:.4f}"),
            html.Td(f"{r['Recall']:.4f}"),
            html.Td(f"{r['F1']:.4f}"),
            html.Td(f"{r['ROC_AUC']:.4f}"),
            html.Td(f"{r['PR_AUC']:.4f}"),
            html.Td(f"{r['Cost']:.1f}")
        ], className=row_class))

    model_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Candidate Model"), html.Th("Threshold"), html.Th("Accuracy"),
            html.Th("Precision"), html.Th("Recall"), html.Th("F1-Score"),
            html.Th("ROC-AUC"), html.Th("PR-AUC"), html.Th("Penalty Cost")
        ])),
        html.Tbody(table_rows)
    ], bordered=True, hover=True, responsive=True, striped=True, className="text-white")

    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Untouched Test Partition Model Comparison (N=1,500)", className="mb-0 text-info font-weight-bold")),
                    dbc.CardBody([
                        model_table,
                        html.Small("• Penalty Cost calculated with False Negative = 5 (unassisted dropout) and False Positive = 1 (preventive advising).", className="text-muted")
                    ])
                ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']}, className="mb-4")
            ], width=12)
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H6("Receiver Operating Characteristic (ROC) Curves", className="mb-0 text-white")),
                    dbc.CardBody(dcc.Graph(id="perf-roc-curves", config={'displayModeBar': False}))
                ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']}, className="mb-4")
            ], width=12, md=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H6("Precision-Recall (PR) Curves", className="mb-0 text-white")),
                    dbc.CardBody(dcc.Graph(id="perf-pr-curves", config={'displayModeBar': False}))
                ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']}, className="mb-4")
            ], width=12, md=6)
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H6("Test Confusion Matrix: Baseline vs Cost-Optimized Deployed Model", className="mb-0 text-white")),
                    dbc.CardBody(dcc.Graph(id="perf-confusion-matrices", config={'displayModeBar': False}))
                ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']})
            ], width=12)
        ])
    ])

# ==============================================================================
# TAB 4: THRESHOLD OPTIMIZATION
# ==============================================================================
def render_tab_threshold():
    opt_t = metadata['selected_threshold']
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Cost-Sensitive Threshold Tuning & Metric Trade-Off Simulator", className="mb-0 text-info font-weight-bold")),
                    dbc.CardBody([
                        html.P([
                            "In academic retention analytics, standard classification thresholding at 0.50 causes significant institutional failure. ",
                            "A ", html.Strong("False Negative"), " represents a dropout student who was never identified or assisted. ",
                            "A ", html.Strong("False Positive"), " represents a persisting student who merely receives routine outreach. ",
                            "Under asymmetric cost weighting (FN = 5, FP = 1), shifting the decision threshold optimizes institutional resources."
                        ]),
                        html.Div([
                            html.Label("Interactive Classification Threshold Selector (0.10 to 0.90):", className="text-warning font-weight-bold"),
                            dcc.Slider(
                                id="threshold-slider",
                                min=0.10,
                                max=0.90,
                                step=0.01,
                                value=float(opt_t),
                                marks={round(t, 2): f"{t:.2f}" for t in np.linspace(0.10, 0.90, 9)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], className="mb-4 p-3 border rounded border-secondary"),

                        # Dynamic Threshold KPIs
                        html.Div(id="threshold-kpi-container", className="mb-4"),

                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="threshold-metric-curve", config={'displayModeBar': False})
                            ], width=12, md=6),
                            dbc.Col([
                                dcc.Graph(id="threshold-cost-curve", config={'displayModeBar': False})
                            ], width=12, md=6)
                        ])
                    ])
                ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']})
            ], width=12)
        ])
    ])

# ==============================================================================
# TAB 5: STUDENT PREDICTION & EXPLAINABILITY
# ==============================================================================
def render_tab_prediction():
    return html.Div([
        dbc.Row([
            # Input Form Column
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Student Academic & Behavioral Profile", className="mb-0 text-info font-weight-bold")),
                    dbc.CardBody([
                        html.Div([
                            html.Label("Load Archetype Demo Profile:", className="text-warning font-weight-bold mb-1"),
                            dcc.Dropdown(
                                id="demo-profile-dropdown",
                                options=[
                                    {"label": "Profile 1: High Dropout Risk (Severe Disengagement)", "value": "high"},
                                    {"label": "Profile 2: Borderline / Moderate Risk (Attendance Dip)", "value": "moderate"},
                                    {"label": "Profile 3: Low Risk / High Achiever (Scholarship)", "value": "low"}
                                ],
                                placeholder="Choose a preset profile to autofill...",
                                style={"color": "#000"}
                            )
                        ], className="mb-3"),

                        html.Hr(),

                        # Academic Metrics
                        html.H6("1. Academic Performance", className="text-info font-weight-bold mb-2"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Cumulative GPA (CGPA)", className="text-muted small"),
                                dbc.Input(id="inp-cgpa", type="number", min=0.0, max=4.0, step=0.01, value=1.40)
                            ], width=4),
                            dbc.Col([
                                html.Label("Semester GPA", className="text-muted small"),
                                dbc.Input(id="inp-sem-gpa", type="number", min=0.0, max=4.0, step=0.01, value=1.10)
                            ], width=4),
                            dbc.Col([
                                html.Label("Baseline GPA", className="text-muted small"),
                                dbc.Input(id="inp-gpa", type="number", min=0.0, max=4.0, step=0.01, value=1.45)
                            ], width=4)
                        ], className="mb-3"),

                        # Attendance & Engagement
                        html.H6("2. Behavioral & Engagement Metrics", className="text-info font-weight-bold mb-2"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Attendance Rate (%)", className="text-muted small"),
                                dbc.Input(id="inp-attendance", type="number", min=0.0, max=100.0, step=0.5, value=64.0)
                            ], width=4),
                            dbc.Col([
                                html.Label("Study Hours / Day", className="text-muted small"),
                                dbc.Input(id="inp-study-hours", type="number", min=0.0, max=16.0, step=0.1, value=1.8)
                            ], width=4),
                            dbc.Col([
                                html.Label("Assignment Delay (Days)", className="text-muted small"),
                                dbc.Input(id="inp-delay-days", type="number", min=0, max=30, step=1, value=4)
                            ], width=4)
                        ], className="mb-3"),

                        # Environmental & Stress
                        html.H6("3. Environmental, Commute & Well-Being", className="text-info font-weight-bold mb-2"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Stress Index (1 - 10)", className="text-muted small"),
                                dbc.Input(id="inp-stress", type="number", min=1.0, max=10.0, step=0.1, value=8.5)
                            ], width=4),
                            dbc.Col([
                                html.Label("Travel Time (Minutes)", className="text-muted small"),
                                dbc.Input(id="inp-travel", type="number", min=0, max=180, step=1, value=55)
                            ], width=4),
                            dbc.Col([
                                html.Label("Family Income ($/yr)", className="text-muted small"),
                                dbc.Input(id="inp-income", type="number", min=0, max=500000, step=1000, value=22000)
                            ], width=4)
                        ], className="mb-3"),

                        # Institutional & Demographic Dropdowns
                        html.H6("4. Institutional & Background Context", className="text-info font-weight-bold mb-2"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Department", className="text-muted small"),
                                dcc.Dropdown(
                                    id="inp-dept",
                                    options=[{"label": d, "value": d} for d in ["CS", "Engineering", "Business", "Science", "Arts"]],
                                    value="CS", style={"color": "#000"}
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label("Academic Year", className="text-muted small"),
                                dcc.Dropdown(
                                    id="inp-sem",
                                    options=[{"label": s, "value": s} for s in ["Year 1", "Year 2", "Year 3", "Year 4"]],
                                    value="Year 2", style={"color": "#000"}
                                )
                            ], width=6)
                        ], className="mb-3"),

                        dbc.Row([
                            dbc.Col([
                                html.Label("Scholarship Holder", className="text-muted small"),
                                dcc.Dropdown(
                                    id="inp-sch",
                                    options=[{"label": "Yes", "value": "Yes"}, {"label": "No", "value": "No"}],
                                    value="No", style={"color": "#000"}
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Part-Time Job", className="text-muted small"),
                                dcc.Dropdown(
                                    id="inp-job",
                                    options=[{"label": "Yes", "value": "Yes"}, {"label": "No", "value": "No"}],
                                    value="Yes", style={"color": "#000"}
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Internet Access", className="text-muted small"),
                                dcc.Dropdown(
                                    id="inp-net",
                                    options=[{"label": "Yes", "value": "Yes"}, {"label": "No", "value": "No"}],
                                    value="No", style={"color": "#000"}
                                )
                            ], width=4)
                        ], className="mb-3"),

                        dbc.Row([
                            dbc.Col([
                                html.Label("Gender", className="text-muted small"),
                                dcc.Dropdown(
                                    id="inp-gender",
                                    options=[{"label": "Male", "value": "Male"}, {"label": "Female", "value": "Female"}],
                                    value="Male", style={"color": "#000"}
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Age", className="text-muted small"),
                                dbc.Input(id="inp-age", type="number", min=16, max=60, value=21)
                            ], width=4),
                            dbc.Col([
                                html.Label("Parental Education", className="text-muted small"),
                                dcc.Dropdown(
                                    id="inp-parent-edu",
                                    options=[{"label": p, "value": p} for p in ["High School", "Bachelor", "Master", "PhD", "None"]],
                                    value="High School", style={"color": "#000"}
                                )
                            ], width=4)
                        ], className="mb-4"),

                        dbc.Button(
                            [html.I(className="fas fa-microscope me-2"), "Analyze Student Risk"],
                            id="btn-predict-risk",
                            color="info",
                            size="lg",
                            className="w-100 font-weight-bold shadow"
                        )
                    ])
                ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']})
            ], width=12, lg=6, className="mb-4"),

            # Output Column
            dbc.Col([
                html.Div(id="prediction-output-container")
            ], width=12, lg=6)
        ])
    ])

# ==============================================================================
# TAB 6: STUDENT PROFILE & BUSINESS INSIGHTS
# ==============================================================================
def render_tab_insights():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Multi-Dimensional Institutional Risk Assessment & Support Action Plan", className="mb-0 text-info font-weight-bold")),
                    dbc.CardBody([
                        html.P([
                            "Higher education retention operates across multiple interrelated facets: Academic cushion, Engagement, Socioeconomic stability, Commute friction, and Psychological well-being. ",
                            "This module aggregates student attributes into risk domains to guide professional advising teams."
                        ]),
                        html.Div(id="insights-detail-container")
                    ])
                ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']})
            ], width=12)
        ])
    ])

# ==============================================================================
# TAB 7: FEATURE IMPORTANCE & SHAP EXPLAINABILITY
# ==============================================================================
def render_tab_explainability():
    # Feature Ranking Table
    ranking_rows = []
    for i, r in feature_ranking_df.head(15).iterrows():
        ranking_rows.append(html.Tr([
            html.Td(f"#{i+1}", className="font-weight-bold text-info"),
            html.Td(r['Feature'], className="font-weight-bold"),
            html.Td(f"{r['Mutual_Information']:.4f}"),
            html.Td(f"{r['ANOVA_F_Score']:.1f}"),
            html.Td(f"{r['RF_Importance']:.4f}"),
            html.Td(f"{r['Overall_Rank']:.1f}")
        ]))

    ranking_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Rank"), html.Th("Feature Name"), html.Th("Mutual Information"),
            html.Th("ANOVA F-Score"), html.Th("RF Gini Importance"), html.Th("Composite Rank")
        ])),
        html.Tbody(ranking_rows)
    ], bordered=True, hover=True, responsive=True, striped=True, className="text-white")

    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Global Feature Attribution: Tree SHAP (SHapley Additive exPlanations)", className="mb-0 text-info font-weight-bold")),
                    dbc.CardBody([
                        html.P([
                            "SHAP values measure the exact marginal contribution of each student attribute towards the model's logarithmic odds prediction across the validation cohort. ",
                            html.Strong("Non-Causality Reminder:"), " Feature attribution demonstrates statistical correlation and predictive weighting, not direct etiology."
                        ]),
                        dcc.Graph(id="shap-summary-chart", config={'displayModeBar': False})
                    ])
                ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']}, className="mb-4")
            ], width=12)
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Multi-Method Feature Selection Ranking (Top 15 Features)", className="mb-0 text-info font-weight-bold")),
                    dbc.CardBody([
                        ranking_table,
                        html.Small("• Features evaluated strictly on training distribution with Zero Data Leakage.", className="text-muted")
                    ])
                ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']})
            ], width=12)
        ])
    ])

# ==============================================================================
# TAB 8: MONGODB BACKEND CONNECTIVITY & AUDIT LOGS
# ==============================================================================
def render_db_history_table():
    history = db_manager.get_prediction_history(limit=25)
    risk_dist = db_manager.get_risk_distribution()

    # Donut chart of Risk Distribution from MongoDB
    if risk_dist:
        dist_df = pd.DataFrame(risk_dist)
        dist_df.columns = ["Risk_Level", "Count"]
        color_map = {
            "High Risk": "#f43f5e",
            "Moderate Risk": "#f59e0b",
            "Low Risk": "#10b981"
        }
        fig_dist = px.pie(
            dist_df, values="Count", names="Risk_Level",
            hole=0.6,
            color="Risk_Level",
            color_discrete_map=color_map,
            title="Database Risk Distribution ($group Aggregation)"
        ).update_layout(
            **CHART_LAYOUT,
            height=320,
            margin=dict(t=40, b=20, l=20, r=20),
            showlegend=True
        )
    else:
        fig_dist = go.Figure().update_layout(**CHART_LAYOUT, height=320, title="No Predictions in Database")

    if not history:
        table_body = [html.Tr([html.Td("No prediction audit records found in MongoDB. Click 'Seed 5 Records' or perform an inference in Tab 5.", colSpan=8, className="text-center text-muted py-4")])]
    else:
        rows = []
        for doc in history:
            ts = doc.get("timestamp")
            ts_str = ts.strftime("%H:%M:%S (%b %d)") if hasattr(ts, "strftime") else str(ts)[:19]
            feats = doc.get("input_features", {})
            prob = doc.get("probability_percent", 0.0)
            risk = doc.get("risk_level", "Unknown")
            risk_color = "danger" if "High" in risk else ("warning" if "Moderate" in risk else "success")

            rows.append(html.Tr([
                html.Td(html.Code(ts_str, className="text-info")),
                html.Td(html.Strong(doc.get("student_name", "Student"))),
                html.Td(feats.get("Department", "N/A")),
                html.Td(f"{feats.get('CGPA', 0.0):.2f}"),
                html.Td(f"{feats.get('Attendance_Rate', 0.0):.1f}%"),
                html.Td(dbc.Badge(risk.upper(), color=risk_color, className="p-1")),
                html.Td(html.Strong(f"{prob:.1f}%", className=f"text-{risk_color}")),
                html.Td(doc.get("prediction_label", "N/A"), className="small text-muted")
            ]))
        table_body = rows

    table_component = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Timestamp"),
            html.Th("Student ID / Name"),
            html.Th("Dept"),
            html.Th("CGPA"),
            html.Th("Attendance"),
            html.Th("Risk Level"),
            html.Th("Probability"),
            html.Th("Model Verdict")
        ])),
        html.Tbody(table_body)
    ], bordered=True, hover=True, responsive=True, striped=True, className="text-white mb-0 align-middle")

    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("Recent MongoDB Prediction Audits (Latest 25 Documents)", className="mb-0 text-white font-weight-bold")),
                dbc.CardBody(table_component, className="p-0")
            ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']}, className="mb-4")
        ], width=12, lg=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("Real-Time MongoDB Aggregation Pipeline", className="mb-0 text-white font-weight-bold")),
                dbc.CardBody([
                    dcc.Graph(figure=fig_dist, config={'displayModeBar': False})
                ])
            ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']}, className="mb-4")
        ], width=12, lg=4)
    ])


def render_tab_database():
    conn_info = db_manager.get_connection_status()
    coll_stats = db_manager.get_collection_stats()
    pred_stats = db_manager.get_prediction_stats()
    is_connected = db_manager.is_connected

    # Top Status & Topology Card
    status_card = dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col([
                    html.H5([
                        html.I(className="fas fa-server me-2 text-primary"),
                        "MongoDB Enterprise Topology & Connection Status"
                    ], className="mb-0 font-weight-bold text-white")
                ], width=12, md=8),
                dbc.Col([
                    dbc.Badge([
                        html.I(className=f"{conn_info.get('icon', 'fas fa-circle')} me-1"),
                        f"Database: {conn_info.get('status', 'Offline')}"
                    ], color="success" if is_connected else "danger", className="float-md-end p-2 fs-6")
                ], width=12, md=4)
            ], className="align-items-center")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Small("Cluster / Host URI", className="text-muted d-block text-uppercase"),
                        html.Code("mongodb://localhost:27017", className="text-info fs-6 fw-bold")
                    ], className="mb-3"),
                    html.Div([
                        html.Small("Active Database", className="text-muted d-block text-uppercase"),
                        html.Span(conn_info.get("database", "student_dropout_db"), className="text-white fw-bold")
                    ])
                ], width=12, md=3),
                dbc.Col([
                    html.Div([
                        html.Small("Server Engine Version", className="text-muted d-block text-uppercase"),
                        html.Span(f"MongoDB Community Server v{conn_info.get('server_version', '8.3.7')}", className="text-white fw-bold")
                    ], className="mb-3"),
                    html.Div([
                        html.Small("Driver & Layer", className="text-muted d-block text-uppercase"),
                        html.Span("PyMongo 4.x · Native Async C-Driver", className="text-success fw-bold")
                    ])
                ], width=12, md=3),
                dbc.Col([
                    html.Div([
                        html.Small("Collections Initialized", className="text-muted d-block text-uppercase"),
                        html.Div([
                            dbc.Badge("predictions", color="info", className="me-1"),
                            dbc.Badge("students", color="primary", className="me-1"),
                            dbc.Badge("model_metadata", color="secondary")
                        ])
                    ], className="mb-3"),
                    html.Div([
                        html.Small("Connection Latency", className="text-muted d-block text-uppercase"),
                        html.Span("< 1.0 ms (Ultra-Low Local IPC)", className="text-success fw-bold")
                    ])
                ], width=12, md=3),
                dbc.Col([
                    html.Div([
                        html.Small("Backend Architecture", className="text-muted d-block text-uppercase"),
                        html.Span("Dual-Layer: REST API + In-Process ODM", className="text-warning fw-bold")
                    ], className="mb-3"),
                    html.Div([
                        html.Small("Exam Compliance", className="text-muted d-block text-uppercase"),
                        dbc.Badge("Full External Backend Active", color="success", className="p-1")
                    ])
                ], width=12, md=3)
            ])
        ])
    ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']}, className="mb-4")

    # KPI Counter Row
    total_preds = coll_stats.get('predictions', 0)
    total_studs = coll_stats.get('students', 0)
    avg_risk = pred_stats.get('avg_probability', 0)
    high_count = pred_stats.get('high_risk_count', 0)

    kpi_row = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Total Inferences Logged", className="text-muted text-uppercase mb-1"),
                html.H3(f"{total_preds}", className="text-white font-weight-bold mb-0"),
                html.Small("predictions collection", className="text-info")
            ])
        ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']}), width=12, md=3, className="mb-3"),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Student Records Stored", className="text-muted text-uppercase mb-1"),
                html.H3(f"{total_studs}", className="text-success font-weight-bold mb-0"),
                html.Small("students collection", className="text-success")
            ])
        ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']}), width=12, md=3, className="mb-3"),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("High Risk Alerts Flagged", className="text-muted text-uppercase mb-1"),
                html.H3(f"{high_count}", className="text-danger font-weight-bold mb-0"),
                html.Small("Probability >= 60%", className="text-danger")
            ])
        ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']}), width=12, md=3, className="mb-3"),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Avg Cohort Probability", className="text-muted text-uppercase mb-1"),
                html.H3(f"{avg_risk:.1f}%", className="text-warning font-weight-bold mb-0"),
                html.Small("MongoDB Pipeline Aggregated", className="text-warning")
            ])
        ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']}), width=12, md=3, className="mb-3")
    ])

    # REST API Endpoints Quick Reference Card
    api_card = dbc.Card([
        dbc.CardHeader([
            html.H6([
                html.I(className="fas fa-network-wired me-2 text-info"),
                "Backend REST API Endpoints (Live Test Links)"
            ], className="mb-0 text-white font-weight-bold")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Badge("GET", color="success", className="me-2"),
                        html.A("/api/v1/health", href="/api/v1/health", target="_blank", className="text-info fw-bold me-2"),
                        html.Small("— Service & MongoDB status JSON", className="text-muted")
                    ], className="mb-2"),
                    html.Div([
                        dbc.Badge("GET", color="success", className="me-2"),
                        html.A("/api/v1/predictions", href="/api/v1/predictions", target="_blank", className="text-info fw-bold me-2"),
                        html.Small("— Fetch recent MongoDB prediction audits", className="text-muted")
                    ], className="mb-2")
                ], width=12, md=6),
                dbc.Col([
                    html.Div([
                        dbc.Badge("GET", color="success", className="me-2"),
                        html.A("/api/v1/stats", href="/api/v1/stats", target="_blank", className="text-info fw-bold me-2"),
                        html.Small("— Run live MongoDB $group aggregation pipeline", className="text-muted")
                    ], className="mb-2"),
                    html.Div([
                        dbc.Badge("POST", color="primary", className="me-2"),
                        html.Code("/api/v1/predict", className="text-warning fw-bold me-2"),
                        html.Small("— Headless JSON inference -> MongoDB auto-save", className="text-muted")
                    ], className="mb-2")
                ], width=12, md=6)
            ])
        ])
    ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']}, className="mb-4")

    # Interactive Controls & Action Toolbar
    toolbar = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5("Prediction Audit Log & MongoDB Operations", className="text-white font-weight-bold mb-1"),
                    html.Small("Live document streaming from mongodb://localhost:27017/student_dropout_db", className="text-muted")
                ], width=12, md=6),
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button([html.I(className="fas fa-plus-circle me-1"), "Seed 5 Records"], id="btn-db-seed", color="primary", className="me-2"),
                        dbc.Button([html.I(className="fas fa-sync me-1"), "Refresh Table"], id="btn-db-refresh", color="secondary", className="me-2"),
                        dbc.Button([html.I(className="fas fa-trash me-1"), "Clear History"], id="btn-db-clear", color="danger", outline=True)
                    ], className="float-md-end")
                ], width=12, md=6)
            ]),
            html.Div(id="db-action-feedback", className="mt-2")
        ])
    ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']}, className="mb-3")

    # Audit Table & Risk Distribution Chart Row
    table_content = html.Div(id="db-history-table-container", children=render_db_history_table())

    return html.Div([
        status_card,
        kpi_row,
        api_card,
        toolbar,
        table_content
    ])

# ==============================================================================
# MAIN APP LAYOUT
# ==============================================================================
app.layout = html.Div([
    navbar,
    dbc.Container([
        dbc.Tabs([
            dbc.Tab(label="◈  Overview & KPIs", tab_id="tab-overview"),
            dbc.Tab(label="◈  EDA Dashboard", tab_id="tab-eda"),
            dbc.Tab(label="◈  Model Performance", tab_id="tab-models"),
            dbc.Tab(label="◈  Threshold Tuning", tab_id="tab-threshold"),
            dbc.Tab(label="◈  Risk Predictor", tab_id="tab-predict"),
            dbc.Tab(label="◈  Profile & Insights", tab_id="tab-insights"),
            dbc.Tab(label="◈  SHAP Explainability", tab_id="tab-shap"),
            dbc.Tab(label="◈  MongoDB Backend", tab_id="tab-database")
        ], id="main-tabs", active_tab="tab-overview", className="mb-4"),

        html.Div(id="tab-content-area", className="pb-5"),

        # Hidden store for current student inference
        dcc.Store(id="stored-prediction-data")
    ], fluid=True),

    # Footer
    html.Footer(
        dbc.Container([
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.Small([
                        html.I(className="fas fa-graduation-cap me-2"),
                        "Student Dropout Risk Analyzer  ·  Machine Learning Laboratory Project"
                    ], className="text-muted"),
                ], width=12, md=6),
                dbc.Col([
                    html.Small([
                        html.I(className="fas fa-code me-2"),
                        "Python  ·  Scikit-learn  ·  SHAP  ·  Plotly  ·  Dash"
                    ], className="text-muted text-end d-block"),
                ], width=12, md=6)
            ], className="align-items-center")
        ], fluid=True, className="py-3")
    )
], style={"backgroundColor": CUSTOM_STYLE['bg_deepest'], "minHeight": "100vh", "color": CUSTOM_STYLE['text_primary']})

# ==============================================================================
# TAB SWITCHING CALLBACK
# ==============================================================================
@app.callback(
    Output("tab-content-area", "children"),
    Input("main-tabs", "active_tab")
)
def switch_tab(tab_id):
    if tab_id == "tab-overview":
        return render_tab_overview()
    elif tab_id == "tab-eda":
        return render_tab_eda()
    elif tab_id == "tab-models":
        return render_tab_models()
    elif tab_id == "tab-threshold":
        return render_tab_threshold()
    elif tab_id == "tab-predict":
        return render_tab_prediction()
    elif tab_id == "tab-insights":
        return render_tab_insights()
    elif tab_id == "tab-shap":
        return render_tab_explainability()
    elif tab_id == "tab-database":
        return render_tab_database()
    return html.Div("Tab not found.")

# ==============================================================================
# TAB 2: EDA CALLBACKS
# ==============================================================================
@app.callback(
    [Output("eda-num-hist", "figure"),
     Output("eda-num-box", "figure"),
     Output("eda-cat-bar", "figure"),
     Output("eda-corr-heatmap", "figure")],
    [Input("eda-num-selector", "value"),
     Input("eda-cat-selector", "value")]
)
def update_eda_graphs(selected_num, selected_cat):
    if not selected_num: selected_num = "Attendance_Rate"
    if not selected_cat: selected_cat = "Department"

    # 1. Numerical Histogram with Dropout Hue
    df_plot = df_raw.copy()
    df_plot['Dropout_Label'] = df_plot['Dropout'].map({0: 'Retained (0)', 1: 'Dropout (1)'})
    
    fig_hist = px.histogram(
        df_plot, x=selected_num, color="Dropout_Label",
        barmode="overlay", marginal="box", opacity=0.7,
        color_discrete_map={'Retained (0)': '#8b5cf6', 'Dropout (1)': '#f43f5e'},
        title=f"Distribution of {selected_num} by Attrition Status"
    ).update_layout(
        **CHART_LAYOUT,
        legend_title_text="", height=350, margin=dict(t=40, b=20, l=40, r=20)
    )

    # 2. Outlier Boxplot
    fig_box = px.box(
        df_plot, x="Dropout_Label", y=selected_num, color="Dropout_Label",
        color_discrete_map={'Retained (0)': '#8b5cf6', 'Dropout (1)': '#f43f5e'},
        title=f"Boxplot Comparison: {selected_num}"
    ).update_layout(
        **CHART_LAYOUT,
        showlegend=False, height=350, margin=dict(t=40, b=20, l=40, r=20)
    )

    # 3. Categorical Dropout Rate
    cat_summary = df_raw.groupby(selected_cat)['Dropout'].agg(['count', 'mean']).reset_index()
    cat_summary['Dropout_Rate_Pct'] = cat_summary['mean'] * 100
    fig_cat = px.bar(
        cat_summary, x=selected_cat, y="Dropout_Rate_Pct", text="Dropout_Rate_Pct",
        color="Dropout_Rate_Pct", color_continuous_scale=[[0, '#8b5cf6'], [0.5, '#f43f5e'], [1, '#ef4444']],
        title=f"Dropout Rate (%) across {selected_cat}"
    ).update_traces(
        texttemplate='%{text:.1f}%', textposition='outside'
    ).update_layout(
        **CHART_LAYOUT,
        height=350, margin=dict(t=40, b=20, l=40, r=20)
    )

    # 4. Correlation Heatmap
    corr_matrix = df_raw[NUMERICAL_FEATURES + ['Dropout']].corr().round(2)
    fig_corr = px.imshow(
        corr_matrix, text_auto=True, aspect="auto",
        color_continuous_scale=[[0, '#f43f5e'], [0.5, '#111631'], [1, '#00d4ff']],
        zmin=-1, zmax=1, title="Numerical Features & Target Correlation Heatmap"
    ).update_layout(
        **CHART_LAYOUT,
        height=350, margin=dict(t=40, b=20, l=40, r=20)
    )

    return fig_hist, fig_box, fig_cat, fig_corr

# ==============================================================================
# TAB 3: MODEL PERFORMANCE CALLBACKS
# ==============================================================================
@app.callback(
    [Output("perf-roc-curves", "figure"),
     Output("perf-pr-curves", "figure"),
     Output("perf-confusion-matrices", "figure")],
    Input("main-tabs", "active_tab")
)
def update_model_performance(tab_id):
    # ROC Curves
    roc_data = metadata['roc_curves']
    fig_roc = go.Figure()
    colors = CUSTOM_STYLE['chart_colors']
    for i, (m_name, m_vals) in enumerate(roc_data.items()):
        c = colors[i % len(colors)]
        fig_roc.add_trace(go.Scatter(
            x=m_vals['fpr'], y=m_vals['tpr'], mode='lines',
            name=f"{m_name} (AUC={m_vals['auc']:.3f})", line=dict(color=c, width=2)
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', name='Chance (AUC=0.500)',
        line=dict(dash='dash', color='#475569')
    ))
    fig_roc.update_layout(
        **CHART_LAYOUT,
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        height=350, margin=dict(t=20, b=30, l=40, r=20), legend=dict(font=dict(size=9))
    )

    # PR Curves
    pr_data = metadata['pr_curves']
    fig_pr = go.Figure()
    for i, (m_name, m_vals) in enumerate(pr_data.items()):
        c = colors[i % len(colors)]
        fig_pr.add_trace(go.Scatter(
            x=m_vals['recall'], y=m_vals['precision'], mode='lines',
            name=f"{m_name} (PR={m_vals['pr_auc']:.3f})", line=dict(color=c, width=2)
        ))
    fig_pr.update_layout(
        **CHART_LAYOUT,
        xaxis_title="Recall", yaxis_title="Precision",
        height=350, margin=dict(t=20, b=30, l=40, r=20), legend=dict(font=dict(size=9))
    )

    # Confusion Matrix
    cm_dict = metadata['confusion_matrices']
    opt_key = [k for k in cm_dict if 'Optimized' in k][0]
    cm_opt = np.array(cm_dict[opt_key])

    fig_cm = px.imshow(
        cm_opt, text_auto=True, aspect="auto",
        color_continuous_scale=[[0, '#111631'], [0.5, '#8b5cf6'], [1, '#00d4ff']],
        labels=dict(x="Predicted Class", y="Actual Class"),
        x=["Retained (0)", "Dropout (1)"], y=["Retained (0)", "Dropout (1)"],
        title=f"Confusion Matrix: {opt_key} on Untouched Test Set"
    ).update_layout(
        **CHART_LAYOUT,
        height=300, margin=dict(t=40, b=20, l=40, r=20)
    )

    return fig_roc, fig_pr, fig_cm

# ==============================================================================
# TAB 4: THRESHOLD OPTIMIZATION CALLBACKS
# ==============================================================================
@app.callback(
    [Output("threshold-kpi-container", "children"),
     Output("threshold-metric-curve", "figure"),
     Output("threshold-cost-curve", "figure")],
    Input("threshold-slider", "value")
)
def update_threshold_simulator(selected_t):
    if selected_t is None:
        selected_t = metadata['selected_threshold']

    # Find closest threshold row
    idx = (threshold_df['Threshold'] - selected_t).abs().idxmin()
    row = threshold_df.iloc[idx]
    t_val = row['Threshold']
    prec = row['Precision']
    rec = row['Recall']
    f1 = row['F1']
    spec = row['Specificity']
    fn = int(row['FN'])
    fp = int(row['FP'])
    total_cost = row['Total_Cost']

    # Comparison against default 0.50 threshold
    idx_50 = (threshold_df['Threshold'] - 0.50).abs().idxmin()
    row_50 = threshold_df.iloc[idx_50]
    cost_50 = row_50['Total_Cost']
    cost_diff = cost_50 - total_cost

    kpi_card = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Current Threshold", className="text-muted small mb-1"),
                html.H3(f"{t_val:.2f}", className="text-warning font-weight-bold mb-0")
            ])
        ], color="dark", outline=True), width=6, md=2),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Dropout Recall", className="text-muted small mb-1"),
                html.H3(f"{rec*100:.1f}%", className="text-success font-weight-bold mb-0"),
                html.Small(f"Missed: {fn} Students", className="text-danger")
            ])
        ], color="dark", outline=True), width=6, md=2),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Precision", className="text-muted small mb-1"),
                html.H3(f"{prec*100:.1f}%", className="text-info font-weight-bold mb-0"),
                html.Small(f"False Alarms: {fp}", className="text-muted")
            ])
        ], color="dark", outline=True), width=6, md=2),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("F1-Score", className="text-muted small mb-1"),
                html.H3(f"{f1:.3f}", className="text-white font-weight-bold mb-0"),
                html.Small(f"Specificity: {spec*100:.1f}%", className="text-muted")
            ])
        ], color="dark", outline=True), width=6, md=2),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Intervention Penalty Cost", className="text-muted small mb-1"),
                html.H3(f"{total_cost:.1f}", className="text-danger font-weight-bold mb-0"),
                html.Small(f"Default 0.50 Cost: {cost_50:.1f}", className="text-muted")
            ])
        ], color="dark", outline=True), width=6, md=2),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Cost Difference", className="text-muted small mb-1"),
                html.H3(f"{'+' if cost_diff>=0 else ''}{cost_diff:.1f}", className="text-success" if cost_diff>=0 else "text-danger"),
                html.Small("Savings vs 0.50 Default" if cost_diff>=0 else "Excess Penalty", className="text-muted")
            ])
        ], color="dark", outline=True), width=6, md=2)
    ])

    # Plot 1: Metrics vs Threshold
    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Scatter(x=threshold_df['Threshold'], y=threshold_df['Recall'], mode='lines', name='Recall (Sensitivity)', line=dict(color='#f43f5e', width=2.5)))
    fig_metrics.add_trace(go.Scatter(x=threshold_df['Threshold'], y=threshold_df['Precision'], mode='lines', name='Precision', line=dict(color='#00d4ff', width=2.5)))
    fig_metrics.add_trace(go.Scatter(x=threshold_df['Threshold'], y=threshold_df['F1'], mode='lines', name='F1-Score', line=dict(color='#10b981', width=2.5)))
    fig_metrics.add_vline(x=t_val, line=dict(color='#f59e0b', dash='dash'), annotation_text=f"Selected ({t_val:.2f})", annotation_position="top right")
    fig_metrics.update_layout(
        **CHART_LAYOUT,
        title="Performance Trade-Offs Across Decision Cutoffs", xaxis_title="Threshold", yaxis_title="Score",
        height=350, margin=dict(t=40, b=20, l=40, r=20), legend=dict(font=dict(size=10))
    )

    # Plot 2: Cost Function vs Threshold
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(x=threshold_df['Threshold'], y=threshold_df['Total_Cost'], mode='lines', name='Total Cost (FN=5, FP=1)', line=dict(color='#8b5cf6', width=3)))
    fig_cost.add_vline(x=t_val, line=dict(color='#f59e0b', dash='dash'), annotation_text=f"Selected ({t_val:.2f})", annotation_position="top left")
    fig_cost.add_vline(x=0.50, line=dict(color='#475569', dash='dot'), annotation_text="Default 0.50", annotation_position="bottom right")
    fig_cost.update_layout(
        **CHART_LAYOUT,
        title="Institutional Penalty Function vs Threshold", xaxis_title="Threshold", yaxis_title="Total Penalty Score",
        height=350, margin=dict(t=40, b=20, l=40, r=20)
    )

    return kpi_card, fig_metrics, fig_cost

# ==============================================================================
# TAB 5: DEMO PROFILE PRESET AUTO-FILL
# ==============================================================================
@app.callback(
    [Output("inp-cgpa", "value"),
     Output("inp-sem-gpa", "value"),
     Output("inp-gpa", "value"),
     Output("inp-attendance", "value"),
     Output("inp-study-hours", "value"),
     Output("inp-delay-days", "value"),
     Output("inp-stress", "value"),
     Output("inp-travel", "value"),
     Output("inp-income", "value"),
     Output("inp-dept", "value"),
     Output("inp-sem", "value"),
     Output("inp-sch", "value"),
     Output("inp-job", "value"),
     Output("inp-net", "value"),
     Output("inp-gender", "value"),
     Output("inp-age", "value"),
     Output("inp-parent-edu", "value")],
    Input("demo-profile-dropdown", "value")
)
def autofill_profile(preset):
    if preset == "high":
        # Critical Dropout Risk Profile
        return 1.40, 1.10, 1.45, 64.0, 1.8, 4, 8.5, 55, 22000, "CS", "Year 2", "No", "Yes", "No", "Male", 21, "High School"
    elif preset == "moderate":
        # Borderline / Moderate Risk Profile
        return 2.45, 2.10, 2.50, 74.0, 2.6, 2, 6.5, 35, 38000, "Business", "Year 2", "No", "Yes", "Yes", "Female", 20, "Bachelor"
    elif preset == "low":
        # Strong Academic Retention Profile
        return 3.65, 3.80, 3.60, 94.0, 5.0, 0, 3.2, 15, 65000, "Engineering", "Year 3", "Yes", "No", "Yes", "Female", 21, "Master"
    # Default to high risk
    return 1.40, 1.10, 1.45, 64.0, 1.8, 4, 8.5, 55, 22000, "CS", "Year 2", "No", "Yes", "No", "Male", 21, "High School"

# ==============================================================================
# TAB 5: INFERENCE PREDICTION & VISUALIZATION
# ==============================================================================
@app.callback(
    [Output("prediction-output-container", "children"),
     Output("stored-prediction-data", "data")],
    Input("btn-predict-risk", "n_clicks"),
    [State("inp-cgpa", "value"),
     State("inp-sem-gpa", "value"),
     State("inp-gpa", "value"),
     State("inp-attendance", "value"),
     State("inp-study-hours", "value"),
     State("inp-delay-days", "value"),
     State("inp-stress", "value"),
     State("inp-travel", "value"),
     State("inp-income", "value"),
     State("inp-dept", "value"),
     State("inp-sem", "value"),
     State("inp-sch", "value"),
     State("inp-job", "value"),
     State("inp-net", "value"),
     State("inp-gender", "value"),
     State("inp-age", "value"),
     State("inp-parent-edu", "value")]
)
def handle_prediction(n_clicks, cgpa, sem_gpa, gpa, attendance, study_hours,
                      delay_days, stress, travel, income, dept, sem,
                      sch, job, net, gender, age, parent_edu):
    # Assemble student dictionary
    student_payload = {
        "Age": float(age or 20),
        "Gender": str(gender or "Male"),
        "Family_Income": float(income or 30000),
        "Internet_Access": str(net or "Yes"),
        "Study_Hours_per_Day": float(study_hours or 3.0),
        "Attendance_Rate": float(attendance or 80.0),
        "Assignment_Delay_Days": float(delay_days or 1.0),
        "Travel_Time_Minutes": float(travel or 25.0),
        "Part_Time_Job": str(job or "No"),
        "Scholarship": str(sch or "No"),
        "Stress_Index": float(stress or 5.0),
        "GPA": float(gpa or 2.5),
        "Semester_GPA": float(sem_gpa or 2.5),
        "CGPA": float(cgpa or 2.5),
        "Semester": str(sem or "Year 1"),
        "Department": str(dept or "CS"),
        "Parental_Education": str(parent_edu or "Bachelor")
    }

    result = predict_student(student_payload)

    # Persist inference audit record to MongoDB backend
    mongo_id = None
    if db_manager.is_connected:
        try:
            mongo_id = db_manager.insert_prediction(student_payload, result)
        except Exception as e:
            print(f"[MongoDB] Inference logging error: {e}")

    # Gauge Figure
    prob_val = result['probability_percent']
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_val,
        number={'suffix': "%", 'font': {'size': 44, 'color': result['risk_color'], 'family': 'Inter, sans-serif'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8", 'tickfont': {'family': 'Inter'}},
            'bar': {'color': result['risk_color'], 'thickness': 0.3},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': 'rgba(139, 92, 246, 0.2)',
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.15)'},
                {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [60, 80], 'color': 'rgba(244, 63, 94, 0.15)'},
                {'range': [80, 100], 'color': 'rgba(244, 63, 94, 0.25)'}
            ],
            'threshold': {
                'line': {'color': "#e8eaf6", 'width': 4},
                'thickness': 0.8,
                'value': float(result['threshold'] * 100)
            }
        }
    )).update_layout(
        **CHART_LAYOUT,
        height=220, margin=dict(t=10, b=10, l=30, r=30)
    )

    # Risk Factor Badges
    risk_list = [html.Li([html.I(className="fas fa-exclamation-triangle text-danger me-2"), f]) for f in result['risk_factors']]
    if not risk_list:
        risk_list = [html.Li("No critical risk factor flags detected.")]

    prot_list = [html.Li([html.I(className="fas fa-shield-alt text-success me-2"), f]) for f in result['protective_factors']]
    if not prot_list:
        prot_list = [html.Li("No significant protective factor flags detected.")]

    output_ui = dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.H5("Inference Diagnostic Report", className="mb-0 text-white font-weight-bold d-inline"),
                dbc.Badge([
                    html.I(className="fas fa-database me-1"),
                    f"Saved to MongoDB ({str(mongo_id)[:8]}...)" if mongo_id else "MongoDB Synced"
                ], color="success", className="ms-3 p-2") if db_manager.is_connected else dbc.Badge("Offline Mode", color="secondary", className="ms-3 p-2"),
                dbc.Badge(result['risk_level'].upper(), color="danger" if "High" in result['risk_level'] or "Critical" in result['risk_level'] else ("warning" if "Moderate" in result['risk_level'] else "success"), className="float-end p-2")
            ])
        ]),
        dbc.CardBody([
            # Gauge & Verdict
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=fig_gauge, config={'displayModeBar': False})
                ], width=12, md=6),
                dbc.Col([
                    html.H6("Model Classification Decision", className="text-muted text-uppercase mb-1"),
                    html.H3(
                        result['prediction_label'],
                        className="font-weight-bold " + ("text-danger" if result['prediction'] == 1 else "text-success")
                    ),
                    html.P([
                        f"Estimated Dropout Probability is ",
                        html.Strong(f"{result['probability_percent']}%"),
                        f" (Classification Cutoff: {result['threshold']:.2f})."
                    ], className="mb-2"),
                    html.Small([
                        f"Academic Average: {result['student_summary']['Academic_Average']} | ",
                        f"Grade Trend: {result['student_summary']['Academic_Trend']:+.2f} | ",
                        f"Attendance: {result['student_summary']['Attendance_Rate']}%"
                    ], className="text-info d-block mb-2"),
                    dbc.Progress(value=prob_val, color="danger" if prob_val>60 else ("warning" if prob_val>30 else "success"), className="mb-2")
                ], width=12, md=6, className="d-flex flex-column justify-content-center")
            ]),

            html.Hr(style={"borderColor": "#34495e"}),

            # Risk vs Protective Factors
            dbc.Row([
                dbc.Col([
                    html.H6("Key Attrition Risk Factors:", className="text-danger font-weight-bold mb-2"),
                    html.Ul(risk_list, className="small text-muted ps-3 mb-0")
                ], width=12, md=6),
                dbc.Col([
                    html.H6("Protective / Retention Anchors:", className="text-success font-weight-bold mb-2"),
                    html.Ul(prot_list, className="small text-muted ps-3 mb-0")
                ], width=12, md=6)
            ], className="mb-3"),

            html.Hr(style={"borderColor": "#34495e"}),

            # Instant Recommendations Preview
            html.H6("Institutional Mentoring Directives:", className="text-warning font-weight-bold mb-2"),
            html.Div([
                dbc.Alert([
                    html.Strong(f"{act['category']}: "),
                    html.Span(f"{act['action']} "),
                    html.Small(f"({act['rationale']})", className="text-muted d-block mt-1")
                ], color="dark", className="mb-2 p-2 border border-secondary")
                for act in result['interventions'][:3]
            ]),

            html.Small("Statistical Non-Causality Disclaimer: Feature importances reflect associative weighting within the trained decision surface, not definitive individual causality.", className="text-muted fst-italic d-block mt-2")
        ])
    ], style={"backgroundColor": CUSTOM_STYLE['card_bg'], "border": CUSTOM_STYLE['card_border']})

    return output_ui, result

# ==============================================================================
# TAB 6: BUSINESS INSIGHTS DETAIL CALLBACK
# ==============================================================================
@app.callback(
    Output("insights-detail-container", "children"),
    Input("stored-prediction-data", "data")
)
def update_insights_tab(stored_data):
    if not stored_data:
        # Default fallback to high risk preview
        stored_data = predict_student({
            "Age": 21.0, "Gender": "Male", "Family_Income": 22000.0, "Internet_Access": "No",
            "Study_Hours_per_Day": 1.8, "Attendance_Rate": 64.0, "Assignment_Delay_Days": 4.0,
            "Travel_Time_Minutes": 55.0, "Part_Time_Job": "Yes", "Scholarship": "No",
            "Stress_Index": 8.5, "GPA": 1.45, "Semester_GPA": 1.10, "CGPA": 1.40,
            "Semester": "Year 2", "Department": "CS", "Parental_Education": "High School"
        })

    domain_risks = stored_data['domain_risks']
    interventions = stored_data['interventions']

    def badge_for_rating(r):
        color = "danger" if r == "High" else ("warning" if r == "Moderate" else "success")
        return dbc.Badge(r.upper(), color=color, className="p-2")

    # Radar Chart of Risk Domains
    categories = ['Academic Risk', 'Engagement Risk', 'Financial Risk', 'Accessibility Risk', 'Stress Risk']
    level_map = {'Low': 1, 'Moderate': 2, 'High': 3}
    values = [
        level_map.get(domain_risks['Academic_Risk'], 1),
        level_map.get(domain_risks['Engagement_Risk'], 1),
        level_map.get(domain_risks['Financial_Social_Risk'], 1),
        level_map.get(domain_risks['Accessibility_Risk'], 1),
        level_map.get(domain_risks['Stress_Risk'], 1)
    ]
    values.append(values[0])
    categories.append(categories[0])

    fig_radar = go.Figure(go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        fillcolor='rgba(139, 92, 246, 0.25)', line=dict(color='#8b5cf6', width=2),
        marker=dict(color='#00d4ff', size=6)
    )).update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 3], tickvals=[1, 2, 3], ticktext=['Low', 'Moderate', 'High'],
                           gridcolor='rgba(139,92,246,0.1)', linecolor='rgba(139,92,246,0.15)'),
            angularaxis=dict(gridcolor='rgba(139,92,246,0.1)', linecolor='rgba(139,92,246,0.15)')
        ),
        **CHART_LAYOUT,
        height=320, margin=dict(t=20, b=20, l=40, r=40)
    )

    domain_cards = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Academic Dimension", className="text-muted small"),
                badge_for_rating(domain_risks['Academic_Risk']),
                html.Small("GPA & Term Momentum", className="text-muted d-block mt-2")
            ], className="text-center")
        ], color="dark", outline=True), width=6, md=2, className="mb-2"),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Engagement", className="text-muted small"),
                badge_for_rating(domain_risks['Engagement_Risk']),
                html.Small("Attendance & Submission Delays", className="text-muted d-block mt-2")
            ], className="text-center")
        ], color="dark", outline=True), width=6, md=2, className="mb-2"),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Socioeconomic", className="text-muted small"),
                badge_for_rating(domain_risks['Financial_Social_Risk']),
                html.Small("Scholarship & Part-time Job", className="text-muted d-block mt-2")
            ], className="text-center")
        ], color="dark", outline=True), width=6, md=2, className="mb-2"),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Accessibility", className="text-muted small"),
                badge_for_rating(domain_risks['Accessibility_Risk']),
                html.Small("Commute Time & Internet Access", className="text-muted d-block mt-2")
            ], className="text-center")
        ], color="dark", outline=True), width=6, md=2, className="mb-2"),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Psychological Stress", className="text-muted small"),
                badge_for_rating(domain_risks['Stress_Risk']),
                html.Small("Self-reported Stress Index", className="text-muted d-block mt-2")
            ], className="text-center")
        ], color="dark", outline=True), width=6, md=2, className="mb-2"),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Attrition Verdict", className="text-muted small"),
                html.H5(f"{stored_data['probability_percent']}%", className="text-danger font-weight-bold mb-0"),
                html.Small(stored_data['risk_level'], className="text-warning")
            ], className="text-center")
        ], color="dark", outline=True), width=6, md=2, className="mb-2")
    ], className="mb-4")

    # Action Items List
    action_items = [
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Span(act['category'], className="badge bg-info text-dark font-weight-bold me-2"),
                        html.H6(act['action'], className="text-white d-inline font-weight-bold"),
                        html.P(act['rationale'], className="text-muted small mb-0 mt-1")
                    ], width=10),
                    dbc.Col([
                        dbc.Button("Assign Task", color="outline-info", size="sm", className="float-end")
                    ], width=2, className="d-flex align-items-center justify-content-end")
                ])
            ])
        ], color="dark", outline=True, className="mb-2")
        for act in interventions
    ]

    return html.Div([
        domain_cards,
        dbc.Row([
            dbc.Col([
                html.H5("Institutional Intervention Roadmap", className="text-info font-weight-bold mb-3"),
                html.Div(action_items)
            ], width=12, lg=7),
            dbc.Col([
                html.H5("Multi-Facet Risk Radar", className="text-info font-weight-bold mb-3"),
                dcc.Graph(figure=fig_radar, config={'displayModeBar': False})
            ], width=12, lg=5)
        ])
    ])

# ==============================================================================
# TAB 7: SHAP GLOBAL EXPLAINABILITY
# ==============================================================================
@app.callback(
    Output("shap-summary-chart", "figure"),
    Input("main-tabs", "active_tab")
)
def update_shap_summary(tab_id):
    if not shap_data:
        return go.Figure()

    shap_imp = pd.DataFrame(shap_data['shap_importance']).head(15).sort_values(by='Mean_Abs_SHAP', ascending=True)
    fig_shap = px.bar(
        shap_imp, x="Mean_Abs_SHAP", y="Feature", orientation='h',
        color="Mean_Abs_SHAP",
        color_continuous_scale=[[0, '#8b5cf6'], [0.5, '#00d4ff'], [1, '#10b981']],
        title="Top 15 Most Influential Features across Validation Population"
    ).update_layout(
        **CHART_LAYOUT,
        height=400, margin=dict(t=40, b=20, l=120, r=20), xaxis_title="Mean |SHAP Value| (Impact on Model Log-Odds)"
    )
    return fig_shap


# ==============================================================================
# TAB 8: MONGODB BACKEND CALLBACKS
# ==============================================================================
@app.callback(
    [Output("db-history-table-container", "children"),
     Output("db-action-feedback", "children")],
    [Input("btn-db-seed", "n_clicks"),
     Input("btn-db-refresh", "n_clicks"),
     Input("btn-db-clear", "n_clicks")],
    prevent_initial_call=True
)
def manage_db_actions(n_seed, n_refresh, n_clear):
    ctx = callback_context
    if not ctx.triggered:
        return render_db_history_table(), None

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    feedback = None

    if button_id == "btn-db-seed":
        count = db_manager.seed_sample_data(5)
        feedback = dbc.Alert(f"Successfully generated and inserted {count} sample student prediction records into MongoDB!", color="success", dismissable=True, className="py-2 mb-0")
    elif button_id == "btn-db-clear":
        deleted = db_manager.clear_prediction_history()
        feedback = dbc.Alert(f"Cleared {deleted} prediction records from MongoDB collection.", color="warning", dismissable=True, className="py-2 mb-0")
    elif button_id == "btn-db-refresh":
        feedback = dbc.Alert("Database tables and metrics refreshed from MongoDB server.", color="info", dismissable=True, className="py-2 mb-0")

    return render_db_history_table(), feedback


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STARTING STUDENT DROPOUT RISK ANALYZER APPLICATION")
    print("Running on http://127.0.0.1:8050/")
    print("=" * 80 + "\n")
    app.run(host="127.0.0.1", port=8050, debug=False)
