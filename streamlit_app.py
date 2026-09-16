"""
CASEFILE — Streamlit UI for Missing Person Location Prediction
==============================================================
A redesigned Streamlit port of the Flask web UI (templates/index.html
+ app.py). Provides a performance dashboard (5x repeated evaluation
summary) and an interactive Top-5 probable-location predictor with
color-coded rank cards, probability charts, and ethical framing.

Run locally:
    streamlit run streamlit_app.py

Artifacts required (from build_model.py):
    models/location_model.pkl, models/label_encoder.pkl,
    results/evaluation_summary.json
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from casefile_common import engineer_features, get_feature_cols

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "location_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
SUMMARY_PATH = os.path.join(BASE_DIR, "results", "evaluation_summary.json")

PRIORITY_LABELS = ["Very High", "High", "Medium", "Low", "Very Low"]
PRIORITY_COLORS = ["#ef4444", "#f97316", "#eab308", "#3b82f6", "#6b7280"]
AREAS = [f"Area_{chr(65 + i)}" for i in range(10)]


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    """Load model, label encoder and evaluation summary once per session."""
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    with open(SUMMARY_PATH, encoding="utf-8") as fh:
        summary = json.load(fh)
    return model, label_encoder, summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_pct(val):
    return f"{val * 100:.1f}%"


def fmt_pct_std(mean, std):
    return f"{mean * 100:.1f}% ± {std * 100:.1f}%"


def build_case_dict(age_group, gender, lat, lon, last_seen_time, day, weather,
                    usual_area, avg_distance, avg_speed, previous_area, time_since):
    return {
        "Case_ID": "UI-0001",
        "Person_ID": "P-UI01",
        "Age_Group": age_group,
        "Gender": gender,
        "Last_Latitude": float(lat),
        "Last_Longitude": float(lon),
        "Last_Seen_Time": last_seen_time,
        "Day": day,
        "Weather": weather,
        "Usual_Area": usual_area,
        "Average_Distance": float(avg_distance),
        "Average_Speed": float(avg_speed),
        "Previous_Area": previous_area,
        "Time_Since_Last_Seen": float(time_since),
    }


def predict_top5(model, label_encoder, row):
    """Return top-5 area predictions with probabilities + priority labels."""
    input_eng = engineer_features(pd.DataFrame([row]))
    feature_cols = get_feature_cols(input_eng)
    probs = model.predict_proba(input_eng[feature_cols])[0]
    top_indices = np.argsort(probs)[::-1][:5]
    predictions = []
    for rank, idx in enumerate(top_indices):
        predictions.append({
            "rank": rank + 1,
            "area": label_encoder.inverse_transform([idx])[0],
            "probability": round(float(probs[idx]) * 100, 1),
            "priority": PRIORITY_LABELS[rank],
            "color": PRIORITY_COLORS[rank],
        })
    return predictions


def make_comparison_figure(models_summary):
    """Grouped horizontal bar chart: Top-1/3/5 accuracy by model."""
    mdf = pd.DataFrame(models_summary).sort_values("top1_mean", ascending=True)
    fig = go.Figure()
    traces = [
        ("Top-1", "top1_mean", "#06b6d4"),
        ("Top-3", "top3_mean", "#a855f7"),
        ("Top-5", "top5_mean", "#22c55e"),
    ]
    for label, key, color in traces:
        fig.add_trace(go.Bar(
            name=label,
            x=mdf[key] * 100,
            y=mdf["model"],
            orientation="h",
            marker_color=color,
            text=[f"{v * 100:.1f}%" for v in mdf[key]],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Mean accuracy (%)",
        xaxis=dict(range=[0, 95]),
        hovermode="y unified",
        legend=dict(orientation="h", y=1.08),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0f172a"),
    )
    return fig


def make_prediction_figure(predictions):
    """Horizontal bar chart of the top-5 prediction probabilities."""
    fig = go.Figure(go.Bar(
        x=[p["probability"] for p in predictions],
        y=[p["area"] for p in predictions][::-1],
        orientation="h",
        marker_color=[p["color"] for p in predictions][::-1],
        text=[f"{p['probability']}%" for p in predictions][::-1],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, max(p["probability"] for p in predictions) + 20],
                   title="Probability (%)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0f172a"),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Page config & global CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CASEFILE · Missing Person Location Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --cf-cyan: #06b6d4;
        --cf-green: #22c55e;
        --cf-purple: #a855f7;
        --cf-orange: #f97316;
        --cf-red: #ef4444;
        --cf-ink: #0f172a;
        --cf-muted: #64748b;
    }

    /* ---------- global ---------- */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--cf-ink);
    }
    .stApp { background: linear-gradient(180deg, #f1f5f9 0%, #f8fafc 40%); }
    .block-container { padding-top: 1.2rem; padding-bottom: 3rem; }

    /* hide streamlit chrome */
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

    /* ---------- typography ---------- */
    h1, h2, h3 { font-weight: 800 !important; color: var(--cf-ink); }
    .cf-mono { font-family: 'JetBrains Mono', monospace; }

    /* ---------- hero ---------- */
    .cf-hero {
        background: linear-gradient(135deg, #0b1220 0%, #134e4a 60%, #155e75 100%);
        border-radius: 16px;
        padding: 28px 34px 24px;
        color: #e2e8f0;
        position: relative;
        overflow: hidden;
        margin-bottom: 8px;
    }
    .cf-hero .cf-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #67e8f9;
        background: rgba(6, 182, 212, 0.15);
        border: 1px solid rgba(103, 232, 249, 0.35);
        border-radius: 99px;
        padding: 4px 12px;
        margin-bottom: 12px;
    }
    .cf-hero h1 {
        color: #ffffff !important;
        font-size: 2.4rem;
        letter-spacing: 0.05em;
        margin: 0;
        line-height: 1.05;
    }
    .cf-hero .cf-subtitle { font-size: 1.02rem; color: #94a3b8; margin: 4px 0 10px; }
    .cf-hero .cf-ethics {
        display: flex; align-items: center; gap: 8px;
        font-size: 0.85rem; color: #cbd5e1;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px; padding: 8px 12px; width: fit-content;
    }

    /* ---------- section ---------- */
    .cf-section {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px 22px;
        margin: 12px 0;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .cf-section-title { font-size: 1.05rem; font-weight: 800; margin: 0 0 2px; }
    .cf-section-hint { font-size: 0.8rem; color: var(--cf-muted); margin-bottom: 14px; }

    /* ---------- metric cards ---------- */
    .cf-metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
    .cf-metric {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid var(--cf-cyan);
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(15,23,42,0.05);
    }
    .cf-metric .cf-metric-label {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.09em;
        text-transform: uppercase; color: var(--cf-muted);
    }
    .cf-metric .cf-metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.35rem; font-weight: 700; margin-top: 4px;
    }
    .cf-metric .cf-metric-sub { font-size: 0.78rem; color: var(--cf-muted); margin-top: 2px; }

    /* ---------- prediction cards ---------- */
    .cf-pred {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid var(--cf-cyan);
        border-radius: 12px;
        padding: 12px 16px;
        margin: 10px 0;
        box-shadow: 0 1px 2px rgba(15,23,42,0.05);
    }
    .cf-pred-top { display: flex; justify-content: space-between; align-items: center; }
    .cf-pred-area { font-weight: 800; font-size: 1.02rem; }
    .cf-pred-rank { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.85rem; }
    .cf-pred-pct { font-family: 'JetBrains Mono', monospace; font-weight: 800; font-size: 1.1rem; }
    .cf-pred-badge {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em;
        border-radius: 99px; padding: 3px 10px; margin-left: 8px;
    }
    .cf-bar-track {
        background: #e8edf3; border-radius: 6px; height: 10px;
        margin-top: 10px; overflow: hidden;
    }
    .cf-bar-fill { height: 100%; border-radius: 6px; }

    /* ---------- case chip ---------- */
    .cf-case-chip {
        display: flex; flex-wrap: wrap; gap: 8px 12px;
        background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 12px 16px; margin-bottom: 14px;
        font-size: 0.86rem; color: #334155;
    }
    .cf-chip { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 3px 10px; }

    /* ---------- ethical notice ---------- */
    .cf-ethics-note {
        margin-top: 14px; padding: 12px 16px;
        border: 1px solid #d7e6f0;
        border-left: 4px solid #94a3b8;
        border-radius: 12px; background: #f1f5f9;
        color: #475569; font-size: 0.85rem; line-height: 1.5;
    }

    /* ---------- form sections ---------- */
    .cf-form-caption {
        font-size: 0.78rem; font-weight: 800; letter-spacing: 0.09em;
        text-transform: uppercase; color: var(--cf-muted);
        margin: 14px 0 4px; padding-top: 10px; border-top: 1px solid #eef2f7;
    }
    .cf-form-caption:first-child { margin-top: 0; border-top: none; padding-top: 0; }

    /* ---------- footer ---------- */
    .cf-footer {
        text-align: center; color: #94a3b8; font-size: 0.8rem;
        margin: 26px 0 0; padding-top: 16px; border-top: 1px solid #e2e8f0;
    }

    /* ---------- tab styling ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px; padding: 6px 18px; font-weight: 700;
    }
    .stTabs [aria-selected="true"] { background: rgba(6,182,212,0.12); }

    /* ---------- sidebar ---------- */
    [data-testid="stSidebar"] {
        background: #0f172a;
    }
    [data-testid="stSidebar"] * { color: #cbd5e1; }
    [data-testid="stSidebar"] .cf-sidebar-title {
        font-size: 1.1rem; font-weight: 800; color: #ffffff; margin: 0;
    }
    [data-testid="stSidebar"] .cf-sidebar-sub { font-size: 0.78rem; color: #94a3b8; }
    .cf-sidebar-box {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px; padding: 10px 12px; margin: 10px 0;
        font-size: 0.82rem;
    }

    @media (max-width: 900px) {
        .cf-metric-grid { grid-template-columns: repeat(2, 1fr); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="cf-sidebar-title">CASEFILE</div>
        <div class="cf-sidebar-sub">Missing Person Location Prediction System</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="cf-sidebar-box"><strong>What this tool does</strong><br>'
        "Ranks the 10 search areas by predicted probability using a trained "
        "classifier, so field teams can prioritize where to look first.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="cf-sidebar-box"><strong>Model</strong><br>'
        "Logistic Regression ensemble · Top-1 ≈ 36.6% · Top-5 ≈ 87.1%<br>"
        "<span class='cf-sidebar-sub'>Trained on synthetic data only</span></div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="cf-hero">
        <span class="cf-badge">ML Prioritization Tool</span>
        <h1>CASEFILE</h1>
        <div class="cf-subtitle">Missing Person Location Prediction System</div>
        <div class="cf-ethics">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#fbbf24"
                 stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg>
            Statistical prioritization aid — probabilities, not certainties
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_dashboard, tab_predictor = st.tabs(["Dashboard", "New Case Prediction"])

# ===========================================================================
# DASHBOARD
# ===========================================================================
with tab_dashboard:
    with st.spinner("Loading performance summary..."):
        _, _, summary = load_artifacts()
    best = next(m for m in summary["models"] if m["model"] == summary["best_model"])

    st.markdown(
        f"""
        <div class="cf-section">
            <div class="cf-section-title">Model Performance</div>
            <div class="cf-section-hint">{summary["repeats"]} repeats · seeds [
            {", ".join(map(str, summary["seeds"]))}] · {summary["n_cases"]} cases ·
            {summary["n_features"]} features</div>
            <div class="cf-metric-grid">
                <div class="cf-metric" style="border-left-color:#06b6d4">
                    <div class="cf-metric-label">Best Model</div>
                    <div class="cf-metric-value">{best["model"]}</div>
                    <div class="cf-metric-sub">by Top-1 accuracy</div>
                </div>
                <div class="cf-metric" style="border-left-color:#22c55e">
                    <div class="cf-metric-label">Top-1 Accuracy</div>
                    <div class="cf-metric-value">{fmt_pct_std(best["top1_mean"], best["top1_std"])}</div>
                    <div class="cf-metric-sub">mean ± std</div>
                </div>
                <div class="cf-metric" style="border-left-color:#a855f7">
                    <div class="cf-metric-label">Top-3 Accuracy</div>
                    <div class="cf-metric-value">{fmt_pct_std(best["top3_mean"], best["top3_std"])}</div>
                    <div class="cf-metric-sub">mean ± std</div>
                </div>
                <div class="cf-metric" style="border-left-color:#f97316">
                    <div class="cf-metric-label">Top-5 Accuracy</div>
                    <div class="cf-metric-value">{fmt_pct_std(best["top5_mean"], best["top5_std"])}</div>
                    <div class="cf-metric-sub">mean ± std</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_chart, col_table = st.columns([1, 1.15])

    with col_chart:
        st.markdown('<div class="cf-section"><div class="cf-section-title">Model Comparison</div>'
                    '<div class="cf-section-hint">Mean accuracy across 5 evaluation repeats</div>')
        fig = make_comparison_figure(summary["models"])
        st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_table:
        st.markdown('<div class="cf-section"><div class="cf-section-title">Per-Model Metrics</div>'
                    '<div class="cf-section-hint">Top-1 shown as mean ± std</div>')
        rows = []
        for m in sorted(summary["models"], key=lambda x: x["top1_mean"], reverse=True):
            rows.append({
                "Model": m["model"],
                "Top-1": fmt_pct_std(m["top1_mean"], m["top1_std"]),
                "Top-3": fmt_pct(m["top3_mean"]),
                "Top-5": fmt_pct(m["top5_mean"]),
                "Macro F1": fmt_pct(m["macro_f1_mean"]),
                "Weighted F1": fmt_pct(m["weighted_f1_mean"]),
            })
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
            height=320,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="cf-ethics-note"><strong>Ethical notice:</strong> All data is 100% '
        "synthetic &amp; fictional. Metrics reflect repeated cross-validation on simulated "
        "cases only and are not a measure of real-world field performance.</div>",
        unsafe_allow_html=True,
    )

# ===========================================================================
# PREDICTOR
# ===========================================================================
with tab_predictor:
    st.markdown(
        '<div class="cf-section"><div class="cf-section-title">New Case Prediction</div>'
        '<div class="cf-section-hint">Enter case details below to rank the 10 search areas '
        "by predicted likelihood</div>"
    )

    form_col, result_col = st.columns([1, 1.1])

    with form_col:
        with st.form("case-form"):
            st.markdown('<div class="cf-form-caption">Person</div>')
            c1, c2 = st.columns(2)
            age_group = c1.selectbox("Age Group", ["18-25", "26-35", "36-45", "46-55", "56+"], index=1)
            gender = c2.selectbox("Gender", ["Male", "Female", "Other"], index=0)

            st.markdown('<div class="cf-form-caption">Location</div>')
            c1, c2 = st.columns(2)
            lat = c1.number_input("Last Latitude", value=23.2500, step=0.0001, format="%.4f")
            lon = c2.number_input("Last Longitude", value=77.4100, step=0.0001, format="%.4f")
            c1, c2 = st.columns(2)
            usual_area = c1.selectbox("Usual Area", AREAS, index=4)
            previous_area = c2.selectbox("Previous Area", AREAS, index=4)

            st.markdown('<div class="cf-form-caption">Time &amp; Conditions</div>')
            c1, c2 = st.columns(2)
            last_seen_time = c1.text_input("Last Seen Time (HH:MM)", value="14:30")
            day = c2.selectbox("Day",
                               ["Monday", "Tuesday", "Wednesday", "Thursday",
                                "Friday", "Saturday", "Sunday"], index=4)
            weather = st.selectbox("Weather", ["Clear", "Cloudy", "Rain", "Fog"], index=0)

            st.markdown('<div class="cf-form-caption">Movement</div>')
            c1, c2, c3 = st.columns(3)
            avg_distance = c1.number_input("Average Distance (km)", value=9.0, step=0.1)
            avg_speed = c2.number_input("Average Speed (km/h)", value=20.0, step=0.1)
            time_since = c3.number_input("Time Since Seen (hrs)", value=4.0, min_value=0.1, step=0.1)

            submitted = st.form_submit_button("Generate Prediction Report", type="primary",
                                              width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with result_col:
        st.markdown(
            '<div class="cf-section"><div class="cf-section-title">Probable Location Report</div>'
        )
        if not submitted:
            st.markdown(
                '<div style="text-align:center; color:#94a3b8; padding:60px 0;">'
                '<div style="font-size:2rem; line-height:1;">&#128270;</div>'
                "<div>Fill in the case form and hit <strong>Generate Prediction Report</strong> "
                "to see the Top-5 probable areas.</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            model, label_encoder, _ = load_artifacts()
            row = build_case_dict(age_group, gender, lat, lon, last_seen_time, day, weather,
                                  usual_area, avg_distance, avg_speed, previous_area, time_since)

            with st.spinner("Computing area probabilities..."):
                predictions = predict_top5(model, label_encoder, row)

            st.markdown(
                f"""
                <div class="cf-case-chip">
                    <span class="cf-chip"><strong>Case</strong> {row['Case_ID']}</span>
                    <span class="cf-chip"><strong>Person</strong> {row['Age_Group']} · {row['Gender']}</span>
                    <span class="cf-chip"><strong>Day</strong> {row['Day']}</span>
                    <span class="cf-chip"><strong>Weather</strong> {row['Weather']}</span>
                    <span class="cf-chip"><strong>Usual</strong> {row['Usual_Area']}</span>
                    <span class="cf-chip"><strong>Previous</strong> {row['Previous_Area']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            pred_html = ""
            for p in predictions:
                pred_html += f"""
                <div class="cf-pred" style="border-left-color:{p['color']}">
                    <div class="cf-pred-top">
                        <span class="cf-pred-area">#{p['rank']} {p['area']}
                            <span class="cf-pred-badge"
                                  style="color:{p['color']}; background:{p['color']}18; border:1px solid {p['color']}44;">{p['priority']}</span>
                        </span>
                        <span class="cf-pred-pct" style="color:{p['color']}">{p['probability']}%</span>
                    </div>
                    <div class="cf-bar-track">
                        <div class="cf-bar-fill" style="width:{p['probability']}%; background:{p['color']};"></div>
                    </div>
                </div>
                """
            st.markdown(pred_html, unsafe_allow_html=True)

            st.plotly_chart(make_prediction_figure(predictions), width="stretch")

            st.markdown(
                '<div class="cf-ethics-note"><strong>Ethical notice:</strong> This output is a '
                "statistical probability estimate for search prioritization only. It does not "
                "constitute a definitive prediction or proof of location. Always cross-reference "
                "with field intelligence.</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="cf-footer">CASEFILE · Missing Person Location Prediction System '
    "· All data is 100% synthetic &amp; fictional</div>",
    unsafe_allow_html=True,
)