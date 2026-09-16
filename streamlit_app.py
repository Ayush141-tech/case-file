"""
CASEFILE — Streamlit UI for Missing Person Location Prediction
==============================================================
A fully native Streamlit app (no HTML/CSS injection) mirroring the
Flask web UI (templates/index.html + app.py). Provides a performance
dashboard (5x repeated evaluation summary) and an interactive Top-5
probable-location predictor.

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

MODEL_COLORS = {
    "Gradient Boosting": "#06b6d4",
    "Logistic Regression": "#22c55e",
    "Random Forest": "#a855f7",
    "XGBoost": "#f97316",
}


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


def predictions_to_frame(predictions):
    """Convert prediction dicts into a tidy DataFrame for display/export."""
    return pd.DataFrame([
        {"Rank": p["rank"], "Area": p["area"],
         "Probability (%)": p["probability"], "Priority": p["priority"]}
        for p in predictions
    ])


def predict_all_areas(model, label_encoder, row):
    """Return a ranked DataFrame of all 10 areas with probabilities."""
    input_eng = engineer_features(pd.DataFrame([row]))
    feature_cols = get_feature_cols(input_eng)
    probs = model.predict_proba(input_eng[feature_cols])[0]
    areas = label_encoder.inverse_transform(np.arange(len(probs)))
    df = pd.DataFrame({"Area": areas, "Probability (%)": np.round(probs * 100, 1)})
    df = df.sort_values("Probability (%)", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    return df


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
        height=340,
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


def make_radar_figure(models_summary):
    """Radar chart comparing model quality across all five metrics."""
    dims = ["Top-1", "Top-3", "Top-5", "Macro F1", "Weighted F1"]
    keys = ["top1_mean", "top3_mean", "top5_mean", "macro_f1_mean", "weighted_f1_mean"]
    fig = go.Figure()
    for m in models_summary:
        r = [m[k] * 100 for k in keys]
        color = MODEL_COLORS.get(m["model"], "#06b6d4")
        fig.add_trace(go.Scatterpolar(
            r=r + [r[0]],
            theta=dims + [dims[0]],
            mode="lines+markers",
            name=m["model"],
            line=dict(color=color, width=2),
            marker=dict(color=color, size=4),
            fill="toself",
            opacity=0.6,
        ))
    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=10, b=10),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#e2e8f0"),
            angularaxis=dict(gridcolor="#e2e8f0", tickfont=dict(color="#475569")),
        ),
        legend=dict(orientation="h", y=-0.15, font=dict(color="#475569")),
        paper_bgcolor="rgba(0,0,0,0)",
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
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, max(p["probability"] for p in predictions) + 20],
                   title="Probability (%)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0f172a"),
        showlegend=False,
        bargap=0.35,
    )
    return fig


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CASEFILE · Missing Person Location Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar (native widgets)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("CASEFILE")
    st.caption("Missing Person Location Prediction System")

    st.divider()

    st.header("About")
    st.write(
        "A machine-learning decision aid that ranks the 10 search areas by "
        "predicted probability, so field teams can prioritize where to look first."
    )

    st.header("How to use")
    st.write(
        "1. Open **New Case Prediction**.\n\n"
        "2. Enter the last-seen case details.\n\n"
        "3. Generate the report and review the Top-5 areas."
    )

    st.header("Priority scale")
    st.dataframe(
        pd.DataFrame({"Priority": PRIORITY_LABELS}),
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.caption("Probabilities, not certainties. All data is 100% synthetic & fictional.")

# ---------------------------------------------------------------------------
# Hero (native typography)
# ---------------------------------------------------------------------------

st.title("CASEFILE")
st.caption("Missing Person Location Prediction System")
st.info("Statistical prioritization aid — probabilities, not certainties.", icon=None)
st.divider()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None


def render_results(model, label_encoder, row, predictions=None):
    """Render the report panel using native Streamlit components."""
    if predictions is None:
        predictions = predict_top5(model, label_encoder, row)

    top = predictions[0]

    st.success(
        f"Highest priority area: **{top['area']}** at **{top['probability']}%** "
        f"(#1 · {top['priority']} priority). Search here first."
    )

    st.write(
        f"**Case:** {row['Case_ID']}  ·  **Person:** {row['Age_Group']} · {row['Gender']}  ·  "
        f"**Day:** {row['Day']}  ·  **Weather:** {row['Weather']}  ·  "
        f"**Usual:** {row['Usual_Area']}  ·  **Previous:** {row['Previous_Area']}"
    )

    st.markdown("#### Ranked predictions")
    for p in predictions:
        with st.container(border=True):
            col_area, col_prob, col_prio = st.columns([2.4, 1, 1.2])
            col_area.markdown(f"**#{p['rank']}** · {p['area']}")
            col_prob.markdown(f"**{p['probability']}%**")
            col_prio.write(p["priority"])
            st.progress(min(p["probability"], 100.0) / 100.0)

    st.markdown("#### Probability chart")
    st.plotly_chart(make_prediction_figure(predictions), width="stretch")

    with st.expander("Full 10-area ranking"):
        st.dataframe(predict_all_areas(model, label_encoder, row), width="stretch", hide_index=True)

    st.download_button(
        "Download predictions (CSV)",
        predictions_to_frame(predictions).to_csv(index=False).encode("utf-8"),
        file_name="casefile_predictions.csv",
        mime="text/csv",
    )

    st.caption(
        "Ethical notice: This output is a statistical probability estimate for search "
        "prioritization only. It does not constitute a definitive prediction or proof of "
        "location. Always cross-reference with field intelligence."
    )


# ---------------------------------------------------------------------------
# Tabs (native)
# ---------------------------------------------------------------------------

tab_dashboard, tab_predictor = st.tabs(["Dashboard", "New Case Prediction"])

# ===========================================================================
# DASHBOARD
# ===========================================================================
with tab_dashboard:
    with st.spinner("Loading performance summary..."):
        _, _, summary = load_artifacts()
    best = next(m for m in summary["models"] if m["model"] == summary["best_model"])

    st.subheader("Model Performance")
    st.caption(
        f"{summary['repeats']} repeats · seeds [{', '.join(map(str, summary['seeds']))}] · "
        f"{summary['n_cases']} cases · {summary['n_features']} features"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Model", best["model"], help="Chosen by top-1 accuracy.")
    m2.metric("Top-1 Accuracy", fmt_pct_std(best["top1_mean"], best["top1_std"]),
              help="Mean ± std across evaluation repeats.")
    m3.metric("Top-3 Accuracy", fmt_pct_std(best["top3_mean"], best["top3_std"]),
              help="Mean ± std across evaluation repeats.")
    m4.metric("Top-5 Accuracy", fmt_pct_std(best["top5_mean"], best["top5_std"]),
              help="Mean ± std across evaluation repeats.")

    st.divider()

    chart_col, radar_col = st.columns(2)

    with chart_col:
        st.subheader("Top-K Accuracy")
        st.caption("Mean accuracy across evaluation repeats")
        st.plotly_chart(make_comparison_figure(summary["models"]), width="stretch")

    with radar_col:
        st.subheader("Model Quality Radar")
        st.caption("All five metrics scaled to %")
        st.plotly_chart(make_radar_figure(summary["models"]), width="stretch")

    st.divider()

    st.subheader("Per-Model Metrics")
    st.caption("Sorted by Top-1 accuracy · Top-1 shown as mean ± std")
    rows = []
    for m in sorted(summary["models"], key=lambda x: x["top1_mean"], reverse=True):
        rows.append({
            "Model": m["model"],
            "Top-1 (%)": fmt_pct_std(m["top1_mean"], m["top1_std"]),
            "Top-3 (%)": fmt_pct(m["top3_mean"]),
            "Top-5 (%)": fmt_pct(m["top5_mean"]),
            "Macro F1 (%)": fmt_pct(m["macro_f1_mean"]),
            "Weighted F1 (%)": fmt_pct(m["weighted_f1_mean"]),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.caption(
        "Ethical notice: All data is 100% synthetic & fictional. Metrics reflect repeated "
        "cross-validation on simulated cases only and are not a measure of real-world field "
        "performance."
    )

# ===========================================================================
# PREDICTOR
# ===========================================================================
with tab_predictor:
    st.subheader("New Case Prediction")
    st.caption("Enter case details below to rank the 10 search areas by predicted likelihood")

    form_col, result_col = st.columns([1, 1.05])

    with form_col:
        with st.form("case-form"):
            st.markdown("**Person**")
            c1, c2 = st.columns(2)
            age_group = c1.selectbox("Age Group", ["18-25", "26-35", "36-45", "46-55", "56+"],
                                     index=1, help="Age band of the missing person.")
            gender = c2.selectbox("Gender", ["Male", "Female", "Other"], index=0)

            st.markdown("**Location**")
            c1, c2 = st.columns(2)
            lat = c1.number_input("Last Latitude", value=23.2500, step=0.0001, format="%.4f",
                                  help="GPS latitude where the person was last seen.")
            lon = c2.number_input("Last Longitude", value=77.4100, step=0.0001, format="%.4f",
                                  help="GPS longitude where the person was last seen.")
            c1, c2 = st.columns(2)
            usual_area = c1.selectbox("Usual Area", AREAS, index=4,
                                      help="Area the person most often visits.")
            previous_area = c2.selectbox("Previous Area", AREAS, index=4,
                                         help="Area from the person's latest known movement.")

            st.markdown("**Time & Conditions**")
            c1, c2 = st.columns(2)
            last_seen_time = c1.text_input("Last Seen Time (HH:MM)", value="14:30",
                                           help="24-hour format, e.g. 14:30.")
            day = c2.selectbox("Day",
                               ["Monday", "Tuesday", "Wednesday", "Thursday",
                                "Friday", "Saturday", "Sunday"], index=4)
            weather = st.selectbox("Weather", ["Clear", "Cloudy", "Rain", "Fog"], index=0)

            st.markdown("**Movement**")
            c1, c2, c3 = st.columns(3)
            avg_distance = c1.number_input("Avg Distance (km)", value=9.0, min_value=0.0, step=0.1,
                                           help="Average distance covered between sightings.")
            avg_speed = c2.number_input("Avg Speed (km/h)", value=20.0, min_value=0.0, step=0.1,
                                        help="Average movement speed of the person.")
            time_since = c3.number_input("Time Since Seen (hrs)", value=4.0, min_value=0.1,
                                         step=0.1, help="Hours elapsed since the last sighting.")

            submitted = st.form_submit_button("Generate Prediction Report", type="primary",
                                              width="stretch")

    with result_col:
        model, label_encoder, _ = load_artifacts()

        if submitted:
            row = build_case_dict(age_group, gender, lat, lon, last_seen_time, day, weather,
                                  usual_area, avg_distance, avg_speed, previous_area, time_since)
            with st.spinner("Computing area probabilities..."):
                predictions = predict_top5(model, label_encoder, row)
            st.session_state["last_prediction"] = {"row": row, "predictions": predictions}
            st.toast("Prediction report generated")

        st.subheader("Probable Location Report")
        st.caption("Top-5 search areas by model likelihood")

        last = st.session_state["last_prediction"]
        if last is None:
            st.info(
                "Fill in the case form and press **Generate Prediction Report** "
                "to see the Top-5 probable areas."
            )
        else:
            render_results(model, label_encoder, last["row"], predictions=last["predictions"])

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption("CASEFILE · Missing Person Location Prediction System · All data is 100% synthetic & fictional")