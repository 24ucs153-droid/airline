"""
app.py
------
Streamlit UI for the Airline Customer Satisfaction Predictor.

Run:
    streamlit run app/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
PLOTS_DIR = PROJECT_ROOT / "models" / "plots"

from predict import load_pipeline, predict_single

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="✈️ Airline Satisfaction Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ---- Main background ---- */
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    /* ---- Hero header ---- */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1.05rem;
        max-width: 650px;
        margin: 0 auto;
    }

    /* ---- Glass cards ---- */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(10px);
    }

    /* ---- Section headers ---- */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a78bfa;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(167, 139, 250, 0.3);
        padding-bottom: 0.4rem;
    }

    /* ---- Predict button ---- */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        font-size: 1.05rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        border: none;
        width: 100%;
        letter-spacing: 0.04em;
        transition: all 0.2s ease;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(124, 58, 237, 0.55);
    }

    /* ---- Result boxes ---- */
    .result-satisfied {
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.15), rgba(16, 185, 129, 0.08));
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-dissatisfied {
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.15), rgba(239, 68, 68, 0.08));
        border: 1px solid rgba(248, 113, 113, 0.4);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-label {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .result-sub {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    /* ---- Metric cards ---- */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #a78bfa;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ---- Slider & input labels ---- */
    .stSlider label, .stSelectbox label, .stNumberInput label {
        color: #cbd5e1 !important;
        font-size: 0.9rem !important;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.85) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model & metrics (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_pipeline():
    return load_pipeline(MODEL_PATH)


@st.cache_data(show_spinner=False)
def get_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>✈️ Airline Satisfaction Predictor</h1>
        <p>
            Predict whether an airline passenger will be <strong>satisfied</strong> or
            <strong>dissatisfied</strong> based on their flight and service experience.
            Powered by a Random Forest classifier trained on 129 880 real passenger records.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
with st.spinner("Loading model …"):
    try:
        pipeline = get_pipeline()
        metrics = get_metrics()
        model_loaded = True
    except FileNotFoundError as e:
        st.error(
            f"⚠️ Model not found.\n\n"
            f"`{e}`\n\n"
            "Please train the model first:\n```bash\n"
            "python src/train.py\n```"
        )
        model_loaded = False

# ---------------------------------------------------------------------------
# Sidebar – about & performance
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 Model Performance")

    if metrics:
        acc = metrics.get("accuracy", 0)
        f1 = metrics.get("f1", 0)
        roc = metrics.get("roc_auc", 0)
        prec = metrics.get("precision", 0)
        rec = metrics.get("recall", 0)

        # Gauge-style accuracy
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=acc * 100,
                number={"suffix": "%", "font": {"size": 28, "color": "#a78bfa"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#475569"},
                    "bar": {"color": "#7c3aed"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "bordercolor": "rgba(255,255,255,0.1)",
                    "steps": [
                        {"range": [0, 70], "color": "rgba(239,68,68,0.15)"},
                        {"range": [70, 90], "color": "rgba(251,191,36,0.15)"},
                        {"range": [90, 100], "color": "rgba(52,211,153,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "#34d399", "width": 3},
                        "thickness": 0.75,
                        "value": acc * 100,
                    },
                },
                title={"text": "Accuracy", "font": {"color": "#94a3b8", "size": 14}},
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=200,
            margin=dict(t=30, b=0, l=20, r=20),
        )
        st.plotly_chart(fig_gauge, width='stretch')

        col1, col2 = st.columns(2)
        col1.metric("F1 Score", f"{f1:.3f}")
        col2.metric("ROC-AUC", f"{roc:.3f}")
        col1.metric("Precision", f"{prec:.3f}")
        col2.metric("Recall", f"{rec:.3f}")

        st.divider()

        train_sz = metrics.get("train_size", "–")
        test_sz = metrics.get("test_size", "–")
        st.markdown(f"🗃️ **Training set:** {train_sz:,}")
        st.markdown(f"🧪 **Test set:** {test_sz:,}")

    else:
        st.info("Metrics not available. Train the model first.")

    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown(
        "This app uses a **scikit-learn Random Forest** pipeline that was trained on the "
        "[Airline Customer Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) "
        "dataset.\n\n"
        "The same preprocessing (imputation, scaling, one-hot encoding) applied during "
        "training is used at inference time, so there is no risk of data leakage."
    )

# ---------------------------------------------------------------------------
# Main input form
# ---------------------------------------------------------------------------
if not model_loaded:
    st.stop()

st.markdown("---")
st.markdown("### 🛫 Enter Passenger & Flight Details")
st.markdown(
    '<div class="section-title">Passenger Information</div>',
    unsafe_allow_html=True,
)

# ---- Helper ---- #
def rating_widget(label: str, key: str, col) -> int:
    return col.select_slider(
        label,
        options=[0, 1, 2, 3, 4, 5],
        value=3,
        key=key,
        help="0 = Not rated / N/A,  1 = Very Poor,  5 = Excellent",
    )


with st.form("prediction_form"):
    # ── Row 1: Passenger basics
    c1, c2, c3, c4 = st.columns([1.5, 1.5, 1.5, 1.5])

    age = c1.slider("Age", min_value=7, max_value=85, value=35, key="age")
    customer_type = c2.selectbox(
        "Customer Type",
        options=["Loyal Customer", "disloyal Customer"],
        key="cust_type",
    )
    travel_type = c3.selectbox(
        "Type of Travel",
        options=["Business travel", "Personal Travel"],
        key="travel_type",
    )
    flight_class = c4.selectbox(
        "Class",
        options=["Business", "Eco Plus", "Eco"],
        key="flight_class",
    )

    # ── Row 2: Flight details
    st.markdown("---")
    st.markdown(
        '<div class="section-title">Flight Details</div>',
        unsafe_allow_html=True,
    )
    c5, c6, c7 = st.columns(3)

    flight_dist = c5.number_input(
        "Flight Distance (miles)", min_value=31, max_value=6951, value=1500, step=10, key="fd"
    )
    dep_delay = c6.number_input(
        "Departure Delay (minutes)", min_value=0, max_value=1600, value=0, step=1, key="dep_delay"
    )
    arr_delay = c7.number_input(
        "Arrival Delay (minutes)", min_value=0, max_value=1600, value=0, step=1, key="arr_delay"
    )

    # ── Row 3: Service ratings
    st.markdown("---")
    st.markdown(
        '<div class="section-title">Service Ratings  (0 = N/A, 1 = Very Poor → 5 = Excellent)</div>',
        unsafe_allow_html=True,
    )

    r1, r2, r3, r4 = st.columns(4)
    seat_comfort = rating_widget("Seat Comfort", "seat_comfort", r1)
    food_drink = rating_widget("Food & Drink", "food_drink", r2)
    inflight_entertainment = rating_widget("Inflight Entertainment", "ife", r3)
    inflight_wifi = rating_widget("Inflight WiFi Service", "wifi", r4)

    r5, r6, r7, r8 = st.columns(4)
    cleanliness = rating_widget("Cleanliness", "clean", r5)
    leg_room = rating_widget("Leg Room Service", "leg", r6)
    onboard_service = rating_widget("On-board Service", "onboard", r7)
    baggage = rating_widget("Baggage Handling", "baggage", r8)

    r9, r10, r11, r12 = st.columns(4)
    checkin = rating_widget("Check-in Service", "checkin", r9)
    online_support = rating_widget("Online Support", "online_sup", r10)
    ease_booking = rating_widget("Ease of Online Booking", "ease_book", r11)
    online_boarding = rating_widget("Online Boarding", "online_board", r12)

    r13, r14 = st.columns([1, 3])
    gate_location = rating_widget("Gate Location", "gate", r13)
    dep_arr_convenient = rating_widget("Departure/Arrival Time Convenient", "dep_arr", r14)

    # ── Submit
    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔮  Predict Satisfaction")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    feature_dict = {
        "Customer Type": customer_type,
        "Age": age,
        "Type of Travel": travel_type,
        "Class": flight_class,
        "Flight Distance": flight_dist,
        "Seat comfort": seat_comfort,
        "Departure/Arrival time convenient": dep_arr_convenient,
        "Food and drink": food_drink,
        "Gate location": gate_location,
        "Inflight wifi service": inflight_wifi,
        "Inflight entertainment": inflight_entertainment,
        "Online support": online_support,
        "Ease of Online booking": ease_booking,
        "On-board service": onboard_service,
        "Leg room service": leg_room,
        "Baggage handling": baggage,
        "Checkin service": checkin,
        "Cleanliness": cleanliness,
        "Online boarding": online_boarding,
        "Departure Delay in Minutes": dep_delay,
        "Arrival Delay in Minutes": float(arr_delay),
    }

    try:
        label, confidence, pos_proba = predict_single(pipeline, feature_dict)

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        res_col, prob_col = st.columns([1, 1])

        # ── Result card
        if label == "satisfied":
            res_col.markdown(
                f"""
                <div class="result-satisfied">
                    <div class="result-label">😊 Satisfied</div>
                    <div class="result-sub">
                        The model predicts this passenger is <strong>likely satisfied</strong>
                        with a confidence of <strong>{confidence*100:.1f}%</strong>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            res_col.markdown(
                f"""
                <div class="result-dissatisfied">
                    <div class="result-label">😞 Dissatisfied</div>
                    <div class="result-sub">
                        The model predicts this passenger is <strong>likely dissatisfied</strong>
                        with a confidence of <strong>{confidence*100:.1f}%</strong>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Probability gauge
        fig_prob = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=pos_proba * 100,
                number={"suffix": "%", "font": {"size": 30, "color": "#a78bfa"}},
                delta={"reference": 50, "valueformat": ".1f"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {
                        "color": "#34d399" if label == "satisfied" else "#f87171"
                    },
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 50], "color": "rgba(248,113,113,0.1)"},
                        {"range": [50, 100], "color": "rgba(52,211,153,0.1)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
                title={
                    "text": "P(Satisfied)",
                    "font": {"color": "#94a3b8", "size": 14},
                },
            )
        )
        fig_prob.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=220,
            margin=dict(t=40, b=0, l=20, r=20),
        )
        prob_col.plotly_chart(fig_prob, width='stretch')

        # ── Interpretation
        st.info(
            "💡 **Interpretation:** "
            "A probability > 50% indicates the passenger is predicted to be satisfied. "
            "The confidence score reflects how certain the model is about the specific "
            "predicted class."
        )

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.exception(e)

# ---------------------------------------------------------------------------
# Feature importance section
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📈 Feature Importance")

fi_path = PLOTS_DIR / "feature_importance.png"
if fi_path.exists():
    col_fi, col_txt = st.columns([2, 1])
    col_fi.image(str(fi_path), width='stretch')
    col_txt.markdown(
        """
        **What drives satisfaction?**

        The chart shows the top features ranked by their contribution
        to the Random Forest model's decisions.

        ⭐ **Inflight WiFi** and **Online Boarding** consistently rank
        as top predictors, followed by seat comfort and inflight entertainment.

        Higher-rated services strongly correlate with satisfied outcomes.
        """
    )
else:
    st.info("Train the model first to see feature importance: `python src/train.py`")

# ---------------------------------------------------------------------------
# Model visualisations (confusion matrix, ROC)
# ---------------------------------------------------------------------------
cm_path = PLOTS_DIR / "confusion_matrix.png"
roc_path = PLOTS_DIR / "roc_curve.png"

if cm_path.exists() or roc_path.exists():
    st.markdown("---")
    st.markdown("### 🔬 Model Evaluation Visualisations")
    v1, v2 = st.columns(2)
    if cm_path.exists():
        v1.image(str(cm_path), caption="Confusion Matrix (Test Set)", width='stretch')
    if roc_path.exists():
        v2.image(str(roc_path), caption="ROC Curve (Test Set)", width='stretch')

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#475569; font-size:0.85rem;'>"
    "Built with ❤️ using scikit-learn & Streamlit • Random Forest Classifier "
    "• Trained on 129 880 airline passenger records"
    "</div>",
    unsafe_allow_html=True,
)
