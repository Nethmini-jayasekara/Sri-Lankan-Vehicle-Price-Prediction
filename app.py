
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import base64
from pathlib import Path
import sklearn.compose._column_transformer as _sklearn_column_transformer


# Models created with scikit-learn 1.6.1 reference this private compatibility
# class, which was removed in newer scikit-learn releases.
if not hasattr(_sklearn_column_transformer, "_RemainderColsList"):
    class _RemainderColsListCompat(list):
        def __init__(self, iterable=(), future_dtype=None):
            super().__init__(iterable)
            self.future_dtype = future_dtype

    _sklearn_column_transformer._RemainderColsList = (
        _RemainderColsListCompat
    )

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Sri Lankan Vehicle Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: "Manrope", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 78% 8%, rgba(74, 131, 255, 0.12), transparent 28rem),
        linear-gradient(145deg, #070e1d 0%, #0b1425 58%, #111c31 100%);
    color: #f1f5f9;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: -18px;
    z-index: 0;
    pointer-events: none;
    background:
        radial-gradient(circle at 22% 82%, rgba(67, 183, 255, 0.10), transparent 22rem),
        radial-gradient(circle at 82% 12%, rgba(189, 100, 255, 0.10), transparent 18rem);
}

.stApp > header,
.stApp > div {
    position: relative;
    z-index: 1;
}

[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] {
    display: none;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.15rem;
    padding-bottom: 2.5rem;
}

.vehicle-hero {
    position: relative;
    height: 240px;
    margin: 0 0 18px;
    overflow: hidden;
    border: 1px solid rgba(164, 195, 240, 0.20);
    border-radius: 22px;
    background: rgba(12, 24, 48, 0.42);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.13),
        0 26px 58px rgba(2, 8, 23, 0.38);
}

.vehicle-hero::after {
    content: "";
    position: absolute;
    inset: 0;
    z-index: 2;
    background:
        linear-gradient(90deg, rgba(7, 16, 36, 0.40) 0%, rgba(7, 16, 36, 0.16) 48%, rgba(7, 16, 36, 0.08) 100%),
        linear-gradient(0deg, rgba(7, 16, 36, 0.44), transparent 68%);
    backdrop-filter: blur(0.5px);
    -webkit-backdrop-filter: blur(0.5px);
}

.vehicle-hero .stImage,
.vehicle-hero img {
    width: 100%;
    height: 240px;
    object-fit: cover;
    object-position: center 52%;
    filter: saturate(1.12) contrast(1.06) brightness(0.92);
}

.vehicle-hero-copy {
    position: absolute;
    z-index: 3;
    left: 28px;
    bottom: 24px;
}

.vehicle-hero-copy h2 {
    margin: 0 0 4px;
    color: #ffffff;
    font-size: 1.65rem;
}

.vehicle-hero-copy p {
    margin: 0;
    color: #bbcbe2;
    font-size: 0.82rem;
}

.block-container {
    background: linear-gradient(
        145deg,
        rgba(14, 28, 55, 0.22),
        rgba(8, 18, 38, 0.12)
    );
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(164, 195, 240, 0.07);
    border-radius: 22px;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.05),
        0 24px 70px rgba(2, 8, 23, 0.22);
}

.brand-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -0.35rem 0 1.15rem;
    padding: 0.55rem 0 0.85rem;
    border-bottom: 1px solid rgba(132, 163, 210, 0.13);
}

.brand-name {
    display: flex;
    align-items: center;
    color: #f8fafc;
    font-size: 1.48rem;
    font-weight: 700;
    letter-spacing: -0.045em;
}

.brand-name .brand-ai {
    margin-left: 7px;
    color: #89a4ff;
    font-size: 0.72em;
    font-weight: 800;
    letter-spacing: 0.10em;
    text-transform: uppercase;
}

.brand-status {
    color: #91a4c4;
    font-size: 0.78rem;
    font-weight: 600;
}

.brand-status span {
    display: inline-block;
    width: 8px;
    height: 8px;
    margin-right: 7px;
    border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 12px rgba(74, 222, 128, 0.75);
}

.st-key-main_navigation {
    background:
        linear-gradient(180deg, rgba(23, 38, 68, 0.72), rgba(12, 24, 48, 0.58));
    border: 1px solid rgba(137, 174, 230, 0.16);
    border-radius: 14px;
    padding: 5px;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 14px 36px rgba(3, 8, 20, 0.18);
}

.st-key-main_navigation [role="radiogroup"] {
    gap: 2px;
}

.st-key-main_navigation [role="radio"] {
    position: relative;
    min-height: 42px;
    border: 1px solid rgba(137, 174, 230, 0.12);
    border-radius: 10px;
    background: rgba(13, 24, 46, 0.76) !important;
    color: #edf4ff !important;
    opacity: 1 !important;
    font-weight: 600;
    letter-spacing: 0.01em;
    transition: all 160ms ease;
}

.st-key-main_navigation [role="radio"]:hover {
    color: #edf4ff;
    background: rgba(112, 149, 214, 0.08);
}

.st-key-main_navigation [role="radio"][aria-checked="true"] {
    border: 1px solid rgba(151, 177, 232, 0.14);
    background: rgba(73, 108, 177, 0.18);
    color: #ffffff !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.07);
}

.st-key-main_navigation [role="radio"][aria-checked="true"]::after {
    content: "";
    position: absolute;
    left: 22%;
    right: 22%;
    bottom: 3px;
    height: 2px;
    border-radius: 999px;
    background: linear-gradient(90deg, #5c86ff, #79b8ff);
}

h1, h2, h3 {
    color: #f8fafc;
    letter-spacing: -0.035em;
}

h1 {
    margin-bottom: 0.2rem;
    font-size: 2rem;
}

p,
label,
.stCaption,
.stMarkdown,
[data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"] {
    color: #e6eef9 !important;
}

.metric-label {
    color: #bac9df !important;
    opacity: 1;
}

.metric-value,
.winner-name,
.winner-score-value,
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    text-shadow: 0 2px 14px rgba(0, 0, 0, 0.28);
}

[data-testid="stMetricLabel"],
.winner-eyebrow,
.winner-category,
.winner-score-label {
    color: #b9c7db !important;
}

.metric-card,
[data-testid="stMetric"],
[data-testid="stDataFrame"],
[data-testid="stDataFrameResizable"] {
    background: linear-gradient(145deg, rgba(24, 38, 67, 0.96), rgba(16, 27, 49, 0.96));
    border: 1px solid rgba(121, 155, 210, 0.14);
    border-radius: 16px;
    box-shadow: 0 18px 44px rgba(2, 8, 23, 0.18);
}

.metric-card {
    position: relative;
    overflow: hidden;
    min-height: 145px;
    padding: 20px 22px 58px;
}

.glass-card {
    background:
        linear-gradient(135deg, rgba(43, 72, 125, 0.48), rgba(18, 30, 54, 0.74)),
        rgba(18, 30, 54, 0.52);
    backdrop-filter: blur(20px) saturate(118%);
    -webkit-backdrop-filter: blur(20px) saturate(118%);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.14),
        inset 0 -1px 0 rgba(128, 163, 215, 0.05),
        0 18px 44px rgba(2, 8, 23, 0.30);
}

.glass-card-blue { border-color: rgba(98, 124, 255, 0.30); }
.glass-card-cyan { border-color: rgba(67, 183, 255, 0.28); }
.glass-card-purple { border-color: rgba(189, 100, 255, 0.28); }
.glass-card-green { border-color: rgba(34, 211, 167, 0.28); }

.metric-icon {
    position: absolute;
    right: 18px;
    top: 17px;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #627cff;
    box-shadow: 0 0 16px rgba(98, 124, 255, 0.85);
}

.glass-card-cyan .metric-icon { background: #43b7ff; box-shadow: 0 0 16px rgba(67,183,255,.85); }
.glass-card-purple .metric-icon { background: #bd64ff; box-shadow: 0 0 16px rgba(189,100,255,.85); }
.glass-card-green .metric-icon { background: #22d3a7; box-shadow: 0 0 16px rgba(34,211,167,.85); }

.mini-vehicle {
    position: absolute;
    right: 18px;
    bottom: 10px;
    width: 82px;
    height: 36px;
    opacity: 0.72;
    filter: drop-shadow(0 8px 8px rgba(3, 8, 20, 0.45));
}

.vehicle-body {
    position: absolute;
    left: 5px;
    bottom: 8px;
    width: 72px;
    height: 20px;
    border-radius: 20px 28px 7px 7px;
    background: linear-gradient(135deg, #7f99d0 0%, #35527e 100%);
    border: 1px solid rgba(196, 215, 255, 0.32);
}

.vehicle-cabin {
    position: absolute;
    left: 23px;
    bottom: 13px;
    width: 34px;
    height: 16px;
    clip-path: polygon(20% 100%, 38% 0, 78% 0, 100% 100%);
    background: linear-gradient(135deg, #afc7e9, #3f5f8c);
    border: 1px solid rgba(220, 235, 255, 0.34);
}

.vehicle-wheel {
    position: absolute;
    bottom: -7px;
    width: 14px;
    height: 14px;
    border: 3px solid #a7b9d6;
    border-radius: 50%;
    background: #10182a;
}

.vehicle-wheel-left { left: 11px; }
.vehicle-wheel-right { right: 9px; }
}

.metric-label {
    color: #97aac7;
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.metric-value {
    color: #ffffff;
    font-size: 1.8rem;
    font-weight: 800;
    margin-top: 8px;
}

div.stButton > button {
    background: linear-gradient(135deg, #3b73ff 0%, #765bf6 100%);
    color: white;
    border-radius: 10px;
    border: none;
    min-height: 46px;
    padding: 10px 24px;
    font-weight: 700;
    box-shadow: 0 12px 28px rgba(66, 98, 230, 0.24);
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #4c80ff 0%, #8469ff 100%);
    color: white;
}

[data-testid="stForm"] {
    background: rgba(17, 28, 50, 0.82);
    border: 1px solid rgba(121, 155, 210, 0.14);
    border-radius: 16px;
    padding: 18px;
}

[data-testid="stMetric"] {
    padding: 16px 18px;
}

[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid rgba(121, 155, 210, 0.12);
}

[data-testid="stDataFrame"] {
    overflow: hidden;
    border: 1px solid rgba(99, 145, 235, 0.32);
    border-radius: 16px;
    background: linear-gradient(145deg, rgba(26, 48, 88, 0.96), rgba(14, 31, 65, 0.96));
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow:
        inset 0 1px 0 rgba(173, 204, 255, 0.12),
        0 18px 42px rgba(2, 8, 23, 0.28);
}

[data-testid="stDataFrame"] table {
    font-size: 0.82rem;
    color: #eef5ff;
}

[data-testid="stDataFrame"] thead tr th {
    background: linear-gradient(135deg, rgba(54, 91, 154, 0.98), rgba(38, 69, 125, 0.98)) !important;
    color: #f7faff !important;
    font-size: 0.78rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.025em;
    border-bottom: 1px solid rgba(139, 174, 230, 0.22) !important;
}

[data-testid="stDataFrame"] tbody tr:nth-child(even) {
    background: rgba(63, 101, 160, 0.18);
}

[data-testid="stDataFrame"] tbody tr:hover {
    background: rgba(83, 133, 210, 0.28) !important;
}

[data-testid="stMetric"] {
    overflow: hidden;
    border: 1px solid rgba(137, 174, 230, 0.16);
    border-radius: 16px;
    background:
        linear-gradient(135deg, rgba(33, 57, 99, 0.55), rgba(13, 25, 48, 0.76));
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
}

.winner-banner {
    position: relative;
    overflow: hidden;
    display: grid;
    grid-template-columns: 72px 1fr auto;
    align-items: center;
    gap: 16px;
    min-height: 112px;
    margin-top: 8px;
    padding: 18px 22px;
    border: 1px solid rgba(34, 211, 167, 0.28);
    border-radius: 18px;
    background:
        radial-gradient(circle at 92% 20%, rgba(189, 100, 255, 0.16), transparent 180px),
        linear-gradient(135deg, rgba(24, 49, 78, 0.92), rgba(16, 30, 52, 0.94));
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.09),
        0 18px 44px rgba(2, 8, 23, 0.24);
}

.winner-banner::after {
    content: "";
    position: absolute;
    width: 150px;
    height: 150px;
    right: -80px;
    bottom: -95px;
    border-radius: 50%;
    background: rgba(34, 211, 167, 0.08);
    border: 1px solid rgba(34, 211, 167, 0.12);
}

.winner-badge {
    display: grid;
    place-items: center;
    width: 62px;
    height: 62px;
    border-radius: 16px;
    background: linear-gradient(135deg, #22d3a7, #627cff 72%, #bd64ff);
    box-shadow:
        0 12px 28px rgba(34, 211, 167, 0.18),
        inset 0 1px 0 rgba(255, 255, 255, 0.28);
    color: white;
    font-size: 28px;
}

.winner-eyebrow {
    margin-bottom: 4px;
    color: #91a4c4;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.10em;
    text-transform: uppercase;
}

.winner-name {
    color: #ffffff;
    font-size: 1.32rem;
    font-weight: 800;
    letter-spacing: -0.025em;
}

.winner-category {
    margin-top: 5px;
    color: #b8c6d9;
    font-size: 0.82rem;
}

.winner-score {
    position: relative;
    z-index: 1;
    min-width: 128px;
    text-align: right;
}

.winner-score-label {
    color: #91a4c4;
    font-size: 0.72rem;
    font-weight: 700;
}

.winner-score-value {
    margin-top: 2px;
    color: #6ee7c4;
    font-size: 1.55rem;
    font-weight: 800;
}

.js-plotly-plot,
.plot-container {
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# FILE PATHS
# ==========================================

DATA_PATH = Path("dataset/car_price_dataset.csv")
RESULT_PATH = Path("final_ensemble_comparison.csv")
MODEL_DIR = Path("models")

MODEL_FILES = {
    "Random Forest": "random_forest_vehicle_price.pkl",
    "XGBoost": "xgboost_vehicle_price.pkl",
    "CatBoost": "catboost_vehicle_price.pkl",
    "Gradient Boosting": "gradient_boosting_vehicle_price.pkl",
    "Stacking": "stacking_vehicle_price.pkl",
    "RF + ANN Hybrid": "hybrid_rf_ann_vehicle_price.pkl"
}

NUMERICAL_FEATURES = [
    "YOM",
    "Engine (cc)",
    "Millage(KM)",
    "Vehicle_Age"
]

CATEGORICAL_FEATURES = [
    "Brand",
    "Model",
    "Gear",
    "Fuel Type",
    "Town",
    "Leasing",
    "Condition",
    "AIR CONDITION",
    "POWER STEERING",
    "POWER MIRROR",
    "POWER WINDOW"
]

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_dataset():
    data = pd.read_csv(DATA_PATH)

    data = data.drop(
        columns=["Unnamed: 0"],
        errors="ignore"
    )

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    return data


@st.cache_data
def load_comparison():
    return pd.read_csv(RESULT_PATH)


@st.cache_resource
def load_model(filename):
    try:
        return joblib.load(MODEL_DIR / filename)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model '{filename}'. "
            "Use model files retrained with the installed ML package "
            f"versions. Original error: {exc}"
        ) from exc


# Check required files
if not DATA_PATH.exists():
    st.error("Dataset not found. Check dataset/car_price_dataset.csv")
    st.stop()

if not RESULT_PATH.exists():
    st.error("Model comparison CSV not found.")
    st.stop()

df = load_dataset()
comparison = load_comparison()

car_image = Path("static/car.jpg")
if car_image.exists():
    car_image_b64 = base64.b64encode(
        car_image.read_bytes()
    ).decode("ascii")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(
                    115deg,
                    rgba(5, 12, 29, 0.78) 0%,
                    rgba(7, 18, 40, 0.66) 52%,
                    rgba(8, 19, 42, 0.48) 100%
                ),
                url("data:image/jpeg;base64,{car_image_b64}") center 52% / cover no-repeat fixed,
                #09101f !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# APP HEADER
# ==========================================

st.markdown("""
<div class="brand-bar">
    <div class="brand-name">
        <span>AutoLens</span><span class="brand-ai">Intelligence</span>
    </div>
    <div class="brand-status"><span></span>Models operational</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# TOP NAVIGATION
# ==========================================

page = st.segmented_control(
    "Navigation",
    [
        "Overview",
        "Dataset Exploration",
        "Vehicle Price Prediction",
        "Model Comparison",
        "Feature Importance"
    ],
    selection_mode="single",
    default="Overview",
    label_visibility="collapsed",
    key="main_navigation"
)

# ==========================================
# HELPER FUNCTION
# ==========================================

def metric_card(label, value, icon="directions_car", accent="blue"):
    st.markdown(
        f"""
        <div class="metric-card glass-card glass-card-{accent}">
            <div class="metric-icon" aria-hidden="true"></div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="mini-vehicle" aria-hidden="true">
                <div class="vehicle-body">
                    <div class="vehicle-cabin"></div>
                    <div class="vehicle-wheel vehicle-wheel-left"></div>
                    <div class="vehicle-wheel vehicle-wheel-right"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def style_dashboard_figure(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(17,28,50,0)",
        plot_bgcolor="#111c32",
        hovermode="x unified",
        bargap=0.22,
        autosize=True,
        font={
            "family": "Manrope, Arial, sans-serif",
            "color": "#d8e3f3"
        },
        title_font={
            "size": 16,
            "color": "#f8fafc"
        },
        margin={
            "l": 55,
            "r": 24,
            "t": 62,
            "b": 48
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1
        }
    )
    fig.update_xaxes(
        gridcolor="rgba(132,163,210,0.10)",
        zeroline=False
    )
    fig.update_yaxes(
        gridcolor="rgba(132,163,210,0.10)",
        zeroline=False
    )
    for trace in fig.data:
        trace.opacity = 0.92
        if hasattr(trace, "marker"):
            trace.marker.line = {"width": 0}
    return fig


def format_feature_name(feature_name):
    name = str(feature_name)

    if name.startswith(("num__", "cat__")):
        name = name.split("__", 1)[1]

    friendly_features = sorted(
        CATEGORICAL_FEATURES,
        key=len,
        reverse=True
    )

    for feature in friendly_features:
        prefix = f"{feature}_"
        if name.startswith(prefix):
            category_value = name[len(prefix):].replace("_", " ")
            return f"{feature}: {category_value}"

    return name.replace("_", " ")


# ==========================================
# PAGE 1: OVERVIEW
# ==========================================

if page == "Overview":

    st.markdown("## Vehicle Intelligence")
    st.caption(
        "Sri Lankan vehicle market overview and ensemble price analytics"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Total Vehicles",
            f"{len(df):,}",
            "directions_car",
            "blue"
        )

    with col2:
        metric_card(
            "Vehicle Brands",
            f"{df['Brand'].nunique():,}",
            "garage",
            "cyan"
        )

    with col3:
        metric_card(
            "Vehicle Models",
            f"{df['Model'].nunique():,}",
            "commute",
            "purple"
        )

    with col4:
        best_model = comparison.loc[
            comparison["RMSE"].idxmin()
        ]
        metric_card(
            "Best ML Model",
            str(best_model["Model"]),
            "workspace_premium",
            "green"
        )

    st.write("")
    st.subheader("Vehicle Price Distribution")

    fig = px.histogram(
        df,
        x="Price",
        nbins=40,
        title="Distribution of Vehicle Prices"
    )

    style_dashboard_figure(fig)

    st.plotly_chart(
        fig,
        key="overview_price_distribution",
        width="stretch",
        alt="Distribution of vehicle prices"
    )

    col1, col2 = st.columns(2)

    with col1:
        brand_data = (
            df["Brand"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        brand_data.columns = ["Brand", "Count"]

        fig = px.bar(
            brand_data,
            x="Brand",
            y="Count",
            title="Top 10 Vehicle Brands"
        )

        style_dashboard_figure(fig)
        st.plotly_chart(
            fig,
            key="overview_top_brands",
            width="stretch",
            alt="Top 10 vehicle brands"
        )

    with col2:
        fig = px.scatter(
            df,
            x="YOM",
            y="Price",
            color="Gear",
            title="Manufacturing Year vs Price"
        )

        style_dashboard_figure(fig)
        st.plotly_chart(
            fig,
            key="overview_year_vs_price",
            width="stretch",
            alt="Manufacturing year versus vehicle price"
        )


# ==========================================
# PAGE 2: DATASET EXPLORATION
# ==========================================

elif page == "Dataset Exploration":

    st.title("Dataset Exploration")

    st.write("Explore the Sri Lankan vehicle dataset.")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(100),
        width="stretch",
        height=420,
        hide_index=True
    )

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", len(df))
        st.metric("Columns", len(df.columns))

    with col2:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

        st.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

    st.subheader("Vehicle Statistics")

    st.dataframe(
        df.describe(include="all").T,
        width="stretch",
        height=420,
        hide_index=False
    )

    st.subheader("Filter Vehicles")

    brands = ["All"] + sorted(
        df["Brand"].dropna().unique().tolist()
    )

    selected_brand = st.selectbox(
        "Select Vehicle Brand",
        brands
    )

    filtered = df.copy()

    if selected_brand != "All":
        filtered = filtered[
            filtered["Brand"] == selected_brand
        ]

    st.dataframe(
        filtered.head(100),
        width="stretch",
        height=420,
        hide_index=True
    )


# ==========================================
# PAGE 3: PRICE PREDICTION
# ==========================================

elif page == "Vehicle Price Prediction":

    st.title("Vehicle Price Prediction")

    st.write(
        "Enter vehicle details to estimate its price "
        "using a trained machine learning model."
    )

    col1, col2 = st.columns(2)

    with col1:

        brand = st.selectbox(
            "Vehicle Brand",
            sorted(df["Brand"].dropna().unique())
        )

        model_options = sorted(
            df.loc[
                df["Brand"] == brand,
                "Model"
            ].dropna().unique()
        )

        vehicle_model = st.selectbox(
            "Vehicle Model",
            model_options
        )

        yom = st.number_input(
            "Year of Manufacture",
            min_value=int(df["YOM"].min()),
            max_value=int(df["YOM"].max()),
            value=int(df["YOM"].median()),
            step=1
        )

        engine = st.number_input(
            "Engine Capacity (cc)",
            min_value=1.0,
            max_value=float(df["Engine (cc)"].max()),
            value=float(df["Engine (cc)"].median())
        )

        mileage = st.number_input(
            "Mileage (KM)",
            min_value=0.0,
            max_value=float(df["Millage(KM)"].max()),
            value=float(df["Millage(KM)"].median())
        )

        gear = st.selectbox(
            "Transmission",
            sorted(df["Gear"].dropna().unique())
        )

    with col2:

        fuel = st.selectbox(
            "Fuel Type",
            sorted(df["Fuel Type"].dropna().unique())
        )

        town = st.selectbox(
            "Town",
            sorted(df["Town"].dropna().unique())
        )

        leasing = st.selectbox(
            "Leasing",
            sorted(df["Leasing"].dropna().unique())
        )

        condition = st.selectbox(
            "Vehicle Condition",
            sorted(df["Condition"].dropna().unique())
        )

        ac = st.selectbox(
            "Air Condition",
            sorted(df["AIR CONDITION"].dropna().unique())
        )

        steering = st.selectbox(
            "Power Steering",
            sorted(df["POWER STEERING"].dropna().unique())
        )

        mirror = st.selectbox(
            "Power Mirror",
            sorted(df["POWER MIRROR"].dropna().unique())
        )

        window = st.selectbox(
            "Power Window",
            sorted(df["POWER WINDOW"].dropna().unique())
        )

    selected_model = st.selectbox(
        "Choose Prediction Model",
        list(MODEL_FILES.keys())
    )

    # Use the latest advertisement year in the dataset
    valuation_year = int(
        df["Date"].dt.year.dropna().max()
    )

    vehicle_age = valuation_year - int(yom)

    if st.button(
        "Predict Vehicle Price",
        width="stretch"
    ):

        if vehicle_age < 0:
            st.error("Manufacturing year is after the valuation year.")

        else:

            input_data = pd.DataFrame([{
                "YOM": int(yom),
                "Engine (cc)": float(engine),
                "Millage(KM)": float(mileage),
                "Vehicle_Age": int(vehicle_age),
                "Brand": brand,
                "Model": vehicle_model,
                "Gear": gear,
                "Fuel Type": fuel,
                "Town": town,
                "Leasing": leasing,
                "Condition": condition,
                "AIR CONDITION": ac,
                "POWER STEERING": steering,
                "POWER MIRROR": mirror,
                "POWER WINDOW": window
            }])

            model_path = (
                MODEL_DIR / MODEL_FILES[selected_model]
            )

            if not model_path.exists():

                st.error(
                    f"Model file not found: {model_path}"
                )

            else:

                try:
                    trained_model = load_model(
                        MODEL_FILES[selected_model]
                    )

                    if selected_model == "RF + ANN Hybrid":

                        rf_prediction = (
                            trained_model["random_forest"]
                            .predict(input_data)[0]
                        )

                        ann_prediction = (
                            trained_model["ann"]
                            .predict(input_data)[0]
                        )

                        prediction = (
                            rf_prediction + ann_prediction
                        ) / 2

                    else:

                        prediction = trained_model.predict(
                            input_data
                        )[0]

                    st.success(
                        "Vehicle price prediction completed!"
                    )

                    st.metric(
                        "Estimated Vehicle Price",
                        f"{prediction:,.2f} dataset price units"
                    )

                    st.caption(
                        "The displayed value uses the original "
                        "dataset's price units. Confirm the unit "
                        "before interpreting it as LKR."
                    )

                except Exception as e:
                    st.error(f"Prediction error: {e}")


# ==========================================
# PAGE 4: MODEL COMPARISON
# ==========================================

elif page == "Model Comparison":

    st.title("Ensemble Model Performance")

    st.write(
        "Comparison of Bagging, Boosting, Stacking "
        "and Hybrid learning algorithms."
    )

    st.dataframe(
        comparison.sort_values("RMSE"),
        width="stretch",
        height=278,
        hide_index=True,
        column_config={
            "Model": st.column_config.TextColumn(
                "Model",
                width="large"
            ),
            "Category": st.column_config.TextColumn(
                "Learning Type",
                width="medium"
            ),
            "MAE": st.column_config.NumberColumn(
                "MAE",
                format="%.2f"
            ),
            "RMSE": st.column_config.NumberColumn(
                "RMSE",
                format="%.2f"
            ),
            "R2 Score": st.column_config.NumberColumn(
                "R2 Score",
                format="%.4f"
            )
        }
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            comparison,
            x="Model",
            y="RMSE",
            color="Category",
            title="RMSE Comparison"
        )

        style_dashboard_figure(fig)
        st.plotly_chart(
            fig,
            key="model_comparison_rmse",
            width="stretch",
            alt="RMSE comparison across machine learning models"
        )

    with col2:

        fig = px.bar(
            comparison,
            x="Model",
            y="R2 Score",
            color="Category",
            title="R2 Score Comparison"
        )

        style_dashboard_figure(fig)
        st.plotly_chart(
            fig,
            key="model_comparison_r2",
            width="stretch",
            alt="R-squared comparison across machine learning models"
        )

    fig = px.bar(
        comparison,
        x="Model",
        y="MAE",
        color="Category",
        title="MAE Comparison"
    )

    style_dashboard_figure(fig)
    st.plotly_chart(
        fig,
        key="model_comparison_mae",
        width="stretch",
        alt="MAE comparison across machine learning models"
    )

    best_model = comparison.loc[
        comparison["RMSE"].idxmin()
    ]

    st.markdown(
        f"""
        <div class="winner-banner">
            <div class="winner-badge" aria-hidden="true">&#9733;</div>
            <div>
                <div class="winner-eyebrow">Best model by RMSE</div>
                <div class="winner-name">{best_model['Model']}</div>
                <div class="winner-category">{best_model['Category']} ensemble model</div>
            </div>
            <div class="winner-score">
                <div class="winner-score-label">Lowest RMSE</div>
                <div class="winner-score-value">{best_model['RMSE']:.2f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# PAGE 5: FEATURE IMPORTANCE
# ==========================================

elif page == "Feature Importance":

    st.title("Feature Importance Analysis")

    st.write(
        "Explore which vehicle characteristics "
        "contribute most to Random Forest predictions."
    )

    model_path = MODEL_DIR / MODEL_FILES["Random Forest"]

    if not model_path.exists():

        st.warning(
            "Upload the Random Forest model file "
            "to view feature importance."
        )

    else:
        try:
            rf = load_model(
                MODEL_FILES["Random Forest"]
            )

            feature_names = (
                rf.named_steps["preprocessor"]
                .get_feature_names_out()
            )

            importances = (
                rf.named_steps["model"]
                .feature_importances_
            )

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            })

            importance_df["Feature"] = importance_df["Feature"].map(
                format_feature_name
            )

            importance_df = importance_df.sort_values(
                "Importance",
                ascending=False
            ).head(15)

            fig = px.bar(
                importance_df.sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 15 Random Forest Features",
                labels={
                    "Importance": "Relative Importance",
                    "Feature": "Vehicle Feature"
                }
            )

            style_dashboard_figure(fig)

            st.plotly_chart(
                fig,
                key="feature_importance",
                width="stretch",
                alt="Top 15 feature importances"
            )
        except Exception as exc:
            st.error(str(exc))


# ==========================================
# FOOTER
# ==========================================

st.caption(
    "Sri Lankan Vehicle Price Prediction | "
    "Ensemble Learning Project"
)

