import streamlit as st
import pandas as pd
import pickle
import os
import xgboost as xgb


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bengaluru House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "house_price_model.json"
)

COLUMNS_PATH = os.path.join(
    BASE_DIR,
    "model",
    "model_columns.pkl"
)

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "house.jpg"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #f6f8fc;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}


/* ================= HEADER ================= */

.hero {
    background: linear-gradient(
        135deg,
        #0f4c81 0%,
        #1769aa 50%,
        #2b83c6 100%
    );

    padding: 42px 45px;
    border-radius: 24px;
    color: white;
    margin-bottom: 25px;

    box-shadow: 0 10px 30px rgba(15, 76, 129, 0.18);
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 18px;
    opacity: 0.92;
}

.hero-badge {
    display: inline-block;
    margin-top: 18px;
    padding: 8px 16px;
    border-radius: 20px;
    background: rgba(255,255,255,0.15);
    font-size: 14px;
}


/* ================= SECTION ================= */

.section-title {
    font-size: 25px;
    font-weight: 750;
    color: #172b4d;
    margin: 15px 0 18px 0;
}


/* ================= CARD ================= */

.card {
    background: white;
    padding: 28px;
    border-radius: 20px;
    border: 1px solid #e6eaf0;

    box-shadow: 0 6px 22px rgba(20, 35, 60, 0.06);
}


/* ================= PRICE CARD ================= */

.price-card {
    background: linear-gradient(
        145deg,
        #ffffff,
        #f0f8ff
    );

    border: 1px solid #cfe2f5;
    border-radius: 22px;

    padding: 35px;

    text-align: center;

    min-height: 365px;

    display: flex;
    flex-direction: column;
    justify-content: center;

    box-shadow: 0 8px 28px rgba(20, 80, 130, 0.08);
}

.price-label {
    color: #1769aa;
    font-size: 18px;
    font-weight: 700;
}

.price-value {
    color: #0f4c81;
    font-size: 48px;
    font-weight: 850;
    margin: 18px 0 5px 0;
}

.price-crore {
    color: #64748b;
    font-size: 17px;
}

.price-sqft {
    margin-top: 25px;

    background: white;

    padding: 12px 18px;

    border-radius: 12px;

    display: inline-block;

    color: #334155;

    font-weight: 650;
}


/* ================= SUMMARY ================= */

.summary-card {
    background: white;

    border: 1px solid #e6eaf0;

    border-radius: 16px;

    padding: 18px 10px;

    text-align: center;

    min-height: 110px;

    box-shadow: 0 4px 15px rgba(20, 35, 60, 0.04);
}

.summary-icon {
    font-size: 22px;
}

.summary-label {
    font-size: 12px;
    color: #64748b;
    margin-top: 5px;
}

.summary-value {
    font-size: 17px;
    color: #172b4d;
    font-weight: 750;
    margin-top: 3px;
}


/* ================= PERFORMANCE ================= */

.performance-card {
    background: white;

    border-radius: 18px;

    padding: 22px;

    border: 1px solid #e6eaf0;

    text-align: center;

    min-height: 120px;

    box-shadow: 0 5px 18px rgba(20,35,60,0.05);
}

.metric-value {
    font-size: 27px;
    font-weight: 800;
    color: #1769aa;
}

.metric-label {
    color: #64748b;
    font-size: 14px;
    margin-top: 4px;
}


/* ================= MODEL INFO ================= */

.model-info {
    background: #eef6fd;

    border: 1px solid #d3e6f7;

    border-radius: 18px;

    padding: 25px;

    color: #334155;
}

.model-heading {
    color: #0f4c81;

    font-size: 21px;

    font-weight: 750;

    margin-bottom: 12px;
}


/* ================= BUTTON ================= */

.stButton > button {
    width: 100%;

    height: 52px;

    border-radius: 12px;

    border: none;

    background: linear-gradient(
        135deg,
        #1769aa,
        #0f4c81
    );

    color: white;

    font-size: 17px;

    font-weight: 750;
}

.stButton > button:hover {
    background: #0f4c81;
    color: white;
}


/* ================= INPUT ================= */

div[data-baseweb="select"] > div {
    border-radius: 10px;
}


/* ================= FOOTER ================= */

.footer {
    text-align: center;

    color: #64748b;

    font-size: 14px;

    padding: 30px 0 10px 0;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CHECK MODEL FILES
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "❌ Model file not found.\n\n"
        "Expected location: `model/house_price_model.json`"
    )

    st.stop()


if not os.path.exists(COLUMNS_PATH):

    st.error(
        "❌ Model columns file not found.\n\n"
        "Expected location: `model/model_columns.pkl`"
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = xgb.XGBRegressor()

    model.load_model(MODEL_PATH)

    return model


# ============================================================
# LOAD MODEL COLUMNS
# ============================================================

@st.cache_data
def load_columns():

    with open(COLUMNS_PATH, "rb") as file:

        return pickle.load(file)


# ============================================================
# LOAD MODEL + COLUMNS
# ============================================================

try:

    model = load_model()

    model_columns = load_columns()

except Exception as e:

    st.error(
        f"❌ Error loading model: {e}"
    )

    st.stop()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

# Final XGBoost model evaluation

R2_SCORE = 0.720262
MAE = 12.770682
RMSE = 18.074255


# ============================================================
# GET LOCATIONS
# ============================================================

locations = sorted(
    [
        col.replace("location_", "")
        for col in model_columns
        if col.startswith("location_")
    ]
)


# ============================================================
# CHECK LOCATIONS
# ============================================================

if len(locations) == 0:

    st.error(
        "❌ No location columns were found in model_columns.pkl."
    )

    st.stop()


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🏠 Bengaluru House Price Prediction
</div>

<div class="hero-subtitle">
Estimate the market value of a Bengaluru property using
Machine Learning.
</div>

<div class="hero-badge">
🤖 Powered by XGBoost Regression
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# HOUSE IMAGE
# ============================================================

if os.path.exists(IMAGE_PATH):

    st.image(
        IMAGE_PATH,
        use_container_width=True
    )


# ============================================================
# PROPERTY DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">📋 Property Details</div>',
    unsafe_allow_html=True
)


input_col, result_col = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# INPUT SECTION
# ============================================================

with input_col:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )


    # Location

    location = st.selectbox(
        "📍 Location",
        locations
    )


    # Area

    sqft = st.number_input(
        "📐 Area (Sq. Ft.)",

        min_value=300.0,

        max_value=10000.0,

        value=1200.0,

        step=50.0
    )


    # BHK + Bathroom

    c1, c2 = st.columns(2)


    with c1:

        bhk = st.selectbox(
            "🛏️ BHK",
            [1, 2, 3, 4, 5],
            index=2
        )


    with c2:

        bath = st.selectbox(
            "🚿 Bathrooms",
            [1, 2, 3, 4, 5],
            index=2
        )


    # Balcony

    balcony = st.selectbox(
        "🌅 Balcony",
        [0, 1, 2, 3],
        index=2
    )


    st.markdown("<br>", unsafe_allow_html=True)


    # Prediction button

    predict_button = st.button(
        "🔮 Predict House Price"
    )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # Create dataframe with exact model columns
        # ----------------------------------------------------

        data = pd.DataFrame(
            0,
            index=[0],
            columns=model_columns
        )


        # ----------------------------------------------------
        # AREA
        # ----------------------------------------------------

        if "sqft" in data.columns:

            data["sqft"] = float(sqft)

        elif "Area" in data.columns:

            data["Area"] = float(sqft)

        else:

            st.error(
                "❌ Area/sqft column not found in model."
            )

            st.stop()


        # ----------------------------------------------------
        # BHK
        # ----------------------------------------------------

        if "bhk" in data.columns:

            data["bhk"] = float(bhk)


        # ----------------------------------------------------
        # BATHROOM
        # ----------------------------------------------------

        if "bath" in data.columns:

            data["bath"] = float(bath)


        # ----------------------------------------------------
        # BALCONY
        # ----------------------------------------------------

        if "balcony" in data.columns:

            data["balcony"] = float(balcony)


        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------

        location_column = "location_" + location


        if location_column in data.columns:

            data[location_column] = 1

        else:

            st.warning(
                "⚠️ Selected location was not found in the model."
            )


        # ----------------------------------------------------
        # EXACT FEATURE ORDER
        # ----------------------------------------------------

        data = data.reindex(
            columns=model_columns,
            fill_value=0
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = float(
            model.predict(data)[0]
        )


        # ----------------------------------------------------
        # PREVENT NEGATIVE PRICE
        # ----------------------------------------------------

        prediction = max(
            0,
            prediction
        )


        # ----------------------------------------------------
        # PRICE PER SQ FT
        # ----------------------------------------------------

        price_per_sqft = (
            prediction * 100000
        ) / sqft


        # ----------------------------------------------------
        # CRORE
        # ----------------------------------------------------

        crore = prediction / 100


        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        st.session_state["prediction"] = prediction

        st.session_state["price_per_sqft"] = price_per_sqft

        st.session_state["crore"] = crore

        st.session_state["location"] = location

        st.session_state["sqft"] = sqft

        st.session_state["bhk"] = bhk

        st.session_state["bath"] = bath

        st.session_state["balcony"] = balcony


    except Exception as e:

        st.error(
            f"❌ Prediction error: {e}"
        )


# ============================================================
# RESULT
# ============================================================

with result_col:

    if "prediction" in st.session_state:

        prediction = st.session_state["prediction"]

        price_per_sqft = st.session_state["price_per_sqft"]

        crore = st.session_state["crore"]


        st.markdown(f"""
        <div class="price-card">

        <div class="price-label">
        🏠 Estimated House Price
        </div>

        <div class="price-value">
        ₹ {prediction:,.2f} Lakhs
        </div>

        <div class="price-crore">
        Approximately ₹ {crore:.2f} Crores
        </div>

        <div class="price-sqft">
        💰 ₹ {price_per_sqft:,.0f} / Sq. Ft.
        </div>

        <p style="
        color:#64748b;
        margin-top:20px;
        font-size:14px;
        ">
        Estimated using the selected property characteristics.
        </p>

        </div>
        """, unsafe_allow_html=True)


    else:

        st.markdown("""
        <div class="price-card">

        <div class="price-label">
        🏠 Estimated House Price
        </div>

        <div style="
        font-size:25px;
        font-weight:700;
        color:#64748b;
        margin-top:35px;
        ">
        Your prediction will appear here
        </div>

        <p style="
        color:#94a3b8;
        margin-top:10px;
        ">
        Enter the property details and click<br>
        <b>Predict House Price</b>
        </p>

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PROPERTY SUMMARY
# ============================================================

if "prediction" in st.session_state:

    st.markdown(
        '<div class="section-title">📋 Property Summary</div>',
        unsafe_allow_html=True
    )


    summary_data = [

        (
            "📍",
            "Location",
            st.session_state["location"]
        ),

        (
            "📐",
            "Area",
            f'{st.session_state["sqft"]:,.0f} Sq. Ft.'
        ),

        (
            "🛏️",
            "BHK",
            str(st.session_state["bhk"])
        ),

        (
            "🚿",
            "Bathrooms",
            str(st.session_state["bath"])
        ),

        (
            "🌅",
            "Balcony",
            str(st.session_state["balcony"])
        ),

        (
            "💰",
            "Price / Sq. Ft.",
            f'₹ {st.session_state["price_per_sqft"]:,.0f}'
        )

    ]


    cols = st.columns(6)


    for col, item in zip(
        cols,
        summary_data
    ):

        icon, label, value = item


        with col:

            st.markdown(f"""
            <div class="summary-card">

            <div class="summary-icon">
            {icon}
            </div>

            <div class="summary-label">
            {label}
            </div>

            <div class="summary-value">
            {value}
            </div>

            </div>
            """, unsafe_allow_html=True)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">📊 Model Performance</div>',
    unsafe_allow_html=True
)


metrics = [

    (
        "R² Score",
        f"{R2_SCORE:.6f}"
    ),

    (
        "MAE",
        f"{MAE:.6f}"
    ),

    (
        "RMSE",
        f"{RMSE:.6f}"
    ),

    (
        "Algorithm",
        "XGBoost"
    )

]


metric_cols = st.columns(4)


for col, metric in zip(
    metric_cols,
    metrics
):

    label, value = metric


    with col:

        st.markdown(f"""
        <div class="performance-card">

        <div class="metric-value">
        {value}
        </div>

        <div class="metric-label">
        {label}
        </div>

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">🤖 Model Information</div>',
    unsafe_allow_html=True
)


st.markdown(f"""
<div class="model-info">

<div class="model-heading">
XGBoost Regression Model
</div>

<b>Algorithm:</b> XGBoost Regressor
&nbsp;&nbsp; | &nbsp;&nbsp;

<b>Features:</b> {len(model_columns)}
&nbsp;&nbsp; | &nbsp;&nbsp;

<b>Dataset:</b> Bengaluru Housing Data

<br><br>

The model estimates house prices using property
characteristics such as location, area, BHK,
bathrooms and balcony.

<br><br>

<b>Model Performance:</b>

R² = {R2_SCORE:.6f}
&nbsp;&nbsp; | &nbsp;&nbsp;

MAE = {MAE:.6f}
&nbsp;&nbsp; | &nbsp;&nbsp;

RMSE = {RMSE:.6f}

</div>
""", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

🏠 Bengaluru House Price Prediction

<br>

Machine Learning Project • XGBoost Regression

</div>
""", unsafe_allow_html=True)