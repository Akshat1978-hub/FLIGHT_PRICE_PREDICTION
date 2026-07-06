import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Flight Price Prediction",
    page_icon="✈️",
    layout="wide"
)

# -------------------------
# CUSTOM CSS
# -------------------------

st.markdown("""
<style>

.main {
    background: linear-gradient(
    135deg,
    #0f172a,
    #1e293b,
    #334155
    );
}

h1,h2,h3 {
    color:white;
}

.metric-card{
    background:#1e293b;
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# FILE PATHS
# -------------------------

DATASET_PATH = "Clean_Dataset.csv"
MODEL_PATH = "model.pkl"

# -------------------------
# LOAD DATA
# -------------------------

@st.cache_data
def load_data():

    if not os.path.exists(DATASET_PATH):
        return None

    df = pd.read_csv(DATASET_PATH)

    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    return df

# -------------------------
# LOAD MODEL
# -------------------------

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

# -------------------------
# SIDEBAR
# -------------------------

st.sidebar.title("✈️ Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "🏠 Home",
        "✈️ Prediction",
        "📊 Dashboard",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
Flight Price Prediction

Machine Learning Project
"""
)

st.sidebar.markdown(
"""
### Developer

Akshat
"""
)

# -------------------------
# HOME PAGE
# -------------------------

if page == "🏠 Home":

    st.title("✈️ Flight Price Prediction")

    st.write(
        """
        Predict flight ticket prices using Machine Learning.
        """
    )

    if df is None:
        st.error("Dataset not found.")
        st.stop()

    rows, cols = df.shape

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", rows)
    c2.metric("Columns", cols)
    c3.metric(
        "Missing Values",
        int(df.isna().sum().sum())
    )

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Statistics")
    st.dataframe(df.describe())

# -------------------------
# PREDICTION PAGE
# -------------------------

elif page == "✈️ Prediction":

    st.title("✈️ Flight Price Prediction")

    if model is None:
        st.error(
            "model.pkl not found. Run train.py first."
        )
        st.stop()

    if df is None:
        st.error(
            "Dataset not found."
        )
        st.stop()

    features = df.drop(columns=["price"])

    input_data = {}

    for col in features.columns:

        if features[col].dtype == "object":

            input_data[col] = st.selectbox(
                col,
                sorted(features[col].unique())
            )

        else:

            input_data[col] = st.number_input(
                col,
                float(features[col].min()),
                float(features[col].max()),
                float(features[col].median())
            )

    if st.button("Predict Price"):

        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]

        st.success("Prediction Successful")

        st.metric(
            "Estimated Flight Price",
            f"₹ {prediction:,.0f}"
        )

# -------------------------
# DASHBOARD PAGE
# -------------------------

elif page == "📊 Dashboard":

    st.title("📊 Flight Dashboard")

    if df is None:
        st.error("Dataset not found.")
        st.stop()

    st.subheader("Price Distribution")

    fig = px.histogram(
        df,
        x="price",
        nbins=50
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if "airline" in df.columns:

        st.subheader("Average Price by Airline")

        airline_price = (
            df.groupby("airline")["price"]
            .mean()
            .sort_values()
            .reset_index()
        )

        fig = px.bar(
            airline_price,
            x="airline",
            y="price"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if "class" in df.columns:

        st.subheader("Price by Class")

        fig = px.box(
            df,
            x="class",
            y="price"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if "stops" in df.columns:

        st.subheader("Price by Stops")

        fig = px.box(
            df,
            x="stops",
            y="price"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if "source_city" in df.columns:

        st.subheader("Source City Distribution")

        fig = px.pie(
            df,
            names="source_city"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if "destination_city" in df.columns:

        st.subheader(
            "Destination Distribution"
        )

        fig = px.pie(
            df,
            names="destination_city"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if (
        "duration" in df.columns
        and
        "price" in df.columns
    ):

        st.subheader(
            "Duration vs Price"
        )

        fig = px.scatter(
            df.sample(
                min(
                    5000,
                    len(df)
                )
            ),
            x="duration",
            y="price"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# -------------------------
# ABOUT PAGE
# -------------------------

elif page == "ℹ️ About":

    st.title("ℹ️ About Project")

    st.write("""
    Flight Price Prediction Project

    Built using:

    - Python
    - Pandas
    - Scikit-Learn
    - Plotly
    - Streamlit
    """)

    st.success(
        "Deployment Ready on Streamlit Cloud"
    )
