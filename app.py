import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Page Configuration Configuration
st.set_page_config(
    page_title="Flight Price Intelligence Portal",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inline Training Engine Definition (Eliminates File Import Errors)
def internal_train_pipeline(data_path, model_path, meta_path):
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    
    # Remove high-cardinality flight column to ensure ultra-fast execution
    if 'flight' in df.columns:
        df = df.drop(columns=['flight'])
    
    X = df.drop(columns=["price"])
    y = df["price"]
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Using high-performance optimized XGBoost Regressor directly
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    pipeline.fit(X_train, y_train)
    
    preds = pipeline.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    joblib.dump(pipeline, model_path)
    
    unique_cat_values = {col: df[col].dropna().unique().tolist() for col in categorical_cols}
    metadata = {
        "training_columns": X.columns.tolist(),
        "categorical_columns": categorical_cols,
        "numerical_columns": numerical_cols,
        "best_model_name": "XGBoost Regressor",
        "best_score": r2,
        "metrics": {"R2": r2, "MAE": mae, "RMSE": rmse},
        "unique_values": unique_cat_values
    }
    joblib.dump(metadata, meta_path)
    return metadata

# 3. Premium Branding Custom CSS Injection
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); color: #f8fafc; }
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
        color: white; border-radius: 8px; border: none; padding: 0.6rem 2rem; font-weight: 600;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    .kpi-card {
        background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px; padding: 1.5rem; text-align: center;
    }
    .premium-price-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
        border: 2px solid #3b82f6; border-radius: 16px; padding: 2.5rem; text-align: center; margin: 2rem 0;
    }
    .premium-price { font-size: 3rem; font-weight: 800; color: #60a5fa; }
    h1, h2, h3 { color: #ffffff !important; font-family: 'Inter', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# 4. Target Variables & File Paths
DATA_FILE = "flight_dataset.csv"
MODEL_FILE = "model.pkl"
META_FILE = "model_metadata.pkl"

@st.cache_data
def load_raw_dataset(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df
    return None

df_raw = load_raw_dataset(DATA_FILE)

if df_raw is None:
    st.error(f"🚨 Essential system dataset file '{DATA_FILE}' is missing from the directory.")
    st.stop()

# Auto-train block if files are missing from directory
if not os.path.exists(MODEL_FILE) or not os.path.exists(META_FILE):
    with st.spinner("⏳ Operational model binaries missing. Automatically generating optimized ML models..."):
        try:
            metadata = internal_train_pipeline(DATA_FILE, MODEL_FILE, META_FILE)
            model_pipeline = joblib.load(MODEL_FILE)
        except Exception as e:
            st.error(f"💥 Failed to auto-train model structure: {str(e)}")
            st.stop()
else:
    model_pipeline = joblib.load(MODEL_FILE)
    metadata = joblib.load(META_FILE)

# 5. Global Sidebar Navigation
st.sidebar.markdown("<h2 style='text-align: center; color: white;'>✈️ Flight Portal</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
page_selection = st.sidebar.radio(
    "Navigate Terminal System",
    ["Dashboard Overview", "Price Engine Unit", "Visual Analytics Analytics", "System Architecture Specs"]
)

# --- PAGE 1: HOME PAGE ---
if page_selection == "Dashboard Overview":
    st.markdown("# ✈️ Flight Price Intelligence Portal")
    st.markdown("### Production-Grade Enterprise Analytics & Predictive Inference Framework")
    st.markdown("---")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='kpi-card'><h3>📊 Observations</h3><p style='font-size:1.8rem; font-weight:700; color:#3b82f6;'>{df_raw.shape[0]:,}</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi-card'><h3>📐 Matrix Features</h3><p style='font-size:1.8rem; font-weight:700; color:#10b981;'>{df_raw.shape[1]}</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi-card'><h3>🚫 Missing Elements</h3><p style='font-size:1.8rem; font-weight:700; color:#ef4444;'>{df_raw.isnull().sum().sum()}</p></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='kpi-card'><h3>🤖 Model Core</h3><p style='font-size:1.2rem; font-weight:700; color:#a855f7;'>{metadata['best_model_name']}</p></div>", unsafe_allow_html=True)

    st.markdown("### 📋 Production Dataset Preview")
    st.dataframe(df_raw.head(15), use_container_width=True)

    col_info_left, col_info_right = st.columns(2)
    with col_info_left:
        st.markdown("### 🩻 High-Level Dimensional Analysis Data Types")
        dtype_df = pd.DataFrame(df_raw.dtypes.astype(str), columns=["Data Type"])
        st.table(dtype_df)
    with col_info_right:
        st.markdown("### 📈 Continuous Value Profile Distributions")
        st.dataframe(df_raw.describe(), use_container_width=True)

# --- PAGE 2: PRICE PREDICTION PAGE ---
elif page_selection == "Price Engine Unit":
    st.markdown("# 🤖 High-Fidelity Price Prediction Unit")
    st.markdown("##### Direct continuous value prediction using downstream pipeline transforms.")
    st.markdown("---")
    
    st.markdown("### Input Trip Configuration Matrix")
    
    cached_uniques = metadata["unique_values"]
    num_cols = metadata["numerical_columns"]
    train_cols = metadata["training_columns"]
    
    input_payload = {}
    ui_columns = st.columns(2)
    
    idx = 0
    for col in train_cols:
        target_ui = ui_columns[idx % 2]
        with target_ui:
            if col in cached_uniques:
                input_payload[col] = st.selectbox(f"Select Feature: {col.replace('_', ' ').title()}", options=cached_uniques[col])
            elif col in num_cols:
                min_val = float(df_raw[col].min())
                max_val = float(df_raw[col].max())
                mean_val = float(df_raw[col].mean())
                
                if col == "days_left":
                    input_payload[col] = st.slider(f"Adjust Matrix: Days Left Before Departure", int(min_val), int(max_val), int(mean_val))
                elif col == "duration":
                    input_payload[col] = st.slider(f"Adjust Matrix: Flight Duration (Hours)", float(min_val), float(max_val), float(mean_val))
                else:
                    input_payload[col] = st.number_input(f"Define Vector Variable: {col.title()}", min_value=min_val, max_value=max_val, value=mean_val)
        idx += 1

    st.markdown("---")
    if st.button("✨ Evaluate & Execute Inference Pipeline"):
        # INTERCEPT IDENTICAL ROUTE ANOMALIES (Kolkata to Kolkata, etc.)
        if input_payload.get('source_city') == input_payload.get('destination_city'):
            st.error("❌ **Invalid Route Selected:** The origin departure city cannot match your target destination city. Please adjust your travel configurations.")
        else:
            try:
                payload_df = pd.DataFrame([input_payload])
                payload_df = payload_df[train_cols]
                
                prediction = model_pipeline.predict(payload_df)[0]
                
                st.markdown(f"""
                    <div class='premium-price-card'>
                        <h2>Calculated Expected Flight Value</h2>
                        <div class='premium-price'>₹ {prediction:,.2f}</div>
                        <p style='color: #94a3b8; margin-top: 1rem;'>System Engine Verification Confidence: <b>{(metadata['best_score']*100):.2f}% (R²)</b></p>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Structural error vector encountered during inference compilation: {str(e)}")

# --- PAGE 3: VISUAL ANALYTICS DASHBOARD ---
elif page_selection == "Visual Analytics Analytics":
    st.markdown("# 📊 Advanced Exploratory Analytics Dashboard")
    st.markdown("---")
    
    st.markdown("### 🛞 Live Analytics Filtering Controls")
    f_cols = st.columns(5)
    
    with f_cols[0]:
        sel_airline = st.multiselect("Airline Operator", options=df_raw["airline"].unique(), default=df_raw["airline"].unique())
    with f_cols[1]:
        sel_class = st.multiselect("Cabin Seating Tier", options=df_raw["class"].unique(), default=df_raw["class"].unique())
    with f_cols[2]:
        sel_source = st.multiselect("Origin City Terminal", options=df_raw["source_city"].unique(), default=df_raw["source_city"].unique())
    with f_cols[3]:
        sel_dest = st.multiselect("Destination City Axis", options=df_raw["destination_city"].unique(), default=df_raw["destination_city"].unique())
    with f_cols[4]:
        sel_stops = st.multiselect("Route Intermediary Stops", options=df_raw["stops"].unique(), default=df_raw["stops"].unique())
        
    mask = (
        df_raw["airline"].isin(sel_airline) & 
        df_raw["class"].isin(sel_class) & 
        df_raw["source_city"].isin(sel_source) & 
        df_raw["destination_city"].isin(sel_dest) & 
        df_raw["stops"].isin(sel_stops)
    )
    df_filtered = df_raw[mask]
    
    if df_filtered.empty:
        st.warning("⚠️ High dimensional search parameters yield an empty matrix selection. Realign dashboard controls.")
    else:
        st.markdown("## 💰 A. Structural Price Distribution Analytics")
        a1, a2 = st.columns(2)
        with a1:
            fig_hist = px.histogram(df_filtered, x="price", nbins=50, title="Price Value Density Count Histogram", color_discrete_sequence=["#3b82f6"], marginal="violin")
            st.plotly_chart(fig_hist, use_container_width=True)
        with a2:
            fig_box = px.box(df_filtered, y="price", x="class", title="Price Outlier Box Metric Layout by Seating Class", color="class", color_discrete_sequence=["#10b981", "#6366f1"])
            st.plotly_chart(fig_box, use_container_width=True)
            
        st.markdown("## ✈️ B. Airline Aviation Market Component Profiles")
        b1, b2 = st.columns(2)
        with b1:
            airline_counts = df_filtered["airline"].value_counts().reset_index()
            fig_pie = px.pie(airline_counts, names="airline", values="count", title="Operator Volumetric Share", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)
        with b2:
            fig_air_avg = px.bar(df_filtered.groupby("airline")["price"].mean().reset_index(), x="airline", y="price", title="Average Target Ticket Cost Across Operator Fleets", color="airline")
            st.plotly_chart(fig_air_avg, use_container_width=True)
            
        st.markdown("## 🗺️ C. Geographical Routing Cross-Reference Matrices")
        c1, c2 = st.columns(2)
        with c1:
            fig_src = px.histogram(df_filtered, x="source_city", title="Operational Density by Origin City", color_discrete_sequence=["#f59e0b"])
            st.plotly_chart(fig_src, use_container_width=True)
        with c2:
            fig_dst = px.histogram(df_filtered, x="destination_city", title="Traffic Destination Density", color_discrete_sequence=["#ec4899"])
            st.plotly_chart(fig_dst, use_container_width=True)
            
        st.markdown("### Hub-to-Hub Flight Operator Target Pricing Heatmap")
        pivot_df = df_filtered.pivot_table(index="airline", columns="source_city", values="price", aggfunc="mean").fillna(0)
        fig_heat = px.imshow(pivot_df, text_auto=True, labels=dict(x="Source City Location Axis", y="Airline Vendor", color="Mean Price (₹)"), color_continuous_scale="Viridis")
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("## 💺 D. Class Configuration & Route Intermediary Profiling")
        d1, d2 = st.columns(2)
        with d1:
            fig_class_violin = px.violin(df_filtered, y="price", x="class", color="class", box=True, points="all", title="Pricing Dispersion Violin Map by Class")
            st.plotly_chart(fig_class_violin, use_container_width=True)
        with d2:
            fig_stops_avg = px.bar(df_filtered.groupby("stops")["price"].mean().reset_index(), x="stops", y="price", title="Mean Ticket Pricing by Stops", color="stops", color_discrete_sequence=px.colors.sequential.Agrid)
            st.plotly_chart(fig_stops_avg, use_container_width=True)

        st.markdown("## ⏰ E. Chronological Temporal Flight Allocation Boundaries")
        e1, e2 = st.columns(2)
        with e1:
            fig_dep = px.box(df_filtered, x="departure_time", y="price", title="Departure Time Period Pricing Spectrum", color="departure_time")
            st.plotly_chart(fig_dep, use_container_width=True)
        with e2:
            fig_arr = px.box(df_filtered, x="arrival_time", y="price", title="Arrival Time Period Pricing Spectrum", color="arrival_time")
            st.plotly_chart(fig_arr, use_container_width=True)

        st.markdown("## 📉 F. Continuous Spatial Variable Interactions")
        f1, f2 = st.columns(2)
        with f1:
            fig_scat1 = px.scatter(df_filtered, x="duration", y="price", color="airline", opacity=0.5, title="Flight Duration Hours vs Price Vector")
            st.plotly_chart(fig_scat1, use_container_width=True)
        with f2:
            fig_scat2 = px.scatter(df_filtered, x="days_left", y="price", color="class", opacity=0.5, title="Days Remaining Prior to Takeoff vs Price Vector")
            st.plotly_chart(fig_scat2, use_container_width=True)

        st.markdown("## 🔗 G. Covariance Correlation Matrix")
        num_corr_features = ["duration", "days_left", "price"]
        available_corr_cols = [c for c in num_corr_features if c in df_filtered.columns]
        if len(available_corr_cols) > 1:
            corr_mat = df_filtered[available_corr_cols].corr()
            fig_corr = px.imshow(corr_mat, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Pearson Feature Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("## 🔍 H. Deep Diagnostic Insights Matrix Profiles")
        h1, h2 = st.columns(2)
        columns_to_show = ["airline", "source_city", "destination_city", "class", "price"]
        available_cols = [col for col in columns_to_show if col in df_filtered.columns]
        with h1:
            st.markdown("#### 🔝 Top 10 Most Expensive Flight Manifests")
            st.dataframe(df_filtered.sort_values(by="price", ascending=False).head(10)[available_cols], use_container_width=True)
        with h2:
            st.markdown("#### 📉 Top 10 High Efficiency Value Flight Entries")
            st.dataframe(df_filtered.sort_values(by="price", ascending=True).head(10)[available_cols], use_container_width=True)

        h3, h4 = st.columns(2)
        with h3:
            st.markdown("#### 🌆 Average Ticket Cost by Source City")
            st.dataframe(df_filtered.groupby("source_city")["price"].mean().reset_index().sort_values(by="price", ascending=False), use_container_width=True)
        with h4:
            st.markdown("#### 🏙️ Average Ticket Cost by Destination City")
            st.dataframe(df_filtered.groupby("destination_city")["price"].mean().reset_index().sort_values(by="price", ascending=False), use_container_width=True)

# --- PAGE 4: ABOUT PAGE ---
elif page_selection == "System Architecture Specs":
    st.markdown("# ⚙️ Engineering Execution Spec Sheet & System Logs")
    st.markdown("---")
    
    st.markdown("""
    ### 🔬 Advanced Core Functional Architecture Overview
    This platform acts as an automated ML deployment instance engineered specifically to streamline high-dimensional, cross-category preprocessing steps. It features integrated categorical safety bounds that guarantee smooth operational workflows and protect against input mismatch disruptions.
    
    ### 🛠️ Production Technology Stack Configuration
    - **Interface Core Layer:** Streamlit (Wide-frame customized responsive rendering block layout)
    - **High-Dimensional Linear Algebra Engine:** NumPy & Pandas DataFrames
    - **Visual Plot Graph Structs:** Plotly Express Advanced Chart Engine
    - **State Deserialization Protocol:** Joblib Compression Serialization 
    - **Mathematical Model Kernels:** Scikit-Learn Framework & Extreme Gradient Boosting Pipeline Architectures
    """)
    
    st.markdown("### 🏆 Active Selected Champion Structural Characteristics")
    st.json(metadata)
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Engineering Core Contact Infrastructure Matrix")
    
    c_btn1, c_btn2, _ = st.columns([1, 1, 4])
    with c_btn1:
        st.link_button("🌐 Access Production GitHub Repository", "https://github.com")
    with c_btn2:
        st.link_button("👔 Interface Professional LinkedIn Profile", "https://linkedin.com")
