import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import sys

# --- CONFIGURATION ---
st.set_page_config(page_title="Tourism AI Guide", layout="wide", initial_sidebar_state="expanded")

# --- PATH DETECTION ---
# Data Path
possible_data_paths = [Path("data"), Path("../data"), Path("notebooks/data"), Path(".")]
DATA_DIR = None
for path in possible_data_paths:
    if (path / "master_enriched_dataset.csv").exists():
        DATA_DIR = path
        break
if DATA_DIR is None: DATA_DIR = Path("data")

# Models Path
possible_model_paths = [Path("models"), Path("notebooks/models"), Path("../models"), Path(".")]
MODEL_DIR = None
for path in possible_model_paths:
    if (path / "best_model.pkl").exists():
        MODEL_DIR = path
        break
if MODEL_DIR is None: MODEL_DIR = Path("models")

# --- LOADER FUNCTIONS ---
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR / "master_enriched_dataset.csv")
    return df

@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_DIR / "best_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    le = joblib.load(MODEL_DIR / "label_encoder.pkl")
    metrics = joblib.load(MODEL_DIR / "model_metrics.pkl")
    
    # Load background data for SHAP (Explainable AI)
    try:
        background_data = joblib.load(MODEL_DIR / "background_data.pkl")
    except:
        background_data = None
        
    return model, scaler, le, metrics, background_data

try:
    df = load_data()
    model, scaler, le, metrics, background_data = load_resources()
except Exception as e:
    st.error(f"Error loading resources. Ensure Step 05 has been run.\nDetails: {e}")
    st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home & Data Exploration", "Visualizations", "Model Prediction", "Model Performance"])

st.sidebar.markdown("---")


# ==========================================
# PAGE 1: HOME & DATA EXPLORATION
# ==========================================
if page == "Home & Data Exploration":
    st.title("📊 Data Exploration & Overview")
    st.markdown("Welcome to the **AI Smart Tourism Guide**. This section provides an overview of the enriched dataset used for training.")

    # 1. Dataset Overview
    st.subheader("1. Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Locations", df.shape[0])
    col2.metric("Total Features", df.shape[1])
    col3.metric("Missing Values", df.isna().sum().sum())

    # 2. Interactive Filtering
    st.subheader("2. Filter Data")
    
    # Filter by Category
    categories = df["category"].dropna().unique().tolist()
    selected_cats = st.multiselect("Filter by Category", categories, default=categories)
    
    # Filter by Crowd Level
    crowd_levels = df["crowd_level"].dropna().unique().tolist()
    selected_crowd = st.multiselect("Filter by Crowd Level", crowd_levels, default=crowd_levels)

    # Apply Filters
    filtered_df = df[
        (df["category"].isin(selected_cats)) & 
        (df["crowd_level"].isin(selected_crowd))
    ]

    st.write(f"Showing {len(filtered_df)} rows:")
    st.dataframe(filtered_df, use_container_width=True)

    # 3. Descriptive Stats
    with st.expander("Show Descriptive Statistics"):
        st.dataframe(filtered_df.describe())

# ==========================================
# PAGE 2: VISUALIZATIONS
# ==========================================
elif page == "Visualizations":
    st.title("📈 Interactive Visualizations")
    
    # Chart 1: Eco vs Cultural Score (Scatter)
    st.subheader("1. Eco vs. Cultural Score")
    st.markdown("Does a high eco score imply a low cultural score?")
    fig1 = px.scatter(
        df, x="eco_score", y="cultural_score", 
        color="category", size="vendor_count", 
        hover_name="place_name", title="Eco vs Cultural Landscape"
    )
    st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)
    
    # Chart 2: Category Distribution (Bar)
    with col_a:
        st.subheader("2. Category Distribution")
        count_data = df["category"].value_counts().reset_index()
        count_data.columns = ["Category", "Count"]
        fig2 = px.bar(count_data, x="Category", y="Count", color="Category", title="Number of Locations per Category")
        st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: Rating Distribution (Box Plot)
    with col_b:
        st.subheader("3. Ratings by Crowd Level")
        fig3 = px.box(df, x="crowd_level", y="avg_rating", color="crowd_level", title="Rating Distribution vs Crowd Level")
        st.plotly_chart(fig3, use_container_width=True)

    # Chart 4: Empowerment Signal (Histogram)
    st.subheader("4. Local Empowerment Signal")
    fig4 = px.histogram(df, x="local_empowerment_signal", nbins=15, title="Distribution of Local Empowerment Scores")
    st.plotly_chart(fig4, use_container_width=True)

# ==========================================
# PAGE 3: MODEL PREDICTION (WITH EXPLAINABLE AI)
# ==========================================
elif page == "Model Prediction":
    st.title("🤖 Real-time Prediction & Explanation")
    st.markdown("Predict the tourism category (**Eco, Cultural, or Mixed**) based on location features and understand **why** the AI made that decision.")

    # Two columns for inputs
    c1, c2 = st.columns([1, 2])

    with c1:
        st.subheader("Input Features")
        
        # Auto-fill option
        place_names = sorted(df["place_name"].dropna().astype(str).unique().tolist())
        selected_loc = st.selectbox("Auto-fill from existing location", ["-- Manual Input --"] + place_names)

        # Defaults
        defaults = {
            "eco": 5.0, "cult": 5.0, "vend": 10, "emp": 5.0, 
            "acc": 5.0, "dang": 3.0, "rat": 4.0, "rev": 50, "sent": 0.2, "crowd": 2
        }

        if selected_loc != "-- Manual Input --":
            row = df[df["place_name"] == selected_loc].iloc[0]
            defaults["eco"] = float(row.get("eco_score", 5))
            defaults["cult"] = float(row.get("cultural_score", 5))
            defaults["vend"] = int(row.get("vendor_count", 10))
            defaults["emp"] = float(row.get("local_empowerment_index", 5))
            defaults["acc"] = float(row.get("accessibility_score", 5))
            defaults["dang"] = float(row.get("danger_level", 3))
            defaults["rat"] = float(row.get("avg_rating", 4))
            defaults["rev"] = int(row.get("review_count", 50))
            defaults["sent"] = float(row.get("avg_sentiment", 0))
            cm_map = {"low":1, "medium":2, "high":3}
            defaults["crowd"] = cm_map.get(str(row.get("crowd_level", "medium")).lower(), 2)

        # Input Widgets
        eco = st.slider("Eco Score", 1.0, 10.0, defaults["eco"])
        cult = st.slider("Cultural Score", 1.0, 10.0, defaults["cult"])
        vendor = st.number_input("Vendor Count", 0, 100, defaults["vend"])
        emp = st.slider("Empowerment Index", 1.0, 10.0, defaults["emp"])
        access = st.slider("Accessibility", 1.0, 10.0, defaults["acc"])
        danger = st.slider("Danger Level", 1.0, 10.0, defaults["dang"])
        rating = st.slider("Avg Rating", 0.0, 5.0, defaults["rat"])
        rev_cnt = st.number_input("Review Count", 0, 10000, defaults["rev"])
        sent = st.slider("Sentiment", -1.0, 1.0, defaults["sent"])
        crowd = st.selectbox("Crowd Level", [1, 2, 3], index=defaults["crowd"]-1, format_func=lambda x: {1:"Low", 2:"Medium", 3:"High"}[x])

    with c2:
        st.subheader("Prediction Results")
        
        # Prepare Data
        features = np.array([[eco, cult, vendor, emp, access, danger, rating, rev_cnt, sent, crowd]])
        feat_scaled = scaler.transform(features)
        
        if st.button("Predict Category & Explain", type="primary"):
            # Predict
            probs = model.predict_proba(feat_scaled)[0]
            pred_idx = np.argmax(probs)
            pred_label = le.inverse_transform([pred_idx])[0]
            confidence = probs[pred_idx]

            # Display
            st.success(f"### Predicted Category: **{pred_label.upper()}**")
            st.metric("Confidence Level", f"{confidence:.2%}")

            # Probability Chart
            prob_df = pd.DataFrame({"Category": le.classes_, "Probability": probs})
            fig = px.bar(prob_df, x="Category", y="Probability", color="Category", range_y=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            # --- EXPLAINABLE AI (SHAP) SECTION ---
            if background_data is not None:
                st.subheader("🔍 Explainable AI: Why this prediction?")
                st.markdown("The waterfall chart below shows how each feature pushed the prediction towards this specific category.")
                
                # Feature names for visualization
                feature_names = ["Eco Score", "Cultural Score", "Vendor Count", "Empowerment", 
                                 "Accessibility", "Danger", "Rating", "Review Count", "Sentiment", "Crowd"]
                
                try:
                    # 1. Initialize Explainer
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(feat_scaled)
                    except:
                        explainer = shap.KernelExplainer(model.predict_proba, background_data)
                        shap_values = explainer.shap_values(feat_scaled)

                    # 2. Extract values for the target class (FIXED LOGIC)
                    # We need a 1D array of SHAP values for the single sample
                    if isinstance(shap_values, list):
                        # List of [ (1, F), (1, F) ... ] -> Take array for pred_idx -> Take first row
                        values = shap_values[pred_idx][0]
                        
                        # Base value
                        if isinstance(explainer.expected_value, list):
                            base_value = explainer.expected_value[pred_idx]
                        else:
                            base_value = explainer.expected_value
                    else:
                        # Single Array case
                        if len(shap_values.shape) == 3:
                             # (1, F, C) -> Take row 0, all features, class pred_idx
                             values = shap_values[0, :, pred_idx]
                        elif len(shap_values.shape) == 2:
                             # (1, F)
                             values = shap_values[0]
                        else:
                             values = shap_values[0]
                        
                        # Base value
                        if hasattr(explainer.expected_value, "__len__") and len(explainer.expected_value) > pred_idx:
                            base_value = explainer.expected_value[pred_idx]
                        else:
                            base_value = explainer.expected_value

                    # 3. Create Plot
                    fig_shap = plt.figure(figsize=(10, 5))
                    
                    explanation = shap.Explanation(
                        values=values, 
                        base_values=base_value, 
                        data=features[0], 
                        feature_names=feature_names
                    )
                    
                    shap.plots.waterfall(explanation, show=False)
                    st.pyplot(fig_shap)
                    
                    st.info("⬆️ **Red bars** push the prediction higher (towards this category). \n⬇️ **Blue bars** push it lower.")
                    
                except Exception as e:
                    st.warning(f"Could not generate SHAP explanation: {e}")
            else:
                st.warning("⚠️ Background data for SHAP not found. Please re-run the training script (Step 05) to generate it.")

# ==========================================
# PAGE 4: MODEL PERFORMANCE
# ==========================================
elif page == "Model Performance":
    st.title("⚙️ Model Performance & Evaluation")
    
    # 1. Model Comparison
    st.subheader("1. Algorithm Comparison")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Random Forest Accuracy", f"{metrics['rf_accuracy']:.2%}")
    col2.metric("Logistic Regression Accuracy", f"{metrics['lr_accuracy']:.2%}")
    
    if 'rf_cv_score' in metrics:
        cv_score = metrics['rf_cv_score'] if metrics['best_model_name'] == 'Random Forest' else metrics['lr_cv_score']
        col3.metric("Cross-Validation Score", f"{cv_score:.2%}")

    if metrics['rf_accuracy'] > metrics['lr_accuracy']:
        st.success(f"**Random Forest** performed better and was selected as the final model.")
    else:
        st.success(f"**Logistic Regression** performed better and was selected as the final model.")

    # 2. Confusion Matrix
    st.subheader("2. Confusion Matrix (Test Set)")
    st.write("Visualizing how well the model predicts each specific category.")
    
    cm = metrics['confusion_matrix']
    labels = metrics['classes']
    
    # Create Annotated Heatmap using Plotly Figure Factory
    fig_cm = ff.create_annotated_heatmap(
        z=cm, 
        x=list(labels), 
        y=list(labels), 
        colorscale='Blues',
        showscale=True
    )
    fig_cm.update_layout(
        title_text='<b>Confusion Matrix</b>',
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")
st.caption("Final Year Project | AI Smart Guide Prototype")