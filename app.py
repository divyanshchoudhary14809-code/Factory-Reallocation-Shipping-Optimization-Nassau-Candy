import streamlit as st
import pandas as pd
import os
import joblib

# ==============================
# Page Config
# ==============================

st.set_page_config(page_title="Factory Optimization System", layout="wide")

st.title("📦 Factory Reallocation & Shipping Optimization")
st.markdown("Nassau Candy Distributor")

# ==============================
# Base Directory
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================
# Load Dataset
# ==============================

data_path = os.path.join(BASE_DIR, "data", "nassau_dataset.csv")

if not os.path.exists(data_path):
    st.error("❌ Dataset not found in data folder.")
    st.stop()

data = pd.read_csv(data_path)

# ==============================
# Load Model Files
# ==============================

models_path = os.path.join(BASE_DIR, "models")

try:
    model = joblib.load(os.path.join(models_path, "lead_time_model.pkl"))
    label_encoders = joblib.load(os.path.join(models_path, "label_encoders.pkl"))
    feature_columns = joblib.load(os.path.join(models_path, "feature_columns.pkl"))
except:
    st.error("❌ Model files not found. Please run model_training.py first.")
    st.stop()

# ==============================
# Sidebar Inputs
# ==============================

st.sidebar.header("Simulation Inputs")

ship_mode = st.sidebar.selectbox(
    "Ship Mode",
    sorted(data["Ship Mode"].dropna().unique())
)

region = st.sidebar.selectbox(
    "Region",
    sorted(data["Region"].dropna().unique())
)

division = st.sidebar.selectbox(
    "Division",
    sorted(data["Division"].dropna().unique())
)

# ==============================
# Prediction Function
# ==============================

def predict_lead_time(ship_mode, region, division):

    input_df = pd.DataFrame(
        [[ship_mode, region, division]],
        columns=feature_columns
    )

    for col in input_df.columns:
        input_df[col] = label_encoders[col].transform(
            input_df[col].astype(str)
        )

    prediction = model.predict(input_df)

    return round(prediction[0], 2)

# ==============================
# Prediction Button
# ==============================

if st.button("Predict Lead Time"):

    result = predict_lead_time(ship_mode, region, division)

    st.success(f"📅 Predicted Lead Time: {result} days")

    if result > 5:
        st.warning("⚠ High Lead Time detected. Consider optimization.")
    else:
        st.success("✅ Lead Time is within acceptable range.")