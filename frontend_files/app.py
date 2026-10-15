
import os
import io
import requests
import pandas as pd
import streamlit as st

# Flask backend address
# "backend" will be the Docker container name on our shared network.
API_URL = os.getenv("API_URL", "http://backend:5000")

st.set_page_config(
    page_title="SuperKart Sales Predictor",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 SuperKart Sales Prediction")
st.write(
    "Predict product-store sales using our trained "
    "Random Forest machine learning model."
)

tab1, tab2 = st.tabs([
    "Individual Prediction",
    "Batch CSV Prediction"
])

# --------------------------------------------------
# TAB 1: Individual Prediction
# --------------------------------------------------

with tab1:
    st.subheader("Enter Product and Store Information")

    col1, col2 = st.columns(2)

    with col1:
        product_weight = st.number_input(
            "Product Weight", min_value=0.0, value=10.0
        )
        product_area = st.number_input(
            "Product Allocated Area", min_value=0.0, value=0.05,
            format="%.4f"
        )
        product_mrp = st.number_input(
            "Product MRP", min_value=0.0, value=150.0
        )
        sugar_content = st.selectbox(
            "Product Sugar Content",
            ["Low Sugar", "Regular", "No Sugar"]
        )
        product_type = st.text_input(
            "Product Type", value="Fruits and Vegetables"
        )

    with col2:
        store_id
