
import os
import requests
import pandas as pd
import streamlit as st

# Flask backend URL within the Docker network
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

# Store information from the training dataset
STORE_DETAILS = {
    "OUT001": {
        "Store_Size": "High",
        "Store_Location_City_Type": "Tier 2",
        "Store_Type": "Supermarket Type1",
        "Store_Establishment_Year": 1987
    },
    "OUT002": {
        "Store_Size": "Small",
        "Store_Location_City_Type": "Tier 3",
        "Store_Type": "Food Mart",
        "Store_Establishment_Year": 1998
    },
    "OUT003": {
        "Store_Size": "Medium",
        "Store_Location_City_Type": "Tier 1",
        "Store_Type": "Departmental Store",
        "Store_Establishment_Year": 1999
    },
    "OUT004": {
        "Store_Size": "Medium",
        "Store_Location_City_Type": "Tier 2",
        "Store_Type": "Supermarket Type2",
        "Store_Establishment_Year": 2009
    }
}

FEATURES = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year",
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Id",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type"
]

tab1, tab2 = st.tabs([
    "Individual Prediction",
    "Batch CSV Prediction"
])

# --------------------------------------------------
# TAB 1 - Individual Prediction
# --------------------------------------------------

with tab1:

    st.subheader("Enter Product and Store Information")

    col1, col2 = st.columns(2)

    with col1:

        product_weight = st.number_input(
            "Product Weight",
            min_value=0.0,
            value=10.0
        )

        product_area = st.number_input(
            "Product Allocated Area",
            min_value=0.0,
            value=0.05,
            format="%.4f"
        )

        product_mrp = st.number_input(
            "Product MRP",
            min_value=0.0,
            value=150.0
        )

        sugar_content = st.selectbox(
            "Product Sugar Content",
            ["Low Sugar", "Regular", "No Sugar"]
        )

        product_type = st.text_input(
            "Product Type",
            value="Fruits and Vegetables"
        )

    with col2:

        store_id = st.selectbox(
            "Store ID",
            list(STORE_DETAILS.keys())
        )

        # Automatically populate store details
        store_info = STORE_DETAILS[store_id]

        st.text_input(
            "Store Size",
            value=store_info["Store_Size"],
            disabled=True
        )

        st.text_input(
            "Store Location City Type",
            value=store_info["Store_Location_City_Type"],
            disabled=True
        )

        st.text_input(
            "Store Type",
            value=store_info["Store_Type"],
            disabled=True
        )

        st.number_input(
            "Store Establishment Year",
            value=store_info["Store_Establishment_Year"],
            disabled=True
        )

    if st.button("Predict Sales", type="primary"):

        payload = {
            "Product_Weight": product_weight,
            "Product_Allocated_Area": product_area,
            "Product_MRP": product_mrp,
            "Store_Establishment_Year":
                store_info["Store_Establishment_Year"],
            "Product_Sugar_Content": sugar_content,
            "Product_Type": product_type,
            "Store_Id": store_id,
            "Store_Size": store_info["Store_Size"],
            "Store_Location_City_Type":
                store_info["Store_Location_City_Type"],
            "Store_Type": store_info["Store_Type"]
        }

        try:

            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30
            )

            response.raise_for_status()

            predicted_sales = response.json()["predicted_sales"]

            st.success("Prediction completed successfully!")

            st.metric(
                "Predicted Sales",
                f"{predicted_sales:,.2f}"
            )

        except requests.exceptions.RequestException as e:
            st.error(f"Unable to retrieve prediction: {e}")

        except (KeyError, ValueError) as e:
            st.error(f"Unexpected prediction response: {e}")


# --------------------------------------------------
# TAB 2 - Batch CSV Prediction
# --------------------------------------------------

with tab2:

    st.subheader("Upload a CSV File for Batch Prediction")

    st.write(
        "Upload a CSV file containing product and store "
        "information to predict sales for multiple records."
    )

    with st.expander("View required CSV columns"):
        st.write(", ".join(FEATURES))

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_file.seek(0)
            preview_df = pd.read_csv(uploaded_file)

            st.write("Uploaded Data Preview")
            st.dataframe(preview_df.head())

            if st.button("Run Batch Prediction"):

                missing_columns = [
                    col for col in FEATURES
                    if col not in preview_df.columns
                ]

                if missing_columns:
                    st.error(
                        f"Missing required columns: {missing_columns}"
                    )

                elif preview_df.empty:
                    st.error("The uploaded CSV file is empty.")

                else:

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "text/csv"
                        )
                    }

                    response = requests.post(
                        f"{API_URL}/predict_batch",
                        files=files,
                        timeout=120
                    )

                    response.raise_for_status()

                    results = pd.DataFrame(
                        response.json()["predictions"]
                    )

                    st.success(
                        f"Successfully predicted {len(results)} records!"
                    )

                    st.dataframe(results)

                    csv_output = results.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "Download Predictions CSV",
                        data=csv_output,
                        file_name="superkart_predictions.csv",
                        mime="text/csv"
                    )

        except requests.exceptions.RequestException as e:
            st.error(f"Batch prediction failed: {e}")

        except (ValueError, KeyError) as e:
            st.error(f"Unable to process CSV file: {e}")
