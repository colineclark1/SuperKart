
import os
import joblib
import pandas as pd

from flask import Flask, request, jsonify

# --------------------------------------------------
# 1. Initialise Flask Application
# --------------------------------------------------

app = Flask(__name__)

# --------------------------------------------------
# 2. Load Trained SuperKart Model
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "superkart_model.joblib"
)

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# 3. Define Required Input Features
# --------------------------------------------------

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

NUMERIC_FEATURES = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year"
]

# --------------------------------------------------
# 4. Validate and Prepare Prediction Data
# --------------------------------------------------

def prepare_data(df):
    """
    Validate incoming data and apply the same
    preprocessing rules used during model training.
    """

    # Check for missing required columns
    missing_columns = [
        col for col in FEATURES
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Select features in the correct order
    df = df[FEATURES].copy()

    # Standardise sugar content
    df["Product_Sugar_Content"] = (
        df["Product_Sugar_Content"]
        .replace({"reg": "Regular"})
    )

    # Convert numeric inputs to appropriate types
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise"
        )

    # Check for missing values
    if df.isnull().any().any():
        raise ValueError(
            "Input data contains missing values."
        )

    return df


# --------------------------------------------------
# 5. API Home / Health Check
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Welcome to the SuperKart Sales Prediction API",
        "status": "running"
    })


# --------------------------------------------------
# 6. Individual Sales Prediction
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Expected a JSON object."
            }), 400

        # Convert JSON input to DataFrame
        input_df = pd.DataFrame([data])

        # Validate and prepare data
        input_df = prepare_data(input_df)

        # Generate prediction
        prediction = model.predict(input_df)[0]

        return jsonify({
            "predicted_sales": float(prediction)
        }), 200

    except (ValueError, TypeError, KeyError) as e:

        return jsonify({
            "error": str(e)
        }), 400

    except Exception:

        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction failed."
        }), 500


# --------------------------------------------------
# 7. Batch Sales Prediction Using CSV
# --------------------------------------------------

@app.route("/predict_batch", methods=["POST"])
def predict_batch():

    try:

        # Check whether a file was uploaded
        if "file" not in request.files:
            return jsonify({
                "error": "No CSV file uploaded."
            }), 400

        uploaded_file = request.files["file"]

        # Read uploaded CSV
        df = pd.read_csv(uploaded_file)

        if df.empty:
            return jsonify({
                "error": "CSV file is empty."
            }), 400

        # Validate and prepare input
        input_df = prepare_data(df)

        # Generate predictions
        predictions = model.predict(input_df)

        # Add predictions to original data
        result = df.copy()

        result["Predicted_Sales"] = predictions

        return jsonify({
            "predictions": result.to_dict(
                orient="records"
            )
        }), 200

    except (
        ValueError,
        TypeError,
        KeyError,
        pd.errors.ParserError
    ) as e:

        return jsonify({
            "error": str(e)
        }), 400

    except Exception:

        app.logger.exception("Batch prediction failed")

        return jsonify({
            "error": "Batch prediction failed."
        }), 500


# --------------------------------------------------
# 8. Run Flask Application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )
