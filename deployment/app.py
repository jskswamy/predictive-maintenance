"""
Predictive Maintenance Streamlit Application

This application provides a web interface for predicting engine maintenance needs
based on sensor readings. It loads a trained AdaBoost model from HuggingFace and
performs real-time inference on user-provided sensor values.

Features:
- Configurable alert sensitivity with two operating modes
- Single engine prediction with interactive sensor inputs
- Bulk CSV import for fleet-wide batch predictions
"""

import io
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# Configuration
MODEL_REPO = "jskswamy/predictive-maintenance-model"
MODEL_FILE = "best_model.joblib"
METADATA_FILE = "model_metadata.json"

# Operating modes with threshold and performance metrics
OPERATING_MODES = {
    "Optimal -- Maximum Safety": {
        "threshold": 0.3163,
        "recall": "99.8%",
        "precision": "63.2%",
        "description": (
            "Catches virtually all failures. Best for fleets where "
            "breakdown costs are high ($350-$700 roadside + $448/hr downtime)."
        ),
    },
    "Default -- Balanced": {
        "threshold": 0.50,
        "recall": "84.6%",
        "precision": "68.4%",
        "description": (
            "Balanced detection vs false alarms. Best for fleets "
            "where inspection costs are significant."
        ),
    },
}
DEFAULT_MODE = "Optimal -- Maximum Safety"

# Column name aliases for CSV import (maps variations to canonical names)
COLUMN_ALIASES = {
    "engine rpm": "Engine RPM",
    "engine_rpm": "Engine RPM",
    "rpm": "Engine RPM",
    "lub oil pressure": "Lub Oil Pressure",
    "lub_oil_pressure": "Lub Oil Pressure",
    "oil pressure": "Lub Oil Pressure",
    "oil_pressure": "Lub Oil Pressure",
    "fuel pressure": "Fuel Pressure",
    "fuel_pressure": "Fuel Pressure",
    "coolant pressure": "Coolant Pressure",
    "coolant_pressure": "Coolant Pressure",
    "lub oil temp": "Lub Oil Temp",
    "lub_oil_temp": "Lub Oil Temp",
    "oil temp": "Lub Oil Temp",
    "oil_temp": "Lub Oil Temp",
    "oil temperature": "Lub Oil Temp",
    "oil_temperature": "Lub Oil Temp",
    "coolant temp": "Coolant Temp",
    "coolant_temp": "Coolant Temp",
    "coolant temperature": "Coolant Temp",
    "coolant_temperature": "Coolant Temp",
}

REQUIRED_COLUMNS = [
    "Engine RPM",
    "Lub Oil Pressure",
    "Fuel Pressure",
    "Coolant Pressure",
    "Lub Oil Temp",
    "Coolant Temp",
]

MAX_BULK_ROWS = 10_000

# Sensor input ranges (from training data)
SENSOR_RANGES = {
    "engine_rpm": {"min": 0, "max": 2239, "default": 800, "unit": "RPM"},
    "lub_oil_pressure": {"min": 0.0, "max": 7.26, "default": 3.0, "unit": "bar"},
    "fuel_pressure": {"min": 0.0, "max": 21.14, "default": 8.0, "unit": "bar"},
    "coolant_pressure": {"min": 0.0, "max": 7.53, "default": 2.0, "unit": "bar"},
    "lub_oil_temp": {"min": 0.0, "max": 164.35, "default": 80.0, "unit": "°C"},
    "coolant_temp": {"min": 0.0, "max": 194.59, "default": 75.0, "unit": "°C"},
}

# Sensor descriptions for help text
SENSOR_HELP = {
    "engine_rpm": "Engine revolutions per minute. Normal idle: 600-800 RPM. Higher values indicate increased load.",
    "lub_oil_pressure": "Lubrication oil pressure. Low pressure may indicate bearing wear or pump issues.",
    "fuel_pressure": "Fuel system delivery pressure. Erratic values may suggest injector problems.",
    "coolant_pressure": "Cooling system pressure. Low values can indicate leaks or pump failure.",
    "lub_oil_temp": "Lubrication oil temperature. High temperatures accelerate oil breakdown.",
    "coolant_temp": "Engine coolant temperature. Overheating can cause severe engine damage.",
}


@st.cache_resource
def load_model() -> tuple[Any, dict] | tuple[None, None]:
    """Load the trained model and metadata from HuggingFace Hub."""
    try:
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        metadata_path = hf_hub_download(repo_id=MODEL_REPO, filename=METADATA_FILE)

        model = joblib.load(model_path)
        with open(metadata_path) as f:
            metadata = json.load(f)

        return model, metadata
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None


def engineer_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the 22 engineered features for a batch of sensor readings.

    Args:
        df: DataFrame with columns: Engine RPM, Lub Oil Pressure, Fuel Pressure,
            Coolant Pressure, Lub Oil Temp, Coolant Temp.

    Returns:
        DataFrame with 22 features in the correct order for model input.
    """
    eps = 1e-6

    rpm = df["Engine RPM"]
    oil_press = df["Lub Oil Pressure"]
    fuel_press = df["Fuel Pressure"]
    cool_press = df["Coolant Pressure"]
    oil_temp = df["Lub Oil Temp"]
    cool_temp = df["Coolant Temp"]

    features = pd.DataFrame(
        {
            # Base features (6)
            "Engine RPM": rpm,
            "Lub Oil Pressure": oil_press,
            "Fuel Pressure": fuel_press,
            "Coolant Pressure": cool_press,
            "Lub Oil Temp": oil_temp,
            "Coolant Temp": cool_temp,
            # Interaction features (5)
            "RPM_x_OilPressure": rpm * oil_press,
            "RPM_x_FuelPressure": rpm * fuel_press,
            "RPM_x_CoolantPressure": rpm * cool_press,
            "OilTemp_x_OilPressure": oil_temp * oil_press,
            "CoolantTemp_x_CoolantPressure": cool_temp * cool_press,
            # Polynomial features (2)
            "RPM_squared": rpm**2,
            "OilPressure_squared": oil_press**2,
            # Domain features (9)
            "TempDiff": oil_temp - cool_temp,
            "OilFuelPressureRatio": oil_press / (fuel_press + eps),
            "CoolantOilPressureRatio": cool_press / (oil_press + eps),
            "OilHealthIndex": oil_press / (oil_temp + eps),
            "CoolantStress": cool_temp / (cool_press + eps),
            "OilTempPerRPM": oil_temp / (rpm + eps),
            "CoolantTempPerRPM": cool_temp / (rpm + eps),
            "PressureSum": oil_press + fuel_press + cool_press,
            "TempSum": oil_temp + cool_temp,
        }
    )

    return features


def engineer_features(
    rpm: float,
    oil_press: float,
    fuel_press: float,
    cool_press: float,
    oil_temp: float,
    cool_temp: float,
) -> pd.DataFrame:
    """Create the 22 engineered features for a single engine reading."""
    row = pd.DataFrame(
        [
            {
                "Engine RPM": rpm,
                "Lub Oil Pressure": oil_press,
                "Fuel Pressure": fuel_press,
                "Coolant Pressure": cool_press,
                "Lub Oil Temp": oil_temp,
                "Coolant Temp": cool_temp,
            }
        ]
    )
    return engineer_features_batch(row)


def validate_input(value: float, sensor: str) -> list[str]:
    """Validate sensor input against expected ranges."""
    warnings = []
    config = SENSOR_RANGES[sensor]

    if value < config["min"]:
        warnings.append(
            f"{sensor.replace('_', ' ').title()}: Value {value} is below expected minimum ({config['min']} {config['unit']})"
        )
    elif value > config["max"]:
        warnings.append(
            f"{sensor.replace('_', ' ').title()}: Value {value} is above expected maximum ({config['max']} {config['unit']})"
        )

    return warnings


def predict(model: Any, features: pd.DataFrame, threshold: float) -> dict:
    """Make a prediction using the model for a single observation."""
    probability = model.predict_proba(features)[0, 1]
    prediction = 1 if probability >= threshold else 0
    label = "Maintenance Required" if prediction == 1 else "Normal Operation"

    return {
        "prediction": prediction,
        "probability": probability,
        "label": label,
        "threshold": threshold,
    }


def predict_batch(
    model: Any, features: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """
    Make predictions for a batch of observations.

    Returns:
        DataFrame with Probability, Prediction, and Status columns.
    """
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    statuses = np.where(predictions == 1, "Maintenance Required", "Normal Operation")

    return pd.DataFrame(
        {
            "Probability": probabilities,
            "Prediction": predictions,
            "Status": statuses,
        }
    )


def validate_csv(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Validate and normalize an uploaded CSV for batch prediction.

    Returns:
        Tuple of (cleaned DataFrame with canonical column names, list of warnings).
    """
    warnings = []

    # Normalize column names: lowercase for matching
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
        elif col in REQUIRED_COLUMNS:
            rename_map[col] = col

    df = df.rename(columns=rename_map)

    # Check for missing required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return pd.DataFrame(), [f"Missing required columns: {', '.join(missing)}"]

    # Keep only required columns
    df = df[REQUIRED_COLUMNS].copy()

    # Coerce non-numeric values
    original_len = len(df)
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN values
    df = df.dropna().reset_index(drop=True)
    dropped = original_len - len(df)
    if dropped > 0:
        warnings.append(
            f"{dropped} row(s) dropped due to missing or non-numeric values."
        )

    # Row limit
    if len(df) > MAX_BULK_ROWS:
        warnings.append(
            f"CSV truncated to {MAX_BULK_ROWS:,} rows (uploaded {len(df):,})."
        )
        df = df.head(MAX_BULK_ROWS)

    if len(df) == 0:
        warnings.append("No valid rows remaining after cleaning.")

    return df, warnings


def generate_sample_csv() -> str:
    """Generate a sample CSV template with correct headers and example rows."""
    sample = pd.DataFrame(
        {
            "Engine RPM": [800, 1200, 1500, 600, 1800],
            "Lub Oil Pressure": [3.0, 2.5, 4.1, 1.2, 3.8],
            "Fuel Pressure": [8.0, 6.5, 10.0, 3.0, 9.5],
            "Coolant Pressure": [2.0, 1.8, 2.5, 0.8, 2.2],
            "Lub Oil Temp": [80.0, 95.0, 75.0, 120.0, 85.0],
            "Coolant Temp": [75.0, 88.0, 70.0, 110.0, 80.0],
        }
    )
    return sample.to_csv(index=False)


def color_prediction_rows(row: pd.Series) -> list[str]:
    """Apply row-level styling based on prediction status."""
    if row.get("Status") == "Maintenance Required":
        return ["background-color: #ffcccc"] * len(row)
    return ["background-color: #ccffcc"] * len(row)


def render_sidebar() -> float:
    """Render the sidebar with operating mode selection. Returns the active threshold."""
    with st.sidebar:
        st.header("Alert Sensitivity")

        mode = st.radio(
            "Operating Mode",
            options=list(OPERATING_MODES.keys()),
            index=list(OPERATING_MODES.keys()).index(DEFAULT_MODE),
            label_visibility="collapsed",
        )

        config = OPERATING_MODES[mode]

        st.metric("Threshold", f"{config['threshold']:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Recall", config["recall"])
        with col2:
            st.metric("Precision", config["precision"])

        st.info(config["description"])

    return config["threshold"]


def render_single_prediction(model: Any, metadata: dict, threshold: float):
    """Render the single engine prediction tab."""
    st.subheader("Sensor Readings")

    col1, col2 = st.columns(2)

    with col1:
        engine_rpm = st.number_input(
            "Engine RPM",
            min_value=0.0,
            max_value=5000.0,
            value=float(SENSOR_RANGES["engine_rpm"]["default"]),
            step=10.0,
            help=SENSOR_HELP["engine_rpm"],
        )

        lub_oil_pressure = st.number_input(
            "Lub Oil Pressure (bar)",
            min_value=0.0,
            max_value=20.0,
            value=float(SENSOR_RANGES["lub_oil_pressure"]["default"]),
            step=0.1,
            help=SENSOR_HELP["lub_oil_pressure"],
        )

        fuel_pressure = st.number_input(
            "Fuel Pressure (bar)",
            min_value=0.0,
            max_value=50.0,
            value=float(SENSOR_RANGES["fuel_pressure"]["default"]),
            step=0.5,
            help=SENSOR_HELP["fuel_pressure"],
        )

    with col2:
        coolant_pressure = st.number_input(
            "Coolant Pressure (bar)",
            min_value=0.0,
            max_value=20.0,
            value=float(SENSOR_RANGES["coolant_pressure"]["default"]),
            step=0.1,
            help=SENSOR_HELP["coolant_pressure"],
        )

        lub_oil_temp = st.number_input(
            "Lub Oil Temp (°C)",
            min_value=0.0,
            max_value=250.0,
            value=float(SENSOR_RANGES["lub_oil_temp"]["default"]),
            step=1.0,
            help=SENSOR_HELP["lub_oil_temp"],
        )

        coolant_temp = st.number_input(
            "Coolant Temp (°C)",
            min_value=0.0,
            max_value=250.0,
            value=float(SENSOR_RANGES["coolant_temp"]["default"]),
            step=1.0,
            help=SENSOR_HELP["coolant_temp"],
        )

    # Validate inputs
    all_warnings = []
    inputs = {
        "engine_rpm": engine_rpm,
        "lub_oil_pressure": lub_oil_pressure,
        "fuel_pressure": fuel_pressure,
        "coolant_pressure": coolant_pressure,
        "lub_oil_temp": lub_oil_temp,
        "coolant_temp": coolant_temp,
    }

    for sensor, value in inputs.items():
        all_warnings.extend(validate_input(value, sensor))

    if all_warnings:
        st.warning("**Input Validation Warnings:**")
        for warning in all_warnings:
            st.caption(f"- {warning}")
        st.caption(
            "*Values outside training data ranges may produce less reliable predictions.*"
        )

    st.divider()

    if st.button(
        "Predict Maintenance Status", type="primary", use_container_width=True
    ):
        try:
            features = engineer_features(
                engine_rpm,
                lub_oil_pressure,
                fuel_pressure,
                coolant_pressure,
                lub_oil_temp,
                coolant_temp,
            )

            result = predict(model, features, threshold)

            st.subheader("Prediction Result")

            if result["prediction"] == 0:
                st.success(f"**{result['label']}**")
                st.markdown(
                    "The engine is operating within normal parameters. "
                    "Continue regular monitoring."
                )
            else:
                st.error(f"**{result['label']}**")
                st.markdown(
                    "The model indicates potential engine issues. "
                    "**Schedule maintenance inspection.**"
                )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{result['probability']:.1%}")
            with col2:
                st.metric("Threshold", f"{result['threshold']:.4f}")
            with col3:
                st.metric("Raw Probability", f"{result['probability']:.4f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


def render_bulk_import(model: Any, metadata: dict, threshold: float):
    """Render the bulk CSV import tab."""
    st.markdown(
        "Upload a CSV file with engine sensor readings to predict maintenance "
        "status for an entire fleet."
    )
    st.markdown(
        f"**Required columns:** {', '.join(REQUIRED_COLUMNS)}"
    )

    # Sample CSV download
    sample_csv = generate_sample_csv()
    st.download_button(
        "Download Sample CSV Template",
        data=sample_csv,
        file_name="sensor_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload sensor readings CSV", type=["csv"])

    if uploaded_file is None:
        return

    # Read and validate CSV
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    if raw_df.empty:
        st.warning("The uploaded CSV file is empty.")
        return

    cleaned_df, warnings = validate_csv(raw_df)

    for w in warnings:
        if "Missing required" in w or "No valid rows" in w:
            st.error(w)
        else:
            st.warning(w)

    if cleaned_df.empty:
        return

    # Data preview
    st.subheader("Data Preview")
    st.dataframe(cleaned_df.head(), use_container_width=True)
    st.caption(f"{len(cleaned_df):,} valid rows loaded.")

    st.divider()

    if st.button(
        "Run Batch Predictions", type="primary", use_container_width=True
    ):
        with st.spinner("Running predictions..."):
            features = engineer_features_batch(cleaned_df)
            results = predict_batch(model, features, threshold)

        # Summary metrics
        st.subheader("Batch Results")
        total = len(results)
        maint_count = int((results["Prediction"] == 1).sum())
        normal_count = total - maint_count

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Engines", f"{total:,}")
        with col2:
            st.metric("Maintenance Required", f"{maint_count:,}")
        with col3:
            st.metric("Normal Operation", f"{normal_count:,}")

        # Results table with color coding
        display_df = pd.concat(
            [cleaned_df.reset_index(drop=True), results.reset_index(drop=True)],
            axis=1,
        )
        display_df["Probability"] = display_df["Probability"].round(4)

        styled = display_df.style.apply(color_prediction_rows, axis=1)
        st.dataframe(styled, use_container_width=True)

        # Download results
        csv_buffer = io.StringIO()
        display_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download Results CSV",
            data=csv_buffer.getvalue(),
            file_name="maintenance_predictions.csv",
            mime="text/csv",
        )


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Predictive Maintenance",
        page_icon="🔧",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.title("Engine Predictive Maintenance")
    st.markdown(
        "Predict engine maintenance needs based on sensor readings. "
        "Select an operating mode in the sidebar and use the tabs below."
    )

    # Load model
    with st.spinner("Loading model from HuggingFace..."):
        model, metadata = load_model()

    if model is None:
        st.error(
            "Unable to load the prediction model. "
            "Please try again later or contact support."
        )
        st.stop()

    # Sidebar: operating mode selection
    threshold = render_sidebar()

    # Model info expander
    active_mode = [
        name
        for name, cfg in OPERATING_MODES.items()
        if cfg["threshold"] == threshold
    ][0]

    with st.expander("Model Information"):
        st.markdown(f"**Model**: {metadata.get('model_name', 'Unknown')}")
        st.markdown(f"**Algorithm**: {metadata.get('algorithm', 'Unknown')}")
        st.markdown(
            f"**Test Recall**: {metadata.get('test_metrics', {}).get('recall', 0):.2%}"
        )
        st.markdown(f"**Active Mode**: {active_mode}")
        st.markdown(f"**Decision Threshold**: {threshold:.4f}")

    st.divider()

    # Tabs
    tab1, tab2 = st.tabs(["Single Prediction", "Bulk Import"])

    with tab1:
        render_single_prediction(model, metadata, threshold)

    with tab2:
        render_bulk_import(model, metadata, threshold)

    # Footer
    st.divider()
    st.caption(
        "Built with Streamlit | Model hosted on HuggingFace | "
        "[PGP-AIML Capstone Project](https://github.com/jskswamy/AIML-LearningBytes)"
    )


if __name__ == "__main__":
    main()
