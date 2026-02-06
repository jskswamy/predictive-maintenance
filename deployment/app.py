"""
Predictive Maintenance Streamlit Application

This application provides a web interface for predicting engine maintenance needs
based on sensor readings. It loads a trained AdaBoost model from HuggingFace and
performs real-time inference on user-provided sensor values.
"""

import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
import joblib

# Configuration
MODEL_REPO = "jskswamy/predictive-maintenance-model"
MODEL_FILE = "best_model.joblib"
METADATA_FILE = "model_metadata.json"
THRESHOLD = 0.3163  # Optimized for 99.78% recall

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
    """
    Load the trained model and metadata from HuggingFace Hub.

    Returns:
        Tuple of (model, metadata) on success, (None, None) on failure.
    """
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


def engineer_features(
    rpm: float,
    oil_press: float,
    fuel_press: float,
    cool_press: float,
    oil_temp: float,
    cool_temp: float,
) -> pd.DataFrame:
    """
    Create the 22 engineered features required by the model.

    Args:
        rpm: Engine RPM
        oil_press: Lubrication oil pressure (bar)
        fuel_press: Fuel pressure (bar)
        cool_press: Coolant pressure (bar)
        oil_temp: Lubrication oil temperature (°C)
        cool_temp: Coolant temperature (°C)

    Returns:
        DataFrame with 22 features in the correct order.
    """
    eps = 1e-6  # Small value to prevent division by zero

    features = {
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

    return pd.DataFrame([features])


def validate_input(value: float, sensor: str) -> list[str]:
    """
    Validate sensor input against expected ranges.

    Args:
        value: The sensor reading value
        sensor: The sensor name key

    Returns:
        List of warning messages (empty if valid)
    """
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
    """
    Make a prediction using the model.

    Args:
        model: Trained sklearn model
        features: DataFrame with engineered features
        threshold: Classification threshold

    Returns:
        Dictionary with prediction, probability, and label
    """
    probability = model.predict_proba(features)[0, 1]
    prediction = 1 if probability >= threshold else 0
    label = "Maintenance Required" if prediction == 1 else "Normal Operation"

    return {
        "prediction": prediction,
        "probability": probability,
        "label": label,
        "threshold": threshold,
    }


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="Predictive Maintenance",
        page_icon="🔧",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # Title and description
    st.title("🔧 Engine Predictive Maintenance")
    st.markdown(
        """
        This application predicts whether an engine requires maintenance based on sensor readings.
        Enter the current sensor values below and click **Predict** to get the maintenance recommendation.
        """
    )

    # Load model
    with st.spinner("Loading model from HuggingFace..."):
        model, metadata = load_model()

    if model is None:
        st.error(
            "Unable to load the prediction model. Please try again later or contact support."
        )
        st.stop()

    # Display model info in expander
    with st.expander("Model Information"):
        st.markdown(f"**Model**: {metadata.get('model_name', 'Unknown')}")
        st.markdown(f"**Algorithm**: {metadata.get('algorithm', 'Unknown')}")
        st.markdown(
            f"**Test Recall**: {metadata.get('test_metrics', {}).get('recall', 0):.2%}"
        )
        st.markdown(f"**Decision Threshold**: {THRESHOLD}")

    st.divider()

    # Input form
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

    # Display warnings
    if all_warnings:
        st.warning("⚠️ **Input Validation Warnings:**")
        for warning in all_warnings:
            st.caption(f"• {warning}")
        st.caption(
            "*Values outside training data ranges may produce less reliable predictions.*"
        )

    st.divider()

    # Prediction button
    if st.button(
        "🔍 Predict Maintenance Status", type="primary", use_container_width=True
    ):
        try:
            # Engineer features
            features = engineer_features(
                engine_rpm,
                lub_oil_pressure,
                fuel_pressure,
                coolant_pressure,
                lub_oil_temp,
                coolant_temp,
            )

            # Make prediction
            result = predict(model, features, THRESHOLD)

            # Display result
            st.subheader("Prediction Result")

            if result["prediction"] == 0:
                st.success(f"✅ **{result['label']}**")
                st.markdown(
                    "The engine is operating within normal parameters. Continue regular monitoring."
                )
            else:
                st.error(f"⚠️ **{result['label']}**")
                st.markdown(
                    "The model indicates potential engine issues. **Schedule maintenance inspection.**"
                )

            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{result['probability']:.1%}")
            with col2:
                st.metric("Threshold", f"{result['threshold']:.4f}")
            with col3:
                st.metric("Raw Probability", f"{result['probability']:.4f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Footer
    st.divider()
    st.caption(
        "Built with Streamlit • Model hosted on HuggingFace • "
        "[PGP-AIML Capstone Project](https://github.com/jskswamy/AIML-LearningBytes)"
    )


if __name__ == "__main__":
    main()
