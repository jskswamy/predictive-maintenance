"""
Data Preparation Script

Loads data from HuggingFace, performs cleaning and feature engineering,
splits into train/validation/test sets, and uploads splits to HuggingFace.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split

# Load environment variables
load_dotenv()

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "jskswamy/predictive-maintenance-data"
TARGET_COLUMN = "Engine Condition"
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.117647  # 10% of total from remaining 85%


def load_data(repo_id: str, token: str) -> pd.DataFrame:
    """Load raw dataset file from HuggingFace."""
    print(f"Downloading engine_data.csv from {repo_id}...")
    raw_path = hf_hub_download(
        repo_id=repo_id,
        filename="engine_data.csv",
        repo_type="dataset",
        token=token,
    )
    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df)} rows")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from base sensor readings.

    Args:
        df: DataFrame with base sensor columns

    Returns:
        DataFrame with additional engineered features
    """
    eps = 1e-6
    df = df.copy()

    # Interaction features
    df["RPM_x_OilPressure"] = df["Engine RPM"] * df["Lub Oil Pressure"]
    df["RPM_x_FuelPressure"] = df["Engine RPM"] * df["Fuel Pressure"]
    df["RPM_x_CoolantPressure"] = df["Engine RPM"] * df["Coolant Pressure"]
    df["OilTemp_x_OilPressure"] = df["Lub Oil Temp"] * df["Lub Oil Pressure"]
    df["CoolantTemp_x_CoolantPressure"] = df["Coolant Temp"] * df["Coolant Pressure"]

    # Polynomial features
    df["RPM_squared"] = df["Engine RPM"] ** 2
    df["OilPressure_squared"] = df["Lub Oil Pressure"] ** 2

    # Domain features
    df["TempDiff"] = df["Lub Oil Temp"] - df["Coolant Temp"]
    df["OilFuelPressureRatio"] = df["Lub Oil Pressure"] / (df["Fuel Pressure"] + eps)
    df["CoolantOilPressureRatio"] = df["Coolant Pressure"] / (df["Lub Oil Pressure"] + eps)
    df["OilHealthIndex"] = df["Lub Oil Pressure"] / (df["Lub Oil Temp"] + eps)
    df["CoolantStress"] = df["Coolant Temp"] / (df["Coolant Pressure"] + eps)
    df["OilTempPerRPM"] = df["Lub Oil Temp"] / (df["Engine RPM"] + eps)
    df["CoolantTempPerRPM"] = df["Coolant Temp"] / (df["Engine RPM"] + eps)
    df["PressureSum"] = df["Lub Oil Pressure"] + df["Fuel Pressure"] + df["Coolant Pressure"]
    df["TempSum"] = df["Lub Oil Temp"] + df["Coolant Temp"]

    print(f"Engineered {len(df.columns)} total features")
    return df


def split_data(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test sets using stratified sampling.

    Args:
        df: Full dataset
        target: Target column name

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    X = df.drop(columns=[target])
    y = df[target]

    # First split: train+val vs test (85/15)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Second split: train vs val (75/10 of original)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_temp
    )

    # Reconstruct DataFrames
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def upload_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, repo_id: str, token: str) -> None:
    """Upload train/val/test splits to HuggingFace as individual CSV files."""
    api = HfApi()
    splits = [("train", train_df), ("val", val_df), ("test", test_df)]

    for name, df in splits:
        print(f"Uploading {name}.csv...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f, index=False)
            temp_path = f.name

        api.upload_file(
            path_or_fileobj=temp_path,
            path_in_repo=f"{name}.csv",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

        # Clean up temporary file
        os.unlink(temp_path)

    print(f"Successfully uploaded splits to https://huggingface.co/datasets/{repo_id}")


def main():
    """Main entry point."""
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable not set")
        sys.exit(1)

    try:
        # Load raw data from the same repo
        df = load_data(DATASET_REPO, HF_TOKEN)

        # Engineer features
        df = engineer_features(df)

        # Split data
        train_df, val_df, test_df = split_data(df, TARGET_COLUMN)

        # Upload processed splits to the same repo
        upload_splits(train_df, val_df, test_df, DATASET_REPO, HF_TOKEN)

        print("Data preparation complete!")

    except Exception as e:
        print(f"Error during data preparation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
