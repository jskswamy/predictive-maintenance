"""
Data Registration Script

Uploads the engine sensor dataset to HuggingFace Datasets Hub.
This script is part of the CI/CD pipeline and demonstrates data versioning.
"""

import os
import sys

import pandas as pd
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load environment variables
load_dotenv()

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "jskswamy/predictive-maintenance-data"
DATA_PATH = "data/engine_data.csv"


def upload_dataset(data_path: str, repo_id: str, token: str) -> None:
    """
    Upload a CSV dataset to HuggingFace Datasets Hub.

    Args:
        data_path: Path to the CSV file
        repo_id: HuggingFace dataset repository ID
        token: HuggingFace API token with write access
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    # Standardize column names for consistency
    column_mapping = {
        "Engine rpm": "Engine RPM",
        "Lub oil pressure": "Lub Oil Pressure",
        "Fuel pressure": "Fuel Pressure",
        "Coolant pressure": "Coolant Pressure",
        "lub oil temp": "Lub Oil Temp",
        "Coolant temp": "Coolant Temp",
    }
    df = df.rename(columns=column_mapping)
    print(f"Standardized column names: {list(df.columns)}")

    # Create HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    print(f"Created dataset with {len(dataset)} samples")

    # Upload to HuggingFace
    print(f"Uploading to {repo_id}...")
    dataset.push_to_hub(repo_id, token=token, private=False)
    print(f"Successfully uploaded dataset to https://huggingface.co/datasets/{repo_id}")


def main():
    """Main entry point."""
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable not set")
        sys.exit(1)

    # Find data file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # model_building/ -> project root
    data_path = os.path.join(project_root, DATA_PATH)

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    try:
        upload_dataset(data_path, DATASET_REPO, HF_TOKEN)
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
