"""
Data Registration Script

Uploads the engine sensor dataset to HuggingFace Datasets Hub.
This script is part of the CI/CD pipeline and demonstrates data versioning.
"""

import os
import sys
import tempfile

import pandas as pd
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
    Upload a CSV dataset to HuggingFace Datasets Hub as a file.

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

    # Save standardized data to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f, index=False)
        temp_path = f.name

    # Upload file to HuggingFace using HfApi
    print(f"Uploading to {repo_id}...")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=temp_path,
        path_in_repo="engine_data.csv",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    print(
        f"Successfully uploaded engine_data.csv to https://huggingface.co/datasets/{repo_id}"
    )

    # Clean up temporary file
    os.unlink(temp_path)

    # Upload README (dataset card)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    readme_path = os.path.join(project_root, "data", "README.md")
    if os.path.exists(readme_path):
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print(f"Uploaded dataset card (README.md)")


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
