"""
HuggingFace Spaces Hosting Script

Deploys the Streamlit application to HuggingFace Spaces by uploading
the deployment artifacts (app.py, requirements.txt, Dockerfile).
"""

import os
import sys

from dotenv import load_dotenv
from huggingface_hub import HfApi, upload_file

# Load environment variables
load_dotenv()

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
SPACE_REPO = "jskswamy/predictive-maintenance"
DEPLOYMENT_DIR = "deployment"

# Files to upload
DEPLOYMENT_FILES = [
    "app.py",
    "requirements.txt",
    "Dockerfile",
]

# Space README with YAML frontmatter
SPACE_README = """---
title: Predictive Maintenance
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Engine Predictive Maintenance

This application predicts whether an engine requires maintenance based on sensor readings.

## Features

- Real-time prediction using a trained AdaBoost model
- 6 sensor inputs: RPM, Oil Pressure, Fuel Pressure, Coolant Pressure, Oil Temp, Coolant Temp
- Input validation with range warnings
- Confidence scores with adjustable threshold display

## Model Information

- **Algorithm**: AdaBoost with Optuna hyperparameter tuning
- **Test Recall**: 99.78% (optimized for catching all maintenance cases)
- **Threshold**: 0.3163 (optimized for high recall)

## Usage

1. Enter the current sensor readings from the engine
2. Click "Predict Maintenance Status"
3. View the prediction result and confidence score

## Links

- [Model on HuggingFace](https://huggingface.co/jskswamy/predictive-maintenance-model)
- [Dataset on HuggingFace](https://huggingface.co/datasets/jskswamy/predictive-maintenance-data)
- [GitHub Repository](https://github.com/jskswamy/AIML-LearningBytes)

## License

MIT License - PGP-AIML Capstone Project
"""


def deploy_to_spaces(deployment_dir: str, repo_id: str, token: str) -> None:
    """
    Deploy Streamlit app to HuggingFace Spaces.

    Args:
        deployment_dir: Path to deployment directory containing app files
        repo_id: HuggingFace Spaces repository ID
        token: HuggingFace API token with write access
    """
    api = HfApi()

    # Create space if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            token=token,
            exist_ok=True,
        )
        print(f"Space repository ready: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"Note: {e}")

    # Upload deployment files
    for filename in DEPLOYMENT_FILES:
        filepath = os.path.join(deployment_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filename} not found, skipping...")
            continue

        print(f"Uploading {filename}...")
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="space",
            token=token,
        )

    # Upload README
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(SPACE_README)
        readme_path = f.name

    print("Uploading README.md...")
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
        token=token,
    )
    os.unlink(readme_path)

    print(f"\nDeployment complete!")
    print(f"Space URL: https://huggingface.co/spaces/{repo_id}")
    print(
        "Note: It may take a few minutes for the Space to build and become available."
    )


def main():
    """Main entry point."""
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable not set")
        sys.exit(1)

    # Find deployment directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # model_building/ -> project root
    deployment_dir = os.path.join(project_root, DEPLOYMENT_DIR)

    if not os.path.exists(deployment_dir):
        print(f"Error: Deployment directory not found at {deployment_dir}")
        sys.exit(1)

    try:
        deploy_to_spaces(deployment_dir, SPACE_REPO, HF_TOKEN)
    except Exception as e:
        print(f"Error deploying to Spaces: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
