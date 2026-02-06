"""
Model Training Script

Loads prepared data from HuggingFace, trains an AdaBoost model,
evaluates performance, and uploads the trained model to HuggingFace Model Hub.

Includes a recall gate check to ensure model meets minimum performance requirements.
"""

import json
import os
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, upload_file
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

# Load environment variables
load_dotenv()

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "jskswamy/predictive-maintenance-data"
MODEL_REPO = "jskswamy/predictive-maintenance-model"
TARGET_COLUMN = "Engine Condition"
RECALL_GATE = 0.95  # Minimum required recall

# Model hyperparameters (from Optuna tuning)
MODEL_PARAMS = {
    "n_estimators": 383,
    "learning_rate": 0.2608158369431769,
    "base_max_depth": 3,
}


def load_splits(repo_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits from HuggingFace."""
    print(f"Loading splits from {repo_id}...")
    dataset = load_dataset(repo_id)

    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    print(f"Loaded - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def get_feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Get the list of feature columns (excluding target)."""
    return [col for col in df.columns if col != target]


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> AdaBoostClassifier:
    """
    Train an AdaBoost classifier with optimized hyperparameters.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Trained AdaBoost model
    """
    print("Training AdaBoost model...")
    base_estimator = DecisionTreeClassifier(max_depth=MODEL_PARAMS["base_max_depth"])
    model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=MODEL_PARAMS["n_estimators"],
        learning_rate=MODEL_PARAMS["learning_rate"],
        random_state=42,
    )

    model.fit(X_train, y_train)
    print("Model training complete")
    return model


def find_optimal_threshold(model: AdaBoostClassifier, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """
    Find the optimal classification threshold by maximizing F2 score.

    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels

    Returns:
        Optimal threshold value
    """
    print("Finding optimal threshold...")
    probas = model.predict_proba(X_val)[:, 1]

    best_threshold = 0.5
    best_f2 = 0

    for threshold in np.arange(0.1, 0.9, 0.01):
        preds = (probas >= threshold).astype(int)
        # F2 weights recall more heavily
        f2 = fbeta_score(y_val, preds, beta=2)
        if f2 > best_f2:
            best_f2 = f2
            best_threshold = threshold

    print(f"Optimal threshold: {best_threshold:.4f} (F2: {best_f2:.4f})")
    return best_threshold


def evaluate_model(model: AdaBoostClassifier, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> dict:
    """
    Evaluate model performance on test set.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    print("Evaluating model...")
    probas = model.predict_proba(X_test)[:, 1]
    predictions = (probas >= threshold).astype(int)

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probas),
        "recall": recall_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "f2": fbeta_score(y_test, predictions, beta=2),
    }

    print("Test Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    return metrics


def save_and_upload_model(
    model: AdaBoostClassifier,
    metadata: dict,
    repo_id: str,
    token: str,
) -> None:
    """
    Save model and metadata, then upload to HuggingFace.

    Args:
        model: Trained model
        metadata: Model metadata dictionary
        repo_id: HuggingFace model repository ID
        token: HuggingFace API token
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save model
        model_path = os.path.join(tmpdir, "best_model.joblib")
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")

        # Save metadata
        metadata_path = os.path.join(tmpdir, "model_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")

        # Upload to HuggingFace
        api = HfApi()
        print(f"Uploading to {repo_id}...")

        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="best_model.joblib",
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )

        api.upload_file(
            path_or_fileobj=metadata_path,
            path_in_repo="model_metadata.json",
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )

        # Upload model card (README)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        readme_path = os.path.join(script_dir, "MODEL_README.md")
        if os.path.exists(readme_path):
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )
            print("Uploaded model card (README.md)")

        print(f"Successfully uploaded model to https://huggingface.co/{repo_id}")


def main():
    """Main entry point."""
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable not set")
        sys.exit(1)

    try:
        # Load data
        train_df, val_df, test_df = load_splits(DATASET_REPO)

        # Prepare features
        feature_cols = get_feature_columns(train_df, TARGET_COLUMN)
        X_train = train_df[feature_cols]
        y_train = train_df[TARGET_COLUMN]
        X_val = val_df[feature_cols]
        y_val = val_df[TARGET_COLUMN]
        X_test = test_df[feature_cols]
        y_test = test_df[TARGET_COLUMN]

        # Train model
        model = train_model(X_train, y_train)

        # Find optimal threshold
        threshold = find_optimal_threshold(model, X_val, y_val)

        # Evaluate on test set
        metrics = evaluate_model(model, X_test, y_test, threshold)

        # Gate check: verify recall meets minimum requirement
        if metrics["recall"] < RECALL_GATE:
            print(f"RECALL GATE FAILED: {metrics['recall']:.4f} < {RECALL_GATE}")
            print("Model does not meet minimum recall requirement. Aborting upload.")
            sys.exit(1)

        print(f"RECALL GATE PASSED: {metrics['recall']:.4f} >= {RECALL_GATE}")

        # Create metadata
        metadata = {
            "model_name": "AdaBoost (Optuna)",
            "algorithm": "AdaBoost",
            "tuning_method": "Optuna",
            "features": feature_cols,
            "n_features": len(feature_cols),
            "target": TARGET_COLUMN,
            "threshold": threshold,
            "test_metrics": metrics,
            "best_params": {
                "base_max_depth": str(MODEL_PARAMS["base_max_depth"]),
                "n_estimators": str(MODEL_PARAMS["n_estimators"]),
                "learning_rate": str(MODEL_PARAMS["learning_rate"]),
            },
            "training_info": {
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
            },
        }

        # Upload model
        save_and_upload_model(model, metadata, MODEL_REPO, HF_TOKEN)

        print("Training pipeline complete!")

    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
