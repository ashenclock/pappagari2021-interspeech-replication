#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config

def evaluate_classification(predictions_file: Path, labels_file: Path):
    """Calculates and prints classification metrics."""
    preds_df = pd.read_csv(predictions_file)
    labels_df = pd.read_csv(labels_file)

    # Normalize label columns (Dx, dx, etc.)
    labels_df = labels_df.rename(columns=lambda c: c.strip())
    if "diagnosis" not in labels_df.columns:
        if "Dx" in labels_df.columns:
            labels_df = labels_df.rename(columns={"Dx": "diagnosis"})
        elif "dx" in labels_df.columns:
            labels_df = labels_df.rename(columns={"dx": "diagnosis"})
    
    # Merge to ensure correct order and alignment
    merged = pd.merge(preds_df, labels_df, on="ID", suffixes=('_pred', '_true'))
    
    if "diagnosis" not in merged.columns:
        available = ', '.join(merged.columns)
        raise KeyError(f"Column 'diagnosis' not found after merging. Columns available: {available}")

    # Map textual labels to numeric where needed
    mapping = {
        "control": 0,
        "probablead": 1,
        "probable_ad": 1,
        "ad": 1,
        "cn": 0,
        "0": 0,
        "1": 1,
    }

    y_true_raw = merged['diagnosis'].astype(str).str.strip()
    y_true = y_true_raw.str.lower().map(mapping)
    if y_true.isnull().any():
        missing = ', '.join(sorted(y_true_raw[y_true.isnull()].unique()))
        raise ValueError(f"Some diagnosis labels could not be mapped to numeric classes: {missing}")
    y_pred = merged['prediction']
    
    if y_true.isnull().any() or y_pred.isnull().any():
        raise ValueError("Mismatch between prediction and label files. Some IDs may be missing.")

    print("----- Classification Evaluation -----")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-Score (Weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['CN', 'AD']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("------------------------------------")

def evaluate_regression(predictions_file: Path, labels_file: Path):
    """Calculates and prints regression metrics."""
    preds_df = pd.read_csv(predictions_file)
    labels_df = pd.read_csv(labels_file)
    
    merged = pd.merge(preds_df, labels_df, on="ID", suffixes=('_pred', '_true'))
    
    y_true = merged['mmse']
    y_pred = merged['prediction']

    if y_true.isnull().any() or y_pred.isnull().any():
        raise ValueError("Mismatch between prediction and label files. Some IDs may be missing.")

    print("----- Regression Evaluation -----")
    print(f"RMSE (Root Mean Squared Error): {mean_squared_error(y_true, y_pred, squared=False):.4f}")
    print(f"MAE (Mean Absolute Error): {mean_absolute_error(y_true, y_pred):.4f}")
    print("------------------------------------")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions against ground truth labels.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--predictions", type=str, default="predictions_task1.csv", help="Path to the generated predictions CSV file.")
    args = parser.parse_args()

    config = load_config(args.config)
    predictions_path = Path(args.predictions)
    
    if not predictions_path.exists():
        print(f"Error: Predictions file not found at '{predictions_path}'")
        sys.exit(1)

    if config.training.task == "classification":
        labels_path = Path(config.data.test_task1_labels)
        if not labels_path.exists():
            print(f"Error: Test labels file not found at '{labels_path}'")
            sys.exit(1)
        evaluate_classification(predictions_path, labels_path)
    elif config.training.task == "regression":
        labels_path = Path(config.data.test_task2_labels)
        if not labels_path.exists():
            print(f"Error: Test labels file not found at '{labels_path}'")
            sys.exit(1)
        evaluate_regression(predictions_path, labels_path)

if __name__ == "__main__":
    main()
