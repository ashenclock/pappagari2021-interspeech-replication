#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.models.bert_model import BertClassifier


class TestDataset(torch.utils.data.Dataset):
    """Dataset for test transcripts which are not in ad/ or cn/ subfolders."""
    def __init__(self, file_paths: list[Path], tokenizer, max_length: int):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            text = file_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            print(f"Warning: Transcript not found for {file_path.name}, using empty string.")
            text = ""

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "file_id": file_path.stem
        }


def find_best_model_paths(output_dir: Path, cv_results_path: Path) -> list[tuple[Path, dict]]:
    """Finds the paths to the best model from each fold and their hparams."""
    if not cv_results_path.exists():
        # Fallback if no CV was run: check for a single_run model
        single_run_path = output_dir / "single_run" / "best_model.pt"
        if single_run_path.exists():
            print("Found model from a single run.")
            # We don't have hparams, so we'll use config defaults
            return [(single_run_path, {})]
        raise FileNotFoundError(f"CV results not found at {cv_results_path} and no single_run model found.")

    with open(cv_results_path, 'r') as f:
        cv_results = json.load(f)

    model_paths = []
    for fold_result in cv_results:
        fold = fold_result['fold']
        hparams = fold_result['best_hparams']
        model_path = output_dir / f"fold_{fold}" / "best_model.pt"
        if model_path.exists():
            model_paths.append((model_path, hparams))
        else:
            print(f"Warning: Best model for fold {fold} not found at {model_path}")

    return model_paths


def predict(config_path: str, test_audio_dir: str, threshold: float):
    """
    Runs inference using an ensemble of models from cross-validation folds.
    """
    config = load_config(config_path)
    device = torch.device(config.training.device)
    output_dir = Path(config.training.output_dir)

    # --- 1. Find Models ---
    cv_results_path = output_dir / "cv_results.json"
    model_info = find_best_model_paths(output_dir, cv_results_path)
    if not model_info:
        print("No models found for prediction. Exiting.")
        return

    print(f"Found {len(model_info)} models for ensembling.")

    # --- 2. Prepare Data ---
    tokenizer = AutoTokenizer.from_pretrained(config.model.bert_model)

    # The test transcripts are in a flat directory, unlike train transcripts
    # We need to find the correct test transcript folder.
    # The structure is transcripts_test/<model_name>/
    test_transcript_dir = Path("transcripts_test") / config.data.transcription_model
    if not test_transcript_dir.exists():
        raise FileNotFoundError(f"Test transcript directory not found at {test_transcript_dir}")

    # Get the list of test files from the provided audio directory
    test_files = sorted([p for p in Path(test_audio_dir).rglob("*.wav")])
    test_file_ids = [p.stem for p in test_files]

    # Create paths to the corresponding transcripts
    test_transcript_paths = [test_transcript_dir / f"{file_id}.txt" for file_id in test_file_ids]

    test_dataset = TestDataset(
        file_paths=test_transcript_paths,
        tokenizer=tokenizer,
        max_length=config.model.max_length
    )
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size)

    # --- 3. Run Inference with Ensemble ---
    all_fold_probs = []
    file_ids_ordered = []

    for i, (model_path, hparams) in enumerate(model_info):
        print(f"Loading model: {model_path}")
        # Use default hparams from config if not available from CV results
        if not hparams:
            hparams = {
                "learning_rate": config.training.learning_rate, "dropout": config.model.dropout,
                "model_type": "simple", "mlp_hidden_size": config.model.mlp_hidden_size,
                "weight_decay": config.training.weight_decay
            }

        model_cfg = config.model.copy(update=hparams)
        model = BertClassifier(
            model_config=model_cfg,
            task=config.training.task,
            model_type=hparams.get("model_type", "simple")
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        fold_probs = []
        batch_file_ids = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Predicting with {model_path.parent.name}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                logits = model(input_ids, attention_mask)

                if config.training.task == "classification":
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    fold_probs.extend(probs.cpu().numpy())
                else:  # Regression
                    fold_probs.extend(logits.cpu().numpy())

                if i == 0:  # Only need to get the order once
                    batch_file_ids.extend(batch["file_id"])

        all_fold_probs.append(fold_probs)
        if i == 0:
            file_ids_ordered = batch_file_ids

    # --- 4. Ensemble and Save Results ---
    mean_probs = np.mean(all_fold_probs, axis=0)

    if config.training.task == "classification":
        predictions = (mean_probs >= threshold).astype(int)
        results_df = pd.DataFrame({'ID': file_ids_ordered, 'prediction': predictions, 'probability': mean_probs})
    else:  # Regression
        predictions = mean_probs
        results_df = pd.DataFrame({'ID': file_ids_ordered, 'prediction': predictions})

    # Ensure the order matches the official label file
    label_file = config.data.test_task1_labels if config.training.task == "classification" else config.data.test_task2_labels
    test_labels_df = pd.read_csv(label_file)
    # The IDs in the CSV might be like 'adrsdt1', but our stems might be different.
    # Let's be robust and use the IDs from the label file directly.
    results_df = results_df.set_index('ID').reindex(test_labels_df['ID']).reset_index()

    output_filename = "predictions_task1.csv" if config.training.task == "classification" else "predictions_task2.csv"
    results_df[['ID', 'prediction']].to_csv(output_filename, index=False)

    print(f"\nPredictions saved to {output_filename}")
    if config.training.task == "classification":
        print("Prediction distribution:")
        print(results_df['prediction'].value_counts())


def main():
    parser = argparse.ArgumentParser(description="Run ensemble prediction on the test set.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--test-audio-dir", required=True, help="Directory of the original test audio files (e.g., 'Adresso21/ADReSSo21-diagnosis-test/ADReSSo21/diagnosis/test-dist/audio').")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for classification.")
    args = parser.parse_args()
    predict(args.config, args.test_audio_dir, args.threshold)


if __name__ == "__main__":
    main()
