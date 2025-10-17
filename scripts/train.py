#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AdamW

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.dataset import TranscriptionDataset, load_and_prepare_data
from src.models.bert_model import BertClassifier


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, epoch_num: int, total_epochs: int):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch_num}/{total_epochs}", leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix(train_loss=f"{loss.item():.4f}")
    return total_loss / len(data_loader)


def eval_model(model, data_loader, loss_fn, device, task_type):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            if task_type == "classification":
                preds = torch.argmax(outputs, dim=1)
            else:  # regression
                preds = outputs

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    if task_type == "classification":
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return avg_loss, {"accuracy": acc, "f1": f1}
    else:
        rmse = mean_squared_error(all_labels, all_preds, squared=False)
        return avg_loss, {"rmse": rmse}


def run_training(config, train_indices, val_indices, labels_df, transcript_dir, hparams, fold_num=None):
    """Runs a complete training and validation loop."""
    fold_str = f"fold_{fold_num}" if fold_num is not None else "single_run"
    print(f"\n----- Starting Training: {fold_str} -----")
    print(f"Hyperparameters: {hparams}")

    # DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(config.model.bert_model)
    train_dataset = TranscriptionDataset(
        file_ids=labels_df.iloc[train_indices].index.tolist(), labels=labels_df,
        transcript_dir=transcript_dir, tokenizer=tokenizer, max_length=config.model.max_length, task=config.training.task
    )
    val_dataset = TranscriptionDataset(
        file_ids=labels_df.iloc[val_indices].index.tolist(), labels=labels_df,
        transcript_dir=transcript_dir, tokenizer=tokenizer, max_length=config.model.max_length, task=config.training.task
    )
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size)

    # Model, Optimizer, Scheduler, Loss
    device = torch.device(config.training.device)
    model_hparams = {k: v for k, v in hparams.items() if k in config.model.to_dict()}
    model_cfg = config.model.copy(update=model_hparams)
    model = BertClassifier(model_config=model_cfg, task=config.training.task, model_type=hparams.get("model_type", "simple")).to(device)

    optimizer = AdamW(model.parameters(), lr=hparams['learning_rate'], weight_decay=hparams.get('weight_decay', 0.01))
    epochs = config.cross_validation.max_epochs if fold_num is not None else config.training.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.training.warmup_steps,
        num_training_steps=len(train_loader) * epochs
    )
    loss_fn = torch.nn.CrossEntropyLoss() if config.training.task == "classification" else torch.nn.MSELoss()

    # Training Loop
    best_metric_val = 0 if config.training.eval_metric in ["accuracy", "f1"] else float('inf')
    patience = config.cross_validation.early_stopping_patience if fold_num is not None else config.training.early_stopping_patience
    patience_counter = 0
    output_dir = Path(config.training.output_dir) / fold_str
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            epoch_num=epoch + 1,
            total_epochs=epochs
        )
        val_loss, metrics = eval_model(model, val_loader, loss_fn, device, config.training.task)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Metrics: {metrics}")

        current_metric = metrics[config.training.eval_metric]
        is_better = (current_metric > best_metric_val) if config.training.eval_metric in ["accuracy", "f1"] else (current_metric < best_metric_val)

        if is_better:
            best_metric_val = current_metric
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"-> Saved new best model with {config.training.eval_metric}: {best_metric_val:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return best_metric_val


def main(config_path: str, grid_search: bool):
    config = load_config(config_path)
    labels_df, transcript_dir = load_and_prepare_data(config.data, config.training.task)

    if not config.cross_validation.enabled:
        indices = np.arange(len(labels_df))
        train_indices, val_indices = train_test_split(
            indices, test_size=config.data.val_split, random_state=config.data.seed,
            stratify=labels_df['label'] if config.training.task == "classification" else None
        )
        hparams = {
            "learning_rate": config.training.learning_rate, "dropout": config.model.dropout,
            "model_type": "simple", "mlp_hidden_size": config.model.mlp_hidden_size,
            "weight_decay": config.training.weight_decay
        }
        run_training(config, train_indices, val_indices, labels_df, transcript_dir, hparams)
        return

    # K-Fold Cross-Validation with Hyperparameter Search
    skf = StratifiedKFold(n_splits=config.cross_validation.n_folds, shuffle=True, random_state=config.data.seed)
    hparam_grid = config.cross_validation.hyperparameters.to_dict()
    hparam_keys = list(hparam_grid.keys())
    if grid_search:
        hparam_combinations = [dict(zip(hparam_keys, values)) for values in itertools.product(*hparam_grid.values())]
    else:
        single_combo = {key: (values[0] if isinstance(values, list) else values) for key, values in hparam_grid.items()}
        print("Running with a single hyperparameter combination for quick testing.")
        hparam_combinations = [single_combo]

    cv_results = []

    for fold, (train_indices, val_indices) in enumerate(skf.split(labels_df, labels_df['label'])):
        best_fold_metric = 0 if config.training.eval_metric in ["accuracy", "f1"] else float('inf')
        best_hparams = None

        for hparams in hparam_combinations:
            # Pass a copy of hparams to avoid issues with aliasing
            metric = run_training(config, train_indices, val_indices, labels_df, transcript_dir, hparams.copy(), fold_num=fold + 1)

            is_better = (metric > best_fold_metric) if config.training.eval_metric in ["accuracy", "f1"] else (metric < best_fold_metric)
            if is_better:
                best_fold_metric = metric
                best_hparams = hparams

        cv_results.append({"fold": fold + 1, "best_metric": best_fold_metric, "best_hparams": best_hparams})
        print(f"\nBest for fold {fold + 1}: {config.training.eval_metric} = {best_fold_metric:.4f} with {best_hparams}")

    # Save and print final results
    (Path(config.training.output_dir) / "cv_results.json").write_text(json.dumps(cv_results, indent=4))
    avg_metric = np.mean([r['best_metric'] for r in cv_results])
    print(f"\n----- CV Summary -----\nAverage best {config.training.eval_metric}: {avg_metric:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BERT model with optional cross-validation.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--grid-search", action="store_true", help="Run full hyperparameter grid search instead of a single combination.")
    args = parser.parse_args()
    main(args.config, args.grid_search)
