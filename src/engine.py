import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, root_mean_squared_error
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.models import build_model
from src.utils import (
    clear_memory, find_diagnosis_column, map_label,
    compute_classification_metrics, load_labels_df, LABEL_MAPPING,
)


def _move_batch_to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def train_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        batch = _move_batch_to_device(batch, device)
        labels = batch.pop('labels')
        outputs = model(batch)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_epoch(model, loader, loss_fn, device, task, metric_name):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = _move_batch_to_device(batch, device)
            labels = batch.pop('labels')
            outputs = model(batch)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            if task == 'classification':
                preds = torch.argmax(outputs, dim=1)
            else:
                preds = outputs

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)

    if task == 'classification':
        if metric_name == 'accuracy':
            metric = accuracy_score(all_labels, all_preds)
        elif metric_name == 'f1':
            metric = f1_score(all_labels, all_preds, average='weighted')
        elif metric_name == 'loss':
            return avg_loss, avg_loss
        else:
            raise ValueError(f"Metric '{metric_name}' not supported for classification.")
    else:
        metric = root_mean_squared_error(all_labels, all_preds)

    return avg_loss, metric


class Trainer:
    def __init__(self, config, train_loader, val_loader, fold):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.fold = fold
        self.device = torch.device(config.device)
        self.output_path = Path(config.output_dir) / f"fold_{fold}"
        self.output_path.mkdir(parents=True, exist_ok=True)

    def train(self):
        clear_memory()
        config = self.config
        model = build_model(config).to(self.device)

        if config.modality == 'text':
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)

        num_training_steps = len(self.train_loader) * config.training.epochs
        if config.modality == 'text':
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(num_training_steps * config.training.warmup_ratio), num_training_steps=num_training_steps)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)

        loss_fn = nn.CrossEntropyLoss() if config.task == 'classification' else nn.MSELoss()
        higher_is_better = config.training.eval_metric in ['accuracy', 'f1']
        best_metric = -np.inf if higher_is_better else np.inf
        patience_counter = 0

        print(f"--- Training Fold {self.fold} ---")
        for epoch in range(config.training.epochs):
            train_loss = train_epoch(model, self.train_loader, optimizer, scheduler, loss_fn, self.device)
            val_loss, val_metric = evaluate_epoch(model, self.val_loader, loss_fn, self.device, config.task, config.training.eval_metric)

            print(f"Epoch {epoch+1}/{config.training.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val {config.training.eval_metric}: {val_metric:.4f}")

            is_better = (val_metric > best_metric) if higher_is_better else (val_metric < best_metric)
            if is_better:
                best_metric = val_metric
                torch.save(model.state_dict(), self.output_path / "best_model.pt")
                print(f"  -> Saved. Best {config.training.eval_metric}: {best_metric:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.training.early_stopping_patience:
                print("  -> Early stopping.")
                break

        print(f"--- End Fold {self.fold} ---")


class Predictor:
    def __init__(self, config, test_loader):
        self.config = config
        self.test_loader = test_loader
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)

    def predict(self):
        all_fold_scores = []

        for fold in range(self.config.k_folds):
            model_path = self.output_dir / f"fold_{fold}" / "best_model.pt"
            if not model_path.exists():
                print(f"WARNING: Model for fold {fold} not found. Skipping.")
                continue

            model = build_model(self.config)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()

            fold_scores = []
            test_ids = []

            with torch.no_grad():
                for batch in tqdm(self.test_loader, desc=f"Predicting Fold {fold}", leave=False):
                    ids = batch.pop("id")
                    test_ids.extend(ids)
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = model(batch)

                    if self.config.task == 'classification':
                        scores = torch.softmax(outputs, dim=1)[:, 1]
                    else:
                        scores = outputs
                    fold_scores.extend(scores.cpu().numpy())

            all_fold_scores.append(fold_scores)

        mean_scores = np.mean(all_fold_scores, axis=0)
        results_df = pd.DataFrame({'ID': test_ids[:len(mean_scores)], 'score': mean_scores})

        if self.config.task == 'classification':
            results_df['prediction'] = (results_df['score'] >= 0.5).astype(int)
        else:
            results_df['prediction'] = results_df['score']

        output_file = self.output_dir / "predictions.csv"
        results_df[['ID', 'prediction']].to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.predictions_path = self.output_dir / "predictions.csv"
        if config.task == 'classification':
            self.labels_path = Path(config.data.test_task1_labels)
        else:
            self.labels_path = Path(config.data.test_task2_labels)

    def evaluate(self):
        if not self.predictions_path.exists():
            raise FileNotFoundError(f"Predictions not found: {self.predictions_path}")

        preds_df = pd.read_csv(self.predictions_path)
        labels_df = load_labels_df(self.labels_path)
        merged_df = pd.merge(preds_df, labels_df, on="ID")

        if merged_df.empty:
            print("ERROR: No matching IDs between predictions and labels.")
            return

        y_pred = merged_df['prediction']
        report_lines = []

        report_lines.append(f"--- Evaluation Report: {self.output_dir.name} ---")
        report_lines.append("=" * 70)
        report_lines.append(f"Task: {self.config.task}")

        if hasattr(self.config, 'tabular_model'):
            report_lines.append(f"Mode: Tabular")
            report_lines.append(f"  Classifier: {self.config.tabular_model.name}")
            if hasattr(self.config, 'embedding_extraction') and self.config.embedding_extraction.name == self.config.feature_extraction.feature_set:
                report_lines.append(f"  Features: Embeddings from '{self.config.embedding_extraction.model_id}'")
            else:
                report_lines.append(f"  Features: '{self.config.feature_extraction.feature_set}'")
        elif hasattr(self.config, 'modality'):
            report_lines.append(f"Mode: Deep Learning ({self.config.modality})")
            if self.config.modality == 'text':
                report_lines.append(f"  Text model: {self.config.model.text.name}")
                report_lines.append(f"  ASR transcriptions: {self.config.transcription_model_for_training}")
            elif self.config.modality == 'audio':
                report_lines.append(f"  Audio model: {self.config.model.audio.name}")
                report_lines.append(f"  Pretrained: {self.config.model.audio.pretrained}")

        report_lines.append("\n" + ("-" * 30))

        if self.config.task == 'classification':
            dx_col = find_diagnosis_column(merged_df)
            if dx_col is None:
                raise ValueError(f"Diagnosis column not found. Available: {merged_df.columns.tolist()}")

            y_true = merged_df[dx_col].map(LABEL_MAPPING)
            metrics = compute_classification_metrics(y_true, y_pred)

            report_lines.append("\n--- Main Metrics ---")
            report_lines.append(f"Accuracy           : {metrics['accuracy']:.4f}")
            report_lines.append(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
            report_lines.append(f"F1-Score (Macro)   : {metrics['f1_macro']:.4f}")
            report_lines.append(f"Sensitivity (Recall): {metrics['sensitivity']:.4f}")
            report_lines.append(f"Specificity        : {metrics['specificity']:.4f}")
            report_lines.append("\n--- Classification Report ---")
            report_lines.append(metrics['classification_report'])
            report_lines.append("\n--- Confusion Matrix ---")
            report_lines.append(str(metrics['confusion_matrix']))
        else:
            y_true = merged_df['MMSE']
            rmse = root_mean_squared_error(y_true, y_pred)
            report_lines.append("\n--- Regression Metrics ---")
            report_lines.append(f"RMSE: {rmse:.4f}")

        report_lines.append("\n" + "=" * 70)
        full_report = "\n".join(report_lines)
        print("\n" + full_report)

        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        report_filepath = reports_dir / f"{self.output_dir.name}_report.txt"
        report_filepath.write_text(full_report, encoding="utf-8")
        print(f"\nReport saved to: {report_filepath}")
