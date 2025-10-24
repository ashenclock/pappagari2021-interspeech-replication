import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, root_mean_squared_error, classification_report, confusion_matrix

from src.models import build_model
from src.utils import clear_memory

def train_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        
        # --- FIX: QUESTA RIGA DEVE ESSERE ATTIVA ---
        # Sposta tutti i tensori del batch sul device corretto (es. GPU)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
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
            # --- FIX: ANCHE QUESTA RIGA DEVE ESSERE ATTIVA ---
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch.pop('labels')
            outputs = model(batch)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            if task == 'classification':
                preds = torch.argmax(outputs, dim=1)
            else: # regression
                preds = outputs
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    
    if task == 'classification':
        if metric_name == 'accuracy':
            metric = accuracy_score(all_labels, all_preds)
        elif metric_name == 'f1':
            metric = f1_score(all_labels, all_preds, average='weighted')
        else:
            raise ValueError(f"Metrica '{metric_name}' non supportata per la classificazione.")
    else: # regression
        metric = root_mean_squared_error(all_labels, all_preds) # RMSE
        
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
        model = build_model(self.config).to(self.device)
        if self.config.modality == 'text':
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.training.learning_rate, weight_decay=self.config.training.weight_decay)
        else: # audio
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.training.learning_rate, weight_decay=self.config.training.weight_decay)
        
        num_training_steps = len(self.train_loader) * self.config.training.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(num_training_steps * self.config.training.warmup_ratio), num_training_steps=num_training_steps)
        
        loss_fn = nn.CrossEntropyLoss() if self.config.task == 'classification' else nn.MSELoss()

        best_metric = -np.inf if self.config.training.eval_metric in ['accuracy', 'f1'] else np.inf
        patience_counter = 0

        print(f"--- Inizio Training Fold {self.fold} ---")
        for epoch in range(self.config.training.epochs):
            train_loss = train_epoch(model, self.train_loader, optimizer, scheduler, loss_fn, self.device)
            val_loss, val_metric = evaluate_epoch(model, self.val_loader, loss_fn, self.device, self.config.task, self.config.training.eval_metric)
            
            print(f"Epoch {epoch+1}/{self.config.training.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val {self.config.training.eval_metric}: {val_metric:.4f}")

            is_better = (val_metric > best_metric) if self.config.training.eval_metric in ['accuracy', 'f1'] else (val_metric < best_metric)
            
            if is_better:
                best_metric = val_metric
                torch.save(model.state_dict(), self.output_path / "best_model.pt")
                print(f"-> Modello salvato. Nuova metrica migliore: {best_metric:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.training.early_stopping_patience:
                print("-> Early stopping attivato.")
                break
        print(f"--- Fine Training Fold {self.fold} ---")

class Predictor:
    def __init__(self, config, test_loader):
        self.config = config
        self.test_loader = test_loader
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)
    
    def predict(self):
        all_fold_scores = []
        
        for fold in range(self.config.k_folds):
            print(f"--- Inferenza con Fold {fold} ---")
            model_path = self.output_dir / f"fold_{fold}" / "best_model.pt"
            if not model_path.exists():
                print(f"ATTENZIONE: Modello per il fold {fold} non trovato in {model_path}. Salto.")
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
                        scores = torch.softmax(outputs, dim=1)[:, 1] # Probabilità della classe positiva
                    else: # regression
                        scores = outputs
                    
                    fold_scores.extend(scores.cpu().numpy())
            
            all_fold_scores.append(fold_scores)
            
        # Ensembling: media degli score/probabilità
        mean_scores = np.mean(all_fold_scores, axis=0)
        
        # Crea DataFrame finale
        results_df = pd.DataFrame({'ID': test_ids[:len(mean_scores)], 'score': mean_scores})
        
        if self.config.task == 'classification':
            results_df['prediction'] = (results_df['score'] >= 0.5).astype(int)
        else:
            results_df['prediction'] = results_df['score']
            
        output_file = self.output_dir / "predictions.csv"
        results_df[['ID', 'prediction']].to_csv(output_file, index=False)
        print(f"\nPredizioni salvate in {output_file}")
        
class Evaluator:
    def __init__(self, config):
        self.config = config
        self.predictions_path = Path(config.output_dir) / "predictions.csv"
        if config.task == 'classification':
            self.labels_path = Path(config.data.test_task1_labels)
        else:
            self.labels_path = Path(config.data.test_task2_labels)
    
    def evaluate(self):
        if not self.predictions_path.exists():
            raise FileNotFoundError(f"File di predizioni non trovato: {self.predictions_path}")
        
        preds_df = pd.read_csv(self.predictions_path)
        labels_df = pd.read_csv(self.labels_path)
        
        # Assicuriamoci che la colonna ID si chiami allo stesso modo per il merge
        # A volte i CSV hanno nomi leggermente diversi (es. 'id', 'ID', 'adressfname')
        if 'ID' not in labels_df.columns:
             # Cerca una colonna che potrebbe essere l'ID
             possible_id_cols = [col for col in labels_df.columns if 'id' in col.lower() or 'name' in col.lower()]
             if possible_id_cols:
                 labels_df = labels_df.rename(columns={possible_id_cols[0]: 'ID'})
             else:
                 raise ValueError("Non riesco a trovare la colonna ID nel file delle etichette.")

        merged_df = pd.merge(preds_df, labels_df, on="ID")
        
        y_pred = merged_df['prediction']
        
        if self.config.task == 'classification':
            # --- FIX CRUCIALE: MAPPING DELLE ETICHETTE ---
            # Convertiamo le stringhe 'Control'/'ProbableAD' in 0/1
            # Adatta questo dizionario se le tue etichette sono diverse (es. 'CN'/'AD')
            label_mapping = {'Control': 0, 'ProbableAD': 1, 'CN': 0, 'AD': 1}
            
            # Usa la colonna corretta per la diagnosi. Potrebbe chiamarsi 'Dx', 'diagnosis', ecc.
            # Cerchiamo di essere flessibili.
            dx_col = None
            for col in ['Dx', 'diagnosis', 'label']:
                if col in merged_df.columns:
                    dx_col = col
                    break
            
            if dx_col is None:
                 raise ValueError(f"Colonna diagnosi non trovata. Colonne disponibili: {merged_df.columns}")

            # Applica il mapping. Se un valore non è nel dizionario, potrebbe dare errore o NaN.
            try:
                y_true = merged_df[dx_col].map(label_mapping).astype(int)
            except ValueError as e:
                 print(f"Errore durante il mapping delle etichette. Valori unici trovati in {dx_col}: {merged_df[dx_col].unique()}")
                 raise e
            # ---------------------------------------------

            print("----- Valutazione Classificazione -----")
            print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
            print(f"F1-Score (Weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=['CN', 'AD']))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_true, y_pred))
            print("------------------------------------")
        else: # regression
            y_true = merged_df['MMSE']
            print("----- Valutazione Regressione -----")
            rmse = root_mean_squared_error(y_true, y_pred)
            print(f"RMSE: {rmse:.4f}")
            print("------------------------------------")