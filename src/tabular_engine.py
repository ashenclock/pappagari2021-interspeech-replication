import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, root_mean_squared_error

from src.data import get_data_splits

class TabularTrainer:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_set = config.feature_extraction.feature_set

    def load_features(self, is_test=False):
        fname = f"test_{self.feature_set}.csv" if is_test else f"train_{self.feature_set}.csv"
        path = Path(self.config.data.features_root) / fname
        if not path.exists():
            raise FileNotFoundError(f"File feature non trovato: {path}. Esegui prima extract_features.py")
        return pd.read_csv(path)

    def get_model_pipeline(self):
        model_name = self.config.tabular_model.name
        if model_name == 'svm':
            clf = SVC(probability=True, random_state=self.config.seed)
        elif model_name == 'xgboost':
            clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=self.config.seed)
        elif model_name == 'lr':
            clf = LogisticRegression(max_iter=1000, random_state=self.config.seed)
        else:
            raise ValueError(f"Modello tabulare '{model_name}' non supportato.")
        
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', clf)
        ])

    def train(self):
        print(f"--- Inizio Training Tabulare ({self.config.tabular_model.name}) su {self.feature_set} ---")
        
        # 1. Caricamento Dati e Merge con Etichette
        feat_df = self.load_features(is_test=False)
        
        # Usiamo get_data_splits solo per ottenere il dataframe completo delle etichette una volta
        # per fare il merge corretto
        labels_df = pd.read_csv(self.config.data.train_labels)
        # Adattamento nomi colonne come in data.py
        if 'adressfname' in labels_df.columns:
             labels_df = labels_df.rename(columns={'adressfname': 'ID'})

        # Troviamo la colonna diagnosi
        dx_col = next((c for c in ['dx', 'diagnosis', 'Dx'] if c in labels_df.columns), None)
        if dx_col is None: raise ValueError("Colonna diagnosi non trovata nelle etichette.")
        
        labels_df['label'] = labels_df[dx_col].apply(lambda x: 1 if x == 'ad' else 0)
        
        # Merge feature e label
        full_train_df = pd.merge(feat_df, labels_df[['ID', 'label']], on='ID')
        X_all = full_train_df.drop(columns=['ID', 'label'])
        y_all = full_train_df['label'].values
        ids_all = full_train_df['ID'].values

        # 2. GridSearch Globale per trovare i migliori iperparametri
        print("Esecuzione GridSearchCV per trovare i migliori iperparametri...")
        pipeline = self.get_model_pipeline()
        # NUOVE RIGHE CORRETTE
        model_name = self.config.tabular_model.name
        # Usiamo getattr per accedere dinamicamente alla proprietà (es. .xgboost) e .to_dict() per convertirla
        grid_dict = getattr(self.config.tabular_model.grids, model_name).to_dict()
        param_grid = {f'clf__{k}': v for k, v in grid_dict.items()}
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.config.seed)
        grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
        grid.fit(X_all, y_all)
        
        best_params = grid.best_params_
        print(f"Migliori parametri trovati: {best_params}")
        print(f"Best CV Score: {grid.best_score_:.4f}")

        # 3. Addestramento Ensemble dei 10 Fold con i migliori parametri
        print("\nAddestramento dei 10 modelli per l'ensemble...")
        skf = StratifiedKFold(n_splits=self.config.k_folds, shuffle=True, random_state=self.config.seed)
        
        oof_preds = np.zeros((len(full_train_df), 2)) # Per salvare le probabilità OOF
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
            print(f"Training Fold {fold}...")
            fold_dir = self.output_dir / f"fold_{fold}"
            fold_dir.mkdir(exist_ok=True)
            
            X_train, y_train = X_all.iloc[train_idx], y_all[train_idx]
            X_val, y_val = X_all.iloc[val_idx], y_all[val_idx]
            
            # Creiamo una nuova pipeline con i parametri migliori fissati
            model = self.get_model_pipeline()
            model.set_params(**best_params)
            model.fit(X_train, y_train)
            
            # Salvataggio modello
            joblib.dump(model, fold_dir / "tabular_model.pkl")
            
            # Predizioni OOF per questo fold
            if hasattr(model, "predict_proba"):
                oof_preds[val_idx] = model.predict_proba(X_val)
            else: # fallback per modelli che non supportano probabilità
                 pred = model.predict(X_val)
                 oof_preds[val_idx, 1] = pred
                 oof_preds[val_idx, 0] = 1 - pred

        # Salviamo le predizioni OOF complete
        oof_df = pd.DataFrame({
            'ID': ids_all,
            'score': oof_preds[:, 1], # Probabilità classe 1 (AD)
            'label': y_all
        })
        oof_df.to_csv(self.output_dir / "oof_predictions.csv", index=False)
        print(f"Training completato. Modelli e OOF salvati in {self.output_dir}")

class TabularPredictor:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.feature_set = config.feature_extraction.feature_set

    def load_features(self, is_test=True):
        fname = f"test_{self.feature_set}.csv" if is_test else f"train_{self.feature_set}.csv"
        path = Path(self.config.data.features_root) / fname
        return pd.read_csv(path)

    def predict(self):
        print(f"--- Inizio Predizione Tabulare su Test Set ---")
        test_df = self.load_features(is_test=True)
        X_test = test_df.drop(columns=['ID'])
        ids_test = test_df['ID'].values
        
        fold_scores = []
        for fold in range(self.config.k_folds):
            model_path = self.output_dir / f"fold_{fold}" / "tabular_model.pkl"
            if not model_path.exists():
                print(f"ATTENZIONE: Modello fold {fold} non trovato. Salto.")
                continue
            
            model = joblib.load(model_path)
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(X_test)[:, 1]
            else:
                scores = model.predict(X_test)
            fold_scores.append(scores)
            
        if not fold_scores:
            raise ValueError("Nessun modello trovato per la predizione.")
            
        # Ensemble averaging
        avg_scores = np.mean(fold_scores, axis=0)
        
        results_df = pd.DataFrame({
            'ID': ids_test,
            'score': avg_scores,
            'prediction': (avg_scores >= 0.5).astype(int)
        })
        
        out_file = self.output_dir / "predictions.csv"
        results_df.to_csv(out_file, index=False)
        print(f"Predizioni salvate in {out_file}")