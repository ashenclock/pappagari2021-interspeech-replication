# scripts/find_best_fuser.py

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import warnings

# Aggiungi la root del progetto al path di Python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importa i modelli e gli strumenti necessari
from sklearn.model_selection import StratifiedKFold, GridSearchCV
# StandardScaler non è più importato
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.config import load_config

# Ignora i warning di convergenza che possono apparire durante la grid search
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def evaluate_predictions(submission_df: pd.DataFrame, config):
    """Valuta le predizioni finali e stampa i report."""
    print("\n--- Valutazione del Fusore Campione sul Test Set ---")
    labels_path = Path(config.data.test_task1_labels)
    if not labels_path.exists(): print(f"ERRORE: File etichette non trovato: {labels_path}"); return
    labels_df = pd.read_csv(labels_path)
    id_col = next((c for c in labels_df.columns if 'id' in c.lower()), None)
    labels_df = labels_df.rename(columns={id_col: 'ID'})
    merged_df = pd.merge(submission_df, labels_df, on="ID")
    if merged_df.empty: print("ERRORE: Nessun ID corrispondente"); return
    
    y_pred = merged_df['prediction']
    dx_col = next((c for c in ['Dx', 'diagnosis', 'label'] if c in merged_df.columns), None)
    y_true = merged_df[dx_col].map({'Control': 0, 'ProbableAD': 1, 'CN': 0, 'AD': 1})
    
    print(f"\nAccuracy Finale: {accuracy_score(y_true, y_pred):.4f}\n")
    print("Classification Report:"); print(classification_report(y_true, y_pred, target_names=['CN', 'AD']))
    print("\nConfusion Matrix:"); print(confusion_matrix(y_true, y_pred))

def main():
    parser = argparse.ArgumentParser(description="Esegue una Grid Search con CV per trovare il miglior modello di fusione.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Percorso al file di configurazione.")
    parser.add_argument("--output_file", type=str, default="best_fused_submission.csv", help="File di output per le predizioni del miglior modello.")
    parser.add_argument("--evaluate", action="store_true", help="Valuta le predizioni finali sul test set.")
    args = parser.parse_args()

    config = load_config(args.config)
    model_dirs = [Path(p['output_dir']) for p in config.score_generation_models]

    # --- PASSO 1: CARICAMENTO E PREPARAZIONE DEI DATI (SCORE OOF E TEST) ---
    print("--- PASSO 1: Caricamento e preparazione degli score dei modelli base ---")
    
    # Carica e unisci i dati di training (OOF)
    oof_dfs = [pd.read_csv(d / "oof_predictions.csv").rename(columns={'score': f'score_{d.name}'}) for d in model_dirs]
    train_df = oof_dfs[0]
    for i in range(1, len(oof_dfs)):
        train_df = pd.merge(train_df, oof_dfs[i][['ID', f'score_{model_dirs[i].name}']], on='ID')

    dx_col = next((c for c in ['dx', 'diagnosis'] if c in train_df.columns), None)
    train_df['label'] = train_df[dx_col].map({'cn': 0, 'ad': 1})
    X_train = train_df.filter(like='score_')
    y_train = train_df['label']

    # Carica e unisci i dati di test
    test_dfs = [pd.read_csv(d / "test_predictions.csv").rename(columns={'score': f'score_{d.name}'}) for d in model_dirs]
    test_df = test_dfs[0]
    for i in range(1, len(test_dfs)):
        test_df = pd.merge(test_df, test_dfs[i], on='ID')
    X_test = test_df.drop(columns=['ID'])

    print(f"Dati di training (OOF) pronti: {X_train.shape[0]} campioni, {X_train.shape[1]} feature (modelli).")
    print(f"Dati di test pronti: {X_test.shape[0]} campioni, {X_test.shape[1]} feature.")

    print("\n--- PASSO 2: Definizione dei modelli e delle griglie di iperparametri ---")

    models_and_grids = [
        {
            "name": "Logistic Regression",
            "estimator": LogisticRegression(random_state=config.seed, max_iter=2000),
            "grid": {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'saga'],
                'penalty': ['l1', 'l2']
            }
        },
        {
            "name": "Support Vector Machine (SVM)",
            "estimator": SVC(probability=True, random_state=config.seed),
            "grid": {
                'C': [0.1, 1, 10, 100],
                'gamma': [1, 0.1, 0.01, 'scale', 'auto'],
                'kernel': ['rbf', 'linear']
            }
        },
        {
            "name": "XGBoost",
            # --- ECCO LA CORREZIONE ---
            "estimator": XGBClassifier(random_state=config.seed, eval_metric='logloss'),
            # --- FINE DELLA CORREZIONE ---
            "grid": {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.7, 0.8, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0]
            }
        }
    ]

    # --- PASSO 3: ESECUZIONE DELLA GRID SEARCH CON 10-FOLD CV ---
    print("\n--- PASSO 3: Avvio della Grid Search... Questo potrebbe richiedere del tempo. ---")
    
    cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=config.seed)
    
    best_overall_score = -1
    best_overall_model = None
    best_model_name = ""

    for model_info in models_and_grids:
        print(f"\n>>> Ricerca per: {model_info['name']} <<<")
        grid_search = GridSearchCV(
            estimator=model_info['estimator'],
            param_grid=model_info['grid'],
            cv=cv_strategy,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        # --- MODIFICA: Uso diretto di X_train e y_train, senza _scaled ---
        print(X_train.shape,X_train,X_train.head())
        grid_search.fit(X_train, y_train)

        print(f"Risultati per {model_info['name']}:")
        print(f"  - Miglior score CV (Accuracy): {grid_search.best_score_:.4f}")
        print(f"  - Migliori iperparametri: {grid_search.best_params_}")

        if grid_search.best_score_ > best_overall_score:
            best_overall_score = grid_search.best_score_
            best_overall_model = grid_search.best_estimator_
            best_model_name = model_info['name']

    # --- PASSO 4: PREDIZIONE FINALE CON IL MODELLO CAMPIONE ---
    print("\n--- PASSO 4: Trovato il modello campione! Addestramento finale e predizione. ---")
    print(f"Modello Campione: {best_model_name}")
    print(f"Score CV stimato: {best_overall_score:.4f}")
    print(f"Iperparametri: {best_overall_model.get_params()}")
    
    # --- MODIFICA: Uso diretto di X_test, senza _scaled ---
    final_predictions = best_overall_model.predict(X_test)
    
    submission_df = pd.DataFrame({
        'ID': test_df['ID'],
        'prediction': final_predictions
    })
    
    submission_df.to_csv(args.output_file, index=False)
    print(f"\nPredizioni finali del modello campione salvate in: {args.output_file}")

    if args.evaluate:
        evaluate_predictions(submission_df, config)


if __name__ == "__main__":
    main()