import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import warnings
import json

# Aggiungi la root del progetto al path di Python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importa i modelli e gli strumenti necessari
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from src.config import load_config

# Ignora i warning
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def evaluate_and_save_report(
    submission_df: pd.DataFrame, 
    config, 
    model_name: str, 
    best_params: dict, # <- Questo ora conterrà solo i parametri puliti
    best_cv_score: float,
    candidate_fusers: list,
    base_asr_models: list
):
    """
    Valuta le predizioni, stampa i report e salva un file dettagliato.
    """
    # --- 1. Calcoli di valutazione ---
    labels_path = Path(config.data.test_task1_labels)
    if not labels_path.exists():
        print(f"ERRORE: File etichette non trovato: {labels_path}"); return
    labels_df = pd.read_csv(labels_path)
    id_col = next((c for c in labels_df.columns if 'id' in c.lower()), None)
    labels_df = labels_df.rename(columns={id_col: 'ID'})
    merged_df = pd.merge(submission_df, labels_df, on="ID")
    if merged_df.empty:
        print("ERRORE: Nessun ID corrispondente tra predizioni ed etichette."); return
    
    y_pred = merged_df['prediction']
    dx_col = next((c for c in ['Dx', 'diagnosis', 'label'] if c in merged_df.columns), None)
    y_true = merged_df[dx_col].map({'Control': 0, 'ProbableAD': 1, 'CN': 0, 'AD': 1})
    cm = confusion_matrix(y_true, y_pred)
    if len(cm.ravel()) == 4:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sensitivity, specificity = float('nan'), float('nan')

    # --- 2. Costruzione della stringa del report ---
    report_lines = []
    report_lines.append(f"--- Report di Valutazione per Fusione di Modelli ---")
    report_lines.append("="*70)
    report_lines.append(f"\nModelli di Base Fusi (ASR): {', '.join(base_asr_models)}")
    report_lines.append(f"Modelli di Fusione Candidati: {', '.join(candidate_fusers)}")
    
    report_lines.append(f"\n--- Modello Campione Selezionato: '{model_name}' ---")
    report_lines.append(f"Miglior Score CV (Accuracy): {best_cv_score:.4f}")
    report_lines.append("Iperparametri Ottimali:")
    report_lines.append(json.dumps(best_params, indent=4))
    
    report_lines.append("\n--- Valutazione Finale del Modello Campione sul Test Set ---")
    report_lines.append(f"Accuracy           : {accuracy_score(y_true, y_pred):.4f}")
    report_lines.append(f"F1-Score (Weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")
    report_lines.append(f"Sensitivity (Recall): {sensitivity:.4f}")
    report_lines.append(f"Specificity        : {specificity:.4f}")

    report_lines.append("\n--- Classification Report Dettagliato ---")
    report_lines.append(classification_report(y_true, y_pred, target_names=['CN', 'AD']))

    report_lines.append("\n--- Confusion Matrix ---")
    report_lines.append(str(cm))
    report_lines.append("\n" + "="*70)
    
    full_report_string = "\n".join(report_lines)
    
    # --- 3. Stampa del report a console ---
    print("\n" + full_report_string)
    
    # --- 4. Salvataggio del report su file ---
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    fused_models_str = "_".join(sorted(base_asr_models))
    report_filepath = reports_dir / f"{safe_model_name}_{fused_models_str}_report.txt"
    
    with open(report_filepath, "w", encoding="utf-8") as f:
        f.write(full_report_string)
        
    print(f"\nReport dettagliato salvato in: {report_filepath}")


def main():
    parser = argparse.ArgumentParser(description="Esegue una Grid Search con CV per trovare il miglior modello di fusione.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Percorso al file di configurazione.")
    parser.add_argument("--output_file", type=str, default="best_fused_submission.csv", help="File di output.")
    parser.add_argument("--evaluate", action="store_true", help="Valuta le predizioni finali e genera un report.")
    args = parser.parse_args()

    config = load_config(args.config)
    model_dirs = [Path(p['output_dir']) for p in config.score_generation_models]
    base_asr_model_names = [d.name.split('_')[-1] for d in model_dirs]

    print("--- PASSO 1: Caricamento e preparazione degli score ---")
    oof_dfs = [pd.read_csv(d / "oof_predictions.csv").rename(columns={'score': f'score_{d.name}'}) for d in model_dirs]
    train_df = oof_dfs[0]
    for i in range(1, len(oof_dfs)):
        train_df = pd.merge(train_df, oof_dfs[i][['ID', f'score_{model_dirs[i].name}']], on='ID')
    dx_col = next((c for c in ['dx', 'diagnosis'] if c in train_df.columns), None)
    train_df['label'] = train_df[dx_col].map({'cn': 0, 'ad': 1})
    X_train = train_df.filter(like='score_')
    y_train = train_df['label']

    test_dfs = [pd.read_csv(d / "test_predictions.csv").rename(columns={'score': f'score_{d.name}'}) for d in model_dirs]
    test_df = test_dfs[0]
    for i in range(1, len(test_dfs)):
        test_df = pd.merge(test_df, test_dfs[i], on='ID')
    X_test = test_df.drop(columns=['ID'])
    print(f"Dati pronti. Modelli di base fusi: {', '.join(base_asr_model_names)}")

    print("\n--- PASSO 2: Definizione dei modelli di fusione ---")
    models_and_grids = [
        {"name": "Logistic Regression", "estimator": LogisticRegression(random_state=config.seed, max_iter=2000), "grid": {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga'], 'penalty': ['l1', 'l2']}},
        {"name": "Support Vector Machine (SVM)", "estimator": SVC(probability=True, random_state=config.seed), "grid": {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 'scale', 'auto'], 'kernel': ['rbf', 'linear']}},
        {"name": "XGBoost", "estimator": XGBClassifier(random_state=config.seed, eval_metric='logloss'), "grid": {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2], 'subsample': [0.7, 0.8, 1.0], 'colsample_bytree': [0.7, 0.8, 1.0]}}
    ]

    print("\n--- PASSO 3: Avvio della Grid Search ---")
    cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=config.seed)
    best_overall_score, best_overall_model, best_model_name = -1, None, ""
    best_params_dict = {} # <- Qui salviamo i parametri puliti

    for model_info in models_and_grids:
        print(f"\n>>> Ricerca per: {model_info['name']} <<<")
        grid_search = GridSearchCV(estimator=model_info['estimator'], param_grid=model_info['grid'], cv=cv_strategy, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        print(f"Risultati per {model_info['name']}:")
        print(f"  - Miglior score CV (Accuracy): {grid_search.best_score_:.4f}")
        print(f"  - Migliori iperparametri: {grid_search.best_params_}")
        if grid_search.best_score_ > best_overall_score:
            best_overall_score = grid_search.best_score_
            best_overall_model = grid_search.best_estimator_
            best_model_name = model_info['name']
            best_params_dict = grid_search.best_params_ # Salva il dizionario pulito

    print("\n--- PASSO 4: Predizione finale con il modello campione ---")
    print(f"Modello Campione: {best_model_name}")
    final_predictions = best_overall_model.predict(X_test)
    submission_df = pd.DataFrame({'ID': test_df['ID'], 'prediction': final_predictions})
    submission_df.to_csv(args.output_file, index=False)
    print(f"\nPredizioni finali salvate in: {args.output_file}")

    if args.evaluate:
        candidate_fuser_names = [m['name'] for m in models_and_grids]
        evaluate_and_save_report(
            submission_df=submission_df, 
            config=config, 
            model_name=best_model_name,
            best_params=best_params_dict, # <- Passa il dizionario pulito alla funzione di report
            best_cv_score=best_overall_score,
            candidate_fusers=candidate_fuser_names,
            base_asr_models=base_asr_model_names
        )

if __name__ == "__main__":
    main()