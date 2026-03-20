"""Late fusion with grid search over multiple classifiers."""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import warnings
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.config import load_config
from src.utils import load_labels_df, find_diagnosis_column, LABEL_MAPPING, compute_classification_metrics

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def evaluate_and_save_report(submission_df, config, model_name, best_params, best_cv_score, candidate_fusers, base_asr_models):
    labels_df = load_labels_df(config.data.test_task1_labels)
    merged_df = pd.merge(submission_df, labels_df, on="ID")
    if merged_df.empty:
        print("ERROR: No matching IDs between predictions and labels.")
        return

    y_pred = merged_df['prediction']
    dx_col = find_diagnosis_column(merged_df)
    y_true = merged_df[dx_col].map(LABEL_MAPPING)
    metrics = compute_classification_metrics(y_true, y_pred)

    report_lines = [
        f"--- Fusion Evaluation Report ---",
        "=" * 70,
        f"\nBase models (ASR): {', '.join(base_asr_models)}",
        f"Candidate fusers: {', '.join(candidate_fusers)}",
        f"\n--- Selected model: '{model_name}' ---",
        f"Best CV Accuracy: {best_cv_score:.4f}",
        "Best hyperparameters:",
        json.dumps(best_params, indent=4),
        f"\n--- Test Set Evaluation ---",
        f"Accuracy           : {metrics['accuracy']:.4f}",
        f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}",
        f"Sensitivity (Recall): {metrics['sensitivity']:.4f}",
        f"Specificity        : {metrics['specificity']:.4f}",
        f"\n--- Classification Report ---",
        metrics['classification_report'],
        f"\n--- Confusion Matrix ---",
        str(metrics['confusion_matrix']),
        "\n" + "=" * 70,
    ]

    full_report = "\n".join(report_lines)
    print("\n" + full_report)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    fused_str = "_".join(sorted(base_asr_models))
    report_path = reports_dir / f"{safe_name}_{fused_str}_report.txt"
    report_path.write_text(full_report, encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Grid search for best fusion model.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output_file", type=str, default="best_fused_submission.csv")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model_dirs = [Path(p['output_dir']) for p in config.score_generation_models]
    base_asr_model_names = [d.name.split('_')[-1] for d in model_dirs]

    # Load OOF scores
    print("--- Step 1: Loading scores ---")
    oof_dfs = [pd.read_csv(d / "oof_predictions.csv").rename(columns={'score': f'score_{d.name}'}) for d in model_dirs]
    train_df = oof_dfs[0]
    for i in range(1, len(oof_dfs)):
        train_df = pd.merge(train_df, oof_dfs[i][['ID', f'score_{model_dirs[i].name}']], on='ID')

    dx_col = next((c for c in ['dx', 'diagnosis'] if c in train_df.columns), None)
    train_df['label'] = train_df[dx_col].map({'cn': 0, 'ad': 1})
    X_train = train_df.filter(like='score_')
    y_train = train_df['label']

    # Load test scores
    test_dfs = [pd.read_csv(d / "test_predictions.csv").rename(columns={'score': f'score_{d.name}'}) for d in model_dirs]
    test_df = test_dfs[0]
    for i in range(1, len(test_dfs)):
        test_df = pd.merge(test_df, test_dfs[i], on='ID')
    X_test = test_df.drop(columns=['ID'])

    # Define fusion models
    print("\n--- Step 2: Grid search ---")
    models_and_grids = [
        {"name": "Logistic Regression", "estimator": LogisticRegression(random_state=config.seed, max_iter=2000),
         "grid": {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga'], 'penalty': ['l1', 'l2']}},
        {"name": "SVM", "estimator": SVC(probability=True, random_state=config.seed),
         "grid": {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 'scale', 'auto'], 'kernel': ['rbf', 'linear']}},
        {"name": "XGBoost", "estimator": XGBClassifier(random_state=config.seed, eval_metric='logloss'),
         "grid": {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2],
                  'subsample': [0.7, 0.8, 1.0], 'colsample_bytree': [0.7, 0.8, 1.0]}},
    ]

    cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=config.seed)
    best_score, best_model, best_name, best_params = -1, None, "", {}

    for info in models_and_grids:
        print(f"\n>>> {info['name']} <<<")
        gs = GridSearchCV(estimator=info['estimator'], param_grid=info['grid'], cv=cv_strategy, scoring='accuracy', n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        print(f"  Best CV accuracy: {gs.best_score_:.4f} | Params: {gs.best_params_}")
        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_model = gs.best_estimator_
            best_name = info['name']
            best_params = gs.best_params_

    # Final prediction
    print(f"\n--- Step 3: Final prediction with {best_name} ---")
    final_predictions = best_model.predict(X_test)
    submission_df = pd.DataFrame({'ID': test_df['ID'], 'prediction': final_predictions})
    submission_df.to_csv(args.output_file, index=False)
    print(f"Predictions saved to: {args.output_file}")

    if args.evaluate:
        evaluate_and_save_report(
            submission_df, config, best_name, best_params, best_score,
            [m['name'] for m in models_and_grids], base_asr_model_names,
        )


if __name__ == "__main__":
    main()
