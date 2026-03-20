import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from src.utils import load_labels_df, find_diagnosis_column, map_label


class TabularTrainer:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_set = config.feature_extraction.feature_set

    def _load_features(self, is_test=False):
        fname = f"test_{self.feature_set}.csv" if is_test else f"train_{self.feature_set}.csv"
        path = Path(self.config.data.features_root) / fname
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}. Run extract_features.py or extract_embeddings_batch.py first.")
        return pd.read_csv(path)

    def _build_pipeline(self):
        model_name = self.config.tabular_model.name
        pca = PCA(n_components=0.95, random_state=self.config.seed)

        if model_name == 'svm':
            clf = SVC(probability=True, class_weight='balanced', random_state=self.config.seed)
        elif model_name == 'xgboost':
            clf = XGBClassifier(
                objective='binary:logistic',
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=1.2,
                random_state=self.config.seed,
                n_jobs=1,
            )
        elif model_name == 'lr':
            clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=self.config.seed)
        else:
            raise ValueError(f"Tabular model '{model_name}' not supported.")

        return Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('pca', pca),
            ('clf', clf),
        ])

    def train(self):
        print(f"--- Tabular Training ({self.config.tabular_model.name}) on {self.feature_set} ---")

        feat_df = self._load_features(is_test=False)
        feat_df['ID'] = feat_df['ID'].astype(str)

        labels_df = load_labels_df(self.config.data.train_labels)
        dx_col = find_diagnosis_column(labels_df)
        if dx_col is None:
            raise ValueError(f"Diagnosis column not found. Available: {labels_df.columns.tolist()}")

        labels_df['target'] = labels_df[dx_col].apply(map_label)

        full_train_df = pd.merge(feat_df, labels_df[['ID', 'target']], on='ID', how='inner')
        if len(full_train_df) == 0:
            raise ValueError("Merge between features and labels produced 0 rows.")

        X_all = full_train_df.drop(columns=['ID', 'target'])
        y_all = full_train_df['target'].values.astype(int)
        ids_all = full_train_df['ID'].values

        print(f"Dataset: {X_all.shape} (samples x features)")
        print(f"Class distribution: {np.bincount(y_all)} (0=CN/MCI, 1=AD)")

        # GridSearch for best hyperparameters
        print("Running GridSearchCV...")
        pipeline = self._build_pipeline()
        model_name = self.config.tabular_model.name
        grid_dict = getattr(self.config.tabular_model.grids, model_name).to_dict()
        param_grid = {f'clf__{k}': v for k, v in grid_dict.items()}

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.seed)
        grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)
        grid.fit(X_all, y_all)

        best_params = grid.best_params_
        print(f"Best params: {best_params}")
        print(f"Best CV F1: {grid.best_score_:.4f}")

        # K-Fold ensemble training
        print("\nTraining K-Fold ensemble...")
        skf = StratifiedKFold(n_splits=self.config.k_folds, shuffle=True, random_state=self.config.seed)
        oof_preds = np.zeros((len(full_train_df), 2))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
            fold_dir = self.output_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            X_train, y_train = X_all.iloc[train_idx], y_all[train_idx]
            X_val = X_all.iloc[val_idx]

            model = self._build_pipeline()
            model.set_params(**best_params)
            model.fit(X_train, y_train)
            joblib.dump(model, fold_dir / "tabular_model.pkl")

            if hasattr(model, "predict_proba"):
                oof_preds[val_idx] = model.predict_proba(X_val)
            else:
                pred = model.predict(X_val)
                oof_preds[val_idx, 1] = pred
                oof_preds[val_idx, 0] = 1 - pred

        oof_df = pd.DataFrame({'ID': ids_all, 'score': oof_preds[:, 1], 'label': y_all})
        oof_df.to_csv(self.output_dir / "oof_predictions.csv", index=False)
        print(f"Training complete. Models saved to {self.output_dir}")


class TabularPredictor:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.feature_set = config.feature_extraction.feature_set

    def _load_features(self, is_test=True):
        fname = f"test_{self.feature_set}.csv" if is_test else f"train_{self.feature_set}.csv"
        path = Path(self.config.data.features_root) / fname
        return pd.read_csv(path)

    def predict(self):
        print(f"--- Tabular Prediction on Test Set ---")

        feat_df = self._load_features(is_test=True)
        feat_df['ID'] = feat_df['ID'].astype(str)

        test_split_df = load_labels_df(self.config.data.test_task1_labels)
        target_ids = test_split_df['ID'].values

        X_test_full = feat_df[feat_df['ID'].isin(target_ids)].copy()
        X_test_full = X_test_full.set_index('ID').reindex(target_ids).reset_index()

        ids_test = X_test_full['ID'].values
        X_test = X_test_full.drop(columns=['ID'])

        fold_scores = []
        for fold in range(self.config.k_folds):
            model_path = self.output_dir / f"fold_{fold}" / "tabular_model.pkl"
            if not model_path.exists():
                print(f"WARNING: Fold {fold} model not found.")
                continue

            model = joblib.load(model_path)
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(X_test)[:, 1]
            else:
                scores = model.predict(X_test)
            fold_scores.append(scores)

        if not fold_scores:
            raise ValueError("No models found for prediction.")

        avg_scores = np.mean(fold_scores, axis=0)

        pd.DataFrame({'ID': ids_test, 'score': avg_scores}).to_csv(self.output_dir / "test_predictions.csv", index=False)
        pd.DataFrame({'ID': ids_test, 'prediction': (avg_scores >= 0.5).astype(int)}).to_csv(self.output_dir / "predictions.csv", index=False)
        print(f"Predictions saved to {self.output_dir}")
