"""Tabular ML train/predict/evaluate (SVM, XGBoost, Logistic Regression)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.utils import set_seed
from src.tabular_engine import TabularTrainer, TabularPredictor
from src.engine import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Tabular ML train/predict/evaluate.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["train", "predict", "evaluate"])
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    if args.mode == 'train':
        TabularTrainer(config).train()
    elif args.mode == 'predict':
        TabularPredictor(config).predict()
    elif args.mode == 'evaluate':
        Evaluator(config).evaluate()


if __name__ == "__main__":
    main()
