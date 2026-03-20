"""Main script for deep learning train/predict/evaluate."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.utils import set_seed
from src.data import get_data_splits, get_dataloaders
from src.engine import Trainer, Predictor, Evaluator


def main():
    parser = argparse.ArgumentParser(description="Deep learning train/predict/evaluate.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["train", "predict", "evaluate"])
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    if args.mode == 'train':
        print(f"Training: task={config.task}, modality={config.modality}, output={config.output_dir}")
        for fold, train_df, val_df, _ in get_data_splits(config):
            train_loader, val_loader = get_dataloaders(config, train_df, val_df)
            Trainer(config, train_loader, val_loader, fold).train()

    elif args.mode == 'predict':
        print(f"Predicting: task={config.task}, modality={config.modality}")
        for _, _, _, test_df in get_data_splits(config):
            _, _, test_loader = get_dataloaders(config, train_df=test_df, val_df=test_df, test_df=test_df)
            break
        Predictor(config, test_loader).predict()

    elif args.mode == 'evaluate':
        Evaluator(config).evaluate()


if __name__ == "__main__":
    main()
