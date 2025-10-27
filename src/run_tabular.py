import argparse
import sys
from pathlib import Path

# Aggiunge la root del progetto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.utils import set_seed
# Importiamo il nuovo engine e l'Evaluator esistente
from src.tabular_engine import TabularTrainer, TabularPredictor
from src.engine import Evaluator 

def main():
    parser = argparse.ArgumentParser(description="Script per training/predizione con modelli tabulari (feature manuali).")
    parser.add_argument("--config", type=str, required=True, help="Path al config.yaml")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "predict", "evaluate"], help="Modalità.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    if args.mode == 'train':
        trainer = TabularTrainer(config)
        trainer.train()
    elif args.mode == 'predict':
        predictor = TabularPredictor(config)
        predictor.predict()
    elif args.mode == 'evaluate':
        print("Valutazione predizioni tabulari...")
        evaluator = Evaluator(config)
        # L'Evaluator esistente funziona perché legge 'predictions.csv' 
        # che abbiamo generato nello stesso formato.
        evaluator.evaluate()

if __name__ == "__main__":
    main()