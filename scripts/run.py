import argparse
import sys
from pathlib import Path

# Aggiunge la root del progetto al path per permettere 'from src import ...'
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.utils import set_seed
from src.data import get_data_splits, get_dataloaders
from src.engine import Trainer, Predictor, Evaluator

def main():
    parser = argparse.ArgumentParser(description="Script principale per training, predizione e valutazione.")
    parser.add_argument("--config", type=str, required=True, help="Percorso al file di configurazione (es. config.yaml)")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "predict", "evaluate"], help="Modalità di esecuzione.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    if args.mode == 'train':
        print(f"Inizio training per il task '{config.task}' con la modalità '{config.modality}'.")
        print(f"Gli output verranno salvati in: {config.output_dir}")
        
        # Il K-Fold è gestito qui
        for fold, train_df, val_df, _ in get_data_splits(config):
            train_loader, val_loader = get_dataloaders(config, train_df, val_df)
            trainer = Trainer(config, train_loader, val_loader, fold)
            trainer.train()
            
    elif args.mode == 'predict':
        print(f"Inizio predizione per il task '{config.task}' con la modalità '{config.modality}'.")
        print(f"Uso i modelli da: {config.output_dir}")
        
        # Ottieni il test loader (usiamo un ciclo fittizio per prendere il primo test_df)
        for _, _, _, test_df in get_data_splits(config):
            _, _, test_loader = get_dataloaders(config, train_df=test_df, val_df=test_df, test_df=test_df)
            break # Usciamo dopo il primo, dato che test_df è sempre lo stesso
            
        predictor = Predictor(config, test_loader)
        predictor.predict()

    elif args.mode == 'evaluate':
        print(f"Inizio valutazione per il task '{config.task}'.")
        print(f"Valuto le predizioni in: {Path(config.output_dir) / 'predictions.csv'}")
        
        evaluator = Evaluator(config)
        evaluator.evaluate()

if __name__ == "__main__":
    main()