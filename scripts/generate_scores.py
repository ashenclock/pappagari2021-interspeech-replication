# scripts/generate_scores.py

import argparse
import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, Config
from src.data import get_data_splits, get_dataloaders
from src.models import build_model
from src.utils import set_seed, clear_memory

# Funzione per aggiornare ricorsivamente i dizionari di configurazione
def recursive_update(base_dict, update_dict):
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = recursive_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

# La funzione get_scores rimane invariata
def get_scores(model, loader, device, task):
    model.eval()
    all_scores, all_ids = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferencing", leave=False):
            ids = batch.pop("id")
            all_ids.extend(ids)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch)
            scores = torch.softmax(outputs, dim=1)[:, 1] if task == 'classification' else outputs
            all_scores.extend(scores.cpu().numpy())
    return all_ids, all_scores

def process_single_model(model_info: dict, base_config: Config):
    """
    Processa un singolo modello usando la configurazione base e le informazioni specifiche.
    """
    output_dir = Path(model_info['output_dir'])
    print(f"\n{'='*20} Processing Model: {output_dir.name} {'='*20}")

    if not output_dir.exists():
        print(f"ATTENZIONE: La directory di output '{output_dir}' non esiste. Salto questo modello.")
        return

    # Crea una configurazione specifica per questo modello
    # Partendo da una copia della configurazione base
    model_config_dict = copy.deepcopy(base_config.to_dict())
    
    # Aggiorna la configurazione con le info specifiche di questo modello
    model_config_dict = recursive_update(model_config_dict, model_info)

    # **Logica Intelligente per i Transcripts**
    # Se il modello è testuale, proviamo a indovinare la cartella dei transcript
    if model_info['modality'] == 'text':
        # Esempio: da "bert_classification_parakeet" estrae "parakeet"
        # Adatta questa logica se i tuoi nomi sono diversi
        potential_transcript_folder = model_info['output_dir'].split('_')[-1]
        model_config_dict['transcription_model_for_training'] = potential_transcript_folder
        print(f"  -> Info: Modello testuale. Cartella transcript impostata su: '{potential_transcript_folder}'")

    # Trasforma il dizionario finale in un oggetto Config
    model_config = Config(model_config_dict)

    device = torch.device(model_config.device)
    clear_memory()

    # --- 1. Generazione OOF ---
    print("\n[Fase 1/2] Generazione predizioni Out-of-Fold (OOF)...")
    oof_predictions = []
    for fold, train_df, val_df, _ in get_data_splits(model_config):
        model_path = output_dir / f"fold_{fold}" / "best_model.pt"
        if not model_path.exists(): continue
        model = build_model(model_config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        _, val_loader = get_dataloaders(model_config, train_df, val_df)
        val_ids, val_scores = get_scores(model, val_loader, device, model_config.task)
        oof_predictions.append(pd.DataFrame({'ID': val_ids, 'score': val_scores}))
        
    if oof_predictions:
        full_oof_df = pd.concat(oof_predictions).set_index('ID')
        train_labels_df = pd.read_csv(model_config.data.train_labels).rename(columns={'adressfname': 'ID'}).set_index('ID')
        final_oof_df = full_oof_df.join(train_labels_df)
        final_oof_df.to_csv(output_dir / "oof_predictions.csv")
        print(f"  -> Predizioni OOF salvate.")

    # --- 2. Generazione Test ---
    print("\n[Fase 2/2] Generazione predizioni sul Test Set...")
    all_test_scores = []
    _, _, _, test_df = next(get_data_splits(model_config))
    _, _, test_loader = get_dataloaders(model_config, test_df, test_df, test_df=test_df)
    test_ids = []
    for fold in range(model_config.k_folds):
        model_path = output_dir / f"fold_{fold}" / "best_model.pt"
        if not model_path.exists(): continue
        model = build_model(model_config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        ids, scores = get_scores(model, test_loader, device, model_config.task)
        if not test_ids: test_ids = ids # Salva gli ID solo la prima volta
        all_test_scores.append(scores)

    if all_test_scores:
        mean_scores = np.mean(all_test_scores, axis=0)
        pd.DataFrame({'ID': test_ids, 'score': mean_scores}).to_csv(output_dir / "test_predictions.csv", index=False)
        print(f"  -> Predizioni sul Test Set salvate.")

def main():
    parser = argparse.ArgumentParser(description="Genera in batch le predizioni per tutti i modelli definiti in config.yaml.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Percorso al file di configurazione principale.")
    args = parser.parse_args()
    
    main_config = load_config(args.config)
    set_seed(main_config.seed)

    if not hasattr(main_config, 'score_generation_models') or not main_config.score_generation_models:
        print("ERRORE: La chiave 'score_generation_models' non è stata trovata o è vuota nel tuo config.yaml.")
        return

    # Itera sulla lista di dizionari nel file di configurazione
    for model_info_dict in main_config.score_generation_models:
        process_single_model(model_info_dict, main_config)

    print(f"\n{'='*20} Generazione Score Completata {'='*20}")

if __name__ == "__main__":
    main()