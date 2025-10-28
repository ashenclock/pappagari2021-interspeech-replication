# scripts/extract_embeddings_batch.py

import argparse
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import warnings
import torch
from transformers import AutoProcessor, AutoModel
import torchaudio
from torchaudio.functional import resample
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Aggiunge la root del progetto al path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.utils import clear_memory

warnings.filterwarnings("ignore")

# --- Dataset custom per caricare i file audio ---
class AudioFileDataset(Dataset):
    """Un Dataset PyTorch per caricare file audio da una lista di percorsi."""
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0)
            else:
                waveform = waveform.squeeze(0)
            if sr != 16000:
                waveform = resample(waveform, sr, 16000)
            return {"waveform": waveform, "id": audio_path.stem}
        except Exception as e:
            print(f"Errore nel caricare {audio_path.name}: {e}. Salto il file.", file=sys.stderr)
            return {"waveform": torch.tensor([]), "id": "error"}

# --- Funzione Collate per gestire il padding ---
def collate_fn(batch):
    """
    Raggruppa i dati e converte le waveform in array NumPy,
    come richiesto dal processor di Hugging Face.
    """
    batch = [item for item in batch if item['id'] != "error"]
    if not batch:
        return None

    # ========================== ECCO LA CORREZIONE ==========================
    # Convertiamo ogni tensore waveform in un array NumPy.
    # Questo è il formato che il feature_extractor si aspetta.
    waveforms = [item['waveform'].numpy() for item in batch]
    # ======================================================================

    ids = [item['id'] for item in batch]
    
    return {"waveforms": waveforms, "ids": ids}


class EmbeddingExtractor:
    """
    Classe che gestisce il caricamento del modello e l'estrazione degli embedding
    in modo efficiente tramite batch.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.model_id = config.embedding_extraction.model_id
        self.pooling = config.embedding_extraction.pooling_strategy
        self.batch_size = config.embedding_extraction.batch_size

        print(f"Caricamento modello '{self.model_id}' su {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        print("Modello caricato.")

    def extract(self, audio_root: Path, output_path: Path):
        """Estrae gli embedding da tutti i file in una directory e li salva in un CSV."""
        if output_path.exists() and not self.config.feature_extraction.overwrite:
            print(f"File di feature già esistente, salto: {output_path}")
            return

        audio_files = sorted(list(audio_root.rglob("*.wav")))
        if not audio_files:
            print(f"Nessun file .wav trovato in {audio_root}")
            return

        dataset = AudioFileDataset(audio_files)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )

        all_embeddings = []
        all_ids = []

        print(f"Estrazione embeddings da {len(audio_files)} file (batch size: {self.batch_size})...")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                if batch is None:
                    continue

                # Ora `batch['waveforms']` è una LISTA di array NumPy, che è l'input corretto.
                inputs = self.processor(
                    batch['waveforms'],
                    return_tensors="pt",
                    sampling_rate=16000,
                    padding=True
                )
                input_features = inputs.input_features.to(self.device)

                last_hidden_state = self.model.encoder(input_features).last_hidden_state

                if self.pooling == "mean":
                    pooled_output = last_hidden_state.mean(dim=1)
                else:
                    raise ValueError(f"Pooling strategy '{self.pooling}' non supportata.")
                
                all_embeddings.append(pooled_output.cpu().numpy())
                all_ids.extend(batch['ids'])

        if not all_ids:
            print("ERRORE: Nessun embedding è stato estratto.")
            return

        embeddings_matrix = np.vstack(all_embeddings)
        feature_names = [f"whisper_{i}" for i in range(embeddings_matrix.shape[1])]
        
        df = pd.DataFrame(embeddings_matrix, columns=feature_names)
        df.insert(0, 'ID', all_ids)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Feature salvate con successo in: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Estrae embeddings audio in batch usando la GPU.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path al file di configurazione.")
    args = parser.parse_args()
    config = load_config(args.config)
    
    feature_set_name = config.feature_extraction.feature_set
    if not hasattr(config, 'embedding_extraction') or feature_set_name != config.embedding_extraction.name:
         print(f"ERRORE: Questo script è progettato solo per l'estrazione di embedding (es. 'whisper_large_v3').")
         print(f"Il feature_set nel config è '{feature_set_name}'. Esegui 'scripts/extract_features.py' per quello.")
         sys.exit(1)

    clear_memory()
    extractor = EmbeddingExtractor(config)
    
    features_root = Path(config.data.features_root)
    
    print("\n--- Processando il set di training ---")
    train_out = features_root / f"train_{feature_set_name}.csv"
    extractor.extract(Path(config.data.audio_root), train_out)

    print("\n--- Processando il set di test ---")
    test_out = features_root / f"test_{feature_set_name}.csv"
    extractor.extract(Path(config.data.test_audio_root), test_out)
    
    print("\nEstrazione completata.")

if __name__ == "__main__":
    main()