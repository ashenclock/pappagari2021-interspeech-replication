# scripts/extract_features.py

import argparse
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import opensmile
import librosa
from scipy.stats import skew, kurtosis
import numpy as np
import warnings
from multiprocessing import Pool, cpu_count
import multiprocessing

# NUOVI IMPORT PER WHISPER
import torch
from transformers import AutoProcessor, AutoModel
import torchaudio
from torchaudio.functional import resample

# Aggiunge la root del progetto al path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config

warnings.filterwarnings("ignore")

# --- Variabili globali e inizializzatori per i processi figli ---
extractor_opensmile = None
extractor_disvoice = None

# NUOVE VARIABILI GLOBALI PER WHISPER
whisper_model = None
whisper_processor = None
whisper_device = None
whisper_pooling = None


def initialize_worker_opensmile(feature_set):
    global extractor_opensmile
    if feature_set == 'egemaps':
        extractor_opensmile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    elif feature_set == 'compare':
        extractor_opensmile = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016, feature_level=opensmile.FeatureLevel.Functionals)

def initialize_worker_mfcc():
    pass

def initialize_worker_disvoice():
    global extractor_disvoice
    try:
        from disvoice.prosody import Prosody
        extractor_disvoice = Prosody()
    except ImportError:
        print("ERRORE: La libreria 'disvoice' non è installata.", file=sys.stderr)
        sys.exit(1)

# --- NUOVO INIZIALIZZATORE PER WHISPER ---
def initialize_worker_whisper(config_dict):
    """Carica il modello e il processore Whisper una sola volta per ogni processo figlio."""
    global whisper_model, whisper_processor, whisper_device, whisper_pooling
    
    # Ricostruiamo l'oggetto config per facilità d'uso
    from src.config import Config
    config = Config(config_dict)
    
    whisper_device = config.device
    whisper_pooling = config.embedding_extraction.pooling_strategy
    model_id = config.embedding_extraction.model_id
    
    print(f"Processo {multiprocessing.current_process().pid}: caricamento modello '{model_id}'...")
    
    whisper_processor = AutoProcessor.from_pretrained(model_id)
    whisper_model = AutoModel.from_pretrained(model_id).to(whisper_device)
    whisper_model.eval() # Mettiamo il modello in modalità valutazione
    
    print(f"Processo {multiprocessing.current_process().pid}: modello caricato.")


# --- Funzioni di elaborazione per i processi figli ---

def process_single_file_opensmile(audio_path):
    # ... (invariato)
    global extractor_opensmile
    try:
        feature_df = extractor_opensmile.process_file(str(audio_path)).reset_index()
        feature_df['ID'] = audio_path.stem
        return feature_df
    except Exception as e:
        print(f"\nATTENZIONE: Errore OpenSMILE su {audio_path.name}: {e}. File saltato.", file=sys.stderr)
        return None

def process_single_file_mfcc(audio_path):
    # ... (invariato)
    try:
        y, sr = librosa.load(str(audio_path), sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13); delta_mfccs = librosa.feature.delta(mfccs); delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        all_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs)); stats = {'mean': np.mean(all_features, axis=1), 'std': np.std(all_features, axis=1), 'skew': skew(all_features, axis=1), 'kurtosis': kurtosis(all_features, axis=1), 'min': np.min(all_features, axis=1), 'max': np.max(all_features, axis=1)}
        feature_dict = {'ID': audio_path.stem}; feature_names = [f'mfcc_{i}' for i in range(13)] + [f'delta_{i}' for i in range(13)] + [f'delta2_{i}' for i in range(13)]
        for stat_name, stat_values in stats.items():
            for i, value in enumerate(stat_values): feature_dict[f'{feature_names[i]}_{stat_name}'] = value
        return pd.DataFrame([feature_dict])
    except Exception as e:
        print(f"\nATTENZIONE: Errore Librosa su {audio_path.name}: {e}. File saltato.", file=sys.stderr)
        return None

def process_single_file_disvoice(audio_path):
    # ... (invariato)
    global extractor_disvoice
    try:
        features_dict = extractor_disvoice.extract_features_file(str(audio_path), static=True)
        feature_df = pd.DataFrame([features_dict])
        feature_df['ID'] = audio_path.stem
        return feature_df
    except Exception as e:
        if "Sound is too short" in str(e): print(f"\nINFO: File {audio_path.name} troppo corto. Saltato.", file=sys.stderr)
        else: print(f"\nATTENZIONE: Errore Disvoice su {audio_path.name}: {e}. File saltato.", file=sys.stderr)
        return None

# --- NUOVA FUNZIONE DI ELABORAZIONE PER WHISPER ---
def process_single_file_whisper(audio_path):
    """Estrae l'embedding da un singolo file audio usando il modello Whisper globale."""
    global whisper_model, whisper_processor, whisper_device, whisper_pooling
    
    try:
        # 1. Carica e ricampiona l'audio
        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1: # Converti a mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != 16000:
            waveform = resample(waveform, sr, 16000)
            
        # 2. Prepara l'input per il modello
        inputs = whisper_processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=16000)
        input_features = inputs.input_features.to(whisper_device)

        # 3. Esegui il forward pass sull'encoder
        with torch.no_grad():
            # Estraiamo gli hidden states dall'encoder
            last_hidden_state = whisper_model.encoder(input_features).last_hidden_state

        # 4. Applica il pooling
        if whisper_pooling == "mean":
            # Calcola la media lungo la dimensione temporale
            pooled_output = last_hidden_state.mean(dim=1)
        else:
            # Qui potresti aggiungere altre strategie di pooling (es. 'max')
            raise ValueError(f"Pooling strategy '{whisper_pooling}' non supportata.")

        # 5. Crea il DataFrame
        embedding = pooled_output.cpu().numpy().flatten()
        feature_dict = {f"whisper_{i}": val for i, val in enumerate(embedding)}
        feature_dict['ID'] = audio_path.stem
        
        return pd.DataFrame([feature_dict])

    except Exception as e:
        print(f"\nATTENZIONE: Errore Whisper su {audio_path.name}: {e}. File saltato.", file=sys.stderr)
        return None


def extract_and_save(audio_root, output_path, config):
    if output_path.exists() and not config.feature_extraction.overwrite:
        print(f"Feature file già esistente: {output_path}")
        return

    audio_files = sorted(list(Path(audio_root).rglob("*.wav")))
    feature_set = config.feature_extraction.feature_set
    
    print(f"Estrazione feature '{feature_set}' da {len(audio_files)} files in parallelo...")

    # --- MODIFICA: Aggiunta la nuova opzione per Whisper ---
    if feature_set in ['egemaps', 'compare']:
        initializer = initialize_worker_opensmile
        initargs = (feature_set,)
        process_func = process_single_file_opensmile
    elif feature_set == 'mfcc_stats':
        initializer = initialize_worker_mfcc
        initargs = ()
        process_func = process_single_file_mfcc
    elif feature_set == 'disvoice_prosody':
        initializer = initialize_worker_disvoice
        initargs = ()
        process_func = process_single_file_disvoice
    # NUOVO BLOCCO ELIF PER WHISPER
    elif hasattr(config, 'embedding_extraction') and feature_set == config.embedding_extraction.name:
        initializer = initialize_worker_whisper
        # Passiamo la configurazione come dizionario perché è più sicuro con il multiprocessing
        initargs = (config.to_dict(),) 
        process_func = process_single_file_whisper
    else:
        raise ValueError(f"Feature set '{feature_set}' non supportato.")

    # Il resto della funzione rimane quasi invariato
    with Pool(initializer=initializer, initargs=initargs) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(process_func, audio_files), total=len(audio_files)):
            if result is not None:
                results.append(result)

    if not results:
        print("Nessuna feature estratta."); return

    full_df = pd.concat(results, ignore_index=True)
    
    if feature_set in ['egemaps', 'compare']:
        cols_to_drop = ['file', 'start', 'end']
        full_df = full_df.drop(columns=[c for c in cols_to_drop if c in full_df.columns])
    
    # Riempi eventuali valori NaN con 0 (può accadere se un file fallisce)
    full_df = full_df.fillna(0)

    cols = ['ID'] + [c for c in full_df.columns if c != 'ID']
    full_df = full_df[cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(output_path, index=False)
    print(f"Feature salvate in: {output_path}")

def main():
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Estrae feature acustiche manuali o embedding in parallelo.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path al file di configurazione.")
    args = parser.parse_args()
    config = load_config(args.config)

    features_root = Path(config.data.features_root)
    features_root.mkdir(exist_ok=True)
    
    # MODIFICA: Determina il nome del file di output in modo dinamico
    feature_set_name = config.feature_extraction.feature_set

    # Train
    train_out = features_root / f"train_{feature_set_name}.csv"
    extract_and_save(config.data.audio_root, train_out, config)

    # Test
    test_out = features_root / f"test_{feature_set_name}.csv"
    extract_and_save(config.data.test_audio_root, test_out, config)
    print(f"\nEstrazione completata usando {cpu_count()} processi.")


if __name__ == "__main__":
    main()