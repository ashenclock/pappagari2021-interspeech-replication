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

# Aggiunge la root del progetto al path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config

warnings.filterwarnings("ignore")

# --- Variabili globali e inizializzatori per i processi figli ---
extractor_opensmile = None
feature_set_global = None

def initialize_worker_opensmile(feature_set):
    global extractor_opensmile
    if feature_set == 'egemaps':
        extractor_opensmile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    elif feature_set == 'compare':
        extractor_opensmile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

def initialize_worker_mfcc():
    # Nessuna inizializzazione pesante richiesta per librosa
    pass

# --- Funzioni di elaborazione per i processi figli ---

def process_single_file_opensmile(audio_path):
    global extractor_opensmile
    try:
        feature_df = extractor_opensmile.process_file(str(audio_path)).reset_index()
        feature_df['ID'] = audio_path.stem
        return feature_df
    except Exception as e:
        print(f"\nATTENZIONE: Errore OpenSMILE su {audio_path.name}: {e}. File saltato.", file=sys.stderr)
        return None

def process_single_file_mfcc(audio_path):
    try:
        y, sr = librosa.load(str(audio_path), sr=None)
        
        # 1. Calcola MFCC, Delta, e Delta-Delta
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Concatena in un'unica matrice
        all_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
        
        # 2. Calcola le statistiche (functionals)
        stats = {
            'mean': np.mean(all_features, axis=1),
            'std': np.std(all_features, axis=1),
            'skew': skew(all_features, axis=1),
            'kurtosis': kurtosis(all_features, axis=1),
            'min': np.min(all_features, axis=1),
            'max': np.max(all_features, axis=1)
        }
        
        # 3. Crea un dizionario flat per il DataFrame
        feature_dict = {'ID': audio_path.stem}
        feature_names = [f'mfcc_{i}' for i in range(13)] + \
                        [f'delta_{i}' for i in range(13)] + \
                        [f'delta2_{i}' for i in range(13)]
        
        for stat_name, stat_values in stats.items():
            for i, value in enumerate(stat_values):
                feature_dict[f'{feature_names[i]}_{stat_name}'] = value
                
        return pd.DataFrame([feature_dict])

    except Exception as e:
        print(f"\nATTENZIONE: Errore Librosa su {audio_path.name}: {e}. File saltato.", file=sys.stderr)
        return None

def extract_and_save(audio_root, output_path, config):
    if output_path.exists() and not config.feature_extraction.overwrite:
        print(f"Feature file già esistente: {output_path}")
        return

    audio_files = sorted(list(Path(audio_root).rglob("*.wav")))
    feature_set = config.feature_extraction.feature_set
    
    print(f"Estrazione feature '{feature_set}' da {len(audio_files)} files in parallelo...")

    if feature_set in ['egemaps', 'compare']:
        initializer = initialize_worker_opensmile
        initargs = (feature_set,)
        process_func = process_single_file_opensmile
    elif feature_set == 'mfcc_stats':
        initializer = initialize_worker_mfcc
        initargs = ()
        process_func = process_single_file_mfcc
    else:
        raise ValueError(f"Feature set '{feature_set}' non supportato.")

    with Pool(initializer=initializer, initargs=initargs) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(process_func, audio_files), total=len(audio_files)):
            if result is not None:
                results.append(result)

    if not results:
        print("Nessuna feature estratta.")
        return

    full_df = pd.concat(results, ignore_index=True)
    
    # Pulizia e riordino colonne
    if feature_set in ['egemaps', 'compare']:
        cols_to_drop = ['file', 'start', 'end']
        full_df = full_df.drop(columns=[c for c in cols_to_drop if c in full_df.columns])
    
    cols = ['ID'] + [c for c in full_df.columns if c != 'ID']
    full_df = full_df[cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(output_path, index=False)
    print(f"Feature salvate in: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Estrae feature acustiche manuali in parallelo.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path al file di configurazione.")
    args = parser.parse_args()
    config = load_config(args.config)

    features_root = Path(config.data.features_root)
    features_root.mkdir(exist_ok=True)

    # Train
    train_out = features_root / f"train_{config.feature_extraction.feature_set}.csv"
    extract_and_save(config.data.audio_root, train_out, config)

    # Test
    test_out = features_root / f"test_{config.feature_extraction.feature_set}.csv"
    extract_and_save(config.data.test_audio_root, test_out, config)
    print(f"\nEstrazione completata usando {cpu_count()} processi.")

if __name__ == "__main__":
    main()