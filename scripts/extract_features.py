"""
Extract acoustic features (eGeMAPS, ComParE, MFCC, Disvoice) in parallel.

For Whisper embeddings, use extract_embeddings_batch.py instead (GPU-batched).
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import warnings
from multiprocessing import Pool, cpu_count
import multiprocessing

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config

warnings.filterwarnings("ignore")

# Global extractors initialized per worker process
_opensmile_extractor = None
_disvoice_extractor = None


def _init_opensmile(feature_set):
    global _opensmile_extractor
    import opensmile
    if feature_set == 'egemaps':
        _opensmile_extractor = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    elif feature_set == 'compare':
        _opensmile_extractor = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016, feature_level=opensmile.FeatureLevel.Functionals)


def _init_disvoice():
    global _disvoice_extractor
    try:
        from disvoice.prosody import Prosody
        _disvoice_extractor = Prosody()
    except ImportError:
        print("ERROR: 'disvoice' is not installed.", file=sys.stderr)
        sys.exit(1)


def _process_opensmile(audio_path):
    global _opensmile_extractor
    try:
        feature_df = _opensmile_extractor.process_file(str(audio_path)).reset_index()
        feature_df['ID'] = audio_path.stem
        return feature_df
    except Exception as e:
        print(f"\nWARNING: OpenSMILE error on {audio_path.name}: {e}", file=sys.stderr)
        return None


def _process_mfcc(audio_path):
    try:
        import librosa
        from scipy.stats import skew, kurtosis
        y, sr = librosa.load(str(audio_path), sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        all_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))

        stats = {
            'mean': np.mean(all_features, axis=1),
            'std': np.std(all_features, axis=1),
            'skew': skew(all_features, axis=1),
            'kurtosis': kurtosis(all_features, axis=1),
            'min': np.min(all_features, axis=1),
            'max': np.max(all_features, axis=1),
        }
        feature_names = (
            [f'mfcc_{i}' for i in range(13)]
            + [f'delta_{i}' for i in range(13)]
            + [f'delta2_{i}' for i in range(13)]
        )
        feature_dict = {'ID': audio_path.stem}
        for stat_name, stat_values in stats.items():
            for i, value in enumerate(stat_values):
                feature_dict[f'{feature_names[i]}_{stat_name}'] = value
        return pd.DataFrame([feature_dict])
    except Exception as e:
        print(f"\nWARNING: MFCC error on {audio_path.name}: {e}", file=sys.stderr)
        return None


def _process_disvoice(audio_path):
    global _disvoice_extractor
    try:
        features_dict = _disvoice_extractor.extract_features_file(str(audio_path), static=True)
        feature_df = pd.DataFrame([features_dict])
        feature_df['ID'] = audio_path.stem
        return feature_df
    except Exception as e:
        if "Sound is too short" in str(e):
            print(f"\nINFO: {audio_path.name} too short. Skipped.", file=sys.stderr)
        else:
            print(f"\nWARNING: Disvoice error on {audio_path.name}: {e}", file=sys.stderr)
        return None


# Feature set registry
_FEATURE_SETS = {
    'egemaps':          (_init_opensmile, ('egemaps',), _process_opensmile),
    'compare':          (_init_opensmile, ('compare',), _process_opensmile),
    'mfcc_stats':       (None, (), _process_mfcc),
    'disvoice_prosody': (_init_disvoice, (), _process_disvoice),
}


def extract_and_save(audio_root, output_path, config):
    if output_path.exists() and not config.feature_extraction.overwrite:
        print(f"Feature file already exists: {output_path}")
        return

    audio_files = sorted(list(Path(audio_root).rglob("*.wav")))
    feature_set = config.feature_extraction.feature_set

    if feature_set not in _FEATURE_SETS:
        raise ValueError(
            f"Feature set '{feature_set}' not supported by this script. "
            f"Supported: {list(_FEATURE_SETS.keys())}. "
            f"For Whisper embeddings, use extract_embeddings_batch.py."
        )

    initializer, initargs, process_func = _FEATURE_SETS[feature_set]
    print(f"Extracting '{feature_set}' from {len(audio_files)} files...")

    pool_kwargs = {}
    if initializer:
        pool_kwargs = {'initializer': initializer, 'initargs': initargs}

    with Pool(**pool_kwargs) as pool:
        results = [r for r in tqdm(pool.imap_unordered(process_func, audio_files), total=len(audio_files)) if r is not None]

    if not results:
        print("No features extracted.")
        return

    full_df = pd.concat(results, ignore_index=True)

    if feature_set in ['egemaps', 'compare']:
        cols_to_drop = ['file', 'start', 'end']
        full_df = full_df.drop(columns=[c for c in cols_to_drop if c in full_df.columns])

    full_df = full_df.fillna(0)
    cols = ['ID'] + [c for c in full_df.columns if c != 'ID']
    full_df = full_df[cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(output_path, index=False)
    print(f"Features saved to: {output_path}")


def main():
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Extract acoustic features in parallel.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    features_root = Path(config.data.features_root)
    features_root.mkdir(exist_ok=True)
    feature_set_name = config.feature_extraction.feature_set

    train_out = features_root / f"train_{feature_set_name}.csv"
    extract_and_save(config.data.audio_root, train_out, config)

    test_out = features_root / f"test_{feature_set_name}.csv"
    extract_and_save(config.data.test_audio_root, test_out, config)

    print(f"\nExtraction complete ({cpu_count()} processes).")


if __name__ == "__main__":
    main()
