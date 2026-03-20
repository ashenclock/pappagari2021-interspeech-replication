"""
GPU-batched Whisper embedding extraction.

Use this script when feature_extraction.feature_set matches embedding_extraction.name
in config.yaml (e.g. 'whisper_large_v3').
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import warnings
import torch
from transformers import AutoProcessor, AutoModel
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.utils import clear_memory, load_audio

warnings.filterwarnings("ignore")


class AudioFileDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        try:
            waveform, _ = load_audio(audio_path, target_sr=16000)
            return {"waveform": waveform, "id": audio_path.stem}
        except Exception as e:
            print(f"Error loading {audio_path.name}: {e}", file=sys.stderr)
            return {"waveform": torch.tensor([]), "id": "error"}


def collate_fn(batch):
    """Convert waveforms to numpy arrays as expected by the HF processor."""
    batch = [item for item in batch if item['id'] != "error"]
    if not batch:
        return None
    waveforms = [item['waveform'].numpy() for item in batch]
    ids = [item['id'] for item in batch]
    return {"waveforms": waveforms, "ids": ids}


class EmbeddingExtractor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.model_id = config.embedding_extraction.model_id
        self.pooling = config.embedding_extraction.pooling_strategy
        self.batch_size = config.embedding_extraction.batch_size

        print(f"Loading model '{self.model_id}' on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        print("Model loaded.")

    def extract(self, audio_root: Path, output_path: Path):
        if output_path.exists() and not self.config.feature_extraction.overwrite:
            print(f"Feature file already exists, skipping: {output_path}")
            return

        audio_files = sorted(list(audio_root.rglob("*.wav")))
        if not audio_files:
            print(f"No .wav files found in {audio_root}")
            return

        dataset = AudioFileDataset(audio_files)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

        all_embeddings = []
        all_ids = []

        print(f"Extracting embeddings from {len(audio_files)} files (batch size: {self.batch_size})...")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                if batch is None:
                    continue

                inputs = self.processor(batch['waveforms'], return_tensors="pt", sampling_rate=16000, padding=True)
                input_features = inputs.input_features.to(self.device)
                last_hidden_state = self.model.encoder(input_features).last_hidden_state

                if self.pooling == "mean":
                    pooled_output = last_hidden_state.mean(dim=1)
                else:
                    raise ValueError(f"Pooling strategy '{self.pooling}' not supported.")

                all_embeddings.append(pooled_output.cpu().numpy())
                all_ids.extend(batch['ids'])

        if not all_ids:
            print("ERROR: No embeddings extracted.")
            return

        embeddings_matrix = np.vstack(all_embeddings)
        feature_names = [f"whisper_{i}" for i in range(embeddings_matrix.shape[1])]

        df = pd.DataFrame(embeddings_matrix, columns=feature_names)
        df.insert(0, 'ID', all_ids)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Features saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract audio embeddings in batch using GPU.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    feature_set_name = config.feature_extraction.feature_set
    if not hasattr(config, 'embedding_extraction') or feature_set_name != config.embedding_extraction.name:
        print(f"ERROR: This script is for embedding extraction only (e.g. 'whisper_large_v3').")
        print(f"Current feature_set is '{feature_set_name}'. Use extract_features.py for acoustic features.")
        sys.exit(1)

    clear_memory()
    extractor = EmbeddingExtractor(config)

    features_root = Path(config.data.features_root)

    print("\n--- Processing training set ---")
    extractor.extract(Path(config.data.audio_root), features_root / f"train_{feature_set_name}.csv")

    print("\n--- Processing test set ---")
    extractor.extract(Path(config.data.test_audio_root), features_root / f"test_{feature_set_name}.csv")

    print("\nExtraction complete.")


if __name__ == "__main__":
    main()
