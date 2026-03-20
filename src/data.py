from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
import numpy as np
from src.utils import load_audio, load_labels_df, find_diagnosis_column, map_label


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config, tokenizer):
        self.df = df
        self.config = config
        self.tokenizer = tokenizer
        self.is_test = 'label' not in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = str(row.name)

        if not self.is_test:
            transcript_dir = Path(self.config.data.transcripts_root)
        else:
            transcript_dir = Path(self.config.data.test_transcripts_root)

        specific_dir = transcript_dir / self.config.transcription_model_for_training
        if specific_dir.exists():
            transcript_path = specific_dir / f"{file_id}.txt"
        else:
            transcript_path = transcript_dir / f"{file_id}.txt"

        try:
            text = transcript_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            text = ""

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.config.model.text.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "id": file_id,
        }

        if not self.is_test and 'label' in row:
            try:
                val = float(row['label'])
                dtype = torch.long if self.config.task == 'classification' else torch.float32
                item['labels'] = torch.tensor(val, dtype=dtype)
            except (ValueError, TypeError):
                pass

        return item


class AudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config):
        self.df = df
        self.config = config
        self.is_test = 'label' not in df.columns
        self.audio_root = Path(config.data.audio_root if not self.is_test else config.data.test_audio_root)
        self.sample_rate = config.model.audio.sample_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = str(row.name)
        audio_path = self.audio_root / f"{file_id}.wav"

        if not audio_path.exists():
            waveform = torch.zeros(self.sample_rate)
        else:
            waveform, _ = load_audio(audio_path, target_sr=self.sample_rate)

        item = {"waveform": waveform, "id": file_id}

        if not self.is_test and 'label' in row:
            try:
                val = float(row['label'])
                dtype = torch.long if self.config.task == 'classification' else torch.float32
                item['labels'] = torch.tensor(val, dtype=dtype)
            except (ValueError, TypeError):
                pass

        return item


def collate_audio(batch):
    waveforms = [item['waveform'] for item in batch]
    ids = [item['id'] for item in batch]
    padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    collated_batch = {"waveform": padded_waveforms, "id": ids}
    if 'labels' in batch[0]:
        labels = torch.stack([item['labels'] for item in batch])
        collated_batch['labels'] = labels
    return collated_batch


def get_data_splits(config):
    """Prepare DataFrames for training, validation and test with K-Fold."""

    train_df = load_labels_df(config.data.train_labels)
    train_df = train_df.set_index('ID')

    test_df = load_labels_df(config.data.test_task1_labels)
    test_df = test_df.rename(columns={'ID': 'id'}).set_index('id')

    if config.task == 'classification':
        dx_col = find_diagnosis_column(train_df)
        if dx_col:
            train_df['label'] = train_df[dx_col].apply(map_label)

        dx_col_test = find_diagnosis_column(test_df)
        if dx_col_test:
            test_df['label'] = test_df[dx_col_test].apply(map_label)
        else:
            if 'label' in test_df.columns:
                test_df = test_df.drop(columns=['label'])
    else:
        if 'mmse' in train_df.columns:
            train_df['label'] = train_df['mmse']
        if 'mmse' in test_df.columns:
            test_df['label'] = test_df['mmse']

    train_df = train_df.sample(frac=1, random_state=config.seed)

    skf = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)
    y_strat = train_df['label'] if 'label' in train_df.columns else [0] * len(train_df)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_strat)):
        train_split = train_df.iloc[train_idx]
        val_split = train_df.iloc[val_idx]
        yield fold, train_split, val_split, test_df


def get_dataloaders(config, train_df, val_df, test_df=None):
    if config.modality == 'text':
        tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
        train_dataset = TextDataset(train_df, config, tokenizer)
        val_dataset = TextDataset(val_df, config, tokenizer)
        collate_fn = None
        if test_df is not None:
            test_dataset = TextDataset(test_df, config, tokenizer)

    elif config.modality == 'audio':
        train_dataset = AudioDataset(train_df, config)
        val_dataset = AudioDataset(val_df, config)
        collate_fn = collate_audio
        if test_df is not None:
            test_dataset = AudioDataset(test_df, config)
    else:
        raise ValueError(f"Modality '{config.modality}' not supported.")

    bs = config.training.batch_size
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=2)

    if test_df is not None:
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=2)
        return train_loader, val_loader, test_loader

    return train_loader, val_loader
