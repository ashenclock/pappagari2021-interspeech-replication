#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.config import Config


class TranscriptionDataset(Dataset):
    """
    PyTorch Dataset for loading transcripts and labels.
    """
    def __init__(self, file_ids: List[str], labels: pd.DataFrame, transcript_dir: Path, tokenizer: PreTrainedTokenizer, max_length: int, task: str = "classification"):
        self.file_ids = file_ids
        self.labels_df = labels
        self.transcript_dir = transcript_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, idx: int) -> dict:
        file_id = self.file_ids[idx]
        label_info = self.labels_df.loc[file_id]
        
        group = label_info['group']
        transcript_path = self.transcript_dir / group / f"{file_id}.txt"
        
        try:
            text = transcript_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            print(f"Warning: Transcript not found for {file_id}, using empty string.")
            text = ""

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        label_tensor = torch.tensor(label_info['label'], dtype=torch.float32)
        if self.task == "classification":
            label_tensor = label_tensor.long()

        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": label_tensor
        }


def load_and_prepare_data(data_config: Config, task: str) -> Tuple[pd.DataFrame, Path]:
    """
    Loads labels and identifies the correct transcript directory.
    """
    labels_df = pd.read_csv(data_config.train_labels)
    labels_df = labels_df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    labels_df = labels_df.rename(columns={"dx": "diagnosis"})
    
    # Use 'ID' if 'file_name' is not present, also check for case variations
    id_col = None
    if 'file_name' in labels_df.columns:
        id_col = 'file_name'
    elif 'ID' in labels_df.columns:
        id_col = 'ID'
    else:
        # Try to find a column that looks like an ID column
        for col in labels_df.columns:
            lower_col = col.lower()
            if 'file_name' in lower_col or 'id' in lower_col or 'fname' in lower_col:
                id_col = col
                break

    # Common fallback for ADReSSo21 naming convention
    if not id_col and 'adressfname' in labels_df.columns:
        id_col = 'adressfname'

    if id_col:
        labels_df['file_id'] = labels_df[id_col].astype(str).str.replace('.wav', '', regex=False)
    else:
        available_cols = ', '.join(str(col) for col in labels_df.columns)
        raise ValueError(
            "Label file must contain an identifier column (e.g. 'file_name', 'ID', 'adressfname'). "
            f"Columns found: {available_cols}"
        )

    labels_df = labels_df.set_index('file_id')
    
    # Determine the group ('ad' or 'cn') for path lookup
    if 'diagnosis' not in labels_df.columns:
        raise ValueError("Label file must contain a 'diagnosis' or 'dx' column for path lookup.")

    def determine_group(value):
        if isinstance(value, str):
            norm = value.strip().lower()
            if norm in {"ad", "1", "alz", "alzheimers", "positive"}:
                return 'ad'
            if norm in {"cn", "0", "control", "healthy", "negative"}:
                return 'cn'
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None:
            if numeric == 1:
                return 'ad'
            if numeric == 0:
                return 'cn'
        # Default to 'cn' if unsure to avoid crashing but warn the user
        print(f"Warning: Unrecognized diagnosis value '{value}', defaulting to 'cn'.")
        return 'cn'

    labels_df['group'] = labels_df['diagnosis'].apply(determine_group)

    # Handle both classification and regression labels based on the task
    if task == "classification":
        if 'diagnosis' not in labels_df.columns:
            raise ValueError("For classification, the label file must contain a 'diagnosis' or 'dx' column.")
        labels_df['label'] = labels_df['group'].map({'ad': 1, 'cn': 0}).astype(int)
    elif task == "regression":
        if 'mmse' not in labels_df.columns:
            raise ValueError("For regression, the label file must contain an 'mmse' column.")
        labels_df['label'] = labels_df['mmse']
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'classification' or 'regression'.")

    transcript_dir = Path(data_config.transcripts_root) / data_config.transcription_model
    if not transcript_dir.exists():
        raise FileNotFoundError(
            f"Transcription directory not found: {transcript_dir}\n"
            f"Please run the transcription script first for model '{data_config.transcription_model}'."
        )
        
    return labels_df[['label', 'group']], transcript_dir
