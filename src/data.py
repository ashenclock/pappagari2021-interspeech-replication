from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
import torchaudio
from torchaudio.functional import resample

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
        file_id = row.name
        
        transcript_dir = Path(self.config.data.transcripts_root if not self.is_test else self.config.data.test_transcripts_root)
        transcript_path = transcript_dir / self.config.transcription_model_for_training / f"{file_id}.txt"
        
        try:
            text = transcript_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            print(f"ATTENZIONE: Trascrizione per {file_id} non trovata. Uso una stringa vuota.")
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

        if not self.is_test:
            target = row['label']
            dtype = torch.long if self.config.task == 'classification' else torch.float32
            item['labels'] = torch.tensor(target, dtype=dtype)
        
        return item

class AudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config):
        self.df = df
        self.config = config
        self.is_test = 'label' not in df.columns
        self.audio_root = Path(self.config.data.audio_root if not self.is_test else self.config.data.test_audio_root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = row.name
        
        audio_path = self.audio_root / f"{file_id}.wav"
        if not audio_path.exists() and not self.is_test: # Cerca nelle sottocartelle per il training set
            group = 'ad' if row['label'] == 1 else 'cn'
            audio_path = self.audio_root / group / f"{file_id}.wav"

        waveform, sr = torchaudio.load(audio_path)
        
        if sr != self.config.model.audio.sample_rate:
            waveform = resample(waveform, sr, self.config.model.audio.sample_rate)
        
        item = {"waveform": waveform.squeeze(0), "id": file_id}

        if not self.is_test:
            target = row['label']
            dtype = torch.long if self.config.task == 'classification' else torch.float32
            item['labels'] = torch.tensor(target, dtype=dtype)
            
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
    """Prepara i DataFrame per training, validazione e test."""
    # Carica etichette di training
    train_df = pd.read_csv(config.data.train_labels)
    train_df = train_df.rename(columns={'adressfname': 'ID', 'dx': 'diagnosis'})
    train_df = train_df.set_index('ID')
    
    if config.task == 'classification':
        train_df['label'] = train_df['diagnosis'].apply(lambda x: 1 if x == 1 else 0)
    else: # regression
        train_df['label'] = train_df['mmse']
        
    # Carica etichette di test
    label_file = config.data.test_task1_labels if config.task == 'classification' else config.data.test_task2_labels
    test_df = pd.read_csv(label_file).rename(columns={'ID': 'id'}).set_index('id')

    # Crea gli split per il cross-validation
    skf = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)
    
    # Per la stratificazione usiamo le etichette binarie anche per la regressione
    stratify_labels = train_df['diagnosis'].apply(lambda x: 1 if x == 1 else 0)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, stratify_labels)):
        train_split = train_df.iloc[train_idx]
        val_split = train_df.iloc[val_idx]
        yield fold, train_split, val_split, test_df
        
def get_dataloaders(config, train_df, val_df, test_df=None):
    """Crea i DataLoaders per un dato split."""
    
    if config.modality == 'text':
        tokenizer = AutoTokenizer.from_pretrained(config.model.text.name)
        train_dataset = TextDataset(train_df, config, tokenizer)
        val_dataset = TextDataset(val_df, config, tokenizer)
        collate_fn = None
    elif config.modality == 'audio':
        train_dataset = AudioDataset(train_df, config)
        val_dataset = AudioDataset(val_df, config)
        collate_fn = collate_audio
    else:
        raise ValueError(f"Modalità '{config.modality}' non supportata.")

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_fn)
    
    if test_df is not None:
        if config.modality == 'text':
            test_dataset = TextDataset(test_df, config, tokenizer)
        else:
            test_dataset = AudioDataset(test_df, config)
        test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_fn)
        return train_loader, val_loader, test_loader
        
    return train_loader, val_loader