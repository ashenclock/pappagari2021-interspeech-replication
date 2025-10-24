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
    # --- QUESTA CLASSE È CORRETTA E RIMANE INVARIATA ---
    def __init__(self, df: pd.DataFrame, config, tokenizer):
        self.df = df; self.config = config; self.tokenizer = tokenizer; self.is_test = 'label' not in df.columns
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = row.name
        transcript_dir = Path(self.config.data.transcripts_root if not self.is_test else self.config.data.test_transcripts_root)
        transcript_path = transcript_dir / self.config.transcription_model_for_training / f"{file_id}.txt"
        try: text = transcript_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError: text = ""
        inputs = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.config.model.text.max_length,
            padding="max_length", truncation=True, return_attention_mask=True, return_tensors="pt",
        )
        item = {"input_ids": inputs["input_ids"].flatten(), "attention_mask": inputs["attention_mask"].flatten(), "id": file_id}
        if not self.is_test:
            item['labels'] = torch.tensor(row['label'], dtype=torch.long if self.config.task == 'classification' else torch.float32)
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
        if not audio_path.exists() and not self.is_test:
            group = 'ad' if row['label'] == 1 else 'cn'
            audio_path = self.audio_root / group / f"{file_id}.wav"

        waveform, sr = torchaudio.load(audio_path)
        
        # --- FIX #1: FORZARE L'AUDIO MONO ---
        # Se la waveform ha più di un canale (es. è stereo), ne facciamo la media
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        # ------------------------------------

        if sr != self.config.model.audio.sample_rate:
            waveform = resample(waveform, sr, self.config.model.audio.sample_rate)
        
        # Ora .squeeze() funzionerà sempre, perché l'input è sempre [1, n_samples]
        item = {"waveform": waveform.squeeze(0), "id": file_id}

        if not self.is_test:
            target = row['label']
            dtype = torch.long if self.config.task == 'classification' else torch.float32
            item['labels'] = torch.tensor(target, dtype=dtype)
            
        return item

def collate_audio(batch):
    waveforms = [item['waveform'] for item in batch]
    ids = [item['id'] for item in batch]
    
    # Eseguiamo il padding delle waveform
    padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    
    # --- FIX: Rimuoviamo il calcolo delle 'lengths' per semplicità ---
    collated_batch = {"waveform": padded_waveforms, "id": ids}
    
    if 'labels' in batch[0]:
        labels = torch.stack([item['labels'] for item in batch])
        collated_batch['labels'] = labels
        
    return collated_batch



def get_data_splits(config):
    """Prepara i DataFrame per training, validazione e test."""
    train_df = pd.read_csv(config.data.train_labels)
    # Rinominiamo la colonna ID in modo standard
    train_df = train_df.rename(columns={'adressfname': 'ID'})
    train_df = train_df.set_index('ID')
    
    # --- FIX CRUCIALE: LOGICA ROBUSTA PER TROVARE LA COLONNA DELLA DIAGNOSI ---
    diagnosis_col = None
    possible_cols = ['dx', 'diagnosis', 'Dx'] # Lista di nomi di colonna comuni per la diagnosi
    for col in possible_cols:
        if col in train_df.columns:
            diagnosis_col = col
            break
            
    if diagnosis_col is None:
        raise ValueError(f"Impossibile trovare la colonna della diagnosi. Colonne disponibili: {train_df.columns}")
        
    print(f"Info: Trovata colonna diagnosi '{diagnosis_col}'. Verrà usata per creare le etichette.")
    # --- FINE FIX ---

    if config.task == 'classification':
        # Usiamo la colonna trovata per creare le etichette
        train_df['label'] = train_df[diagnosis_col].apply(lambda x: 1 if x == 'ad' else 0)
    else:
        # Per la regressione, assumiamo che la colonna mmse esista
        if 'mmse' not in train_df.columns:
            raise ValueError(f"Task di regressione selezionato ma colonna 'mmse' non trovata. Colonne: {train_df.columns}")
        train_df['label'] = train_df['mmse']
        
    train_df = train_df.sample(frac=1, random_state=config.seed)
    
    test_df = pd.read_csv(config.data.test_task1_labels).rename(columns={'ID': 'id'}).set_index('id')

    skf = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        train_split = train_df.iloc[train_idx]
        val_split = train_df.iloc[val_idx]
        yield fold, train_split, val_split, test_df
        
def get_dataloaders(config, train_df, val_df, test_df=None):
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
        test_dataset = TextDataset(test_df, config, tokenizer) if config.modality == 'text' else AudioDataset(test_df, config)
        test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_fn)
        return train_loader, val_loader, test_loader
        
    return train_loader, val_loader