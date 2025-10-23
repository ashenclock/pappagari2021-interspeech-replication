import torch
import torch.nn as nn
from transformers import AutoModel
from speechbrain.pretrained import EncoderClassifier

class BertClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.model.text.name)
        self.dropout = nn.Dropout(config.model.text.dropout)
        
        num_classes = 2 if config.task == 'classification' else 1
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits.squeeze(-1)

class EcapaTdnnClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = EncoderClassifier.from_hparams(source=config.model.audio.pretrained)
        
        for param in self.encoder.parameters():
            param.requires_grad = config.model.audio.trainable_encoder

        num_features = self.encoder.mods.classifier.in_features
        num_classes = 2 if config.task == 'classification' else 1
        
        self.encoder.mods.classifier = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.ReLU(),
            nn.Dropout(config.model.audio.dropout),
            nn.Linear(num_features // 2, num_classes)
        )

    def forward(self, batch):
        waveforms = batch['waveform']
        wav_lens = torch.ones(waveforms.shape[0], device=waveforms.device)
        
        embeddings = self.encoder.encode_batch(waveforms, wav_lens)
        # Squeeze per rimuovere la dimensione temporale (che è 1)
        embeddings = embeddings.squeeze(1) 
        outputs = self.encoder.mods.classifier(embeddings)
        
        return outputs.squeeze(-1)

def build_model(config):
    """Factory function per costruire il modello in base alla configurazione."""
    if config.modality == 'text':
        return BertClassifier(config)
    elif config.modality == 'audio':
        return EcapaTdnnClassifier(config)
    else:
        raise ValueError(f"Modello per la modalità '{config.modality}' non supportato.")