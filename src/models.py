import torch
import torch.nn as nn
from transformers import AutoModel
from speechbrain.pretrained import EncoderClassifier
import sys
# ... (la classe BertClassifier rimane invariata) ...
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
        if self.classifier.out_features > 1:
            return logits
        else:
            return logits.squeeze(-1)


class EcapaTdnnClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # --- FIX: Carica l'encoder direttamente sul device specificato in config.yaml ---
        # Questo è il metodo più sicuro per i modelli pre-addestrati complessi.
        self.encoder = EncoderClassifier.from_hparams(
            source=config.model.audio.pretrained,
            run_opts={"device": config.device}
        )
        
        if not config.model.audio.trainable_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Ottiene la dimensione dell'embedding in modo sicuro
        with torch.no_grad():
            # Assicurati che anche il tensore fittizio sia sul device corretto
            dummy_input = torch.zeros(1, config.model.audio.sample_rate, device=config.device)
            dummy_embedding = self.encoder.encode_batch(dummy_input)
            num_features = dummy_embedding.shape[-1]
        
        num_classes = 2 if config.task == 'classification' else 1
        
        self.classifier_head = nn.Sequential(
            nn.Dropout(config.model.audio.dropout),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, batch):
        waveforms = batch['waveform']
        
        is_trainable = any(p.requires_grad for p in self.encoder.parameters())
        with torch.set_grad_enabled(is_trainable):
            # Non è necessario passare 'lengths', encode_batch gestisce il padding
            embeddings = self.encoder.encode_batch(waveforms)

        embeddings = embeddings.squeeze(1) 
        outputs = self.classifier_head(embeddings)
        
        if self.classifier_head[-1].out_features > 1:
            return outputs
        else:
            return outputs.squeeze(-1)

def build_model(config):
    if config.modality == 'text':
        return BertClassifier(config)
    elif config.modality == 'audio':
        return EcapaTdnnClassifier(config)
    else:
        raise ValueError(f"Modello per la modalità '{config.modality}' non supportato.")
