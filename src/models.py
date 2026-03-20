import torch
import torch.nn as nn
from transformers import AutoModel
from speechbrain.inference.classifiers import EncoderClassifier


class BertClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.model.text.name)
        self.dropout = nn.Dropout(config.model.text.dropout)
        num_classes = 2 if config.task == 'classification' else 1
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, batch):
        outputs = self.bert(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        return logits if self.classifier.out_features > 1 else logits.squeeze(-1)


class EcapaTdnnClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = EncoderClassifier.from_hparams(
            source=config.model.audio.pretrained,
            run_opts={"device": config.device},
        )

        if not config.model.audio.trainable_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        with torch.no_grad():
            dummy_input = torch.zeros(1, config.model.audio.sample_rate, device=config.device)
            dummy_embedding = self.encoder.encode_batch(dummy_input)
            num_features = dummy_embedding.shape[-1]

        num_classes = 2 if config.task == 'classification' else 1
        dropout = config.model.audio.dropout

        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_features // 2, num_classes),
        )

    def forward(self, batch):
        waveforms = batch['waveform']
        is_trainable = any(p.requires_grad for p in self.encoder.parameters())
        with torch.set_grad_enabled(is_trainable):
            embeddings = self.encoder.encode_batch(waveforms)

        embeddings = embeddings.squeeze(1)
        outputs = self.classifier_head(embeddings)
        return outputs if self.classifier_head[-1].out_features > 1 else outputs.squeeze(-1)


def build_model(config):
    if config.modality == 'text':
        return BertClassifier(config)
    elif config.modality == 'audio':
        return EcapaTdnnClassifier(config)
    else:
        raise ValueError(f"Modality '{config.modality}' not supported.")
