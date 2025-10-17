#!/usr/bin/env python3
"""
BERT model with classification/regression head.
"""
from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
from transformers import AutoModel

from src.config import Config


class BertClassifier(nn.Module):
    """
    A BERT-based model for classification or regression.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        task: str = "classification",
        model_type: str = "simple",
        num_classes: int = 2
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_config.bert_model)
        self.dropout = nn.Dropout(model_config.dropout)
        self.task = task
        self.model_type = model_type

        output_size = 1 if task == "regression" else num_classes

        if self.model_type == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, model_config.mlp_hidden_size),
                nn.ReLU(),
                nn.Dropout(model_config.dropout),
                nn.Linear(model_config.mlp_hidden_size, output_size)
            )
        else:  # simple
            self.classifier = nn.Linear(self.bert.config.hidden_size, output_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        
        logits = self.classifier(self.dropout(pooled_output))
        
        if self.task == "regression":
            return logits.squeeze(-1)
        return logits


def create_model(config: Dict, device: str = "cuda") -> nn.Module:
    """
    Factory function to create model based on task.
    
    Args:
        config: Configuration dictionary
        device: Device to place model on
    
    Returns:
        Model (BERTClassifier or BERTRegressor)
    """
    task = config['training']['task']
    bert_model_name = config['model']['bert_model']
    dropout = config['model']['dropout']
    
    model = BertClassifier(
        model_config=config['model'],
        task=task,
        num_classes=2 if task == "classification" else None
    )
    
    return model.to(device)
