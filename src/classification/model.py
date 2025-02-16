# src/classification/model.py
from __future__ import annotations

import logging
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from typing import Dict, Any, Optional, Tuple, Union
import optuna

from src.common.managers import (
    get_cuda_manager,
    get_batch_manager,
    get_tensor_manager
)

logger = logging.getLogger(__name__)

class ClassificationBert(BertPreTrainedModel):
    """BERT model for classification tasks."""

    def __init__(
        self,
        config: BertConfig,
        num_labels: int = 2,
        dropout_prob: float = 0.1,
        hidden_size: Optional[int] = None
    ):
        """Initialize model."""
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        bert_hidden_size = config.hidden_size
        if hidden_size is None:
            hidden_size = bert_hidden_size

        self.dropout = nn.Dropout(dropout_prob)
        if hidden_size != bert_hidden_size:
            self.intermediate = nn.Linear(bert_hidden_size, hidden_size)
            self.activation = nn.GELU()
        else:
            self.intermediate = None
        self.classifier = nn.Linear(hidden_size, num_labels)

        self._init_weights(self.classifier)
        if self.intermediate is not None:
            self._init_weights(self.intermediate)

        logger.info(
            f"Initialized ClassificationBert with:\n"
            f"- BERT hidden size: {bert_hidden_size}\n"
            f"- Classifier hidden size: {hidden_size}\n"
            f"- Number of labels: {num_labels}\n"
            f"- Dropout probability: {dropout_prob}"
        )

    def _init_weights(self, module):
        """ Initialize the weights (identical to BERT's initialization)."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True
    ) -> Union[Tuple, Dict[str, torch.Tensor]]:

        cuda_manager = get_cuda_manager()
        device = cuda_manager.get_device()
        batch_manager = get_batch_manager()
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        if token_type_ids is not None:
            inputs['token_type_ids'] = token_type_ids
        if position_ids is not None:
            inputs['position_ids'] = position_ids
        if labels is not None:
            inputs['labels'] = labels

        inputs = batch_manager.prepare_batch(inputs, device)

        outputs = self.bert(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids= inputs.get('token_type_ids'),
            position_ids= inputs.get('position_ids'),
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True
        )

        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)

        if self.intermediate is not None:
            pooled_output = self.intermediate(pooled_output)
            pooled_output = self.activation(pooled_output)
            pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        loss = None
        if 'labels' in inputs:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), inputs['labels'].view(-1))

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if output_hidden_states else None,
            'attentions': outputs.attentions if output_attentions else None
        }
def classification_model_factory(config: Dict[str, Any], trial: Optional[optuna.Trial] = None) -> ClassificationBert:
    """
    Factory function for creating a ClassificationBert model.
    Args:
        config: Model configuration.
        trial: Optuna trial object for hyperparameter optimization.
    Returns:
        ClassificationBert: The created ClassificationBert model.
    """

    if trial:
        # Suggest hyperparameters from the ranges defined in config
        hidden_dropout_prob = trial.suggest_float(
            "hidden_dropout_prob",
            config['hyperparameters']['hidden_dropout_prob']['min'],
            config['hyperparameters']['hidden_dropout_prob']['max']
        )
        attention_probs_dropout_prob = trial.suggest_float(
            "attention_probs_dropout_prob",
            config['hyperparameters']['attention_probs_dropout_prob']['min'],
            config['hyperparameters']['attention_probs_dropout_prob']['max']
        )
    else:
        hidden_dropout_prob = config['model']['hidden_dropout_prob']
        attention_probs_dropout_prob = config['model']['attention_probs_dropout_prob']

    model_config = BertConfig.from_pretrained(config['model']['name'])
    model_config.hidden_dropout_prob = hidden_dropout_prob
    model_config.attention_probs_dropout_prob = attention_probs_dropout_prob

    model = ClassificationBert(
        config=model_config,
        num_labels=config['model']['num_labels']
    )
    return model


__all__ = ['ClassificationBert', 'classification_model_factory']