
# src/embedding/model.py
# src/embedding/model.py (CONFIRM ABSOLUTE IMPORTS)
from __future__ import annotations
import logging
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from typing import Dict, Any, Optional, Tuple, Union
import optuna

from src.common import (
    get_cuda_manager,
    get_batch_manager,
    get_tensor_manager
)


logger = logging.getLogger(__name__)

class EmbeddingBert(BertPreTrainedModel):
    """BERT model for learning embeddings through masked token prediction."""

    def __init__(
        self,
        config: BertConfig,
        tie_weights: bool = True
    ):
        """
        Initialize model.
        """
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertEmbeddingHead(config)

        self.post_init()

        if tie_weights:
            self._tie_or_clone_weights(
                self.cls.predictions.decoder,
                self.bert.embeddings.word_embeddings
            )

        logger.info(
            f"Initialized EmbeddingBert with:\n"
            f"- Hidden size: {config.hidden_size}\n"
            f"- Vocab size: {config.vocab_size}\n"
            f"- Tied weights: {tie_weights}"
        )

    def get_output_embeddings(self) -> nn.Linear:
        """Get output embeddings layer."""
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        """Set output embeddings layer."""
        self.cls.predictions.decoder = new_embeddings

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
        """
        Forward pass.
        """
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

        inputs = batch_manager.prepare_batch(inputs, self.device)

        outputs = self.bert(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids= inputs.get('token_type_ids'),
            position_ids= inputs.get('position_ids'),
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True
        )

        sequence_output = outputs.last_hidden_state
        prediction_scores = self.cls(sequence_output)

        loss = None
        if 'labels' in inputs:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

            active_loss = inputs['labels'].view(-1) != -100
            if active_loss.any():
                active_logits = prediction_scores.view(-1, self.config.vocab_size)[active_loss]
                active_labels = inputs['labels'].view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = torch.tensor(0.0, device=self.device)

        if return_dict:
            return {
                'loss': loss,
                'logits': prediction_scores,
                'hidden_states': outputs.hidden_states if output_hidden_states else None,
                'attentions': outputs.attentions if output_attentions else None
            }

        output = (prediction_scores,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

class BertEmbeddingHead(nn.Module):
    """BERT embedding prediction head with proper initialization."""

    def __init__(self, config: BertConfig):
        """
        Initialize embedding prediction head.
        """
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertLMPredictionHead(nn.Module):
    """BERT language model prediction head."""

    def __init__(self, config: BertConfig):
        """
        Initialize prediction head.
        """
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertPredictionHeadTransform(nn.Module):
    """BERT prediction head transform."""

    def __init__(self, config: BertConfig):
        """
        Initialize transform layer.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        if isinstance(config.hidden_act, str):
            self.transform_act_fn = nn.GELU()
        else:
            self.transform_act_fn = config.hidden_act

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

def embedding_model_factory(config: Dict[str, Any], trial: Optional[optuna.Trial] = None) -> EmbeddingBert:
    """
    Factory function for creating an EmbeddingBert model.
    """
    model_config = BertConfig.from_pretrained(config['model']['name'])

    if trial:
        model_config.hidden_dropout_prob = trial.suggest_float(
            "hidden_dropout_prob",
            config['hyperparameters']['hidden_dropout_prob']['min'],
            config['hyperparameters']['hidden_dropout_prob']['max']
        )
        model_config.attention_probs_dropout_prob = trial.suggest_float(
            "attention_probs_dropout_prob",
            config['hyperparameters']['attention_probs_dropout_prob']['min'],
            config['hyperparameters']['attention_probs_dropout_prob']['max']
        )
    else:
        model_config.hidden_dropout_prob = config['model']['hidden_dropout_prob']
        model_config.attention_probs_dropout_prob = config['model']['attention_probs_dropout_prob']


    model = EmbeddingBert(config=model_config, tie_weights= config['model']['tie_weights'])
    return model