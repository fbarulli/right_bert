"""
BERT model implementation for embedding learning through masked language modeling.
"""
from src.embedding.imports import (
    dataclass,
    torch,
    nn,
    BertPreTrainedModel,
    BertModel,
    BertConfig,
    PreTrainedModel,
    Dict, Any, Optional, Tuple, Union, cast,
    Tensor,
    optuna,
    logger,
    log_function,
    LogConfig,
)

@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding model."""
    name: str
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float
    tie_weights: bool = True
    log_level: str = 'log'

class EmbeddingBert(BertPreTrainedModel):
    """BERT model for learning embeddings through masked token prediction."""

    def __init__(
        self,
        config: BertConfig,
        tie_weights: bool = True,
        log_config: Optional[LogConfig] = None
    ) -> None:
        """
        Initialize model.

        Args:
            config: BertConfig instance with model parameters
            tie_weights: Whether to tie embedding and output weights
            log_config: Logging configuration
        """
        super().__init__(config)
        self.log_config = log_config or LogConfig()

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

    @log_function()
    def get_output_embeddings(self) -> nn.Linear:
        """Get output embeddings layer."""
        return self.cls.predictions.decoder

    @log_function()
    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        """Set output embeddings layer."""
        self.cls.predictions.decoder = new_embeddings

    @log_function()
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True
    ) -> Union[Tuple[Tensor, ...], Dict[str, Tensor]]:
        """Forward pass through the model."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True
        )

        sequence_output = outputs.last_hidden_state
        prediction_scores = self.cls(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

            active_loss = labels.view(-1) != -100
            if active_loss.any():
                active_logits = prediction_scores.view(-1, self.config.vocab_size)[active_loss]
                active_labels = labels.view(-1)[active_loss]
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

    def __init__(self, config: BertConfig) -> None:
        """Initialize embedding prediction head."""
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    @log_function()
    def forward(self, sequence_output: Tensor) -> Tensor:
        """Forward pass through the head."""
        return self.predictions(sequence_output)

class BertLMPredictionHead(nn.Module):
    """BERT language model prediction head."""

    def __init__(self, config: BertConfig) -> None:
        """Initialize prediction head."""
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    @log_function()
    def forward(self, hidden_states: Tensor) -> Tensor:
        """Forward pass through the prediction head."""
        hidden_states = self.transform(hidden_states)
        return self.decoder(hidden_states)

class BertPredictionHeadTransform(nn.Module):
    """BERT prediction head transform."""

    def __init__(self, config: BertConfig) -> None:
        """Initialize transform layer."""
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    @log_function()
    def forward(self, hidden_states: Tensor) -> Tensor:
        """Forward pass through the transform."""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

@log_function()
def embedding_model_factory(
    config: Dict[str, Any],
    trial: Optional[optuna.Trial] = None
) -> EmbeddingBert:
    """Factory function for creating an EmbeddingBert model."""
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

    log_config = LogConfig(level=config['training'].get('log_level', 'log'))

    model = EmbeddingBert(
        config=model_config,
        tie_weights=config['model']['tie_weights'],
        log_config=log_config
    )
    return cast(EmbeddingBert, model)

__all__ = [
    'EmbeddingBert',
    'EmbeddingModelConfig',
    'embedding_model_factory',
]