"""
Loss functions for embedding learning, including InfoNCE loss implementation.
"""
from src.embedding.imports import (
    torch,
    nn,
    F,
    dataclass,
    Dict, Optional, Any, cast,
    Tensor,
    logger,
    log_function,
    LogConfig,
)

@dataclass
class InfoNCEConfig:
    """Configuration for InfoNCE loss."""
    temperature: float
    reduction: str
    contrast_mode: str
    chunk_size: int
    log_level: str = 'log'

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.contrast_mode != 'all':
            raise ValueError("Only contrast_mode='all' is currently supported")
        if not 0 < self.temperature < 1:
            raise ValueError(f"Temperature must be between 0 and 1, got {self.temperature}")
        if self.reduction not in ['mean', 'sum']:
            raise ValueError(f"Reduction must be 'mean' or 'sum', got {self.reduction}")
        if self.chunk_size < 1:
            raise ValueError(f"Chunk size must be positive, got {self.chunk_size}")
        if self.log_level not in ['debug', 'log', 'none']:
            raise ValueError(f"Invalid log level: {self.log_level}")

class InfoNCELoss(nn.Module):
    """InfoNCE loss function with improved type safety and error handling."""

    def __init__(
        self,
        config: InfoNCEConfig
    ) -> None:
        """Initialize InfoNCE Loss."""
        super().__init__()
        self.temperature = config.temperature
        self.reduction = config.reduction
        self.contrast_mode = config.contrast_mode
        self.chunk_size = config.chunk_size
        self.log_config = LogConfig(level=config.log_level)

        logger.info(
            f"InfoNCE Loss initialized with:\n"
            f"- Temperature: {self.temperature}\n"
            f"- Reduction: {self.reduction}\n"
            f"- Contrast mode: {self.contrast_mode}\n"
            f"- Chunk size: {self.chunk_size}\n"
            f"- Log level: {self.log_config.level}"
        )

    @log_function()
    def compute_similarity_chunk(
        self,
        features: Tensor,
        chunk_start: int,
        chunk_size: int
    ) -> Tensor:
        """Compute similarity between a chunk and all features."""
        chunk_end = min(chunk_start + chunk_size, features.size(0))
        chunk_features = features[chunk_start:chunk_end]
        return torch.matmul(chunk_features, features.T)

    @log_function()
    def forward(
        self,
        features: Tensor,
        labels: Optional[Tensor] = None,
        mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Compute InfoNCE loss."""
        device = features.device
        if labels is not None:
            labels = labels.to(device)
        if mask is not None:
            mask = mask.to(device)

        # Normalize features
        features = F.normalize(features, dim=1)
        batch_size = features.size(0)

        total_loss = torch.tensor(0.0, device=device)
        total_pairs = torch.tensor(0, device=device)

        for chunk_start in range(0, batch_size, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, batch_size)
            chunk_features = features[chunk_start:chunk_end]
            chunk_labels = labels[chunk_start:chunk_end] if labels is not None else None

            # Compute similarities
            chunk_features = torch.clamp(chunk_features, min=-1e3, max=1e3)
            features_clipped = torch.clamp(features, min=-1e3, max=1e3)
            sim_chunk = torch.matmul(chunk_features, features_clipped.T)

            # Apply temperature scaling
            temperature = max(self.temperature, 1e-4)  # Prevent division by zero
            sim_chunk = sim_chunk / temperature

            # Create masks
            chunk_mask_self = torch.ones_like(sim_chunk, dtype=torch.bool, device=device)
            chunk_mask_self[:, chunk_start:chunk_end].fill_diagonal_(False)

            if chunk_labels is not None:
                chunk_labels = chunk_labels.contiguous().view(-1, 1)
                chunk_mask_pos = chunk_labels == labels.view(1, -1)
                chunk_mask_pos = chunk_mask_pos & chunk_mask_self
                if not chunk_mask_pos.any():
                    chunk_mask_pos = chunk_mask_self
            else:
                chunk_mask_pos = chunk_mask_self

            # Compute log probabilities
            sim_max, _ = torch.max(sim_chunk, dim=1, keepdim=True)
            sim_chunk = sim_chunk - sim_max.detach()
            sim_chunk = torch.clamp(sim_chunk, min=-1e3, max=1e3)

            exp_sim = torch.exp(sim_chunk)
            exp_sim = torch.clamp(exp_sim, min=1e-8)
            exp_sim = exp_sim * chunk_mask_self
            log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

            log_prob = sim_chunk - log_sum_exp
            log_prob = torch.clamp(log_prob, min=-1e3)

            # Compute chunk loss
            pos_pairs = chunk_mask_pos.sum(1)
            chunk_loss = -(chunk_mask_pos * log_prob).sum(1)
            valid_pairs = pos_pairs > 0
            if valid_pairs.any():
                chunk_loss = chunk_loss[valid_pairs] / pos_pairs[valid_pairs]
            else:
                chunk_loss = torch.zeros(1, device=device)

            total_loss += chunk_loss.sum()
            total_pairs += (pos_pairs > 0).sum()

        # Compute final loss
        mean_loss = total_loss / (total_pairs + 1e-8)
        loss = mean_loss if self.reduction == 'mean' else mean_loss * total_pairs

        return {
            'loss': loss,
            'num_pairs': total_pairs,
            'mean_loss': mean_loss
        }

@log_function()
def info_nce_loss_factory(config: Dict[str, Any]) -> InfoNCELoss:
    """Factory function for creating an InfoNCELoss instance."""
    loss_config = InfoNCEConfig(
        temperature=config['training']['loss_temperature'],
        reduction='mean',  # Fixed as mean
        contrast_mode='all',  # Fixed as all
        chunk_size=256,  # Fixed size for memory efficiency
        log_level=config['training'].get('log_level', 'log')
    )
    
    return InfoNCELoss(config=loss_config)

__all__ = [
    'InfoNCELoss',
    'InfoNCEConfig',
    'info_nce_loss_factory',
]