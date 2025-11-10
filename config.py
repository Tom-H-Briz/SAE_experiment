from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ModelConfig:
    """Vision Transformer architecture specifications."""
    
    # Image and patch settings
    image_size: int = 28  # MNIST native resolution
    num_channels: int = 1  # Grayscale
    patch_size: int = 7
    num_patches: int = 16  # 4x4 grid (28/7 = 4 per dimension)
    patch_embedding_dim: int = 192
    
    # Transformer architecture
    num_layers: int = 4
    hidden_dim: int = 192
    num_heads: int = 3  # 192/3 = 64 dims per head
    mlp_expansion_ratio: int = 4  # MLP hidden = 192 * 4 = 768
    mlp_hidden_dim: int = 768
    activation: str = "gelu"
    dropout: float = 0.1
    
    # Pre-norm architecture (LayerNorm before attention/MLP)
    use_pre_norm: bool = True
    
    # Attention and MLP details
    qkv_bias: bool = True
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    
    # Classification head
    num_classes: int = 10  # MNIST
    
    def validate(self):
        """Sanity checks on config."""
        assert self.image_size == 28, "Image size must be 28 (MNIST)"
        assert self.num_channels == 1, "Must be grayscale"
        assert self.patch_size == 7, "Patch size must be 7"
        assert (self.image_size // self.patch_size) ** 2 == self.num_patches, \
            f"Patch grid mismatch: {self.image_size}/{self.patch_size} should give {self.num_patches} patches"
        assert self.hidden_dim % self.num_heads == 0, \
            f"Hidden dim {self.hidden_dim} must be divisible by num_heads {self.num_heads}"
        assert self.mlp_hidden_dim == self.hidden_dim * self.mlp_expansion_ratio, \
            "MLP hidden dim must equal hidden_dim * expansion_ratio"


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""
    
    # Data loading
    batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True
    
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    optimizer: str = "adam"  # Only option for now
    max_norm_grad_clip: float = 1.0
    
    # Learning rate schedule
    num_epochs: int = 30
    warmup_epochs: int = 2  # ~5% of 30 epochs
    lr_schedule: str = "cosine"  # cosine annealing
    
    # Mixed precision
    use_mixed_precision: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = False  # Save all checkpoints for analysis
    keep_last_n: int = 3  # Keep last 3 checkpoints + best
    
    # Logging and validation
    log_frequency: int = 100  # Log every N batches
    validate_frequency: int = 1  # Validate every N epochs
    early_stopping_patience: Optional[int] = None  # None = no early stopping
    
    # Data normalization (MNIST standard)
    norm_mean: float = 0.5
    norm_std: float = 0.5
    
    # Reproducibility
    seed: int = 42
    
    # Device
    device: str = "cuda"  # Will check availability at runtime


@dataclass
class Config:
    """Master config combining model and training settings."""
    model: ModelConfig = None
    training: TrainingConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        
        self.model.validate()
    
    def to_dict(self):
        """Convert entire config to nested dict for saving."""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training)
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Reconstruct config from dict."""
        model_cfg = ModelConfig(**config_dict['model'])
        training_cfg = TrainingConfig(**config_dict['training'])
        return cls(model=model_cfg, training=training_cfg)


# Default config instance
DEFAULT_CONFIG = Config()
