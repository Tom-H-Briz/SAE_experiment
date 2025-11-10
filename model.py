import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings."""
    
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.num_patches = config.num_patches
        self.embed_dim = config.patch_embedding_dim
        
        # Linear projection from flattened patches to embedding dimension
        # Input: (batch, 1, 28, 28) -> patches: (batch, num_patches, patch_size*patch_size*1)
        patch_flattened_dim = config.num_channels * config.patch_size * config.patch_size
        self.projection = nn.Linear(patch_flattened_dim, self.embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            patches: (batch, num_patches, embed_dim)
        """
        batch_size, channels, height, width = x.shape
        
        # Reshape into patches: (batch, channels, num_patches_h, patch_size, num_patches_w, patch_size)
        patches = x.reshape(
            batch_size,
            channels,
            height // self.patch_size,
            self.patch_size,
            width // self.patch_size,
            self.patch_size
        )
        
        # Rearrange to (batch, num_patches, channels, patch_size, patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.reshape(batch_size, self.num_patches, -1)
        
        # Project to embedding dimension
        patches = self.projection(patches)
        
        return patches


class Attention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(config.hidden_dim, config.hidden_dim * 3, bias=config.qkv_bias)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.proj_dropout = nn.Dropout(config.proj_dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            out: (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
        out = self.proj(out)
        out = self.proj_dropout(out)
        
        return out


class MLP(nn.Module):
    """Feed-forward MLP."""
    
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_dim, config.mlp_hidden_dim)
        self.fc2 = nn.Linear(config.mlp_hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # GELU activation
        self.activation = nn.GELU()
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            out: (batch, seq_len, hidden_dim)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""
    
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attention = Attention(config)
        
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = MLP(config)
    
    def forward(self, x, return_mlp_out=False):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            return_mlp_out: If True, also return post-MLP output (before residual)
        Returns:
            out: (batch, seq_len, hidden_dim)
            mlp_out: (batch, seq_len, hidden_dim) if return_mlp_out=True
        """
        # Attention with residual (pre-norm)
        x_normed = self.norm1(x)
        attn_out = self.attention(x_normed)
        x = x + attn_out
        
        # MLP with residual (pre-norm)
        x_normed = self.norm2(x)
        mlp_out = self.mlp(x_normed)
        x = x + mlp_out
        
        if return_mlp_out:
            return x, mlp_out
        return x


class ViT(nn.Module):
    """Vision Transformer for MNIST classification with activation capture."""
    
    def __init__(self, config, capture_layer=None):
        """
        Args:
            config: ModelConfig instance
            capture_layer: If int, return activations from this layer (0-indexed).
                          Layer 3 post-MLP is typically the target for SAE training.
        """
        super().__init__()
        self.config = config
        self.capture_layer = capture_layer
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(config)
        
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.patch_embedding_dim))
        
        # Positional embeddings (learnable 1D)
        # seq_len = num_patches + 1 (for CLS token)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.patch_embedding_dim)
        )
        self.pos_dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.classification_head = nn.Linear(config.hidden_dim, config.num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize learnable parameters."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # Initialize linear layers with trunc_normal
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x):
        """
        Args:
            x: (batch, 1, 28, 28)
        Returns:
            logits: (batch, num_classes)
            captured_activations: (batch, num_patches+1, hidden_dim) if capture_layer set, else None
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(x)  # (batch, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches+1, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embedding
        x = self.pos_dropout(x)
        
        # Transformer blocks
        captured_activations = None
        for i, block in enumerate(self.transformer_blocks):
            if i == self.capture_layer:
                # Capture post-MLP activations for this layer
                x, mlp_out = block(x, return_mlp_out=True)
                captured_activations = mlp_out
            else:
                x = block(x)
        
        # Classification head on CLS token
        x = self.norm(x)
        cls_output = x[:, 0]  # (batch, hidden_dim)
        logits = self.classification_head(cls_output)  # (batch, num_classes)
        
        return logits, captured_activations
    
    def count_parameters(self):
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
