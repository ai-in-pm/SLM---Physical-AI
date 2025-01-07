import torch
import torch.nn as nn

class WorldFoundationModel(nn.Module):
    """
    World Foundation Model that processes both image and text tokens to understand
    the world state and generate appropriate action tokens
    """
    def __init__(
        self,
        token_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        num_action_tokens: int = 256
    ):
        super().__init__()
        
        # Multi-layer transformer for token processing
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(token_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Action token generator
        self.action_generator = nn.Sequential(
            nn.Linear(token_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, num_action_tokens)
        )
        
        # Learnable action token embeddings
        self.action_token_embeddings = nn.Parameter(
            torch.randn(1, num_action_tokens, token_dim)
        )
        
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor = None
    ):
        """
        Process tokens through the World Foundation Model
        Args:
            image_tokens: Image tokens [B, num_image_tokens, token_dim]
            text_tokens: Text tokens [B, num_text_tokens, token_dim]
            attention_mask: Attention mask for text tokens [B, num_text_tokens]
        Returns:
            action_logits: Action token logits [B, num_action_tokens]
            fused_features: Fused token features [B, seq_len, token_dim]
        """
        # Concatenate image and text tokens
        fused_tokens = torch.cat([image_tokens, text_tokens], dim=1)
        
        # Create attention mask for both image and text tokens
        if attention_mask is not None:
            image_mask = torch.ones(
                image_tokens.size(0),
                image_tokens.size(1),
                device=image_tokens.device
            )
            full_mask = torch.cat([image_mask, attention_mask], dim=1)
        else:
            full_mask = None
        
        # Process through transformer layers
        for layer in self.transformer_layers:
            fused_tokens = layer(fused_tokens, full_mask)
        
        # Generate action tokens
        action_query = self.action_token_embeddings.expand(
            fused_tokens.size(0), -1, -1
        )
        
        # Cross-attend between action queries and fused tokens
        action_context = self.cross_attention(
            action_query,
            fused_tokens,
            fused_tokens,
            full_mask
        )
        
        # Generate action logits
        action_logits = self.action_generator(action_context.mean(dim=1))
        
        return action_logits, fused_tokens

class TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float
    ):
        super().__init__()
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feedforward network
        self.ff_network = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # Self-attention
        attended = self.self_attention(
            x, x, x,
            attn_mask=mask,
            need_weights=False
        )[0]
        x = self.norm1(x + self.dropout(attended))
        
        # Feedforward
        ff_output = self.ff_network(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class ActionTokenDecoder(nn.Module):
    def __init__(
        self,
        token_dim: int = 256,
        num_action_tokens: int = 256,
        num_heads: int = 4
    ):
        super().__init__()
        
        self.cross_attention = nn.MultiheadAttention(
            token_dim,
            num_heads,
            batch_first=True
        )
        
        self.action_embeddings = nn.Parameter(
            torch.randn(1, num_action_tokens, token_dim)
        )
        
        self.output_layer = nn.Linear(token_dim, 1)
        
    def forward(
        self,
        fused_features: torch.Tensor,
        attention_mask: torch.Tensor = None
    ):
        """
        Decode action tokens from fused features
        Args:
            fused_features: Fused token features [B, seq_len, token_dim]
            attention_mask: Attention mask [B, seq_len]
        Returns:
            action_probs: Action token probabilities [B, num_action_tokens]
        """
        batch_size = fused_features.size(0)
        
        # Expand action embeddings for batch
        action_queries = self.action_embeddings.expand(batch_size, -1, -1)
        
        # Cross-attend between action queries and fused features
        attended_features = self.cross_attention(
            action_queries,
            fused_features,
            fused_features,
            key_padding_mask=attention_mask,
            need_weights=False
        )[0]
        
        # Generate action scores
        action_scores = self.output_layer(attended_features).squeeze(-1)
        
        # Convert to probabilities
        action_probs = torch.softmax(action_scores, dim=-1)
        
        return action_probs
