import torch
import torch.nn as nn
from typing import List, Tuple, Dict

class TokenProcessor(nn.Module):
    """
    Processes both image and text into tokens that can be consumed by the World Foundation Model
    """
    def __init__(
        self,
        image_token_dim: int = 256,
        text_token_dim: int = 256,
        num_image_tokens: int = 8,
        max_text_tokens: int = 128
    ):
        super().__init__()
        self.image_token_dim = image_token_dim
        self.text_token_dim = text_token_dim
        self.num_image_tokens = num_image_tokens
        self.max_text_tokens = max_text_tokens

        # Image to token conversion layers
        self.image_tokenizer = nn.Sequential(
            nn.Conv2d(2048, 512, 1),  # Reduce channel dimension
            nn.ReLU(),
            nn.Conv2d(512, num_image_tokens, 1),  # Generate token weights
            nn.Softmax(dim=2)  # Spatial attention weights
        )

        # Text to token conversion layers
        self.text_token_embedder = nn.Linear(768, text_token_dim)  # Assuming BERT-like embeddings
        
        # Token type embeddings
        self.token_type_embeddings = nn.Embedding(2, text_token_dim)  # 0 for image, 1 for text

    def extract_image_tokens(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert image features to a sequence of tokens
        Args:
            features: Image features [B, C, H, W]
        Returns:
            tokens: Image tokens [B, num_tokens, token_dim]
            attention_weights: Token attention weights [B, num_tokens, H*W]
        """
        B, C, H, W = features.shape
        
        # Generate attention weights for tokens
        attention = self.image_tokenizer(features)  # [B, num_tokens, H, W]
        attention_weights = attention.view(B, self.num_image_tokens, -1)  # [B, num_tokens, H*W]
        
        # Extract tokens through weighted averaging
        features_flat = features.view(B, C, -1)  # [B, C, H*W]
        tokens = torch.bmm(attention_weights, features_flat.transpose(1, 2))  # [B, num_tokens, C]
        
        return tokens, attention_weights

    def process_text_tokens(
        self,
        text_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Process text embeddings into tokens
        Args:
            text_embeddings: Text embeddings from language model [B, L, D]
            attention_mask: Attention mask [B, L]
        Returns:
            tokens: Text tokens [B, L, token_dim]
        """
        # Project to token dimension
        tokens = self.text_token_embedder(text_embeddings)
        
        # Add token type embeddings
        token_type_ids = torch.ones_like(attention_mask)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        tokens = tokens + token_type_embeddings
        return tokens

    def forward(
        self,
        image_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Process both image and text into tokens
        Args:
            image_features: Image features [B, C, H, W]
            text_embeddings: Text embeddings [B, L, D]
            text_attention_mask: Text attention mask [B, L]
        Returns:
            Dictionary containing:
                - image_tokens: Image tokens [B, num_image_tokens, token_dim]
                - text_tokens: Text tokens [B, L, token_dim]
                - attention_weights: Image token attention weights [B, num_tokens, H*W]
        """
        # Process image tokens
        image_tokens, attention_weights = self.extract_image_tokens(image_features)
        
        # Process text tokens
        text_tokens = self.process_text_tokens(text_embeddings, text_attention_mask)
        
        return {
            'image_tokens': image_tokens,
            'text_tokens': text_tokens,
            'attention_weights': attention_weights
        }
