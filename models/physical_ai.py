import torch
import torch.nn as nn
from .vision_processor import LightweightVisionProcessor
from .text_processor import LightweightTextProcessor
from .token_processor import TokenProcessor
from .fusion import WorldFoundationModel, ActionTokenDecoder

class PhysicalAI(nn.Module):
    def __init__(
        self,
        vision_model="efficientnet-b0",
        language_model="gpt2",
        token_dim=256,
        num_image_tokens=8,
        num_action_tokens=256
    ):
        super().__init__()
        
        # Vision processing
        self.vision_processor = LightweightVisionProcessor(
            feature_dim=2048,  # Match token processor input
            efficient_net_version=vision_model.split('-')[1]
        )
        
        # Text processing
        self.text_processor = LightweightTextProcessor(
            model_name=language_model,
            feature_dim=768  # Match token processor input
        )
        
        # Token processing
        self.token_processor = TokenProcessor(
            image_token_dim=token_dim,
            text_token_dim=token_dim,
            num_image_tokens=num_image_tokens,
            max_text_tokens=128
        )
        
        # World Foundation Model
        self.world_model = WorldFoundationModel(
            token_dim=token_dim,
            num_action_tokens=num_action_tokens
        )
        
        # Action token decoder
        self.action_decoder = ActionTokenDecoder(
            token_dim=token_dim,
            num_action_tokens=num_action_tokens
        )
        
    def forward(self, images, text):
        """
        Process multimodal input and generate action tokens
        Args:
            images: Batch of images [B, C, H, W]
            text: List of text instructions
        Returns:
            action_probs: Action token probabilities [B, num_action_tokens]
            fused_features: Fused token features [B, seq_len, token_dim]
        """
        # Extract visual features
        visual_features = self.vision_processor(images)
        
        # Process text
        text_encoded = self.text_processor.encode_text(text)
        text_features = self.text_processor(
            text_encoded['input_ids'],
            text_encoded['attention_mask']
        )
        
        # Convert to tokens
        token_outputs = self.token_processor(
            visual_features,
            text_features,
            text_encoded['attention_mask']
        )
        
        # Process through World Foundation Model
        action_logits, fused_features = self.world_model(
            token_outputs['image_tokens'],
            token_outputs['text_tokens'],
            text_encoded['attention_mask']
        )
        
        # Decode action tokens
        action_probs = self.action_decoder(
            fused_features,
            text_encoded['attention_mask']
        )
        
        return action_probs, fused_features
    
    def configure_optimizers(self, lr=1e-4):
        """
        Configure optimizers for training
        Args:
            lr (float): Learning rate
        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        # Group parameters by component
        vision_params = list(self.vision_processor.parameters())
        text_params = list(self.text_processor.parameters())
        token_params = list(self.token_processor.parameters())
        world_model_params = list(self.world_model.parameters()) + \
                           list(self.action_decoder.parameters())
        
        # Use different learning rates for different components
        param_groups = [
            {'params': vision_params, 'lr': lr * 0.1},  # Lower LR for pretrained vision
            {'params': text_params, 'lr': lr * 0.1},    # Lower LR for pretrained LM
            {'params': token_params, 'lr': lr},         # Full LR for token processor
            {'params': world_model_params, 'lr': lr}    # Full LR for world model
        ]
        
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=0.01)
