import torch
import torch.nn as nn
from transformers import AutoTokenizer
from .language_model_manager import LanguageModelManager

class LightweightTextProcessor(nn.Module):
    def __init__(
        self,
        model_name="gpt2",
        feature_dim=768,
        max_length=128,
        use_api=False,
        model_preference=None
    ):
        super().__init__()
        
        # Initialize language model manager
        self.lm_manager = LanguageModelManager(
            model_name=model_name,
            use_api=use_api
        )
        
        self.feature_dim = feature_dim
        self.max_length = max_length
        self.model_preference = model_preference
        
        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(
                self.lm_manager.model.config.hidden_size if not use_api else 1024,
                feature_dim
            ),
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, input_ids, attention_mask=None):
        """
        Process text through the language model and project to feature space
        Args:
            input_ids (torch.Tensor): Tokenized input text [B, L]
            attention_mask (torch.Tensor): Attention mask for padding [B, L]
        Returns:
            torch.Tensor: Text features [B, feature_dim]
        """
        if self.lm_manager.use_api:
            # Use API-based model
            text_features = self._process_with_api(input_ids)
        else:
            # Use local model
            outputs = self.lm_manager.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            text_features = outputs.last_hidden_state
        
        # Project to common feature space
        return self.feature_projection(text_features)
    
    async def _process_with_api(self, input_ids):
        """Process text using API-based model"""
        # Convert input_ids back to text
        text_batch = self.lm_manager.tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=True
        )
        
        # Process each text in batch
        features = []
        for text in text_batch:
            response = await self.lm_manager.generate_text(
                text,
                max_length=self.max_length,
                model_preference=self.model_preference
            )
            # Convert response to feature vector (implementation depends on API response format)
            feature = self._convert_api_response_to_features(response)
            features.append(feature)
        
        return torch.stack(features)
    
    def _convert_api_response_to_features(self, response):
        """Convert API response to feature vector"""
        # This implementation will depend on the specific API response format
        # For now, we'll use a simple embedding approach
        tokens = self.lm_manager.tokenizer(
            response,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            features = self.lm_manager.model(
                **tokens,
                output_hidden_states=True
            ).last_hidden_state.mean(dim=1)
        
        return features
    
    def encode_text(self, text):
        """
        Tokenize and encode text input
        Args:
            text (str or List[str]): Input text or batch of texts
        Returns:
            dict: Tokenized inputs with input_ids and attention_mask
        """
        return self.lm_manager.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def get_available_models(self):
        """Get list of available language models"""
        return self.lm_manager.get_available_models()
    
    def switch_model(self, model_name: str, use_api: bool = None):
        """Switch to a different language model"""
        if use_api is not None:
            self.lm_manager.use_api = use_api
        self.lm_manager = LanguageModelManager(
            model_name=model_name,
            use_api=self.lm_manager.use_api
        )
        # Update feature projection if necessary
        if self.lm_manager.use_api:
            input_dim = 1024  # Default for API-based models
        else:
            input_dim = self.lm_manager.model.config.hidden_size
        
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.1)
        )
