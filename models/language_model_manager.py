import os
from typing import Dict, Optional, List, Union
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
from anthropic import Anthropic
from google.cloud import aiplatform
import logging

class LanguageModelManager:
    """
    Manages multiple language models for the Physical AI system,
    supporting both local and API-based models
    """
    def __init__(self, model_name: str = "gpt2", use_api: bool = False):
        load_dotenv()  # Load environment variables from .env file
        
        self.model_name = model_name
        self.use_api = use_api
        self.model = None
        self.tokenizer = None
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # API clients
        self.api_clients = {}
        
        if use_api:
            self._initialize_api_clients()
        else:
            self._initialize_local_model()
    
    def _initialize_api_clients(self):
        """Initialize API clients for various language models"""
        # OpenAI (GPT-4)
        if os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.api_clients["gpt4"] = openai.Client()
        
        # Anthropic (Claude)
        if os.getenv("ANTHROPIC_API_KEY"):
            self.api_clients["claude"] = Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        
        # PaLM 2
        if os.getenv("PALM_2_API_KEY"):
            aiplatform.init(project=os.getenv("PALM_2_API_KEY"))
            self.api_clients["palm2"] = aiplatform
        
        # Add other API clients as needed
        self._load_additional_apis()
    
    def _load_additional_apis(self):
        """Load additional API clients based on environment variables"""
        api_mapping = {
            "GROK_1_API_KEY": "grok1",
            "MISTRAL_7B_API_KEY": "mistral7b",
            "FALCON_180B_API_KEY": "falcon180b",
            "STABLE_LM_2_API_KEY": "stablelm2",
            "GEMINI_1_5_API_KEY": "gemini15",
            "LLAMA_3_1_API_KEY": "llama31",
            "MIXTRAL_8X22B_API_KEY": "mixtral8x22b",
            "INFLECTION_2_5_API_KEY": "inflection25",
            "JAMBA_API_KEY": "jamba",
            "COMMAND_R_API_KEY": "commandr",
            "GEMMA_API_KEY": "gemma",
            "PHI_3_API_KEY": "phi3",
            "XGEN_7B_API_KEY": "xgen7b",
            "DBRX_API_KEY": "dbrx",
            "PYTHIA_API_KEY": "pythia",
            "SORA_API_KEY": "sora",
            "ALPACA_7B_API_KEY": "alpaca7b",
            "NEMOTRON_4_API_KEY": "nemotron4"
        }
        
        for env_key, model_key in api_mapping.items():
            api_key = os.getenv(env_key)
            if api_key:
                self.api_clients[model_key] = {
                    "api_key": api_key,
                    "initialized": False  # Will be initialized on first use
                }
    
    def _initialize_local_model(self):
        """Initialize local transformer model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                
        except Exception as e:
            self.logger.error(f"Error loading local model: {str(e)}")
            raise
    
    async def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        model_preference: Optional[str] = None
    ) -> str:
        """
        Generate text using either local model or API
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            model_preference: Preferred model to use (if using APIs)
        Returns:
            Generated text
        """
        if self.use_api:
            return await self._generate_text_api(prompt, max_length, model_preference)
        else:
            return self._generate_text_local(prompt, max_length)
    
    async def _generate_text_api(
        self,
        prompt: str,
        max_length: int,
        model_preference: Optional[str]
    ) -> str:
        """Generate text using API-based models"""
        if model_preference and model_preference in self.api_clients:
            client = self.api_clients[model_preference]
        else:
            # Default to GPT-4 if available
            client = self.api_clients.get("gpt4")
            if not client:
                # Fallback to first available client
                client = next(iter(self.api_clients.values()))
        
        try:
            if model_preference == "gpt4":
                response = await client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_length
                )
                return response.choices[0].message.content
                
            elif model_preference == "claude":
                response = await client.messages.create(
                    model="claude-3",
                    max_tokens=max_length,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
            # Add handling for other API-based models
            
        except Exception as e:
            self.logger.error(f"API generation error: {str(e)}")
            raise
    
    def _generate_text_local(self, prompt: str, max_length: int) -> str:
        """Generate text using local transformer model"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            self.logger.error(f"Local generation error: {str(e)}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        models = []
        if not self.use_api and self.model is not None:
            models.append(self.model_name)
        models.extend(list(self.api_clients.keys()))
        return models
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a specific model"""
        if model_name in self.api_clients:
            return {
                "type": "api",
                "name": model_name,
                "initialized": self.api_clients[model_name].get("initialized", True)
            }
        elif model_name == self.model_name and not self.use_api:
            return {
                "type": "local",
                "name": model_name,
                "initialized": self.model is not None
            }
        return None
