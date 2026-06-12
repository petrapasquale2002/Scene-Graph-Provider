import os
from typing import List, Dict, Union, Optional, Any
from abc import ABC, abstractmethod

import pandas as pd
from dotenv import load_dotenv

# Import provider SDKs
try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

vlm_node_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nebius_env = os.path.join(vlm_node_dir, "NebiusAPI.env")
groq_env = os.path.join(vlm_node_dir, "GroqAPI.env")

if os.path.exists(nebius_env):
    load_dotenv(dotenv_path=nebius_env, override=True)
if os.path.exists(groq_env):
    load_dotenv(dotenv_path=groq_env, override=True)

class ModelRegistry:
    """
    Centralized registry for model names across providers.
    Maps simplified/internal names to official API model IDs.
    """
    
    # Groq Models
    GROQ_MODELS = {
        "llama3.1-8b": "llama-3.1-8b-instant",
        "llama3.3-70b": "llama-3.3-70b-versatile",
        "llama4-maverick-17b": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "moonshotai-kimik2-32b": "moonshotai/kimi-k2-instruct-0905",
        "qwen3-32b": "qwen/qwen3-32b",
        "openai-oss-20b": "openai/gpt-oss-20b",
        "openai-oss-120b": "openai/gpt-oss-120b",
        # vision enabled models
        "llama4-scout-17b": "meta-llama/llama-4-scout-17b-16e-instruct",
    }

    # Nebius Models
    NEBIUS_MODELS = {
        'nvidia-nemotron-30b': "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        'nvidia-nemotron-120b': "nvidia/nemotron-3-super-120b-a12b",
        'kimi-k2.6': "moonshotai/Kimi-K2.6",
        'qwen3-embedding-8b': 'Qwen/Qwen3-Embedding-8B',
        'qwen3-2.5-70b': 'Qwen/Qwen2.5-VL-72B-Instruct',
    }

    # OpenAI Models
    OPENAI_MODELS = {
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4-turbo": "gpt-4-turbo",
        "gpt-4": "gpt-4",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        # Vision is implicitly supported in 4o and 4-turbo
    }

    # Anthropic Models
    ANTHROPIC_MODELS = {
        "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307",
    }

    # Gemini Models
    GEMINI_MODELS = {
        "gemini-1.5-pro": "gemini-1.5-pro",
        "gemini-1.5-flash": "gemini-1.5-flash",
        "gemini-1.0-pro": "gemini-1.0-pro",
    }

    @classmethod
    def get_model_id(cls, provider: str, model_name: str) -> str:
        """Resolves the internal model name to the provider's API model ID."""
        if provider == "groq":
            return cls.GROQ_MODELS.get(model_name, model_name)
        elif provider == "openai":
            return cls.OPENAI_MODELS.get(model_name, model_name)
        elif provider == "nebius":
            return cls.NEBIUS_MODELS.get(model_name, model_name)
        elif provider == "anthropic":
            return cls.ANTHROPIC_MODELS.get(model_name, model_name)
        elif provider == "gemini":
            return cls.GEMINI_MODELS.get(model_name, model_name)
        else:
            return model_name


class BaseFoundationClient(ABC):
    """Base class for all foundation model clients."""
    
    def __init__(self, **model_parameters):
        
        self.model_parameters = model_parameters
        full_model_name = model_parameters.get("model_name", "")
        if "/" in full_model_name:
            self.provider, self.raw_model_name = full_model_name.split("/", 1)
        else:
            raise ValueError(f"Model name '{full_model_name}' must be in format 'provider/model_name'")

        self.model_name = ModelRegistry.get_model_id(self.provider, self.raw_model_name)
        
        self.temperature = model_parameters.get("temperature", 0.7)
        self.max_tokens = model_parameters.get("max_tokens", 1024)
        self.top_p = model_parameters.get("top_p", 1.0)
        self.stream = model_parameters.get("stream", False)
        
        self.api_key = model_parameters.get("api_key", os.getenv(f"{self.provider.upper()}_API_KEY"))
        self.base_url = model_parameters.get("base_url", os.getenv(f"{self.provider.upper()}_BASE_URL"))
        
        self.client = self._initialize_client()
        self.usage_metrics = None

    def _initialize_client(self):
        if self.provider == "groq":
            if not Groq: raise ImportError("Groq SDK not installed.")
            return Groq(api_key=self.api_key)
        elif self.provider == "openai":
            if not OpenAI: raise ImportError("OpenAI SDK not installed.")
            return OpenAI(api_key=self.api_key)
        elif self.provider == "nebius":
            if not OpenAI: raise ImportError("OpenAI SDK not installed.")
            base_url = self.base_url or "https://api.tokenfactory.nebius.com/v1/"
            return OpenAI(api_key=self.api_key, base_url=base_url)
        elif self.provider == "anthropic":
            if not Anthropic: raise ImportError("Anthropic SDK not installed.")
            return Anthropic(api_key=self.api_key)
        elif self.provider == "gemini":
            if not genai: raise ImportError("Google GenAI SDK not installed.")
            # Initialize Client directly
            return genai.Client(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _update_metrics(self, input_tokens: int, output_tokens: int, search_provider: str = None):
        """Standardized metric collection."""
        new_metric = {
            "timestamp": pd.Timestamp.now(),
            "provider": self.provider,
            "model": self.model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        if search_provider:
             new_metric["search_provider"] = search_provider
             
        if self.usage_metrics is None:
            self.usage_metrics = pd.DataFrame([new_metric])
        else:
            self.usage_metrics = pd.concat([self.usage_metrics, pd.DataFrame([new_metric])], ignore_index=True)

    def get_total_usage(self) -> Dict[str, int]:
        if self.usage_metrics is None:
             return {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
        return {
            "total_tokens": int(self.usage_metrics["total_tokens"].sum()),
            "input_tokens": int(self.usage_metrics["input_tokens"].sum()),
            "output_tokens": int(self.usage_metrics["output_tokens"].sum()),
        }

    def log_metrics(self):
        if self.usage_metrics is not None:
             print(f"\n[{self.__class__.__name__}] Usage Metrics:")
             print(self.usage_metrics)