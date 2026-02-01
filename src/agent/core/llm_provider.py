import requests
import json
from typing import Optional
from src.agent.core.config import MigrationConfig

class LLMProvider:
    """
    Abstract base class for Large Language Model interaction.
    Facilitates structured code generation and transformation.
    """
    def is_available(self) -> bool:
        raise NotImplementedError
        
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        raise NotImplementedError

class OllamaProvider(LLMProvider):
    """
    Implementation for local Ollama server.
    Ensures data privacy by keeping inference local.
    """
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or MigrationConfig.LLM_BASE_URL
        self.model = model or MigrationConfig.LLM_MODEL

    def is_available(self) -> bool:
        """
        Check if the local Ollama service is reachable.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Request code generation from the local model.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": MigrationConfig.LLM_TEMPERATURE
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=MigrationConfig.LLM_TIMEOUT)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            return ""
        except Exception:
            return ""

class OpenAIProvider(LLMProvider):
    """
    Implementation for OpenAI API.
    Used for higher accuracy in complex logic transformations.
    """
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or MigrationConfig.OPENAI_API_KEY
        self.model = model or "gpt-4-turbo"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Securely interaction with OpenAI API.
        """
        if not self.is_available():
            return "Error: OpenAI API key missing."
            
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": MigrationConfig.LLM_TEMPERATURE
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=MigrationConfig.LLM_TIMEOUT)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            return ""
        except Exception:
            return ""

def get_llm_provider() -> LLMProvider:
    """
    Factory method to retrieve the configured LLM provider.
    """
    provider_type = MigrationConfig.LLM_PROVIDER
    if provider_type == "openai":
        return OpenAIProvider()
    # Default to Ollama for local privacy
    return OllamaProvider()
