import logging
from typing import Protocol

from .llm_client import OllamaClient
from .hf_llm import HFChatLLM

logger = logging.getLogger(__name__)


class LLMBackend(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...
    def stream_generate(self, system_prompt: str, user_prompt: str): ...
    def is_available(self) -> bool: ...


def get_llm_backend(config: dict) -> LLMBackend:
    backend = config.get("llm", {}).get("backend", "ollama")
    if backend == "hf":
        client = HFChatLLM(config)
        if client.is_available():
            return client
        logger.warning("HF backend failed; falling back to Ollama")
    return OllamaClient(config)
