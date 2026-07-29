"""Base Model adapter."""
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Base class for LLM adapters."""

    name: str = ""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response from prompt."""
        pass
