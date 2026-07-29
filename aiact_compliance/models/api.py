"""API-based model adapters — Claude Sonnet (judge)."""
import os
from typing import Optional
from .base import BaseModel


class ClaudeSonnet(BaseModel):
    """Anthropic Claude Sonnet — used as judge for compliance scoring."""

    name = "claude-sonnet-5"

    def __init__(self, api_key: Optional[str] = None, model_id: str = "claude-sonnet-5"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. "
                "Set ANTHROPIC_API_KEY env var or pass api_key argument."
            )
        self.model_id = model_id
        self.client = None

    def _init_client(self):
        """Lazy-init Anthropic client."""
        if self.client is None:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate response from Claude."""
        self._init_client()

        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        # Extract text from response
        text_blocks = [
            block.text for block in response.content
            if hasattr(block, "text")
        ]
        return "\n".join(text_blocks).strip()
