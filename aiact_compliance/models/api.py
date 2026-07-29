"""API-based model adapters (DeepSeek V3)."""
from .base import BaseModel


class DeepSeekV3(BaseModel):
    name = "deepseek-v3"

    def __init__(self, api_key):
        self.api_key = api_key

    def generate(self, prompt, max_tokens=512):
        raise NotImplementedError
