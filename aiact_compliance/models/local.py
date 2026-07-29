"""Local model adapters (Mistral 7B, v1 QLoRA)."""
from .base import BaseModel


class MistralLocal(BaseModel):
    name = "mistral-7b-instruct"

    def __init__(self):
        pass

    def generate(self, prompt, max_tokens=512):
        raise NotImplementedError
