"""Local model adapters — Mistral 7B, v1 QLoRA."""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from .base import BaseModel


class MistralLocal(BaseModel):
    """Mistral 7B Instruct v0.2 with 4-bit quantization."""

    name = "mistral-7b-instruct-v0.2"

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load model and tokenizer."""
        print(f"Loading {self.name}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        print(f"  Loaded.")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response using Mistral instruction format."""
        if self.model is None:
            self.load()

        # Mistral instruction format
        formatted = f"[INST] {prompt} [/INST]"
        inputs = self.tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Extract only the generated portion (after [/INST])
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response.split("[/INST]")[-1].strip()
        return response


class QLoRAv1(BaseModel):
    """v1 QLoRA fine-tuned Mistral 7B for Finnish-English."""

    name = "mistral-7b-fi-en-qlora-v1"

    def __init__(self, adapter_repo: str = "indrani191919/mistral-7b-fi-en-qlora-v1"):
        self.adapter_repo = adapter_repo
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load base Mistral + LoRA adapter."""
        print(f"Loading {self.name}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, self.adapter_repo)
        self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_repo)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        print(f"  Loaded.")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response using Mistral instruction format."""
        if self.model is None:
            self.load()

        formatted = f"[INST] {prompt} [/INST]"
        inputs = self.tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response.split("[/INST]")[-1].strip()
        return response
