"""Model management for the inference CLI."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer
)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Optional
import warnings

from config import AVAILABLE_MODELS

warnings.filterwarnings("ignore")

console = Console()


class ModelManager:
    """Manages loading and inference for language models."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        self.device = self._detect_device()
        console.print(f"[dim]Using device: {self.device}[/dim]")

    def _detect_device(self) -> str:
        """Detect the best available device."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def load_model(self, model_key: str, force_reload: bool = False):
        """Load a model by key."""
        if self.current_model_name == model_key and not force_reload:
            return

        model_config = AVAILABLE_MODELS.get(model_key)
        if not model_config:
            raise ValueError(f"Model {model_key} not found")

        model_name = model_config["name"]
        use_float16 = model_config.get("use_float16", False)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Loading {model_key}...", total=None)

            try:
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    padding_side="left"
                )

                if not self.tokenizer.pad_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Determine dtype
                dtype = self._get_dtype(use_float16)

                # Load model
                self.model = self._load_model_with_config(model_name, dtype)

                self.current_model_name = model_key
                progress.update(task, description=f"Loaded {model_key}")

            except Exception as e:
                console.print(f"[red]Error loading model: {e}[/red]")
                raise

    def _get_dtype(self, use_float16: bool) -> torch.dtype:
        """Determine the appropriate dtype for the device."""
        if self.device in ["mps", "cuda"] and use_float16:
            return torch.float16
        return torch.float32

    def _load_model_with_config(self, model_name: str, dtype: torch.dtype):
        """Load model with appropriate configuration."""
        if self.device in ["cuda", "mps"]:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            model = model.to(self.device)

        return model

    def generate(self,
                prompt: str,
                max_tokens: int = 512,
                temperature: float = 0.7,
                top_p: float = 0.95,
                stream: bool = False) -> str:
        """Generate text from a prompt."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded")

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        if self.device in ["cuda", "mps"]:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if stream:
            return self._generate_streaming(inputs, max_tokens, temperature, top_p)
        else:
            return self._generate_batch(inputs, max_tokens, temperature, top_p)

    def _generate_streaming(self, inputs, max_tokens, temperature, top_p) -> str:
        """Generate text with streaming output."""
        streamer = TextStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)
        with torch.no_grad():
            self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                streamer=streamer,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return ""

    def _generate_batch(self, inputs, max_tokens, temperature, top_p) -> str:
        """Generate text without streaming."""
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return response

    def get_memory_usage(self) -> str:
        """Get current memory usage information."""
        if self.device == "cuda":
            return f"Memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        elif self.device == "mps":
            return "Using MPS (Metal Performance Shaders) on Apple Silicon"
        else:
            return "CPU mode"