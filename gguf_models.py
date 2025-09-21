"""GGUF model support for quantized models with working progress tracking."""

import os
import sys
from typing import Optional
from pathlib import Path
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
    TaskProgressColumn
)
import logging

console = Console()

# Configure logging to capture HF progress
logging.basicConfig(level=logging.INFO)


class GGUFModelManager:
    """Manages GGUF quantized models using llama-cpp-python."""

    def __init__(self, cache_dir: str = "~/.cache/ubercli/models"):
        self.model = None
        self.current_model_name = None
        self.cache_dir = os.path.expanduser(cache_dir)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # Detect if Metal is available for GPU acceleration
        self.n_gpu_layers = -1 if self._has_metal() else 0

    def _has_metal(self) -> bool:
        """Check if Metal GPU acceleration is available."""
        try:
            import platform
            return platform.system() == "Darwin" and platform.machine() == "arm64"
        except:
            return False

    def download_model(self, repo_id: str, filename: str = None) -> str:
        """Download a GGUF model from HuggingFace Hub."""

        # First, determine which file to download
        if not filename:
            from huggingface_hub import list_repo_files

            console.print(f"[dim]Checking available files for {repo_id}...[/dim]")

            try:
                files = list_repo_files(repo_id)
                gguf_files = [f for f in files if f.endswith(".gguf")]

                if not gguf_files:
                    raise ValueError(f"No GGUF files found in {repo_id}")

                # Sort files by quantization preference
                quant_priority = {
                    "Q4_K_M": 1,
                    "Q5_K_M": 2,
                    "Q4_K_S": 3,
                    "Q5_K_S": 4,
                    "Q4_0": 5,
                    "Q5_0": 6,
                    "Q3_K_M": 7,
                    "Q6_K": 8,
                    "Q8_0": 9,
                    "Q2_K": 10  # Smallest but lowest quality
                }

                def get_priority(filename):
                    for quant, priority in quant_priority.items():
                        if quant.lower() in filename.lower():
                            return priority
                    return 999

                gguf_files.sort(key=get_priority)
                selected_file = gguf_files[0]
                filename = selected_file

                # Show available quantizations
                console.print("\n[cyan]Available quantizations:[/cyan]")
                for i, file in enumerate(gguf_files[:5]):  # Show top 5
                    size_indicator = ""
                    if "Q2" in file:
                        size_indicator = " [dim](smallest, lowest quality)[/dim]"
                    elif "Q3" in file:
                        size_indicator = " [dim](small, lower quality)[/dim]"
                    elif "Q4" in file:
                        size_indicator = " [dim](balanced)[/dim]"
                    elif "Q5" in file or "Q6" in file:
                        size_indicator = " [dim](higher quality)[/dim]"
                    elif "Q8" in file:
                        size_indicator = " [dim](best quality, largest)[/dim]"

                    marker = ">" if i == 0 else " "
                    console.print(f"  {marker} {file}{size_indicator}")

                console.print(f"\n[green]Selected: {filename}[/green]")

            except Exception as e:
                console.print(f"[red]Error listing files: {e}[/red]")
                raise

        # Check if file already exists in cache
        from huggingface_hub import try_to_load_from_cache

        cached_file = try_to_load_from_cache(
            cache_dir=self.cache_dir,
            repo_id=repo_id,
            filename=filename
        )

        if cached_file and os.path.exists(cached_file):
            file_size_mb = os.path.getsize(cached_file) / (1024 * 1024)
            console.print(f"[green]✓ Model already cached: {filename} ({file_size_mb:.1f} MB)[/green]")
            return cached_file

        # Get file size for display
        file_size = None
        try:
            from huggingface_hub import get_hf_file_metadata, hf_hub_url
            url = hf_hub_url(repo_id=repo_id, filename=filename)
            metadata = get_hf_file_metadata(url)
            if metadata and metadata.size:
                file_size = metadata.size
                file_size_gb = file_size / (1024 ** 3)
                console.print(f"[dim]File size: {file_size_gb:.2f} GB[/dim]")
        except:
            pass

        console.print(f"\n[cyan]Downloading {filename}...[/cyan]")
        console.print("[dim]Note: HuggingFace Hub download progress will appear below[/dim]\n")

        # For now, use the standard HF download with its own progress
        # The tqdm progress bar from HF will show
        try:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=self.cache_dir,
                resume_download=True,
                force_download=False
            )

            console.print(f"\n[green]✓ Download complete![/green]")
            return model_path

        except Exception as e:
            console.print(f"\n[red]Error downloading model: {e}[/red]")
            raise

    def load_model(self,
                  model_path: str = None,
                  repo_id: str = None,
                  filename: str = None,
                  n_ctx: int = 2048,
                  n_threads: int = None,
                  force_reload: bool = False):
        """Load a GGUF model."""

        if repo_id and (self.current_model_name != repo_id or force_reload):
            # Download from HuggingFace
            model_path = self.download_model(repo_id, filename)
            self.current_model_name = repo_id
        elif model_path and (self.current_model_name != model_path or force_reload):
            # Use local path
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.current_model_name = model_path
        elif not force_reload and self.model:
            # Model already loaded
            return

        if not model_path:
            raise ValueError("Either model_path or repo_id must be provided")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading model into memory...", total=None)

            try:
                self.model = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False
                )

                progress.update(task, description="✓ Model loaded")

                if self.n_gpu_layers > 0:
                    console.print("[green]✓ Using Metal GPU acceleration[/green]")
                else:
                    console.print("[dim]Running on CPU (Metal not available)[/dim]")

            except Exception as e:
                console.print(f"[red]Error loading model: {e}[/red]")
                raise

    def generate(self,
                prompt: str,
                max_tokens: int = 512,
                temperature: float = 0.7,
                top_p: float = 0.95,
                stream: bool = False) -> str:
        """Generate text from a prompt."""
        if not self.model:
            raise ValueError("Model not loaded")

        if stream:
            return self._generate_streaming(prompt, max_tokens, temperature, top_p)
        else:
            return self._generate_batch(prompt, max_tokens, temperature, top_p)

    def _generate_streaming(self, prompt, max_tokens, temperature, top_p) -> str:
        """Generate text with streaming output."""
        response = ""
        for chunk in self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True
        ):
            text = chunk['choices'][0]['text']
            console.print(text, end="")
            response += text
        return response

    def _generate_batch(self, prompt, max_tokens, temperature, top_p) -> str:
        """Generate text without streaming."""
        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response['choices'][0]['text']

    def chat(self,
            messages: list,
            max_tokens: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.95,
            stream: bool = False) -> str:
        """Chat completion with message format."""
        if not self.model:
            raise ValueError("Model not loaded")

        if stream:
            response = ""
            for chunk in self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True
            ):
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {})
                    text = delta.get('content', '')
                    if text:
                        console.print(text, end="")
                        response += text
            return response
        else:
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            return response['choices'][0]['message']['content']