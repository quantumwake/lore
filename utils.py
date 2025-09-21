"""Utilities for the inference CLI."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown

console = Console()


def load_conversation_history(history_file: str) -> List[Dict[str, str]]:
    """Load conversation history from file."""
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[yellow]Could not load history: {e}[/yellow]")
    return []


def save_conversation_history(history_file: str, messages: List[Dict[str, str]]):
    """Save conversation history to file."""
    try:
        with open(history_file, 'w') as f:
            json.dump(messages, f, indent=2)
    except Exception as e:
        console.print(f"[yellow]Could not save history: {e}[/yellow]")


def format_prompt(user_input: str, system: str = None, context: str = None) -> str:
    """Format a prompt with optional system and context."""
    prompt_parts = []

    if system:
        prompt_parts.append(f"System: {system}")

    if context:
        prompt_parts.append(f"Context: {context}")

    prompt_parts.append(f"User: {user_input}")
    prompt_parts.append("Assistant:")

    return "\n\n".join(prompt_parts)


def format_chat_messages(messages: List[Dict[str, str]]) -> str:
    """Format chat messages for prompt."""
    prompt = ""
    for msg in messages:
        role = msg['role'].capitalize()
        content = msg['content']
        prompt += f"{role}: {content}\n"
    return prompt


def display_models_table(models: Dict[str, Any]):
    """Display available models in a formatted table."""
    table = Table(title="Available Models", show_header=True, header_style="bold cyan")
    table.add_column("Model ID", style="bold")
    table.add_column("Type", style="dim")
    table.add_column("Description")
    table.add_column("Context", justify="right")

    for model_id, model_info in models.items():
        model_type = model_info.get("type", "unknown")
        description = model_info.get("description", "")
        context = f"{model_info.get('context_length', 0):,}"

        table.add_row(model_id, model_type, description, context)

    console.print(table)


def print_markdown(text: str):
    """Print text as formatted markdown."""
    md = Markdown(text)
    console.print(md)


def confirm_action(message: str, default: bool = False) -> bool:
    """Ask user for confirmation."""
    suffix = " [Y/n]" if default else " [y/N]"
    response = console.input(f"[yellow]{message}{suffix}[/yellow] ").lower().strip()

    if not response:
        return default

    return response in ['y', 'yes']


def get_model_path(model_key: str, models_config: Dict[str, Any]) -> tuple:
    """Get model path and type from configuration."""
    model_info = models_config.get(model_key)
    if not model_info:
        raise ValueError(f"Model {model_key} not found in configuration")

    model_type = model_info.get("type")

    if model_type == "transformers":
        return model_info.get("name"), model_type
    elif model_type == "gguf":
        return model_info.get("repo_id"), model_type
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def ensure_cache_dir(cache_dir: str = "~/.cache/ubercli") -> Path:
    """Ensure cache directory exists."""
    cache_path = Path(os.path.expanduser(cache_dir))
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path