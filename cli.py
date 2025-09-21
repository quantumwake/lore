#!/usr/bin/env python3
"""Main CLI entry point for the inference tool."""

import click
import sys
import json
from rich.console import Console

from config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P,
    HISTORY_FILE
)
from models import ModelManager
from gguf_models import GGUFModelManager
from mcp_integration import MCPClient
from utils import (
    load_conversation_history,
    save_conversation_history,
    format_prompt,
    format_chat_messages,
    display_models_table,
    print_markdown,
    confirm_action,
    get_model_path
)

console = Console()


class InferenceContext:
    """Context manager for model and MCP client."""

    def __init__(self):
        self.transformer_manager = None
        self.gguf_manager = None
        self.mcp_client = None
        self.current_model = None
        self.current_model_type = None

    def get_model_manager(self, model_key: str):
        """Get appropriate model manager based on model type."""
        model_info = AVAILABLE_MODELS.get(model_key)
        if not model_info:
            raise ValueError(f"Model {model_key} not found")

        model_type = model_info.get("type")

        if model_type == "transformers":
            if not self.transformer_manager:
                self.transformer_manager = ModelManager()
            self.current_model_type = "transformers"
            return self.transformer_manager
        elif model_type == "gguf":
            if not self.gguf_manager:
                self.gguf_manager = GGUFModelManager()
            self.current_model_type = "gguf"
            return self.gguf_manager
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def get_mcp_client(self):
        """Get MCP client instance."""
        if not self.mcp_client:
            self.mcp_client = MCPClient()
        return self.mcp_client


@click.group()
@click.pass_context
def cli(ctx):
    """Inference CLI for running language models with MCP integration."""
    ctx.ensure_object(dict)
    ctx.obj['context'] = InferenceContext()


@cli.command()
@click.option('--model', '-m', default=DEFAULT_MODEL,
              type=click.Choice(list(AVAILABLE_MODELS.keys())),
              help='Model to use for inference')
@click.option('--temperature', '-t', default=DEFAULT_TEMPERATURE, type=float,
              help='Temperature for sampling (0.0-2.0)')
@click.option('--max-tokens', default=DEFAULT_MAX_TOKENS, type=int,
              help='Maximum tokens to generate')
@click.option('--top-p', default=DEFAULT_TOP_P, type=float,
              help='Top-p sampling parameter')
@click.option('--system', '-s', default=None,
              help='System prompt')
@click.option('--mcp-context', is_flag=True,
              help='Fetch and include MCP server context')
@click.option('--stream', is_flag=True,
              help='Stream the response')
@click.option('--json-output', is_flag=True,
              help='Output raw JSON response')
@click.argument('prompt', required=False)
@click.pass_context
def chat(ctx, model, temperature, max_tokens, top_p, system, mcp_context, stream, json_output, prompt):
    """Run a single chat interaction."""
    if not prompt:
        console.print("[cyan]Enter your prompt (Ctrl+D when done):[/cyan]")
        prompt = sys.stdin.read().strip()

    if not prompt:
        console.print("[red]Error: No prompt provided[/red]")
        return

    inference_ctx = ctx.obj['context']

    # Get MCP context if requested
    context = None
    if mcp_context:
        mcp_client = inference_ctx.get_mcp_client()
        context = mcp_client.get_context(prompt)

    # Format prompt
    model_info = AVAILABLE_MODELS[model]

    if model_info.get("type") == "gguf":
        # GGUF models use chat format
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})
        messages.append({"role": "user", "content": prompt})

        manager = inference_ctx.get_model_manager(model)
        console.print(f"[dim]Using model: {model}[/dim]")
        console.print("[dim]---[/dim]")

        try:
            # Load model
            manager.load_model(
                repo_id=model_info.get("repo_id"),
                n_ctx=model_info.get("context_length", 2048)
            )

            # Generate response
            response = manager.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream
            )

            if not stream:
                if json_output:
                    console.print(json.dumps({"response": response}, indent=2))
                else:
                    print_markdown(response)

        except Exception as e:
            console.print(f"[red]Error during inference: {e}[/red]")
            sys.exit(1)

    else:
        # Transformer models use prompt format
        full_prompt = format_prompt(prompt, system, context)

        manager = inference_ctx.get_model_manager(model)
        console.print(f"[dim]Using model: {model}[/dim]")
        console.print("[dim]---[/dim]")

        try:
            # Load model
            manager.load_model(model)

            # Generate response
            response = manager.generate(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream
            )

            if not stream:
                if json_output:
                    console.print(json.dumps({"response": response}, indent=2))
                else:
                    print_markdown(response)

        except Exception as e:
            console.print(f"[red]Error during inference: {e}[/red]")
            sys.exit(1)


@cli.command()
@click.option('--model', '-m', default=DEFAULT_MODEL,
              type=click.Choice(list(AVAILABLE_MODELS.keys())),
              help='Model to use for inference')
@click.option('--temperature', '-t', default=DEFAULT_TEMPERATURE, type=float,
              help='Temperature for sampling')
@click.option('--max-tokens', default=DEFAULT_MAX_TOKENS, type=int,
              help='Maximum tokens to generate')
@click.option('--history-file', default=HISTORY_FILE,
              help='File to store conversation history')
@click.option('--mcp-context', is_flag=True,
              help='Fetch and include MCP server context for each message')
@click.pass_context
def interactive(ctx, model, temperature, max_tokens, history_file, mcp_context):
    """Start an interactive chat session."""
    inference_ctx = ctx.obj['context']

    # Load conversation history
    conversation_history = load_conversation_history(history_file)

    if conversation_history:
        console.print(f"[dim]Loaded {len(conversation_history)} messages from history[/dim]")

    console.print(f"[cyan]Interactive mode with {model}[/cyan]")
    console.print("[dim]Commands: 'exit'/'quit' to end, 'clear' to reset history, 'save' to save history[/dim]")
    console.print("[dim]---[/dim]")

    # Get model manager and load model
    manager = inference_ctx.get_model_manager(model)
    model_info = AVAILABLE_MODELS[model]

    if model_info.get("type") == "gguf":
        manager.load_model(
            repo_id=model_info.get("repo_id"),
            n_ctx=model_info.get("context_length", 2048)
        )
    else:
        manager.load_model(model)

    # Get MCP client if needed
    mcp_client = None
    if mcp_context:
        mcp_client = inference_ctx.get_mcp_client()

    while True:
        try:
            user_input = click.prompt("You", default="", show_default=False)

            if user_input.lower() in ['exit', 'quit']:
                if confirm_action("Save conversation history before exiting?", default=True):
                    save_conversation_history(history_file, conversation_history)
                break

            if user_input.lower() == 'clear':
                conversation_history = []
                console.print("[yellow]History cleared[/yellow]")
                continue

            if user_input.lower() == 'save':
                save_conversation_history(history_file, conversation_history)
                console.print("[green]History saved[/green]")
                continue

            if not user_input:
                continue

            # Get MCP context if enabled
            context = None
            if mcp_client:
                context = mcp_client.get_context(user_input)

            console.print(f"[green]Assistant:[/green] ", end="")

            # Generate response based on model type
            if model_info.get("type") == "gguf":
                messages = []

                # Add recent history
                for msg in conversation_history[-6:]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

                if context:
                    messages.insert(0, {"role": "system", "content": f"Context: {context}"})

                messages.append({"role": "user", "content": user_input})

                response = manager.chat(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                )
            else:
                # Format prompt with history
                prompt = format_chat_messages(conversation_history[-4:])

                if context:
                    prompt = f"Context: {context}\n\n" + prompt

                prompt += f"User: {user_input}\nAssistant:"

                response = manager.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                )

            # Save to history
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted[/yellow]")
            if confirm_action("Save conversation history?", default=True):
                save_conversation_history(history_file, conversation_history)
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


@cli.command()
@click.option('--tool', '-t', required=True,
              help='MCP tool name to execute')
@click.option('--params', '-p', default='{}',
              help='JSON parameters for the tool')
@click.pass_context
def mcp_tool(ctx, tool, params):
    """Execute an MCP tool."""
    inference_ctx = ctx.obj['context']
    mcp_client = inference_ctx.get_mcp_client()

    try:
        params_dict = json.loads(params)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON parameters: {e}[/red]")
        return

    console.print(f"[cyan]Executing MCP tool: {tool}[/cyan]")
    result = mcp_client.execute_tool(tool, params_dict)

    if result:
        console.print(json.dumps(result, indent=2))
    else:
        console.print("[red]Tool execution failed[/red]")


@cli.command()
def list_models():
    """List all available models."""
    display_models_table(AVAILABLE_MODELS)


@cli.command()
@click.option('--model', '-m', default=DEFAULT_MODEL,
              type=click.Choice(list(AVAILABLE_MODELS.keys())),
              help='Model to preload')
@click.pass_context
def preload(ctx, model):
    """Preload a model into memory."""
    inference_ctx = ctx.obj['context']

    console.print(f"[cyan]Preloading {model}...[/cyan]")

    try:
        manager = inference_ctx.get_model_manager(model)
        model_info = AVAILABLE_MODELS[model]

        if model_info.get("type") == "gguf":
            manager.load_model(
                repo_id=model_info.get("repo_id"),
                n_ctx=model_info.get("context_length", 2048)
            )
        else:
            manager.load_model(model)

        console.print(f"[green]Successfully loaded {model}[/green]")

        if hasattr(manager, 'get_memory_usage'):
            console.print(manager.get_memory_usage())

    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")


@cli.command()
def list_mcp_tools():
    """List available MCP tools."""
    mcp_client = MCPClient()
    tools = mcp_client.list_tools()

    if tools:
        console.print("[cyan]Available MCP Tools:[/cyan]\n")
        for tool in tools:
            console.print(f"  [bold]{tool.get('name', 'unknown')}[/bold]")
            if 'description' in tool:
                console.print(f"    {tool['description']}")
            console.print()
    else:
        console.print("[yellow]No MCP tools available or server not reachable[/yellow]")


if __name__ == "__main__":
    cli(obj={})