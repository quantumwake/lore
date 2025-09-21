# UberCLI - Modular Inference CLI

A modular command-line interface for running inference with both standard transformer models and GGUF quantized models locally on Apple Silicon (MPS), CUDA, or CPU with MCP server integration.

## Features

- ✅ Support for standard transformer models (GPT-Neo, GPT-J)
- ✅ Support for GGUF quantized models (Llama, Mistral, Mixtral, etc.)
- ✅ Automatic device detection (MPS for Apple Silicon, CUDA, CPU)
- ✅ MCP (Model Context Protocol) server integration
- ✅ Interactive chat mode with history
- ✅ Streaming responses
- ✅ Modular architecture for easy extension

## Project Structure

```
ubercli/
├── cli.py              # Main CLI entry point
├── models.py           # Transformer model management
├── gguf_models.py      # GGUF quantized model support
├── mcp_integration.py  # MCP server client
├── utils.py            # Utility functions
├── config.py           # Configuration loader
├── models.json         # Model definitions
├── requirements.txt    # Python dependencies
└── .env.example        # Environment variables template
```

## Setup

```bash
# Install dependencies with uv
uv pip install -r requirements.txt

# For Metal acceleration on Apple Silicon (optional)
CMAKE_ARGS="-DLLAMA_METAL=on" uv pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

# Copy and configure environment variables
cp .env.example .env

# Make the CLI executable
chmod +x cli.py
```

## Usage

### Basic Chat
```bash
# Using default model (GPT-Neo 2.7B)
python cli.py chat "What is machine learning?"

# Using GGUF quantized model
python cli.py chat -m llama2:7b-gguf "Explain quantum computing"

# With system prompt
python cli.py chat -s "You are a helpful coding assistant" "Write a Python function"

# Stream response
python cli.py chat --stream "Tell me a story"

# With MCP context
python cli.py chat --mcp-context "Analyze this data"
```

### Interactive Mode
```bash
# Start interactive chat
python cli.py interactive

# With specific model
python cli.py interactive -m mistral:7b-gguf

# With MCP context enabled
python cli.py interactive --mcp-context
```

### Model Management
```bash
# List all available models
python cli.py list-models

# Preload a model
python cli.py preload -m gpt-j:6b
```

### MCP Integration
```bash
# Execute MCP tool
python cli.py mcp-tool -t search -p '{"query": "test"}'

# List available MCP tools
python cli.py list-mcp-tools
```

## Available Models

### Transformer Models
- `gpt-neo:2.7b` - GPT-Neo 2.7B (default)
- `gpt-j:6b` - GPT-J 6B

### GGUF Quantized Models
- `gpt-oss:120b-gguf` - 120B parameter quantized model
- `llama2:7b-gguf` - Llama 2 7B quantized
- `llama2:13b-gguf` - Llama 2 13B quantized
- `llama2:70b-gguf` - Llama 2 70B quantized
- `mistral:7b-gguf` - Mistral 7B Instruct quantized
- `mixtral:8x7b-gguf` - Mixtral 8x7B MoE quantized
- `codellama:7b-gguf` - Code Llama 7B quantized
- `phi2:3b-gguf` - Microsoft Phi-2 2.7B quantized

## Adding Custom Models

Edit `models.json` to add new models:

```json
{
  "models": {
    "custom-model": {
      "type": "gguf",
      "repo_id": "username/model-name-GGUF",
      "context_length": 4096,
      "description": "Custom model description"
    }
  }
}
```

## Configuration

Set environment variables in `.env`:

- `MCP_SERVER_URL`: MCP server URL (default: http://localhost:3000)# lore
