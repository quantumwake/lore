"""Configuration for the inference CLI."""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Load model configurations from JSON
CONFIG_DIR = Path(__file__).parent
MODELS_FILE = CONFIG_DIR / "models.json"

with open(MODELS_FILE, "r") as f:
    models_config = json.load(f)

AVAILABLE_MODELS = models_config["models"]
DEFAULT_MODEL = models_config.get("default_model", "gpt-neo:2.7b")
DEFAULT_TEMPERATURE = models_config["default_settings"]["temperature"]
DEFAULT_MAX_TOKENS = models_config["default_settings"]["max_tokens"]
DEFAULT_TOP_P = models_config["default_settings"]["top_p"]

# API configurations
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:3000")

# File paths
HISTORY_FILE = ".chat_history.json"