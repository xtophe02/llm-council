"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of OpenRouter model identifiers
COUNCIL_MODELS = [
    "openai/gpt-5.2",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "google/gemini-3-pro-preview"

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"

# Authentication password (required for production)
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD")

# CORS origins (comma-separated for multiple)
# Include port range 3000-3010 for Docker's dynamic port allocation
_default_origins = ",".join([
    "http://localhost:5173",
    *[f"http://localhost:{p}" for p in range(3000, 3011)]
])
CORS_ORIGINS = os.getenv("CORS_ORIGINS", _default_origins).split(",")

# Tavily API key for web search (optional, enables recent info retrieval)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
