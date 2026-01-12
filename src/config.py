# src/config.py
"""Configuration module for DSPy Newsletter Research Tool.

This module handles:
- Loading environment variables from .env
- Configuring DSPy with the appropriate language model
- Providing configuration constants for the application
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import dspy


def load_env():
    """Load environment variables from .env file.

    Searches for .env in the project root directory.
    """
    # Find project root (parent of src/)
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'

    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try loading from current working directory
        load_dotenv()


def configure_dspy(model: str = "gpt-4o-mini") -> None:
    """Configure DSPy with OpenAI language model.

    Args:
        model: The OpenAI model to use. Defaults to 'gpt-4o-mini'.

    Raises:
        ValueError: If OPENAI_API_KEY is not set.
    """
    # Load environment variables
    load_env()

    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    # Configure DSPy with OpenAI LM
    lm = dspy.LM(
        model=f"openai/{model}",
        api_key=api_key
    )
    dspy.settings.configure(lm=lm)


# Configuration constants
WEAVIATE_URL = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
WEAVIATE_GRPC_PORT = 50051

# Collection configuration
NEWSLETTER_COLLECTION_NAME = "NewsletterArticles"
