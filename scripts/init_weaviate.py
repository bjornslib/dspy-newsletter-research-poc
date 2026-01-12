#!/usr/bin/env python3
# scripts/init_weaviate.py
"""Initialize Weaviate with NewsletterArticles collection schema.

This script creates the NewsletterArticles collection with:
- Required properties for newsletter article storage
- text2vec-openai vectorizer configuration
- reranker-cohere reranker configuration

Usage:
    python scripts/init_weaviate.py
"""

import weaviate
from weaviate.classes.config import (
    Configure,
    Property,
    DataType,
)
import sys


def create_newsletter_collection(client: weaviate.WeaviateClient) -> bool:
    """Create the NewsletterArticles collection with schema.

    Args:
        client: Connected Weaviate client.

    Returns:
        True if collection was created successfully, False otherwise.
    """
    collection_name = "NewsletterArticles"

    # Check if collection already exists
    if client.collections.exists(collection_name):
        print(f"Collection '{collection_name}' already exists.")
        return True

    try:
        # Create collection with schema
        client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.text2vec_openai(
                model="text-embedding-3-small"
            ),
            reranker_config=Configure.Reranker.cohere(
                model="rerank-english-v3.0"
            ),
            properties=[
                Property(
                    name="title",
                    data_type=DataType.TEXT,
                    description="Article title"
                ),
                Property(
                    name="content",
                    data_type=DataType.TEXT,
                    description="Full article content"
                ),
                Property(
                    name="summary",
                    data_type=DataType.TEXT,
                    description="Article summary"
                ),
                Property(
                    name="source_url",
                    data_type=DataType.TEXT,
                    description="Source URL of the article"
                ),
                Property(
                    name="source",
                    data_type=DataType.TEXT,
                    description="Source name/publication"
                ),
                Property(
                    name="region",
                    data_type=DataType.TEXT,
                    description="Geographic region",
                    skip_vectorization=True
                ),
                Property(
                    name="country",
                    data_type=DataType.TEXT,
                    description="Country",
                    skip_vectorization=True
                ),
                Property(
                    name="topics",
                    data_type=DataType.TEXT_ARRAY,
                    description="Article topics",
                    skip_vectorization=True
                ),
                Property(
                    name="relevance_score",
                    data_type=DataType.NUMBER,
                    description="Relevance score (0-1)",
                    skip_vectorization=True
                ),
                Property(
                    name="reasoning",
                    data_type=DataType.TEXT,
                    description="Reasoning for relevance score"
                ),
                Property(
                    name="published_date",
                    data_type=DataType.DATE,
                    description="Publication date",
                    skip_vectorization=True
                ),
                Property(
                    name="ingested_at",
                    data_type=DataType.DATE,
                    description="Ingestion timestamp",
                    skip_vectorization=True
                ),
            ]
        )
        print(f"Collection '{collection_name}' created successfully.")
        return True

    except Exception as e:
        print(f"Error creating collection: {e}")
        return False


def main():
    """Main function to initialize Weaviate."""
    print("Connecting to Weaviate...")

    try:
        client = weaviate.connect_to_local()
        print("Connected to Weaviate successfully.")

        # Check if Weaviate is ready
        if not client.is_ready():
            print("Error: Weaviate is not ready.")
            sys.exit(1)

        # Create the collection
        success = create_newsletter_collection(client)

        # List all collections
        print("\nExisting collections:")
        collections = client.collections.list_all()
        for name in collections:
            print(f"  - {name}")

        client.close()

        if success:
            print("\nWeaviate initialization complete!")
            sys.exit(0)
        else:
            print("\nWeaviate initialization failed!")
            sys.exit(1)

    except Exception as e:
        print(f"Error connecting to Weaviate: {e}")
        print("Make sure Weaviate is running: docker compose up -d weaviate")
        sys.exit(1)


if __name__ == "__main__":
    main()
