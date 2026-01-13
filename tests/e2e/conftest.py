# tests/e2e/conftest.py
"""E2E test fixtures for live Weaviate testing.

These fixtures connect to a real Weaviate instance and provide
utilities for testing the full pipeline end-to-end.
"""

import pytest
import uuid
from typing import Generator, Optional

import weaviate
from weaviate.classes.config import Property, DataType, Configure


# Test collection name to avoid polluting production data
E2E_COLLECTION_NAME = "E2ETestArticles"


def is_weaviate_available() -> bool:
    """Check if Weaviate is running and accessible."""
    try:
        client = weaviate.connect_to_local()
        ready = client.is_ready()
        client.close()
        return ready
    except Exception:
        return False


# Skip all E2E tests if Weaviate is not available
pytestmark = pytest.mark.e2e


@pytest.fixture(scope="session")
def weaviate_available() -> bool:
    """Session-scoped check for Weaviate availability."""
    return is_weaviate_available()


@pytest.fixture(scope="session")
def skip_without_weaviate(weaviate_available):
    """Skip E2E tests if Weaviate is not running."""
    if not weaviate_available:
        pytest.skip("Weaviate is not available - skipping E2E tests")


@pytest.fixture(scope="module")
def live_weaviate_client(skip_without_weaviate) -> Generator[weaviate.WeaviateClient, None, None]:
    """Provide a live Weaviate client for E2E tests.

    Module-scoped to reuse connection across tests in the same file.
    Automatically cleans up the test collection after all tests.
    """
    client = weaviate.connect_to_local()

    try:
        yield client
    finally:
        # Clean up test collection if it exists
        try:
            if client.collections.exists(E2E_COLLECTION_NAME):
                client.collections.delete(E2E_COLLECTION_NAME)
        except Exception:
            pass
        client.close()


@pytest.fixture(scope="function")
def clean_test_collection(live_weaviate_client) -> str:
    """Ensure a clean test collection exists for each test.

    Creates a fresh collection before each test and cleans it up after.
    Returns the collection name for use in tests.
    """
    client = live_weaviate_client

    # Delete if exists from previous failed test
    if client.collections.exists(E2E_COLLECTION_NAME):
        client.collections.delete(E2E_COLLECTION_NAME)

    # Create fresh collection with schema
    properties = [
        Property(name="title", data_type=DataType.TEXT),
        Property(name="content", data_type=DataType.TEXT),
        Property(name="source_url", data_type=DataType.TEXT),
        Property(name="published_date", data_type=DataType.DATE),
        Property(name="region", data_type=DataType.TEXT),
        Property(name="topics", data_type=DataType.TEXT_ARRAY),
        Property(name="relevance_score", data_type=DataType.NUMBER),
        Property(name="confidence", data_type=DataType.NUMBER),
        Property(name="rationale", data_type=DataType.TEXT),
        Property(name="summary", data_type=DataType.TEXT),
        Property(name="content_hash", data_type=DataType.TEXT),
        Property(name="source", data_type=DataType.TEXT),
        Property(name="source_category", data_type=DataType.TEXT),
    ]

    # Try to create with text2vec-openai if available, otherwise no vectorizer
    try:
        client.collections.create(
            name=E2E_COLLECTION_NAME,
            properties=properties,
            vectorizer_config=Configure.Vectorizer.text2vec_openai()
        )
    except Exception:
        # Fallback: create without vectorizer for basic CRUD tests
        client.collections.create(
            name=E2E_COLLECTION_NAME,
            properties=properties,
            vectorizer_config=Configure.Vectorizer.none()
        )

    yield E2E_COLLECTION_NAME

    # Cleanup after test
    if client.collections.exists(E2E_COLLECTION_NAME):
        client.collections.delete(E2E_COLLECTION_NAME)


@pytest.fixture
def e2e_article_store(live_weaviate_client, clean_test_collection):
    """Provide an ArticleStore connected to the test collection."""
    from src.storage import ArticleStore

    store = ArticleStore(
        client=live_weaviate_client,
        collection_name=clean_test_collection
    )
    yield store
    # Don't close - client is managed by live_weaviate_client fixture


@pytest.fixture
def sample_live_articles():
    """Sample articles for E2E testing.

    These articles simulate real ingested content with proper
    classification and metadata.
    """
    return [
        {
            'title': 'FCRA Compliance Update: New Requirements for 2026',
            'content': '''
            The Fair Credit Reporting Act (FCRA) continues to be a critical
            framework for background screening companies. Recent amendments
            require enhanced consumer notification procedures and stricter
            accuracy standards for all consumer reporting agencies.

            Key changes effective 2026:
            - 48-hour notification window for adverse actions
            - Enhanced dispute resolution requirements
            - Mandatory annual accuracy audits
            ''',
            'source_url': 'https://example.com/fcra-2026-update',
            'published_date': '2026-01-10T12:00:00Z',
            'region': 'N_AMERICA_CARIBBEAN',
            'topics': ['REGULATORY', 'CRIMINAL_RECORDS'],
            'relevance_score': 0.95,
            'confidence': 0.9,
            'rationale': 'Highly relevant FCRA regulatory update',
            'source': 'HR Compliance News',
            'source_category': 'legal',
        },
        {
            'title': 'GDPR Impact on Employment Background Checks in Europe',
            'content': '''
            European data protection authorities continue to scrutinize
            employment background screening practices under GDPR. Companies
            must ensure explicit consent and data minimization for all
            candidate verification activities.

            Recent enforcement actions have resulted in significant fines
            for non-compliant screening providers.
            ''',
            'source_url': 'https://example.com/gdpr-employment',
            'published_date': '2026-01-08T10:00:00Z',
            'region': 'EUROPE',
            'topics': ['REGULATORY', 'CREDENTIALS'],
            'relevance_score': 0.92,
            'confidence': 0.88,
            'rationale': 'GDPR compliance for background screening',
            'source': 'EU Compliance Watch',
            'source_category': 'regulatory',
        },
        {
            'title': 'Ban the Box Laws Expand to Additional States',
            'content': '''
            Several US states have enacted new ban the box legislation
            affecting employment screening practices. These laws delay
            criminal history inquiries until later in the hiring process.

            Employers must update their application processes to comply
            with these new state-level requirements.
            ''',
            'source_url': 'https://example.com/ban-the-box-expansion',
            'published_date': '2026-01-05T14:00:00Z',
            'region': 'N_AMERICA_CARIBBEAN',
            'topics': ['REGULATORY', 'CRIMINAL_RECORDS'],
            'relevance_score': 0.88,
            'confidence': 0.85,
            'rationale': 'Ban the box legislation update',
            'source': 'Employment Law Update',
            'source_category': 'legal',
        },
    ]


@pytest.fixture
def live_rss_feed_urls():
    """URLs for real RSS feeds to test with.

    These are publicly available RSS feeds that can be used
    for live ingestion testing.
    """
    return [
        # Note: These are example URLs - replace with actual working feeds
        'https://www.shrm.org/rss/pages/rss.aspx',  # SHRM HR News
        'https://www.hrdive.com/feeds/news/',  # HR Dive News
    ]
