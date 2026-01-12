# tests/conftest.py
"""
Shared pytest fixtures for DSPy Newsletter Research Tool PoC.

These fixtures provide mock objects and sample data for testing
DSPy modules, Weaviate connections, and pipeline components.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime


# =============================================================================
# Environment Setup
# =============================================================================

@pytest.fixture(scope="session")
def mock_openai_key():
    """Ensure OpenAI key is available for testing (mocked)."""
    os.environ.setdefault('OPENAI_API_KEY', 'test-key-for-mocking')
    yield
    # Cleanup not needed - test environment


@pytest.fixture(scope="session")
def mock_cohere_key():
    """Ensure Cohere key is available for testing (mocked)."""
    os.environ.setdefault('COHERE_API_KEY', 'test-cohere-key-for-mocking')
    yield


# =============================================================================
# DSPy Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_dspy_lm():
    """
    Mock DSPy language model for deterministic tests.

    This fixture patches the DSPy settings to prevent actual API calls
    while allowing module structure to be tested.
    """
    with patch('dspy.settings') as mock_settings:
        mock_lm = Mock()
        mock_lm.return_value = Mock()
        mock_settings.lm = mock_lm
        mock_settings.configure = Mock()
        yield mock_settings


@pytest.fixture
def mock_dspy_predict():
    """Mock dspy.Predict for unit testing signatures."""
    with patch('dspy.Predict') as mock_predict:
        mock_instance = Mock()
        mock_predict.return_value = mock_instance
        yield mock_predict


@pytest.fixture
def mock_dspy_chain_of_thought():
    """Mock dspy.ChainOfThought for testing reasoning modules."""
    with patch('dspy.ChainOfThought') as mock_cot:
        mock_instance = Mock()
        mock_cot.return_value = mock_instance
        yield mock_cot


# =============================================================================
# Sample Article Fixtures
# =============================================================================

@pytest.fixture
def sample_article():
    """
    Sample article for testing - highly relevant to background screening.

    This represents an ideal article that should pass prefilter
    and be classified as REGULATORY, N_AMERICA_CARIBBEAN region.
    """
    return {
        'title': 'FCRA Compliance Update 2026',
        'content': '''
        The Fair Credit Reporting Act continues to evolve with new
        amendments proposed for employment background screening. Key changes
        include enhanced consumer notification requirements and stricter
        accuracy standards for consumer reporting agencies.

        The proposed amendments would require:
        - 72-hour notification window for adverse actions
        - Enhanced dispute resolution procedures
        - Annual accuracy audits for CRAs

        Industry stakeholders have until March 2026 to submit comments.
        ''',
        'source_url': 'https://example.com/fcra-update',
        'published_date': '2026-01-10',
        'source_category': 'legal',
        'author': 'Legal Analysis Team'
    }


@pytest.fixture
def irrelevant_article():
    """
    Irrelevant article for negative testing.

    This represents content that should be filtered out in prefilter
    stage with low relevance score.
    """
    return {
        'title': 'Best Pizza Recipes for Weekend Dinners',
        'content': '''
        Today we explore the art of making perfect pizza dough.
        Start with high-quality flour and let the dough rest for
        at least 24 hours for optimal texture. Add your favorite
        toppings and bake at 450 degrees for 12-15 minutes.
        ''',
        'source_url': 'https://example.com/pizza',
        'published_date': '2026-01-10',
        'source_category': 'lifestyle',
        'author': 'Food Blog'
    }


@pytest.fixture
def european_gdpr_article():
    """
    GDPR-focused article for region classification testing.

    Should be classified as EUROPE region with REGULATORY topic.
    """
    return {
        'title': 'GDPR Enforcement Intensifies in Germany',
        'content': '''
        German data protection authorities have announced increased
        enforcement actions targeting employment background check providers.
        Companies must ensure explicit consent and data minimization
        principles are followed for all candidate screening activities.

        Recent fines have exceeded 5 million euros for non-compliance.
        ''',
        'source_url': 'https://example.com/gdpr-germany',
        'published_date': '2026-01-08',
        'source_category': 'regulatory',
        'author': 'EU Compliance Watch'
    }


@pytest.fixture
def apac_article():
    """
    APAC region article for testing geographic classification.
    """
    return {
        'title': 'Singapore Updates Background Check Requirements',
        'content': '''
        The Ministry of Manpower has released new guidelines for
        employment background verification in Singapore. Foreign
        worker screening will now include enhanced credential
        verification through the new SkillsFuture API integration.
        ''',
        'source_url': 'https://example.com/singapore-bg',
        'published_date': '2026-01-09',
        'source_category': 'government',
        'author': 'APAC Business News'
    }


@pytest.fixture
def court_case_article():
    """
    Article about a court case for multi-topic classification testing.

    Should have topics: COURT_CASES, REGULATORY
    """
    return {
        'title': 'FCRA Class Action Settlement Reaches $10M',
        'content': '''
        A major background screening firm has agreed to a $10 million
        settlement in a class action lawsuit alleging FCRA violations.
        The case involved failure to provide proper adverse action notices
        to thousands of job applicants over a five-year period.

        The settlement includes injunctive relief requiring updated
        compliance procedures and third-party auditing.
        ''',
        'source_url': 'https://example.com/fcra-settlement',
        'published_date': '2026-01-05',
        'source_category': 'legal',
        'author': 'Court Watch'
    }


# =============================================================================
# Weaviate Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_weaviate_client():
    """
    Mock Weaviate client for testing without actual database.
    """
    with patch('weaviate.connect_to_local') as mock_connect:
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_client.collections.list_all.return_value = []
        mock_connect.return_value = mock_client
        yield mock_client


@pytest.fixture
def weaviate_test_client():
    """
    Real Weaviate client for integration tests.

    Note: This fixture requires a running Weaviate instance.
    Tests using this fixture should be marked with @pytest.mark.integration
    """
    try:
        import weaviate
        client = weaviate.connect_to_local()
        yield client
        client.close()
    except Exception:
        pytest.skip("Weaviate not available for integration testing")


# =============================================================================
# Pipeline Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_cohere_client():
    """Mock Cohere client for embedding tests."""
    with patch('cohere.Client') as mock_cohere:
        mock_instance = Mock()
        mock_instance.embed.return_value = Mock(embeddings=[[0.1] * 384])
        mock_cohere.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def sample_rss_feed():
    """Sample RSS feed XML for ingestion testing."""
    return '''<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
        <channel>
            <title>Background Screening News</title>
            <link>https://example.com/feed</link>
            <item>
                <title>New FCRA Guidelines Released</title>
                <link>https://example.com/article1</link>
                <description>The CFPB has released new guidance...</description>
                <pubDate>Mon, 10 Jan 2026 12:00:00 GMT</pubDate>
            </item>
            <item>
                <title>Ban the Box Legislation Update</title>
                <link>https://example.com/article2</link>
                <description>Several states have expanded...</description>
                <pubDate>Sun, 09 Jan 2026 10:00:00 GMT</pubDate>
            </item>
        </channel>
    </rss>'''


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
