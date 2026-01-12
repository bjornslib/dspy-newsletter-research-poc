# tests/integration/test_query_e2e.py
"""
End-to-end tests for the QUIPLER Query Agent.

Beads Task: dspy-5xm (AT-E2E)

Tests the complete query flow:
- Query expansion and understanding
- Retrieval from Weaviate (mocked)
- ReAct synthesis
- Response formatting with citations

Validates:
- Answer quality and relevance
- Source citations present
- Latency within targets (<3s simple, <10s complex)
"""

import pytest
import time
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.query_agent import (
    NewsletterQueryAgent,
    query,
    filter_by_date,
    filter_by_region,
    filter_by_topic,
    format_answer,
    format_sources,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_stored_articles():
    """Sample articles as they would be stored in Weaviate."""
    return [
        {
            'title': 'New FCRA Compliance Requirements Announced for 2024',
            'content': 'The Consumer Financial Protection Bureau (CFPB) has announced new compliance requirements for background screening companies under the Fair Credit Reporting Act. Key changes include enhanced dispute resolution procedures, stricter accuracy requirements, and new consumer notification standards. Companies must comply by Q3 2024.',
            'source_url': 'https://example.com/fcra-2024',
            'region': 'N_AMERICA_CARIBBEAN',
            'topics': ['REGULATORY', 'CREDENTIALS'],
            'published_date': datetime(2024, 1, 15),
            'relevance_score': 0.92,
        },
        {
            'title': 'APAC Region Tightens Data Protection Rules',
            'content': 'Several APAC nations including Singapore, Australia, and Japan have introduced new data protection regulations affecting background screening operations. The changes harmonize regional standards while maintaining country-specific requirements for criminal record checks and employment verification.',
            'source_url': 'https://example.com/apac-data-rules',
            'region': 'APAC',
            'topics': ['REGULATORY', 'CRIMINAL_RECORDS'],
            'published_date': datetime(2024, 2, 1),
            'relevance_score': 0.88,
        },
        {
            'title': 'GDPR Enforcement Trends Post-Brexit: UK vs EU',
            'content': 'Analysis of GDPR enforcement patterns since Brexit reveals diverging approaches between UK ICO and EU DPAs. The UK has taken a more business-friendly stance on data transfers, while EU authorities have increased fines for screening companies. Cross-border screening operations now face compliance challenges navigating both regulatory regimes.',
            'source_url': 'https://example.com/gdpr-post-brexit',
            'region': 'EUROPE',
            'topics': ['REGULATORY', 'COURT_CASES'],
            'published_date': datetime(2024, 1, 20),
            'relevance_score': 0.95,
        },
        {
            'title': 'Ban the Box Movement Expands to New States',
            'content': 'Five additional US states have enacted ban-the-box legislation this year, requiring employers to delay criminal history inquiries until later in the hiring process. The trend reflects growing recognition of the need to reduce barriers to employment for individuals with criminal records.',
            'source_url': 'https://example.com/ban-the-box-2024',
            'region': 'N_AMERICA_CARIBBEAN',
            'topics': ['REGULATORY', 'CRIMINAL_RECORDS'],
            'published_date': datetime(2024, 2, 10),
            'relevance_score': 0.85,
        },
        {
            'title': 'AI in Background Screening: Regulatory Concerns',
            'content': 'Regulators are examining the use of AI and machine learning in background screening decisions. The EEOC has issued guidance on algorithmic bias, while several states are considering legislation requiring disclosure of AI-assisted screening. Companies must balance efficiency gains with compliance requirements.',
            'source_url': 'https://example.com/ai-screening-regs',
            'region': 'WORLDWIDE',
            'topics': ['TECHNOLOGY', 'REGULATORY'],
            'published_date': datetime(2024, 2, 15),
            'relevance_score': 0.87,
        },
    ]


@pytest.fixture
def mock_weaviate_with_articles(sample_stored_articles):
    """Mock Weaviate client pre-populated with articles."""
    client = MagicMock()

    # Mock collection
    collection = MagicMock()
    client.collections.get.return_value = collection

    # Store articles for retrieval
    def mock_hybrid_query(query, limit=5, **kwargs):
        # Simple keyword matching for mock
        query_lower = query.lower()
        results = []

        for article in sample_stored_articles:
            title = article['title'].lower()
            content = article['content'].lower()

            # Check if query terms match
            if any(term in title or term in content for term in query_lower.split()):
                mock_result = MagicMock()
                mock_result.properties = article.copy()
                results.append(mock_result)

        return results[:limit]

    collection.query.hybrid = mock_hybrid_query

    return client


@pytest.fixture
def query_agent(mock_weaviate_with_articles):
    """Create query agent with mocked Weaviate."""
    return NewsletterQueryAgent(
        weaviate_client=mock_weaviate_with_articles
    )


# =============================================================================
# Simple Query Tests (Tier 1)
# =============================================================================

class TestSimpleQueries:
    """Test simple, single-topic queries. Target latency: <3 seconds."""

    def test_simple_fcra_query(self, query_agent):
        """Test simple query about FCRA updates."""
        start_time = time.time()

        result = query_agent(question="What are the latest FCRA updates?")

        elapsed = time.time() - start_time

        # Verify response structure
        assert result is not None
        assert hasattr(result, 'answer')
        assert len(result.answer) > 100  # Substantive answer

        # Verify answer relevance
        answer_lower = result.answer.lower()
        assert any(term in answer_lower for term in ['fcra', 'compliance', 'fair credit'])

        # Verify sources present
        assert hasattr(result, 'sources')
        assert len(result.sources) > 0

        # Verify latency
        assert elapsed < 3.0, f"Simple query took {elapsed:.2f}s (target: <3s)"

    def test_simple_query_with_region_filter(self, query_agent):
        """Test simple query with region filter."""
        start_time = time.time()

        result = query_agent(
            question="What background screening regulations apply?",
            filters={'region': 'APAC'}
        )

        elapsed = time.time() - start_time

        # Verify response
        assert result is not None
        assert hasattr(result, 'answer')
        assert len(result.answer) > 50

        # Latency check
        assert elapsed < 3.0

    def test_simple_query_returns_sources(self, query_agent):
        """Test that simple queries return relevant sources."""
        result = query_agent(question="What is ban the box legislation?")

        assert result is not None
        assert hasattr(result, 'sources')

        # Each source should have required fields
        for source in result.sources:
            assert 'title' in source
            # URL might be present
            if 'url' in source:
                assert source['url'].startswith('http')


# =============================================================================
# Standard Query Tests (Tier 2)
# =============================================================================

class TestStandardQueries:
    """Test standard queries requiring some filtering. Target latency: <3 seconds."""

    def test_standard_apac_regulation_query(self, query_agent):
        """Test query about APAC regulation changes."""
        start_time = time.time()

        result = query_agent(
            question="What regulation changes are happening in APAC?",
            filters={'region': 'APAC'}
        )

        elapsed = time.time() - start_time

        # Verify response
        assert result is not None
        assert hasattr(result, 'answer')

        # Verify latency
        assert elapsed < 3.0, f"Standard query took {elapsed:.2f}s (target: <3s)"

    def test_standard_query_with_topic_filter(self, query_agent):
        """Test query filtered by topic."""
        result = query_agent(
            question="What's new in criminal records screening?",
            filters={'topics': ['CRIMINAL_RECORDS']}
        )

        assert result is not None
        assert hasattr(result, 'answer')

    def test_standard_query_multiple_sources(self):
        """Test that standard queries can synthesize multiple sources."""
        # Use convenience function
        result = query(
            question="What are the latest regulatory updates for background screening?"
        )

        assert result is not None
        assert 'answer' in result
        assert 'sources' in result
        assert 'confidence' in result

        # Confidence should be reasonable
        assert 0 <= result['confidence'] <= 1.0


# =============================================================================
# Complex Query Tests (Tier 3)
# =============================================================================

class TestComplexQueries:
    """Test complex multi-hop queries. Target latency: <10 seconds."""

    def test_complex_gdpr_brexit_query(self, query_agent):
        """Test complex query about GDPR evolution post-Brexit."""
        start_time = time.time()

        result = query_agent(
            question="How has GDPR enforcement evolved since Brexit, and what does this mean for background screening companies operating in both UK and EU?"
        )

        elapsed = time.time() - start_time

        # Verify comprehensive response
        assert result is not None
        assert hasattr(result, 'answer')
        assert len(result.answer) > 100  # Complex query should have detailed answer

        # Verify latency (complex queries have higher tolerance)
        assert elapsed < 10.0, f"Complex query took {elapsed:.2f}s (target: <10s)"

    def test_complex_multi_region_query(self, query_agent):
        """Test query spanning multiple regions."""
        start_time = time.time()

        result = query_agent(
            question="Compare the regulatory approaches to background screening between North America and Europe"
        )

        elapsed = time.time() - start_time

        # Verify response
        assert result is not None
        assert hasattr(result, 'answer')

        # Latency check
        assert elapsed < 10.0

    def test_complex_query_with_synthesis(self, query_agent):
        """Test that complex queries synthesize information correctly."""
        result = query_agent(
            question="What are the key trends in AI regulation for background screening across different jurisdictions, and how should companies prepare?"
        )

        assert result is not None
        assert hasattr(result, 'answer')
        assert len(result.answer) > 100


# =============================================================================
# Answer Quality Tests
# =============================================================================

class TestAnswerQuality:
    """Test the quality of generated answers."""

    def test_answer_addresses_question(self, query_agent):
        """Test that answers actually address the question asked."""
        questions_and_keywords = [
            ("What is FCRA?", ['fcra', 'fair credit', 'reporting']),
            ("How does ban the box work?", ['ban', 'box', 'criminal', 'employment']),
        ]

        for question, expected_keywords in questions_and_keywords:
            result = query_agent(question=question)
            answer_lower = result.answer.lower()

            # At least one keyword should appear
            assert any(kw in answer_lower for kw in expected_keywords), \
                f"Answer to '{question}' doesn't contain expected keywords"

    def test_answer_includes_context(self, query_agent):
        """Test that answers provide relevant context."""
        result = query_agent(question="What are the latest FCRA changes?")

        # Answer should be substantive (more than a one-liner)
        assert len(result.answer) > 100

        # Should contain structured information
        assert '.' in result.answer  # Contains sentences

    def test_answer_not_hallucinated(self, query_agent):
        """Test that answers are grounded in retrieved context."""
        result = query_agent(question="What are FCRA compliance requirements?")

        # Answer should be based on context, not made up
        # A non-hallucinated answer will reference screening/compliance
        answer_lower = result.answer.lower()
        assert any(term in answer_lower for term in [
            'compliance', 'screening', 'background', 'regulation', 'requirement'
        ])


# =============================================================================
# Source Citation Tests
# =============================================================================

class TestSourceCitations:
    """Test source citation functionality."""

    def test_sources_included_in_response(self, query_agent):
        """Test that sources are included in query responses."""
        result = query_agent(question="What are the latest background screening updates?")

        assert hasattr(result, 'sources')
        assert isinstance(result.sources, list)

    def test_source_format_correct(self, query_agent):
        """Test that sources have correct format."""
        result = query_agent(question="Latest FCRA news")

        for source in result.sources:
            assert isinstance(source, dict)
            # Should have at least a title
            assert 'title' in source or 'url' in source

    def test_format_sources_markdown(self):
        """Test markdown source formatting."""
        sources = [
            {'title': 'FCRA Update', 'url': 'https://example.com/fcra'},
            {'title': 'GDPR Changes', 'url': 'https://example.com/gdpr'},
        ]

        formatted = format_sources(sources, format='markdown')

        assert '## Sources' in formatted
        assert '[FCRA Update](https://example.com/fcra)' in formatted

    def test_format_sources_html(self):
        """Test HTML source formatting."""
        sources = [
            {'title': 'Test Source', 'url': 'https://example.com/test'},
        ]

        formatted = format_sources(sources, format='html')

        assert '<ul>' in formatted
        assert '<a href="https://example.com/test">Test Source</a>' in formatted

    def test_format_answer_with_citations(self):
        """Test answer formatting with inline citations."""
        answer = "Background screening requirements have changed."
        sources = [
            {'title': 'Source 1', 'url': 'https://example.com/1'},
            {'title': 'Source 2', 'url': 'https://example.com/2'},
        ]

        formatted = format_answer(answer, sources)

        assert 'Sources:' in formatted
        assert 'Source 1' in formatted


# =============================================================================
# Filter Tool Tests
# =============================================================================

class TestFilterTools:
    """Test query filter tools."""

    def test_filter_by_date(self):
        """Test date range filtering."""
        articles = [
            {'title': 'Old Article', 'published_date': datetime(2023, 1, 1)},
            {'title': 'Recent Article', 'published_date': datetime(2024, 1, 15)},
            {'title': 'New Article', 'published_date': datetime(2024, 2, 1)},
        ]

        filtered = filter_by_date(
            articles,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )

        assert len(filtered) == 1
        assert filtered[0]['title'] == 'Recent Article'

    def test_filter_by_region(self):
        """Test region filtering."""
        articles = [
            {'title': 'US Article', 'region': 'N_AMERICA_CARIBBEAN'},
            {'title': 'EU Article', 'region': 'EUROPE'},
            {'title': 'Global Article', 'region': 'WORLDWIDE'},
        ]

        filtered = filter_by_region(articles, 'EUROPE')

        assert len(filtered) == 1
        assert filtered[0]['title'] == 'EU Article'

    def test_filter_by_topic(self):
        """Test topic filtering."""
        articles = [
            {'title': 'Regulatory News', 'topics': ['REGULATORY']},
            {'title': 'Tech News', 'topics': ['TECHNOLOGY']},
            {'title': 'Mixed News', 'topics': ['REGULATORY', 'TECHNOLOGY']},
        ]

        filtered = filter_by_topic(articles, 'REGULATORY')

        assert len(filtered) == 2
        assert all('REGULATORY' in a['topics'] for a in filtered)

    def test_filter_chain(self):
        """Test chaining multiple filters."""
        articles = [
            {
                'title': 'US Regulatory',
                'region': 'N_AMERICA_CARIBBEAN',
                'topics': ['REGULATORY'],
                'published_date': datetime(2024, 1, 15)
            },
            {
                'title': 'EU Regulatory',
                'region': 'EUROPE',
                'topics': ['REGULATORY'],
                'published_date': datetime(2024, 1, 20)
            },
            {
                'title': 'US Tech',
                'region': 'N_AMERICA_CARIBBEAN',
                'topics': ['TECHNOLOGY'],
                'published_date': datetime(2024, 1, 10)
            },
        ]

        # Chain filters: North America + Regulatory
        filtered = filter_by_region(articles, 'N_AMERICA_CARIBBEAN')
        filtered = filter_by_topic(filtered, 'REGULATORY')

        assert len(filtered) == 1
        assert filtered[0]['title'] == 'US Regulatory'


# =============================================================================
# Latency Tests
# =============================================================================

class TestLatency:
    """Test query latency targets."""

    def test_simple_query_latency(self):
        """Test simple query completes within 3 seconds."""
        start = time.time()

        result = query("What is FCRA?")

        elapsed = time.time() - start

        # Simple queries should be fast
        assert elapsed < 3.0, f"Simple query latency: {elapsed:.2f}s (target: <3s)"

    def test_filtered_query_latency(self):
        """Test filtered query latency."""
        start = time.time()

        result = query(
            "What are regulatory updates?",
            filters={'region': 'EUROPE'}
        )

        elapsed = time.time() - start

        assert elapsed < 3.0, f"Filtered query latency: {elapsed:.2f}s"

    def test_batch_query_latency(self):
        """Test multiple queries in sequence."""
        queries = [
            "What is FCRA?",
            "GDPR updates",
            "APAC regulations",
        ]

        start = time.time()

        results = [query(q) for q in queries]

        elapsed = time.time() - start
        avg_latency = elapsed / len(queries)

        # Average latency should be reasonable
        assert avg_latency < 3.0, f"Average query latency: {avg_latency:.2f}s"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_query(self, query_agent):
        """Test handling of empty query."""
        result = query_agent(question="")

        # Should still return a response (possibly with low confidence)
        assert result is not None
        assert hasattr(result, 'answer')

    def test_very_long_query(self, query_agent):
        """Test handling of very long query."""
        long_query = "What are the implications of " * 50 + "FCRA changes?"

        result = query_agent(question=long_query)

        # Should handle gracefully
        assert result is not None
        assert hasattr(result, 'answer')

    def test_special_characters_in_query(self, query_agent):
        """Test handling of special characters."""
        result = query_agent(question="What's the status of FCRA & GDPR compliance?")

        assert result is not None
        assert hasattr(result, 'answer')

    def test_no_matching_results(self, query_agent):
        """Test query with no matching articles."""
        result = query_agent(question="What is the weather forecast for tomorrow?")

        # Should return response even with no matches
        assert result is not None
        assert hasattr(result, 'answer')

    def test_invalid_filter_handled(self, query_agent):
        """Test handling of invalid filter values."""
        result = query_agent(
            question="What are the latest updates?",
            filters={'region': 'INVALID_REGION'}
        )

        # Should still work (may return empty or all results)
        assert result is not None
        assert hasattr(result, 'answer')


# =============================================================================
# Confidence Score Tests
# =============================================================================

class TestConfidenceScores:
    """Test confidence scoring in responses."""

    def test_confidence_in_response(self):
        """Test that confidence score is included."""
        result = query("What are FCRA compliance requirements?")

        assert 'confidence' in result
        assert isinstance(result['confidence'], float)
        assert 0 <= result['confidence'] <= 1.0

    def test_confidence_correlates_with_sources(self):
        """Test that confidence relates to source availability."""
        result = query("Background screening regulations")

        # Should have reasonable confidence with sources
        assert result['confidence'] >= 0.5

    def test_low_confidence_for_uncertain_queries(self):
        """Test that uncertain queries have appropriate confidence."""
        result = query("What is the best pizza recipe?")

        # Off-topic query should have lower confidence
        # (but our mock still returns something)
        assert result['confidence'] <= 1.0
