# tests/test_query_agent.py
"""
RED PHASE: Query Agent tests for DSPy Newsletter Research Tool PoC.

Beads Task: dspy-v72 (QUIPLER Retrieval, ReAct Synthesis)

These tests verify:
- QUIPLER retriever configuration
- ReAct agent for query synthesis
- Tool usage for filtering
- Multi-hop reasoning capabilities

Expected Behavior in RED Phase:
- All tests should FAIL because query_agent module is not yet implemented
- ImportError for 'from src.query_agent import ...'
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import inspect


# =============================================================================
# Query Response Signature Tests
# =============================================================================

class TestQueryResponseSignature:
    """RED: Test ReAct signature for query responses."""

    def test_signature_exists(self):
        """Test QueryResponseSignature can be imported."""
        from src.query_agent import QueryResponseSignature
        assert QueryResponseSignature is not None

    def test_signature_has_context_input(self):
        """Test signature accepts context from retrieval."""
        from src.query_agent import QueryResponseSignature
        assert 'context' in QueryResponseSignature.input_fields

    def test_signature_has_question_input(self):
        """Test signature accepts user question."""
        from src.query_agent import QueryResponseSignature
        assert 'question' in QueryResponseSignature.input_fields

    def test_signature_outputs_answer(self):
        """Test signature produces answer."""
        from src.query_agent import QueryResponseSignature
        assert 'answer' in QueryResponseSignature.output_fields

    def test_signature_outputs_sources(self):
        """Test signature includes sources."""
        from src.query_agent import QueryResponseSignature
        assert 'sources' in QueryResponseSignature.output_fields


# =============================================================================
# Query Tools Tests
# =============================================================================

class TestQueryTools:
    """Test ReAct tool functions."""

    def test_filter_by_date_exists(self):
        """Test filter_by_date tool can be imported."""
        from src.query_agent import filter_by_date
        assert filter_by_date is not None

    def test_filter_by_date_signature(self):
        """Test date filter tool has correct signature."""
        from src.query_agent import filter_by_date

        sig = inspect.signature(filter_by_date)
        params = list(sig.parameters.keys())
        assert 'start_date' in params
        assert 'end_date' in params

    def test_filter_by_date_filters_correctly(self):
        """Test date filter returns articles in range."""
        from src.query_agent import filter_by_date
        from datetime import datetime, timedelta

        now = datetime.now()
        articles = [
            {'id': '1', 'published_date': now - timedelta(days=5)},
            {'id': '2', 'published_date': now - timedelta(days=15)},
            {'id': '3', 'published_date': now - timedelta(days=25)}
        ]

        result = filter_by_date(
            articles,
            start_date=now - timedelta(days=20),
            end_date=now
        )

        assert len(result) == 2
        assert all(a['id'] in ['1', '2'] for a in result)

    def test_filter_by_region_exists(self):
        """Test filter_by_region tool can be imported."""
        from src.query_agent import filter_by_region
        assert filter_by_region is not None

    def test_filter_by_region_returns_filtered(self):
        """Test region filter reduces results."""
        from src.query_agent import filter_by_region

        articles = [
            {'id': '1', 'region': 'EUROPE'},
            {'id': '2', 'region': 'APAC'},
            {'id': '3', 'region': 'EUROPE'},
        ]

        result = filter_by_region(articles, region='EUROPE')
        assert len(result) == 2
        assert all(a['region'] == 'EUROPE' for a in result)

    def test_filter_by_topic_exists(self):
        """Test filter_by_topic tool can be imported."""
        from src.query_agent import filter_by_topic
        assert filter_by_topic is not None

    def test_filter_by_topic_returns_filtered(self):
        """Test topic filter reduces results."""
        from src.query_agent import filter_by_topic

        articles = [
            {'id': '1', 'topics': ['REGULATORY', 'COURT_CASES']},
            {'id': '2', 'topics': ['TECHNOLOGY']},
            {'id': '3', 'topics': ['REGULATORY']},
        ]

        result = filter_by_topic(articles, topic='REGULATORY')
        assert len(result) == 2


# =============================================================================
# QUIPLER Retriever Tests
# =============================================================================

class TestQUIPLERRetriever:
    """Test QUIPLER retrieval module."""

    def test_quipler_retriever_exists(self):
        """Test QUIPLERRetriever can be imported."""
        from src.query_agent import QUIPLERRetriever
        assert QUIPLERRetriever is not None

    def test_quipler_accepts_weaviate_client(self, mock_weaviate_client):
        """Test QUIPLER accepts Weaviate client."""
        from src.query_agent import QUIPLERRetriever

        retriever = QUIPLERRetriever(weaviate_client=mock_weaviate_client)
        assert retriever.client is not None

    def test_quipler_retrieves_documents(self, mock_weaviate_client):
        """Test QUIPLER retrieves relevant documents."""
        from src.query_agent import QUIPLERRetriever

        mock_weaviate_client.collections.get.return_value.query.hybrid.return_value = [
            Mock(properties={'title': 'Article 1', 'content': 'Content 1'}),
            Mock(properties={'title': 'Article 2', 'content': 'Content 2'})
        ]

        retriever = QUIPLERRetriever(weaviate_client=mock_weaviate_client)
        results = retriever.retrieve("FCRA compliance", k=5)

        assert isinstance(results, list)
        assert len(results) > 0

    def test_quipler_returns_passages(self, mock_weaviate_client):
        """Test QUIPLER returns passage objects."""
        from src.query_agent import QUIPLERRetriever

        retriever = QUIPLERRetriever(weaviate_client=mock_weaviate_client)
        results = retriever.retrieve("background screening", k=3)

        for passage in results:
            assert 'content' in passage or 'text' in passage
            assert 'score' in passage or 'relevance' in passage


# =============================================================================
# Newsletter Query Agent Tests
# =============================================================================

class TestNewsletterQueryAgent:
    """Integration tests for ReAct query agent."""

    def test_agent_exists(self):
        """Test NewsletterQueryAgent can be imported."""
        from src.query_agent import NewsletterQueryAgent
        assert NewsletterQueryAgent is not None

    def test_agent_is_dspy_module(self, mock_weaviate_client, mock_cohere_client):
        """Test agent is a proper DSPy module."""
        import dspy
        from src.query_agent import NewsletterQueryAgent

        agent = NewsletterQueryAgent(
            weaviate_client=mock_weaviate_client,
            cohere_client=mock_cohere_client
        )
        assert isinstance(agent, dspy.Module)

    def test_agent_uses_quipler_retriever(self, mock_weaviate_client, mock_cohere_client):
        """Test agent uses QUIPLER for retrieval."""
        from src.query_agent import NewsletterQueryAgent

        agent = NewsletterQueryAgent(
            weaviate_client=mock_weaviate_client,
            cohere_client=mock_cohere_client
        )

        assert hasattr(agent, 'retriever')
        assert 'QUIPLER' in str(type(agent.retriever).__name__)

    def test_agent_uses_react_for_synthesis(self, mock_weaviate_client, mock_cohere_client):
        """Test agent uses ReAct for answer synthesis."""
        from src.query_agent import NewsletterQueryAgent

        agent = NewsletterQueryAgent(
            weaviate_client=mock_weaviate_client,
            cohere_client=mock_cohere_client
        )

        assert hasattr(agent, 'synthesize')
        # ReAct module check
        assert 'ReAct' in str(type(agent.synthesize))

    def test_agent_query_returns_answer(self, mock_weaviate_client, mock_cohere_client, mock_dspy_lm):
        """Test query returns answer."""
        from src.query_agent import NewsletterQueryAgent

        agent = NewsletterQueryAgent(
            weaviate_client=mock_weaviate_client,
            cohere_client=mock_cohere_client
        )

        result = agent(question="What are recent APAC regulations?")

        assert hasattr(result, 'answer')
        assert len(result.answer) > 0

    def test_agent_query_returns_sources(self, mock_weaviate_client, mock_cohere_client, mock_dspy_lm):
        """Test query returns source citations."""
        from src.query_agent import NewsletterQueryAgent

        agent = NewsletterQueryAgent(
            weaviate_client=mock_weaviate_client,
            cohere_client=mock_cohere_client
        )

        result = agent(question="What are recent APAC regulations?")

        assert hasattr(result, 'sources')
        assert isinstance(result.sources, list)


# =============================================================================
# Multi-Hop Reasoning Tests
# =============================================================================

class TestMultiHopReasoning:
    """Test multi-hop reasoning capabilities."""

    def test_agent_handles_complex_query(self, mock_weaviate_client, mock_cohere_client, mock_dspy_lm):
        """Test multi-hop reasoning for complex queries."""
        from src.query_agent import NewsletterQueryAgent

        agent = NewsletterQueryAgent(
            weaviate_client=mock_weaviate_client,
            cohere_client=mock_cohere_client
        )

        result = agent(
            question="How has GDPR enforcement evolved in the UK post-Brexit and what does this mean for US screening firms?"
        )

        # Should provide substantive answer
        assert len(result.answer) > 100
        # Should cite sources
        assert len(result.sources) > 0

    def test_agent_uses_tools_for_filtering(self, mock_weaviate_client, mock_cohere_client, mock_dspy_lm):
        """Test agent uses filter tools when appropriate."""
        from src.query_agent import NewsletterQueryAgent

        agent = NewsletterQueryAgent(
            weaviate_client=mock_weaviate_client,
            cohere_client=mock_cohere_client
        )

        # Query that should trigger region filter
        result = agent(
            question="What are the latest background screening regulations in Europe?"
        )

        # Agent should have used filter_by_region
        assert result is not None

    def test_agent_handles_temporal_queries(self, mock_weaviate_client, mock_cohere_client, mock_dspy_lm):
        """Test agent handles time-based queries."""
        from src.query_agent import NewsletterQueryAgent

        agent = NewsletterQueryAgent(
            weaviate_client=mock_weaviate_client,
            cohere_client=mock_cohere_client
        )

        result = agent(
            question="What FCRA changes happened in the last 30 days?"
        )

        # Should understand temporal constraint
        assert result is not None


# =============================================================================
# Query Function Tests
# =============================================================================

class TestQueryFunction:
    """Test convenience query function."""

    def test_query_function_exists(self):
        """Test query convenience function exists."""
        from src.query_agent import query
        assert query is not None

    def test_query_returns_structured_response(self, mock_weaviate_client, mock_cohere_client, mock_dspy_lm):
        """Test query returns structured response."""
        from src.query_agent import query

        result = query("What are current FCRA requirements?")

        assert 'answer' in result
        assert 'sources' in result
        assert 'confidence' in result

    def test_query_accepts_filters(self, mock_weaviate_client, mock_cohere_client, mock_dspy_lm):
        """Test query accepts filter parameters."""
        from src.query_agent import query

        result = query(
            "background screening trends",
            filters={
                "region": "N_AMERICA_CARIBBEAN",
                "topics": ["REGULATORY"]
            }
        )

        assert result is not None

    def test_query_respects_max_sources(self, mock_weaviate_client, mock_cohere_client, mock_dspy_lm):
        """Test query respects max_sources limit."""
        from src.query_agent import query

        result = query(
            "compliance requirements",
            max_sources=3
        )

        assert len(result['sources']) <= 3


# =============================================================================
# Response Formatting Tests
# =============================================================================

class TestResponseFormatting:
    """Test response formatting utilities."""

    def test_format_answer_exists(self):
        """Test format_answer function exists."""
        from src.query_agent import format_answer
        assert format_answer is not None

    def test_format_answer_includes_citations(self):
        """Test formatted answer includes source citations."""
        from src.query_agent import format_answer

        answer = "FCRA requires consent"
        sources = [
            {"title": "FCRA Guide", "url": "https://example.com/1"},
            {"title": "Compliance Tips", "url": "https://example.com/2"}
        ]

        formatted = format_answer(answer, sources)

        assert "[1]" in formatted or "FCRA Guide" in formatted

    def test_format_sources_as_markdown(self):
        """Test sources formatted as markdown."""
        from src.query_agent import format_sources

        sources = [
            {"title": "Article 1", "url": "https://example.com/1"},
            {"title": "Article 2", "url": "https://example.com/2"}
        ]

        formatted = format_sources(sources, format="markdown")

        assert "https://example.com/1" in formatted
        assert "Article 1" in formatted
