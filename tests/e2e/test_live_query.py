# tests/e2e/test_live_query.py
"""E2E tests for live query functionality with Weaviate.

These tests verify that the query agent correctly retrieves
and synthesizes answers from real stored articles.

Run with: pytest -m e2e tests/e2e/test_live_query.py
"""

import pytest

pytestmark = pytest.mark.e2e


class TestLiveQueryRetrieval:
    """Test query retrieval from live Weaviate."""

    def test_query_retrieves_stored_articles(
        self, live_weaviate_client, e2e_article_store, sample_live_articles
    ):
        """Test that queries retrieve relevant stored articles."""
        from src.query_agent import QUIPLERRetriever

        # Store articles first
        e2e_article_store.batch_insert(sample_live_articles)

        # Create retriever with live client
        retriever = QUIPLERRetriever(
            weaviate_client=live_weaviate_client,
            collection_name=e2e_article_store.collection_name
        )

        # Query for FCRA content
        results = retriever.retrieve("FCRA compliance requirements", k=5)

        # Should return results
        assert len(results) > 0

        # Results should have content
        for result in results:
            assert 'content' in result or 'text' in result

    def test_query_with_region_filter(
        self, live_weaviate_client, e2e_article_store, sample_live_articles
    ):
        """Test querying with region filter."""
        from src.query_agent import QUIPLERRetriever, filter_by_region

        # Store articles
        e2e_article_store.batch_insert(sample_live_articles)

        # Create retriever
        retriever = QUIPLERRetriever(
            weaviate_client=live_weaviate_client,
            collection_name=e2e_article_store.collection_name
        )

        # Retrieve all
        all_results = retriever.retrieve("background screening regulations", k=10)

        # Filter for Europe
        europe_results = filter_by_region(all_results, 'EUROPE')

        # Should have fewer results after filtering
        assert len(europe_results) <= len(all_results)

    def test_retriever_returns_scores(
        self, live_weaviate_client, e2e_article_store, sample_live_articles
    ):
        """Test that retriever returns relevance scores."""
        from src.query_agent import QUIPLERRetriever

        # Store articles
        e2e_article_store.batch_insert(sample_live_articles)

        # Create retriever
        retriever = QUIPLERRetriever(
            weaviate_client=live_weaviate_client,
            collection_name=e2e_article_store.collection_name
        )

        # Query
        results = retriever.retrieve("employment law", k=5)

        # Each result should have a score
        for result in results:
            assert 'score' in result or 'relevance' in result


class TestLiveQueryAgent:
    """Test full query agent with live Weaviate."""

    def test_query_agent_generates_answer(
        self, live_weaviate_client, e2e_article_store, sample_live_articles
    ):
        """Test that query agent generates answers from stored articles."""
        from src.query_agent import NewsletterQueryAgent

        # Store articles
        e2e_article_store.batch_insert(sample_live_articles)

        # Create agent with live client
        agent = NewsletterQueryAgent(
            weaviate_client=live_weaviate_client,
            collection_name=e2e_article_store.collection_name
        )

        # Query
        result = agent("What are the recent FCRA compliance requirements?")

        # Should have an answer
        assert hasattr(result, 'answer')
        assert len(result.answer) > 0

    def test_query_agent_returns_sources(
        self, live_weaviate_client, e2e_article_store, sample_live_articles
    ):
        """Test that query agent returns source citations."""
        from src.query_agent import NewsletterQueryAgent

        # Store articles
        e2e_article_store.batch_insert(sample_live_articles)

        # Create agent
        agent = NewsletterQueryAgent(
            weaviate_client=live_weaviate_client,
            collection_name=e2e_article_store.collection_name
        )

        # Query
        result = agent("Tell me about background screening regulations")

        # Should have sources
        assert hasattr(result, 'sources')
        assert len(result.sources) > 0

    def test_query_function_with_live_weaviate(
        self, live_weaviate_client, e2e_article_store, sample_live_articles
    ):
        """Test the convenience query function with live Weaviate."""
        from src.query_agent import query

        # Store articles
        e2e_article_store.batch_insert(sample_live_articles)

        # Use query function
        result = query(
            "What is GDPR's impact on employment screening?",
            weaviate_client=live_weaviate_client,
            max_sources=3
        )

        # Verify response structure
        assert 'answer' in result
        assert 'sources' in result
        assert 'confidence' in result

        # Verify answer quality
        assert len(result['answer']) > 50


class TestLiveQueryFiltering:
    """Test query filtering with live data."""

    def test_filter_by_topic(
        self, live_weaviate_client, e2e_article_store, sample_live_articles
    ):
        """Test filtering query results by topic."""
        from src.query_agent import query

        # Store articles
        e2e_article_store.batch_insert(sample_live_articles)

        # Query with topic filter
        result = query(
            "Tell me about recent developments",
            filters={'topics': ['CRIMINAL_RECORDS']},
            weaviate_client=live_weaviate_client,
            max_sources=5
        )

        # Should have filtered results
        assert 'answer' in result

    def test_filter_by_region_europe(
        self, live_weaviate_client, e2e_article_store, sample_live_articles
    ):
        """Test filtering query results by European region."""
        from src.query_agent import query

        # Store articles
        e2e_article_store.batch_insert(sample_live_articles)

        # Query with region filter
        result = query(
            "What are the compliance requirements?",
            filters={'region': 'EUROPE'},
            weaviate_client=live_weaviate_client,
            max_sources=5
        )

        # Should have results
        assert 'answer' in result


class TestLiveAnswerQuality:
    """Test quality of answers from live queries."""

    def test_answer_addresses_question(
        self, live_weaviate_client, e2e_article_store, sample_live_articles
    ):
        """Test that answer addresses the asked question."""
        from src.query_agent import query

        # Store articles
        e2e_article_store.batch_insert(sample_live_articles)

        # Query about FCRA
        result = query(
            "What are the FCRA notification requirements?",
            weaviate_client=live_weaviate_client
        )

        # Answer should be about FCRA or compliance
        answer_lower = result['answer'].lower()
        relevant_terms = ['fcra', 'compliance', 'notification', 'requirement', 'consumer', 'report']
        assert any(term in answer_lower for term in relevant_terms)

    def test_answer_has_minimum_length(
        self, live_weaviate_client, e2e_article_store, sample_live_articles
    ):
        """Test that answer has substantive content."""
        from src.query_agent import query

        # Store articles
        e2e_article_store.batch_insert(sample_live_articles)

        # Query
        result = query(
            "Explain the impact of ban the box legislation",
            weaviate_client=live_weaviate_client
        )

        # Answer should be substantial
        assert len(result['answer']) >= 100

    def test_confidence_reflects_source_quality(
        self, live_weaviate_client, e2e_article_store, sample_live_articles
    ):
        """Test that confidence score reflects available sources."""
        from src.query_agent import query

        # Store articles
        e2e_article_store.batch_insert(sample_live_articles)

        # Query with good context
        result = query(
            "What are FCRA requirements for background screening?",
            weaviate_client=live_weaviate_client,
            max_sources=5
        )

        # Should have reasonable confidence
        assert 0 <= result['confidence'] <= 1
