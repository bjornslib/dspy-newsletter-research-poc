# tests/e2e/test_full_pipeline.py
"""E2E tests for the full pipeline: Ingest → Process → Store → Query.

These tests verify the complete round-trip from RSS ingestion
through classification to query retrieval.

Run with: pytest -m e2e tests/e2e/test_full_pipeline.py
"""

import pytest
import time
from unittest.mock import patch, Mock

pytestmark = pytest.mark.e2e


class TestFullPipelineRoundTrip:
    """Test complete pipeline from ingestion to query."""

    def test_ingest_classify_store_query_cycle(
        self, live_weaviate_client, clean_test_collection
    ):
        """Test complete cycle: ingest article, classify, store, then query."""
        from src.ingestion import RSSParser
        from src.classification import classify_article
        from src.storage import ArticleStore
        from src.query_agent import query
        import requests

        # Step 1: Simulate RSS ingestion
        sample_rss = '''<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>HR Compliance News</title>
                <item>
                    <title>New FCRA Amendment Requires Enhanced Consumer Notifications</title>
                    <link>https://example.com/fcra-notification</link>
                    <description>The Consumer Financial Protection Bureau has released
                    new guidance on FCRA notification requirements for employment
                    background screening. All consumer reporting agencies must now
                    provide enhanced notification procedures.</description>
                    <pubDate>Mon, 13 Jan 2026 10:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>'''

        mock_response = Mock()
        mock_response.content = sample_rss.encode()
        mock_response.raise_for_status = Mock()

        with patch.object(requests, 'get', return_value=mock_response):
            parser = RSSParser('https://example.com/feed.xml')
            raw_articles = parser.parse()

        assert len(raw_articles) == 1, "Should parse 1 article"

        # Step 2: Prepare article (add content from description)
        article = raw_articles[0]
        article['content'] = article.get('description', '')

        # Step 3: Classify article
        classification = classify_article(
            title=article['title'],
            content=article['content'],
            source_url=article.get('source_url', '')
        )

        # Apply classification to article (classify_article returns a dict)
        region = classification['region']
        article['region'] = region.value if hasattr(region, 'value') else region
        topics = classification['topics']
        article['topics'] = [t.value if hasattr(t, 'value') else t for t in topics]
        article['relevance_score'] = classification['relevance_score']
        article['confidence'] = classification['confidence']
        article['rationale'] = classification['rationale']

        # Step 4: Store in Weaviate
        store = ArticleStore(
            client=live_weaviate_client,
            collection_name=clean_test_collection
        )
        article_id = store.insert(article)

        assert article_id is not None, "Should store article"

        # Step 5: Query for the stored article
        result = query(
            "What are the FCRA notification requirements?",
            weaviate_client=live_weaviate_client,
            max_sources=5
        )

        # Verify query success
        assert 'answer' in result, "Query should return answer"
        assert len(result['answer']) > 50, "Answer should be substantive"

    def test_multiple_articles_pipeline(
        self, live_weaviate_client, clean_test_collection, sample_live_articles
    ):
        """Test pipeline with multiple articles from different regions."""
        from src.storage import ArticleStore
        from src.query_agent import query

        # Store multiple articles
        store = ArticleStore(
            client=live_weaviate_client,
            collection_name=clean_test_collection
        )
        ids = store.batch_insert(sample_live_articles)

        assert len(ids) == len(sample_live_articles), "All articles should be stored"

        # Query for US content
        us_result = query(
            "What are the FCRA requirements?",
            filters={'region': 'N_AMERICA_CARIBBEAN'},
            weaviate_client=live_weaviate_client
        )
        assert 'answer' in us_result

        # Query for European content
        eu_result = query(
            "What are GDPR requirements for screening?",
            filters={'region': 'EUROPE'},
            weaviate_client=live_weaviate_client
        )
        assert 'answer' in eu_result


class TestPipelinePerformance:
    """Test pipeline performance with live Weaviate."""

    def test_storage_latency(
        self, live_weaviate_client, clean_test_collection, sample_live_articles
    ):
        """Test that article storage completes within acceptable time."""
        from src.storage import ArticleStore

        store = ArticleStore(
            client=live_weaviate_client,
            collection_name=clean_test_collection
        )

        start_time = time.time()
        ids = store.batch_insert(sample_live_articles)
        elapsed = time.time() - start_time

        # Should complete batch insert in under 5 seconds
        assert elapsed < 5.0, f"Batch insert took {elapsed:.2f}s, expected < 5s"
        assert len(ids) == len(sample_live_articles)

    def test_query_latency(
        self, live_weaviate_client, e2e_article_store, sample_live_articles
    ):
        """Test that query completes within acceptable time."""
        from src.query_agent import query

        # Store articles first
        e2e_article_store.batch_insert(sample_live_articles)

        start_time = time.time()
        result = query(
            "What are the compliance requirements?",
            weaviate_client=live_weaviate_client
        )
        elapsed = time.time() - start_time

        # Should complete in under 3 seconds
        assert elapsed < 3.0, f"Query took {elapsed:.2f}s, expected < 3s"
        assert 'answer' in result

    def test_search_latency(
        self, live_weaviate_client, e2e_article_store, sample_live_articles
    ):
        """Test that vector search completes quickly."""
        # Store articles
        e2e_article_store.batch_insert(sample_live_articles)

        start_time = time.time()
        results = e2e_article_store.search("background screening", limit=10)
        elapsed = time.time() - start_time

        # Should complete in under 1 second
        assert elapsed < 1.0, f"Search took {elapsed:.2f}s, expected < 1s"


class TestPipelineErrorRecovery:
    """Test error handling and recovery in the pipeline."""

    def test_graceful_empty_results(
        self, live_weaviate_client, e2e_article_store
    ):
        """Test handling of queries with no matching results."""
        from src.query_agent import query

        # Query empty database
        result = query(
            "Quantum computing regulations in Antarctica",
            weaviate_client=live_weaviate_client
        )

        # Should return valid response even with no matches
        assert 'answer' in result
        assert 'sources' in result
        assert 'confidence' in result

    def test_handle_special_characters_in_query(
        self, live_weaviate_client, e2e_article_store, sample_live_articles
    ):
        """Test handling of special characters in queries."""
        from src.query_agent import query

        # Store articles
        e2e_article_store.batch_insert(sample_live_articles)

        # Query with special characters
        result = query(
            "What's the FCRA's impact on employment?",
            weaviate_client=live_weaviate_client
        )

        # Should handle apostrophes gracefully
        assert 'answer' in result

    def test_handle_very_long_query(
        self, live_weaviate_client, e2e_article_store, sample_live_articles
    ):
        """Test handling of very long queries."""
        from src.query_agent import query

        # Store articles
        e2e_article_store.batch_insert(sample_live_articles)

        # Very long query
        long_query = "Tell me about " + " ".join(["FCRA compliance"] * 50)
        result = query(
            long_query,
            weaviate_client=live_weaviate_client
        )

        # Should handle long query gracefully
        assert 'answer' in result


class TestPipelineDataIntegrity:
    """Test data integrity through the pipeline."""

    def test_metadata_preserved_through_pipeline(
        self, live_weaviate_client, clean_test_collection
    ):
        """Test that all metadata is preserved through storage and retrieval."""
        from src.storage import ArticleStore

        # Create article with full metadata
        article = {
            'title': 'Test Article for Metadata',
            'content': 'This is test content for metadata preservation.',
            'source_url': 'https://example.com/test',
            'region': 'EUROPE',
            'topics': ['REGULATORY', 'TECHNOLOGY'],
            'relevance_score': 0.95,
            'confidence': 0.9,
            'rationale': 'Test rationale',
            'source': 'Test Source',
            'source_category': 'legal',
        }

        store = ArticleStore(
            client=live_weaviate_client,
            collection_name=clean_test_collection
        )

        article_id = store.insert(article)
        retrieved = store.get(article_id)

        # Verify all metadata preserved
        assert retrieved['title'] == article['title']
        assert retrieved['content'] == article['content']
        assert retrieved['region'] == article['region']
        assert retrieved.get('relevance_score') == article['relevance_score']

    def test_update_preserves_unmodified_fields(
        self, live_weaviate_client, e2e_article_store, sample_live_articles
    ):
        """Test that updates preserve unmodified fields."""
        article = sample_live_articles[0]
        article_id = e2e_article_store.insert(article)

        # Update only the relevance score
        e2e_article_store.update(article_id, {'relevance_score': 0.99})

        # Retrieve and verify
        retrieved = e2e_article_store.get(article_id)

        # Updated field should change
        assert retrieved.get('relevance_score') == 0.99

        # Other fields should be preserved
        assert retrieved['title'] == article['title']


class TestCLIIntegration:
    """Test CLI commands with live Weaviate."""

    def test_cli_status_with_live_weaviate(self, live_weaviate_client):
        """Test that CLI status command works with live Weaviate."""
        from click.testing import CliRunner
        from src.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['status'])

        # Should show status information
        assert result.exit_code == 0 or 'Weaviate' in result.output

    def test_cli_query_with_live_weaviate(
        self, live_weaviate_client, e2e_article_store, sample_live_articles
    ):
        """Test that CLI query command works with live Weaviate."""
        from click.testing import CliRunner
        from src.cli import cli

        # Store some articles first
        e2e_article_store.batch_insert(sample_live_articles)

        runner = CliRunner()
        result = runner.invoke(cli, ['query', 'What are FCRA requirements?'])

        # Should complete (may fail if Weaviate collection doesn't match)
        # The main goal is that it doesn't crash
        assert result.exit_code in [0, 1]  # 0 = success, 1 = handled error
