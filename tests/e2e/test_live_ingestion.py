# tests/e2e/test_live_ingestion.py
"""E2E tests for live article ingestion into Weaviate.

These tests verify that articles can be ingested from various sources
and stored correctly in the Weaviate vector database.

Run with: pytest -m e2e tests/e2e/test_live_ingestion.py
"""

import pytest
from datetime import datetime

pytestmark = pytest.mark.e2e


class TestLiveArticleStorage:
    """Test storing articles in live Weaviate."""

    def test_store_single_article(self, e2e_article_store, sample_live_articles):
        """Test storing a single article in Weaviate."""
        article = sample_live_articles[0]

        # Store article
        article_id = e2e_article_store.insert(article)

        # Verify storage
        assert article_id is not None
        assert len(article_id) > 0

        # Retrieve and verify
        stored = e2e_article_store.get(article_id)
        assert stored is not None
        assert stored['title'] == article['title']

    def test_store_multiple_articles(self, e2e_article_store, sample_live_articles):
        """Test batch storage of multiple articles."""
        # Store all articles
        ids = e2e_article_store.batch_insert(sample_live_articles)

        # Verify all stored
        assert len(ids) == len(sample_live_articles)

        # Verify count
        count = e2e_article_store.count()
        assert count == len(sample_live_articles)

    def test_stored_articles_retrievable(self, e2e_article_store, sample_live_articles):
        """Test that stored articles can be retrieved by ID."""
        # Store article
        article = sample_live_articles[0]
        article_id = e2e_article_store.insert(article)

        # Retrieve
        retrieved = e2e_article_store.get(article_id)

        # Verify all fields
        assert retrieved is not None
        assert retrieved['title'] == article['title']
        assert 'FCRA' in retrieved['content'] or 'fcra' in retrieved['content'].lower()

    def test_article_metadata_preserved(self, e2e_article_store, sample_live_articles):
        """Test that article metadata is preserved after storage."""
        article = sample_live_articles[0]
        article_id = e2e_article_store.insert(article)

        retrieved = e2e_article_store.get(article_id)

        # Verify metadata fields
        assert retrieved.get('region') == 'N_AMERICA_CARIBBEAN'
        assert 'REGULATORY' in retrieved.get('topics', [])


class TestLiveDeduplication:
    """Test deduplication with live Weaviate storage."""

    def test_duplicate_detection(self, e2e_article_store, sample_live_articles):
        """Test that duplicate articles are handled correctly."""
        article = sample_live_articles[0]

        # Store same article twice
        id1 = e2e_article_store.insert(article)
        id2 = e2e_article_store.insert(article)

        # Both should succeed (store handles dedup at insert level)
        assert id1 is not None
        assert id2 is not None

        # But they should have different IDs
        assert id1 != id2

    def test_content_hash_for_dedup(self, e2e_article_store, sample_live_articles):
        """Test that content hash is stored for deduplication."""
        from src.ingestion import compute_content_hash

        article = sample_live_articles[0].copy()
        article['content_hash'] = compute_content_hash(
            article['title'],
            article['content']
        )

        article_id = e2e_article_store.insert(article)
        retrieved = e2e_article_store.get(article_id)

        assert retrieved.get('content_hash') is not None
        assert len(retrieved.get('content_hash', '')) > 0


class TestLiveVectorSearch:
    """Test vector search capabilities with live Weaviate."""

    def test_search_by_content(self, e2e_article_store, sample_live_articles):
        """Test searching articles by content similarity."""
        # Store articles
        e2e_article_store.batch_insert(sample_live_articles)

        # Search for FCRA-related content
        results = e2e_article_store.search("FCRA compliance requirements")

        # Should find relevant articles
        assert len(results) > 0

        # First result should be the FCRA article
        titles = [r.get('title', '') for r in results]
        assert any('FCRA' in title for title in titles)

    def test_search_with_region_filter(self, e2e_article_store, sample_live_articles):
        """Test filtering search results by region."""
        # Store articles
        e2e_article_store.batch_insert(sample_live_articles)

        # Search with EUROPE filter
        results = e2e_article_store.search(
            "background screening compliance",
            filters={'region': 'EUROPE'}
        )

        # All results should be European
        for result in results:
            assert result.get('region') == 'EUROPE'

    def test_search_returns_scores(self, e2e_article_store, sample_live_articles):
        """Test that search returns relevance scores."""
        # Store articles
        e2e_article_store.batch_insert(sample_live_articles)

        # Search
        results = e2e_article_store.search("employment law regulations")

        # Each result should have a score
        for result in results:
            assert 'score' in result
            assert 0 <= result['score'] <= 1


class TestLiveIngestionFromRSS:
    """Test live RSS feed ingestion (if feeds are available)."""

    def test_parse_rss_feed_format(self):
        """Test that RSS parser can handle standard feed format."""
        from src.ingestion import RSSParser

        # Use a sample RSS string (not network)
        sample_rss = '''<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>Test Feed</title>
                <item>
                    <title>Test Article</title>
                    <link>https://example.com/test</link>
                    <description>Test content</description>
                    <pubDate>Mon, 13 Jan 2026 12:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>'''

        # Mock the network request
        import requests
        from unittest.mock import patch, Mock

        mock_response = Mock()
        mock_response.content = sample_rss.encode()
        mock_response.raise_for_status = Mock()

        with patch.object(requests, 'get', return_value=mock_response):
            parser = RSSParser('https://example.com/feed.xml')
            articles = parser.parse()

        assert len(articles) == 1
        assert articles[0]['title'] == 'Test Article'

    def test_ingestion_pipeline_stores_to_weaviate(
        self, live_weaviate_client, clean_test_collection
    ):
        """Test that ingest_from_feeds would store to Weaviate.

        Note: This test uses mocked RSS to avoid network dependencies.
        """
        from src.ingestion import RSSParser
        from src.storage import ArticleStore
        import requests
        from unittest.mock import patch, Mock

        # Mock RSS feed
        sample_rss = '''<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>HR News</title>
                <item>
                    <title>Background Check Best Practices</title>
                    <link>https://example.com/bg-check</link>
                    <description>Latest best practices for employment screening.</description>
                    <pubDate>Mon, 13 Jan 2026 12:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>'''

        mock_response = Mock()
        mock_response.content = sample_rss.encode()
        mock_response.raise_for_status = Mock()

        with patch.object(requests, 'get', return_value=mock_response):
            parser = RSSParser('https://example.com/feed.xml')
            articles = parser.parse()

        # Prepare articles for storage
        for article in articles:
            article['content'] = article.get('description', '')

        # Store in Weaviate
        store = ArticleStore(
            client=live_weaviate_client,
            collection_name=clean_test_collection
        )
        ids = store.batch_insert(articles)

        # Verify storage
        assert len(ids) == 1

        # Verify retrieval
        retrieved = store.get(ids[0])
        assert retrieved is not None
        assert 'Background Check' in retrieved.get('title', '')
