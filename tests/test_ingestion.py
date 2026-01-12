# tests/test_ingestion.py
"""
RED PHASE: Ingestion tests for DSPy Newsletter Research Tool PoC.

Beads Task: dspy-7eh (RSS Parsing, Article Extraction)

These tests verify:
- RSS feed parsing and article extraction
- Content extraction from URLs
- Metadata handling (title, date, author)
- Batch ingestion functionality

Expected Behavior in RED Phase:
- All tests should FAIL because ingestion module is not yet implemented
- ImportError for 'from src.ingestion import ...'
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock


# =============================================================================
# RSS Parser Tests
# =============================================================================

class TestRSSParser:
    """RED: Test RSS feed parsing before implementation."""

    def test_rss_parser_exists(self):
        """Test RSSParser class can be imported."""
        from src.ingestion import RSSParser
        assert RSSParser is not None

    def test_rss_parser_accepts_url(self):
        """Test parser can be initialized with feed URL."""
        from src.ingestion import RSSParser
        parser = RSSParser(feed_url="https://example.com/feed.xml")
        assert parser.feed_url == "https://example.com/feed.xml"

    def test_rss_parser_parses_valid_feed(self, sample_rss_feed):
        """Test parser extracts articles from valid RSS feed."""
        from src.ingestion import RSSParser

        with patch('requests.get') as mock_get:
            mock_get.return_value.content = sample_rss_feed.encode()
            mock_get.return_value.status_code = 200

            parser = RSSParser(feed_url="https://example.com/feed.xml")
            articles = parser.parse()

            assert isinstance(articles, list)
            assert len(articles) == 2

    def test_rss_parser_extracts_title(self, sample_rss_feed):
        """Test parser extracts article titles."""
        from src.ingestion import RSSParser

        with patch('requests.get') as mock_get:
            mock_get.return_value.content = sample_rss_feed.encode()
            mock_get.return_value.status_code = 200

            parser = RSSParser(feed_url="https://example.com/feed.xml")
            articles = parser.parse()

            assert articles[0]['title'] == "New FCRA Guidelines Released"

    def test_rss_parser_extracts_link(self, sample_rss_feed):
        """Test parser extracts article links."""
        from src.ingestion import RSSParser

        with patch('requests.get') as mock_get:
            mock_get.return_value.content = sample_rss_feed.encode()
            mock_get.return_value.status_code = 200

            parser = RSSParser(feed_url="https://example.com/feed.xml")
            articles = parser.parse()

            assert articles[0]['source_url'] == "https://example.com/article1"

    def test_rss_parser_extracts_publication_date(self, sample_rss_feed):
        """Test parser extracts and parses publication dates."""
        from src.ingestion import RSSParser

        with patch('requests.get') as mock_get:
            mock_get.return_value.content = sample_rss_feed.encode()
            mock_get.return_value.status_code = 200

            parser = RSSParser(feed_url="https://example.com/feed.xml")
            articles = parser.parse()

            assert 'published_date' in articles[0]
            assert isinstance(articles[0]['published_date'], datetime)

    def test_rss_parser_handles_malformed_feed(self):
        """Test parser handles malformed RSS gracefully."""
        from src.ingestion import RSSParser, RSSParseError

        with patch('requests.get') as mock_get:
            mock_get.return_value.content = b"<invalid>xml"
            mock_get.return_value.status_code = 200

            parser = RSSParser(feed_url="https://example.com/bad.xml")

            with pytest.raises(RSSParseError):
                parser.parse()

    def test_rss_parser_handles_network_error(self):
        """Test parser handles network failures gracefully."""
        from src.ingestion import RSSParser, NetworkError

        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Connection timeout")

            parser = RSSParser(feed_url="https://example.com/feed.xml")

            with pytest.raises(NetworkError):
                parser.parse()


# =============================================================================
# Content Extractor Tests
# =============================================================================

class TestContentExtractor:
    """Test full article content extraction from URLs."""

    def test_content_extractor_exists(self):
        """Test ContentExtractor class can be imported."""
        from src.ingestion import ContentExtractor
        assert ContentExtractor is not None

    def test_extractor_fetches_article_content(self):
        """Test extractor retrieves full article text."""
        from src.ingestion import ContentExtractor

        mock_html = """
        <html>
            <article>
                <h1>FCRA Compliance Update</h1>
                <p>The Fair Credit Reporting Act continues to evolve...</p>
            </article>
        </html>
        """

        with patch('requests.get') as mock_get:
            mock_get.return_value.text = mock_html
            mock_get.return_value.status_code = 200

            extractor = ContentExtractor()
            content = extractor.extract("https://example.com/article")

            assert "Fair Credit Reporting Act" in content

    def test_extractor_removes_html_tags(self):
        """Test extractor returns clean text without HTML."""
        from src.ingestion import ContentExtractor

        mock_html = """
        <html>
            <p><strong>Bold</strong> and <em>italic</em> text</p>
        </html>
        """

        with patch('requests.get') as mock_get:
            mock_get.return_value.text = mock_html
            mock_get.return_value.status_code = 200

            extractor = ContentExtractor()
            content = extractor.extract("https://example.com/article")

            assert "<strong>" not in content
            assert "Bold" in content

    def test_extractor_handles_javascript_content(self):
        """Test extractor handles JS-rendered pages."""
        from src.ingestion import ContentExtractor

        extractor = ContentExtractor(use_javascript=True)
        assert extractor.use_javascript is True

    def test_extractor_respects_robots_txt(self):
        """Test extractor checks robots.txt."""
        from src.ingestion import ContentExtractor

        extractor = ContentExtractor(respect_robots=True)
        assert extractor.respect_robots is True

    def test_extractor_handles_paywall(self):
        """Test extractor detects paywalled content."""
        from src.ingestion import ContentExtractor, PaywallDetectedError

        mock_html = """
        <html>
            <div class="paywall">Subscribe to continue reading</div>
        </html>
        """

        with patch('requests.get') as mock_get:
            mock_get.return_value.text = mock_html
            mock_get.return_value.status_code = 200

            extractor = ContentExtractor()

            with pytest.raises(PaywallDetectedError):
                extractor.extract("https://example.com/premium")


# =============================================================================
# Feed Manager Tests
# =============================================================================

class TestFeedManager:
    """Test feed subscription and management."""

    def test_feed_manager_exists(self):
        """Test FeedManager class can be imported."""
        from src.ingestion import FeedManager
        assert FeedManager is not None

    def test_feed_manager_add_feed(self):
        """Test adding a new feed to the manager."""
        from src.ingestion import FeedManager

        manager = FeedManager()
        manager.add_feed(
            url="https://example.com/feed.xml",
            category="legal",
            priority=1
        )

        assert len(manager.feeds) == 1
        assert manager.feeds[0]['category'] == "legal"

    def test_feed_manager_list_feeds(self):
        """Test listing all subscribed feeds."""
        from src.ingestion import FeedManager

        manager = FeedManager()
        manager.add_feed(url="https://a.com/feed", category="legal")
        manager.add_feed(url="https://b.com/feed", category="industry")

        feeds = manager.list_feeds()
        assert len(feeds) == 2

    def test_feed_manager_remove_feed(self):
        """Test removing a feed from manager."""
        from src.ingestion import FeedManager

        manager = FeedManager()
        manager.add_feed(url="https://example.com/feed", category="legal")
        manager.remove_feed(url="https://example.com/feed")

        assert len(manager.feeds) == 0

    def test_feed_manager_get_by_category(self):
        """Test filtering feeds by category."""
        from src.ingestion import FeedManager

        manager = FeedManager()
        manager.add_feed(url="https://a.com/feed", category="legal")
        manager.add_feed(url="https://b.com/feed", category="legal")
        manager.add_feed(url="https://c.com/feed", category="industry")

        legal_feeds = manager.get_feeds_by_category("legal")
        assert len(legal_feeds) == 2


# =============================================================================
# Ingestion Pipeline Tests
# =============================================================================

class TestIngestionPipeline:
    """Test the complete ingestion pipeline."""

    def test_ingest_function_exists(self):
        """Test ingest convenience function exists."""
        from src.ingestion import ingest_from_feeds
        assert ingest_from_feeds is not None

    def test_ingest_returns_articles(self, sample_rss_feed):
        """Test ingest returns list of article dictionaries."""
        from src.ingestion import ingest_from_feeds

        with patch('requests.get') as mock_get:
            mock_get.return_value.content = sample_rss_feed.encode()
            mock_get.return_value.status_code = 200

            articles = ingest_from_feeds(
                feed_urls=["https://example.com/feed.xml"]
            )

            assert isinstance(articles, list)
            assert len(articles) > 0

    def test_ingest_includes_source_category(self, sample_rss_feed):
        """Test ingested articles include source category."""
        from src.ingestion import ingest_from_feeds

        with patch('requests.get') as mock_get:
            mock_get.return_value.content = sample_rss_feed.encode()
            mock_get.return_value.status_code = 200

            articles = ingest_from_feeds(
                feed_urls=[("https://example.com/feed.xml", "legal")]
            )

            assert articles[0]['source_category'] == "legal"

    def test_ingest_handles_multiple_feeds(self):
        """Test ingesting from multiple feeds."""
        from src.ingestion import ingest_from_feeds

        feed1 = """<?xml version="1.0"?>
        <rss><channel><item><title>Article 1</title></item></channel></rss>"""
        feed2 = """<?xml version="1.0"?>
        <rss><channel><item><title>Article 2</title></item></channel></rss>"""

        with patch('requests.get') as mock_get:
            mock_get.side_effect = [
                Mock(content=feed1.encode(), status_code=200),
                Mock(content=feed2.encode(), status_code=200)
            ]

            articles = ingest_from_feeds(
                feed_urls=["https://a.com/feed", "https://b.com/feed"]
            )

            assert len(articles) == 2

    def test_ingest_deduplicates_across_feeds(self):
        """Test duplicate articles are removed."""
        from src.ingestion import ingest_from_feeds

        # Same article in two feeds
        feed = """<?xml version="1.0"?>
        <rss><channel>
            <item>
                <title>Same Article</title>
                <link>https://example.com/article</link>
            </item>
        </channel></rss>"""

        with patch('requests.get') as mock_get:
            mock_get.return_value.content = feed.encode()
            mock_get.return_value.status_code = 200

            articles = ingest_from_feeds(
                feed_urls=["https://a.com/feed", "https://b.com/feed"],
                deduplicate=True
            )

            # Should only have 1 article, not 2
            unique_urls = set(a['source_url'] for a in articles)
            assert len(unique_urls) == len(articles)


# =============================================================================
# Metadata Extraction Tests
# =============================================================================

class TestMetadataExtraction:
    """Test metadata extraction from articles."""

    def test_extract_author_from_html(self):
        """Test extracting author from article HTML."""
        from src.ingestion import extract_metadata

        html = """
        <html>
            <meta name="author" content="John Doe">
            <article><p>Content here</p></article>
        </html>
        """

        metadata = extract_metadata(html)
        assert metadata.get('author') == "John Doe"

    def test_extract_publication_date_from_html(self):
        """Test extracting date from article HTML."""
        from src.ingestion import extract_metadata

        html = """
        <html>
            <meta property="article:published_time" content="2026-01-10T12:00:00Z">
            <article><p>Content here</p></article>
        </html>
        """

        metadata = extract_metadata(html)
        assert metadata.get('published_date') is not None

    def test_extract_og_image(self):
        """Test extracting Open Graph image."""
        from src.ingestion import extract_metadata

        html = """
        <html>
            <meta property="og:image" content="https://example.com/image.jpg">
        </html>
        """

        metadata = extract_metadata(html)
        assert metadata.get('image_url') == "https://example.com/image.jpg"
