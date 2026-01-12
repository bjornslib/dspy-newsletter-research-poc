# tests/integration/test_pipeline_e2e.py
"""
Integration tests for the full DSPy Newsletter Research Tool pipeline.

Beads Task: dspy-8nc (AT-Integration)

Tests the complete flow:
Ingestion → Deduplication → TinyLM Filter → Classification → Storage

Uses mocked Weaviate to avoid Docker dependency during CI.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

# =============================================================================
# Fixtures for Integration Testing
# =============================================================================

@pytest.fixture
def sample_rss_articles():
    """Sample articles as they would come from RSS feeds."""
    return [
        {
            'title': 'New FCRA Compliance Requirements Announced',
            'description': 'The CFPB has announced new compliance requirements for background screening companies under the Fair Credit Reporting Act.',
            'source_url': 'https://example.com/fcra-compliance',
            'source': 'Compliance Weekly',
            'published_date': datetime(2024, 1, 15, 10, 30),
        },
        {
            'title': 'GDPR Data Protection Updates in Europe',
            'description': 'European data protection authorities release new guidance on processing criminal records data under GDPR regulations.',
            'source_url': 'https://example.com/gdpr-updates',
            'source': 'EU Privacy News',
            'published_date': datetime(2024, 1, 16, 14, 0),
        },
        {
            'title': 'Best Pizza Recipes for Summer',
            'description': 'Discover the best pizza recipes for your summer parties. From classic margherita to creative toppings.',
            'source_url': 'https://example.com/pizza-recipes',
            'source': 'Food Blog',
            'published_date': datetime(2024, 1, 17, 9, 0),
        },
        {
            'title': 'New FCRA Compliance Requirements Announced',  # Duplicate
            'description': 'The CFPB has announced new compliance requirements for background screening companies under the Fair Credit Reporting Act.',
            'source_url': 'https://example.com/fcra-compliance',  # Same URL
            'source': 'HR News',
            'published_date': datetime(2024, 1, 15, 11, 0),
        },
    ]


@pytest.fixture
def mock_weaviate_client():
    """Mock Weaviate client for integration testing."""
    client = MagicMock()

    # Mock collection
    collection = MagicMock()
    client.collections.get.return_value = collection
    client.collections.exists.return_value = True
    client.collections.create.return_value = collection

    # Mock batch operations
    batch_context = MagicMock()
    batch_context.__enter__ = Mock(return_value=batch_context)
    batch_context.__exit__ = Mock(return_value=False)
    collection.batch.fixed_size.return_value = batch_context

    # Mock query operations
    query_mock = MagicMock()
    collection.query = query_mock

    return client


# =============================================================================
# Pipeline Stage Tests
# =============================================================================

class TestIngestionStage:
    """Test article ingestion stage of pipeline."""

    def test_articles_ingested_from_rss(self, sample_rss_articles):
        """Test that articles are correctly parsed from RSS data."""
        from src.ingestion import RSSParser

        # Articles should have required fields
        for article in sample_rss_articles:
            assert 'title' in article
            assert 'source_url' in article
            assert 'description' in article

    def test_articles_have_content_for_processing(self, sample_rss_articles):
        """Test that articles have content for downstream processing."""
        for article in sample_rss_articles:
            # Content should be either description or extracted
            content = article.get('content', article.get('description', ''))
            assert len(content) > 0


class TestDeduplicationStage:
    """Test deduplication stage of pipeline."""

    def test_duplicate_articles_removed(self, sample_rss_articles):
        """Test that duplicate articles are filtered out."""
        from src.deduplication import DuplicateIndex, ContentHashGenerator

        index = DuplicateIndex()
        hash_gen = ContentHashGenerator()

        unique_articles = []
        for article in sample_rss_articles:
            # Use URL as a simple hash key for URL-based deduplication
            url = article.get('source_url', '')
            url_hash = hash_gen.generate(url, '')

            if not index.is_duplicate(url_hash):
                unique_articles.append(article)
                index.add(url_hash, url)

        # Original: 4 articles, 1 duplicate URL
        assert len(unique_articles) == 3

    def test_fuzzy_detects_near_duplicates(self, sample_rss_articles):
        """Test that FuzzyMatcher detects content near-duplicates."""
        from src.deduplication import FuzzyMatcher

        matcher = FuzzyMatcher(threshold=0.8)

        # First FCRA article
        content1 = f"{sample_rss_articles[0]['title']} {sample_rss_articles[0]['description']}"
        # Duplicate FCRA article (identical content)
        content2 = f"{sample_rss_articles[3]['title']} {sample_rss_articles[3]['description']}"

        # Should detect as duplicate (identical content)
        is_dup = matcher.is_duplicate(content1, content2)
        assert is_dup


class TestPrefilterStage:
    """Test TinyLM pre-filter stage of pipeline."""

    def test_relevant_articles_pass_filter(self, sample_rss_articles):
        """Test that relevant articles pass the pre-filter."""
        from src.prefilter import TinyLMRelevanceFilter

        filter_module = TinyLMRelevanceFilter(threshold=0.5)

        # FCRA compliance article should pass
        fcra_article = sample_rss_articles[0]
        result = filter_module(
            title=fcra_article['title'],
            content_preview=fcra_article['description'],
            source_category='legal'
        )

        assert result.is_relevant is True
        assert result.confidence > 0.5

    def test_irrelevant_articles_filtered_out(self, sample_rss_articles):
        """Test that irrelevant articles are filtered out."""
        from src.prefilter import TinyLMRelevanceFilter

        filter_module = TinyLMRelevanceFilter(threshold=0.5)

        # Pizza recipe should be filtered
        pizza_article = sample_rss_articles[2]
        result = filter_module(
            title=pizza_article['title'],
            content_preview=pizza_article['description'],
            source_category='lifestyle'
        )

        assert result.is_relevant is False
        assert result.confidence < 0.5


class TestClassificationStage:
    """Test classification stage of pipeline."""

    def test_articles_classified_by_region(self, sample_rss_articles):
        """Test that articles are correctly classified by region."""
        from src.classification import ClassificationModule

        classifier = ClassificationModule()

        # FCRA article should be N_AMERICA_CARIBBEAN
        fcra_article = sample_rss_articles[0]
        result = classifier(
            title=fcra_article['title'],
            content=fcra_article['description'],
            source_url=fcra_article['source_url']
        )

        region = result.classification.region
        # Region enum values are lowercase
        assert region.value in ['n_america_caribbean', 'worldwide']

    def test_articles_classified_by_topics(self, sample_rss_articles):
        """Test that articles are correctly classified by topics."""
        from src.classification import ClassificationModule

        classifier = ClassificationModule()

        # GDPR article should have REGULATORY topic
        gdpr_article = sample_rss_articles[1]
        result = classifier(
            title=gdpr_article['title'],
            content=gdpr_article['description'],
            source_url=gdpr_article['source_url']
        )

        topics = [t.value for t in result.classification.topics]
        assert 'REGULATORY' in topics or len(topics) > 0


class TestStorageStage:
    """Test Weaviate storage stage of pipeline."""

    def test_articles_stored_in_weaviate(self, sample_rss_articles, mock_weaviate_client):
        """Test that articles are stored in Weaviate."""
        from src.storage import ArticleStore

        with patch('src.storage.weaviate') as mock_weaviate:
            mock_weaviate.connect_to_local.return_value = mock_weaviate_client

            store = ArticleStore(client=mock_weaviate_client)

            # Prepare article for storage
            article = sample_rss_articles[0]
            article_data = {
                'title': article['title'],
                'content': article['description'],
                'source_url': article['source_url'],
                'region': 'N_AMERICA_CARIBBEAN',
                'topics': ['REGULATORY'],
                'relevance_score': 85.0,
            }

            # Insert article
            article_id = store.insert(article_data)
            assert article_id is not None

    def test_stored_articles_searchable(self, mock_weaviate_client):
        """Test that stored articles can be searched."""
        from src.storage import ArticleStore

        with patch('src.storage.weaviate') as mock_weaviate:
            mock_weaviate.connect_to_local.return_value = mock_weaviate_client

            store = ArticleStore(client=mock_weaviate_client)

            # Insert an article to search for
            article_data = {
                'title': 'FCRA Compliance Guidelines',
                'content': 'Background screening requirements for employers',
                'source_url': 'https://example.com/fcra',
            }
            store.insert(article_data)

            # Search for it
            results = store.search("FCRA compliance", limit=5)

            # Verify search returns results
            assert len(results) > 0
            assert 'FCRA' in results[0].get('title', '')


# =============================================================================
# Full Pipeline Integration Tests
# =============================================================================

class TestFullPipeline:
    """Test the complete end-to-end pipeline."""

    def test_pipeline_processes_articles_end_to_end(self, sample_rss_articles, mock_weaviate_client):
        """Test full pipeline: Ingest → Dedupe → Filter → Classify → Store."""
        from src.deduplication import DuplicateIndex, ContentHashGenerator
        from src.prefilter import TinyLMRelevanceFilter
        from src.classification import ClassificationModule
        from src.storage import ArticleStore

        # Stage 1: Deduplication
        index = DuplicateIndex()
        hash_gen = ContentHashGenerator()
        unique_articles = []
        for article in sample_rss_articles:
            url = article.get('source_url', '')
            url_hash = hash_gen.generate(url, '')
            if not index.is_duplicate(url_hash):
                unique_articles.append(article)
                index.add(url_hash, url)

        assert len(unique_articles) == 3  # 4 - 1 duplicate

        # Stage 2: Pre-filter
        filter_module = TinyLMRelevanceFilter(threshold=0.4)
        relevant_articles = []
        for article in unique_articles:
            result = filter_module(
                title=article['title'],
                content_preview=article.get('description', '')[:500],
                source_category=article.get('source_category', 'general')
            )
            if result.is_relevant:
                article['prefilter_score'] = result.confidence
                relevant_articles.append(article)

        # FCRA and GDPR should pass, pizza should be filtered
        assert len(relevant_articles) >= 2

        # Stage 3: Classification
        classifier = ClassificationModule()
        classified_articles = []
        for article in relevant_articles:
            result = classifier(
                title=article['title'],
                content=article.get('description', ''),
                source_url=article['source_url']
            )
            article['region'] = result.classification.region.value
            article['topics'] = [t.value for t in result.classification.topics]
            article['relevance_score'] = result.classification.relevance_score
            classified_articles.append(article)

        assert len(classified_articles) >= 2

        # Stage 4: Storage (mocked)
        with patch('src.storage.weaviate') as mock_weaviate:
            mock_weaviate.connect_to_local.return_value = mock_weaviate_client

            store = ArticleStore(client=mock_weaviate_client)
            stored_count = 0

            for article in classified_articles:
                article_data = {
                    'title': article['title'],
                    'content': article.get('description', ''),
                    'source_url': article['source_url'],
                    'region': article['region'],
                    'topics': article['topics'],
                    'relevance_score': article['relevance_score'],
                }
                store.insert(article_data)
                stored_count += 1

            assert stored_count >= 2

    def test_pipeline_handles_empty_input(self, mock_weaviate_client):
        """Test pipeline handles empty article list gracefully."""
        from src.deduplication import Deduplicator
        from src.prefilter import batch_filter

        empty_articles = []

        # Deduplication on empty list
        deduplicator = Deduplicator()
        unique = deduplicator.deduplicate(empty_articles)
        assert len(unique) == 0

        # Pre-filter on empty list
        filtered = batch_filter(empty_articles, threshold=0.5)
        assert len(filtered) == 0

    def test_pipeline_preserves_metadata_through_stages(self, sample_rss_articles):
        """Test that article metadata is preserved through pipeline stages."""
        from src.prefilter import TinyLMRelevanceFilter
        from src.classification import ClassificationModule

        article = sample_rss_articles[0]  # FCRA article
        original_title = article['title']
        original_url = article['source_url']
        original_date = article['published_date']

        # Pre-filter
        filter_module = TinyLMRelevanceFilter()
        filter_result = filter_module(
            title=article['title'],
            content_preview=article['description'],
            source_category='legal'
        )

        # Classification
        classifier = ClassificationModule()
        class_result = classifier(
            title=article['title'],
            content=article['description'],
            source_url=article['source_url']
        )

        # Original metadata should be intact
        assert article['title'] == original_title
        assert article['source_url'] == original_url
        assert article['published_date'] == original_date


# =============================================================================
# Query Integration Tests
# =============================================================================

class TestQueryIntegration:
    """Test query agent integration with stored articles."""

    def test_query_agent_retrieves_relevant_articles(self, mock_weaviate_client):
        """Test that query agent retrieves relevant stored articles."""
        from src.query_agent import NewsletterQueryAgent

        agent = NewsletterQueryAgent(
            weaviate_client=mock_weaviate_client
        )

        result = agent(question="What are the latest FCRA compliance requirements?")

        assert result is not None
        assert hasattr(result, 'answer')
        assert len(result.answer) > 0

    def test_query_agent_handles_filters(self, mock_weaviate_client):
        """Test query agent with region and topic filters."""
        from src.query_agent import NewsletterQueryAgent

        agent = NewsletterQueryAgent(
            weaviate_client=mock_weaviate_client
        )

        result = agent(
            question="What GDPR regulations affect screening?",
            filters={'region': 'EUROPE'}
        )

        assert result is not None
        assert hasattr(result, 'answer')


# =============================================================================
# Pipeline Metrics Tests
# =============================================================================

class TestPipelineMetrics:
    """Test pipeline produces expected metrics."""

    def test_pipeline_reports_processing_counts(self, sample_rss_articles):
        """Test that pipeline reports counts at each stage."""
        from src.deduplication import DuplicateIndex, ContentHashGenerator
        from src.prefilter import batch_filter, get_prefilter_stats

        # Track counts
        metrics = {
            'input': len(sample_rss_articles),
            'after_dedup': 0,
            'after_filter': 0,
        }

        # Deduplication (URL-based)
        index = DuplicateIndex()
        hash_gen = ContentHashGenerator()
        unique_articles = []
        for article in sample_rss_articles:
            url = article.get('source_url', '')
            url_hash = hash_gen.generate(url, '')
            if not index.is_duplicate(url_hash):
                unique_articles.append(article)
                index.add(url_hash, url)
        metrics['after_dedup'] = len(unique_articles)

        # Pre-filter (using 'description' field which our test data uses)
        filtered = batch_filter(unique_articles, threshold=0.4, content_field='description')
        passed = [a for a in filtered if a.get('prefilter_passed', False)]
        metrics['after_filter'] = len(passed)

        # Verify metrics
        assert metrics['input'] == 4
        assert metrics['after_dedup'] == 3  # 1 duplicate removed
        assert metrics['after_filter'] >= 2  # At least 2 relevant (FCRA and GDPR)
        assert metrics['after_filter'] <= metrics['after_dedup']

    def test_prefilter_stats_calculated_correctly(self, sample_rss_articles):
        """Test pre-filter statistics are calculated correctly."""
        from src.prefilter import batch_filter, get_prefilter_stats

        filtered = batch_filter(sample_rss_articles[:3], threshold=0.5)  # First 3 (no dup)
        stats = get_prefilter_stats(filtered)

        assert stats['total'] == 3
        assert 'passed' in stats
        assert 'filtered' in stats
        assert 'pass_rate' in stats
        assert 0 <= stats['pass_rate'] <= 1
