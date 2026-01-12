# tests/test_storage.py
"""
RED PHASE: Storage tests for DSPy Newsletter Research Tool PoC.

Beads Task: dspy-13p (Weaviate CRUD Operations)

These tests verify:
- Article storage in Weaviate
- Retrieval with vector search
- Update and delete operations
- Batch operations and performance

Expected Behavior in RED Phase:
- All tests should FAIL because storage module is not yet implemented
- ImportError for 'from src.storage import ...'
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock


# =============================================================================
# Article Store Tests
# =============================================================================

class TestArticleStore:
    """RED: Test article storage operations."""

    def test_article_store_exists(self):
        """Test ArticleStore class can be imported."""
        from src.storage import ArticleStore
        assert ArticleStore is not None

    def test_store_connects_to_weaviate(self, mock_weaviate_client):
        """Test store establishes Weaviate connection."""
        from src.storage import ArticleStore

        store = ArticleStore()
        assert store.client is not None

    def test_store_creates_collection_if_missing(self, mock_weaviate_client):
        """Test store creates NewsletterArticles collection."""
        from src.storage import ArticleStore

        mock_weaviate_client.collections.exists.return_value = False

        store = ArticleStore()
        store.ensure_collection()

        mock_weaviate_client.collections.create.assert_called()

    def test_store_insert_article(self, mock_weaviate_client, sample_article):
        """Test inserting a single article."""
        from src.storage import ArticleStore

        store = ArticleStore()
        article_id = store.insert(sample_article)

        assert article_id is not None
        assert isinstance(article_id, str)

    def test_store_insert_generates_embedding(self, mock_weaviate_client, sample_article):
        """Test insertion generates content embedding."""
        from src.storage import ArticleStore

        store = ArticleStore()
        store.insert(sample_article)

        # Should have called embedding generation
        # Implementation will use Cohere or OpenAI embeddings

    def test_store_get_by_id(self, mock_weaviate_client, sample_article):
        """Test retrieving article by ID."""
        from src.storage import ArticleStore

        store = ArticleStore()
        article_id = store.insert(sample_article)

        retrieved = store.get(article_id)
        assert retrieved['title'] == sample_article['title']

    def test_store_get_returns_none_for_missing(self, mock_weaviate_client):
        """Test get returns None for non-existent ID."""
        from src.storage import ArticleStore

        store = ArticleStore()
        retrieved = store.get("non-existent-id")

        assert retrieved is None


# =============================================================================
# Vector Search Tests
# =============================================================================

class TestVectorSearch:
    """Test vector similarity search."""

    def test_search_by_text_query(self, mock_weaviate_client):
        """Test searching by text query."""
        from src.storage import ArticleStore

        store = ArticleStore()
        results = store.search(
            query="FCRA compliance requirements",
            limit=10
        )

        assert isinstance(results, list)

    def test_search_returns_relevance_scores(self, mock_weaviate_client):
        """Test search results include relevance scores."""
        from src.storage import ArticleStore

        store = ArticleStore()
        results = store.search(query="background screening", limit=5)

        for result in results:
            assert 'score' in result or 'distance' in result

    def test_search_with_filters(self, mock_weaviate_client):
        """Test search with metadata filters."""
        from src.storage import ArticleStore

        store = ArticleStore()
        results = store.search(
            query="regulations",
            filters={"region": "EUROPE"},
            limit=10
        )

        # All results should match filter
        for result in results:
            assert result.get('region') == "EUROPE"

    def test_search_with_date_range(self, mock_weaviate_client):
        """Test search with date range filter."""
        from src.storage import ArticleStore
        from datetime import datetime, timedelta

        store = ArticleStore()
        results = store.search(
            query="compliance",
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            limit=10
        )

        assert isinstance(results, list)

    def test_hybrid_search(self, mock_weaviate_client):
        """Test hybrid BM25 + vector search."""
        from src.storage import ArticleStore

        store = ArticleStore()
        results = store.hybrid_search(
            query="FCRA background check",
            alpha=0.5,  # Balance between BM25 and vector
            limit=10
        )

        assert isinstance(results, list)


# =============================================================================
# Update and Delete Tests
# =============================================================================

class TestUpdateDelete:
    """Test update and delete operations."""

    def test_update_article(self, mock_weaviate_client, sample_article):
        """Test updating an existing article."""
        from src.storage import ArticleStore

        store = ArticleStore()
        article_id = store.insert(sample_article)

        updated = store.update(
            article_id,
            {"relevance_score": 95}
        )

        assert updated is True

    def test_update_regenerates_embedding_if_content_changed(self, mock_weaviate_client):
        """Test embedding is regenerated when content changes."""
        from src.storage import ArticleStore

        store = ArticleStore()
        article_id = store.insert({
            "title": "Original Title",
            "content": "Original content"
        })

        store.update(article_id, {"content": "Updated content"})
        # Should trigger re-embedding

    def test_delete_article(self, mock_weaviate_client, sample_article):
        """Test deleting an article."""
        from src.storage import ArticleStore

        store = ArticleStore()
        article_id = store.insert(sample_article)

        deleted = store.delete(article_id)
        assert deleted is True

        # Should not be retrievable
        assert store.get(article_id) is None

    def test_delete_nonexistent_returns_false(self, mock_weaviate_client):
        """Test deleting non-existent article returns False."""
        from src.storage import ArticleStore

        store = ArticleStore()
        deleted = store.delete("non-existent-id")

        assert deleted is False


# =============================================================================
# Batch Operations Tests
# =============================================================================

class TestBatchOperations:
    """Test batch insert/update/delete operations."""

    def test_batch_insert(self, mock_weaviate_client, sample_article, european_gdpr_article):
        """Test batch inserting multiple articles."""
        from src.storage import ArticleStore

        store = ArticleStore()
        articles = [sample_article, european_gdpr_article]

        ids = store.batch_insert(articles)

        assert len(ids) == 2
        assert all(isinstance(id, str) for id in ids)

    def test_batch_insert_with_errors(self, mock_weaviate_client):
        """Test batch insert handles partial failures."""
        from src.storage import ArticleStore

        store = ArticleStore()
        articles = [
            {"title": "Valid", "content": "Valid content"},
            {"title": "", "content": ""},  # Invalid - empty
            {"title": "Valid 2", "content": "More content"}
        ]

        ids, errors = store.batch_insert(articles, return_errors=True)

        assert len(ids) == 2
        assert len(errors) == 1

    def test_batch_delete(self, mock_weaviate_client):
        """Test batch deleting multiple articles."""
        from src.storage import ArticleStore

        store = ArticleStore()

        # Insert articles first
        ids = store.batch_insert([
            {"title": "A", "content": "A"},
            {"title": "B", "content": "B"}
        ])

        deleted_count = store.batch_delete(ids)
        assert deleted_count == 2

    def test_batch_update(self, mock_weaviate_client):
        """Test batch updating multiple articles."""
        from src.storage import ArticleStore

        store = ArticleStore()

        ids = store.batch_insert([
            {"title": "A", "content": "A", "relevance_score": 50},
            {"title": "B", "content": "B", "relevance_score": 50}
        ])

        updated_count = store.batch_update(
            ids,
            {"relevance_score": 75}
        )

        assert updated_count == 2


# =============================================================================
# Schema Management Tests
# =============================================================================

class TestSchemaManagement:
    """Test Weaviate schema management."""

    def test_get_collection_schema(self, mock_weaviate_client):
        """Test retrieving collection schema."""
        from src.storage import ArticleStore

        store = ArticleStore()
        schema = store.get_schema()

        assert 'properties' in schema
        assert 'vectorizer' in schema

    def test_schema_has_required_properties(self, mock_weaviate_client):
        """Test schema includes all required properties."""
        from src.storage import ArticleStore

        store = ArticleStore()
        schema = store.get_schema()

        required = ['title', 'content', 'source_url', 'published_date',
                    'region', 'topics', 'relevance_score']

        property_names = [p['name'] for p in schema['properties']]
        for prop in required:
            assert prop in property_names

    def test_migrate_schema(self, mock_weaviate_client):
        """Test schema migration adds new properties."""
        from src.storage import ArticleStore

        store = ArticleStore()

        # Add a new property
        store.add_property(
            name="new_field",
            data_type="text"
        )

        schema = store.get_schema()
        property_names = [p['name'] for p in schema['properties']]
        assert "new_field" in property_names


# =============================================================================
# Integration Tests (marked for optional execution)
# =============================================================================

@pytest.mark.integration
class TestStorageIntegration:
    """Integration tests requiring running Weaviate."""

    def test_full_crud_cycle(self, weaviate_test_client, sample_article):
        """Test complete Create-Read-Update-Delete cycle."""
        from src.storage import ArticleStore

        store = ArticleStore(client=weaviate_test_client)

        # Create
        article_id = store.insert(sample_article)
        assert article_id is not None

        # Read
        retrieved = store.get(article_id)
        assert retrieved['title'] == sample_article['title']

        # Update
        store.update(article_id, {"relevance_score": 99})
        updated = store.get(article_id)
        assert updated['relevance_score'] == 99

        # Delete
        store.delete(article_id)
        assert store.get(article_id) is None

    def test_vector_search_integration(self, weaviate_test_client):
        """Test vector search with real Weaviate."""
        from src.storage import ArticleStore

        store = ArticleStore(client=weaviate_test_client)

        # Insert test articles
        store.batch_insert([
            {"title": "FCRA Update", "content": "Fair Credit Reporting Act changes"},
            {"title": "GDPR News", "content": "European data protection updates"},
            {"title": "Recipe", "content": "How to make pasta"}
        ])

        # Search should rank FCRA article first
        results = store.search("credit reporting regulations", limit=3)

        assert results[0]['title'] == "FCRA Update"


# =============================================================================
# Connection Management Tests
# =============================================================================

class TestConnectionManagement:
    """Test Weaviate connection handling."""

    def test_connection_retry_on_failure(self):
        """Test connection retries on temporary failure."""
        from src.storage import ArticleStore

        with patch('weaviate.connect_to_local') as mock_connect:
            # Fail twice, succeed third time
            mock_connect.side_effect = [
                Exception("Connection refused"),
                Exception("Connection refused"),
                MagicMock()
            ]

            store = ArticleStore(max_retries=3)
            assert store.client is not None

    def test_connection_closes_properly(self, mock_weaviate_client):
        """Test connection is closed when store is destroyed."""
        from src.storage import ArticleStore

        store = ArticleStore()
        store.close()

        mock_weaviate_client.close.assert_called_once()

    def test_context_manager_usage(self, mock_weaviate_client):
        """Test store can be used as context manager."""
        from src.storage import ArticleStore

        with ArticleStore() as store:
            store.insert({"title": "Test", "content": "Test"})

        # Should have closed connection
        mock_weaviate_client.close.assert_called()
