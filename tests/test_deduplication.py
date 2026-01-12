# tests/test_deduplication.py
"""
RED PHASE: Deduplication tests for DSPy Newsletter Research Tool PoC.

Beads Task: dspy-50z (Hash + Fuzzy Matching Deduplication)

These tests verify:
- Content hash generation for exact duplicate detection
- Fuzzy/semantic matching for near-duplicate detection
- Deduplication pipeline integration
- Performance with large article sets

Expected Behavior in RED Phase:
- All tests should FAIL because deduplication module is not yet implemented
- ImportError for 'from src.deduplication import ...'
"""

import pytest
from unittest.mock import Mock, patch


# =============================================================================
# Content Hash Tests
# =============================================================================

class TestContentHash:
    """RED: Test content hashing for exact duplicate detection."""

    def test_hash_generator_exists(self):
        """Test ContentHashGenerator can be imported."""
        from src.deduplication import ContentHashGenerator
        assert ContentHashGenerator is not None

    def test_hash_generator_produces_hash(self):
        """Test generator produces a hash string."""
        from src.deduplication import ContentHashGenerator

        generator = ContentHashGenerator()
        hash_value = generator.generate(
            title="Test Article",
            content="This is test content"
        )

        assert hash_value is not None
        assert isinstance(hash_value, str)
        assert len(hash_value) > 0

    def test_identical_content_produces_same_hash(self):
        """Test same content always produces same hash."""
        from src.deduplication import ContentHashGenerator

        generator = ContentHashGenerator()

        hash1 = generator.generate(title="Test", content="Content")
        hash2 = generator.generate(title="Test", content="Content")

        assert hash1 == hash2

    def test_different_content_produces_different_hash(self):
        """Test different content produces different hashes."""
        from src.deduplication import ContentHashGenerator

        generator = ContentHashGenerator()

        hash1 = generator.generate(title="Article A", content="Content A")
        hash2 = generator.generate(title="Article B", content="Content B")

        assert hash1 != hash2

    def test_hash_ignores_whitespace_differences(self):
        """Test hash normalizes whitespace."""
        from src.deduplication import ContentHashGenerator

        generator = ContentHashGenerator()

        hash1 = generator.generate(title="Test", content="Hello World")
        hash2 = generator.generate(title="Test", content="Hello  World")
        hash3 = generator.generate(title="Test", content="Hello\nWorld")

        assert hash1 == hash2 == hash3

    def test_hash_is_case_insensitive_for_title(self):
        """Test hash normalizes title case."""
        from src.deduplication import ContentHashGenerator

        generator = ContentHashGenerator()

        hash1 = generator.generate(title="TEST ARTICLE", content="Content")
        hash2 = generator.generate(title="test article", content="Content")

        assert hash1 == hash2


# =============================================================================
# Fuzzy Matching Tests
# =============================================================================

class TestFuzzyMatcher:
    """Test fuzzy/semantic matching for near-duplicates."""

    def test_fuzzy_matcher_exists(self):
        """Test FuzzyMatcher can be imported."""
        from src.deduplication import FuzzyMatcher
        assert FuzzyMatcher is not None

    def test_matcher_computes_similarity_score(self):
        """Test matcher returns similarity score between 0-1."""
        from src.deduplication import FuzzyMatcher

        matcher = FuzzyMatcher()
        score = matcher.similarity(
            text_a="The FCRA requires consumer notification",
            text_b="FCRA mandates consumer notification requirements"
        )

        assert 0.0 <= score <= 1.0

    def test_identical_texts_have_score_one(self):
        """Test identical texts get similarity 1.0."""
        from src.deduplication import FuzzyMatcher

        matcher = FuzzyMatcher()
        score = matcher.similarity(
            text_a="Exact same text here",
            text_b="Exact same text here"
        )

        assert score == 1.0

    def test_completely_different_texts_have_low_score(self):
        """Test unrelated texts get low similarity."""
        from src.deduplication import FuzzyMatcher

        matcher = FuzzyMatcher()
        score = matcher.similarity(
            text_a="FCRA compliance requirements for employers",
            text_b="Pizza recipe with tomato sauce"
        )

        assert score < 0.3

    def test_near_duplicate_detection(self):
        """Test detecting near-duplicate articles."""
        from src.deduplication import FuzzyMatcher

        matcher = FuzzyMatcher(threshold=0.8)

        # Same article with minor rewording
        original = "The CFPB issued new FCRA guidance today"
        reworded = "Today the CFPB released new guidance on FCRA"

        score = matcher.similarity(original, reworded)
        assert score >= 0.7  # Should be high

    def test_matcher_uses_configurable_threshold(self):
        """Test threshold determines duplicate detection."""
        from src.deduplication import FuzzyMatcher

        matcher = FuzzyMatcher(threshold=0.9)
        assert matcher.threshold == 0.9


# =============================================================================
# Semantic Similarity Tests
# =============================================================================

class TestSemanticSimilarity:
    """Test embedding-based semantic similarity."""

    def test_semantic_matcher_exists(self):
        """Test SemanticMatcher can be imported."""
        from src.deduplication import SemanticMatcher
        assert SemanticMatcher is not None

    def test_semantic_matcher_uses_embeddings(self):
        """Test matcher uses embedding model."""
        from src.deduplication import SemanticMatcher

        matcher = SemanticMatcher(model="text-embedding-3-small")
        assert matcher.model == "text-embedding-3-small"

    def test_semantic_similarity_with_embeddings(self, mock_cohere_client):
        """Test semantic similarity using embeddings."""
        from src.deduplication import SemanticMatcher

        matcher = SemanticMatcher()

        # Mock embedding vectors
        with patch.object(matcher, 'get_embedding') as mock_embed:
            mock_embed.side_effect = [
                [0.1, 0.2, 0.3],  # First text
                [0.1, 0.2, 0.3]   # Second text (identical)
            ]

            score = matcher.similarity(
                text_a="Background check compliance",
                text_b="Employment screening regulations"
            )

            assert score == 1.0  # Identical vectors

    def test_semantic_batch_comparison(self):
        """Test batch semantic comparison for efficiency."""
        from src.deduplication import SemanticMatcher

        matcher = SemanticMatcher()
        articles = [
            "FCRA compliance update",
            "Background check regulations",
            "Pizza recipe for dinner"
        ]

        similarities = matcher.batch_similarity(articles)

        # Should return similarity matrix
        assert len(similarities) == 3
        assert len(similarities[0]) == 3


# =============================================================================
# Deduplication Pipeline Tests
# =============================================================================

class TestDeduplicationPipeline:
    """Test the complete deduplication pipeline."""

    def test_deduplicator_exists(self):
        """Test Deduplicator class can be imported."""
        from src.deduplication import Deduplicator
        assert Deduplicator is not None

    def test_deduplicator_accepts_strategy(self):
        """Test deduplicator accepts strategy configuration."""
        from src.deduplication import Deduplicator

        dedup = Deduplicator(strategy="hash_then_fuzzy")
        assert dedup.strategy == "hash_then_fuzzy"

    def test_deduplicate_removes_exact_duplicates(self):
        """Test exact duplicates are removed."""
        from src.deduplication import Deduplicator

        dedup = Deduplicator()

        articles = [
            {"title": "Article A", "content": "Same content"},
            {"title": "Article A", "content": "Same content"},  # Duplicate
            {"title": "Article B", "content": "Different content"}
        ]

        result = dedup.deduplicate(articles)
        assert len(result) == 2

    def test_deduplicate_removes_near_duplicates(self):
        """Test near-duplicates are detected and removed."""
        from src.deduplication import Deduplicator

        dedup = Deduplicator(fuzzy_threshold=0.85)

        articles = [
            {"title": "FCRA Update Released", "content": "The CFPB released new FCRA guidance"},
            {"title": "FCRA Guidance Released", "content": "New FCRA guidance was released by CFPB"},  # Near-dup
            {"title": "Unrelated Article", "content": "Something completely different"}
        ]

        result = dedup.deduplicate(articles)
        assert len(result) == 2

    def test_deduplicate_keeps_newest_article(self):
        """Test deduplication keeps most recent of duplicates."""
        from src.deduplication import Deduplicator
        from datetime import datetime, timedelta

        dedup = Deduplicator(keep="newest")

        now = datetime.now()
        articles = [
            {"title": "Same", "content": "Same", "published_date": now - timedelta(days=1)},
            {"title": "Same", "content": "Same", "published_date": now}  # Newer
        ]

        result = dedup.deduplicate(articles)
        assert len(result) == 1
        assert result[0]['published_date'] == now

    def test_deduplicate_returns_dedup_stats(self):
        """Test deduplication returns statistics."""
        from src.deduplication import Deduplicator

        dedup = Deduplicator()

        articles = [
            {"title": "A", "content": "A"},
            {"title": "A", "content": "A"},  # Duplicate
            {"title": "B", "content": "B"}
        ]

        result, stats = dedup.deduplicate(articles, return_stats=True)

        assert stats['input_count'] == 3
        assert stats['output_count'] == 2
        assert stats['duplicates_removed'] == 1


# =============================================================================
# Duplicate Index Tests
# =============================================================================

class TestDuplicateIndex:
    """Test persistent duplicate index for cross-session deduplication."""

    def test_duplicate_index_exists(self):
        """Test DuplicateIndex can be imported."""
        from src.deduplication import DuplicateIndex
        assert DuplicateIndex is not None

    def test_index_stores_hashes(self):
        """Test index stores content hashes."""
        from src.deduplication import DuplicateIndex

        index = DuplicateIndex()
        index.add("hash123", article_id="article-1")

        assert index.contains("hash123")

    def test_index_checks_for_duplicates(self):
        """Test index can check if article is duplicate."""
        from src.deduplication import DuplicateIndex

        index = DuplicateIndex()
        index.add("hash123", article_id="article-1")

        assert index.is_duplicate("hash123") is True
        assert index.is_duplicate("hash456") is False

    def test_index_persists_to_storage(self):
        """Test index can be saved and loaded."""
        from src.deduplication import DuplicateIndex
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.json")

            # Save
            index1 = DuplicateIndex()
            index1.add("hash123", article_id="article-1")
            index1.save(path)

            # Load
            index2 = DuplicateIndex.load(path)
            assert index2.contains("hash123")

    def test_index_supports_bloom_filter(self):
        """Test index uses bloom filter for memory efficiency."""
        from src.deduplication import DuplicateIndex

        index = DuplicateIndex(use_bloom_filter=True)
        assert index.use_bloom_filter is True


# =============================================================================
# Performance Tests
# =============================================================================

class TestDeduplicationPerformance:
    """Test deduplication performance with large datasets."""

    @pytest.mark.slow
    def test_handles_1000_articles(self):
        """Test deduplication with 1000 articles."""
        from src.deduplication import Deduplicator

        dedup = Deduplicator()

        # Generate 1000 articles with 10% duplicates
        articles = [
            {"title": f"Article {i % 900}", "content": f"Content {i % 900}"}
            for i in range(1000)
        ]

        result = dedup.deduplicate(articles)
        assert len(result) == 900  # 100 duplicates removed

    @pytest.mark.slow
    def test_hash_generation_is_fast(self):
        """Test hash generation performance."""
        from src.deduplication import ContentHashGenerator
        import time

        generator = ContentHashGenerator()
        content = "A" * 10000  # 10KB content

        start = time.time()
        for _ in range(1000):
            generator.generate(title="Test", content=content)
        elapsed = time.time() - start

        # Should process 1000 hashes in under 1 second
        assert elapsed < 1.0
