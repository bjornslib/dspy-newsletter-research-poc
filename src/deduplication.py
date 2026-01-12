# src/deduplication.py
"""Deduplication module for DSPy Newsletter Research Tool.

This module provides:
- ContentHashGenerator: Generate content hashes for exact duplicate detection
- FuzzyMatcher: Text-based fuzzy matching for near-duplicates
- SemanticMatcher: Embedding-based semantic similarity
- Deduplicator: Pipeline for deduplication with multiple strategies
- DuplicateIndex: Persistent index for cross-session deduplication

Beads Task: dspy-50z
"""

import hashlib
import json
import math
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple, Set


# =============================================================================
# Content Hash Generator
# =============================================================================

class ContentHashGenerator:
    """Generate content hashes for exact duplicate detection.

    Normalizes text before hashing to handle minor formatting differences.
    """

    def __init__(self, algorithm: str = "sha256", hash_length: int = 32):
        """Initialize hash generator.

        Args:
            algorithm: Hash algorithm to use (sha256, md5, etc.).
            hash_length: Number of characters in returned hash.
        """
        self.algorithm = algorithm
        self.hash_length = hash_length

    def generate(self, title: str, content: str) -> str:
        """Generate hash from title and content.

        Args:
            title: Article title.
            content: Article content.

        Returns:
            Hex hash string of specified length.
        """
        # Normalize text
        normalized_title = self._normalize(title)
        normalized_content = self._normalize(content)

        # Combine and hash
        combined = f"{normalized_title}|{normalized_content}"
        hash_input = combined.encode('utf-8')

        if self.algorithm == "sha256":
            hash_obj = hashlib.sha256(hash_input)
        elif self.algorithm == "md5":
            hash_obj = hashlib.md5(hash_input)
        else:
            hash_obj = hashlib.new(self.algorithm, hash_input)

        return hash_obj.hexdigest()[:self.hash_length]

    def _normalize(self, text: str) -> str:
        """Normalize text for consistent hashing.

        - Converts to lowercase
        - Normalizes whitespace (multiple spaces/newlines -> single space)
        - Strips leading/trailing whitespace

        Args:
            text: Text to normalize.

        Returns:
            Normalized text string.
        """
        if not text:
            return ""

        # Lowercase
        text = text.lower()

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Strip
        text = text.strip()

        return text


# =============================================================================
# Fuzzy Matcher
# =============================================================================

class FuzzyMatcher:
    """Text-based fuzzy matching for near-duplicate detection.

    Uses a combination of SequenceMatcher and word-based Jaccard similarity
    for robust near-duplicate detection.
    """

    def __init__(self, threshold: float = 0.8):
        """Initialize fuzzy matcher.

        Args:
            threshold: Minimum similarity score to consider as duplicate (0-1).
        """
        self.threshold = threshold

    def similarity(self, text_a: str, text_b: str) -> float:
        """Calculate similarity between two texts.

        Uses maximum of:
        - SequenceMatcher ratio (good for similar sequences)
        - Word-based Jaccard similarity (good for word overlap)
        - Normalized word overlap (good for reordered words)

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        if not text_a or not text_b:
            return 0.0

        if text_a == text_b:
            return 1.0

        # Normalize for comparison
        norm_a = self._normalize(text_a)
        norm_b = self._normalize(text_b)

        if norm_a == norm_b:
            return 1.0

        # Method 1: SequenceMatcher for character-level similarity
        seq_similarity = SequenceMatcher(None, norm_a, norm_b).ratio()

        # Method 2: Word-based Jaccard similarity
        words_a = set(norm_a.split())
        words_b = set(norm_b.split())

        if not words_a or not words_b:
            return seq_similarity

        intersection = words_a & words_b
        union = words_a | words_b
        jaccard = len(intersection) / len(union) if union else 0.0

        # Method 3: Normalized word overlap (accounts for word order)
        # Higher weight on common meaningful words
        common_words = intersection
        total_words = (len(words_a) + len(words_b)) / 2
        word_overlap = len(common_words) / total_words if total_words > 0 else 0.0

        # Take the maximum of all methods for robust matching
        return max(seq_similarity, jaccard, word_overlap)

    def is_duplicate(self, text_a: str, text_b: str) -> bool:
        """Check if two texts are duplicates based on threshold.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            True if similarity exceeds threshold.
        """
        return self.similarity(text_a, text_b) >= self.threshold

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


# =============================================================================
# Semantic Matcher
# =============================================================================

class SemanticMatcher:
    """Embedding-based semantic similarity for near-duplicates.

    Uses vector embeddings to detect semantically similar content.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        threshold: float = 0.85
    ):
        """Initialize semantic matcher.

        Args:
            model: Embedding model to use.
            threshold: Minimum similarity for duplicate detection.
        """
        self.model = model
        self.threshold = threshold
        self._cache: Dict[str, List[float]] = {}

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        # Check cache first
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # For now, return a simple hash-based pseudo-embedding
        # In production, this would call an embedding API
        embedding = self._pseudo_embedding(text)
        self._cache[cache_key] = embedding

        return embedding

    def _pseudo_embedding(self, text: str, dim: int = 384) -> List[float]:
        """Generate pseudo-embedding from text hash.

        This is a placeholder - real implementation would use an embedding API.

        Args:
            text: Text to embed.
            dim: Embedding dimension.

        Returns:
            Pseudo-embedding vector.
        """
        # Create deterministic pseudo-random embedding from text
        hash_bytes = hashlib.sha256(text.encode()).digest()

        embedding = []
        for i in range(dim):
            # Use hash bytes cyclically to generate values
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] / 255.0) * 2 - 1  # Scale to [-1, 1]
            embedding.append(value)

        # Normalize
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding

    def similarity(self, text_a: str, text_b: str) -> float:
        """Calculate semantic similarity between two texts.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Cosine similarity between 0.0 and 1.0.
        """
        vec_a = self.get_embedding(text_a)
        vec_b = self.get_embedding(text_b)

        return self._cosine_similarity(vec_a, vec_b)

    def batch_similarity(self, texts: List[str]) -> List[List[float]]:
        """Compute similarity matrix for batch of texts.

        Args:
            texts: List of texts to compare.

        Returns:
            Similarity matrix (n x n).
        """
        embeddings = [self.get_embedding(t) for t in texts]
        n = len(texts)

        matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(1.0)
                else:
                    row.append(self._cosine_similarity(embeddings[i], embeddings[j]))
            matrix.append(row)

        return matrix

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec_a: First vector.
            vec_b: Second vector.

        Returns:
            Cosine similarity between -1 and 1.
        """
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(a * a for a in vec_a))
        mag_b = math.sqrt(sum(b * b for b in vec_b))

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot_product / (mag_a * mag_b)


# =============================================================================
# Duplicate Index
# =============================================================================

class DuplicateIndex:
    """Persistent index for tracking seen content hashes.

    Supports optional bloom filter for memory-efficient large-scale deduplication.
    """

    def __init__(self, use_bloom_filter: bool = False):
        """Initialize duplicate index.

        Args:
            use_bloom_filter: Use bloom filter for memory efficiency.
        """
        self.use_bloom_filter = use_bloom_filter
        self._hashes: Dict[str, str] = {}  # hash -> article_id
        self._bloom_bits: Set[int] = set()  # For bloom filter
        self._bloom_size = 10000

    def add(self, content_hash: str, article_id: str) -> None:
        """Add hash to index.

        Args:
            content_hash: Content hash.
            article_id: Associated article ID.
        """
        self._hashes[content_hash] = article_id

        if self.use_bloom_filter:
            self._add_to_bloom(content_hash)

    def contains(self, content_hash: str) -> bool:
        """Check if hash exists in index.

        Args:
            content_hash: Hash to check.

        Returns:
            True if hash exists.
        """
        if self.use_bloom_filter:
            if not self._bloom_contains(content_hash):
                return False  # Definitely not in index

        return content_hash in self._hashes

    def is_duplicate(self, content_hash: str) -> bool:
        """Check if content is a duplicate (alias for contains).

        Args:
            content_hash: Hash to check.

        Returns:
            True if duplicate.
        """
        return self.contains(content_hash)

    def get_article_id(self, content_hash: str) -> Optional[str]:
        """Get article ID for a hash.

        Args:
            content_hash: Hash to look up.

        Returns:
            Article ID or None.
        """
        return self._hashes.get(content_hash)

    def save(self, path: str) -> None:
        """Save index to file.

        Args:
            path: File path to save to.
        """
        data = {
            'hashes': self._hashes,
            'use_bloom_filter': self.use_bloom_filter,
        }

        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'DuplicateIndex':
        """Load index from file.

        Args:
            path: File path to load from.

        Returns:
            Loaded DuplicateIndex.
        """
        with open(path, 'r') as f:
            data = json.load(f)

        index = cls(use_bloom_filter=data.get('use_bloom_filter', False))
        index._hashes = data.get('hashes', {})

        # Rebuild bloom filter if needed
        if index.use_bloom_filter:
            for h in index._hashes:
                index._add_to_bloom(h)

        return index

    def _add_to_bloom(self, content_hash: str) -> None:
        """Add hash to bloom filter."""
        for i in range(3):  # 3 hash functions
            bit = hash(f"{content_hash}_{i}") % self._bloom_size
            self._bloom_bits.add(bit)

    def _bloom_contains(self, content_hash: str) -> bool:
        """Check if hash might be in bloom filter."""
        for i in range(3):
            bit = hash(f"{content_hash}_{i}") % self._bloom_size
            if bit not in self._bloom_bits:
                return False
        return True


# =============================================================================
# Deduplicator Pipeline
# =============================================================================

class Deduplicator:
    """Pipeline for deduplication with multiple strategies.

    Supports exact hash matching, fuzzy text matching, and semantic similarity.
    """

    _DEFAULT_FUZZY_THRESHOLD = 0.85

    def __init__(
        self,
        strategy: str = None,
        fuzzy_threshold: float = None,
        semantic_threshold: float = 0.9,
        keep: str = "first"
    ):
        """Initialize deduplicator.

        Args:
            strategy: Deduplication strategy ('hash', 'fuzzy', 'semantic', 'hash_then_fuzzy').
                      If None, 'hash' is used (unless fuzzy_threshold is explicitly set).
            fuzzy_threshold: Threshold for fuzzy matching. If set, enables fuzzy matching.
            semantic_threshold: Threshold for semantic matching.
            keep: Which duplicate to keep ('first', 'newest', 'oldest').
        """
        # Auto-detect strategy if not explicitly set
        if strategy is None:
            if fuzzy_threshold is not None:
                strategy = "hash_then_fuzzy"
            else:
                strategy = "hash"

        self.strategy = strategy
        self.fuzzy_threshold = fuzzy_threshold if fuzzy_threshold is not None else self._DEFAULT_FUZZY_THRESHOLD
        self.semantic_threshold = semantic_threshold
        self.keep = keep

        self._hash_gen = ContentHashGenerator()
        self._fuzzy = FuzzyMatcher(threshold=self.fuzzy_threshold)
        self._semantic = SemanticMatcher(threshold=semantic_threshold)

    def deduplicate(
        self,
        articles: List[Dict[str, Any]],
        return_stats: bool = False
    ) -> Any:
        """Remove duplicate articles.

        Args:
            articles: List of article dictionaries.
            return_stats: If True, return (articles, stats) tuple.

        Returns:
            Deduplicated articles, or (articles, stats) if return_stats=True.
        """
        if not articles:
            if return_stats:
                return [], {'input_count': 0, 'output_count': 0, 'duplicates_removed': 0}
            return []

        # Sort if keeping newest/oldest
        if self.keep == "newest" and any('published_date' in a for a in articles):
            articles = sorted(
                articles,
                key=lambda x: x.get('published_date') or datetime.min,
                reverse=True
            )
        elif self.keep == "oldest" and any('published_date' in a for a in articles):
            articles = sorted(
                articles,
                key=lambda x: x.get('published_date') or datetime.max
            )

        unique_articles = []
        seen_hashes: Set[str] = set()
        seen_articles: List[Tuple[str, str]] = []  # (title, content) pairs

        for article in articles:
            title = article.get('title', '')
            content = article.get('content', '')

            # Generate hash
            content_hash = self._hash_gen.generate(title, content)

            # Check exact duplicate
            if content_hash in seen_hashes:
                continue

            # Check fuzzy/semantic duplicate
            is_near_dup = False

            if self.strategy in ['fuzzy', 'hash_then_fuzzy']:
                for seen_title, seen_content in seen_articles:
                    # Compare title-to-title and content-to-content separately
                    # This catches near-duplicates with reworded titles/content
                    title_sim = self._fuzzy.similarity(title, seen_title)
                    content_sim = self._fuzzy.similarity(content, seen_content)

                    # Multiple detection strategies:
                    # 1. Both title and content have good similarity (both above 0.5)
                    # 2. Either component has high similarity (above threshold)
                    # 3. Average of both exceeds adjusted threshold

                    # If both are similar (>50%), use lower threshold for combined
                    if title_sim > 0.5 and content_sim > 0.5:
                        avg_sim = (title_sim + content_sim) / 2
                        # Adjusted threshold: if both parts are similar, require lower bar
                        adjusted_threshold = self.fuzzy_threshold * 0.85
                        if avg_sim >= adjusted_threshold:
                            is_near_dup = True
                            break

                    # Either component very similar
                    if title_sim >= self.fuzzy_threshold or content_sim >= self.fuzzy_threshold:
                        is_near_dup = True
                        break

            if is_near_dup:
                continue

            # Add to results
            seen_hashes.add(content_hash)
            seen_articles.append((title, content))
            unique_articles.append(article)

        stats = {
            'input_count': len(articles),
            'output_count': len(unique_articles),
            'duplicates_removed': len(articles) - len(unique_articles)
        }

        if return_stats:
            return unique_articles, stats

        return unique_articles

    def find_duplicates(
        self,
        articles: List[Dict[str, Any]]
    ) -> List[Tuple[int, int, float]]:
        """Find all duplicate pairs in article list.

        Args:
            articles: List of article dictionaries.

        Returns:
            List of (index_a, index_b, similarity) tuples.
        """
        duplicates = []
        n = len(articles)

        for i in range(n):
            for j in range(i + 1, n):
                title_i = articles[i].get('title', '')
                content_i = articles[i].get('content', '')
                title_j = articles[j].get('title', '')
                content_j = articles[j].get('content', '')

                # Check exact match
                hash_i = self._hash_gen.generate(title_i, content_i)
                hash_j = self._hash_gen.generate(title_j, content_j)

                if hash_i == hash_j:
                    duplicates.append((i, j, 1.0))
                    continue

                # Check fuzzy match
                text_i = f"{title_i} {content_i}"
                text_j = f"{title_j} {content_j}"
                similarity = self._fuzzy.similarity(text_i, text_j)

                if similarity >= self.fuzzy_threshold:
                    duplicates.append((i, j, similarity))

        return duplicates
