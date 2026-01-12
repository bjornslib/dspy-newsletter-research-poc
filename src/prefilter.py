# src/prefilter.py
"""Pre-filter module for DSPy Newsletter Research Tool.

This module provides:
- RelevancePreFilter: DSPy signature for relevance classification
- TinyLMRelevanceFilter: Lightweight relevance filter module
- batch_filter: Batch processing function for articles
- DEFAULT_RELEVANCE_THRESHOLD: Default threshold for relevance filtering

Beads Task: dspy-2yy
"""

from typing import List, Dict, Any, Optional
import re

import dspy
from pydantic import Field


# =============================================================================
# Constants
# =============================================================================

DEFAULT_RELEVANCE_THRESHOLD = 0.6

# Keywords indicating high relevance to background screening industry
RELEVANCE_KEYWORDS = {
    'high': [
        'background check', 'background screening', 'fcra', 'fair credit reporting',
        'consumer reporting', 'employment screening', 'criminal record', 'criminal background',
        'pre-employment', 'adverse action', 'ban the box', 'expungement',
        'credential verification', 'education verification', 'employment verification',
        'cra', 'consumer reporting agency', 'cfpb', 'eeoc',
        'immigration', 'i-9', 'e-verify', 'work authorization',
        'drug test', 'drug screening', 'medical screening',
        'gdpr', 'data protection', 'privacy regulation', 'data privacy',
        'compliance', 'regulatory', 'legislation', 'amendment',
    ],
    'medium': [
        'hr', 'human resources', 'hiring', 'recruitment', 'talent acquisition',
        'onboarding', 'workforce', 'employee', 'employer',
        'verification', 'vetting', 'screening', 'due diligence',
        'court case', 'lawsuit', 'settlement', 'litigation',
        'merger', 'acquisition', 'm&a',
        'technology', 'ai', 'automation', 'machine learning',
        'conference', 'webinar', 'industry event',
    ],
    'low': [
        'job', 'work', 'career', 'professional', 'business',
    ]
}

# Keywords indicating irrelevant content
IRRELEVANCE_KEYWORDS = [
    'recipe', 'cooking', 'food', 'pizza', 'restaurant',
    'celebrity', 'entertainment', 'gossip', 'movie', 'tv show', 'music',
    'sports', 'game', 'match', 'tournament', 'player',
    'weather', 'forecast', 'temperature',
    'vacation', 'travel', 'hotel', 'tourism',
    'fashion', 'style', 'outfit', 'clothing',
    'celebrity', 'award show', 'red carpet', 'hollywood',
]

# Source categories with relevance weights
CATEGORY_WEIGHTS = {
    'legal': 1.0,
    'regulatory': 1.0,
    'government': 0.9,
    'compliance': 0.9,
    'industry': 0.8,
    'hr': 0.7,
    'technology': 0.6,
    'business': 0.5,
    'news': 0.4,
    'general': 0.3,
    'lifestyle': 0.1,
    'entertainment': 0.1,
}


# =============================================================================
# DSPy Signature
# =============================================================================

class RelevancePreFilter(dspy.Signature):
    """Determine if an article is relevant to background screening industry.

    The filter analyzes article title, content preview, and source category
    to assess relevance to the pre-employment screening industry.
    """
    # Input fields
    title: str = dspy.InputField(desc="Article title")
    content_preview: str = dspy.InputField(desc="First 500 characters of article content")
    source_category: str = dspy.InputField(desc="Category of the source (e.g., 'legal', 'industry')")

    # Output fields
    is_relevant: bool = dspy.OutputField(desc="Whether article is relevant to background screening")
    confidence: float = dspy.OutputField(desc="Confidence score between 0 and 1")
    reason: str = dspy.OutputField(desc="Explanation for the relevance decision")


# =============================================================================
# TinyLM Relevance Filter Module
# =============================================================================

class TinyLMRelevanceFilter(dspy.Module):
    """Lightweight relevance filter using keyword matching.

    This module provides fast, deterministic relevance scoring without
    requiring expensive LLM calls. It uses keyword matching and heuristics
    to filter out obviously irrelevant content.

    Attributes:
        threshold: Minimum confidence score to consider relevant.
    """

    def __init__(self, threshold: float = DEFAULT_RELEVANCE_THRESHOLD):
        """Initialize the filter.

        Args:
            threshold: Minimum confidence score for relevance (default: DEFAULT_RELEVANCE_THRESHOLD).
        """
        super().__init__()
        self.threshold = threshold
        self.predict = dspy.Predict(RelevancePreFilter)

    def forward(
        self,
        title: str,
        content_preview: str,
        source_category: str
    ) -> dspy.Prediction:
        """Score article relevance using keyword matching.

        Args:
            title: Article title.
            content_preview: First ~500 characters of content.
            source_category: Source category string.

        Returns:
            dspy.Prediction with is_relevant, confidence, and reason.
        """
        # Compute relevance score using keywords
        confidence, reason = self._compute_relevance(
            title, content_preview, source_category
        )

        # Determine if relevant based on threshold
        is_relevant = confidence >= self.threshold

        return dspy.Prediction(
            is_relevant=is_relevant,
            confidence=confidence,
            reason=reason
        )

    def _compute_relevance(
        self,
        title: str,
        content_preview: str,
        source_category: str
    ) -> tuple:
        """Compute relevance score using keyword matching.

        Args:
            title: Article title.
            content_preview: Content preview text.
            source_category: Source category.

        Returns:
            Tuple of (confidence_score, reason_string).
        """
        # Combine text for analysis
        combined_text = f"{title} {content_preview}".lower()
        title_lower = title.lower()

        # Check for irrelevance first
        irrelevance_hits = self._count_keyword_hits(combined_text, IRRELEVANCE_KEYWORDS)
        if irrelevance_hits >= 2:
            # Strong irrelevance signal
            return 0.15, f"Content appears unrelated to background screening (found {irrelevance_hits} off-topic indicators)"

        # Count relevance keyword hits
        high_hits = self._count_keyword_hits(combined_text, RELEVANCE_KEYWORDS['high'])
        medium_hits = self._count_keyword_hits(combined_text, RELEVANCE_KEYWORDS['medium'])
        low_hits = self._count_keyword_hits(combined_text, RELEVANCE_KEYWORDS['low'])

        # Title keywords are weighted more heavily
        title_high_hits = self._count_keyword_hits(title_lower, RELEVANCE_KEYWORDS['high'])
        title_medium_hits = self._count_keyword_hits(title_lower, RELEVANCE_KEYWORDS['medium'])

        # Calculate base score
        base_score = 0.0
        reasons = []

        # High relevance keywords
        if high_hits > 0:
            high_contribution = min(0.5, high_hits * 0.15)
            base_score += high_contribution
            reasons.append(f"high-relevance keywords ({high_hits})")

        # Medium relevance keywords
        if medium_hits > 0:
            medium_contribution = min(0.25, medium_hits * 0.08)
            base_score += medium_contribution
            reasons.append(f"medium-relevance keywords ({medium_hits})")

        # Low relevance keywords (small boost)
        if low_hits > 0:
            low_contribution = min(0.1, low_hits * 0.02)
            base_score += low_contribution

        # Title bonus - keywords in title are stronger signals
        if title_high_hits > 0:
            base_score += min(0.2, title_high_hits * 0.1)
            reasons.append(f"relevant terms in title ({title_high_hits})")
        if title_medium_hits > 0:
            base_score += min(0.1, title_medium_hits * 0.05)

        # Source category weight
        category_weight = CATEGORY_WEIGHTS.get(source_category.lower(), 0.3)
        base_score *= (0.5 + category_weight * 0.5)

        # Ensure category contributes to reason
        if category_weight >= 0.7:
            reasons.append(f"relevant source category ({source_category})")

        # Normalize to 0-1 range with ceiling
        confidence = min(0.95, base_score)

        # Build reason string
        if reasons:
            reason = f"Relevance indicators: {', '.join(reasons)}"
        elif confidence < 0.3:
            reason = "No significant relevance indicators found"
        else:
            reason = "Moderate relevance based on general context"

        # Final adjustments
        # Very low scores get a floor
        if confidence < 0.1 and irrelevance_hits == 0:
            confidence = 0.1

        return confidence, reason

    def _count_keyword_hits(self, text: str, keywords: List[str]) -> int:
        """Count how many keywords appear in text.

        Args:
            text: Text to search (should be lowercased).
            keywords: List of keywords to find.

        Returns:
            Number of unique keywords found.
        """
        hits = 0
        for keyword in keywords:
            if keyword in text:
                hits += 1
        return hits


# =============================================================================
# Batch Processing
# =============================================================================

def batch_filter(
    articles: List[Dict[str, Any]],
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    content_field: str = 'content'
) -> List[Dict[str, Any]]:
    """Filter a batch of articles for relevance.

    Args:
        articles: List of article dictionaries.
        threshold: Relevance threshold for filtering.
        content_field: Field name containing article content.

    Returns:
        List of articles enriched with prefilter_score and prefilter_passed fields.
    """
    filter_module = TinyLMRelevanceFilter(threshold=threshold)
    results = []

    for article in articles:
        # Extract fields
        title = article.get('title', '')
        content = article.get(content_field, '')
        source_category = article.get('source_category', 'general')

        # Create content preview
        content_preview = content[:500] if len(content) > 500 else content

        # Run filter
        prediction = filter_module(
            title=title,
            content_preview=content_preview,
            source_category=source_category
        )

        # Create enriched article copy
        enriched = article.copy()
        enriched['prefilter_score'] = prediction.confidence
        enriched['prefilter_passed'] = prediction.is_relevant
        enriched['prefilter_reason'] = prediction.reason
        enriched['is_relevant'] = prediction.is_relevant

        results.append(enriched)

    return results


# =============================================================================
# Utility Functions
# =============================================================================

def filter_relevant_only(
    articles: List[Dict[str, Any]],
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD
) -> List[Dict[str, Any]]:
    """Filter articles and return only relevant ones.

    Args:
        articles: List of article dictionaries.
        threshold: Relevance threshold.

    Returns:
        List containing only relevant articles.
    """
    filtered = batch_filter(articles, threshold=threshold)
    return [a for a in filtered if a['prefilter_passed']]


def get_prefilter_stats(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get statistics from pre-filtered articles.

    Args:
        articles: List of articles after batch_filter.

    Returns:
        Dictionary with filtering statistics.
    """
    if not articles:
        return {
            'total': 0,
            'passed': 0,
            'filtered': 0,
            'pass_rate': 0.0,
            'avg_score': 0.0,
        }

    passed = sum(1 for a in articles if a.get('prefilter_passed', False))
    scores = [a.get('prefilter_score', 0) for a in articles]

    return {
        'total': len(articles),
        'passed': passed,
        'filtered': len(articles) - passed,
        'pass_rate': passed / len(articles) if articles else 0.0,
        'avg_score': sum(scores) / len(scores) if scores else 0.0,
    }
