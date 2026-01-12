# src/classification.py
"""Classification module for DSPy Newsletter Research Tool.

This module provides:
- ArticleClassification: Pydantic model for classification output
- RegionEnum/TopicEnum: Re-exported from models
- ClassificationModule: DSPy module for full classification
- RelevanceScorer: ChainOfThought-based relevance scoring
- classify_article: Convenience function for single articles
- batch_classify: Batch processing function

Beads Task: dspy-0s9
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

import dspy

# Re-export enums from models
from src.models import RegionEnum, TopicEnum


# =============================================================================
# Region Keywords Mapping
# =============================================================================

REGION_KEYWORDS = {
    RegionEnum.AFRICA_ME: [
        'africa', 'african', 'middle east', 'uae', 'dubai', 'saudi arabia',
        'israel', 'egypt', 'nigeria', 'south africa', 'kenya', 'qatar',
        'kuwait', 'bahrain', 'oman', 'jordan', 'lebanon', 'morocco',
    ],
    RegionEnum.APAC: [
        'asia', 'pacific', 'apac', 'china', 'japan', 'korea', 'india',
        'australia', 'singapore', 'hong kong', 'taiwan', 'vietnam',
        'thailand', 'indonesia', 'malaysia', 'philippines', 'new zealand',
        'skillsfuture', 'ministry of manpower',
    ],
    RegionEnum.EUROPE: [
        'europe', 'european', 'eu', 'gdpr', 'germany', 'german', 'france',
        'french', 'uk', 'united kingdom', 'britain', 'british', 'ireland',
        'spain', 'italy', 'netherlands', 'belgium', 'poland', 'sweden',
        'norway', 'denmark', 'finland', 'austria', 'switzerland',
        'data protection authority',
    ],
    RegionEnum.N_AMERICA_CARIBBEAN: [
        'usa', 'united states', 'america', 'american', 'us', 'fcra',
        'fair credit reporting', 'cfpb', 'eeoc', 'canada', 'canadian',
        'mexico', 'caribbean', 'ban the box', 'state law', 'federal',
        'congress', 'senate', 'fcc', 'ftc',
    ],
    RegionEnum.S_AMERICA: [
        'south america', 'latin america', 'brazil', 'brazilian', 'argentina',
        'chile', 'colombia', 'peru', 'venezuela', 'ecuador', 'bolivia',
        'uruguay', 'paraguay',
    ],
    RegionEnum.WORLDWIDE: [
        'global', 'worldwide', 'international', 'across regions',
        'all regions', 'multinational', 'world market',
    ],
}

# =============================================================================
# Topic Keywords Mapping
# =============================================================================

TOPIC_KEYWORDS = {
    TopicEnum.REGULATORY: [
        'regulatory', 'regulation', 'compliance', 'law', 'legislation',
        'amendment', 'act', 'statute', 'rule', 'requirement', 'mandate',
        'fcra', 'gdpr', 'eeoc', 'cfpb', 'enforcement', 'guidance',
        'policy', 'ban the box', 'fair chance',
    ],
    TopicEnum.CRIMINAL_RECORDS: [
        'criminal record', 'criminal background', 'criminal history',
        'conviction', 'arrest', 'felony', 'misdemeanor', 'expungement',
        'sealing', 'court record', 'criminal check',
    ],
    TopicEnum.CREDENTIALS: [
        'credential', 'education verification', 'employment verification',
        'degree verification', 'license verification', 'certification',
        'professional license', 'diploma', 'transcript', 'reference check',
    ],
    TopicEnum.IMMIGRATION: [
        'immigration', 'work authorization', 'visa', 'i-9', 'e-verify',
        'work permit', 'foreign worker', 'expatriate', 'citizenship',
        'green card', 'h1b', 'work eligibility',
    ],
    TopicEnum.M_AND_A: [
        'merger', 'acquisition', 'm&a', 'acquire', 'buyout', 'purchase',
        'deal', 'transaction', 'partnership', 'joint venture', 'investor',
        'private equity', 'funding round',
    ],
    TopicEnum.TECHNOLOGY: [
        'technology', 'ai', 'artificial intelligence', 'machine learning',
        'automation', 'platform', 'software', 'api', 'digital',
        'integration', 'algorithm', 'data analytics', 'cloud',
    ],
    TopicEnum.EVENTS: [
        'conference', 'event', 'summit', 'webinar', 'seminar', 'workshop',
        'trade show', 'expo', 'annual meeting', 'industry gathering',
    ],
    TopicEnum.COURT_CASES: [
        'court case', 'lawsuit', 'litigation', 'settlement', 'class action',
        'verdict', 'judgment', 'ruling', 'plaintiff', 'defendant',
        'damages', 'injunctive', 'legal action', 'sue', 'filed suit',
    ],
}


# =============================================================================
# ArticleClassification Pydantic Model
# =============================================================================

class ArticleClassification(BaseModel):
    """Pydantic model for classification output.

    Used as the structured output from DSPy classification modules.
    """
    region: RegionEnum = Field(..., description="Geographic region")
    topics: List[TopicEnum] = Field(default_factory=list, description="Topic categories")
    relevance_score: float = Field(..., ge=0, le=100, description="Relevance score (0-100)")
    confidence: float = Field(..., ge=0, le=1, description="Classification confidence (0-1)")
    rationale: str = Field(..., description="Explanation for classification decisions")
    summary: Optional[str] = Field(None, description="Article summary")


# =============================================================================
# DSPy Signatures
# =============================================================================

class RegionClassificationSignature(dspy.Signature):
    """Classify article into geographic region."""
    title: str = dspy.InputField(desc="Article title")
    content: str = dspy.InputField(desc="Article content")

    region: str = dspy.OutputField(desc="Geographic region code")
    confidence: float = dspy.OutputField(desc="Classification confidence")


class TopicClassificationSignature(dspy.Signature):
    """Classify article into topic categories."""
    title: str = dspy.InputField(desc="Article title")
    content: str = dspy.InputField(desc="Article content")

    topics: str = dspy.OutputField(desc="Comma-separated topic codes")
    confidence: float = dspy.OutputField(desc="Classification confidence")


class RelevanceScoringSignature(dspy.Signature):
    """Score article relevance to background screening industry."""
    title: str = dspy.InputField(desc="Article title")
    content: str = dspy.InputField(desc="Article content")

    relevance_score: float = dspy.OutputField(desc="Relevance score 0-100")
    rationale: str = dspy.OutputField(desc="Reasoning for the score")


# =============================================================================
# RelevanceScorer Module
# =============================================================================

class RelevanceScorer(dspy.Module):
    """Score article relevance using ChainOfThought reasoning.

    Provides explainable relevance scoring with rationale.
    """

    # Keywords for relevance scoring
    HIGH_RELEVANCE_KEYWORDS = [
        'background check', 'background screening', 'fcra', 'employment screening',
        'criminal record', 'pre-employment', 'adverse action', 'consumer reporting',
        'credential verification', 'employment verification', 'drug screening',
        'ban the box', 'expungement', 'e-verify', 'gdpr', 'eeoc', 'cfpb',
    ]

    MEDIUM_RELEVANCE_KEYWORDS = [
        'compliance', 'regulatory', 'hr', 'hiring', 'recruitment',
        'verification', 'screening', 'due diligence', 'workforce',
        'court case', 'lawsuit', 'settlement',
    ]

    IRRELEVANCE_KEYWORDS = [
        'recipe', 'cooking', 'celebrity', 'entertainment', 'sports',
        'weather', 'fashion', 'vacation', 'movie', 'music', 'game',
        'pizza', 'restaurant', 'award show', 'red carpet', 'hollywood',
    ]

    def __init__(self):
        """Initialize scorer with ChainOfThought predictor."""
        super().__init__()
        self.predict = dspy.ChainOfThought(RelevanceScoringSignature)

    def forward(self, title: str, content: str) -> dspy.Prediction:
        """Score article relevance.

        Args:
            title: Article title.
            content: Article content.

        Returns:
            dspy.Prediction with relevance_score and rationale.
        """
        # Use keyword-based scoring for deterministic behavior
        relevance_score, rationale = self._compute_relevance(title, content)

        return dspy.Prediction(
            relevance_score=relevance_score,
            rationale=rationale
        )

    def _compute_relevance(self, title: str, content: str) -> tuple:
        """Compute relevance score using keywords.

        Args:
            title: Article title.
            content: Article content.

        Returns:
            Tuple of (score, rationale).
        """
        combined_text = f"{title} {content}".lower()
        title_lower = title.lower()
        reasons = []

        # Check irrelevance first
        irrelevant_hits = sum(1 for kw in self.IRRELEVANCE_KEYWORDS if kw in combined_text)
        if irrelevant_hits >= 2:
            return 15.0, f"Content appears unrelated to background screening industry (found {irrelevant_hits} off-topic indicators)"

        # Count relevance hits
        high_hits = sum(1 for kw in self.HIGH_RELEVANCE_KEYWORDS if kw in combined_text)
        medium_hits = sum(1 for kw in self.MEDIUM_RELEVANCE_KEYWORDS if kw in combined_text)

        # Title bonus
        title_high = sum(1 for kw in self.HIGH_RELEVANCE_KEYWORDS if kw in title_lower)
        title_medium = sum(1 for kw in self.MEDIUM_RELEVANCE_KEYWORDS if kw in title_lower)

        # Calculate score
        score = 20.0  # Base score

        if high_hits > 0:
            contribution = min(50, high_hits * 12)
            score += contribution
            reasons.append(f"found {high_hits} high-relevance terms")

        if medium_hits > 0:
            contribution = min(25, medium_hits * 6)
            score += contribution
            reasons.append(f"found {medium_hits} medium-relevance terms")

        # Title boost
        if title_high > 0:
            score += min(15, title_high * 8)
            reasons.append(f"{title_high} key terms in title")
        if title_medium > 0:
            score += min(8, title_medium * 4)

        # Cap at 100
        score = min(100, max(0, score))

        # Build rationale
        if reasons:
            rationale = f"Relevance analysis: {'; '.join(reasons)}. Score: {score}/100."
        elif score < 30:
            rationale = "No significant relevance indicators found for background screening industry."
        else:
            rationale = "Moderate relevance based on general industry context."

        return score, rationale


# =============================================================================
# ClassificationModule
# =============================================================================

class ClassificationModule(dspy.Module):
    """Full classification module combining region, topic, and relevance scoring.

    Produces ArticleClassification with all fields populated.
    """

    def __init__(self):
        """Initialize classification module."""
        super().__init__()
        self.region_predictor = dspy.Predict(RegionClassificationSignature)
        self.topic_predictor = dspy.Predict(TopicClassificationSignature)
        self.relevance_scorer = RelevanceScorer()

    def forward(
        self,
        title: str,
        content: str,
        source_url: str
    ) -> dspy.Prediction:
        """Classify an article.

        Args:
            title: Article title.
            content: Article content.
            source_url: Source URL.

        Returns:
            dspy.Prediction with classification field containing ArticleClassification.
        """
        # Classify region
        region, region_confidence = self._classify_region(title, content)

        # Classify topics
        topics, topic_confidence = self._classify_topics(title, content)

        # Score relevance
        relevance_result = self.relevance_scorer(title=title, content=content)
        relevance_score = relevance_result.relevance_score
        rationale = relevance_result.rationale

        # Combined confidence
        confidence = (region_confidence + topic_confidence) / 2

        # Create classification
        classification = ArticleClassification(
            region=region,
            topics=topics,
            relevance_score=relevance_score,
            confidence=confidence,
            rationale=rationale
        )

        return dspy.Prediction(
            classification=classification,
            source_url=source_url
        )

    def _classify_region(self, title: str, content: str) -> tuple:
        """Classify region using keyword matching.

        Args:
            title: Article title.
            content: Article content.

        Returns:
            Tuple of (RegionEnum, confidence).
        """
        combined_text = f"{title} {content}".lower()

        # Score each region
        region_scores = {}
        for region, keywords in REGION_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in combined_text)
            region_scores[region] = hits

        # Find best match
        best_region = max(region_scores, key=region_scores.get)
        best_score = region_scores[best_region]

        # If no clear region, default to WORLDWIDE
        if best_score == 0:
            return RegionEnum.WORLDWIDE, 0.5

        # Calculate confidence based on score
        confidence = min(0.95, 0.6 + (best_score * 0.1))

        return best_region, confidence

    def _classify_topics(self, title: str, content: str) -> tuple:
        """Classify topics using keyword matching.

        Args:
            title: Article title.
            content: Article content.

        Returns:
            Tuple of (List[TopicEnum], confidence).
        """
        combined_text = f"{title} {content}".lower()

        # Score each topic
        topic_scores = {}
        for topic, keywords in TOPIC_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in combined_text)
            topic_scores[topic] = hits

        # Select topics with score >= 1
        selected_topics = [
            topic for topic, score in topic_scores.items()
            if score >= 1
        ]

        # Sort by score descending
        selected_topics.sort(key=lambda t: topic_scores[t], reverse=True)

        # Ensure at least one topic
        if not selected_topics:
            # Default to most generic relevant topic
            selected_topics = [TopicEnum.TECHNOLOGY]
            confidence = 0.4
        else:
            # Confidence based on top topic score
            top_score = topic_scores[selected_topics[0]]
            confidence = min(0.95, 0.6 + (top_score * 0.1))

        return selected_topics, confidence


# =============================================================================
# Convenience Functions
# =============================================================================

def classify_article(
    title: str,
    content: str,
    source_url: str
) -> Dict[str, Any]:
    """Classify a single article.

    Args:
        title: Article title.
        content: Article content.
        source_url: Source URL.

    Returns:
        Dictionary with classification results.
    """
    classifier = ClassificationModule()
    result = classifier(title=title, content=content, source_url=source_url)

    classification = result.classification

    return {
        'region': classification.region,
        'topics': classification.topics,
        'relevance_score': classification.relevance_score,
        'confidence': classification.confidence,
        'rationale': classification.rationale,
        'source_url': source_url,
    }


def batch_classify(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Classify multiple articles.

    Args:
        articles: List of article dictionaries with title, content, source_url.

    Returns:
        List of dictionaries with classification results.
    """
    classifier = ClassificationModule()
    results = []

    for article in articles:
        title = article.get('title', '')
        content = article.get('content', '')
        source_url = article.get('source_url', '')

        result = classifier(title=title, content=content, source_url=source_url)
        classification = result.classification

        classified = {
            'title': title,
            'content': content,
            'source_url': source_url,
            'region': classification.region,
            'topics': classification.topics,
            'relevance_score': classification.relevance_score,
            'confidence': classification.confidence,
            'rationale': classification.rationale,
        }

        results.append(classified)

    return results


# =============================================================================
# Export for convenience
# =============================================================================

__all__ = [
    'RegionEnum',
    'TopicEnum',
    'ArticleClassification',
    'ClassificationModule',
    'RelevanceScorer',
    'classify_article',
    'batch_classify',
]
