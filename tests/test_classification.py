# tests/test_classification.py
"""
RED PHASE: Classification tests for DSPy Newsletter Research Tool PoC.

Beads Task: dspy-0s9 (Region, Topic, Scoring DSPy Modules)

These tests verify:
- ArticleClassification Pydantic model for TypedPredictor output
- RegionClassifier DSPy module
- TopicClassifier DSPy module
- RelevanceScorer with ChainOfThought
- Combined ClassificationModule

Expected Behavior in RED Phase:
- All tests should FAIL because classification modules are not yet implemented
- ImportError for 'from src.classification import ...'
"""

import pytest
from unittest.mock import Mock, patch
from enum import Enum


# =============================================================================
# ArticleClassification Model Tests
# =============================================================================

class TestArticleClassificationModel:
    """RED: Test Pydantic model before implementation."""

    def test_classification_model_exists(self):
        """Test ArticleClassification model can be imported."""
        from src.classification import ArticleClassification
        assert ArticleClassification is not None

    def test_classification_model_has_region(self):
        """Test model includes region field."""
        from src.classification import ArticleClassification
        fields = ArticleClassification.model_fields
        assert 'region' in fields

    def test_classification_model_has_topics(self):
        """Test model includes topics field as list."""
        from src.classification import ArticleClassification
        fields = ArticleClassification.model_fields
        assert 'topics' in fields

    def test_classification_model_has_relevance_score(self):
        """Test model includes relevance_score."""
        from src.classification import ArticleClassification
        fields = ArticleClassification.model_fields
        assert 'relevance_score' in fields

    def test_classification_model_has_confidence(self):
        """Test model includes confidence score."""
        from src.classification import ArticleClassification
        fields = ArticleClassification.model_fields
        assert 'confidence' in fields

    def test_classification_model_has_rationale(self):
        """Test model includes rationale for explainability."""
        from src.classification import ArticleClassification
        fields = ArticleClassification.model_fields
        assert 'rationale' in fields

    def test_classification_model_validates_region(self):
        """Test region field uses RegionEnum."""
        from src.classification import ArticleClassification, RegionEnum
        from pydantic import ValidationError

        # Valid region should work
        classification = ArticleClassification(
            region=RegionEnum.EUROPE,
            topics=[],
            relevance_score=75,
            confidence=0.9,
            rationale="Test rationale"
        )
        assert classification.region == RegionEnum.EUROPE

    def test_classification_model_validates_topics(self):
        """Test topics field uses TopicEnum list."""
        from src.classification import ArticleClassification, RegionEnum, TopicEnum

        classification = ArticleClassification(
            region=RegionEnum.N_AMERICA_CARIBBEAN,
            topics=[TopicEnum.REGULATORY, TopicEnum.COURT_CASES],
            relevance_score=85,
            confidence=0.95,
            rationale="Test rationale"
        )
        assert TopicEnum.REGULATORY in classification.topics


# =============================================================================
# RegionEnum Tests (imported from models or classification)
# =============================================================================

class TestRegionEnumFromClassification:
    """Test region taxonomy availability in classification module."""

    def test_region_enum_importable(self):
        """Test RegionEnum can be imported from classification."""
        from src.classification import RegionEnum
        assert RegionEnum is not None

    def test_region_enum_has_required_values(self):
        """Test all required regions exist."""
        from src.classification import RegionEnum
        required_regions = [
            'AFRICA_ME', 'APAC', 'EUROPE',
            'N_AMERICA_CARIBBEAN', 'S_AMERICA', 'WORLDWIDE'
        ]
        for region in required_regions:
            assert hasattr(RegionEnum, region)


# =============================================================================
# TopicEnum Tests
# =============================================================================

class TestTopicEnumFromClassification:
    """Test topic taxonomy availability in classification module."""

    def test_topic_enum_importable(self):
        """Test TopicEnum can be imported from classification."""
        from src.classification import TopicEnum
        assert TopicEnum is not None

    def test_topic_enum_has_required_values(self):
        """Test all required topics exist."""
        from src.classification import TopicEnum
        required_topics = [
            'REGULATORY', 'CRIMINAL_RECORDS', 'CREDENTIALS',
            'IMMIGRATION', 'M_AND_A', 'TECHNOLOGY',
            'EVENTS', 'COURT_CASES'
        ]
        for topic in required_topics:
            assert hasattr(TopicEnum, topic)


# =============================================================================
# Classification Module Tests
# =============================================================================

class TestClassificationModule:
    """Integration tests for TypedPredictor classification module."""

    def test_classification_module_exists(self):
        """Test ClassificationModule can be imported."""
        from src.classification import ClassificationModule
        assert ClassificationModule is not None

    def test_classification_module_is_dspy_module(self):
        """Test module is a proper DSPy module."""
        import dspy
        from src.classification import ClassificationModule
        classifier = ClassificationModule()
        assert isinstance(classifier, dspy.Module)

    def test_classification_returns_valid_pydantic(self, mock_dspy_lm, sample_article):
        """Test output is valid ArticleClassification."""
        from src.classification import ClassificationModule, ArticleClassification

        classifier = ClassificationModule()
        result = classifier(
            title=sample_article['title'],
            content=sample_article['content'],
            source_url=sample_article['source_url']
        )

        # Should return ArticleClassification instance
        assert isinstance(result.classification, ArticleClassification)

    def test_classification_assigns_region(self, mock_dspy_lm, sample_article):
        """Test classification assigns a region."""
        from src.classification import ClassificationModule, RegionEnum

        classifier = ClassificationModule()
        result = classifier(
            title=sample_article['title'],
            content=sample_article['content'],
            source_url=sample_article['source_url']
        )

        assert result.classification.region is not None
        assert isinstance(result.classification.region, RegionEnum)

    def test_classification_assigns_topics(self, mock_dspy_lm, sample_article):
        """Test classification assigns topic(s)."""
        from src.classification import ClassificationModule, TopicEnum

        classifier = ClassificationModule()
        result = classifier(
            title=sample_article['title'],
            content=sample_article['content'],
            source_url=sample_article['source_url']
        )

        assert result.classification.topics is not None
        assert isinstance(result.classification.topics, list)
        assert len(result.classification.topics) > 0


# =============================================================================
# Region Classification Accuracy Tests
# =============================================================================

class TestRegionClassificationAccuracy:
    """Test region assignment accuracy with known inputs."""

    def test_european_article_classified_as_europe(self, mock_dspy_lm, european_gdpr_article):
        """Test GDPR Germany article gets EUROPE region."""
        from src.classification import ClassificationModule, RegionEnum

        classifier = ClassificationModule()
        result = classifier(
            title=european_gdpr_article['title'],
            content=european_gdpr_article['content'],
            source_url=european_gdpr_article['source_url']
        )

        assert result.classification.region == RegionEnum.EUROPE

    def test_apac_article_classified_as_apac(self, mock_dspy_lm, apac_article):
        """Test Singapore article gets APAC region."""
        from src.classification import ClassificationModule, RegionEnum

        classifier = ClassificationModule()
        result = classifier(
            title=apac_article['title'],
            content=apac_article['content'],
            source_url=apac_article['source_url']
        )

        assert result.classification.region == RegionEnum.APAC

    def test_fcra_article_classified_as_n_america(self, mock_dspy_lm, sample_article):
        """Test FCRA article gets N_AMERICA_CARIBBEAN region."""
        from src.classification import ClassificationModule, RegionEnum

        classifier = ClassificationModule()
        result = classifier(
            title=sample_article['title'],
            content=sample_article['content'],
            source_url=sample_article['source_url']
        )

        assert result.classification.region == RegionEnum.N_AMERICA_CARIBBEAN

    def test_global_article_classified_as_worldwide(self, mock_dspy_lm):
        """Test global industry article gets WORLDWIDE region."""
        from src.classification import ClassificationModule, RegionEnum

        classifier = ClassificationModule()
        result = classifier(
            title="Global Background Screening Market Report",
            content="""
            The worldwide background screening market continues to grow
            across all regions. Companies in North America, Europe, APAC,
            and emerging markets are all expanding verification services.
            """,
            source_url="https://example.com/global-report"
        )

        assert result.classification.region == RegionEnum.WORLDWIDE


# =============================================================================
# Topic Classification Accuracy Tests
# =============================================================================

class TestTopicClassificationAccuracy:
    """Test topic assignment accuracy with known inputs."""

    def test_fcra_article_has_regulatory_topic(self, mock_dspy_lm, sample_article):
        """Test FCRA article gets REGULATORY topic."""
        from src.classification import ClassificationModule, TopicEnum

        classifier = ClassificationModule()
        result = classifier(
            title=sample_article['title'],
            content=sample_article['content'],
            source_url=sample_article['source_url']
        )

        assert TopicEnum.REGULATORY in result.classification.topics

    def test_court_case_article_has_court_cases_topic(self, mock_dspy_lm, court_case_article):
        """Test settlement article gets COURT_CASES topic."""
        from src.classification import ClassificationModule, TopicEnum

        classifier = ClassificationModule()
        result = classifier(
            title=court_case_article['title'],
            content=court_case_article['content'],
            source_url=court_case_article['source_url']
        )

        assert TopicEnum.COURT_CASES in result.classification.topics

    def test_multi_topic_classification(self, mock_dspy_lm, court_case_article):
        """Test articles can have multiple topics."""
        from src.classification import ClassificationModule, TopicEnum

        classifier = ClassificationModule()
        result = classifier(
            title=court_case_article['title'],
            content=court_case_article['content'],
            source_url=court_case_article['source_url']
        )

        # Court case about FCRA should have both COURT_CASES and REGULATORY
        assert TopicEnum.COURT_CASES in result.classification.topics
        assert TopicEnum.REGULATORY in result.classification.topics

    def test_technology_article_has_technology_topic(self, mock_dspy_lm):
        """Test AI screening article gets TECHNOLOGY topic."""
        from src.classification import ClassificationModule, TopicEnum

        classifier = ClassificationModule()
        result = classifier(
            title="AI-Powered Background Screening Platform Launches",
            content="""
            A new artificial intelligence platform promises to reduce
            background check turnaround times by 80%. The machine learning
            system automates court record searches and credential verification.
            """,
            source_url="https://example.com/ai-screening"
        )

        assert TopicEnum.TECHNOLOGY in result.classification.topics


# =============================================================================
# Relevance Scoring Tests (ChainOfThought)
# =============================================================================

class TestRelevanceScoring:
    """Test relevance scoring with ChainOfThought reasoning."""

    def test_relevance_scorer_exists(self):
        """Test RelevanceScorer can be imported."""
        from src.classification import RelevanceScorer
        assert RelevanceScorer is not None

    def test_scorer_provides_rationale(self, mock_dspy_lm, sample_article):
        """Test ChainOfThought provides reasoning."""
        from src.classification import RelevanceScorer

        scorer = RelevanceScorer()
        result = scorer(
            title=sample_article['title'],
            content=sample_article['content']
        )

        # ChainOfThought should provide rationale
        assert hasattr(result, 'rationale')
        assert len(result.rationale) > 0

    def test_relevance_score_bounded(self, mock_dspy_lm, sample_article):
        """Test relevance score is 0-100."""
        from src.classification import RelevanceScorer

        scorer = RelevanceScorer()
        result = scorer(
            title=sample_article['title'],
            content=sample_article['content']
        )

        assert 0 <= result.relevance_score <= 100

    def test_high_relevance_content_scores_high(self, mock_dspy_lm, sample_article):
        """Test highly relevant content gets high score."""
        from src.classification import RelevanceScorer

        scorer = RelevanceScorer()
        result = scorer(
            title=sample_article['title'],
            content=sample_article['content']
        )

        assert result.relevance_score >= 70

    def test_irrelevant_content_scores_low(self, mock_dspy_lm, irrelevant_article):
        """Test irrelevant content gets low score."""
        from src.classification import RelevanceScorer

        scorer = RelevanceScorer()
        result = scorer(
            title=irrelevant_article['title'],
            content=irrelevant_article['content']
        )

        assert result.relevance_score <= 30


# =============================================================================
# Combined Classification Pipeline Tests
# =============================================================================

class TestClassificationPipeline:
    """Test the complete classification pipeline."""

    def test_classify_article_function_exists(self):
        """Test classify_article convenience function exists."""
        from src.classification import classify_article
        assert classify_article is not None

    def test_classify_article_returns_complete_result(self, mock_dspy_lm, sample_article):
        """Test classify_article returns all fields."""
        from src.classification import classify_article

        result = classify_article(
            title=sample_article['title'],
            content=sample_article['content'],
            source_url=sample_article['source_url']
        )

        assert 'region' in result
        assert 'topics' in result
        assert 'relevance_score' in result
        assert 'confidence' in result
        assert 'rationale' in result

    def test_classification_includes_metadata(self, mock_dspy_lm, sample_article):
        """Test classification preserves source metadata."""
        from src.classification import classify_article

        result = classify_article(
            title=sample_article['title'],
            content=sample_article['content'],
            source_url=sample_article['source_url']
        )

        # Should include original metadata
        assert 'source_url' in result or hasattr(result, 'source_url')

    def test_batch_classification_function_exists(self):
        """Test batch_classify function exists."""
        from src.classification import batch_classify
        assert batch_classify is not None

    def test_batch_classification_processes_multiple(self, mock_dspy_lm, sample_article, european_gdpr_article):
        """Test batch_classify handles multiple articles."""
        from src.classification import batch_classify

        articles = [sample_article, european_gdpr_article]
        results = batch_classify(articles)

        assert len(results) == 2
        # Different regions expected
        assert results[0]['region'] != results[1]['region']
