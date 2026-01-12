# tests/test_models.py
"""
RED PHASE: Data model tests for DSPy Newsletter Research Tool PoC.

Beads Task: dspy-4tp (Data Models)

These tests verify:
- Article Pydantic model with all required fields
- Region and Topic taxonomy enums
- ProcessingLog for audit trail
- Validation rules and constraints

Expected Behavior in RED Phase:
- All tests should FAIL because models are not yet implemented
- ImportError for 'from src.models import ...'
"""

import pytest
from datetime import datetime
from pydantic import ValidationError


# =============================================================================
# Article Model Tests
# =============================================================================

class TestArticleModel:
    """RED: Test Article model before implementation."""

    def test_article_model_exists(self):
        """Test Article model can be imported."""
        from src.models import Article
        assert Article is not None

    def test_article_requires_title(self):
        """Test Article requires title field."""
        from src.models import Article
        with pytest.raises(ValidationError):
            Article(content="Some content without title")

    def test_article_requires_content(self):
        """Test Article requires content field."""
        from src.models import Article
        with pytest.raises(ValidationError):
            Article(title="Title without content")

    def test_article_accepts_valid_data(self):
        """Test Article accepts complete valid data."""
        from src.models import Article
        article = Article(
            title="Test Article",
            content="Test content about background screening",
            source_url="https://example.com/article",
            published_date=datetime.now()
        )
        assert article.title == "Test Article"
        assert article.content == "Test content about background screening"

    def test_article_has_source_url_field(self):
        """Test Article has source_url field."""
        from src.models import Article
        article = Article(
            title="Test",
            content="Content",
            source_url="https://example.com"
        )
        assert hasattr(article, 'source_url')
        assert article.source_url == "https://example.com"

    def test_article_has_published_date_field(self):
        """Test Article has published_date field."""
        from src.models import Article
        now = datetime.now()
        article = Article(
            title="Test",
            content="Content",
            published_date=now
        )
        assert hasattr(article, 'published_date')
        assert article.published_date == now

    def test_article_has_region_field(self):
        """Test Article has region field for geographic classification."""
        from src.models import Article, RegionEnum
        article = Article(
            title="Test",
            content="Content",
            region=RegionEnum.N_AMERICA_CARIBBEAN
        )
        assert hasattr(article, 'region')

    def test_article_has_topics_field(self):
        """Test Article has topics field as list."""
        from src.models import Article, TopicEnum
        article = Article(
            title="Test",
            content="Content",
            topics=[TopicEnum.REGULATORY, TopicEnum.COURT_CASES]
        )
        assert hasattr(article, 'topics')
        assert isinstance(article.topics, list)

    def test_article_has_relevance_score_field(self):
        """Test Article has relevance_score field."""
        from src.models import Article
        article = Article(
            title="Test",
            content="Content",
            relevance_score=85.5
        )
        assert hasattr(article, 'relevance_score')
        assert article.relevance_score == 85.5

    def test_article_relevance_score_bounded(self):
        """Test relevance_score is bounded 0-100."""
        from src.models import Article
        with pytest.raises(ValidationError):
            Article(title="Test", content="Content", relevance_score=150)
        with pytest.raises(ValidationError):
            Article(title="Test", content="Content", relevance_score=-10)

    def test_article_has_content_hash_field(self):
        """Test Article has content_hash for deduplication."""
        from src.models import Article
        article = Article(title="Test", content="Content")
        assert hasattr(article, 'content_hash')

    def test_article_generates_content_hash(self):
        """Test content_hash is automatically generated."""
        from src.models import Article
        article = Article(title="Test", content="Content")
        assert article.content_hash is not None
        assert len(article.content_hash) > 0


# =============================================================================
# Region Enum Tests
# =============================================================================

class TestRegionEnum:
    """Test geographic region taxonomy."""

    def test_region_enum_exists(self):
        """Test RegionEnum can be imported."""
        from src.models import RegionEnum
        assert RegionEnum is not None

    def test_region_enum_has_africa_me(self):
        """Test AFRICA_ME region exists."""
        from src.models import RegionEnum
        assert hasattr(RegionEnum, 'AFRICA_ME')

    def test_region_enum_has_apac(self):
        """Test APAC region exists."""
        from src.models import RegionEnum
        assert hasattr(RegionEnum, 'APAC')

    def test_region_enum_has_europe(self):
        """Test EUROPE region exists."""
        from src.models import RegionEnum
        assert hasattr(RegionEnum, 'EUROPE')

    def test_region_enum_has_n_america_caribbean(self):
        """Test N_AMERICA_CARIBBEAN region exists."""
        from src.models import RegionEnum
        assert hasattr(RegionEnum, 'N_AMERICA_CARIBBEAN')

    def test_region_enum_has_s_america(self):
        """Test S_AMERICA region exists."""
        from src.models import RegionEnum
        assert hasattr(RegionEnum, 'S_AMERICA')

    def test_region_enum_has_worldwide(self):
        """Test WORLDWIDE region exists for global content."""
        from src.models import RegionEnum
        assert hasattr(RegionEnum, 'WORLDWIDE')

    def test_region_enum_has_all_required_values(self):
        """Test all required regions exist."""
        from src.models import RegionEnum
        required_regions = [
            'AFRICA_ME', 'APAC', 'EUROPE',
            'N_AMERICA_CARIBBEAN', 'S_AMERICA', 'WORLDWIDE'
        ]
        for region in required_regions:
            assert hasattr(RegionEnum, region), f"Missing region: {region}"


# =============================================================================
# Topic Enum Tests
# =============================================================================

class TestTopicEnum:
    """Test topic taxonomy for article classification."""

    def test_topic_enum_exists(self):
        """Test TopicEnum can be imported."""
        from src.models import TopicEnum
        assert TopicEnum is not None

    def test_topic_enum_has_regulatory(self):
        """Test REGULATORY topic exists."""
        from src.models import TopicEnum
        assert hasattr(TopicEnum, 'REGULATORY')

    def test_topic_enum_has_criminal_records(self):
        """Test CRIMINAL_RECORDS topic exists."""
        from src.models import TopicEnum
        assert hasattr(TopicEnum, 'CRIMINAL_RECORDS')

    def test_topic_enum_has_credentials(self):
        """Test CREDENTIALS topic exists."""
        from src.models import TopicEnum
        assert hasattr(TopicEnum, 'CREDENTIALS')

    def test_topic_enum_has_immigration(self):
        """Test IMMIGRATION topic exists."""
        from src.models import TopicEnum
        assert hasattr(TopicEnum, 'IMMIGRATION')

    def test_topic_enum_has_m_and_a(self):
        """Test M_AND_A (mergers and acquisitions) topic exists."""
        from src.models import TopicEnum
        assert hasattr(TopicEnum, 'M_AND_A')

    def test_topic_enum_has_technology(self):
        """Test TECHNOLOGY topic exists."""
        from src.models import TopicEnum
        assert hasattr(TopicEnum, 'TECHNOLOGY')

    def test_topic_enum_has_events(self):
        """Test EVENTS topic exists."""
        from src.models import TopicEnum
        assert hasattr(TopicEnum, 'EVENTS')

    def test_topic_enum_has_court_cases(self):
        """Test COURT_CASES topic exists."""
        from src.models import TopicEnum
        assert hasattr(TopicEnum, 'COURT_CASES')

    def test_topic_enum_has_all_required_values(self):
        """Test all required topics exist."""
        from src.models import TopicEnum
        required_topics = [
            'REGULATORY', 'CRIMINAL_RECORDS', 'CREDENTIALS',
            'IMMIGRATION', 'M_AND_A', 'TECHNOLOGY',
            'EVENTS', 'COURT_CASES'
        ]
        for topic in required_topics:
            assert hasattr(TopicEnum, topic), f"Missing topic: {topic}"


# =============================================================================
# ProcessingLog Model Tests
# =============================================================================

class TestProcessingLogModel:
    """Test ProcessingLog for audit trail tracking."""

    def test_processing_log_exists(self):
        """Test ProcessingLog model can be imported."""
        from src.models import ProcessingLog
        assert ProcessingLog is not None

    def test_processing_log_requires_article_id(self):
        """Test ProcessingLog requires article_id field."""
        from src.models import ProcessingLog
        with pytest.raises(ValidationError):
            ProcessingLog()  # No article_id

    def test_processing_log_has_ingestion_timestamp(self):
        """Test ProcessingLog tracks ingestion time."""
        from src.models import ProcessingLog
        log = ProcessingLog(article_id="test-123")
        assert hasattr(log, 'ingestion_timestamp')

    def test_processing_log_has_prefilter_result(self):
        """Test ProcessingLog tracks prefilter stage."""
        from src.models import ProcessingLog
        log = ProcessingLog(article_id="test-123")
        assert hasattr(log, 'prefilter_result')

    def test_processing_log_has_classification_result(self):
        """Test ProcessingLog tracks classification stage."""
        from src.models import ProcessingLog
        log = ProcessingLog(article_id="test-123")
        assert hasattr(log, 'classification_result')

    def test_processing_log_has_storage_timestamp(self):
        """Test ProcessingLog tracks storage completion."""
        from src.models import ProcessingLog
        log = ProcessingLog(article_id="test-123")
        assert hasattr(log, 'storage_timestamp')

    def test_processing_log_has_status_field(self):
        """Test ProcessingLog has overall status."""
        from src.models import ProcessingLog
        log = ProcessingLog(article_id="test-123")
        assert hasattr(log, 'status')

    def test_processing_log_has_error_message_field(self):
        """Test ProcessingLog can store error messages."""
        from src.models import ProcessingLog
        log = ProcessingLog(
            article_id="test-123",
            error_message="Failed at prefilter stage"
        )
        assert hasattr(log, 'error_message')
        assert log.error_message == "Failed at prefilter stage"


# =============================================================================
# Classification Result Model Tests
# =============================================================================

class TestArticleClassificationModel:
    """Test ArticleClassification model for DSPy output."""

    def test_article_classification_exists(self):
        """Test ArticleClassification model can be imported."""
        from src.models import ArticleClassification
        assert ArticleClassification is not None

    def test_classification_has_region(self):
        """Test ArticleClassification includes region field."""
        from src.models import ArticleClassification
        fields = ArticleClassification.model_fields
        assert 'region' in fields

    def test_classification_has_topics(self):
        """Test ArticleClassification includes topics field."""
        from src.models import ArticleClassification
        fields = ArticleClassification.model_fields
        assert 'topics' in fields

    def test_classification_has_relevance_score(self):
        """Test ArticleClassification includes relevance_score."""
        from src.models import ArticleClassification
        fields = ArticleClassification.model_fields
        assert 'relevance_score' in fields

    def test_classification_has_confidence(self):
        """Test ArticleClassification includes confidence score."""
        from src.models import ArticleClassification
        fields = ArticleClassification.model_fields
        assert 'confidence' in fields

    def test_classification_has_rationale(self):
        """Test ArticleClassification includes rationale for explainability."""
        from src.models import ArticleClassification
        fields = ArticleClassification.model_fields
        assert 'rationale' in fields
