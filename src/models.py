# src/models.py
"""Core data models for DSPy Newsletter Research Tool.

This module defines:
- Article: Main data model for newsletter articles
- RegionEnum: Geographic region taxonomy
- TopicEnum: Topic classification taxonomy
- ProcessingLog: Audit trail for pipeline stages
- ArticleClassification: DSPy classification output model
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional
import hashlib

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Region Taxonomy
# =============================================================================

class RegionEnum(str, Enum):
    """Geographic region taxonomy for article classification.

    Based on the background screening industry's regional focus areas.
    """
    AFRICA_ME = "africa_me"  # Africa & Middle East
    APAC = "apac"  # Asia Pacific
    EUROPE = "europe"  # Europe
    N_AMERICA_CARIBBEAN = "n_america_caribbean"  # North America & Caribbean
    S_AMERICA = "s_america"  # South America
    WORLDWIDE = "worldwide"  # Global/Worldwide content


# =============================================================================
# Topic Taxonomy
# =============================================================================

class TopicEnum(str, Enum):
    """Topic taxonomy for article classification.

    Based on background screening industry topics of interest.
    """
    REGULATORY = "regulatory"  # Regulatory changes, compliance
    CRIMINAL_RECORDS = "criminal_records"  # Criminal background checks
    CREDENTIALS = "credentials"  # Education, employment verification
    IMMIGRATION = "immigration"  # Immigration, work authorization
    M_AND_A = "m_and_a"  # Mergers & Acquisitions
    TECHNOLOGY = "technology"  # Technology, AI, automation
    EVENTS = "events"  # Industry events, conferences
    COURT_CASES = "court_cases"  # Legal cases, litigation


# =============================================================================
# Processing Status
# =============================================================================

class ProcessingStatus(str, Enum):
    """Status of article processing through the pipeline."""
    PENDING = "pending"
    INGESTED = "ingested"
    PREFILTERED = "prefiltered"
    CLASSIFIED = "classified"
    STORED = "stored"
    FAILED = "failed"


# =============================================================================
# Article Model
# =============================================================================

class Article(BaseModel):
    """Main data model for newsletter articles.

    Represents an article at any stage of the processing pipeline.
    """
    # Required fields
    title: str = Field(..., min_length=1, description="Article title")
    content: str = Field(..., min_length=1, description="Full article content")

    # Optional fields
    source_url: Optional[str] = Field(None, description="Source URL of the article")
    source: Optional[str] = Field(None, description="Source name/publication")
    published_date: Optional[datetime] = Field(None, description="Publication date")

    # Classification fields
    region: Optional[RegionEnum] = Field(None, description="Geographic region")
    topics: Optional[List[TopicEnum]] = Field(default_factory=list, description="Article topics")
    relevance_score: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Relevance score (0-100)"
    )
    summary: Optional[str] = Field(None, description="AI-generated summary")
    reasoning: Optional[str] = Field(None, description="Classification reasoning")

    # Deduplication
    content_hash: Optional[str] = Field(None, description="Content hash for deduplication")

    # Metadata
    ingested_at: Optional[datetime] = Field(default_factory=datetime.now, description="Ingestion timestamp")
    country: Optional[str] = Field(None, description="Country")

    @model_validator(mode='after')
    def generate_content_hash(self):
        """Generate content hash if not provided."""
        if self.content_hash is None:
            # Create hash from title + content for deduplication
            hash_input = f"{self.title}{self.content}".encode('utf-8')
            self.content_hash = hashlib.sha256(hash_input).hexdigest()[:32]
        return self

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None
        }
    }


# =============================================================================
# Processing Log Model
# =============================================================================

class ProcessingLog(BaseModel):
    """Audit trail for tracking article through processing pipeline.

    Each stage of processing updates relevant fields.
    """
    article_id: str = Field(..., description="Unique identifier for the article")

    # Timestamps for each stage
    ingestion_timestamp: Optional[datetime] = Field(
        default_factory=datetime.now,
        description="When article was ingested"
    )
    prefilter_timestamp: Optional[datetime] = Field(None, description="When prefilter completed")
    classification_timestamp: Optional[datetime] = Field(None, description="When classification completed")
    storage_timestamp: Optional[datetime] = Field(None, description="When stored to vector DB")

    # Results from each stage
    prefilter_result: Optional[bool] = Field(None, description="Prefilter pass/fail")
    classification_result: Optional[dict] = Field(None, description="Classification output")

    # Status tracking
    status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description="Current processing status"
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None
        }
    }


# =============================================================================
# Article Classification Model
# =============================================================================

class ArticleClassification(BaseModel):
    """DSPy classification output model.

    Structured output from the classification pipeline.
    """
    region: RegionEnum = Field(..., description="Classified region")
    topics: List[TopicEnum] = Field(default_factory=list, description="Classified topics")
    relevance_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Relevance score (0-100)"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Classification confidence (0-1)"
    )
    rationale: str = Field(..., description="Explanation for classification")
    summary: Optional[str] = Field(None, description="Article summary")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "region": "n_america_caribbean",
                    "topics": ["regulatory", "criminal_records"],
                    "relevance_score": 85.0,
                    "confidence": 0.92,
                    "rationale": "Article discusses new FCRA regulations affecting criminal background checks in the US."
                }
            ]
        }
    }


# =============================================================================
# Raw Article (for ingestion stage)
# =============================================================================

class RawArticle(BaseModel):
    """Article data as ingested from RSS feed before processing."""
    title: str
    content: str
    source_url: str
    source: str
    published_date: Optional[datetime] = None
    feed_url: str = Field(..., description="RSS feed this was ingested from")
    raw_html: Optional[str] = Field(None, description="Original HTML content")

    @property
    def content_hash(self) -> str:
        """Generate content hash for deduplication."""
        hash_input = f"{self.title}{self.content}".encode('utf-8')
        return hashlib.sha256(hash_input).hexdigest()[:32]


# =============================================================================
# Processed Article (for storage stage)
# =============================================================================

class ProcessedArticle(Article):
    """Fully processed article ready for vector storage.

    Extends Article with required classification fields.
    """
    # Make classification fields required for storage
    region: RegionEnum = Field(..., description="Classified region")
    topics: List[TopicEnum] = Field(..., min_length=1, description="Classified topics")
    relevance_score: float = Field(..., ge=0, le=100, description="Relevance score")
    summary: str = Field(..., description="AI-generated summary")
