# DSPy Newsletter Research Tool - Solution Design
## Architect 3: Systematic Decomposition Approach

**Author**: Solution Architect (Systematic Decomposition Strategy)
**Date**: 2026-01-12
**Version**: 1.0

---

## Executive Summary

This solution design presents a modular, DSPy-based architecture for PreEmploymentDirectory's automated newsletter research system. Using systematic decomposition, the system is divided into six core modules with well-defined interfaces and contracts, enabling independent development, testing, and optimization.

**Key Design Principles**:
- **Clear Module Boundaries**: Each module has explicit input/output contracts
- **Progressive Filtering**: Funnel architecture reduces API costs (500+ → 20-50 articles/day)
- **DSPy-Native**: Leverages signatures, modules, and optimizers for LLM programming
- **PoC-Appropriate**: Balances sophistication with practical constraints
- **Incremental Delivery**: 4-phase plan delivers value at each milestone

**Expected Outcomes**:
- 70%+ recall vs manual process
- 90%+ cost reduction through intelligent filtering
- <5 second query response time
- 20-50 high-quality candidate articles per day

---

## 1. System Architecture

### 1.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION SOURCES (247+)                     │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│ RSS Feeds    │  Scrapers    │ Email Parser │ Social Monitors   │
│ (Tier 1-2)   │  (Tier 3)    │  (Tier 4)    │ (Future)          │
└──────┬───────┴──────┬───────┴──────┬───────┴───────┬───────────┘
       │              │              │               │
       └──────────────┴──────────────┴───────────────┘
                      │
                      ▼
       ┌──────────────────────────────────────┐
       │      INGESTION MODULE                │
       │  Contract: RawArticle → Article      │
       │  {title, url, content, source, date} │
       └──────────────┬───────────────────────┘
                      │
                      ▼
       ┌──────────────────────────────────────┐
       │   RELEVANCE FILTER (Stage 1)         │
       │   DSPy Module: dspy.Predict          │
       │   Contract: Article → FilterResult   │
       │   {is_relevant: bool, confidence}    │
       └──────────────┬───────────────────────┘
                      │ (Keep 40%)
                      ▼
       ┌──────────────────────────────────────┐
       │   DEDUPLICATION MODULE               │
       │   Hybrid: Hash + Semantic Clustering │
       │   Contract: Article[] → ClusterMap   │
       │   {cluster_id, is_primary}           │
       └──────────────┬───────────────────────┘
                      │ (Reduce 30%)
                      ▼
       ┌──────────────────────────────────────┐
       │   CLASSIFICATION MODULE              │
       │   DSPy Module: TypedPredictor        │
       │   Contract: Article → Classification │
       │   {region, topics[], confidence}     │
       └──────────────┬───────────────────────┘
                      │
                      ▼
       ┌──────────────────────────────────────┐
       │   RELEVANCE SCORING MODULE           │
       │   DSPy Module: ChainOfThought        │
       │   Contract: Article → ScoredArticle  │
       │   {score: 0-1, reasoning, keywords}  │
       └──────────────┬───────────────────────┘
                      │
                      ▼
       ┌──────────────────────────────────────┐
       │        STORAGE LAYER                 │
       │  ChromaDB (vectors) + JSON (metadata)│
       │  Contract: ScoredArticle → article_id│
       └──────────────┬───────────────────────┘
                      │
                      ▼
       ┌──────────────────────────────────────┐
       │   DAILY REPORT GENERATOR             │
       │   Filter: score > 0.6, top 50        │
       │   Output: Ranked list for review     │
       └──────────────────────────────────────┘
                      │
                      ▼
       ┌──────────────────────────────────────┐
       │      QUERY MODULE (CLI)              │
       │   DSPy Module: ReAct Agent           │
       │   Tools: Filter, Retrieve, Summarize │
       │   Contract: Query → Answer           │
       └──────────────────────────────────────┘
                      ▲
                      │
       ┌──────────────┴───────────────────────┐
       │   OPTIMIZATION FEEDBACK LOOP         │
       │   BootstrapFewShot → Compiled Models │
       └──────────────────────────────────────┘
```

### 1.2 Module Boundaries and Contracts

| Module | Input Contract | Output Contract | Responsibility |
|--------|---------------|-----------------|----------------|
| **Ingestion** | Source configs | `{title, url, content, source, published_date}` | Fetch & parse articles |
| **Relevance Filter** | Article | `{is_relevant: bool, confidence: float}` | Quick triage (keyword + LLM) |
| **Deduplication** | Article[] | `{cluster_id: str, is_primary: bool}` | Group duplicates |
| **Classification** | Article | `{region: Enum, topics: List[Enum], confidence: Dict}` | Multi-label categorization |
| **Scoring** | Article + Classification | `{score: float, reasoning: str, keywords: List[str]}` | Detailed relevance |
| **Storage** | ScoredArticle | `{article_id: UUID}` | Persist vectors + metadata |
| **Query** | Natural language query | `{answer: str, sources: List[Article]}` | RAG-based Q&A |

---

## 2. DSPy Signatures (Code Examples)

### 2.1 Relevance Filter Signature

```python
import dspy
from enum import Enum
from typing import List

class RelevanceCategory(str, Enum):
    HIGHLY_RELEVANT = "highly_relevant"
    POSSIBLY_RELEVANT = "possibly_relevant"
    NOT_RELEVANT = "not_relevant"

class ArticleRelevanceFilter(dspy.Signature):
    """
    Quick triage to filter articles related to background screening,
    employment verification, and regulatory compliance.

    Focus on primary keywords: background check, employment screening,
    criminal records, FCRA, GDPR, right to work, Ban the Box.
    """

    title: str = dspy.InputField(desc="Article headline")
    content_preview: str = dspy.InputField(desc="First 500 characters of article")
    source: str = dspy.InputField(desc="Publication name")

    category: RelevanceCategory = dspy.OutputField(
        desc="Relevance category based on keyword analysis"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score 0.0-1.0"
    )
    matched_keywords: List[str] = dspy.OutputField(
        desc="Primary or secondary keywords found in text"
    )
```

### 2.2 Classification Signature

```python
from typing import List, Dict
from pydantic import BaseModel, Field

class Region(str, Enum):
    AFRICA_MIDDLE_EAST = "africa_middle_east"
    ASIA_PACIFIC = "asia_pacific"
    EUROPE = "europe"
    NORTH_AMERICA = "north_america_caribbean"
    SOUTH_AMERICA = "south_america"
    WORLDWIDE = "worldwide"

class Topic(str, Enum):
    REGULATORY = "regulatory_legal_changes"
    CRIMINAL_BACKGROUND = "criminal_background_checks"
    EDUCATION_VERIFICATION = "education_credential_verification"
    IMMIGRATION = "immigration_right_to_work"
    INDUSTRY_NEWS = "industry_ma_company_news"
    TECHNOLOGY = "technology_product_announcements"
    EVENTS = "conference_event_news"
    LEGAL_CASES = "court_cases_legal_precedents"

class ClassificationOutput(BaseModel):
    """Structured output for article classification."""
    region: Region = Field(description="Primary geographic region")
    topics: List[Topic] = Field(
        description="All applicable topics (multi-label)",
        min_items=1,
        max_items=4
    )
    confidence_by_topic: Dict[str, float] = Field(
        description="Confidence score per topic"
    )
    country: str = Field(
        description="Specific country if identifiable, else 'Multiple' or 'Unknown'"
    )

class ArticleClassifier(dspy.Signature):
    """
    Classify background screening articles by region and topic.

    Regions represent geographic focus. Topics are multi-label
    (an article can cover multiple topics). Use country-specific
    legislation mentions for region identification.
    """

    title: str = dspy.InputField()
    content: str = dspy.InputField(desc="Full article text or substantial excerpt")
    source: str = dspy.InputField(desc="Publication name for context")

    classification: ClassificationOutput = dspy.OutputField(
        desc="Structured classification with region, topics, and confidence"
    )
```

### 2.3 Relevance Scoring Signature

```python
class ScoringOutput(BaseModel):
    """Detailed relevance scoring output."""
    relevance_score: float = Field(
        description="Score from 0.0 (irrelevant) to 1.0 (highly relevant)",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="2-3 sentence explanation of score",
        min_length=50,
        max_length=300
    )
    key_insights: List[str] = Field(
        description="Bullet points of why this matters to background screeners",
        min_items=1,
        max_items=3
    )
    matched_keywords: List[str] = Field(
        description="Primary and secondary keywords found"
    )

class RelevanceScorer(dspy.Signature):
    """
    Score article relevance for background screening industry professionals.

    High scores (0.7-1.0): Direct impact on screening operations, new regulations,
    major compliance changes, technology affecting verification processes.

    Medium scores (0.4-0.6): Industry trends, tangential legal changes, competitor news.

    Low scores (0.0-0.3): General HR news, loosely related topics.
    """

    title: str = dspy.InputField()
    content: str = dspy.InputField()
    region: str = dspy.InputField(desc="From classification module")
    topics: List[str] = dspy.InputField(desc="From classification module")
    source: str = dspy.InputField()

    scoring: ScoringOutput = dspy.OutputField(
        desc="Detailed relevance assessment with reasoning"
    )
```

### 2.4 Query/RAG Signature

```python
class QueryAnswer(BaseModel):
    """Structured answer to user query."""
    answer: str = Field(
        description="Comprehensive answer synthesizing retrieved articles",
        min_length=100
    )
    article_count: int = Field(
        description="Number of articles synthesized"
    )
    key_sources: List[Dict[str, str]] = Field(
        description="Top 3-5 source articles with titles and URLs"
    )
    follow_up_suggestions: List[str] = Field(
        description="Suggested related queries",
        max_items=3
    )

class NewsletterRAG(dspy.Signature):
    """
    Answer questions about background screening industry trends using
    retrieved newsletter articles.

    Synthesize information across multiple sources. Cite specific articles.
    Identify patterns and trends. Flag conflicting information.
    """

    question: str = dspy.InputField(desc="User's natural language query")
    context: str = dspy.InputField(
        desc="Retrieved article excerpts (from vector search)"
    )

    answer: QueryAnswer = dspy.OutputField(
        desc="Comprehensive answer with sources"
    )
```

---

## 3. Module Selection Rationale

### 3.1 Module-to-DSPy Component Mapping

| Processing Stage | DSPy Module | Justification |
|-----------------|-------------|---------------|
| **Relevance Filter** | `dspy.Predict` | Fast, simple classification without reasoning overhead. Optimized for throughput. |
| **Classification** | `dspy.TypedPredictor` | Enforces structured output (Pydantic models). Ensures valid enum values and confidence scores. |
| **Relevance Scoring** | `dspy.ChainOfThought` | Requires reasoning trail to explain scores. Improves explainability and allows debugging. |
| **Query/RAG** | `dspy.ReAct` | Multi-step reasoning with tools (filter, retrieve, summarize). Handles complex queries like "compare GDPR vs CCPA impact". |
| **Optimization** | `BootstrapFewShot` | PoC-appropriate. Requires minimal training data (20-50 examples). Fast compilation. |

### 3.2 Alternative Considerations

**Why NOT dspy.ChainOfThought for Classification?**
- Classification is straightforward pattern matching (region/topic identification)
- Reasoning overhead increases latency and cost without quality improvement
- TypedPredictor enforces output structure better

**Why NOT ColBERTv2 for RAG?**
- ChromaDB simpler setup for PoC
- ColBERTv2 requires index building and more infrastructure
- Phase 2 upgrade path if retrieval quality insufficient

**Why NOT MIPROv2 for Initial Optimization?**
- MIPROv2 requires larger training sets (100+ examples)
- BootstrapFewShot sufficient for PoC validation
- Can upgrade in Phase 3 if performance targets unmet

---

## 4. Optimization Strategy

### 4.1 Training Data Requirements

| Signature | Training Examples | Composition | Collection Method |
|-----------|------------------|-------------|-------------------|
| **RelevanceFilter** | 50 | 25 relevant, 25 irrelevant | Manual labeling of daily feed sample |
| **Classifier** | 40 | 5 per region, 5 per topic (overlap OK) | Ensure coverage of all labels |
| **RelevanceScorer** | 30 | 10 high (0.7+), 10 medium (0.4-0.6), 10 low (<0.4) | Gold standard scores from domain expert |
| **QueryRAG** | 20 | Mix of factual, comparative, trend queries | Simulate real user questions |

**Total Labeling Effort**: ~140 examples × 5 minutes each = ~12 hours one-time investment

### 4.2 Optimization Workflow

```python
import dspy
from dspy.teleprompt import BootstrapFewShot
from typing import List

# Define metric for RelevanceFilter
def relevance_filter_metric(example, prediction, trace=None) -> float:
    """
    Metric evaluating filter accuracy and efficiency.
    Penalize false negatives (missed relevant articles) more than false positives.
    """
    correct_category = (
        example.category == prediction.category
    )

    # False negative penalty
    if example.category != RelevanceCategory.NOT_RELEVANT and \
       prediction.category == RelevanceCategory.NOT_RELEVANT:
        return 0.0  # Critical failure

    # False positive tolerance (human review catches these)
    if example.category == RelevanceCategory.NOT_RELEVANT and \
       prediction.category != RelevanceCategory.NOT_RELEVANT:
        return 0.5  # Acceptable

    # Correct classification
    if correct_category:
        # Bonus for confidence calibration
        confidence_diff = abs(example.confidence - prediction.confidence)
        return 1.0 - (confidence_diff * 0.5)

    return 0.3  # Wrong category but not critical

# Optimization process
class FilterOptimizer:
    def __init__(self, training_data: List[dspy.Example]):
        self.training_data = training_data

    def optimize_filter(self, base_filter):
        """Compile optimized relevance filter."""
        optimizer = BootstrapFewShot(
            metric=relevance_filter_metric,
            max_bootstrapped_demos=8,  # Keep prompt manageable
            max_labeled_demos=4,        # Use some labeled examples
            max_rounds=2                # Quick compilation for PoC
        )

        compiled_filter = optimizer.compile(
            base_filter,
            trainset=self.training_data
        )

        return compiled_filter

    def save_optimized_model(self, compiled_filter, path: str):
        """Persist optimized prompts for production."""
        compiled_filter.save(path)

# Usage
base_filter = dspy.Predict(ArticleRelevanceFilter)
optimizer = FilterOptimizer(training_examples)
optimized_filter = optimizer.optimize_filter(base_filter)
optimized_filter.save("models/relevance_filter_v1.json")
```

### 4.3 Continuous Improvement Loop

```
┌─────────────────────────────────────────────────────┐
│  WEEK 1-2: Collect Training Data                    │
│  - Run unoptimized pipeline                         │
│  - Human labels 50 articles/day = 500+ examples     │
│  - Focus on edge cases and errors                   │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│  WEEK 3: Optimize & Validate                        │
│  - Run BootstrapFewShot per signature               │
│  - Validate on holdout set (20% of labeled data)    │
│  - A/B test optimized vs baseline                   │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│  WEEK 4+: Production & Monitoring                   │
│  - Deploy optimized models                          │
│  - Track precision/recall on daily outputs          │
│  - Quarterly re-optimization with new examples      │
└─────────────────────────────────────────────────────┘
```

---

## 5. RAG Implementation

### 5.1 Vector Storage Architecture

```python
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import hashlib

class NewsletterVectorStore:
    """
    ChromaDB-backed vector storage for newsletter articles.
    Stores full-text embeddings with metadata for hybrid retrieval.
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="newsletter_articles",
            metadata={"hnsw:space": "cosine"},  # Cosine similarity
            embedding_function=None  # Use OpenAI embeddings externally
        )

    def add_article(
        self,
        article_id: str,
        text: str,
        embedding: List[float],
        metadata: Dict
    ):
        """
        Add article to vector store.

        Args:
            article_id: UUID string
            text: Full article text
            embedding: Vector from OpenAI/Anthropic
            metadata: {region, topics, score, published_date, source, url}
        """
        self.collection.add(
            ids=[article_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Hybrid search: vector similarity + metadata filtering.

        Example filter:
            {"region": "asia_pacific", "score": {"$gte": 0.6}}
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=metadata_filter  # ChromaDB metadata filter
        )

        return self._format_results(results)

    def _format_results(self, raw_results) -> List[Dict]:
        """Convert ChromaDB results to structured format."""
        formatted = []
        for i in range(len(raw_results['ids'][0])):
            formatted.append({
                'id': raw_results['ids'][0][i],
                'text': raw_results['documents'][0][i],
                'distance': raw_results['distances'][0][i],
                'metadata': raw_results['metadatas'][0][i]
            })
        return formatted
```

### 5.2 RAG Module Implementation

```python
import dspy
from typing import List, Dict, Optional
import openai

class NewsletterRAGModule(dspy.Module):
    """
    RAG module for querying newsletter knowledge base.
    Uses dspy.Retrieve with custom retriever + generation.
    """

    def __init__(self, vector_store: NewsletterVectorStore):
        super().__init__()
        self.vector_store = vector_store

        # Custom retriever wrapping ChromaDB
        self.retrieve = self._create_retriever()

        # ReAct agent for complex queries
        self.agent = dspy.ReAct(
            signature=NewsletterRAG,
            tools=[self.filter_by_metadata, self.get_article_details],
            max_iters=3
        )

    def _create_retriever(self):
        """Create custom DSPy retriever."""
        class ChromaRetriever:
            def __init__(self, store):
                self.store = store

            def __call__(self, query: str, k: int = 10) -> List[str]:
                # Generate query embedding
                embedding = self._get_embedding(query)

                # Retrieve from ChromaDB
                results = self.store.search(
                    query_embedding=embedding,
                    n_results=k
                )

                # Format for DSPy (returns passages as strings)
                passages = [
                    f"[{r['metadata']['source']}] {r['text'][:500]}..."
                    for r in results
                ]
                return passages

            def _get_embedding(self, text: str) -> List[float]:
                response = openai.Embedding.create(
                    input=text,
                    model="text-embedding-3-small"
                )
                return response['data'][0]['embedding']

        return ChromaRetriever(self.vector_store)

    def filter_by_metadata(
        self,
        region: Optional[str] = None,
        topics: Optional[List[str]] = None,
        min_score: float = 0.0,
        days_back: Optional[int] = None
    ) -> str:
        """
        Tool for ReAct agent: filter articles by metadata.
        Returns summary of filtered results.
        """
        filter_dict = {}

        if region:
            filter_dict["region"] = region
        if min_score > 0:
            filter_dict["score"] = {"$gte": min_score}
        if topics:
            # ChromaDB supports $in operator
            filter_dict["topics"] = {"$in": topics}

        # Date filtering would require preprocessing dates
        # to Unix timestamps in metadata

        # Use dummy embedding for metadata-only search
        dummy_embedding = [0.0] * 1536
        results = self.vector_store.search(
            query_embedding=dummy_embedding,
            n_results=50,
            metadata_filter=filter_dict if filter_dict else None
        )

        summary = f"Found {len(results)} articles matching filters.\n"
        summary += "\n".join([
            f"- {r['metadata']['title']} ({r['metadata']['source']})"
            for r in results[:10]
        ])

        return summary

    def get_article_details(self, article_id: str) -> str:
        """Tool: Retrieve full article by ID."""
        # Implementation would fetch from storage
        pass

    def forward(self, question: str):
        """Main entry point for queries."""
        # ReAct agent handles tool use + answer generation
        return self.agent(question=question)
```

### 5.3 Query CLI Example

```python
# CLI interface
import click

@click.command()
@click.argument('query')
@click.option('--region', help='Filter by region')
@click.option('--topic', help='Filter by topic')
@click.option('--days', type=int, help='Articles from last N days')
def query_newsletter(query: str, region: str, topic: str, days: int):
    """Query the newsletter knowledge base."""

    # Initialize RAG module
    vector_store = NewsletterVectorStore()
    rag_module = NewsletterRAGModule(vector_store)

    # Construct augmented query with filters
    if region or topic or days:
        filter_context = f"Focus on: "
        if region:
            filter_context += f"region={region} "
        if topic:
            filter_context += f"topic={topic} "
        if days:
            filter_context += f"last {days} days"

        augmented_query = f"{query}. {filter_context}"
    else:
        augmented_query = query

    # Execute query
    result = rag_module(question=augmented_query)

    # Display results
    click.echo("\n" + "="*60)
    click.echo(f"QUERY: {query}")
    click.echo("="*60 + "\n")
    click.echo(result.answer.answer)
    click.echo("\n" + "-"*60)
    click.echo("KEY SOURCES:")
    for source in result.answer.key_sources:
        click.echo(f"  - {source['title']}")
        click.echo(f"    {source['url']}\n")

    if result.answer.follow_up_suggestions:
        click.echo("-"*60)
        click.echo("RELATED QUERIES:")
        for suggestion in result.answer.follow_up_suggestions:
            click.echo(f"  - {suggestion}")

if __name__ == '__main__':
    query_newsletter()
```

**Example Usage**:
```bash
# Basic query
python query.py "recent GDPR enforcement actions"

# Filtered query
python query.py "background check regulations" --region asia_pacific --days 30

# Complex query (ReAct will decompose)
python query.py "compare ban the box adoption in US vs Europe"
```

---

## 6. Phased Implementation Plan

### Phase 1: Foundation (Weeks 1-2)
**Goal**: Functional data pipeline with basic filtering

**Deliverables**:
1. **Ingestion Module**
   - RSS parser for Tier 1 feeds (20 feeds)
   - JSON file storage for raw articles
   - Scheduled daily batch job (cron)

2. **Keyword-Based Filter**
   - Rule-based relevance filter (no LLM)
   - Uses primary keyword list (background check, FCRA, etc.)
   - Outputs: `relevant_articles.json`

3. **Storage Setup**
   - ChromaDB initialization
   - Metadata schema definition
   - Basic deduplication (URL normalization)

**Success Criteria**:
- Ingest 100+ articles/day from RSS feeds
- Keyword filter achieves 60%+ recall (tested on 50 manual labels)
- No crashes or data loss

**Tasks**:
- [ ] Configure RSS feed parsers (feedparser library)
- [ ] Implement JSON storage schema
- [ ] Create keyword matching engine (regex + fuzzy)
- [ ] Set up ChromaDB with persistence
- [ ] Write deduplication logic (URL + title hashing)

---

### Phase 2: Intelligence (Weeks 3-4)
**Goal**: DSPy-powered classification and scoring

**Deliverables**:
1. **DSPy Integration**
   - Implement ArticleRelevanceFilter signature
   - Implement ArticleClassifier (TypedPredictor)
   - Implement RelevanceScorer (ChainOfThought)

2. **Pipeline Integration**
   - Replace keyword filter with DSPy filter
   - Add classification step
   - Add scoring step
   - Output: `scored_articles.json` with full metadata

3. **Embedding Generation**
   - Generate OpenAI embeddings for all articles
   - Store in ChromaDB with metadata

**Success Criteria**:
- Classification accuracy 70%+ on 40-article validation set
- Relevance scores correlate with human judgment (Spearman ρ > 0.7)
- Process 200+ articles/day within API budget ($5/day target)

**Tasks**:
- [ ] Define all DSPy signatures (code from Section 2)
- [ ] Implement dspy.Predict for relevance filter
- [ ] Implement dspy.TypedPredictor for classification
- [ ] Implement dspy.ChainOfThought for scoring
- [ ] Create embedding generation pipeline
- [ ] Integration testing with real feeds

---

### Phase 3: Optimization (Week 5)
**Goal**: Optimized models via BootstrapFewShot

**Deliverables**:
1. **Training Data Collection**
   - Manual labeling interface (simple CLI)
   - Collect 50 examples for RelevanceFilter
   - Collect 40 examples for Classifier
   - Collect 30 examples for Scorer

2. **Model Optimization**
   - Implement metric functions (Section 4.2)
   - Run BootstrapFewShot per signature
   - Save optimized models to disk
   - A/B testing harness

3. **Performance Validation**
   - Holdout set evaluation (20% of labeled data)
   - Precision/recall analysis
   - Error analysis and iteration

**Success Criteria**:
- RelevanceFilter recall improves to 75%+
- Classification accuracy improves to 80%+
- Scoring correlation improves to ρ > 0.8
- False positive rate <20%

**Tasks**:
- [ ] Build labeling CLI tool
- [ ] Collect and version training data (JSON)
- [ ] Implement BootstrapFewShot optimization
- [ ] Run optimization for each signature
- [ ] Validate on holdout set
- [ ] Document performance metrics

---

### Phase 4: Query & Reporting (Week 6)
**Goal**: RAG-powered CLI and daily reporting

**Deliverables**:
1. **RAG Module**
   - Implement NewsletterRAGModule (Section 5.2)
   - Custom ChromaDB retriever
   - ReAct agent with tools

2. **CLI Interface**
   - Query command (Section 5.3 example)
   - Interactive mode (follow-up questions)
   - Export results to markdown

3. **Daily Report**
   - Automated daily digest email
   - Top 20-50 articles ranked by score
   - Grouped by region/topic
   - Summary statistics

**Success Criteria**:
- Query response time <5 seconds
- Retrieval precision 70%+ (returns relevant articles)
- Daily report accepted by stakeholders
- CLI usable by non-technical staff

**Tasks**:
- [ ] Implement RAG module with dspy.ReAct
- [ ] Create CLI with click library
- [ ] Build daily report generator (HTML email)
- [ ] User acceptance testing
- [ ] Documentation (user guide + API docs)

---

## 7. Risks and Mitigations

### 7.1 Technical Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| **API Cost Overrun** (LLM calls exceed budget) | HIGH | HIGH | • Aggressive keyword pre-filtering (Stage 0)<br>• Batch API calls (OpenAI Batch API)<br>• Cache embeddings aggressively<br>• Use cheaper models for filter (GPT-3.5-turbo) |
| **Low Recall on Niche Topics** (miss relevant articles) | MEDIUM | HIGH | • Hybrid keyword + semantic scoring<br>• Over-inclusive filtering (human review catches FPs)<br>• Weekly review of missed articles to update keywords |
| **Classification Accuracy Insufficient** | MEDIUM | MEDIUM | • Multi-stage classification (coarse → fine)<br>• Ensemble approach (keyword + LLM voting)<br>• Upgrade to MIPROv2 with more training data |
| **Deduplication False Positives** (merge distinct articles) | LOW | MEDIUM | • Conservative similarity threshold (0.85+)<br>• Manual review of cluster edges<br>• Preserve all duplicates in storage, mark primary |
| **Source Instability** (RSS feeds change/break) | MEDIUM | LOW | • Per-source health monitoring<br>• Graceful degradation (log failures, continue)<br>• Fallback to scraping for critical sources |

### 7.2 Data Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| **Multilingual Content** (non-English articles) | HIGH | MEDIUM | • Phase 1: English-only filter<br>• Phase 2: Detect language, translate titles<br>• Phase 3: Full translation for high-value content |
| **Content Behind Paywalls** | MEDIUM | LOW | • Extract title + abstract only<br>• Flag for manual review<br>• Track conversion rate (how many paywalls) |
| **Dynamic Content/JavaScript** | MEDIUM | MEDIUM | • Use headless browser (Playwright) for Tier 3 scraping<br>• RSS feeds avoid this issue<br>• Fallback to static scraping where possible |
| **Regional Legislation Naming** (GDPR vs country-specific) | LOW | LOW | • Taxonomy includes both general and specific terms<br>• Classification examples cover edge cases<br>• Use LLM's knowledge of legal frameworks |

### 7.3 Operational Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| **Training Data Bias** (labels reflect current priorities) | MEDIUM | MEDIUM | • Quarterly re-labeling and re-optimization<br>• Track shifting priorities in domain<br>• Diverse labelers (multiple stakeholders) |
| **Model Drift** (LLM API changes) | LOW | HIGH | • Pin API versions (OpenAI model snapshots)<br>• Monitor output quality metrics daily<br>• Alert on sudden performance changes |
| **Scaling Beyond PoC** (500 → 5000 articles/day) | LOW | MEDIUM | • Architecture supports horizontal scaling<br>• Replace ChromaDB with Qdrant/Weaviate<br>• Use streaming processing (Apache Beam) |
| **User Adoption** (CLI too complex) | MEDIUM | MEDIUM | • User testing in Phase 4<br>• Web UI as Phase 5 extension<br>• Chatbot interface option |

### 7.4 Contingency Plans

**If API costs exceed budget**:
1. Reduce filter to top 20 sources only
2. Switch to Anthropic Claude Haiku (cheaper)
3. Increase keyword filtering aggressiveness
4. Move to weekly batch instead of daily

**If recall <70% after optimization**:
1. Expand training data to 200+ examples
2. Use MIPROv2 optimizer
3. Add human-in-the-loop for borderline cases
4. Ensemble multiple classifiers

**If ChromaDB performance insufficient**:
1. Migrate to Qdrant (better performance)
2. Implement query caching layer
3. Use ColBERTv2 for retrieval
4. Pre-compute common query results

---

## 8. Success Metrics & Monitoring

### 8.1 Key Performance Indicators

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Recall** | 70%+ | Weekly manual review of 50 articles vs system output |
| **Precision** | 60%+ | Human rating of daily top-50 list (relevant vs not) |
| **Daily Output Volume** | 20-50 articles | Automatic tracking of score >0.6 articles |
| **Query Response Time** | <5 sec | Logging + percentile analysis |
| **API Cost** | <$10/day | OpenAI/Anthropic billing API monitoring |
| **Classification Accuracy** | 75%+ | Confusion matrix on labeled validation set |

### 8.2 Monitoring Dashboard

```python
# Simplified monitoring schema
{
    "date": "2026-01-12",
    "ingestion": {
        "articles_fetched": 523,
        "sources_failed": 2,
        "dedup_clusters": 45
    },
    "filtering": {
        "passed_filter": 210,
        "filter_rate": 0.40
    },
    "classification": {
        "by_region": {"asia_pacific": 45, "europe": 38, ...},
        "by_topic": {"regulatory": 67, "criminal_background": 52, ...}
    },
    "scoring": {
        "avg_score": 0.58,
        "high_confidence": 48,  # score >0.7
        "borderline": 22        # 0.5-0.7
    },
    "api_usage": {
        "total_calls": 735,
        "cost_usd": 4.23,
        "avg_latency_ms": 1250
    },
    "query_stats": {
        "queries_executed": 12,
        "avg_response_time_sec": 3.8
    }
}
```

### 8.3 Alerting Rules

- **Alert**: API cost >$15/day → Email admin + pause ingestion
- **Alert**: Recall drops <60% on validation set → Review pipeline
- **Alert**: Source failure >5 sources → Check configurations
- **Alert**: Query latency >10sec → Investigate retrieval bottleneck

---

## Appendix A: Technology Stack Summary

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Language** | Python 3.11+ | DSPy ecosystem, ML/NLP libraries |
| **LLM Framework** | DSPy 2.x | Declarative LLM programming, optimizers |
| **LLM APIs** | OpenAI GPT-4o-mini, GPT-3.5-turbo | Cost-effective, reliable |
| **Embeddings** | OpenAI text-embedding-3-small | Good quality, low cost ($0.02/1M tokens) |
| **Vector DB** | ChromaDB | Simple PoC setup, easy migration path |
| **Storage** | JSON files (PoC), PostgreSQL (production) | Simple for PoC, scalable later |
| **Orchestration** | Cron (PoC), Airflow (production) | Daily batch scheduling |
| **CLI** | Click library | User-friendly command-line interface |
| **RSS Parsing** | feedparser | Standard Python RSS library |
| **Web Scraping** | BeautifulSoup + Playwright | Static + dynamic content |
| **Testing** | pytest, DSPy evaluate | Unit tests + LLM evaluation |

---

## Appendix B: Example End-to-End Flow

**Scenario**: Process daily batch on 2026-01-12

```
06:00 - INGESTION
  ├─ Fetch RSS feeds (247 sources)
  ├─ Parse 523 articles
  ├─ Store raw JSON: /data/raw/2026-01-12/
  └─ Time: 10 minutes

06:10 - DEDUPLICATION
  ├─ Hash-based exact duplicate removal (15 removed)
  ├─ Semantic clustering (0.85 threshold, 30 clusters)
  ├─ Keep primary from each cluster
  └─ Remaining: 478 unique articles

06:15 - RELEVANCE FILTER (Stage 1)
  ├─ Keyword pre-filter (60% pass = 287 articles)
  ├─ DSPy RelevanceFilter on 287 articles
  │  ├─ API calls: 287 × GPT-3.5-turbo
  │  ├─ Cost: $0.50
  │  └─ Time: 5 minutes (batched)
  ├─ Keep: 210 articles (POSSIBLY_RELEVANT or higher)
  └─ Reject: 77 articles

06:20 - CLASSIFICATION
  ├─ DSPy ArticleClassifier (TypedPredictor)
  │  ├─ API calls: 210 × GPT-4o-mini
  │  ├─ Cost: $1.20
  │  └─ Time: 8 minutes
  └─ Output: Region + Topics for each article

06:28 - RELEVANCE SCORING
  ├─ DSPy RelevanceScorer (ChainOfThought)
  │  ├─ API calls: 210 × GPT-4o-mini
  │  ├─ Cost: $1.80
  │  └─ Time: 12 minutes
  └─ Output: Scores 0.0-1.0 + reasoning

06:40 - EMBEDDING & STORAGE
  ├─ Generate embeddings (OpenAI)
  │  ├─ Cost: $0.15
  │  └─ Time: 3 minutes
  ├─ Store in ChromaDB with metadata
  └─ Save scored_articles.json

06:43 - DAILY REPORT GENERATION
  ├─ Filter: score ≥0.6 → 48 articles
  ├─ Rank by score (descending)
  ├─ Group by region, topic
  ├─ Generate HTML email with summaries
  └─ Send to stakeholders

TOTAL:
  ├─ Time: 43 minutes
  ├─ Cost: $3.65
  ├─ Output: 48 high-quality candidate articles
  └─ Success: Within budget, <1 hour, meets volume target
```

---

## Appendix C: Migration Path Beyond PoC

**When to scale**:
- Ingestion >1000 articles/day
- User base >10 people
- Query volume >100/day
- Need for real-time updates

**Upgrade Path**:
1. **Storage**: JSON → PostgreSQL (metadata) + Qdrant (vectors)
2. **Vector DB**: ChromaDB → Qdrant/Weaviate (better performance, filtering)
3. **Retrieval**: Basic similarity → ColBERTv2 (contextual retrieval)
4. **Orchestration**: Cron → Apache Airflow (dependency management, monitoring)
5. **API**: CLI → FastAPI REST API + React frontend
6. **Optimization**: BootstrapFewShot → MIPROv2 (better prompt optimization)
7. **LLM**: OpenAI → Fine-tuned models (lower cost, better quality)

**Cost projections** (production scale):
- 1000 articles/day × 30 days = 30K articles/month
- Estimated cost: $300-500/month (LLM APIs + embeddings + vector DB hosting)
- ROI: 80+ hours/month manual labor saved = $2000+ value

---

*End of Solution Design Document*

**Next Steps for Project Manager**:
1. Review phased implementation plan for resourcing
2. Allocate time for training data labeling (12 hours Week 3)
3. Set up development environment (Python 3.11, DSPy, ChromaDB)
4. Schedule weekly checkpoint meetings during each phase
5. Identify domain expert for validation and labeling

**Handoff to Development Team**:
- This document serves as technical blueprint
- Each phase has clear deliverables and tasks
- Code examples provide implementation guidance
- Risks and mitigations inform development decisions
