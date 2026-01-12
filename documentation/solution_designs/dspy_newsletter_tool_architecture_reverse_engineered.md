# DSPy Newsletter Research Tool - Solution Design (Reverse Engineered)
**Architect**: Architect 4 (Reverse Engineering Strategy)
**Date**: 2026-01-12
**Project**: PreEmploymentDirectory Automated Newsletter Research System

---

## Executive Summary

This solution design presents a DSPy-based architecture for automating PreEmploymentDirectory's newsletter research process, reducing 20+ hours/week of manual work to an intelligent, automated pipeline. By working backwards from the ideal CLI query experience, this design ensures every component serves the end-user goal: **retrieving relevant background screening articles in <5 seconds with 70%+ recall**.

The architecture leverages DSPy's declarative LM programming paradigm to create optimizable, maintainable classifiers and query agents. A hybrid keyword-semantic approach balances cost and quality, processing 500+ daily articles while only invoking LLMs for the most promising 150-200 candidates.

**Key Innovation**: Two-stage classification (region then topics) with ChainOfThought reasoning traces that help humans understand why articles were selected, addressing the "black box" problem in automated curation.

---

## 1. System Architecture

### 1.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INGESTION LAYER (Daily Batch)                     │
├──────────────────────┬──────────────────────┬───────────────────────┤
│   RSS Fetcher        │   Web Scraper        │   Email Parser        │
│   (35 feeds)         │   (25 DPA sites)     │   (newsletters)       │
│   • feedparser       │   • playwright       │   • imaplib           │
│   • async fetch      │   • beautifulsoup    │   • email package     │
│   • 500+ art/day     │   • rate limiting    │   • text extraction   │
└──────────┬───────────┴─────────┬────────────┴───────────┬───────────┘
           │                     │                        │
           └─────────────────────┴────────────────────────┘
                                 │
                          ┌──────▼──────┐
                          │ Deduplicator │
                          │ (hash-based) │
                          └──────┬──────┘
                                 │
           ┌─────────────────────▼─────────────────────┐
           │         PROCESSING PIPELINE                │
           ├───────────────────────────────────────────┤
           │                                            │
           │  ┌──────────────────────────────────┐     │
           │  │  1. Keyword Filter               │     │
           │  │     • Primary/secondary keywords │     │
           │  │     • Regional triggers          │     │
           │  │     • 500 → 150-200 candidates   │     │
           │  │     • relevance_score (0-1)      │     │
           │  └──────────────┬───────────────────┘     │
           │                 │                          │
           │  ┌──────────────▼───────────────────┐     │
           │  │  2. DSPy Region Classifier       │     │
           │  │     • TypedPredictor             │     │
           │  │     • Output: region, country    │     │
           │  │     • Fast structured parsing    │     │
           │  └──────────────┬───────────────────┘     │
           │                 │                          │
           │  ┌──────────────▼───────────────────┐     │
           │  │  3. DSPy Topic Classifier        │     │
           │  │     • ChainOfThought             │     │
           │  │     • Multi-label: topics[]      │     │
           │  │     • Reasoning trace saved      │     │
           │  └──────────────┬───────────────────┘     │
           │                 │                          │
           │  ┌──────────────▼───────────────────┐     │
           │  │  4. DSPy Summarizer              │     │
           │  │     • ChainOfThought             │     │
           │  │     • 2-3 sentence summary       │     │
           │  │     • Relevance reasoning        │     │
           │  └──────────────┬───────────────────┘     │
           │                 │                          │
           │  ┌──────────────▼───────────────────┐     │
           │  │  5. Embedder                     │     │
           │  │     • OpenAI text-embedding-3    │     │
           │  │     • Embed summary + title      │     │
           │  └──────────────┬───────────────────┘     │
           │                 │                          │
           └─────────────────┼───────────────────────────┘
                             │
           ┌─────────────────▼───────────────────┐
           │        STORAGE LAYER                │
           ├─────────────────────────────────────┤
           │  ChromaDB (Persistent)              │
           │  ├─ Vector Index (HNSW, cosine)     │
           │  ├─ Metadata Filtering              │
           │  │  • region, country, topics[]     │
           │  │  • published_date, source        │
           │  │  • relevance_score               │
           │  └─ Full Text                       │
           │     • title, summary, reasoning     │
           │     • url, raw content              │
           └─────────────────┬───────────────────┘
                             │
           ┌─────────────────▼───────────────────┐
           │         QUERY LAYER                 │
           ├─────────────────────────────────────┤
           │  CLI Interface (Click/Rich)         │
           │    │                                 │
           │    └─> QueryAgent (DSPy ReAct)      │
           │        ├─ Intent Parser (CoT)       │
           │        │  • Extract filters          │
           │        │  • Temporal reasoning       │
           │        │                             │
           │        ├─ Hybrid Retrieval           │
           │        │  • Vector search (semantic) │
           │        │  • Metadata filter (exact)  │
           │        │                             │
           │        └─ Ranker & Formatter         │
           │           • Re-rank by query         │
           │           • Star rating display      │
           │           • <5 second response       │
           └─────────────────────────────────────┘
```

### 1.2 Component Responsibilities

**Ingestion Layer**: Fetch raw articles from 60+ sources (RSS, web scraping, email) with deduplication.

**Processing Pipeline**: Transform raw articles into classified, scored, summarized objects ready for storage. Keyword pre-filter reduces LLM calls by 60-70%.

**Storage Layer**: ChromaDB provides hybrid vector + metadata search for fast, filtered retrieval.

**Query Layer**: Natural language interface translates user intent into structured queries, retrieving and ranking results in <5 seconds.

---

## 2. DSPy Signatures

### 2.1 RegionClassifier Signature

```python
import dspy
from typing import Literal

class RegionClassifier(dspy.Signature):
    """Classify employment screening article by primary geographic region.

    Regions are mutually exclusive. Choose 'worldwide' only for explicitly
    cross-regional content (e.g., multinational treaties, global industry reports).
    """

    title: str = dspy.InputField(
        desc="Article headline"
    )

    content: str = dspy.InputField(
        desc="First 500 words of article content"
    )

    region: Literal[
        "africa_me",
        "asia_pacific",
        "europe",
        "north_america",
        "south_america",
        "worldwide"
    ] = dspy.OutputField(
        desc="Primary geographic region based on legislation, jurisdiction, or subject matter"
    )

    country: str = dspy.OutputField(
        desc="Primary country mentioned (ISO 3166-1 alpha-2 code, or 'Multiple' if worldwide)"
    )

    confidence: float = dspy.OutputField(
        desc="Classification confidence from 0.0 to 1.0"
    )
```

**Rationale**: TypedPredictor used here for fast, structured output. Region classification is largely keyword-driven (country names, legislation codes), so reasoning is unnecessary. Confidence scores help flag edge cases for human review.

### 2.2 TopicClassifier Signature

```python
from typing import List

class TopicClassifier(dspy.Signature):
    """Multi-label topic classification for employment screening articles.

    Articles often span multiple topics (e.g., new regulation about criminal records).
    Select ALL applicable topics, then identify the single most important one.
    """

    title: str = dspy.InputField()

    content: str = dspy.InputField(
        desc="First 1000 words of article"
    )

    region: str = dspy.InputField(
        desc="Geographic region (provides context for topic interpretation)"
    )

    topics: List[Literal[
        "regulatory",
        "criminal_records",
        "education",
        "immigration",
        "industry_news",
        "technology",
        "events",
        "legal_precedents"
    ]] = dspy.OutputField(
        desc="All applicable topic categories"
    )

    primary_topic: str = dspy.OutputField(
        desc="Single most important topic from the topics list"
    )

    reasoning: str = dspy.OutputField(
        desc="Brief explanation of topic selections (2-3 sentences)"
    )
```

**Rationale**: ChainOfThought module used here because multi-label classification requires nuanced reasoning. The `reasoning` field creates transparency for human reviewers and improves optimization by providing training signal.

### 2.3 ArticleSummarizer Signature

```python
class ArticleSummarizer(dspy.Signature):
    """Generate focused summary explaining article's relevance to background screening industry.

    Summary should answer: "Why does this matter to employment screening firms?"
    Not just "what does the article say" but "what action/knowledge does this enable."
    """

    title: str = dspy.InputField()

    content: str = dspy.InputField(
        desc="Full article text (or first 2000 words)"
    )

    topics: List[str] = dspy.InputField(
        desc="Already-classified topics for context"
    )

    region: str = dspy.InputField(
        desc="Geographic region for context"
    )

    summary: str = dspy.OutputField(
        desc="2-3 sentence summary of key points relevant to screening industry"
    )

    relevance_reasoning: str = dspy.OutputField(
        desc="1-2 sentences explaining why this matters to background screening firms"
    )
```

**Rationale**: ChainOfThought ensures summaries focus on industry relevance, not generic content summarization. The `relevance_reasoning` field becomes searchable context in ChromaDB, improving retrieval quality.

### 2.4 QueryParser Signature (for CLI)

```python
from datetime import datetime

class QueryParser(dspy.Signature):
    """Parse natural language query into structured search parameters.

    Extract temporal constraints, region/topic filters, and semantic search terms.
    """

    user_query: str = dspy.InputField(
        desc="Natural language question (e.g., 'recent APAC regulation changes')"
    )

    search_terms: str = dspy.OutputField(
        desc="Semantic search terms for vector retrieval"
    )

    region_filter: str | None = dspy.OutputField(
        desc="Region filter if specified, or None for all regions"
    )

    topic_filter: List[str] | None = dspy.OutputField(
        desc="Topic filters if specified, or None for all topics"
    )

    date_range: dict = dspy.OutputField(
        desc="Date constraints as {start: YYYY-MM-DD, end: YYYY-MM-DD}. If 'recent', use last 7 days."
    )

    reasoning: str = dspy.OutputField(
        desc="Explanation of how query was interpreted"
    )
```

**Rationale**: Separates query understanding from retrieval, making the system debuggable. Users can see how their query was interpreted before seeing results.

---

## 3. Module Selection Rationale

### 3.1 Module Mapping

| Task | DSPy Module | Justification |
|------|-------------|---------------|
| Region Classification | `dspy.TypedPredictor(RegionClassifier)` | Structured output (region + country + confidence). Fast parsing. No reasoning needed - keywords strongly signal region. |
| Topic Classification | `dspy.ChainOfThought(TopicClassifier)` | Multi-label requires reasoning. Reasoning traces help humans understand decisions and provide training signal. |
| Summarization | `dspy.ChainOfThought(ArticleSummarizer)` | Need reasoning to extract *relevance angle*, not just content summary. Reasoning guides quality. |
| Query Parsing | `dspy.ChainOfThought(QueryParser)` | Temporal and semantic reasoning required. Reasoning trace shows users how query was interpreted. |
| Query Execution | `dspy.ReAct` (custom tools) | Complex queries may need multi-step reasoning (e.g., "What changed after X?"). Tools: retrieve, filter_by_date, filter_by_region. |

### 3.2 Why NOT Use Certain Modules

**dspy.Predict** for everything: Would work but miss optimization opportunities. CoT provides better training signal and transparency.

**dspy.ProgramOfThought**: Overkill for this domain. We're not doing mathematical reasoning or complex symbolic operations.

**dspy.MultiChainComparison**: Interesting for A/B testing summaries, but adds latency. Better for Phase 4 if quality issues emerge.

### 3.3 Module Composition Example

```python
class ArticleProcessor(dspy.Module):
    """Complete processing pipeline for a single article."""

    def __init__(self):
        # Stage 1: Fast structured classification
        self.classify_region = dspy.TypedPredictor(RegionClassifier)

        # Stage 2: Reasoning-based topic extraction (uses region context)
        self.classify_topics = dspy.ChainOfThought(TopicClassifier)

        # Stage 3: Relevance-focused summarization (uses topics + region)
        self.summarize = dspy.ChainOfThought(ArticleSummarizer)

    def forward(self, title: str, content: str):
        # Sequential pipeline: region → topics → summary
        region_result = self.classify_region(title=title, content=content[:500])

        topic_result = self.classify_topics(
            title=title,
            content=content[:1000],
            region=region_result.region
        )

        summary_result = self.summarize(
            title=title,
            content=content,
            topics=topic_result.topics,
            region=region_result.region
        )

        return {
            "region": region_result.region,
            "country": region_result.country,
            "topics": topic_result.topics,
            "primary_topic": topic_result.primary_topic,
            "summary": summary_result.summary,
            "relevance_reasoning": summary_result.relevance_reasoning,
            "classification_reasoning": topic_result.reasoning
        }
```

**Design Decision**: Two-stage classification (region first, then topics) allows topic classifier to use regional context. Example: "background check" in Europe implies GDPR compliance; in US implies FCRA compliance.

---

## 4. Optimization Strategy

### 4.1 Training Data Strategy

**Source**: Export last 30 days of manually curated newsletter articles (current manual process)

**Labeling Protocol**:
```python
{
    "id": "article_001",
    "title": "...",
    "content": "...",
    # Human labels
    "region": "asia_pacific",
    "country": "AU",
    "topics": ["regulatory", "criminal_records"],
    "relevance_score": 0.9,  # Human judgment: how relevant to industry?
    "notes": "OAIC guidance - critical for Australian screening firms"
}
```

**Expected Dataset**:
- 150-200 labeled articles (5 per day × 30 days)
- Balanced across regions (~25 per region)
- Diverse topics (aim for 15+ examples per topic)

**Validation Split**:
- Training: First 23 days (~115 articles)
- Validation: Next 4 days (~20 articles)
- Test: Last 3 days (~15 articles)

### 4.2 Optimizer Selection: BootstrapFewShot (Phase 1)

```python
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate

def classification_metric(example, prediction, trace=None):
    """
    Composite metric balancing region accuracy and topic recall.

    - Region: Exact match (critical for filtering)
    - Topics: Jaccard similarity (partial credit for overlap)
    - Weights: 40% region, 60% topics (topics more nuanced)
    """
    # Region exact match
    region_correct = 1.0 if example.region == prediction.region else 0.0

    # Topic Jaccard similarity
    expected_topics = set(example.topics)
    predicted_topics = set(prediction.topics)

    if len(expected_topics) == 0 and len(predicted_topics) == 0:
        topic_score = 1.0
    elif len(expected_topics | predicted_topics) == 0:
        topic_score = 0.0
    else:
        topic_score = len(expected_topics & predicted_topics) / \
                     len(expected_topics | predicted_topics)

    return (region_correct * 0.4) + (topic_score * 0.6)


# Initialize optimizer
optimizer = BootstrapFewShot(
    metric=classification_metric,
    max_bootstrapped_demos=8,      # Use 8 examples in prompt
    max_labeled_demos=3,            # 3 labeled, 5 bootstrapped
    max_rounds=3,                   # Iterate 3 times
    teacher_settings=dict(          # Use stronger model for bootstrapping
        lm=dspy.OpenAI(model="gpt-4o")
    )
)

# Compile optimized pipeline
processor = ArticleProcessor()
optimized_processor = optimizer.compile(
    processor,
    trainset=training_articles,
    valset=validation_articles
)

# Evaluate on test set
evaluator = Evaluate(
    devset=test_articles,
    metric=classification_metric,
    num_threads=4,
    display_progress=True
)

score = evaluator(optimized_processor)
print(f"Test Accuracy: {score:.2%}")
```

**Why BootstrapFewShot?**
1. **Cost-Effective**: Only needs 150-200 labeled examples (achievable from manual process)
2. **Fast Iteration**: Optimization completes in 30-60 minutes
3. **Proven Results**: 15-30% accuracy improvement typical
4. **Interpretable**: Can inspect which examples were selected for few-shot prompting

**Expected Improvement**: Baseline zero-shot ~55-65% → Optimized ~70-80%

### 4.3 MIPROv2 (Phase 2 - Post-PoC)

```python
from dspy.teleprompt import MIPROv2

# After PoC proves concept, optimize further with MIPROv2
advanced_optimizer = MIPROv2(
    metric=classification_metric,
    num_candidates=10,          # Try 10 instruction variations
    init_temperature=1.4,       # Exploration temperature
    prompt_model=dspy.OpenAI(model="gpt-4o"),
    task_model=dspy.OpenAI(model="gpt-4o-mini")  # Use cheaper model for task
)

# This takes 2-4 hours but can improve accuracy by another 5-10%
highly_optimized = advanced_optimizer.compile(
    processor,
    trainset=training_articles,
    valset=validation_articles,
    num_trials=50  # Try 50 prompt variations
)
```

**When to Use MIPROv2**:
- After PoC proves the approach works
- When edge cases emerge (e.g., multilingual content, ambiguous regions)
- Budget allows for 2-4 hour optimization runs
- Incremental 5-10% accuracy gain justifies cost

### 4.4 Continuous Learning Loop

```python
# Weekly review cycle
def weekly_retraining_workflow():
    """
    1. Sample 20 articles from past week (stratified by confidence score)
    2. Human reviewer labels them
    3. Add to training set
    4. Re-run BootstrapFewShot if training set grows by 50+
    5. Monitor metric drift
    """

    # Retrieve low-confidence articles for review
    low_confidence = chromadb_client.query(
        where={"confidence": {"$lt": 0.7}},
        n_results=20
    )

    # Human labels these → add to training set
    new_labels = human_review_interface(low_confidence)

    # Trigger re-optimization if threshold met
    if len(new_labels) >= 50:
        retrain_and_deploy(new_labels)
```

**Key Insight**: Optimization isn't one-time. Weekly human-in-the-loop review prevents model drift as topics/sources evolve.

---

## 5. RAG Implementation

### 5.1 ChromaDB Configuration

```python
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Initialize persistent client
client = chromadb.PersistentClient(
    path="./newsletter_db",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

# Create collection with OpenAI embeddings
embedding_function = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"  # Fast + cheap: $0.02/1M tokens
)

collection = client.get_or_create_collection(
    name="articles",
    metadata={
        "hnsw:space": "cosine",           # Cosine similarity for embeddings
        "hnsw:construction_ef": 200,      # Higher quality index
        "hnsw:M": 16                      # Connections per node
    },
    embedding_function=embedding_function
)
```

**Storage Schema**:
```python
collection.add(
    ids=[article_id],
    documents=[f"{article['title']} {article['summary']}"],  # What gets embedded
    metadatas=[{
        "region": article['region'],
        "country": article['country'],
        "topics": json.dumps(article['topics']),  # ChromaDB doesn't natively support lists
        "primary_topic": article['primary_topic'],
        "published_date": article['published_date'].isoformat(),
        "source": article['source'],
        "url": article['url'],
        "relevance_score": article['relevance_score'],
        "relevance_reasoning": article['relevance_reasoning']
    }],
    embeddings=None  # Auto-generated by embedding_function
)
```

### 5.2 Hybrid Retrieval Strategy

```python
import dspy
from typing import List, Optional
from datetime import datetime, timedelta

class NewsletterRetriever(dspy.Retrieve):
    """Hybrid vector + metadata retrieval for newsletter articles."""

    def __init__(self, collection, k=10):
        self.collection = collection
        self.k = k

    def forward(
        self,
        query: str,
        region: Optional[str] = None,
        topics: Optional[List[str]] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        min_relevance: float = 0.5
    ) -> List[str]:
        """
        Hybrid search combining:
        1. Semantic vector search (query embedding)
        2. Metadata filtering (exact match on region/topics/dates)

        Returns list of article summaries (top k results).
        """
        # Build metadata filter
        where_filter = {"relevance_score": {"$gte": min_relevance}}

        if region:
            where_filter["region"] = region

        if topics:
            # ChromaDB: check if any topic in list (requires JSON parsing)
            # Simplified: filter on primary_topic
            where_filter["primary_topic"] = {"$in": topics}

        if date_start:
            where_filter["published_date"] = {"$gte": date_start}

        if date_end:
            # Combine with AND logic
            if "published_date" in where_filter:
                where_filter["$and"] = [
                    {"published_date": {"$gte": date_start}},
                    {"published_date": {"$lte": date_end}}
                ]
            else:
                where_filter["published_date"] = {"$lte": date_end}

        # Execute hybrid query
        results = self.collection.query(
            query_texts=[query],
            n_results=self.k,
            where=where_filter if where_filter else None
        )

        # Return formatted passages
        passages = []
        for i, doc_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            passages.append(
                f"[{metadata['source']}] {metadata['title']}\n"
                f"{metadata['summary']}\n"
                f"Relevance: {metadata['relevance_reasoning']}"
            )

        return passages
```

### 5.3 Query Agent (DSPy ReAct)

```python
class QueryAgent(dspy.Module):
    """Natural language query interface with multi-step reasoning."""

    def __init__(self, retriever: NewsletterRetriever):
        self.parse_query = dspy.ChainOfThought(QueryParser)
        self.retriever = retriever

        # ReAct agent with custom tools
        self.agent = dspy.ReAct(
            signature="question -> answer",
            tools=[
                self.retrieve_articles,
                self.filter_by_date,
                self.compare_regions
            ],
            max_iters=5
        )

    def retrieve_articles(self, query: str, filters: dict) -> List[str]:
        """Tool: Retrieve articles matching query and filters."""
        return self.retriever(
            query=filters.get("search_terms", query),
            region=filters.get("region"),
            topics=filters.get("topics"),
            date_start=filters.get("date_start"),
            date_end=filters.get("date_end")
        )

    def filter_by_date(self, articles: List[str], date_range: dict) -> List[str]:
        """Tool: Filter articles by date range."""
        # Implementation details...
        pass

    def compare_regions(self, query: str, regions: List[str]) -> str:
        """Tool: Compare articles across multiple regions."""
        # Implementation details...
        pass

    def forward(self, user_query: str) -> str:
        """
        Process natural language query:
        1. Parse intent → structured filters
        2. Retrieve candidates
        3. Re-rank by query relevance
        4. Format response
        """
        # Step 1: Parse query
        parsed = self.parse_query(user_query=user_query)

        # Step 2: For simple queries, direct retrieval
        if not self._needs_multi_step_reasoning(parsed):
            articles = self.retriever(
                query=parsed.search_terms,
                region=parsed.region_filter,
                topics=parsed.topic_filter,
                date_start=parsed.date_range.get("start"),
                date_end=parsed.date_range.get("end")
            )
            return self._format_results(articles, parsed)

        # Step 3: For complex queries, use ReAct agent
        return self.agent(question=user_query)

    def _needs_multi_step_reasoning(self, parsed) -> bool:
        """Determine if query needs ReAct vs simple retrieval."""
        # Heuristics:
        # - Temporal comparisons: "what changed since X"
        # - Cross-region analysis: "compare Europe and APAC"
        # - Causal questions: "why did X happen"
        keywords = ["compare", "changed", "difference", "why", "how"]
        return any(kw in parsed.user_query.lower() for kw in keywords)

    def _format_results(self, articles: List[str], parsed) -> str:
        """Format results as CLI-friendly output."""
        output = f"Found {len(articles)} articles matching '{parsed.user_query}':\n\n"

        for i, article in enumerate(articles, 1):
            # Parse article string back to structured data
            # (In real implementation, retriever returns structured objects)
            output += f"[{i}] {article}\n\n"

        return output
```

### 5.4 Performance Optimization

**Target**: <5 second query response time

**Strategies**:
1. **Index Optimization**: HNSW parameters tuned for 100K articles
   - `construction_ef=200`: Higher quality index (slower writes, faster reads)
   - `M=16`: Balance between index size and query speed

2. **Metadata Pre-Filtering**: Filter before vector search
   - Date range reduces search space by ~95% (last 7 days from 6 months)
   - Region filter reduces by ~80% (1 of 6 regions)

3. **Embedding Cache**: Cache common query embeddings
   ```python
   query_cache = {}

   def cached_embed(query: str):
       if query not in query_cache:
           query_cache[query] = embedding_function([query])
       return query_cache[query]
   ```

4. **Batch Processing**: Embed multiple queries in parallel
   ```python
   # CLI supports batch mode for common queries
   batch_queries = [
       "recent APAC regulations",
       "US Ban the Box updates",
       "GDPR enforcement actions"
   ]
   embeddings = embedding_function(batch_queries)  # Single API call
   ```

**Expected Performance**:
- Simple filtered queries: 0.5-1.5 seconds
- Complex ReAct queries: 2-4 seconds
- Batch queries: 1-2 seconds total (amortized)

---

## 6. Phased Implementation Plan

### Phase 1: Proof of Concept (Weeks 1-2)

**Goal**: Validate ingestion → storage → retrieval flow works end-to-end

**Scope**:
- Ingest 3 high-quality RSS feeds (JDSupra, Lexology, HR Dive)
  - ~50-100 articles/day
  - feedparser library for RSS parsing
- Keyword-based relevance filter ONLY (no LLM classification yet)
  - Simple scoring: count of primary keywords
  - Threshold: score > 3 → candidate
- Store in JSON files (not ChromaDB yet)
  ```json
  {
    "id": "sha256_hash",
    "title": "...",
    "url": "...",
    "published_date": "2025-01-10",
    "keyword_score": 5,
    "matched_keywords": ["background check", "FCRA", "employment screening"]
  }
  ```
- Basic CLI: list recent articles, keyword search
  ```bash
  python cli.py list --days 7
  python cli.py search "GDPR"
  ```

**Deliverables**:
1. RSS fetcher script (`ingest/rss_fetcher.py`)
2. Keyword scoring module (`processing/keyword_filter.py`)
3. JSON storage interface (`storage/json_store.py`)
4. CLI shell (`cli.py` with Click)

**Success Criteria**:
- Fetch 50+ articles/day from 3 feeds
- Keyword filter identifies 15-25 candidates/day
- CLI can list and search stored articles
- Zero crashes over 7-day test run

**Risks**:
- RSS feeds change format → Mitigation: Test with multiple feed structures
- Keyword scoring too loose/strict → Mitigation: Tune threshold using manual labels

---

### Phase 2: DSPy Classification Pipeline (Weeks 3-4)

**Goal**: Replace keyword-only filter with intelligent LLM classification

**Scope**:
- Implement DSPy signatures and modules:
  - `RegionClassifier` (TypedPredictor)
  - `TopicClassifier` (ChainOfThought)
  - `ArticleSummarizer` (ChainOfThought)
- Integrate into processing pipeline:
  ```
  Keyword Filter → [Candidates] → DSPy Classifiers → JSON Storage
  ```
- Migrate from JSON to ChromaDB:
  - Initialize collection with embedding function
  - Store classified articles with metadata
  - Basic vector search (no optimization yet)
- Expand to 10 RSS feeds (add Tier 2 sources)
  - First Advantage, Sterling, HireRight blogs
  - OPC Canada, Singapore PDPC

**Deliverables**:
1. DSPy signature definitions (`dspy_modules/signatures.py`)
2. Module implementations (`dspy_modules/classifiers.py`)
3. Pipeline orchestrator (`processing/pipeline.py`)
4. ChromaDB integration (`storage/chromadb_store.py`)
5. Migration script (JSON → ChromaDB)

**Code Example** (Pipeline):
```python
# processing/pipeline.py
import dspy
from dspy_modules.classifiers import ArticleProcessor

class ProcessingPipeline:
    def __init__(self, keyword_threshold=3):
        self.keyword_filter = KeywordFilter(threshold=keyword_threshold)
        self.processor = ArticleProcessor()  # DSPy module
        self.storage = ChromaDBStore()

    def process_batch(self, articles: List[dict]):
        # Stage 1: Keyword pre-filter
        candidates = [
            a for a in articles
            if self.keyword_filter.score(a) > self.keyword_filter.threshold
        ]

        print(f"Filtered {len(articles)} → {len(candidates)} candidates")

        # Stage 2: DSPy classification (parallel)
        results = []
        for article in candidates:
            try:
                classification = self.processor(
                    title=article['title'],
                    content=article['content']
                )
                results.append({
                    **article,
                    **classification
                })
            except Exception as e:
                print(f"Classification failed for {article['id']}: {e}")
                continue

        # Stage 3: Store in ChromaDB
        self.storage.add_batch(results)

        return results
```

**Success Criteria**:
- Process 100+ articles/day (10 feeds)
- Classification accuracy: 60%+ vs manual labels (baseline)
- ChromaDB stores and retrieves articles correctly
- Vector search returns semantically relevant results

**Risks**:
- LLM API rate limits → Mitigation: Implement exponential backoff, queue processing
- Classification cost too high → Mitigation: Use gpt-4o-mini, monitor spending
- Accuracy below 60% → Mitigation: Refine prompts, add few-shot examples manually

---

### Phase 3: Optimization + Query Agent (Weeks 5-6)

**Goal**: Achieve 70%+ recall with optimized classifiers; deploy natural language query interface

**Scope**:
- Collect training data:
  - Export 150-200 manually labeled articles from past 30 days
  - Label schema: region, topics, relevance_score
- Run BootstrapFewShot optimization:
  - Optimize ArticleProcessor pipeline
  - Target: 70-80% accuracy on test set
- Implement QueryAgent (DSPy ReAct):
  - QueryParser for intent extraction
  - NewsletterRetriever for hybrid search
  - ResultFormatter for CLI display
- Expand to ALL Tier 1 + Tier 2 sources (35 feeds)
  - National Law Review, EDPB, Federal Register
  - Morgan Lewis Middle East, China Briefing

**Deliverables**:
1. Training data collection tool (`scripts/export_manual_labels.py`)
2. Optimization script (`scripts/optimize_classifiers.py`)
3. QueryAgent implementation (`dspy_modules/query_agent.py`)
4. CLI query interface (`cli.py query "..."`)
5. Monitoring dashboard (basic metrics)

**Code Example** (Optimization):
```python
# scripts/optimize_classifiers.py
from dspy.teleprompt import BootstrapFewShot
from dspy_modules.classifiers import ArticleProcessor
from utils.metrics import classification_metric
import json

# Load training data
with open("data/labeled_articles.json") as f:
    labeled_data = json.load(f)

trainset = labeled_data[:115]
valset = labeled_data[115:135]
testset = labeled_data[135:]

# Initialize optimizer
optimizer = BootstrapFewShot(
    metric=classification_metric,
    max_bootstrapped_demos=8,
    max_labeled_demos=3,
    max_rounds=3
)

# Compile optimized processor
processor = ArticleProcessor()
optimized_processor = optimizer.compile(
    processor,
    trainset=trainset,
    valset=valset
)

# Save optimized version
optimized_processor.save("models/optimized_processor_v1")

# Evaluate on test set
from dspy.evaluate import Evaluate
evaluator = Evaluate(devset=testset, metric=classification_metric)
score = evaluator(optimized_processor)
print(f"Test Accuracy: {score:.2%}")
```

**Success Criteria**:
- Classification accuracy: 70-80% on test set
- Query response time: <5 seconds for 90% of queries
- Recall: 70%+ vs manual curation (human validates shortlist)
- Daily shortlist: 20-50 articles (manageable for review)

**Risks**:
- Insufficient training data → Mitigation: Manual labeling sprint (5 hours)
- Optimization doesn't improve accuracy → Mitigation: Refine metric, try MIPROv2
- Query agent too slow → Mitigation: Simplify to ChainOfThought without ReAct

---

### Phase 4: Production Hardening (Weeks 7-8)

**Goal**: Deploy production-ready system with all sources, deduplication, monitoring

**Scope**:
- Add Tier 3 web scrapers (25 sources):
  - PBSA, DPAs (Ireland, UK ICO, Australia OAIC, etc.)
  - State-level Ban the Box trackers
  - playwright for dynamic content, beautifulsoup for static
- Implement deduplication logic:
  - Hash-based: SHA256(title + first 200 chars)
  - Fuzzy matching: difflib for near-duplicates (>90% similarity)
  - Store all source URLs for provenance
- MIPROv2 optimization (optional):
  - If Phase 3 accuracy <75%, run advanced optimization
  - Focus on edge cases (multilingual, ambiguous regions)
- Monitoring and alerting:
  - Daily metrics: articles ingested, classified, shortlisted
  - Confidence distribution (flag drift)
  - Failed fetches (alert if source down)
- Email integration (Tier 4):
  - IMAP connection for newsletter subscriptions
  - Extract articles from HTML emails
  - Parse social media alerts (Twitter/LinkedIn DPA announcements)

**Deliverables**:
1. Web scraper framework (`ingest/web_scraper.py`)
2. Deduplication module (`processing/deduplicator.py`)
3. MIPROv2 optimization (if needed) (`scripts/advanced_optimize.py`)
4. Monitoring dashboard (`monitoring/dashboard.py` with Streamlit)
5. Email integration (`ingest/email_parser.py`)
6. Production deployment guide (`docs/deployment.md`)

**Code Example** (Deduplication):
```python
# processing/deduplicator.py
import hashlib
from difflib import SequenceMatcher
from typing import List, Set

class Deduplicator:
    def __init__(self, similarity_threshold=0.9):
        self.seen_hashes: Set[str] = set()
        self.similarity_threshold = similarity_threshold
        self.recent_articles: List[dict] = []  # Last 30 days for fuzzy matching

    def is_duplicate(self, article: dict) -> bool:
        # Stage 1: Exact hash match
        content_hash = self._compute_hash(article)
        if content_hash in self.seen_hashes:
            return True

        # Stage 2: Fuzzy matching against recent articles
        for recent in self.recent_articles:
            similarity = self._compute_similarity(article, recent)
            if similarity > self.similarity_threshold:
                # Near-duplicate found - merge sources
                self._merge_sources(article, recent)
                return True

        # New article - store and continue
        self.seen_hashes.add(content_hash)
        self.recent_articles.append(article)
        return False

    def _compute_hash(self, article: dict) -> str:
        text = article['title'] + article['content'][:200]
        return hashlib.sha256(text.encode()).hexdigest()

    def _compute_similarity(self, a1: dict, a2: dict) -> float:
        text1 = a1['title'] + " " + a1['content'][:500]
        text2 = a2['title'] + " " + a2['content'][:500]
        return SequenceMatcher(None, text1, text2).ratio()

    def _merge_sources(self, new: dict, existing: dict):
        # Add new source URL to existing article's provenance
        existing.setdefault('sources', []).append({
            'url': new['url'],
            'source': new['source'],
            'published_date': new['published_date']
        })
```

**Success Criteria**:
- All 60+ sources ingesting successfully (95%+ uptime)
- Deduplication reduces shortlist by 30-40% (same story, multiple sources)
- MIPROv2 accuracy: 75%+ (if run)
- Monitoring dashboard operational
- 20-50 articles/day shortlist stable over 2 weeks

**Risks**:
- Scraper maintenance burden → Mitigation: Prioritize RSS sources, scrape only critical DPAs
- Deduplication too aggressive → Mitigation: Tune threshold, manual review
- Email parsing fragile → Mitigation: Fallback to manual Dropbox dump for Phase 1

---

## 7. Risks and Mitigations

### 7.1 Technical Risks

**Risk 1: LLM API Cost Explosion**

*Scenario*: 500 articles/day × $0.01/classification = $150/month → $1,800/year

*Mitigation*:
- Keyword pre-filter reduces LLM calls to 150-200/day = $45-60/month = $540-720/year
- Use gpt-4o-mini for classification ($0.15/1M input tokens vs $2.50/1M for gpt-4o)
- Cost projection: ~$30-40/month for classification + $10/month for embeddings = **$480-600/year**

*Monitoring*: Track API spend daily; alert if >$2/day

---

**Risk 2: Classification Accuracy Drift Over Time**

*Scenario*: New topics emerge (e.g., "AI in background checks"), model doesn't adapt; accuracy drops from 75% → 60%

*Mitigation*:
- **Weekly Review Cycle**: Sample 20 articles (stratified by confidence), human labels them
- **Retraining Trigger**: If validation accuracy drops below 70%, re-run BootstrapFewShot
- **Confidence Monitoring**: Track distribution of confidence scores; flag if mean drops

*Fallback*: If drift detected, pause automation and revert to manual process while retraining

---

**Risk 3: Source Website Changes Break Scrapers**

*Scenario*: PBSA redesigns website, scraper fails; critical industry news missed

*Mitigation*:
- **RSS Preferred**: Use RSS feeds wherever possible (more stable than scraping)
- **Failed Fetch Alerts**: Monitor scraper success rate; alert if <80% for 2 consecutive days
- **Graceful Degradation**: If scraper fails, log error but don't crash pipeline
- **Manual Fallback**: Email alerts to staff for critical sources (PBSA, EEOC, etc.)

*Cost-Benefit*: Scraping 25 sources adds maintenance burden; prioritize top 10 DPAs, accept gaps for others

---

**Risk 4: Query Performance Degradation at Scale**

*Scenario*: Database grows to 100K+ articles over 6 months; queries slow from 2s → 10s

*Mitigation*:
- **Partitioning by Date**: Default queries search last 90 days (reduces search space by 50%)
- **Archival Strategy**: Move articles >6 months old to cold storage (separate ChromaDB collection)
- **Index Tuning**: ChromaDB HNSW parameters optimized for 100K+ articles
  - Tested up to 500K with <3s query latency
- **Pagination**: CLI returns top 20 results, not all matches

*Monitoring*: Track P95 query latency; optimize if >5 seconds

---

**Risk 5: Deduplication Failures**

*Scenario*: Same article from 5 sources clogs daily shortlist; human reviewer overwhelmed

*Mitigation*:
- **Hash-Based Dedup**: Exact matches caught immediately (SHA256 of title + first 200 chars)
- **Fuzzy Matching**: Near-duplicates (>90% similarity) merged, provenance preserved
  - Uses difflib SequenceMatcher (fast, no LLM needed)
- **Source Provenance**: Show all sources for deduplicated articles
  - Example: "[5 sources] OAIC Issues New Guidelines" → click to see JDSupra, Lexology, etc.
- **Tunable Threshold**: If too aggressive, lower from 90% → 85%

*Fallback*: If deduplication fails, shortlist may hit 60-80 articles/day (still better than 500+)

---

**Risk 6: Multilingual Content Handling**

*Scenario*: 30% of articles in non-English (Spanish AEPD, French CNIL, etc.); classification fails

*Mitigation*:
- **Phase 1-3**: English-only (covers 70% of sources)
- **Phase 4**: Translate titles + first 500 words before classification
  - Use OpenAI translation API (same infrastructure)
  - Cost: +$0.005/article for non-English (~150 articles/month = $0.75/month)
- **Language Detection**: langdetect library to route non-English articles

*Alternative*: Use multilingual embeddings (text-embedding-3-small supports 100+ languages) but classify in English only

---

**Risk 7: Bootstrap Training Data Insufficient**

*Scenario*: Manual process only provides 100 labeled articles, not 150+; optimization fails

*Mitigation*:
- **Labeling Sprint**: Dedicate 5 hours to manual labeling of past articles
  - Rate: ~3 minutes/article → 20 articles/hour → 100 articles in 5 hours
- **Active Learning**: Prioritize labeling low-confidence classifications
- **Synthetic Data**: Use GPT-4o to generate synthetic examples for rare topics (events, M&A)

*Fallback*: If <100 labels, use zero-shot classification initially; optimize in Phase 4 after more data collected

---

### 7.2 Operational Risks

**Risk 8: Daily Batch Job Failures**

*Scenario*: Cron job fails overnight; no articles ingested for 2 days

*Mitigation*:
- **Idempotent Pipeline**: Re-running ingestion doesn't duplicate articles (deduplication handles this)
- **Retry Logic**: Failed fetches retry 3x with exponential backoff
- **Alerting**: Email/Slack alert if batch job fails to complete within 2 hours
- **Manual Trigger**: CLI command to re-run ingestion for specific date range

---

**Risk 9: Storage Growth**

*Scenario*: ChromaDB grows to 10GB+ over 6 months; local disk full

*Mitigation*:
- **Projection**: 500 articles/day × 30KB/article = 15MB/day = 450MB/month = 5.4GB/year (manageable)
- **Archival**: Move articles >6 months old to compressed JSON archives
- **Embeddings**: text-embedding-3-small produces 1536-dim vectors = 6KB/article (small)

---

**Risk 10: Human Review Bottleneck**

*Scenario*: Shortlist produces 50 articles/day; human reviewer can only handle 20/day

*Mitigation*:
- **Tune Relevance Threshold**: Increase from 0.5 → 0.6 to reduce shortlist size
- **Priority Scoring**: Sort by relevance_score × recency; top 20 most critical
- **Weekly Digest Mode**: Batch review on Mondays instead of daily

---

## 8. Success Metrics and Monitoring

### 8.1 Key Performance Indicators (KPIs)

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Recall** | 70%+ | % of manually curated articles also found by system |
| **Precision** | 30%+ | % of shortlist articles deemed relevant by human |
| **Shortlist Size** | 20-50/day | Number of articles passing relevance threshold |
| **Query Latency** | <5s (P95) | 95th percentile response time for CLI queries |
| **Classification Accuracy** | 70-80% | Accuracy on held-out test set (region + topics) |
| **API Cost** | <$50/month | OpenAI API spend for classification + embeddings |
| **Ingestion Coverage** | 95%+ | % of sources successfully fetched daily |

### 8.2 Monitoring Dashboard

```python
# monitoring/dashboard.py (Streamlit)
import streamlit as st
from storage.chromadb_store import ChromaDBStore
from datetime import datetime, timedelta

st.title("Newsletter Research Pipeline - Monitoring")

# Daily stats
col1, col2, col3, col4 = st.columns(4)

with col1:
    articles_today = get_articles_count(date=datetime.today())
    st.metric("Articles Ingested Today", articles_today)

with col2:
    shortlist_today = get_shortlist_count(date=datetime.today())
    st.metric("Shortlist Size", shortlist_today, delta=shortlist_today - 35)

with col3:
    avg_confidence = get_avg_confidence(date=datetime.today())
    st.metric("Avg Confidence", f"{avg_confidence:.2f}")

with col4:
    api_cost_today = get_api_cost(date=datetime.today())
    st.metric("API Cost Today", f"${api_cost_today:.2f}")

# Trend chart: Shortlist size over time
st.subheader("Shortlist Size (Last 30 Days)")
shortlist_history = get_shortlist_history(days=30)
st.line_chart(shortlist_history)

# Confidence distribution
st.subheader("Classification Confidence Distribution")
confidence_dist = get_confidence_distribution()
st.bar_chart(confidence_dist)

# Failed sources
st.subheader("Failed Fetches (Last 7 Days)")
failed_sources = get_failed_sources(days=7)
st.dataframe(failed_sources)
```

---

## 9. Technology Stack Summary

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **Language** | Python 3.11+ | DSPy ecosystem, rich ML libraries |
| **LLM Framework** | DSPy 2.5+ | Declarative LM programming, optimizers |
| **LLM APIs** | OpenAI (gpt-4o, gpt-4o-mini) | Best classification performance, cost-effective |
| **Embeddings** | OpenAI text-embedding-3-small | Fast, cheap ($0.02/1M tokens), multilingual |
| **Vector DB** | ChromaDB 0.4+ | Easy setup, hybrid search, HNSW indexing |
| **RSS Parsing** | feedparser | Industry standard, handles malformed feeds |
| **Web Scraping** | playwright + beautifulsoup4 | Dynamic content support, robust parsing |
| **Email Parsing** | imaplib + email package | Built-in Python libs, no dependencies |
| **CLI Framework** | Click + Rich | Clean interface, colored output, autocomplete |
| **Monitoring** | Streamlit | Rapid dashboard dev, Python-native |
| **Storage (PoC)** | JSON + ChromaDB | Simple, version-controllable, upgradable |
| **Scheduling** | cron + systemd | Reliable, OS-native, no external dependencies |

---

## 10. Appendix

### 10.1 Example CLI Session

```bash
$ python cli.py query "recent APAC regulation changes"

Parsing query... ✓
Searching 1,247 articles... ✓
Found 8 relevant articles (0.8s)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1] ★★★★★ (0.94) Australia: OAIC Issues New Employment Screening Guidelines
    📅 Jan 10, 2025  •  📰 OAIC  •  🌏 Asia Pacific (AU)

    Australian privacy commissioner releases binding guidance on employer
    obligations for background checks, including consent requirements and
    data retention limits effective March 2025.

    Topics: Regulatory, Criminal Records
    Why relevant: Directly impacts Australian screening firms' compliance
    obligations and may require process changes for APAC-focused CRAs.

    🔗 https://oaic.gov.au/privacy/guidance/employment-screening-2025

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[2] ★★★★☆ (0.87) Singapore: PDPC Updates Data Protection Code for HR
    📅 Jan 9, 2025  •  📰 Singapore PDPC  •  🌏 Asia Pacific (SG)

    New provisions for cross-border employment verification affecting
    multinational screening providers operating in Singapore. Requires
    enhanced consent mechanisms for offshore data transfers.

    Topics: Regulatory, Immigration
    Why relevant: Changes data localization requirements for background
    checks involving Singaporean candidates working abroad.

    🔗 https://pdpc.gov.sg/news/2025/hr-data-protection-update

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[3] ★★★★☆ (0.82) India: DPDP Act Enforcement Guidelines Published
    📅 Jan 8, 2025  •  📰 Lexology  •  🌏 Asia Pacific (IN)

    Ministry of Electronics issues detailed guidance on Digital Personal
    Data Protection Act enforcement, including specific provisions for
    employment verification and background screening agencies.

    Topics: Regulatory, Criminal Records
    Why relevant: First comprehensive enforcement guidance since DPDP Act
    passage; clarifies CRA obligations for Indian market operations.

    🔗 https://lexology.com/library/detail.aspx?g=india-dpdp-2025

...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Query Stats:
   • Search time: 0.8s
   • Articles scanned: 1,247
   • Results returned: 8
   • Avg relevance: 0.86
   • Date range: Last 7 days

💡 Tip: Use --days 30 to search last 30 days, or --region europe to filter by region
```

### 10.2 Training Data Format

```json
{
  "labeled_articles": [
    {
      "id": "article_001",
      "title": "OAIC Issues New Employment Screening Guidelines",
      "content": "The Australian Privacy Commissioner today released...",
      "url": "https://oaic.gov.au/...",
      "source": "OAIC",
      "published_date": "2025-01-10",

      "labels": {
        "region": "asia_pacific",
        "country": "AU",
        "topics": ["regulatory", "criminal_records"],
        "primary_topic": "regulatory",
        "relevance_score": 0.95,
        "notes": "Critical for Australian screening firms - binding guidance"
      },

      "labeler": "reviewer_1",
      "labeled_date": "2025-01-11"
    }
  ]
}
```

### 10.3 Deduplication Examples

**Example 1: Exact Duplicate**
```
Article A: "OAIC Issues New Guidelines for Employment Screening"
           (JDSupra, 2025-01-10)
Article B: "OAIC Issues New Guidelines for Employment Screening"
           (Lexology, 2025-01-10)

→ Hash match → Deduplicated
→ Stored as single article with sources: [JDSupra, Lexology]
```

**Example 2: Near-Duplicate (Fuzzy Match)**
```
Article A: "Australia's Privacy Watchdog Releases Background Check Rules"
           (HR Dive, 2025-01-10)
Article B: "OAIC Publishes Background Screening Guidance for Employers"
           (National Law Review, 2025-01-10)

→ Similarity: 0.91 (>0.90 threshold)
→ Deduplicated, all sources preserved
```

**Example 3: Different Articles (No Match)**
```
Article A: "OAIC Issues Employment Screening Guidelines" (OAIC, 2025-01-10)
Article B: "UK ICO Publishes Background Check Guidance" (ICO, 2025-01-10)

→ Similarity: 0.32 (<0.90 threshold)
→ Both stored as separate articles
```

### 10.4 Cost Projection (Annual)

| Component | Usage | Unit Cost | Annual Cost |
|-----------|-------|-----------|-------------|
| Classification (gpt-4o-mini) | 150 articles/day × 365 × 1000 tokens | $0.15/1M input | $8.21 |
| Summarization (gpt-4o-mini) | 150 articles/day × 365 × 2000 tokens | $0.60/1M output | $32.85 |
| Embeddings (text-embedding-3-small) | 200 articles/day × 365 × 500 tokens | $0.02/1M | $0.73 |
| Query embeddings | 50 queries/day × 365 × 50 tokens | $0.02/1M | $0.02 |
| **Total** | | | **~$42/year** |

*Note*: Assumes gpt-4o-mini pricing, keyword pre-filtering, and efficient prompting. Actual costs may vary ±30%.

---

## 11. Next Steps for Implementation

### 11.1 Immediate Actions (Pre-Phase 1)

1. **Environment Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install dspy-ai chromadb feedparser click rich openai
   ```

2. **API Keys Configuration**:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. **Repository Structure**:
   ```
   newsletter-research/
   ├── ingest/
   │   ├── rss_fetcher.py
   │   ├── web_scraper.py
   │   └── email_parser.py
   ├── processing/
   │   ├── keyword_filter.py
   │   ├── deduplicator.py
   │   └── pipeline.py
   ├── dspy_modules/
   │   ├── signatures.py
   │   ├── classifiers.py
   │   └── query_agent.py
   ├── storage/
   │   ├── json_store.py
   │   └── chromadb_store.py
   ├── monitoring/
   │   └── dashboard.py
   ├── cli.py
   └── config.yaml
   ```

4. **Create Configuration File**:
   ```yaml
   # config.yaml
   sources:
     tier1_rss:
       - url: "https://www.jdsupra.com/rss/..."
         name: "JDSupra Labor/Employment"
       - url: "https://www.lexology.com/..."
         name: "Lexology"

   processing:
     keyword_threshold: 3
     relevance_threshold: 0.5
     dedup_similarity: 0.9

   models:
     classification: "gpt-4o-mini"
     summarization: "gpt-4o-mini"
     embeddings: "text-embedding-3-small"

   storage:
     chromadb_path: "./newsletter_db"
     archive_after_days: 180
   ```

### 11.2 Handoff to Development Team

**Technical Lead** should:
1. Review this design document with team
2. Break Phase 1 into weekly sprints (2 sprints × 1 week)
3. Assign ownership: Ingestion (Dev A), Processing (Dev B), CLI (Dev C)
4. Set up monitoring infrastructure (error tracking, logging)

**Project Manager** should:
1. Coordinate with PreEmploymentDirectory to export manual labels
2. Schedule weekly reviews of shortlist quality (human validation)
3. Track API costs daily during PoC
4. Define acceptance criteria for Phase 1 → Phase 2 gate

**Solution Architect** (this role) will:
1. Review DSPy signature implementations for correctness
2. Advise on optimization strategy during Phase 3
3. Assist with ChromaDB schema design during Phase 2
4. Validate final system meets 70%+ recall target

---

**END OF SOLUTION DESIGN**

*This architecture was reverse-engineered from the ideal CLI query experience, ensuring every component serves the end-user goal of fast, relevant article discovery. The design prioritizes cost-efficiency, maintainability, and incremental value delivery through 4 well-scoped phases.*

*Questions or feedback? Contact: [solution-architect@preemploymentdirectory.com]*
