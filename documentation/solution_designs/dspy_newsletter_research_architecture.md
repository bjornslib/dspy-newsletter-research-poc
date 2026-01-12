# DSPy Newsletter Research Tool - Solution Design
**Constraint-First Architecture for PreEmploymentDirectory**

## Executive Summary

This solution design addresses PreEmploymentDirectory's need to automate their newsletter research process, transforming a 20+ hour/week manual workflow into an intelligent DSPy-based pipeline. The architecture is designed **constraint-first**: prioritizing PoC budget limitations, 500+ articles/day throughput, cost minimization, and multilingual content handling.

**Core Philosophy**: Offline optimization, online efficiency. Use DSPy's BootstrapFewShot and MIPROv2 optimizers to learn effective prompts once, then serve a cheap, stable batch inference pipeline.

**Target Metrics**:
- 70%+ recall vs manual process
- 20-50 curated articles/day output
- <$50/month operational cost (PoC phase)
- <5 second CLI query response time
- Support for 500+ articles/day ingestion

---

## 1. CONSTRAINT ANALYSIS & BOUNDARIES

### 1.1 Hard Constraints (Non-Negotiable)

| Constraint | Impact | Architectural Response |
|------------|--------|------------------------|
| **PoC Budget** | Minimize API costs | - Batch processing (not real-time)<br>- Cheap models (GPT-4o-mini) for classification<br>- Aggressive text truncation<br>- Caching & deduplication<br>- Offline optimization only |
| **500+ Articles/Day** | High throughput | - Scheduled batch jobs (15-min intervals)<br>- Parallel processing capability<br>- Efficient embedding generation<br>- Database indexing strategy |
| **70%+ Recall Target** | Over-inclusion bias | - Prefer false positives over false negatives<br>- Multi-stage filtering (keyword → semantic)<br>- Lower confidence thresholds<br>- Human review queue for borderline cases |
| **Multilingual Content** | Language diversity | - ~70% English, 30% non-English<br>- Translation strategy only for high-relevance signals<br>- Language-agnostic embeddings<br>- Keyword matching in original language |

### 1.2 Soft Constraints (Design Goals)

- **Maintainability**: Simple Python codebase, clear module boundaries
- **Extensibility**: Easy to add new sources, regions, topics
- **Observability**: Clear logging, metrics tracking, error monitoring
- **Human-in-the-Loop**: Review queue for quality assurance and continuous improvement

### 1.3 PoC Scope Boundaries

**IN SCOPE (Phase 1-2)**:
- RSS feed ingestion (Tier 1: 20 feeds)
- Single-stage classification (region + topics)
- Basic relevance scoring (keyword + semantic)
- ChromaDB vector storage
- CLI query interface
- JSON file storage for metadata

**OUT OF SCOPE (Future)**:
- Web scraping (Tier 3 sources)
- Email parsing integration
- Real-time processing
- Advanced deduplication (cross-source clustering)
- Multi-language translation
- Web UI/dashboard
- Advanced agent-based query processing

---

## 2. SYSTEM ARCHITECTURE

### 2.1 High-Level Architecture (ASCII Diagram)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OFFLINE OPTIMIZATION PLANE                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  DSPy Training Pipeline (Weekly/On-Demand)                    │  │
│  │                                                               │  │
│  │  [Labeled Dataset] ──→ BootstrapFewShot ──→ [Optimized       │  │
│  │       (200-500        (5-8 examples)          Prompts]       │  │
│  │        examples)            ↓                    ↓            │  │
│  │                       MIPROv2 ──────→ [Tuned Hyperparams]    │  │
│  │                    (instruction opt)         ↓               │  │
│  │                                    [classifier_v1.json]      │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    ONLINE SERVING PLANE (Batch)                      │
│                                                                       │
│  ┌─────────────┐       ┌──────────────┐       ┌─────────────────┐  │
│  │   Ingest    │  →    │  Preprocess  │  →    │   Classify      │  │
│  │  (RSS/API)  │       │  & Filter    │       │  (DSPy Frozen)  │  │
│  └─────────────┘       └──────────────┘       └─────────────────┘  │
│        ↓                      ↓                        ↓             │
│   Every 15min          - Dedupe hash           - Region + Topics    │
│   ~125 articles        - Truncate 1.5k         - Relevance score    │
│                        - Language detect       - GPT-4o-mini        │
│                                                                       │
│  ┌─────────────┐       ┌──────────────┐       ┌─────────────────┐  │
│  │   Embed &   │  →    │   Store &    │  →    │  Review Queue   │  │
│  │   Index     │       │   Cache      │       │  (Human QA)     │  │
│  └─────────────┘       └──────────────┘       └─────────────────┘  │
│        ↓                      ↓                        ↓             │
│   ChromaDB            - JSON metadata          - Score < 0.7        │
│   Embeddings          - Hash cache             - Daily top 50       │
│   (OpenAI)            - PostgreSQL/SQLite      - Feedback loop      │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         QUERY INTERFACE                              │
│                                                                       │
│  CLI: "recent APAC regulation changes"                               │
│    ↓                                                                  │
│  [DSPy RAG Module] → Retrieve (ChromaDB k=10)                       │
│                   → Generate (ChainOfThought)                        │
│                   → Response (ranked articles + summaries)           │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Breakdown

#### 2.2.1 Ingestion Layer
- **RSS Feed Parser**: `feedparser` library for Tier 1 sources
- **Schedule**: Every 15 minutes via `cron` or `schedule` library
- **Deduplication**: SHA256 hash of (title + published_date + source)
- **Storage**: Append to `articles_raw.jsonl` (one article per line)

#### 2.2.2 Preprocessing Layer
- **Text Cleaning**: Remove HTML, normalize whitespace, handle encoding
- **Truncation**: Cap at 1,500 tokens (~2,000 chars) for cost control
- **Language Detection**: `langdetect` library (cached per source)
- **Keyword Pre-Filter**: Regex match on primary keywords (background check, employment screening, etc.)
  - **Purpose**: Drop 60-70% of irrelevant articles before LLM calls
  - **Threshold**: Must match ≥1 primary keyword OR ≥2 secondary keywords

#### 2.2.3 Classification Layer (DSPy Modules)
**Core Module**: `ArticleClassifier`
- **Input**: Truncated article text (1,500 tokens max)
- **Output**: Region, topics[], relevance_score, reasoning
- **Model**: GPT-4o-mini (cost-optimized)
- **Temperature**: 0.1 (deterministic for caching)
- **Batch Size**: 16 articles per API call

#### 2.2.4 Storage Layer
- **Metadata**: SQLite database (`articles.db`)
  - Schema: `id, title, source, url, published_date, region, country, topics[], relevance_score, summary, reasoning, hash, created_at`
  - Indexes: `hash`, `published_date`, `region`, `relevance_score`
- **Vector Embeddings**: ChromaDB collection
  - Embeddings: OpenAI `text-embedding-3-small` (512 dimensions, low cost)
  - Metadata: `article_id`, `region`, `topics`, `published_date`
  - Per-article embedding (not chunked for PoC)

#### 2.2.5 Query Layer (DSPy RAG)
- **Retrieval**: ChromaDB similarity search (k=10)
- **Reranking**: Optional ColBERTv2 for precision (Phase 2)
- **Generation**: DSPy ChainOfThought for natural language summaries
- **Caching**: Query hash → response cache (Redis or in-memory)

---

## 3. DSPy SIGNATURES & MODULES

### 3.1 Core Signatures

```python
import dspy
from typing import List, Literal
from pydantic import BaseModel, Field

# Type definitions for structured output
RegionType = Literal[
    "africa_middle_east",
    "asia_pacific",
    "europe",
    "north_america",
    "south_america",
    "worldwide"
]

TopicType = Literal[
    "regulatory_legal",
    "criminal_background",
    "education_verification",
    "immigration_work_auth",
    "industry_ma_news",
    "technology_products",
    "conferences_events",
    "court_cases_precedents"
]

# ─────────────────────────────────────────────────────────────
# Signature 1: Relevance Pre-Filter (Fast, Cheap)
# ─────────────────────────────────────────────────────────────
class RelevanceFilter(dspy.Signature):
    """
    Determine if article is relevant to background screening industry.
    Focus on employment screening, criminal checks, compliance, FCRA/GDPR.
    """
    title: str = dspy.InputField(desc="Article headline")
    snippet: str = dspy.InputField(desc="First 300 chars of article")

    is_relevant: bool = dspy.OutputField(
        desc="True if article discusses background screening, employment checks, "
             "data protection in hiring, or related compliance topics"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score 0.0-1.0"
    )
    reasoning: str = dspy.OutputField(
        desc="One sentence explaining relevance determination"
    )

# ─────────────────────────────────────────────────────────────
# Signature 2: Multi-Label Classification (Region + Topics)
# ─────────────────────────────────────────────────────────────
class ArticleClassifier(dspy.Signature):
    """
    Classify article by geographic region and applicable topics.
    Can assign multiple topics. Region must be exactly one.
    """
    title: str = dspy.InputField()
    content: str = dspy.InputField(desc="Truncated article text, max 1500 tokens")

    region: RegionType = dspy.OutputField(
        desc="Primary geographic region. Use 'worldwide' if multi-regional or global."
    )
    country: str = dspy.OutputField(
        desc="Specific country if identifiable, else 'multiple' or 'unknown'"
    )
    topics: List[TopicType] = dspy.OutputField(
        desc="All applicable topic categories (array, can be multiple)"
    )
    reasoning: str = dspy.OutputField(
        desc="2-3 sentences explaining region and topic selections"
    )

# ─────────────────────────────────────────────────────────────
# Signature 3: Relevance Scoring (Semantic)
# ─────────────────────────────────────────────────────────────
class RelevanceScorer(dspy.Signature):
    """
    Score article relevance to background screening industry (0.0-1.0).
    Consider: directness to topic, actionability for newsletter readers,
    recency of information, regulatory/legal importance.
    """
    title: str = dspy.InputField()
    content: str = dspy.InputField()
    region: str = dspy.InputField()
    topics: List[str] = dspy.InputField()

    relevance_score: float = dspy.OutputField(
        desc="Score 0.0-1.0 where 1.0 is highly relevant, must-include content"
    )
    priority: Literal["high", "medium", "low"] = dspy.OutputField(
        desc="Editorial priority: high (regulatory/legal), medium (news/product), low (events/general)"
    )
    reasoning: str = dspy.OutputField(
        desc="2-3 sentences justifying score and priority"
    )

# ─────────────────────────────────────────────────────────────
# Signature 4: RAG Query Response
# ─────────────────────────────────────────────────────────────
class QueryResponse(dspy.Signature):
    """
    Answer user question about background screening news using retrieved articles.
    Provide concise summary with article references.
    """
    question: str = dspy.InputField()
    context: str = dspy.InputField(desc="Retrieved article texts and metadata")

    answer: str = dspy.OutputField(
        desc="3-5 sentence answer synthesizing information from context"
    )
    cited_articles: List[str] = dspy.OutputField(
        desc="List of article titles that support the answer"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence in answer completeness (0.0-1.0)"
    )
```

### 3.2 DSPy Module Implementations

```python
import dspy
from typing import Dict, List, Any

# ─────────────────────────────────────────────────────────────
# Module 1: Two-Stage Pipeline (Filter → Classify → Score)
# ─────────────────────────────────────────────────────────────
class NewsletterPipeline(dspy.Module):
    """
    Complete classification pipeline with cost-optimized two-stage design:
    1. Fast relevance filter (drops 60-70% of articles)
    2. Full classification + scoring for relevant articles only
    """

    def __init__(self):
        super().__init__()

        # Stage 1: Cheap, fast filter using Predict (no CoT)
        self.relevance_filter = dspy.Predict(RelevanceFilter)

        # Stage 2: Full classification with ChainOfThought reasoning
        self.classifier = dspy.ChainOfThought(ArticleClassifier)

        # Stage 3: Relevance scoring with reasoning
        self.scorer = dspy.ChainOfThought(RelevanceScorer)

    def forward(self, title: str, content: str, snippet: str = None) -> Dict[str, Any]:
        """
        Process article through pipeline.
        Returns None if filtered out, else full classification result.
        """
        # Stage 1: Quick relevance check (uses only title + snippet)
        if snippet is None:
            snippet = content[:300]

        filter_result = self.relevance_filter(
            title=title,
            snippet=snippet
        )

        # Early exit if not relevant (saves 60-70% of classification costs)
        if not filter_result.is_relevant or filter_result.confidence < 0.5:
            return None

        # Stage 2: Full classification
        classification = self.classifier(
            title=title,
            content=content
        )

        # Stage 3: Relevance scoring
        score_result = self.scorer(
            title=title,
            content=content,
            region=classification.region,
            topics=classification.topics
        )

        return {
            "region": classification.region,
            "country": classification.country,
            "topics": classification.topics,
            "relevance_score": score_result.relevance_score,
            "priority": score_result.priority,
            "classification_reasoning": classification.reasoning,
            "score_reasoning": score_result.reasoning,
            "filter_confidence": filter_result.confidence
        }

# ─────────────────────────────────────────────────────────────
# Module 2: RAG Query Module
# ─────────────────────────────────────────────────────────────
class NewsletterRAG(dspy.Module):
    """
    Retrieval-augmented generation for natural language queries.
    Uses ChromaDB retrieval + ChainOfThought generation.
    """

    def __init__(self, retriever: dspy.Retrieve, k: int = 10):
        super().__init__()
        self.retrieve = retriever
        self.k = k
        self.generate = dspy.ChainOfThought(QueryResponse)

    def forward(self, question: str) -> dspy.Prediction:
        """
        Answer question using retrieved context.
        """
        # Retrieve top-k relevant articles
        retrieved = self.retrieve(question, k=self.k)

        # Format context from retrieved passages
        context_parts = []
        for idx, passage in enumerate(retrieved.passages, 1):
            context_parts.append(
                f"[Article {idx}] {passage.get('title', 'Untitled')}\n"
                f"Region: {passage.get('region', 'Unknown')}\n"
                f"Topics: {', '.join(passage.get('topics', []))}\n"
                f"Content: {passage.get('text', '')[:500]}...\n"
            )

        context = "\n\n".join(context_parts)

        # Generate answer with citations
        response = self.generate(
            question=question,
            context=context
        )

        return response

# ─────────────────────────────────────────────────────────────
# Module 3: Batch Processing Wrapper
# ─────────────────────────────────────────────────────────────
class BatchClassifier(dspy.Module):
    """
    Batch processing wrapper for efficient API usage.
    Processes articles in parallel batches.
    """

    def __init__(self, pipeline: NewsletterPipeline, batch_size: int = 16):
        super().__init__()
        self.pipeline = pipeline
        self.batch_size = batch_size

    def forward(self, articles: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Process multiple articles in batches.
        Returns list of classification results (None for filtered articles).
        """
        results = []

        # Process in batches to optimize API calls
        for i in range(0, len(articles), self.batch_size):
            batch = articles[i:i + self.batch_size]

            batch_results = []
            for article in batch:
                result = self.pipeline(
                    title=article['title'],
                    content=article['content'],
                    snippet=article.get('snippet')
                )
                batch_results.append(result)

            results.extend(batch_results)

        return results
```

---

## 4. MODULE SELECTION RATIONALE

### 4.1 Why This Module Architecture?

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **Two-Stage Pipeline** (Filter → Classify → Score) | - Drops 60-70% of articles early<br>- Saves ~$30-40/month in API costs<br>- Faster processing | - Additional latency overhead<br>- Risk of false negatives in filter |
| **Predict for Filter, CoT for Classification** | - Filter needs speed, not reasoning<br>- Classification benefits from explicit reasoning<br>- Scorer needs to justify priority decisions | - Inconsistent reasoning depth<br>- Harder to optimize together |
| **Separate Scorer Module** | - Decouples classification from editorial priority<br>- Allows independent optimization<br>- Provides transparency for human review | - Extra API call per article<br>- More complex pipeline |
| **ChainOfThought over TypedPredictor** | - Better reasoning quality for complex multi-label tasks<br>- Easier debugging via exposed reasoning<br>- More suitable for BootstrapFewShot optimization | - Slightly higher token costs<br>- Longer output sequences |

### 4.2 Alternative Architectures Considered

**❌ Single-Stage Monolithic Classifier**
- Reason rejected: High API costs (all 500 articles classified), poor cost control

**❌ Agent-Based with ReAct**
- Reason rejected: Overkill for PoC, high latency, unpredictable costs, complex optimization

**❌ Fine-Tuned Small Model** (DistilBERT, etc.)
- Reason rejected: Requires large labeled dataset (>5k examples), poor multilingual support, hard to iterate

**✅ CHOSEN: Modular Two-Stage with DSPy Optimization**
- Best balance of cost, quality, and iteration speed
- Clear separation of concerns
- Each module independently optimizable

---

## 5. OPTIMIZATION STRATEGY

### 5.1 Training Data Requirements

**Target Dataset Size**: 200-500 labeled examples
- **Distribution**: Stratified by region (6 classes) and topics (8 classes)
- **Quality over Quantity**: 200 high-quality examples > 1000 noisy examples
- **Labeling Strategy**:
  1. **Week 1**: Staff manually label 100 articles from current newsletter archive
  2. **Week 2**: Run pipeline on unlabeled data, human review top 100 by score
  3. **Week 3**: Active learning: label articles where classifier has low confidence
  4. **Week 4**: Achieve 200-300 labeled examples

**Example Distribution** (Target):
```
Region Distribution:
  north_america: 80 examples (40%)
  europe: 60 examples (30%)
  asia_pacific: 30 examples (15%)
  worldwide: 20 examples (10%)
  africa_middle_east: 5 examples (2.5%)
  south_america: 5 examples (2.5%)

Topic Distribution (multi-label, overlapping):
  regulatory_legal: 100 examples (50%)
  criminal_background: 80 examples (40%)
  technology_products: 40 examples (20%)
  industry_ma_news: 30 examples (15%)
  ... (other topics)
```

### 5.2 BootstrapFewShot Configuration

```python
from dspy.teleprompt import BootstrapFewShot

# ─────────────────────────────────────────────────────────────
# Metric: Balanced F1 Score (Region + Topics)
# ─────────────────────────────────────────────────────────────
def classification_metric(example, pred, trace=None) -> float:
    """
    Combined metric: Region accuracy + Topics macro F1.
    Weights: 30% region, 70% topics (topics more important).
    """
    if pred is None:
        return 0.0

    # Region accuracy (binary)
    region_correct = 1.0 if pred.region == example.region else 0.0

    # Topics F1 (set-based)
    pred_topics = set(pred.topics)
    true_topics = set(example.topics)

    if len(pred_topics) == 0 and len(true_topics) == 0:
        topic_f1 = 1.0
    elif len(pred_topics) == 0 or len(true_topics) == 0:
        topic_f1 = 0.0
    else:
        precision = len(pred_topics & true_topics) / len(pred_topics)
        recall = len(pred_topics & true_topics) / len(true_topics)
        topic_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Combined score (weighted)
    return 0.3 * region_correct + 0.7 * topic_f1

# ─────────────────────────────────────────────────────────────
# Optimizer Configuration
# ─────────────────────────────────────────────────────────────
optimizer = BootstrapFewShot(
    metric=classification_metric,
    max_bootstrapped_demos=8,        # 8 few-shot examples max
    max_labeled_demos=16,             # Draw from pool of 16 candidates
    max_rounds=3,                     # Iterate 3 times for stability
    max_errors=5,                     # Allow 5 errors before stopping
    teacher_settings=dict(
        lm=dspy.OpenAI(model='gpt-4o-mini', max_tokens=500)
    )
)

# Optimize pipeline (runs offline, ~30 min on 200 examples)
compiled_pipeline = optimizer.compile(
    student=NewsletterPipeline(),
    trainset=labeled_articles[:200],
    valset=labeled_articles[200:250]  # Hold out 50 for validation
)

# Save optimized version
compiled_pipeline.save('models/classifier_v1.json')
```

### 5.3 MIPROv2 Configuration (Optional, Phase 2)

**Use MIPROv2 when**:
- Initial BootstrapFewShot results plateau
- Need to optimize multi-module pipeline jointly
- Have 300+ labeled examples to prevent overfitting

```python
from dspy.teleprompt import MIPROv2

# MIPROv2 for instruction + example joint optimization
mipro_optimizer = MIPROv2(
    metric=classification_metric,
    auto="light",                     # Light mode: faster, fewer trials
    num_candidates=10,                # Generate 10 instruction candidates
    init_temperature=1.0,             # Instruction generation creativity
    verbose=True
)

# Optimize (runs 40-60 trials, ~2 hours on 300 examples)
mipro_pipeline = mipro_optimizer.compile(
    student=NewsletterPipeline(),
    trainset=labeled_articles[:300],
    num_trials=40,                    # Cost control: limit trials
    max_bootstrapped_demos=6,         # Slightly fewer examples for stability
    max_labeled_demos=20,
    eval_kwargs=dict(num_threads=4, display_progress=True)
)

mipro_pipeline.save('models/classifier_v2_mipro.json')
```

### 5.4 Optimization Cost Control

**Budget Management**:
- **BootstrapFewShot**: ~$2-5 per optimization run (200 examples, GPT-4o-mini)
- **MIPROv2**: ~$10-20 per optimization run (300 examples, 40 trials)
- **Schedule**: Run optimization weekly during PoC, monthly in production
- **Strategy**: Start with BootstrapFewShot, add MIPROv2 only if F1 < 0.75

**Cost Reduction Tactics**:
1. Use cached LM calls during optimization (DSPy built-in)
2. Subsample large datasets (200 examples sufficient for PoC)
3. Limit max_rounds and num_trials aggressively
4. Use cheaper model for teacher (GPT-4o-mini vs GPT-4)

---

## 6. RAG IMPLEMENTATION

### 6.1 Vector Database Setup (ChromaDB)

```python
import chromadb
from chromadb.config import Settings
import dspy

# ─────────────────────────────────────────────────────────────
# ChromaDB Initialization
# ─────────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(
    path="./data/chromadb",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

# Create collection with metadata filtering
collection = chroma_client.get_or_create_collection(
    name="newsletter_articles",
    metadata={"hnsw:space": "cosine"},  # Cosine similarity
    embedding_function=None  # We'll handle embeddings manually
)

# ─────────────────────────────────────────────────────────────
# DSPy Embedder Setup
# ─────────────────────────────────────────────────────────────
embedder = dspy.Embedder(
    model='openai/text-embedding-3-small',
    dimensions=512,  # Cost-optimized (vs 1536 default)
    api_key=os.getenv('OPENAI_API_KEY')
)

# ─────────────────────────────────────────────────────────────
# Embedding & Indexing Function
# ─────────────────────────────────────────────────────────────
def index_article(article: Dict[str, Any], embedding_text: str = None):
    """
    Embed and index article in ChromaDB.

    Args:
        article: Article metadata dict with keys:
                 id, title, content, region, topics, relevance_score, etc.
        embedding_text: Optional custom text for embedding
                       (default: title + first 500 chars of content)
    """
    # Prepare embedding text
    if embedding_text is None:
        embedding_text = f"{article['title']}\n\n{article['content'][:500]}"

    # Generate embedding (batched internally by DSPy)
    embedding = embedder(embedding_text)

    # Prepare metadata (ChromaDB supports filtering)
    metadata = {
        "article_id": article['id'],
        "title": article['title'],
        "source": article['source'],
        "url": article['url'],
        "published_date": article['published_date'],
        "region": article['region'],
        "country": article.get('country', 'unknown'),
        "topics": ','.join(article['topics']),  # Store as comma-separated
        "relevance_score": article['relevance_score'],
        "priority": article.get('priority', 'medium')
    }

    # Add to collection
    collection.add(
        ids=[str(article['id'])],
        embeddings=[embedding],
        documents=[embedding_text],
        metadatas=[metadata]
    )

# ─────────────────────────────────────────────────────────────
# Batch Indexing (for initial load)
# ─────────────────────────────────────────────────────────────
def batch_index_articles(articles: List[Dict[str, Any]], batch_size: int = 100):
    """Index multiple articles efficiently."""
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i+batch_size]

        # Prepare batch data
        ids = [str(a['id']) for a in batch]
        texts = [f"{a['title']}\n\n{a['content'][:500]}" for a in batch]
        embeddings = embedder(texts)  # Batch embedding call
        metadatas = [
            {
                "article_id": a['id'],
                "title": a['title'],
                "source": a['source'],
                "url": a['url'],
                "published_date": a['published_date'],
                "region": a['region'],
                "country": a.get('country', 'unknown'),
                "topics": ','.join(a['topics']),
                "relevance_score": a['relevance_score'],
                "priority": a.get('priority', 'medium')
            }
            for a in batch
        ]

        # Add batch to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        print(f"Indexed {len(batch)} articles (total: {i+len(batch)})")
```

### 6.2 DSPy Retriever Integration

```python
# ─────────────────────────────────────────────────────────────
# Custom ChromaDB Retriever for DSPy
# ─────────────────────────────────────────────────────────────
class ChromaDBRetriever(dspy.Retrieve):
    """
    Custom DSPy retriever using ChromaDB.
    Supports metadata filtering (region, topics, date range).
    """

    def __init__(self, collection, embedder, k: int = 10):
        super().__init__(k=k)
        self.collection = collection
        self.embedder = embedder

    def forward(
        self,
        query: str,
        k: int = None,
        region: str = None,
        topics: List[str] = None,
        min_score: float = None,
        date_range: tuple = None
    ) -> dspy.Prediction:
        """
        Retrieve articles matching query with optional filters.

        Args:
            query: Natural language search query
            k: Number of results (default: self.k)
            region: Filter by region
            topics: Filter by topics (OR logic)
            min_score: Minimum relevance score
            date_range: (start_date, end_date) tuple

        Returns:
            dspy.Prediction with .passages attribute
        """
        k = k or self.k

        # Build metadata filter (ChromaDB where clause)
        where_filter = {}
        if region:
            where_filter["region"] = region
        if min_score:
            where_filter["relevance_score"] = {"$gte": min_score}
        # Note: ChromaDB metadata filters are limited;
        # for complex filters (topics OR, date ranges), post-filter

        # Generate query embedding
        query_embedding = self.embedder(query)

        # Query ChromaDB (retrieves more than k for post-filtering)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k * 2, 50),  # Over-fetch for filtering
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"]
        )

        # Post-process results
        passages = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            # Apply complex filters (topics, date range)
            if topics:
                article_topics = metadata['topics'].split(',')
                if not any(t in article_topics for t in topics):
                    continue

            if date_range:
                article_date = metadata['published_date']
                if not (date_range[0] <= article_date <= date_range[1]):
                    continue

            # Convert to passage format
            passages.append({
                "text": doc,
                "title": metadata['title'],
                "source": metadata['source'],
                "url": metadata['url'],
                "region": metadata['region'],
                "topics": metadata['topics'].split(','),
                "relevance_score": metadata['relevance_score'],
                "similarity": 1 - distance,  # Convert distance to similarity
                "article_id": metadata['article_id']
            })

            if len(passages) >= k:
                break

        return dspy.Prediction(passages=passages)

# ─────────────────────────────────────────────────────────────
# Initialize RAG System
# ─────────────────────────────────────────────────────────────
retriever = ChromaDBRetriever(
    collection=collection,
    embedder=embedder,
    k=10
)

rag_system = NewsletterRAG(retriever=retriever, k=10)
```

### 6.3 Query Examples

```python
# Example 1: Simple query
response = rag_system("What are recent APAC regulation changes?")
print(f"Answer: {response.answer}")
print(f"Citations: {response.cited_articles}")
print(f"Confidence: {response.confidence}")

# Example 2: Filtered query (via custom retriever)
retriever_filtered = ChromaDBRetriever(collection, embedder, k=5)
response = rag_system.retrieve(
    query="criminal background check requirements",
    region="north_america",
    topics=["regulatory_legal", "criminal_background"],
    min_score=0.7
)

# Example 3: Date-filtered query
from datetime import datetime, timedelta
last_30_days = (
    (datetime.now() - timedelta(days=30)).isoformat(),
    datetime.now().isoformat()
)
response = rag_system.retrieve(
    query="Ban the Box legislation updates",
    region="north_america",
    date_range=last_30_days
)
```

---

## 7. PHASED IMPLEMENTATION PLAN

### Phase 1: Foundation & Core Pipeline (Weeks 1-3)

**Goal**: Functional end-to-end pipeline with basic classification

**Deliverables**:
1. **Week 1: Infrastructure Setup**
   - ✅ Project scaffolding (Python, Poetry/pipenv)
   - ✅ RSS feed ingester for Tier 1 sources (20 feeds)
   - ✅ SQLite database schema + migrations
   - ✅ Basic preprocessing (cleaning, truncation, deduplication)
   - ✅ Keyword pre-filter (regex-based)

2. **Week 2: DSPy Classification Pipeline**
   - ✅ Define DSPy signatures (ArticleClassifier, RelevanceScorer)
   - ✅ Implement NewsletterPipeline module
   - ✅ Manual labeling of 100 articles (staff time: 4-6 hours)
   - ✅ Baseline classification (no optimization)
   - ✅ Evaluation metrics (F1, recall, precision)

3. **Week 3: Optimization & Storage**
   - ✅ Label 100 more articles (active learning based)
   - ✅ Run BootstrapFewShot optimizer
   - ✅ ChromaDB setup + embedding generation
   - ✅ Batch processing script (15-min schedule)
   - ✅ Basic logging and error handling

**Success Criteria**:
- [ ] Process 500 articles/day without errors
- [ ] 60%+ recall on held-out test set (100 articles)
- [ ] <$30/month API costs
- [ ] 20-50 high-relevance articles flagged daily

**Risks**:
- ⚠️ Labeling bottleneck (staff time)
  - **Mitigation**: Use semi-automated labeling (classifier suggests, human confirms)
- ⚠️ API rate limits during batch processing
  - **Mitigation**: Implement exponential backoff, respect OpenAI rate limits

---

### Phase 2: RAG & Query Interface (Weeks 4-5)

**Goal**: Natural language query interface with retrieval

**Deliverables**:
1. **Week 4: RAG Implementation**
   - ✅ ChromaDBRetriever with metadata filtering
   - ✅ NewsletterRAG module (QueryResponse signature)
   - ✅ CLI interface (argparse or Typer)
   - ✅ Query caching (hash → response)
   - ✅ Result ranking and formatting

2. **Week 5: Refinement & Testing**
   - ✅ User acceptance testing with staff
   - ✅ Query performance tuning (k-value, reranking)
   - ✅ Documentation (README, usage guide)
   - ✅ Logging and monitoring dashboard (simple)

**Success Criteria**:
- [ ] <5 second query response time (p95)
- [ ] >80% query satisfaction rate (staff feedback)
- [ ] Correct metadata filtering (region, topics, dates)

**Risks**:
- ⚠️ Poor retrieval quality (irrelevant results)
  - **Mitigation**: Tune embedding model, add reranking, adjust k-value
- ⚠️ Query ambiguity (vague questions)
  - **Mitigation**: Prompt engineering, query expansion, clarification prompts

---

### Phase 3: Production Hardening (Weeks 6-7)

**Goal**: Reliable, monitored production system

**Deliverables**:
1. **Week 6: Robustness**
   - ✅ Error handling and retry logic
   - ✅ Dead letter queue for failed articles
   - ✅ Database backups and recovery
   - ✅ API cost monitoring and alerts
   - ✅ Performance profiling and optimization

2. **Week 7: Human-in-the-Loop**
   - ✅ Review queue interface (simple web UI or CLI)
   - ✅ Feedback collection mechanism
   - ✅ Incremental retraining pipeline
   - ✅ A/B testing framework (v1 vs v2 classifiers)

**Success Criteria**:
- [ ] 99%+ uptime during business hours
- [ ] <1% article processing failure rate
- [ ] Feedback loop functional (corrections → retraining)

**Risks**:
- ⚠️ Staff adoption resistance (prefer manual process)
  - **Mitigation**: Gradual rollout, transparent explanations, easy override
- ⚠️ Classifier drift over time
  - **Mitigation**: Weekly metrics tracking, monthly retraining

---

### Phase 4: Expansion & Advanced Features (Weeks 8-10)

**Goal**: Scale to full source coverage + advanced capabilities

**Deliverables**:
1. **Week 8: Tier 2 Sources**
   - ✅ Add 15 company/regional RSS feeds
   - ✅ Multilingual keyword lists (non-English sources)
   - ✅ Optional translation for high-relevance non-English articles

2. **Week 9: Advanced Deduplication**
   - ✅ Cross-source clustering (similar stories)
   - ✅ MinHash LSH for near-duplicate detection
   - ✅ Canonical article selection (prefer official sources)

3. **Week 10: MIPROv2 & Fine-Tuning**
   - ✅ Run MIPROv2 optimization (if BootstrapFewShot insufficient)
   - ✅ Experiment with small model fine-tuning (DistilBERT for filter)
   - ✅ Performance benchmarking and cost analysis

**Success Criteria**:
- [ ] 70%+ recall on full source set (247 sources)
- [ ] <10% duplicate articles in daily output
- [ ] <$50/month total operational cost

**Risks**:
- ⚠️ Multilingual performance degradation
  - **Mitigation**: Language-specific prompt engineering, translation fallback
- ⚠️ Scaling costs with more sources
  - **Mitigation**: More aggressive keyword filtering, tiered source priority

---

## 8. RISKS & MITIGATIONS

### 8.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| **API Cost Overruns** | Medium | High | - Aggressive keyword pre-filtering<br>- Cached classifications (hash-based)<br>- Batch processing with rate limiting<br>- Cost monitoring alerts |
| **Poor Classification Accuracy** | Medium | High | - BootstrapFewShot optimization<br>- Human review queue for low-confidence<br>- Continuous feedback loop<br>- A/B testing of classifier versions |
| **ChromaDB Performance Issues** | Low | Medium | - Index tuning (HNSW parameters)<br>- Pagination for large result sets<br>- Consider Weaviate/Pinecone if >100k articles |
| **RSS Feed Failures** | High | Low | - Retry logic with exponential backoff<br>- Dead letter queue<br>- Monitoring and alerts<br>- Graceful degradation (skip source) |
| **Multilingual Content Handling** | Medium | Medium | - Language-agnostic embeddings (multilingual models)<br>- Translation only for high-relevance articles<br>- Language-specific keyword lists |

### 8.2 Operational Risks

| Risk | Likelihood | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| **Staff Labeling Bottleneck** | High | Medium | - Semi-automated labeling (classifier suggests)<br>- Prioritize labeling high-impact regions<br>- Active learning for efficient labeling |
| **User Adoption Resistance** | Medium | High | - Gradual rollout (augment, don't replace)<br>- Transparent explanations (reasoning fields)<br>- Easy override and feedback mechanisms<br>- Regular check-ins with staff |
| **Classifier Drift** | Medium | Medium | - Weekly performance monitoring<br>- Monthly retraining schedule<br>- Version control for classifiers<br>- A/B testing before rollout |
| **Data Quality Issues** | Medium | Medium | - RSS feed validation<br>- Content length filtering<br>- HTML cleaning and normalization<br>- Manual spot-checks |

### 8.3 Business Risks

| Risk | Likelihood | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| **ROI Not Achieved** (time savings) | Low | High | - Track time savings metrics<br>- Measure before/after newsletter creation time<br>- Highlight high-value catches (missed by manual) |
| **False Negatives** (miss important news) | Medium | High | - Prefer recall over precision (70%+ target)<br>- Redundant sources (same story, multiple feeds)<br>- Weekly manual spot-checks of dropped articles |
| **Scope Creep** (feature requests) | High | Medium | - Strict PoC scope boundaries<br>- Feature backlog for post-PoC<br>- Prioritize core functionality |

---

## 9. COST ANALYSIS (PoC Budget)

### 9.1 Estimated Monthly Costs

**Assumptions**:
- 500 articles/day × 30 days = 15,000 articles/month
- 40% filtered by keywords = 9,000 articles to classify
- 1,500 tokens/article average (truncated)

| Component | Usage | Unit Cost | Monthly Cost |
|-----------|-------|-----------|--------------|
| **Classification API** (GPT-4o-mini) | 9,000 articles × 1,500 tokens | $0.15/1M input | ~$20.25 |
| **Embedding API** (text-embedding-3-small) | 9,000 articles × 500 tokens | $0.02/1M tokens | ~$0.90 |
| **Query API** (RAG responses) | 100 queries/month × 5,000 tokens | $0.15/1M input | ~$0.75 |
| **Optimization** (weekly BootstrapFewShot) | 200 examples × 4 runs | $2/run | ~$8.00 |
| **Infrastructure** (ChromaDB, SQLite) | Local storage | Free | $0.00 |
| **Total Estimated Cost** | | | **~$30/month** |

**Cost Control Levers**:
1. **Increase keyword filtering threshold** → 60% filtered = $15/month
2. **Reduce truncation limit** (1,500 → 1,000 tokens) → $13.50/month
3. **Bi-weekly optimization** instead of weekly → $26/month
4. **Use local embedding model** (sentence-transformers) → $20/month

**Buffer for Overruns**: Recommend $50/month budget for PoC safety margin

### 9.2 Scaling Costs (Post-PoC)

**At Full Source Coverage** (247 sources, ~1,000 articles/day):
- Classification: ~$40/month
- Embeddings: ~$2/month
- Queries: ~$2/month
- **Total**: ~$44/month + optimization costs

**Cost remains <$50/month even at 2x PoC volume** due to efficient batching and filtering.

---

## 10. SUCCESS METRICS & MONITORING

### 10.1 Key Performance Indicators (KPIs)

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Recall** | ≥70% | Weekly evaluation on 100-article test set |
| **Precision** | ≥40% | Human review of top 50 daily articles |
| **Daily Article Output** | 20-50 articles | Count of articles with score ≥0.7 |
| **API Cost** | <$50/month | OpenAI usage dashboard |
| **Query Response Time** | <5 seconds (p95) | Application logging |
| **Processing Failure Rate** | <1% | Error logs and dead letter queue |
| **Staff Time Savings** | >15 hours/week | Before/after time tracking |

### 10.2 Monitoring Dashboard

**Daily Metrics** (logged to JSON, visualized in simple script):
```json
{
  "date": "2025-01-15",
  "articles_ingested": 523,
  "articles_filtered_keywords": 312,
  "articles_classified": 211,
  "articles_high_relevance": 34,
  "api_calls": 211,
  "api_cost_usd": 0.95,
  "errors": 2,
  "avg_processing_time_sec": 1.2,
  "region_distribution": {
    "north_america": 18,
    "europe": 9,
    "asia_pacific": 4,
    "worldwide": 3
  },
  "topic_distribution": {
    "regulatory_legal": 21,
    "criminal_background": 15,
    "technology_products": 8
  }
}
```

**Weekly Reports**:
- Precision/recall on test set
- False negative analysis (manually review dropped articles)
- Cost tracking vs budget
- User feedback summary

---

## 11. TECHNICAL SPECIFICATIONS

### 11.1 Technology Stack

| Layer | Technology | Version | Justification |
|-------|-----------|---------|---------------|
| **Language** | Python | 3.11+ | DSPy compatibility, rich ecosystem |
| **LM Framework** | DSPy | Latest | Core requirement |
| **LLM Provider** | OpenAI | GPT-4o-mini | Cost-optimized, reliable |
| **Vector DB** | ChromaDB | 0.4.x | Simple, embedded, free |
| **Embeddings** | OpenAI | text-embedding-3-small | Cost-optimized (512-dim) |
| **Web Framework** | FastAPI | 0.109+ | Future API if needed |
| **CLI** | Typer | 0.9+ | User-friendly CLI |
| **Database** | SQLite | 3.x | Lightweight, local |
| **Scheduler** | APScheduler | 3.x | Simple Python scheduler |
| **RSS Parser** | feedparser | 6.x | Industry standard |
| **Testing** | pytest | 7.x | Standard Python testing |

### 11.2 Data Schemas

**SQLite Schema** (`articles.db`):
```sql
CREATE TABLE articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hash TEXT UNIQUE NOT NULL,           -- SHA256(title + published_date + source)
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    url TEXT UNIQUE NOT NULL,
    published_date TEXT NOT NULL,        -- ISO format
    content TEXT NOT NULL,               -- Full article text
    snippet TEXT,                         -- First 300 chars
    language TEXT,                        -- Detected language code

    -- Classification results
    region TEXT,                          -- africa_middle_east, asia_pacific, etc.
    country TEXT,
    topics TEXT,                          -- JSON array as string
    relevance_score REAL,
    priority TEXT,                        -- high, medium, low

    -- Reasoning/metadata
    classification_reasoning TEXT,
    score_reasoning TEXT,
    filter_confidence REAL,

    -- Timestamps
    ingested_at TEXT NOT NULL,           -- When RSS fetch occurred
    processed_at TEXT,                    -- When classification completed
    reviewed_at TEXT,                     -- When human reviewed (if applicable)

    -- Human feedback
    human_label_region TEXT,
    human_label_topics TEXT,
    human_label_relevance REAL,
    feedback_notes TEXT,

    -- Indexing
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_hash ON articles(hash);
CREATE INDEX idx_published_date ON articles(published_date DESC);
CREATE INDEX idx_region ON articles(region);
CREATE INDEX idx_relevance_score ON articles(relevance_score DESC);
CREATE INDEX idx_processed_at ON articles(processed_at);
```

**ChromaDB Metadata Schema**:
```python
{
    "article_id": int,               # References articles.id
    "title": str,
    "source": str,
    "url": str,
    "published_date": str,           # ISO format
    "region": str,
    "country": str,
    "topics": str,                   # Comma-separated
    "relevance_score": float,
    "priority": str
}
```

### 11.3 File Structure

```
dspy_newsletter_poc/
├── data/
│   ├── chromadb/                   # ChromaDB persistent storage
│   ├── articles.db                 # SQLite database
│   ├── articles_raw.jsonl          # Raw ingested articles
│   └── cache/                      # Classification cache (hash → result)
├── models/
│   ├── classifier_v1.json          # Optimized DSPy pipeline
│   └── training_data/
│       ├── labeled_articles.jsonl
│       └── test_set.jsonl
├── src/
│   ├── ingestion/
│   │   ├── rss_fetcher.py
│   │   ├── preprocessor.py
│   │   └── keyword_filter.py
│   ├── classification/
│   │   ├── signatures.py           # DSPy signatures
│   │   ├── modules.py              # DSPy modules
│   │   ├── pipeline.py             # Main pipeline
│   │   └── optimizer.py            # Training & optimization
│   ├── storage/
│   │   ├── database.py             # SQLite operations
│   │   ├── vectordb.py             # ChromaDB operations
│   │   └── cache.py                # Hash-based caching
│   ├── query/
│   │   ├── rag.py                  # RAG module
│   │   ├── retriever.py            # ChromaDB retriever
│   │   └── cli.py                  # Command-line interface
│   └── utils/
│       ├── logging.py
│       ├── metrics.py
│       └── config.py
├── scripts/
│   ├── run_ingestion.py            # Scheduled batch job
│   ├── train_classifier.py         # Optimization script
│   ├── evaluate.py                 # Testing & metrics
│   └── export_review_queue.py      # Export for human review
├── tests/
│   ├── test_classification.py
│   ├── test_retrieval.py
│   └── test_pipeline.py
├── config/
│   ├── sources.yaml                # RSS feed URLs
│   ├── keywords.yaml               # Keyword lists
│   └── settings.yaml               # App configuration
├── pyproject.toml                  # Dependencies
└── README.md
```

---

## 12. DEVELOPMENT WORKFLOW

### 12.1 Daily Operations

**Automated (Scheduled)**:
1. **Every 15 minutes**: RSS ingestion → Preprocessing → Classification → Storage
2. **Daily 6 AM**: Generate review queue (top 50 articles by relevance_score)
3. **Weekly Sunday 2 AM**: Performance evaluation + metrics report

**Manual (On-Demand)**:
1. **Labeling**: `python scripts/label_articles.py` (human review interface)
2. **Training**: `python scripts/train_classifier.py` (run optimization)
3. **Querying**: `python -m src.query.cli "your question here"`
4. **Evaluation**: `python scripts/evaluate.py --test-set data/test_set.jsonl`

### 12.2 Continuous Improvement Loop

```
┌──────────────────────────────────────────────────────────┐
│  Week 1: Initial Deployment                              │
│  - Run baseline classifier                               │
│  - Collect 100 human reviews from review queue           │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│  Week 2: Label & Optimize                                │
│  - Add 100 labeled examples to training set              │
│  - Run BootstrapFewShot → classifier_v2                  │
│  - A/B test v1 vs v2 on held-out set                     │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│  Week 3: Deploy & Monitor                                │
│  - Deploy v2 if performance improved                     │
│  - Track precision/recall on production data             │
│  - Collect edge cases for next iteration                 │
└────────────────────┬─────────────────────────────────────┘
                     ↓
                  (Repeat)
```

### 12.3 Version Control Strategy

**Classifier Versioning**:
- `classifier_v1.json`: Initial BootstrapFewShot (200 examples)
- `classifier_v2.json`: Improved BootstrapFewShot (300 examples)
- `classifier_v3_mipro.json`: MIPROv2 optimization
- **Rollback**: Keep last 3 versions, easy revert via config file

**Data Versioning**:
- Training sets: `labeled_articles_v1.jsonl`, `v2.jsonl`, etc.
- Test sets: Frozen `test_set_2025_01.jsonl` (never changes for consistent evaluation)

---

## 13. APPENDIX

### 13.1 Keyword Lists (Abbreviated)

**Primary Keywords** (High Signal):
```yaml
primary:
  - "background check"
  - "background screening"
  - "employment screening"
  - "pre-employment"
  - "criminal record"
  - "criminal history check"
  - "Ban the Box"
  - "fair chance hiring"
  - "FCRA"
  - "consumer reporting agency"
  - "CRA compliance"
  - "right to work"
  - "work authorization"
  - "E-Verify"
  - "credential verification"
  - "education verification"
  - "adverse action"
```

**Secondary Keywords** (Supporting):
```yaml
secondary:
  - "drug testing"
  - "drug screening"
  - "reference check"
  - "identity verification"
  - "continuous monitoring"
  - "occupational screening"
  - "tenant screening"
  - "volunteer screening"
```

**Regional Legislation** (Triggers):
```yaml
legislation:
  us:
    - "FCRA"
    - "Title VII"
    - "EEOC"
    - "Ban the Box"
    - "E-Verify"
  eu:
    - "GDPR"
    - "Data Protection Directive"
    - "DPA"
  canada:
    - "PIPEDA"
    - "provincial privacy act"
  asia:
    - "PDPA" # Singapore, Malaysia
    - "DPDP" # India
    - "APPI" # Japan
  australia:
    - "Privacy Act 1988"
    - "Australian Privacy Principles"
```

### 13.2 Example RSS Sources (Tier 1)

```yaml
tier1_sources:
  - name: "JDSupra Labor & Employment"
    url: "https://www.jdsupra.com/rss/labor-employment.xml"
    language: "en"
    volume: "~80 articles/day"

  - name: "Lexology Employment Law"
    url: "https://www.lexology.com/rss/employment.xml"
    language: "en"
    volume: "~60 articles/day"

  - name: "National Law Review"
    url: "https://www.natlawreview.com/rss/employment-labor.xml"
    language: "en"
    volume: "~40 articles/day"

  - name: "HR Dive"
    url: "https://www.hrdive.com/feeds/news/"
    language: "en"
    volume: "~20 articles/day"

  - name: "EDPB News"
    url: "https://edpb.europa.eu/rss_en.xml"
    language: "en"
    volume: "~5 articles/day"
    region_hint: "europe"

  - name: "EEOC Newsroom"
    url: "https://www.eeoc.gov/rss/newsroom.xml"
    language: "en"
    volume: "~3 articles/day"
    region_hint: "north_america"
```

### 13.3 Research Citations

**DSPy Resources**:
- DSPy Documentation: https://dspy.ai
- BootstrapFewShot Guide: https://dspy.ai/api/optimizers/BootstrapFewShot
- MIPROv2 Paper: https://dspy.ai/api/optimizers/MIPROv2
- RAG Tutorial: https://dspy.ai/tutorials/rag

**Best Practices**:
- Perplexity Research: "Production DSPy pipelines for document classification"
- Weaviate Blog: "Your Language Model Deserves Better Prompting"
- Context7: DSPy framework documentation

### 13.4 Glossary

- **BootstrapFewShot**: DSPy optimizer that learns effective few-shot examples
- **ChainOfThought**: DSPy module that generates reasoning before answers
- **ChromaDB**: Open-source vector database for embeddings
- **ColBERT**: Contextualized late interaction over BERT (advanced retrieval)
- **MIPROv2**: DSPy optimizer for instruction + example joint optimization
- **RAG**: Retrieval-Augmented Generation (combining search with LLMs)
- **Signature**: DSPy's typed input/output specification for LLM tasks
- **TypedPredictor**: DSPy module for structured JSON output

---

## 14. NEXT STEPS & HANDOFF

### 14.1 Immediate Actions (Week 1)

1. **Project Setup**:
   - [ ] Create GitHub repository
   - [ ] Initialize Python project with Poetry
   - [ ] Set up development environment (Python 3.11+, DSPy, dependencies)
   - [ ] Configure OpenAI API keys (`.env` file)

2. **Data Collection**:
   - [ ] Export 100 recent articles from current newsletter archive
   - [ ] Set up RSS feed ingestion for 5 pilot sources
   - [ ] Create SQLite database with schema

3. **Baseline Implementation**:
   - [ ] Implement keyword pre-filter
   - [ ] Create basic ArticleClassifier signature
   - [ ] Test classification on 20 articles manually

### 14.2 Handoff to Implementation Team

**Primary Contact**: Project Manager (coordinate labeling with newsletter staff)

**Key Artifacts**:
1. This solution design document
2. Example DSPy code snippets (signatures, modules)
3. Database schema SQL
4. RSS source list (YAML)
5. Keyword lists (YAML)

**Critical Decisions Needed from Stakeholders**:
- [ ] Confirm PoC budget ($50/month acceptable?)
- [ ] Identify 2-3 staff members for labeling (4-6 hours/week commitment)
- [ ] Approve phased rollout plan (gradual vs immediate)
- [ ] Define success criteria for post-PoC production decision

**Support Resources**:
- DSPy Discord: https://discord.gg/dspy
- Weekly check-ins with architect (first month)
- On-call support for critical issues

### 14.3 Risk Flag for Leadership

⚠️ **Biggest Risk**: Staff labeling bottleneck (200-300 examples needed for good performance)
- **Recommendation**: Allocate 6-8 hours/week of staff time for first month
- **Alternative**: Consider hiring contractor for labeling if staff unavailable

✅ **Biggest Opportunity**: If successful, this system can scale to 1,000+ articles/day at <$50/month, 10x faster than manual process

---

**Document Version**: 1.0
**Author**: Architect 5 (Constraint Analysis Strategy)
**Date**: 2025-01-12
**Status**: Ready for Implementation

---

*End of Solution Design Document*
