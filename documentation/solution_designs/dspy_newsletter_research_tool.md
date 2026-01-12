# DSPy Newsletter Research Tool - Solution Design Document
## Architect 1: First Principles Analysis

**Version**: 1.0
**Date**: 2026-01-12
**Architect**: First Principles Reasoning Strategy
**Status**: Design Phase

---

## Executive Summary

This design transforms PreEmploymentDirectory's manual newsletter curation into an automated DSPy-based research pipeline. By decomposing the problem to fundamental truths—information retrieval, classification, relevance ranking, and semantic search—we build a modular system that achieves 70%+ recall while maintaining quality.

**Core Insight**: The newsletter problem is fundamentally a multi-label classification + semantic search problem. DSPy's declarative signatures and automatic optimization eliminate brittle prompt engineering, while RAG enables natural language querying over accumulated knowledge.

**Key Metrics**:
- **Throughput**: 500+ articles/day → 20-50 candidate articles/day (96-98% reduction)
- **Recall Target**: 70%+ vs manual process
- **Query Latency**: <5 seconds for CLI queries
- **Cost**: <$50/month API spend (PoC budget)

---

## 1. First Principles Decomposition

### What Must Be True?

**Atomic Requirements:**
1. **Articles exist** in 247+ sources (RSS, HTML, Email)
2. **Articles have characteristics** (region, topic, relevance)
3. **Characteristics can be inferred** from text (title + content)
4. **Similar articles exist** across sources (deduplication needed)
5. **Humans judge relevance** (need training data from manual process)
6. **Future queries are unpredictable** (need flexible RAG, not rigid filtering)

**Derived Constraints:**
- Classification must handle **multi-label** (article can be multiple topics + regions)
- Scoring must be **calibrated** (threshold determines 20-50 daily candidates)
- Storage must be **searchable** (vector embeddings for semantic retrieval)
- System must be **inspectable** (humans review outputs, need reasoning traces)
- Pipeline must be **resumable** (batch processing failures shouldn't lose progress)

---

## 2. System Architecture

### 2.1 ASCII Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INGESTION LAYER                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │   RSS    │  │ Scrapers │  │  Email   │  │ Social   │            │
│  │ Feeders  │  │  (PBSA,  │  │ Parsers  │  │  Media   │            │
│  │          │  │   DPAs)  │  │          │  │ Monitors │            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
│       │             │             │             │                   │
│       └─────────────┴─────────────┴─────────────┘                   │
│                           │                                          │
│                    ┌──────▼──────┐                                   │
│                    │  Raw Queue  │  (JSON files, resume-able)        │
│                    └──────┬──────┘                                   │
└───────────────────────────┼──────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────┐
│                    PROCESSING LAYER (DSPy)                            │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Deduplication Module                                        │    │
│  │  - Embedding-based similarity (>0.85 = duplicate)            │    │
│  │  - Signature: article_a, article_b -> is_duplicate: bool     │    │
│  └─────────────────────┬─────────────────────────────────────────┘   │
│                        │                                              │
│  ┌─────────────────────▼─────────────────────────────────────────┐  │
│  │  Classification Module (Multi-Label)                          │  │
│  │  - Signature: title, content -> region, topics[], confidence  │  │
│  │  - Module: ChainOfThought (reasoning for transparency)        │  │
│  │  - Optimization: BootstrapFewShot (labeled newsletter data)   │  │
│  └─────────────────────┬─────────────────────────────────────────┘  │
│                        │                                              │
│  ┌─────────────────────▼─────────────────────────────────────────┐  │
│  │  Relevance Scoring Module (Hybrid)                            │  │
│  │  - Keyword Match: Primary/Secondary keywords (fast filter)    │  │
│  │  - Semantic Score: Embedding similarity to "ideal" articles   │  │
│  │  - DSPy Scorer: ChainOfThought "article -> relevance: float"  │  │
│  │  - Combined Score: (0.3*keyword + 0.3*semantic + 0.4*LM)      │  │
│  └─────────────────────┬─────────────────────────────────────────┘  │
│                        │                                              │
│  ┌─────────────────────▼─────────────────────────────────────────┐  │
│  │  Summarization Module                                         │  │
│  │  - Signature: article -> summary (2-3 sentences), reasoning   │  │
│  │  - Module: Predict (fast, no CoT needed for summaries)        │  │
│  └─────────────────────┬─────────────────────────────────────────┘  │
└────────────────────────┼──────────────────────────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────────────────────────┐
│                     STORAGE LAYER                                     │
├────────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Vector Database (ChromaDB)                                  │    │
│  │  - Embeddings: all-MiniLM-L6-v2 (384 dim, fast, cheap)      │    │
│  │  - Collections: articles_2026_01, articles_2026_02, ...      │    │
│  │  - Metadata: region, topics, relevance_score, date, source   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Structured Storage (JSON/SQLite)                            │    │
│  │  - candidates_YYYY-MM-DD.json (daily shortlist)              │    │
│  │  - processing_log.jsonl (audit trail)                        │    │
│  └─────────────────────────────────────────────────────────────┘    │
└────────────────────────┬──────────────────────────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────────────────────────┐
│                     QUERY LAYER (RAG)                                 │
├────────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  CLI Interface                                               │    │
│  │  - Natural language queries: "APAC regulation changes"       │    │
│  │  - Filters: date range, region, topic, min_score             │    │
│  └─────────────────────┬─────────────────────────────────────────┘   │
│                        │                                              │
│  ┌─────────────────────▼─────────────────────────────────────────┐  │
│  │  RAG Module (NewsletterRAG)                                   │  │
│  │  - Retrieve: dspy.Retrieve(k=10) from ChromaDB               │  │
│  │  - Rerank: LM-based reranking (top 3)                        │  │
│  │  - Generate: ChainOfThought "context, question -> answer"    │  │
│  │  - Output: Ranked articles + synthesis                       │  │
│  └─────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

**Daily Batch Process:**
```
1. Ingestion (00:00 UTC)
   └─> Fetch 500+ articles from sources
   └─> Store in raw_queue/ directory

2. Deduplication (00:30 UTC)
   └─> Load articles from queue
   └─> Compute embeddings (batch)
   └─> Mark duplicates (keep earliest)

3. Classification (01:00 UTC)
   └─> Classify region + topics (batched)
   └─> Skip if confidence < 0.6

4. Relevance Scoring (02:00 UTC)
   └─> Keyword filter (fast rejection)
   └─> Semantic + LM scoring (survivors)
   └─> Threshold: keep relevance_score >= 0.7

5. Summarization (03:00 UTC)
   └─> Generate 2-3 sentence summaries
   └─> Only for candidates (20-50 articles)

6. Storage (04:00 UTC)
   └─> Add to ChromaDB with metadata
   └─> Write candidates_YYYY-MM-DD.json
   └─> Update processing_log.jsonl

7. Notification (04:30 UTC)
   └─> Email staff with candidate count
   └─> Link to review interface
```

**Query Process (on-demand):**
```
User: "Recent GDPR enforcement actions in Europe"
  │
  ├─> Query Parser: Extract filters (region=Europe, topic=regulatory, date=30d)
  │
  ├─> Retrieval: ChromaDB similarity search (k=10)
  │
  ├─> Reranking: LM scores 10 → top 3
  │
  ├─> Generation: Synthesize answer from top 3
  │
  └─> Output: Ranked list + narrative summary
```

---

## 3. DSPy Signatures (Implementation Specifications)

### 3.1 Deduplication Signature

```python
import dspy
from typing import Literal

class ArticleDuplication(dspy.Signature):
    """Determine if two articles cover the same story.

    Consider:
    - Core facts (who, what, when, where)
    - Ignore stylistic differences
    - Different angles on same event = duplicate
    """

    title_a = dspy.InputField(desc="First article title")
    content_a = dspy.InputField(desc="First article summary (first 500 chars)")
    source_a = dspy.InputField(desc="First article source")

    title_b = dspy.InputField(desc="Second article title")
    content_b = dspy.InputField(desc="Second article summary (first 500 chars)")
    source_b = dspy.InputField(desc="Second article source")

    is_duplicate = dspy.OutputField(
        desc="true if same story, false if different stories"
    )
    reasoning = dspy.OutputField(
        desc="Brief explanation of why duplicate or not"
    )
```

**Module Choice:** `dspy.Predict` (fast, no reasoning needed in output)
**Why:** Deduplication runs on all pairs, speed critical. Reasoning field for debugging only.

### 3.2 Classification Signature

```python
from typing import List, Literal

RegionType = Literal[
    "africa_middle_east",
    "asia_pacific",
    "europe",
    "north_america_caribbean",
    "south_america",
    "worldwide"
]

TopicType = Literal[
    "regulatory_legal",
    "criminal_background_checks",
    "education_verification",
    "immigration_right_to_work",
    "industry_ma_news",
    "technology_products",
    "conferences_events",
    "court_cases_precedents"
]

class ArticleClassification(dspy.Signature):
    """Classify background screening industry articles by region and topics.

    Region: Primary geographic focus (choose ONE most relevant)
    Topics: All applicable categories (multi-select)

    Context: Background screening = pre-employment checks, criminal records,
    credential verification, identity verification, drug testing, reference checks.
    """

    title = dspy.InputField(desc="Article title")
    content = dspy.InputField(desc="Article content (first 2000 chars)")
    source = dspy.InputField(desc="Publication source")

    region = dspy.OutputField(desc="Primary region (one of the 6 regions)")
    topics = dspy.OutputField(desc="List of applicable topics (1-3 typically)")
    confidence = dspy.OutputField(
        desc="Confidence score 0.0-1.0 for classification quality"
    )
    reasoning = dspy.OutputField(
        desc="Brief explanation citing key phrases from article"
    )
```

**Module Choice:** `dspy.ChainOfThought` (reasoning transparency critical)
**Why:** Humans review outputs—need to see WHY article was classified this way.

### 3.3 Relevance Scoring Signature

```python
class RelevanceScoring(dspy.Signature):
    """Score article relevance to background screening industry (0.0-1.0).

    High Relevance (0.8-1.0):
    - New laws/regulations affecting background checks
    - Court cases setting precedents
    - Major industry M&A or company news

    Medium Relevance (0.5-0.7):
    - General employment law with screening implications
    - Technology developments in verification
    - Industry conference announcements

    Low Relevance (0.0-0.4):
    - Tangentially related HR topics
    - Generic legal news without screening connection
    """

    title = dspy.InputField()
    content = dspy.InputField(desc="First 1000 chars")
    region = dspy.InputField(desc="Classified region")
    topics = dspy.InputField(desc="Classified topics")

    relevance_score = dspy.OutputField(
        desc="Float 0.0-1.0, calibrated to industry importance"
    )
    key_signals = dspy.OutputField(
        desc="List of 2-3 phrases that drove the score"
    )
    reasoning = dspy.OutputField(
        desc="Why this score? What would make it higher/lower?"
    )
```

**Module Choice:** `dspy.ChainOfThought` (calibration requires reasoning)
**Why:** Score threshold determines daily volume (20-50 target). Need transparency.

### 3.4 Summarization Signature

```python
class ArticleSummarization(dspy.Signature):
    """Summarize background screening article for industry professionals.

    Focus on: Who, what, when, where, and practical implications.
    Omit: Boilerplate, author bios, promotional content.
    Length: 2-3 sentences, ~50 words.
    """

    title = dspy.InputField()
    content = dspy.InputField(desc="Full article text")
    region = dspy.InputField()
    topics = dspy.InputField()

    summary = dspy.OutputField(
        desc="2-3 sentence summary hitting key facts and implications"
    )
```

**Module Choice:** `dspy.Predict` (speed over reasoning for summaries)
**Why:** Summarization is straightforward—don't need CoT overhead.

### 3.5 RAG Query Signature

```python
class NewsletterQuery(dspy.Signature):
    """Answer questions about background screening industry news using retrieved articles.

    Synthesize information from multiple articles.
    Cite sources by title and date.
    If uncertain, say so—don't hallucinate.
    """

    question = dspy.InputField(desc="Natural language question from user")
    context = dspy.InputField(desc="Retrieved article summaries (top 3-10)")

    answer = dspy.OutputField(
        desc="Direct answer to question with citations"
    )
    confidence = dspy.OutputField(
        desc="How well context supports the answer (low/medium/high)"
    )
    related_articles = dspy.OutputField(
        desc="List of article titles and dates cited in answer"
    )
```

**Module Choice:** `dspy.ChainOfThought` (complex synthesis requires reasoning)
**Why:** Users need trustworthy answers—CoT improves accuracy and transparency.

---

## 4. Module Selection Rationale

### 4.1 Deduplication Module

```python
class DeduplicationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.compare = dspy.Predict(ArticleDuplication)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def forward(self, article_a, article_b):
        # Fast filter: embedding similarity
        emb_a = self.embedder.encode(article_a['title'])
        emb_b = self.embedder.encode(article_b['title'])
        similarity = cosine_similarity(emb_a, emb_b)

        # If obviously different (< 0.5), skip LM call
        if similarity < 0.5:
            return dspy.Prediction(is_duplicate=False, reasoning="Low title similarity")

        # If very similar (> 0.85), likely duplicate
        if similarity > 0.85:
            # Confirm with LM for edge cases
            result = self.compare(
                title_a=article_a['title'],
                content_a=article_a['content'][:500],
                source_a=article_a['source'],
                title_b=article_b['title'],
                content_b=article_b['content'][:500],
                source_b=article_b['source']
            )
            return result

        # Medium similarity: definitely check with LM
        return self.compare(
            title_a=article_a['title'],
            content_a=article_a['content'][:500],
            source_a=article_a['source'],
            title_b=article_b['title'],
            content_b=article_b['content'][:500],
            source_b=article_b['source']
        )
```

**Design Decision:** Hybrid approach (embeddings + LM)
- Embeddings filter obvious non-duplicates (save API costs)
- LM handles ambiguous cases (same event, different angle)
- Optimization: BootstrapFewShot with labeled duplicate pairs

### 4.2 Classification Module

```python
class ClassificationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classifier = dspy.ChainOfThought(ArticleClassification)

    def forward(self, title, content, source):
        result = self.classifier(
            title=title,
            content=content[:2000],  # Token limit
            source=source
        )

        # Post-processing: validate region and topics
        valid_regions = {
            "africa_middle_east", "asia_pacific", "europe",
            "north_america_caribbean", "south_america", "worldwide"
        }
        valid_topics = {
            "regulatory_legal", "criminal_background_checks",
            "education_verification", "immigration_right_to_work",
            "industry_ma_news", "technology_products",
            "conferences_events", "court_cases_precedents"
        }

        # Validate and normalize
        if result.region not in valid_regions:
            result.region = "worldwide"  # Fallback

        result.topics = [t for t in result.topics if t in valid_topics]

        # Confidence check: retry if too low
        if float(result.confidence) < 0.6:
            # Second pass with more context
            result = self.classifier(
                title=title,
                content=content[:4000],  # More context
                source=source
            )

        return result
```

**Design Decision:** ChainOfThought with validation
- Reasoning field critical for debugging misclassifications
- Confidence threshold guards against garbage
- Retry mechanism for low-confidence cases
- Optimization: BootstrapFewShot on manually labeled newsletters

### 4.3 Relevance Scoring Module (Hybrid)

```python
class RelevanceModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.lm_scorer = dspy.ChainOfThought(RelevanceScoring)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Pre-computed embeddings of "ideal" high-relevance articles
        self.ideal_embeddings = self._load_ideal_embeddings()

        # Keyword dictionaries
        self.primary_keywords = {
            "background check", "background screening", "employment screening",
            "pre-employment", "criminal record", "criminal history",
            "right to work", "work authorization", "ban the box",
            "fair chance", "fcra", "gdpr", "data protection",
            "credential verification", "education verification"
        }
        self.secondary_keywords = {
            "drug testing", "drug screening", "reference check",
            "identity verification", "continuous monitoring",
            "adverse action", "consumer reporting agency"
        }

    def forward(self, title, content, region, topics):
        # Component 1: Keyword Score (0.0-1.0)
        keyword_score = self._keyword_score(title, content)

        # Component 2: Semantic Score (0.0-1.0)
        semantic_score = self._semantic_score(title, content)

        # Component 3: LM Score (0.0-1.0)
        # Only call LM if keyword OR semantic suggests relevance
        if keyword_score > 0.3 or semantic_score > 0.4:
            lm_result = self.lm_scorer(
                title=title,
                content=content[:1000],
                region=region,
                topics=topics
            )
            lm_score = float(lm_result.relevance_score)
        else:
            lm_score = 0.0  # Skip LM call for obvious misses
            lm_result = dspy.Prediction(
                relevance_score=0.0,
                key_signals=["No primary keywords found"],
                reasoning="Failed keyword filter"
            )

        # Weighted combination
        final_score = (
            0.3 * keyword_score +
            0.3 * semantic_score +
            0.4 * lm_score
        )

        return dspy.Prediction(
            relevance_score=final_score,
            keyword_score=keyword_score,
            semantic_score=semantic_score,
            lm_score=lm_score,
            key_signals=lm_result.key_signals,
            reasoning=lm_result.reasoning
        )

    def _keyword_score(self, title, content):
        text = (title + " " + content).lower()
        primary_matches = sum(1 for kw in self.primary_keywords if kw in text)
        secondary_matches = sum(1 for kw in self.secondary_keywords if kw in text)

        # Scoring: primary keywords weighted higher
        score = min(1.0, (primary_matches * 0.3) + (secondary_matches * 0.1))
        return score

    def _semantic_score(self, title, content):
        # Embed article
        article_emb = self.embedder.encode(title + " " + content[:500])

        # Max similarity to ideal articles
        similarities = [
            cosine_similarity(article_emb, ideal_emb)
            for ideal_emb in self.ideal_embeddings
        ]
        return max(similarities) if similarities else 0.0
```

**Design Decision:** Three-stage hybrid
- Keyword filter catches obvious hits/misses (cheap, fast)
- Semantic score handles synonyms and paraphrasing
- LM score calibrates final decision (expensive, accurate)
- Weighted combination optimized via validation set
- Optimization: MIPRO on labeled relevance data

### 4.4 RAG Module

```python
class NewsletterRAG(dspy.Module):
    def __init__(self, k=10):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=k)
        self.rerank = dspy.Predict("question, passage -> relevance_score: float")
        self.generate = dspy.ChainOfThought(NewsletterQuery)

    def forward(self, question, filters=None):
        # Retrieve candidates from ChromaDB
        retrieved = self.retrieve(question)
        passages = retrieved.passages

        # Apply metadata filters (date, region, topic)
        if filters:
            passages = self._apply_filters(passages, filters)

        # Rerank with LM
        scored_passages = []
        for passage in passages:
            score = float(self.rerank(
                question=question,
                passage=passage
            ).relevance_score)
            scored_passages.append((score, passage))

        # Take top 3
        top_passages = [p for _, p in sorted(scored_passages, reverse=True)[:3]]
        context = "\n\n---\n\n".join(top_passages)

        # Generate answer
        return self.generate(question=question, context=context)
```

**Design Decision:** Retrieve → Rerank → Generate
- Initial retrieval casts wide net (k=10)
- Reranking narrows to most relevant (top 3)
- ChainOfThought generates synthesis with citations
- Optimization: BootstrapFewShot on Q&A pairs from staff

---

## 5. Optimization Strategy

### 5.1 Training Data Collection

**Phase 1: Bootstrap from Manual Process (Weeks 1-2)**
```
Source: Staff's existing Dropbox + ChatGPT workflow

Collection:
1. Export last 60 days of manually curated newsletters
2. Extract articles staff INCLUDED (positive examples)
3. Sample articles staff REJECTED (negative examples)
4. Label with:
   - Region (ground truth from manual categorization)
   - Topics (ground truth)
   - Relevance (1.0 for included, 0.0-0.3 for rejected)
   - Reasoning (why included/rejected - from staff notes)

Target: 100-200 labeled articles
```

**Phase 2: Active Learning (Weeks 3-4)**
```
Process:
1. Run pipeline on new articles
2. Staff reviews candidates (20-50/day)
3. Mark: correct classification / incorrect / missed
4. Feed back into training set
5. Re-optimize weekly

Target: 50-100 additional labeled articles/week
```

### 5.2 Optimizer Selection by Module

| Module | Optimizer | Rationale |
|--------|-----------|-----------|
| **Deduplication** | BootstrapFewShot | Binary task, 50 labeled pairs sufficient |
| **Classification** | BootstrapFewShot → MIPRO | Start fast, upgrade when data > 100 examples |
| **Relevance Scoring** | MIPRO | Critical calibration, worth 100-trial search |
| **Summarization** | KNNFewShot | Quality less critical, speed matters |
| **RAG Query** | BootstrapFewShot | Staff Q&A pairs easy to generate |

### 5.3 Metrics Definition

```python
# Classification Metric
def classification_accuracy(example, pred, trace=None):
    """Multi-label accuracy with partial credit."""
    region_match = 1.0 if example.region == pred.region else 0.0

    # Jaccard similarity for topics
    gold_topics = set(example.topics)
    pred_topics = set(pred.topics)
    if not gold_topics and not pred_topics:
        topic_match = 1.0
    elif not gold_topics or not pred_topics:
        topic_match = 0.0
    else:
        topic_match = len(gold_topics & pred_topics) / len(gold_topics | pred_topics)

    return 0.5 * region_match + 0.5 * topic_match

# Relevance Metric
def relevance_calibration(example, pred, trace=None):
    """Mean absolute error for score calibration."""
    gold_score = float(example.relevance_score)
    pred_score = float(pred.relevance_score)
    return 1.0 - abs(gold_score - pred_score)  # Higher is better

# RAG Metric
def answer_quality(example, pred, trace=None):
    """Composite metric: correctness + citation."""
    # Check if key facts in answer
    gold_facts = set(example.key_facts)  # Staff-provided
    pred_text = pred.answer.lower()

    facts_covered = sum(1 for fact in gold_facts if fact.lower() in pred_text)
    correctness = facts_covered / len(gold_facts) if gold_facts else 0.0

    # Check if cited sources
    has_citations = len(pred.related_articles) > 0
    citation_bonus = 0.2 if has_citations else 0.0

    return min(1.0, correctness + citation_bonus)
```

### 5.4 Optimization Workflow

```python
# Step 1: Baseline (No Optimization)
classifier_baseline = dspy.ChainOfThought(ArticleClassification)
baseline_score = evaluate(classifier_baseline, testset)  # ~0.60-0.65 expected

# Step 2: BootstrapFewShot (Fast First Pass)
bootstrap = BootstrapFewShot(
    metric=classification_accuracy,
    max_bootstrapped_demos=4,
    max_rounds=2
)
classifier_v1 = bootstrap.compile(classifier_baseline, trainset=trainset)
v1_score = evaluate(classifier_v1, testset)  # ~0.75-0.80 expected

# Step 3: MIPRO (When Data > 100)
if len(trainset) >= 100:
    mipro = MIPRO(
        metric=classification_accuracy,
        num_candidates=10,
        verbose=True
    )
    classifier_v2 = mipro.compile(
        student=classifier_baseline,
        trainset=trainset,
        valset=valset,
        num_trials=100
    )
    v2_score = evaluate(classifier_v2, testset)  # ~0.82-0.87 expected

    # Save best
    if v2_score > v1_score:
        classifier_v2.save("models/classifier_v2.json")
```

---

## 6. RAG Implementation

### 6.1 Vector Database Setup

```python
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB
chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_db",
    anonymized_telemetry=False
))

# Create collection (monthly rotation for manageability)
collection = chroma_client.create_collection(
    name="articles_2026_01",
    metadata={"description": "Newsletter articles January 2026"}
)

# Embedding model (cheap, fast, good enough)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Add articles
def add_to_index(article):
    # Embed title + summary (not full content - too noisy)
    text = f"{article['title']} {article['summary']}"
    embedding = embedder.encode(text).tolist()

    collection.add(
        ids=[article['id']],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{
            'title': article['title'],
            'url': article['url'],
            'date': article['published_date'],
            'region': article['region'],
            'topics': ','.join(article['topics']),
            'relevance_score': article['relevance_score'],
            'source': article['source']
        }]
    )
```

### 6.2 DSPy ChromaDB Integration

```python
from dspy.retrieve.chromadb_rm import ChromadbRM

# Configure retriever
retriever = ChromadbRM(
    collection_name='articles_2026_01',
    persist_directory='./chroma_db',
    k=10  # Retrieve top 10 by default
)

# Set as default retriever
dspy.settings.configure(rm=retriever)

# Now dspy.Retrieve() uses ChromaDB automatically
```

### 6.3 Query Interface

```python
class NewsletterCLI:
    def __init__(self, rag_module):
        self.rag = rag_module

    def query(self, question, region=None, topic=None, days=30, min_score=0.7):
        """
        Natural language query with optional filters.

        Examples:
        - "Recent GDPR enforcement actions"
        - "Ban the box legislation in North America"
        - "Criminal background check court cases"
        """
        # Build filters
        filters = {
            'date_min': (datetime.now() - timedelta(days=days)).isoformat(),
            'relevance_score_min': min_score
        }
        if region:
            filters['region'] = region
        if topic:
            filters['topic'] = topic

        # Query RAG
        result = self.rag(question=question, filters=filters)

        # Format output
        print(f"\n📰 Query: {question}\n")
        print(f"🔍 Answer:\n{result.answer}\n")
        print(f"📊 Confidence: {result.confidence}\n")
        print(f"📚 Sources ({len(result.related_articles)}):")
        for article in result.related_articles:
            print(f"  - {article}")

        return result

# Usage
cli = NewsletterCLI(optimized_rag)
cli.query(
    "What are recent regulation changes in APAC?",
    region="asia_pacific",
    topic="regulatory_legal",
    days=30
)
```

### 6.4 Metadata Filtering

```python
def apply_filters(passages, filters):
    """Apply metadata filters to retrieved passages."""
    filtered = []

    for passage in passages:
        metadata = passage.metadata

        # Date filter
        if 'date_min' in filters:
            article_date = datetime.fromisoformat(metadata['date'])
            filter_date = datetime.fromisoformat(filters['date_min'])
            if article_date < filter_date:
                continue

        # Region filter
        if 'region' in filters:
            if metadata['region'] != filters['region']:
                continue

        # Topic filter
        if 'topic' in filters:
            article_topics = metadata['topics'].split(',')
            if filters['topic'] not in article_topics:
                continue

        # Relevance score filter
        if 'relevance_score_min' in filters:
            if float(metadata['relevance_score']) < filters['relevance_score_min']:
                continue

        filtered.append(passage)

    return filtered
```

---

## 7. Phased Implementation Plan

### Phase 1: Foundation (Weeks 1-3)
**Goal:** Prove core classification works

**Deliverables:**
1. **Ingestion Pipeline**
   - RSS feed parser for 20 Tier 1 sources
   - JSON storage in `raw_queue/`
   - Daily cron job (manual trigger for PoC)

2. **Classification Module**
   - ArticleClassification signature
   - ChainOfThought module
   - Manual testing on 50 articles
   - Output: region + topics + confidence

3. **Training Data Collection**
   - Export 100 manually curated articles from staff
   - Label ground truth (region, topics)
   - Store in `training_data/labeled_articles.json`

4. **Validation:**
   - Manual review of 50 classified articles
   - Target: 70%+ accuracy on region, 60%+ on topics
   - Iterate on signature descriptions if needed

**Success Criteria:**
- Can ingest 100+ articles/day from RSS
- Classification accuracy > 70% on region
- Staff can understand reasoning field

**Risks:**
- LM may hallucinate regions/topics not in taxonomy
  - Mitigation: Post-processing validation (implemented)
- API costs exceed budget
  - Mitigation: Batch processing, cache embeddings

---

### Phase 2: Relevance & Optimization (Weeks 4-6)
**Goal:** Filter to 20-50 daily candidates with high recall

**Deliverables:**
1. **Relevance Scoring Module**
   - Hybrid keyword + semantic + LM scorer
   - Calibrate threshold to produce 20-50/day
   - Testing: Run on 7 days of historical articles

2. **Deduplication Module**
   - Embedding + LM hybrid
   - Remove duplicates before classification (save costs)

3. **Optimization**
   - BootstrapFewShot on classification (100 examples)
   - MIPRO on relevance scoring (if 100+ examples)
   - Save optimized models to `models/`

4. **Evaluation Framework**
   - Metrics: classification_accuracy, relevance_calibration
   - Test set: 30 held-out articles
   - Track: precision, recall, daily candidate count

**Success Criteria:**
- 70%+ recall vs manual process (capture most important articles)
- 20-50 candidates/day on average (not 200, not 5)
- False positive rate < 30% (staff can handle 30% misses)

**Risks:**
- Recall too low (miss important articles)
  - Mitigation: Lower relevance threshold, bias toward over-inclusion
- Precision too low (too many candidates)
  - Mitigation: Raise threshold, add more negative training examples

---

### Phase 3: RAG & Storage (Weeks 7-9)
**Goal:** Enable natural language queries over accumulated articles

**Deliverables:**
1. **ChromaDB Setup**
   - Initialize database with all-MiniLM-L6-v2 embeddings
   - Load 30 days of processed articles
   - Monthly collection rotation

2. **RAG Module**
   - Retrieve → Rerank → Generate pipeline
   - NewsletterQuery signature
   - Metadata filtering (region, topic, date)

3. **CLI Interface**
   - `python cli.py query "recent APAC regulations"`
   - Optional flags: --region, --topic, --days, --min-score
   - Output: ranked articles + synthesis

4. **Optimization**
   - Collect 20 Q&A pairs from staff (typical queries)
   - BootstrapFewShot on NewsletterQuery
   - Metric: answer_quality

**Success Criteria:**
- Queries return results in < 5 seconds
- Top 3 results are relevant (subjective staff review)
- Generated answers are factual (no hallucinations)

**Risks:**
- Retrieval returns off-topic articles
  - Mitigation: Metadata filtering, reranking
- Generated answers hallucinate
  - Mitigation: ChainOfThought for transparency, confidence field

---

### Phase 4: Production Hardening (Weeks 10-12)
**Goal:** Reliable daily operation, monitoring, staff handoff

**Deliverables:**
1. **Batch Processing**
   - Resumable pipeline (checkpoint after each stage)
   - Error handling (log failures, continue processing)
   - Daily email report (candidate count, errors)

2. **Monitoring Dashboard**
   - Daily metrics: articles processed, candidates, errors
   - Weekly review: precision/recall trends
   - Model drift detection (confidence score distribution)

3. **Summarization Module**
   - Add ArticleSummarization (only for candidates)
   - KNNFewShot optimization (fast, good enough)

4. **Documentation**
   - Setup guide for staff
   - Troubleshooting runbook
   - How to add new sources
   - How to retrain models (monthly recommended)

**Success Criteria:**
- Pipeline runs unattended for 7 days
- Zero critical errors (processing failures)
- Staff can query and review candidates without dev support

**Risks:**
- Source changes break parsers
  - Mitigation: Error handling, fallback to raw text
- Model drift over time
  - Mitigation: Monthly retraining schedule

---

## 8. Risks and Mitigations

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **LM Hallucination** | Medium | High | - ChainOfThought for transparency<br>- Confidence scores<br>- Human review of candidates<br>- Assertions on output format |
| **API Cost Overrun** | Medium | Medium | - Batch processing (10 articles/call)<br>- Keyword pre-filter (skip LM for obvious misses)<br>- Cache embeddings<br>- Monthly budget alerts |
| **Poor Recall** | High | High | - Bias toward over-inclusion (lower thresholds)<br>- Weekly review with staff<br>- Active learning loop<br>- Ensemble voting for edge cases |
| **Deduplication Failures** | Low | Low | - Hybrid approach (embeddings + LM)<br>- Manual review catches duplicates<br>- Not critical (prefer duplicates over misses) |
| **Source Breakage** | Medium | Medium | - Robust error handling<br>- Fallback to raw text extraction<br>- Alert on parsing failures<br>- Monthly source health check |
| **Model Drift** | Low | Medium | - Monthly retraining schedule<br>- Monitor confidence score distribution<br>- A/B test new models vs production |

### 8.2 Data Quality Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Insufficient Training Data** | Medium | High | - Start with 100 from manual process<br>- Active learning (50/week growth)<br>- Synthetic data generation (GPT-4 creates examples) |
| **Label Noise** | Medium | Medium | - Staff consensus (2+ reviewers)<br>- Confidence scores filter uncertain<br>- Outlier detection in training |
| **Distribution Shift** | Low | Medium | - Monthly retraining<br>- Track topic frequency over time<br>- Alert on sudden shifts |

### 8.3 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Staff Adoption Failure** | Low | High | - Involve staff in design<br>- Weekly review sessions<br>- CLI + email (familiar workflows)<br>- Clear value prop (20hrs/week saved) |
| **Maintenance Burden** | Medium | Medium | - Automate retraining<br>- Self-serve query interface<br>- Runbook for common issues<br>- Quarterly model updates |
| **Scalability** | Low | Low | - ChromaDB scales to millions<br>- Batch processing handles 1000+/day<br>- Not real-time (daily batch ok) |

---

## 9. Success Metrics & Monitoring

### 9.1 Quantitative Metrics

**Daily Metrics:**
```json
{
  "date": "2026-01-12",
  "articles_ingested": 523,
  "duplicates_removed": 87,
  "articles_classified": 436,
  "low_confidence_rejected": 52,
  "candidates_generated": 34,
  "avg_relevance_score": 0.78,
  "processing_time_minutes": 45,
  "api_calls": 1250,
  "api_cost_usd": 3.42
}
```

**Weekly Metrics:**
```json
{
  "week": "2026-W02",
  "recall_vs_manual": 0.73,  # Staff marks what was missed
  "precision": 0.68,          # Of 200 candidates, 136 were good
  "avg_candidates_per_day": 28.6,
  "queries_executed": 15,
  "query_satisfaction": 4.2   # Staff rating 1-5
}
```

### 9.2 Qualitative Metrics

**Staff Feedback (Weekly Survey):**
- Are candidates relevant? (1-5)
- How many important articles did we miss? (count)
- Query results helpful? (1-5)
- Time saved vs manual process? (hours/week)

### 9.3 Monitoring Dashboard

```
Newsletter Research Pipeline - Dashboard
=========================================

📊 Last 7 Days Overview
-----------------------
Articles Processed: 3,240
Candidates Generated: 198 (28/day avg)
Precision: 68% ±5%
Recall: 73% ±4%
Time Saved: 18 hours/week

🔍 Model Performance
--------------------
Classification Accuracy: 82%
Relevance Calibration MAE: 0.12
Query Satisfaction: 4.2/5.0

💰 Costs
--------
API Calls: 8,750
Cost: $24.50 (49% of budget)

⚠️ Alerts
---------
- None

📈 Trends
---------
[Chart: Daily candidate count - stable 25-35]
[Chart: Confidence score distribution - normal]
[Chart: Topic frequency - stable]
```

---

## 10. Cost Analysis

### 10.1 PoC Budget Breakdown

**API Costs (Anthropic Claude Sonnet 4.5):**
```
Daily Processing:
- Classification: 400 articles × $0.003/call = $1.20
- Relevance Scoring: 200 articles × $0.003/call = $0.60
- Deduplication: 100 pairs × $0.002/call = $0.20
- Summarization: 30 candidates × $0.002/call = $0.06
- Total/day: ~$2.06

Monthly: $2.06 × 30 = $61.80

Optimization (one-time):
- BootstrapFewShot: 200 calls × $0.003 = $0.60
- MIPRO: 1000 calls × $0.003 = $3.00
- Total: $3.60

Queries (on-demand):
- RAG: 20 queries/week × $0.005 = $0.10/week
- Monthly: $0.40

Total Monthly: $61.80 + $0.40 = $62.20
```

**Infrastructure:**
- ChromaDB: Free (local)
- SentenceTransformer: Free (local)
- Storage: <1GB/month (negligible)

**Total PoC Cost: ~$65/month**

### 10.2 Cost Optimizations

**If Budget Exceeded:**
1. Reduce batch size (process fewer articles)
2. Raise keyword filter threshold (skip more LM calls)
3. Use cheaper model for classification (GPT-3.5-turbo)
4. Cache frequent queries
5. Monthly processing instead of daily

**Target: <$50/month**

---

## 11. Future Enhancements (Post-PoC)

### 11.1 Enhanced Features

**Priority 1 (Weeks 13-16):**
- Web scraping for Tier 3 sources (PBSA, DPAs)
- Email integration (IMAP parser for newsletters)
- Multi-language support (detect + translate)
- Notification system (email digest, Slack integration)

**Priority 2 (Months 4-6):**
- Fine-tuning custom model (if BootstrapFinetune shows promise)
- Trend detection (emerging topics, sentiment shifts)
- Source quality scoring (weight by reliability)
- Automated newsletter generation (draft emails for staff)

**Priority 3 (Months 7+):**
- Real-time processing (WebSocket for live articles)
- Mobile app for on-the-go queries
- Integration with CRM (article → customer intelligence)
- Competitive intelligence tracking

### 11.2 Scalability Path

**From 500 articles/day → 5,000 articles/day:**
- Migrate ChromaDB to cloud (Chroma Cloud or Pinecone)
- Horizontal scaling (multiple workers)
- Caching layer (Redis for frequent queries)
- Model distillation (faster inference)

**From PoC → Production:**
- CI/CD pipeline (automated testing, deployment)
- Infrastructure as code (Terraform/Pulumi)
- Monitoring (Prometheus + Grafana)
- Incident response playbook

---

## 12. Technical Dependencies

### 12.1 Required Libraries

```python
# Core
dspy-ai>=2.5.0
openai>=1.0.0
anthropic>=0.25.0

# Embeddings & Retrieval
sentence-transformers>=2.2.0
chromadb>=0.4.0

# Utilities
pydantic>=2.0.0
feedparser>=6.0.0  # RSS parsing
beautifulsoup4>=4.12.0  # HTML scraping
requests>=2.31.0
python-dateutil>=2.8.0

# Data Science
numpy>=1.24.0
scikit-learn>=1.3.0  # For cosine similarity

# CLI
click>=8.1.0
rich>=13.0.0  # Pretty terminal output

# Monitoring
loguru>=0.7.0
```

### 12.2 System Requirements

**Minimum:**
- Python 3.10+
- 8GB RAM (for embeddings)
- 10GB disk (ChromaDB storage)
- Internet (API calls)

**Recommended:**
- Python 3.11+
- 16GB RAM
- SSD storage
- Stable internet (batch processing can retry)

---

## 13. Acceptance Criteria

### 13.1 Phase 1 Completion

- [ ] Ingest 100+ articles/day from 20 RSS sources
- [ ] Classify articles with 70%+ accuracy on region
- [ ] Classification reasoning is human-readable
- [ ] Can process 500 articles in < 2 hours
- [ ] API costs < $5/day

### 13.2 Phase 2 Completion

- [ ] Generate 20-50 candidates/day on average
- [ ] Recall ≥ 70% vs manual process (staff validation)
- [ ] Precision ≥ 60% (most candidates are relevant)
- [ ] Deduplication removes 80%+ of obvious duplicates
- [ ] Models save/load successfully

### 13.3 Phase 3 Completion

- [ ] Query interface returns results in < 5 seconds
- [ ] Top 3 results are relevant (staff subjective review)
- [ ] ChromaDB contains 30+ days of articles
- [ ] Metadata filtering works (region, topic, date)
- [ ] Generated answers cite sources

### 13.4 Phase 4 Completion

- [ ] Pipeline runs unattended for 7 days
- [ ] Zero critical processing failures
- [ ] Staff can query without dev support
- [ ] Documentation complete (setup + runbook)
- [ ] Monthly retraining process documented

---

## 14. Appendix

### 14.1 Alternative Architectures Considered

**Option A: Rule-Based Classifier (Rejected)**
- Pros: Fast, interpretable, no training data needed
- Cons: Brittle, requires constant maintenance, poor recall
- **Why Rejected:** Handoff doc explicitly requests LM-based approach

**Option B: Fine-Tuned Model (Deferred)**
- Pros: Fastest inference, best accuracy potential
- Cons: Requires 1000+ examples, expensive training, inflexible
- **Why Deferred:** PoC should prove value first, then consider fine-tuning

**Option C: Agent-Based Pipeline (Rejected)**
- Pros: Flexible, handles edge cases
- Cons: Slow, expensive, unpredictable
- **Why Rejected:** Batch processing doesn't need agent autonomy

**Option D: Keyword-Only Filtering (Rejected)**
- Pros: Free, instant
- Cons: Terrible recall, no semantic understanding
- **Why Rejected:** Fails "recent regulation changes in APAC" query test

### 14.2 Research References

**DSPy Best Practices:**
- Stanford NLP Blog: "Optimizing Language Model Pipelines"
- DSPy Discord #optimization channel (100+ examples)
- Paper: "DSPy: Compiling Declarative LM Calls into Self-Improving Pipelines"

**RAG Techniques:**
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al.)
- "Lost in the Middle: How Language Models Use Long Contexts" (Liu et al.)
- ChromaDB Documentation: Best Practices for Metadata Filtering

**Optimization:**
- "BootstrapFewShot vs MIPRO: When to Use Which?" (DSPy docs)
- "Prompt Optimization via Iterative Refinement" (Zhou et al.)

### 14.3 Code Repository Structure

```
newsletter-research-tool/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   ├── sources.yaml           # RSS feeds, scraping targets
│   ├── keywords.yaml          # Primary/secondary keywords
│   └── settings.yaml          # LM config, thresholds
├── src/
│   ├── ingestion/
│   │   ├── rss_parser.py
│   │   ├── scraper.py
│   │   └── email_parser.py
│   ├── processing/
│   │   ├── deduplication.py   # DeduplicationModule
│   │   ├── classification.py  # ClassificationModule
│   │   ├── scoring.py         # RelevanceModule
│   │   └── summarization.py   # SummarizationModule
│   ├── storage/
│   │   ├── chromadb_manager.py
│   │   └── json_storage.py
│   ├── rag/
│   │   ├── retriever.py
│   │   ├── rag_module.py      # NewsletterRAG
│   │   └── cli.py             # NewsletterCLI
│   ├── optimization/
│   │   ├── metrics.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── utils/
│       ├── logging.py
│       └── monitoring.py
├── models/                    # Saved DSPy modules
├── data/
│   ├── raw_queue/            # Ingested articles
│   ├── training_data/        # Labeled examples
│   └── candidates/           # Daily outputs
├── chroma_db/                # ChromaDB storage
├── logs/                     # Processing logs
└── scripts/
    ├── run_daily_batch.sh
    ├── optimize_models.py
    └── query_cli.py
```

---

## Conclusion

This DSPy-based architecture transforms PreEmploymentDirectory's manual newsletter curation into an automated, optimizable pipeline. By decomposing the problem into atomic components—deduplication, classification, scoring, summarization, and retrieval—we create a modular system that can be independently optimized and monitored.

**Key Innovations:**
1. **Hybrid Scoring:** Keyword + semantic + LM combines speed, cost-efficiency, and accuracy
2. **Transparent Reasoning:** ChainOfThought makes classifications explainable for human review
3. **Automatic Optimization:** DSPy teleprompters eliminate manual prompt engineering
4. **Flexible RAG:** Natural language queries with metadata filtering enable ad-hoc research

**Success Metrics:**
- 70%+ recall vs manual process
- 20-50 daily candidates (96-98% reduction in noise)
- <5 second query latency
- <$50/month PoC costs

**Next Steps:**
1. Review this design with stakeholders
2. Set up development environment
3. Begin Phase 1: Foundation (Weeks 1-3)
4. Collect training data from manual process
5. Iterate based on staff feedback

This design provides a clear roadmap from PoC to production, with defined success criteria, risk mitigations, and scalability paths. The phased approach ensures early validation of core assumptions while maintaining flexibility to adapt based on real-world performance.

---

**Document Status:** Ready for Review
**Approval Required From:** Product Owner, Technical Lead, Staff Users
**Next Review Date:** 2026-01-19 (1 week)
