# Product Requirements Document: DSPy Newsletter Research Tool

**Project**: PreEmploymentDirectory Automated Newsletter Research System
**Version**: 2.0
**Date**: 2026-01-12
**Status**: Ready for Implementation
**Methodology**: Parallel Architect Consensus (7 Architects) + User Feedback Integration

---

## Executive Summary

This PRD defines the requirements for an automated newsletter research tool that transforms PreEmploymentDirectory's manual article discovery process into an intelligent DSPy-based agent system. The system will serve 2,100+ background screening firms worldwide by automatically identifying, classifying, and curating relevant industry news.

**Business Impact**: Replace 20+ hours/week of manual research with an agent-based query system, achieving 70%+ recall while maintaining quality signal-to-noise ratio.

**Key Architecture Decisions**:
- **retrieve-dspy + QUIPLER** for state-of-the-art RAG retrieval
- **Weaviate** (local Docker) for vector storage with advanced hybrid search
- **Tiny LM pre-filtering** (gpt-4o-mini) instead of hardcoded keywords
- **Cohere reranking** for precision optimization
- **dspy.ReAct** for multi-step query synthesis

---

## 1. Problem Statement

### 1.1 Current State

PreEmploymentDirectory staff manually:
1. Search LinkedIn and monitor email subscriptions
2. Dump discovered articles into Dropbox
3. Use ChatGPT to summarize content
4. Manually categorize articles by region and topic
5. Curate final selection for newsletters

**Pain Points**:
- 20+ hours/week of repetitive manual research
- Inconsistent coverage across 6 global regions
- High latency between news event and newsletter inclusion
- No searchable historical archive
- Difficult to answer ad-hoc questions about industry trends

### 1.2 Target State

An agent-based system that:
1. **Ingests** articles from 247+ sources (RSS, scraping, email)
2. **Filters** using tiny LM-based semantic analysis (not hardcoded keywords)
3. **Classifies** by 6 regions × 8 topics using DSPy modules
4. **Scores** relevance with transparent reasoning
5. **Stores** in Weaviate vector database for RAG retrieval
6. **Queries** via CLI using QUIPLER-powered natural language agent

---

## 2. Success Criteria

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Recall** | ≥70% | % of manually selected articles also found by system |
| **Daily Candidates** | 20-50 | Articles passing relevance threshold for human review |
| **Query Latency (Simple)** | <3 sec | P95 response time for basic queries |
| **Query Latency (Complex)** | <10 sec | P95 response time for multi-hop queries |
| **Classification Accuracy** | ≥70% | Test set accuracy (region + topics combined) |
| **Monthly API Cost** | <$50 | OpenAI/Cohere API spend |
| **Processing Time** | <2 hrs | Daily batch completion window |

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION LAYER (Daily Batch)                │
├──────────────┬─────────────┬──────────────┬────────────────────┤
│  RSS Feeds   │  Scrapers   │ Email Parser │  (Future: Social)  │
│  (35 feeds)  │ (25 DPAs)   │ (newsletters)│                    │
└──────┬───────┴──────┬──────┴──────┬───────┴────────────────────┘
       └──────────────┴─────────────┘
                      │
              ┌───────▼────────┐
              │ Deduplication  │  (Hash + Fuzzy Matching)
              └───────┬────────┘
                      │
┌─────────────────────▼─────────────────────────────────────────┐
│                 PROCESSING PIPELINE (DSPy)                     │
├────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Stage 1: Tiny LM Relevance Filter (dspy.Predict)      │   │
│  │  • gpt-4o-mini based semantic filtering                 │   │
│  │  • Catches nuanced relevance keywords would miss        │   │
│  │  • 500+ articles/day → 150-200 candidates              │   │
│  └────────────────────┬───────────────────────────────────┘   │
│                       │                                        │
│  ┌────────────────────▼───────────────────────────────────┐   │
│  │  Stage 2: DSPy Classification (dspy.TypedPredictor)    │   │
│  │  • Region: 1 of 6 regions                              │   │
│  │  • Topics: 1-4 from 8 categories (multi-label)         │   │
│  │  • Country: ISO 3166-1 code                            │   │
│  └────────────────────┬───────────────────────────────────┘   │
│                       │                                        │
│  ┌────────────────────▼───────────────────────────────────┐   │
│  │  Stage 3: Relevance Scoring (dspy.ChainOfThought)      │   │
│  │  • Score: 0.0-1.0                                      │   │
│  │  • Reasoning: Why this matters to screening industry   │   │
│  │  • Key signals: Phrases that drove the score           │   │
│  └────────────────────┬───────────────────────────────────┘   │
│                       │                                        │
│  ┌────────────────────▼───────────────────────────────────┐   │
│  │  Stage 4: Summarization (dspy.Predict)                 │   │
│  │  • 2-3 sentence industry-focused summary               │   │
│  │  • Only for candidates (score ≥ threshold)             │   │
│  └────────────────────┬───────────────────────────────────┘   │
└────────────────────────┼──────────────────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────────────────┐
│                    STORAGE LAYER                               │
├────────────────────────────────────────────────────────────────┤
│  Weaviate (Local Docker)                                       │
│  ├─ Vector Index: text2vec-openai (text-embedding-3-small)     │
│  ├─ Hybrid Search: BM25 + vector fusion                        │
│  ├─ Metadata: region, country, topics[], score, date, source   │
│  └─ Full Text: title, summary, reasoning, url                  │
│                                                                │
│  JSON Files (Audit)                                            │
│  └─ processing_log.jsonl (audit trail)                         │
└────────────────────────┬──────────────────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────────────────┐
│              QUERY LAYER (retrieve-dspy + QUIPLER)             │
├────────────────────────────────────────────────────────────────┤
│  CLI Interface (Click + Rich)                                  │
│    │                                                           │
│    ├─> QUIPLER Retrieval (retrieve-dspy)                       │
│    │   • Multi-query expansion (3-5 variations)                │
│    │   • Parallel hybrid search per query                      │
│    │   • Cross-encoder reranking (Cohere rerank-v3.5)          │
│    │   • Reciprocal Rank Fusion (RRF) for result merging       │
│    │                                                           │
│    └─> Answer Synthesis (dspy.ReAct)                           │
│        • Multi-step reasoning with tools                       │
│        • Filter by date/region/topic as tool calls             │
│        • Synthesize answer with source citations               │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow (Daily Batch Processing)

```
06:00 UTC - INGESTION
├─ Fetch 247+ sources (RSS, scrapers, email)
├─ Parse ~500+ articles
└─ Store in raw_queue/

06:15 UTC - DEDUPLICATION
├─ Hash-based exact duplicate removal
├─ Fuzzy matching (>90% similarity threshold)
└─ Remaining: ~400 unique articles

06:20 UTC - TINY LM PRE-FILTER (gpt-4o-mini)
├─ Semantic relevance filtering
├─ Catches nuanced industry relevance
└─ Output: ~150-200 candidates

06:30 UTC - DSPy PROCESSING
├─ Classification: region + topics
├─ Scoring: 0.0-1.0 with reasoning
└─ Summarization: 2-3 sentences (candidates only)

07:30 UTC - STORAGE
├─ Generate embeddings (OpenAI)
├─ Store in Weaviate with metadata
└─ Write processing_log.jsonl
```

### 3.3 Query Agent Flow (Ad-Hoc)

```
USER QUERY: "What regulation changes in APAC affect background screening?"
                │
                ▼
┌───────────────────────────────────────────────────────────────┐
│  QUIPLER (Query Expansion + Parallel Reranking + RRF)         │
├───────────────────────────────────────────────────────────────┤
│  1. EXPAND: Generate 3-5 query variations                      │
│     • "APAC background screening regulations 2026"             │
│     • "Asia Pacific employment screening law changes"          │
│     • "Australia privacy background check guidelines"          │
│                                                                │
│  2. RETRIEVE: Parallel hybrid search (k=50 per query)          │
│     • BM25 + vector similarity in Weaviate                     │
│                                                                │
│  3. RERANK: Cohere cross-encoder (rerank-v3.5)                 │
│     • Parallel reranking per query variation                   │
│                                                                │
│  4. FUSE: Reciprocal Rank Fusion (RRF, k=60)                   │
│     • Merge results from all query variations                  │
│     • Final top-k=20 documents                                 │
└───────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────┐
│  dspy.ReAct Synthesis                                          │
├───────────────────────────────────────────────────────────────┤
│  Tools available:                                              │
│  • filter_by_date(start, end)                                  │
│  • filter_by_region(region)                                    │
│  • filter_by_topic(topic)                                      │
│  • get_article_details(id)                                     │
│                                                                │
│  Reasoning steps:                                              │
│  1. "I need to focus on regulation changes, not general news"  │
│  2. "Let me filter to articles with regulatory topic"          │
│  3. "Found 8 relevant articles, synthesizing answer..."        │
└───────────────────────────────────────────────────────────────┘
                │
                ▼
        STRUCTURED RESPONSE
        ├─ Answer with citations
        ├─ Confidence level
        └─ Source articles list
```

---

## 4. Classification Taxonomy

### 4.1 Regions (6)

| Code | Region | Example Sources |
|------|--------|-----------------|
| `africa_me` | Africa & Middle East | Morgan Lewis ME, SAHRC |
| `asia_pacific` | Asia Pacific | OAIC, Singapore PDPC, China Briefing |
| `europe` | Europe | EDPB, UK ICO, CNIL |
| `north_america` | North America & Caribbean | EEOC, FTC, CFPB, Canadian OPC |
| `south_america` | South America | Brazil ANPD, LGPD updates |
| `worldwide` | Worldwide/Cross-regional | PBSA, multinational treaties |

### 4.2 Topics (8)

| Code | Topic | Example Keywords |
|------|-------|------------------|
| `regulatory` | Regulatory/Legal Changes | FCRA, GDPR, data protection, employment law |
| `criminal_records` | Criminal Background Checks | criminal history, arrest records, ban the box |
| `education` | Education/Credential Verification | credential fraud, degree verification |
| `immigration` | Immigration/Right-to-Work | E-Verify, work authorization, visa |
| `industry_news` | Industry M&A/Company News | acquisition, merger, Sterling, First Advantage |
| `technology` | Technology/Product Announcements | AI, automation, continuous monitoring |
| `events` | Conference/Event News | PBSA conference, webinar |
| `legal_precedents` | Court Cases/Legal Precedents | lawsuit, ruling, settlement |

---

## 5. DSPy Signatures

### 5.1 Tiny LM Relevance Filter Signature

```python
class RelevancePreFilter(dspy.Signature):
    """
    Quick semantic triage using tiny LM.
    Catches nuanced relevance that keyword matching would miss.
    Model: gpt-4o-mini (low cost, fast)
    """
    title: str = dspy.InputField(desc="Article headline")
    content_preview: str = dspy.InputField(desc="First 500 characters")
    source_category: str = dspy.InputField(
        desc="Source type: legal, news, government, industry"
    )

    is_relevant: bool = dspy.OutputField(
        desc="True if potentially relevant to background screening industry"
    )
    confidence: float = dspy.OutputField(desc="0.0-1.0 confidence score")
    reason: str = dspy.OutputField(desc="One-line reason for decision")
```

**Module**: `dspy.Predict` (fast, minimal reasoning overhead)
**Cost**: ~$0.01 per 100 articles

### 5.2 Article Classifier Signature

```python
class ArticleClassifier(dspy.Signature):
    """Classify background screening article by region and topics."""

    title: str = dspy.InputField()
    content: str = dspy.InputField(desc="Article text (truncated to 2000 chars)")
    source: str = dspy.InputField()

    region: Literal["africa_me", "asia_pacific", "europe",
                    "north_america", "south_america", "worldwide"] = dspy.OutputField()
    country: str = dspy.OutputField(desc="ISO 3166-1 alpha-2 code or 'Multiple'")
    topics: list[Literal["regulatory", "criminal_records", "education",
                         "immigration", "industry_news", "technology",
                         "events", "legal_precedents"]] = dspy.OutputField()
    confidence: float = dspy.OutputField()
```

**Module**: `dspy.TypedPredictor` (structured Pydantic output with validation)

### 5.3 Relevance Scorer Signature

```python
class RelevanceScorer(dspy.Signature):
    """Score article relevance to background screening industry (0.0-1.0)."""

    title: str = dspy.InputField()
    content: str = dspy.InputField(desc="First 1500 chars")
    region: str = dspy.InputField()
    topics: list[str] = dspy.InputField()

    relevance_score: float = dspy.OutputField(
        desc="Score 0.0-1.0 calibrated to industry importance"
    )
    key_signals: list[str] = dspy.OutputField(
        desc="2-3 phrases that drove the score"
    )
    reasoning: str = dspy.OutputField(
        desc="Why this score? What would make it higher/lower?"
    )
```

**Module**: `dspy.ChainOfThought` (reasoning for transparency and debugging)

### 5.4 Query Response Signature (ReAct Agent)

```python
class QueryResponseSignature(dspy.Signature):
    """
    Answer questions about background screening news using QUIPLER-retrieved articles.
    Uses ReAct for multi-step reasoning with tool calls.
    """
    question: str = dspy.InputField()
    context: str = dspy.InputField(desc="QUIPLER-retrieved article summaries")
    filters_applied: str = dspy.InputField(desc="Metadata filters used")

    answer: str = dspy.OutputField(desc="Direct answer with inline citations [1], [2]")
    confidence: Literal["low", "medium", "high"] = dspy.OutputField()
    cited_articles: list[dict] = dspy.OutputField(
        desc="List of {id, title, url, date} cited in answer"
    )
    follow_up_suggestions: list[str] = dspy.OutputField(
        desc="2-3 related questions the user might want to explore"
    )
```

**Module**: `dspy.ReAct` (multi-step reasoning with tool use)

---

## 6. RAG Implementation: retrieve-dspy + QUIPLER

### 6.1 Vector Database: Weaviate (Local Docker)

**Docker Compose Configuration**:

```yaml
# docker-compose.yml
version: '3.8'

services:
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.35.2
    ports:
      - "8080:8080"
      - "50051:50051"
    volumes:
      - weaviate_data:/var/lib/weaviate
    environment:
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'text2vec-openai'
      ENABLE_MODULES: 'text2vec-openai,reranker-cohere'
      OPENAI_APIKEY: ${OPENAI_API_KEY}
      COHERE_APIKEY: ${COHERE_API_KEY}
      CLUSTER_HOSTNAME: 'node1'

volumes:
  weaviate_data:
```

**Collection Schema**:

```python
import weaviate
from weaviate.classes.config import Configure, Property, DataType

client = weaviate.connect_to_local()

client.collections.create(
    name="NewsletterArticles",
    vectorizer_config=Configure.Vectorizer.text2vec_openai(
        model="text-embedding-3-small"
    ),
    reranker_config=Configure.Reranker.cohere(
        model="rerank-english-v3.0"
    ),
    properties=[
        Property(name="title", data_type=DataType.TEXT),
        Property(name="content", data_type=DataType.TEXT),
        Property(name="summary", data_type=DataType.TEXT),
        Property(name="url", data_type=DataType.TEXT),
        Property(name="source", data_type=DataType.TEXT),
        Property(name="region", data_type=DataType.TEXT,
                 skip_vectorization=True),
        Property(name="country", data_type=DataType.TEXT,
                 skip_vectorization=True),
        Property(name="topics", data_type=DataType.TEXT_ARRAY,
                 skip_vectorization=True),
        Property(name="relevance_score", data_type=DataType.NUMBER,
                 skip_vectorization=True),
        Property(name="reasoning", data_type=DataType.TEXT),
        Property(name="published_date", data_type=DataType.DATE,
                 skip_vectorization=True),
        Property(name="ingested_at", data_type=DataType.DATE,
                 skip_vectorization=True),
    ]
)
```

### 6.2 QUIPLER Query Agent

```python
import dspy
import cohere
import weaviate
from retrieve_dspy import QUIPLER
from retrieve_dspy.models import RerankerClient

class NewsletterQueryAgent(dspy.Module):
    """
    Production query agent using QUIPLER for state-of-the-art retrieval.
    Combines: Query expansion + Parallel reranking + RRF + ReAct synthesis
    """

    def __init__(self, weaviate_client, cohere_client):
        super().__init__()

        # QUIPLER: State-of-the-art retrieval
        self.retriever = QUIPLER(
            collection_name="NewsletterArticles",
            target_property_name="content",
            weaviate_client=weaviate_client,
            retrieved_k=50,   # Initial retrieval per query
            reranked_k=20,    # Final results after reranking
            rrf_k=60,         # RRF constant
            verbose=False
        )

        # ReAct for synthesis with tool use
        self.synthesize = dspy.ReAct(
            QueryResponseSignature,
            tools=[
                self.filter_by_date,
                self.filter_by_region,
                self.filter_by_topic,
                self.get_article_details
            ]
        )

        self.weaviate_client = weaviate_client
        self.cohere_client = cohere_client

    def forward(self, question: str, filters: dict = None):
        # Step 1: QUIPLER retrieval (expansion + reranking + fusion)
        retrieved = self.retriever.forward(
            question=question,
            weaviate_client=self.weaviate_client,
            reranker_clients=[
                RerankerClient(name="cohere", client=self.cohere_client)
            ]
        )

        # Step 2: Format context from retrieved documents
        context = self._format_context(retrieved.sources)

        # Step 3: Apply any user-specified filters
        if filters:
            context = self._apply_filters(context, filters)

        # Step 4: ReAct synthesis with citations
        result = self.synthesize(
            question=question,
            context=context,
            filters_applied=str(filters) if filters else "none"
        )

        return dspy.Prediction(
            answer=result.answer,
            confidence=result.confidence,
            cited_articles=result.cited_articles,
            follow_up_suggestions=result.follow_up_suggestions,
            queries_generated=retrieved.searches,
            total_documents_searched=len(retrieved.sources)
        )

    # Tool methods for ReAct
    def filter_by_date(self, start_date: str, end_date: str) -> str:
        """Filter articles by date range (YYYY-MM-DD format)"""
        # Implementation using Weaviate where clause
        pass

    def filter_by_region(self, region: str) -> str:
        """Filter articles by region code"""
        pass

    def filter_by_topic(self, topic: str) -> str:
        """Filter articles by topic code"""
        pass

    def get_article_details(self, article_id: str) -> str:
        """Get full details for a specific article"""
        pass

    def _format_context(self, sources) -> str:
        """Format retrieved sources as context string"""
        context_parts = []
        for i, source in enumerate(sources[:15], 1):
            context_parts.append(
                f"[{i}] {source.title}\n"
                f"Date: {source.published_date}\n"
                f"Region: {source.region} | Topics: {', '.join(source.topics)}\n"
                f"Summary: {source.summary}\n"
            )
        return "\n---\n".join(context_parts)
```

### 6.3 CLI Interface

```bash
# Basic query
python cli.py query "recent APAC regulation changes"

# Query with filters
python cli.py query "ban the box" --region north_america --days 30

# Complex multi-hop query
python cli.py query "How has UK GDPR evolved post-Brexit and what does this mean for US screening firms?"

# List recent candidates
python cli.py list --date 2026-01-12 --min-score 0.7

# Show statistics
python cli.py stats --days 7
```

---

## 7. Optimization Strategy

### 7.1 Training Data Requirements

| Signature | Examples Needed | Collection Method |
|-----------|-----------------|-------------------|
| RelevancePreFilter | 50 | Manual labeling from daily feed |
| ArticleClassifier | 100-150 | Export from manual newsletter process |
| RelevanceScorer | 50-100 | Domain expert gold-standard scores |
| QueryResponse | 20-30 | Simulate staff queries with answers |

**Total Effort**: ~200 labeled examples, ~12-15 hours one-time labeling

### 7.2 Optimizer Progression

**Phase 2 (PoC)**: BootstrapFewShot
- Fast compilation (30-60 minutes)
- Works with 50-150 examples
- Expected improvement: +15-25% accuracy over zero-shot

**Phase 3+ (Post-PoC)**: MIPROv2
- Deeper prompt optimization
- Requires 100+ examples
- Expected improvement: +5-10% over BootstrapFewShot

---

## 8. Phased Implementation Plan

### Overview: Parallelizable Phases

```
                    ┌─────────────────────────────────────┐
                    │  PHASE PARALLELIZATION GUIDE        │
                    └─────────────────────────────────────┘

SEQUENTIAL (Must Complete First):
┌──────────────────────────────────────────────────────────────┐
│  PHASE 1A: Infrastructure (Week 1)                           │
│  • Docker compose with Weaviate                              │
│  • Python project structure                                  │
│  • Base DSPy configuration                                   │
└──────────────────────────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌─────────────────────┐     ┌─────────────────────┐
│  PHASE 1B-INGEST    │     │  PHASE 1B-QUERY     │
│  (Weeks 1-2)        │     │  (Weeks 1-2)        │
│  ▸ RSS fetchers     │     │  ▸ QUIPLER setup    │
│  ▸ Deduplication    │     │  ▸ ReAct agent      │
│  ▸ Tiny LM filter   │     │  ▸ CLI interface    │
│  ▸ Classification   │     │  ▸ Test with        │
│  ▸ Scoring          │     │    synthetic data   │
│  ▸ Summarization    │     │                     │
│     CAN RUN IN      │     │     CAN RUN IN      │
│      PARALLEL       │     │      PARALLEL       │
└─────────────────────┘     └─────────────────────┘
         │                               │
         └───────────────┬───────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  PHASE 2: Integration (Week 3)                               │
│  • Connect ingestion pipeline to Weaviate storage            │
│  • Connect query agent to live data                          │
│  • End-to-end testing                                        │
└──────────────────────────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌─────────────────────┐     ┌─────────────────────┐
│  PHASE 3A-DATA      │     │  PHASE 3B-SOURCES   │
│  (Week 4-5)         │     │  (Week 4-5)         │
│  ▸ Collect 150+     │     │  ▸ Add 25 DPA       │
│    training examples│     │    scrapers         │
│  ▸ BootstrapFewShot │     │  ▸ Add email parser │
│    optimization     │     │  ▸ Source expansion │
│     CAN RUN IN      │     │     CAN RUN IN      │
│      PARALLEL       │     │      PARALLEL       │
└─────────────────────┘     └─────────────────────┘
         │                               │
         └───────────────┬───────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  PHASE 4: Production Hardening (Week 6-8)                    │
│  • Monitoring & alerting                                     │
│  • Performance tuning                                        │
│  • Documentation & handoff                                   │
└──────────────────────────────────────────────────────────────┘
```

---

### Phase 1A: Infrastructure Foundation (Week 1) - SEQUENTIAL

**Goal**: Establish core infrastructure that all other work depends on

**Deliverables**:
1. Docker compose with Weaviate (local)
2. Python project structure with uv/poetry
3. DSPy configuration with OpenAI + Cohere
4. Weaviate collection schema created
5. Basic connection tests passing

**Success Criteria**:
- `docker compose up` starts Weaviate
- Python can connect and query Weaviate
- DSPy configured with LM providers
- Collection schema deployed

**Estimated Effort**: 15 hours

---

### Phase 1B-INGEST: Processing Pipeline (Weeks 1-2) - PARALLEL

**Goal**: Build complete article processing pipeline

**Deliverables**:
1. RSS fetcher for 20 Tier 1 feeds
2. Deduplication module (hash + fuzzy)
3. Tiny LM relevance filter (gpt-4o-mini)
4. Article classifier (TypedPredictor)
5. Relevance scorer (ChainOfThought)
6. Summarizer (Predict)
7. Weaviate storage integration

**Success Criteria**:
- Ingest 100+ articles/day from RSS
- Tiny LM filter reduces to 150-200 candidates
- Classification accuracy ≥60% (baseline)
- Articles stored in Weaviate with embeddings

**Estimated Effort**: 50 hours

**Can Run In Parallel With**: Phase 1B-QUERY (use synthetic test data)

---

### Phase 1B-QUERY: Query Agent System (Weeks 1-2) - PARALLEL

**Goal**: Build QUIPLER-powered query agent

**Deliverables**:
1. retrieve-dspy integration
2. QUIPLER configuration with Cohere reranking
3. ReAct synthesis agent
4. CLI interface (Click + Rich)
5. Tool implementations (filter_by_date, etc.)

**Success Criteria**:
- Query agent responds with synthetic data
- QUIPLER generates 3-5 query variations
- Cohere reranking integrated
- CLI interface functional

**Estimated Effort**: 40 hours

**Can Run In Parallel With**: Phase 1B-INGEST (use synthetic test data)

---

### Phase 2: Integration & Testing (Week 3) - SEQUENTIAL

**Goal**: Connect parallel workstreams and validate end-to-end

**Deliverables**:
1. Connect ingestion pipeline to Weaviate
2. Connect query agent to live article data
3. End-to-end test suite
4. Performance benchmarking

**Success Criteria**:
- Full pipeline: RSS → Processing → Storage → Query
- Query latency <5s for simple queries
- API costs <$3/day
- No data corruption or loss

**Estimated Effort**: 25 hours

---

### Phase 3A-DATA: Optimization (Weeks 4-5) - PARALLEL

**Goal**: Achieve 70%+ accuracy through optimization

**Deliverables**:
1. Training data collection (150-200 examples)
2. BootstrapFewShot optimization per signature
3. Evaluation metrics dashboard
4. A/B testing framework

**Success Criteria**:
- Classification accuracy ≥70%
- Recall ≥70% vs manual process
- Optimized prompts saved

**Estimated Effort**: 35 hours

**Can Run In Parallel With**: Phase 3B-SOURCES

---

### Phase 3B-SOURCES: Source Expansion (Weeks 4-5) - PARALLEL

**Goal**: Expand to full 247+ sources

**Deliverables**:
1. Web scrapers for 25 DPAs (Playwright)
2. Email parser integration
3. Source monitoring dashboard
4. Graceful degradation for failures

**Success Criteria**:
- 60+ sources ingesting reliably
- 95% source uptime
- Failures handled gracefully

**Estimated Effort**: 40 hours

**Can Run In Parallel With**: Phase 3A-DATA

---

### Phase 4: Production Hardening (Weeks 6-8) - SEQUENTIAL

**Goal**: Production-ready system with monitoring

**Deliverables**:
1. Monitoring & alerting (API costs, source health)
2. Performance tuning (query latency)
3. Error recovery procedures
4. Documentation & handoff
5. Staff training

**Success Criteria**:
- 95%+ system uptime
- 20-50 candidates/day (stable)
- Staff can operate independently
- Complete documentation

**Estimated Effort**: 45 hours

---

### Timeline Summary

| Week | Phase(s) | Focus | Parallelization |
|------|----------|-------|-----------------|
| 1 | 1A | Infrastructure | Sequential (foundation) |
| 1-2 | 1B-INGEST + 1B-QUERY | Pipeline + Query Agent | **PARALLEL** |
| 3 | 2 | Integration | Sequential (merge) |
| 4-5 | 3A-DATA + 3B-SOURCES | Optimization + Sources | **PARALLEL** |
| 6-8 | 4 | Hardening | Sequential (polish) |

**Total**: 8 weeks, ~250 hours
**Parallelization Savings**: ~2 weeks (if two developers available)

---

## 9. Technology Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **Language** | Python 3.11+ | DSPy ecosystem, ML libraries |
| **LLM Framework** | DSPy 2.5+ | Declarative LM programming |
| **RAG Library** | retrieve-dspy | Pre-built QUIPLER, HyDE, RAGFusion |
| **LLM APIs** | OpenAI (gpt-4o-mini) | Cost-effective, reliable |
| **Embeddings** | text-embedding-3-small | Fast, cheap, multilingual |
| **Vector DB** | Weaviate (local Docker) | Hybrid search, retrieve-dspy support |
| **Reranking** | Cohere (rerank-v3.5) | High quality cross-encoder |
| **RSS Parsing** | feedparser | Industry standard |
| **Web Scraping** | playwright + beautifulsoup | Dynamic + static content |
| **CLI** | Click + Rich | User-friendly interface |
| **Scheduling** | cron | Reliable, OS-native |

---

## 10. Cost Projection

### 10.1 Monthly Operating Costs

| Component | Usage | Unit Cost | Monthly Cost |
|-----------|-------|-----------|--------------|
| Tiny LM Filter (gpt-4o-mini) | 500/day × 1K tokens | $0.15/1M input | $2.25 |
| Classification (gpt-4o-mini) | 200/day × 1.5K tokens | $0.15/1M input | $1.35 |
| Scoring (gpt-4o-mini) | 200/day × 2K tokens | $0.60/1M output | $7.20 |
| Embeddings (OpenAI) | 200/day × 500 tokens | $0.02/1M | $0.60 |
| **Cohere Reranking** | 100 queries/day × 50 docs | $0.002/search | $6.00 |
| Query synthesis | 50/day × 2K tokens | $0.60/1M output | $1.80 |
| **Total API** | | | **~$20/month** |

### 10.2 Infrastructure Costs

| Component | Cost |
|-----------|------|
| Weaviate (local Docker) | $0 |
| Compute (local dev machine) | $0 |
| **Total Infrastructure** | **$0** |

### 10.3 One-Time Optimization Costs

| Phase | Calls | Cost |
|-------|-------|------|
| BootstrapFewShot | 500 | ~$2 |
| MIPROv2 (if needed) | 2,000 | ~$10 |
| **Total** | | **~$12** |

### 10.4 Total Budget

- **Monthly Operations**: ~$20-30/month
- **One-Time Optimization**: ~$15
- **Infrastructure**: $0 (local Docker)
- **Buffer**: $10/month

**Total PoC Budget**: **<$50/month** (well under target)

---

## 11. Risk Assessment

### 11.1 High-Priority Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Low Recall** (<70%) | MEDIUM | HIGH | Tiny LM catches more than keywords; active learning loop |
| **QUIPLER Latency** | LOW | MEDIUM | Tune retrieved_k and reranked_k; cache frequent queries |
| **Weaviate Docker Issues** | LOW | HIGH | Documented compose file; volume persistence |

### 11.2 Medium-Priority Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Cohere API Limits** | LOW | MEDIUM | Fallback to without-reranking mode |
| **Source Breakage** | MEDIUM | LOW | RSS preferred; graceful degradation |
| **Classification Drift** | MEDIUM | MEDIUM | Weekly retraining; confidence monitoring |

### 11.3 Contingency Plans

**If recall <70%**:
1. Expand training data to 200+ examples
2. Run MIPROv2 optimization
3. Lower relevance threshold temporarily

**If query latency exceeds target**:
1. Reduce retrieved_k from 50 to 30
2. Cache frequent query patterns
3. Use HybridSearch for simple queries (skip QUIPLER)

---

## 12. Acceptance Criteria

### 12.1 Phase 1A Complete When:
- [ ] Docker compose starts Weaviate successfully
- [ ] Python connects to Weaviate
- [ ] DSPy configured with OpenAI + Cohere
- [ ] Collection schema deployed

### 12.2 Phase 1B Complete When:
- [ ] 20 RSS feeds ingesting successfully
- [ ] Tiny LM filter operational (gpt-4o-mini)
- [ ] Classification accuracy ≥60% baseline
- [ ] QUIPLER query agent returns results
- [ ] CLI interface functional

### 12.3 Phase 2 Complete When:
- [ ] End-to-end pipeline working
- [ ] Query latency <5 seconds
- [ ] API costs <$3/day

### 12.4 Phase 3 Complete When:
- [ ] 150+ training examples collected
- [ ] BootstrapFewShot optimization complete
- [ ] Classification accuracy ≥70%
- [ ] 60+ sources ingesting

### 12.5 Phase 4 Complete When:
- [ ] 95%+ system uptime over 7 days
- [ ] 20-50 candidates/day (stable)
- [ ] Documentation complete
- [ ] Staff can operate independently

---

## 13. Dependencies

### 13.1 Python Dependencies

```txt
# Core
dspy-ai>=2.5.0
retrieve-dspy>=0.1.0
weaviate-client>=4.0.0

# LLM Providers
openai>=1.0.0
cohere>=5.0.0

# Processing
feedparser>=6.0.0
beautifulsoup4>=4.12.0
playwright>=1.40.0

# CLI
click>=8.1.0
rich>=13.0.0

# Utilities
pydantic>=2.0.0
python-dotenv>=1.0.0
```

### 13.2 Environment Variables

```bash
# .env
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...

# Weaviate (local)
WEAVIATE_URL=http://localhost:8080
```

---

## 14. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-12 | Orchestrator | Initial PRD from 7-architect synthesis |
| 2.0 | 2026-01-12 | Orchestrator | Major revision: QUIPLER, Weaviate, tiny LM filter, parallelization |

---

**END OF PRD**

*This PRD incorporates parallel architect consensus with user feedback. Key changes from v1.0:*
- *ChromaDB → Weaviate (local Docker)*
- *Hardcoded keywords → Tiny LM pre-filter (gpt-4o-mini)*
- *Basic RAG → retrieve-dspy QUIPLER*
- *ChainOfThought queries → dspy.ReAct synthesis*
- *Daily report focus → Agent-based query system*
- *Added phase parallelization guidance*
