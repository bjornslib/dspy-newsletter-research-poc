# Architect 4 Solution Summary - Reverse Engineering Approach

## Core Philosophy: Work Backwards from Perfect Outcome

**Starting Point**: Envisioned the ideal CLI query response
```
User: "recent regulation changes in APAC"
System: Returns 8 highly relevant articles in <5 seconds with star ratings, summaries, and relevance reasoning
```

**Question**: What must exist to produce this output?

---

## Architecture Working Backwards

### Layer 5 (User Interface) → Layer 1 (Ingestion)

```
CLI Query
    ↓ needs
QueryAgent (ReAct) + NewsletterRetriever
    ↓ needs
ChromaDB with embeddings + metadata
    ↓ needs
Classified/Scored/Summarized articles
    ↓ needs
DSPy Processing Pipeline (Region → Topics → Summary)
    ↓ needs
Keyword-filtered candidates
    ↓ needs
Raw articles from sources
    ↓ needs
RSS/Scraping/Email ingestion
```

---

## Key Innovations

### 1. Two-Stage Classification
- **Region first** (TypedPredictor): Fast, keyword-driven
- **Topics second** (ChainOfThought): Uses region context for better accuracy
- **Why**: "background check" in Europe = GDPR context; in US = FCRA context

### 2. Hybrid Cost Optimization
- **Keyword pre-filter**: 500 articles → 150-200 LLM candidates
- **Cost savings**: 60% reduction in API calls
- **Annual projection**: ~$42/year vs $1,800 without filtering

### 3. Transparent Reasoning
- All ChainOfThought modules include `reasoning` field
- Humans see WHY articles were classified/selected
- Improves trust and provides training signal for optimization

### 4. DSPy Module Selection
| Module | Use Case | Rationale |
|--------|----------|-----------|
| TypedPredictor | Region classification | Fast structured output, no reasoning needed |
| ChainOfThought | Topics + Summary | Reasoning improves quality and transparency |
| ReAct | Complex queries | Multi-step reasoning for temporal/comparative questions |

---

## DSPy Signatures (Complete Set)

```python
# 1. Fast region classification
class RegionClassifier(dspy.Signature):
    title: str
    content: str
    region: Literal[6_options]
    country: str
    confidence: float

# 2. Multi-label topic classification with reasoning
class TopicClassifier(dspy.Signature):
    title: str
    content: str
    region: str  # Context from previous stage
    topics: List[str]
    primary_topic: str
    reasoning: str

# 3. Relevance-focused summarization
class ArticleSummarizer(dspy.Signature):
    title: str
    content: str
    topics: List[str]
    region: str
    summary: str
    relevance_reasoning: str

# 4. Natural language query parsing
class QueryParser(dspy.Signature):
    user_query: str
    search_terms: str
    region_filter: str | None
    topic_filter: List[str] | None
    date_range: dict
    reasoning: str
```

---

## Optimization Strategy

### Phase 1: BootstrapFewShot
- **Training data**: 150-200 manually labeled articles (from current process)
- **Metric**: 40% region accuracy + 60% topic Jaccard similarity
- **Expected improvement**: 55-65% baseline → 70-80% optimized
- **Time**: 30-60 minutes to run

### Phase 2: MIPROv2 (if needed)
- **When**: If Phase 1 accuracy <75% or edge cases emerge
- **Improvement**: Additional 5-10% accuracy gain
- **Cost**: 2-4 hours optimization time

### Continuous Learning
- **Weekly review**: 20 articles sampled by confidence score
- **Retraining trigger**: If accuracy drops below 70%
- **Prevents drift**: As topics/sources evolve

---

## RAG Implementation

### ChromaDB Configuration
```python
embedding_function = OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small"  # $0.02/1M tokens
)

collection = client.create_collection(
    name="articles",
    metadata={"hnsw:space": "cosine"},
    embedding_function=embedding_function
)
```

### Hybrid Search
1. **Semantic**: Vector search on embedded summaries
2. **Metadata**: Exact filtering (region, topics, date)
3. **Performance**: <5s for 90% of queries (even with 100K+ articles)

---

## Phased Implementation (8 Weeks)

### Phase 1 (Weeks 1-2): PoC
- 3 RSS feeds
- Keyword-only filtering
- JSON storage
- Basic CLI
- **Goal**: Prove ingestion → retrieval works

### Phase 2 (Weeks 3-4): DSPy Classification
- All 3 DSPy signatures implemented
- ChromaDB migration
- 10 RSS feeds
- **Goal**: 60%+ accuracy baseline

### Phase 3 (Weeks 5-6): Optimization + Query Agent
- BootstrapFewShot optimization
- QueryAgent with ReAct
- 35 RSS feeds (all Tier 1+2)
- **Goal**: 70%+ recall, <5s queries

### Phase 4 (Weeks 7-8): Production
- Web scrapers (25 sources)
- Deduplication
- MIPROv2 (if needed)
- Monitoring dashboard
- **Goal**: 20-50 articles/day shortlist

---

## Risk Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Cost explosion | High API bills | Keyword pre-filter (60% reduction) |
| Classification drift | Accuracy degrades | Weekly review + monthly retraining |
| Scraper breakage | Missed articles | RSS preferred; alerts for failures |
| Query slowness | Poor UX | Date partitioning; HNSW indexing |
| Poor deduplication | Cluttered shortlist | Hash + fuzzy matching (90% threshold) |

---

## Success Metrics

| Metric | Target | Current (Manual) |
|--------|--------|------------------|
| Recall | 70%+ | 100% (baseline) |
| Precision | 30%+ | ~40% (estimated) |
| Time Investment | <2 hours/week | 20+ hours/week |
| Shortlist Size | 20-50/day | Varies widely |
| Query Speed | <5s | N/A (manual search) |
| Cost | <$50/month | ~$1,000/month (labor) |

---

## Technology Stack

- **Language**: Python 3.11+
- **LLM Framework**: DSPy 2.5+
- **LLM APIs**: OpenAI (gpt-4o-mini for cost)
- **Embeddings**: text-embedding-3-small
- **Vector DB**: ChromaDB 0.4+
- **Ingestion**: feedparser, playwright, imaplib
- **CLI**: Click + Rich
- **Monitoring**: Streamlit

---

## Why This Approach Works

### 1. Reverse Engineering Ensures Focus
- Every component justified by end-user need
- No "nice to have" features that don't serve queries
- Clear path from outcome → requirements

### 2. Cost-Conscious Design
- Keyword pre-filter saves 60% API costs
- gpt-4o-mini instead of gpt-4o (10x cheaper)
- Small embeddings model (sufficient quality, lower cost)
- **Result**: $42/year vs $1,800/year naive approach

### 3. Incremental Value Delivery
- Phase 1: Working prototype in 2 weeks
- Phase 2: Intelligent classification in 4 weeks
- Phase 3: Production-quality in 6 weeks
- Phase 4: Full feature set in 8 weeks

### 4. Human-in-the-Loop
- Reasoning traces make decisions transparent
- Weekly review prevents drift
- Shortlist (not full automation) keeps humans in control
- **Result**: Trust + adaptability

### 5. DSPy Optimization Leverage
- BootstrapFewShot uses existing manual labels
- 15-30% accuracy improvement typical
- Continuous learning as data grows
- **Result**: System improves over time automatically

---

## Next Steps

1. **Review this design** with development team
2. **Set up environment** (Python, DSPy, ChromaDB, OpenAI keys)
3. **Export manual labels** (last 30 days of curated articles)
4. **Start Phase 1** (RSS ingestion + keyword filtering)
5. **Weekly check-ins** with PM to validate progress

---

## Document Location

Full design document:
`/Users/theb/Documents/Windsurf/DSPY_PreEmploymentDirectory_PoC/documentation/solution_designs/dspy_newsletter_tool_architecture_reverse_engineered.md`

---

**Architect 4 (Reverse Engineering)**
Date: 2026-01-12
Status: Ready for Development Handoff
