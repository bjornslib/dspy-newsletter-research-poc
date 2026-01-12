# RAG Architecture Recommendations for Newsletter Research Tool

**Date**: 2026-01-12
**Author**: Orchestrator (Post-Research Synthesis)
**Status**: Recommendations for User Review

---

## Executive Summary

After researching DSPy RAG patterns using context7 and Perplexity, I recommend adopting the **retrieve-dspy** library from Weaviate, which provides pre-built Compound AI Systems optimized for retrieval-augmented generation.

### Key Architectural Changes from Original PRD

| Original PRD | Recommended Change | Rationale |
|--------------|-------------------|-----------|
| Hardcoded keyword pre-filter | **Tiny LM-based filter** | User feedback: Keywords miss important articles |
| ChromaDB storage | **Weaviate** | retrieve-dspy only supports Weaviate; provides superior retrieval patterns |
| dspy.ChainOfThought for queries | **retrieve-dspy + dspy.ReAct** | Pre-built multi-hop, reranking, query expansion |
| Daily batch report focus | **Agent-based query system** | User feedback: Focus on ad-hoc queries in Phase 1 |

---

## Research Findings

### retrieve-dspy Library Overview

The `retrieve-dspy` library (from Weaviate) provides production-ready RAG patterns that would take months to build from scratch:

| Module | Purpose | Use Case |
|--------|---------|----------|
| `HybridSearch` | Baseline hybrid search | Simple queries |
| `MultiQueryWriter` | Generates multiple search queries | Improves recall through query diversity |
| `HyDE_QueryExpander` | Hypothetical Document Embeddings | Better semantic matching |
| `RAGFusion` | Query expansion + RRF fusion | Multi-faceted queries |
| `CrossEncoderReranker` | Neural reranking (Cohere/Voyage) | Precision-focused retrieval |
| `SimplifiedBaleenWithCrossEncoder` | Multi-hop iterative retrieval | Complex queries requiring reasoning across documents |
| `QUIPLER` | State-of-the-art combined approach | Best quality, higher latency |

### Recommended RAG Pattern for Newsletter Research

For queries like *"What regulation changes in APAC affect background screening?"*, I recommend:

**Primary Pattern: MultiQueryWriter + CrossEncoderReranker**

```python
import dspy
from retrieve_dspy import MultiQueryWriter, CrossEncoderReranker
from retrieve_dspy.models import RerankerClient

class NewsletterRAGAgent(dspy.Module):
    """
    Two-stage RAG: Multi-query expansion for recall + Cross-encoder for precision
    """
    def __init__(self, weaviate_client, cohere_client):
        super().__init__()

        # Stage 1: Multi-query expansion (improves recall)
        self.query_expander = MultiQueryWriter(
            collection_name="NewsletterArticles",
            target_property_name="content",
            retrieved_k=50,  # Cast wide net
            search_with_queries_concatenated=False,
            verbose=True
        )

        # Stage 2: Cross-encoder reranking (improves precision)
        self.reranker = CrossEncoderReranker(
            collection_name="NewsletterArticles",
            target_property_name="content",
            weaviate_client=weaviate_client,
            retrieved_k=50,
            reranked_k=10,  # Top 10 final results
            verbose=True
        )

        # Stage 3: ReAct agent for synthesis
        self.synthesize = dspy.ReAct(
            "context, question, filters -> answer, sources",
            tools=[self.search_metadata, self.filter_by_date]
        )

        self.clients = {
            'weaviate': weaviate_client,
            'cohere': cohere_client
        }

    def forward(self, question: str, filters: dict = None):
        # Step 1: Expand query and retrieve
        expansion_result = self.query_expander.forward(question=question)

        # Step 2: Rerank results
        reranked = self.reranker.forward(
            question=question,
            weaviate_client=self.clients['weaviate'],
            reranker_clients=[RerankerClient(name="cohere", client=self.clients['cohere'])]
        )

        # Step 3: Apply metadata filters if provided
        context = self._apply_filters(reranked.sources, filters)

        # Step 4: Synthesize answer with ReAct reasoning
        result = self.synthesize(
            context=context,
            question=question,
            filters=str(filters) if filters else "none"
        )

        return dspy.Prediction(
            answer=result.answer,
            sources=[s.object_id for s in reranked.sources[:5]],
            queries_generated=expansion_result.searches
        )
```

**Alternative for Complex Multi-hop Queries: QUIPLER**

For complex queries like *"How has GDPR enforcement evolved in the UK post-Brexit and what does this mean for US screening firms?"*:

```python
from retrieve_dspy import QUIPLER

class ComplexQueryRAGAgent(dspy.Module):
    """
    QUIPLER: State-of-the-art retrieval combining all techniques
    """
    def __init__(self, weaviate_client, cohere_client):
        super().__init__()

        self.retriever = QUIPLER(
            collection_name="NewsletterArticles",
            target_property_name="content",
            weaviate_client=weaviate_client,
            retrieved_k=100,  # Large initial pool
            reranked_k=20,    # Aggressive reranking
            rrf_k=60,         # RRF constant
            verbose=True
        )

        self.extract_facts = dspy.ChainOfThought("context, question -> key_facts: list[str]")
        self.synthesize = dspy.ReAct("facts, question -> comprehensive_answer, recommendations")

        self.clients = {
            'weaviate': weaviate_client,
            'cohere': cohere_client
        }

    def forward(self, question: str):
        # QUIPLER: Query expansion + Parallel reranking + RRF
        retrieved = self.retriever.forward(
            question=question,
            weaviate_client=self.clients['weaviate'],
            reranker_clients=[RerankerClient(name="cohere", client=self.clients['cohere'])]
        )

        # Extract key facts from diverse sources
        facts = self.extract_facts(
            context="\n\n".join([s.content for s in retrieved.sources[:10]]),
            question=question
        )

        # ReAct synthesis for complex reasoning
        result = self.synthesize(
            facts=facts.key_facts,
            question=question
        )

        return dspy.Prediction(
            answer=result.comprehensive_answer,
            recommendations=result.recommendations,
            source_count=len(retrieved.sources),
            queries_used=retrieved.searches
        )
```

---

## Architecture Recommendation: Weaviate over ChromaDB

### Why Switch to Weaviate

| Feature | ChromaDB | Weaviate |
|---------|----------|----------|
| **retrieve-dspy support** | No | Full native support |
| **Hybrid search** | Basic | Advanced (BM25 + vector fusion) |
| **Multi-tenancy** | Limited | Built-in |
| **Reranking** | Manual | Integrated via retrieve-dspy |
| **Query expansion** | Manual | MultiQueryWriter, HyDE, RAGFusion |
| **Multi-hop retrieval** | Manual | SimplifiedBaleenWithCrossEncoder |
| **Cost** | Free | Free (self-hosted) or Cloud |
| **Production readiness** | PoC-suitable | Enterprise-ready |

**Recommendation**: Switch to **Weaviate** to leverage retrieve-dspy's pre-built Compound AI Systems. This provides months of engineering effort out-of-the-box.

---

## Updated Processing Pipeline

### Original Pipeline (from PRD)
```
Ingestion → Deduplication → Keyword Pre-Filter → DSPy Processing → Storage → Query
```

### Recommended Pipeline
```
Ingestion → Deduplication → Tiny LM Filter → DSPy Processing → Weaviate Storage → retrieve-dspy Query Agent
```

### Stage-by-Stage Changes

#### Stage 1: Relevance Pre-Filter (CHANGED)

**Original**: Hardcoded keyword matching
**New**: Tiny LM-based filtering using `dspy.Predict` with small model

```python
class RelevancePreFilter(dspy.Signature):
    """
    Quick triage using tiny LM - catches nuanced relevance keywords miss.
    Model: gpt-4o-mini or claude-3-haiku (low cost, fast)
    """
    title: str = dspy.InputField(desc="Article headline")
    content_preview: str = dspy.InputField(desc="First 500 characters")
    source_category: str = dspy.InputField(desc="Source type: legal, news, government, industry")

    is_relevant: bool = dspy.OutputField(desc="True if potentially relevant to background screening industry")
    confidence: float = dspy.OutputField(desc="0.0-1.0 confidence score")
    reason: str = dspy.OutputField(desc="One-line reason for decision")

# Implementation
class TinyLMRelevanceFilter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.filter = dspy.Predict(RelevancePreFilter)

    def forward(self, title: str, content_preview: str, source_category: str):
        # Use tiny LM (configured at dspy.configure level)
        result = self.filter(
            title=title,
            content_preview=content_preview,
            source_category=source_category
        )
        return result
```

**Cost Comparison**:
- Keyword filter: $0 (but misses nuanced articles)
- Tiny LM filter: ~$0.01 per 100 articles with gpt-4o-mini
- **Break-even**: If tiny LM catches 10 valuable articles/month that keywords miss, it's worth it

#### Stage 5: Query Response (CHANGED)

**Original**: `dspy.ChainOfThought` or `dspy.ReAct`
**New**: `retrieve-dspy` patterns + `dspy.ReAct` for synthesis

See code examples above for full implementation.

---

## Tiered Query Strategy

Based on query complexity, route to different retrieve-dspy patterns:

| Query Type | Example | retrieve-dspy Pattern | Latency | Cost |
|------------|---------|----------------------|---------|------|
| **Simple** | "Latest FCRA updates" | `HybridSearch` | <2s | Low |
| **Standard** | "Regulation changes in APAC" | `MultiQueryWriter` + `CrossEncoderReranker` | 3-5s | Medium |
| **Complex** | "How has UK GDPR evolved post-Brexit..." | `QUIPLER` or `SimplifiedBaleenWithCrossEncoder` | 5-10s | Higher |

**Query Router Implementation**:

```python
class QueryRouter(dspy.Module):
    """Routes queries to appropriate retrieve-dspy pattern based on complexity"""

    def __init__(self, weaviate_client, cohere_client):
        super().__init__()
        self.classify = dspy.Predict("question -> complexity: Literal['simple', 'standard', 'complex']")

        self.simple_retriever = HybridSearch(...)
        self.standard_retriever = NewsletterRAGAgent(weaviate_client, cohere_client)
        self.complex_retriever = ComplexQueryRAGAgent(weaviate_client, cohere_client)

    def forward(self, question: str):
        complexity = self.classify(question=question).complexity

        if complexity == "simple":
            return self.simple_retriever.forward(question)
        elif complexity == "standard":
            return self.standard_retriever.forward(question)
        else:
            return self.complex_retriever.forward(question)
```

---

## Phase 1 Reframe: Agent-Based Query System

Per user feedback, Phase 1 should focus on an **agent system for ad-hoc queries**, not daily reports.

### Phase 1 Deliverables (Revised)

1. **Core Processing Pipeline**
   - Ingestion from RSS feeds
   - Tiny LM relevance filtering
   - Classification + scoring
   - Weaviate storage with embeddings

2. **Query Agent System**
   - CLI interface for ad-hoc queries
   - Query routing (simple → standard → complex)
   - retrieve-dspy patterns for RAG
   - dspy.ReAct for synthesis with source citations

3. **No Daily Report** (deferred)
   - Daily digest moved to Phase 2
   - Focus on reliable query response first

### Success Criteria (Phase 1)

| Metric | Target |
|--------|--------|
| Query response accuracy | ≥80% user satisfaction |
| Query latency (simple) | <3 seconds |
| Query latency (complex) | <10 seconds |
| Articles processed/day | 300-500 |
| Monthly cost | <$30 |

---

## Additional Dependencies

To use retrieve-dspy, add these to requirements:

```txt
retrieve-dspy>=0.1.0
weaviate-client>=4.0.0
cohere>=5.0.0  # For reranking
voyageai>=0.3.0  # Alternative reranker
```

---

## My Recommendations Summary

1. **Adopt retrieve-dspy** - Pre-built Compound AI Systems save months of engineering
2. **Switch ChromaDB → Weaviate** - Required for retrieve-dspy, better retrieval quality
3. **Use tiny LMs for pre-filtering** - gpt-4o-mini catches what keywords miss
4. **Implement tiered query routing** - Match query complexity to retrieval pattern
5. **Use dspy.ReAct for synthesis** - Multi-step reasoning with tool use
6. **Reframe Phase 1** - Agent-based query system, not daily reports

---

## Open Questions for User

1. **Weaviate hosting**: Self-hosted or Weaviate Cloud? (Cloud is easier, self-hosted is free)
2. **Reranker choice**: Cohere or Voyage AI? (Cohere has better free tier)
3. **Query complexity threshold**: Should we start with just the "standard" pattern?

---

*This document synthesizes research from context7 (retrieve-dspy documentation) and Perplexity (DSPy RAG patterns) with user feedback on architecture direction.*
