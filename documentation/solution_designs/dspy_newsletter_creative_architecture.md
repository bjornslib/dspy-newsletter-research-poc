# DSPy Newsletter Research Tool - Creative Exploration Architecture
**Architect**: Architect 7 (Creative Exploration Strategy)
**Date**: 2026-01-12
**Status**: Design Exploration
**Approach**: Recommendation System + Knowledge Graph + Active Learning Hybrid

---

## Executive Summary

**What if newsletter curation were a recommendation problem, not a classification problem?**

This architecture reimagines PreEmploymentDirectory's newsletter research as a **self-improving recommendation engine** where articles compete for attention, sources build reputation, and the system learns from editorial taste over time.

Instead of asking "What category does this belong to?", we ask:
- "How likely would our editors select this article?" (Recommendation score)
- "Which sources consistently surface valuable content?" (Authority graph)
- "What patterns distinguish must-read from skip-worthy?" (Active learning)

**Core Innovation**: A three-layer architecture where:
1. **Content Layer** (Traditional): Classification, relevance scoring, deduplication
2. **Reputation Layer** (Graph-based): Sources, entities, and relationships build authority scores via PageRank
3. **Learning Layer** (Active): System identifies its own blind spots and prioritizes human feedback where it matters most

**Key Outcomes**:
- 70%+ recall with self-improving precision through feedback loops
- Source reputation tracking reveals hidden gems and filters noise
- Active learning minimizes labeling burden (200 examples vs 2000+)
- Knowledge graph enables "who cites whom?" and temporal relationship queries
- <$50/month operational cost through clever caching and tiered processing

---

## 1. Architectural Analogies - Thinking Outside the Pipeline

### 1.1 Analogy: Netflix Recommendation System

**Traditional Approach**: Classify → Filter → Output
**Recommendation Approach**: Predict editorial appeal → Rank by certainty → Learn from selections

| Netflix Pattern | Our Application |
|-----------------|-----------------|
| **User ratings** | Editorial approve/reject decisions |
| **Content features** | Region, topics, source reputation |
| **Collaborative filtering** | "Articles similar to past selections" |
| **Cold start problem** | New sources need bootstrapping |
| **Exploration vs exploitation** | Balance known-good sources vs discovering new ones |

**Implementation**:
```python
class EditorialRecommender(dspy.Module):
    """
    Predict probability an editor would select this article.
    Learn from historical accept/reject patterns.
    """
    def __init__(self):
        super().__init__()
        self.feature_extractor = dspy.ChainOfThought(ArticleFeatures)
        self.appeal_scorer = dspy.ChainOfThought(EditorialAppeal)

    def forward(self, article, editor_history=None):
        # Extract features (topics, novelty, source reputation)
        features = self.feature_extractor(
            title=article['title'],
            content=article['content'],
            source_reputation=get_source_score(article['source'])
        )

        # Score appeal based on what editors historically value
        appeal = self.appeal_scorer(
            features=features,
            similar_articles=find_similar_accepted(article),
            recency_bonus=compute_freshness(article['date'])
        )

        return appeal.score, appeal.confidence, appeal.reasoning
```

**Signature for Editorial Appeal**:
```python
class EditorialAppeal(dspy.Signature):
    """
    Predict whether this article matches editorial selection criteria.
    Consider: regulatory impact, audience relevance, actionability.
    """
    features: str = dspy.InputField(desc="Extracted article features")
    similar_articles: str = dspy.InputField(desc="Past accepted articles with similar characteristics")
    recency_bonus: float = dspy.InputField(desc="Freshness multiplier 0.5-1.5")

    appeal_score: float = dspy.OutputField(
        desc="Probability editor would select this (0.0-1.0)"
    )
    confidence: float = dspy.OutputField(
        desc="Model confidence in prediction (0.0-1.0)"
    )
    reasoning: str = dspy.OutputField(
        desc="Explain why this article would/wouldn't appeal to editors"
    )
    novelty_factor: float = dspy.OutputField(
        desc="How different is this from recent selections? (0.0=duplicate, 1.0=novel)"
    )
```

### 1.2 Analogy: PageRank for Source Authority

**Traditional Approach**: All sources treated equally
**PageRank Approach**: Sources gain authority through citation patterns and editorial acceptance

**Implementation Concept**:
```
Source Authority = 0.3 * CitationScore + 0.4 * EditorialAcceptanceRate + 0.2 * FirstToReport + 0.1 * Diversity
```

**Knowledge Graph Structure**:
```
[Article] --cites--> [Article]
[Article] --mentions--> [Entity: FCRA]
[Article] --published_by--> [Source: EEOC]
[Source] --first_reported--> [Story: Ban the Box CA 2026]
[Entity] --regulated_by--> [Entity: DOL]
```

**DSPy Module for Authority Propagation**:
```python
import networkx as nx

class SourceReputationEngine:
    """
    Track source authority using graph-based scoring.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.authority_scores = {}

    def add_article_with_citations(self, article_id, source, cited_sources):
        """Build citation graph"""
        self.graph.add_node(source, type='source')
        for cited in cited_sources:
            self.graph.add_edge(source, cited, relation='cites')

    def compute_authority(self):
        """Run PageRank on source subgraph"""
        source_nodes = [n for n,d in self.graph.nodes(data=True) if d.get('type')=='source']
        subgraph = self.graph.subgraph(source_nodes)
        self.authority_scores = nx.pagerank(subgraph, alpha=0.85)

    def adjust_by_editorial_feedback(self, source, accept_rate):
        """Boost/penalize based on editorial decisions"""
        base_score = self.authority_scores.get(source, 0.5)
        adjusted = 0.6 * base_score + 0.4 * accept_rate
        self.authority_scores[source] = adjusted
```

### 1.3 Analogy: Active Learning for Labeling Efficiency

**Traditional Approach**: Randomly sample 1000 articles for labeling
**Active Learning Approach**: System identifies 200 most informative examples

**Uncertainty Sampling Strategy**:
```python
class ActiveLearner:
    """
    Identify articles where model is most uncertain.
    Prioritize labeling these for maximum learning gain.
    """
    def select_for_labeling(self, unlabeled_articles, budget=50):
        """
        Sampling strategies:
        1. Low confidence classifications
        2. Borderline relevance scores (near decision threshold)
        3. Novel entity combinations (not seen in training)
        4. High-volume sources with changing editorial patterns
        """
        candidates = []

        for article in unlabeled_articles:
            # Score information gain potential
            uncertainty = self.compute_uncertainty(article)
            novelty = self.compute_novelty(article)
            source_drift = self.detect_source_pattern_drift(article['source'])

            priority = 0.5 * uncertainty + 0.3 * novelty + 0.2 * source_drift
            candidates.append((article, priority))

        # Return top-N by priority
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [a for a,_ in candidates[:budget]]

    def compute_uncertainty(self, article):
        """Entropy of classification probability distribution"""
        # If model predicts: 40% regulatory, 35% criminal, 25% other
        # High entropy = uncertain
        import numpy as np
        probs = article['classification_probabilities']
        return -np.sum(probs * np.log(probs + 1e-10))

    def compute_novelty(self, article):
        """Distance to nearest neighbor in training set"""
        # Embed article, find closest training example
        # Large distance = novel
        embedding = self.embed(article['content'])
        distance = self.nearest_neighbor_distance(embedding)
        return min(distance / 0.5, 1.0)  # Normalize

    def detect_source_pattern_drift(self, source):
        """Has source's editorial acceptance rate changed?"""
        recent_rate = get_recent_acceptance_rate(source, days=30)
        historical_rate = get_historical_acceptance_rate(source)
        return abs(recent_rate - historical_rate)
```

---

## 2. System Architecture - Three-Layer Design

### 2.1 ASCII Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LAYER 3: LEARNING & ADAPTATION                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Active Learning Controller                                       │  │
│  │  • Uncertainty sampling (low confidence → review)                 │  │
│  │  • Novelty detection (new entity patterns)                        │  │
│  │  • Source drift monitoring (changing editorial rates)             │  │
│  │  • Feedback loop orchestration                                    │  │
│  │                                                                    │  │
│  │  Optimization Scheduler:                                          │  │
│  │  Weekly: BootstrapFewShot (new examples)                         │  │
│  │  Monthly: MIPROv2 (if F1 <0.75)                                  │  │
│  │  Continuous: Source reputation updates                            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                ↕ Feedback
┌─────────────────────────────────────────────────────────────────────────┐
│                  LAYER 2: REPUTATION & AUTHORITY                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Knowledge Graph Engine (NetworkX)                                │  │
│  │                                                                    │  │
│  │  Nodes:                          Edges:                           │  │
│  │  • Articles                      • mentions(entity)               │  │
│  │  • Sources                       • cites(article)                 │  │
│  │  • Entities (FCRA, EEOC, etc)   • published_by(source)           │  │
│  │  • Topics/Regulations            • supersedes(law)                │  │
│  │                                  • regulated_by(agency)            │  │
│  │  Scores:                                                          │  │
│  │  • PageRank authority (sources)                                   │  │
│  │  • Entity centrality (importance)                                 │  │
│  │  • First-to-report tracking                                       │  │
│  │  • Editorial acceptance rates (per source)                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Recommendation Engine                                            │  │
│  │  • Predict editorial appeal (collaborative filtering)             │  │
│  │  • Rank by confidence-adjusted score                              │  │
│  │  • Diversity reranking (avoid redundancy)                         │  │
│  │  • Temporal decay (recency bonus)                                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                ↕ Context
┌─────────────────────────────────────────────────────────────────────────┐
│                   LAYER 1: CONTENT PROCESSING                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Ingestion Pipeline (15-min batches)                             │  │
│  │  RSS Fetcher → Deduplication → Keyword Filter (40% reduction)    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                ↓                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  DSPy Classification Pipeline (Two-Stage)                         │  │
│  │                                                                    │  │
│  │  Stage 1: Quick Relevance Filter (Predict module)                │  │
│  │    → 60-70% dropped, saves $20-30/month                          │  │
│  │                                                                    │  │
│  │  Stage 2: Full Classification (ChainOfThought)                   │  │
│  │    → Region + Topics + Reasoning                                  │  │
│  │                                                                    │  │
│  │  Stage 3: Entity Extraction (TypedPredictor)                     │  │
│  │    → Organizations, Laws, Jurisdictions                           │  │
│  │                                                                    │  │
│  │  Stage 4: Relevance Scoring (ChainOfThought)                     │  │
│  │    → Editorial appeal prediction                                  │  │
│  │    → Incorporates source reputation from Layer 2                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                ↓                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Storage Layer                                                    │  │
│  │  • SQLite (metadata, feedback, graph edges)                      │  │
│  │  • ChromaDB (vector embeddings, semantic search)                 │  │
│  │  • NetworkX (in-memory graph, periodic snapshots)                │  │
│  │  • Redis (optional: query cache, session state)                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         QUERY INTERFACE (RAG)                            │
│                                                                          │
│  Natural Language Query: "Recent Ban the Box developments in CA"        │
│       ↓                                                                  │
│  1. Intent Analysis (region=north_america, topic=regulatory)            │
│  2. Hybrid Retrieval:                                                   │
│     • Vector search (ChromaDB): semantic similarity                     │
│     • Graph traversal: related entities (Ban the Box → CA → articles)   │
│     • Temporal filter: last 90 days                                     │
│  3. Reranking by:                                                       │
│     • Source authority scores                                           │
│     • Entity centrality                                                  │
│     • Diversity (dedup similar stories)                                 │
│  4. Answer Generation (ChainOfThought)                                  │
│     • Multi-document synthesis                                          │
│     • Source citations                                                   │
│     • Confidence scoring                                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Information Flow

**Daily Batch Processing Flow**:
```
1. RSS Fetch (500 articles)
   ↓
2. Hash-based Dedup (→ 400 unique)
   ↓
3. Keyword Pre-filter (→ 160 relevant)
   ↓
4. Stage 1: Quick Relevance Check (→ 120 pass filter)
   ↓
5. Stage 2: Classification + Entity Extraction (→ 120 classified)
   ↓
6. Stage 3: Editorial Appeal Scoring (→ ranked by score)
   ↓
7. Source Reputation Adjustment (boost/penalty)
   ↓
8. Knowledge Graph Update (add nodes/edges)
   ↓
9. ChromaDB Embedding (store vectors)
   ↓
10. Top 40-50 → Review Queue for editors
    ↓
11. Editorial Feedback → Active Learning Queue
    ↓
12. Weekly: Retrain on new labels
```

---

## 3. DSPy Signatures - The Creative Twist

### 3.1 Editorial Appeal Prediction (Recommendation-Style)

```python
class EditorialAppealSignature(dspy.Signature):
    """
    Predict likelihood this article matches newsletter editorial standards.
    Think like a recommender system: what signals predict selection?
    """
    # Input fields
    title: str = dspy.InputField()
    content_snippet: str = dspy.InputField(desc="First 1000 chars")
    source_name: str = dspy.InputField()
    source_authority: float = dspy.InputField(desc="PageRank score 0.0-1.0")
    historical_context: str = dspy.InputField(
        desc="Recent similar articles and editorial decisions on them"
    )
    recency_days: int = dspy.InputField(desc="Days since publication")

    # Multi-dimensional output
    appeal_score: float = dspy.OutputField(
        desc="Probability editor would select (0.0-1.0). "
             "Based on: relevance to audience, regulatory significance, "
             "novelty vs recent coverage, source credibility."
    )

    novelty_score: float = dspy.OutputField(
        desc="How different is this from recent selections? "
             "1.0 = completely new angle, 0.0 = redundant"
    )

    urgency_tier: Literal["breaking", "important", "routine", "fyi"] = dspy.OutputField(
        desc="Editorial urgency. Breaking = must-include, FYI = nice-to-have"
    )

    reasoning: str = dspy.OutputField(
        desc="3-4 sentences explaining appeal score, citing specific factors"
    )

    confidence: float = dspy.OutputField(
        desc="Model confidence in this prediction (0.0-1.0)"
    )

    risk_flags: List[str] = dspy.OutputField(
        desc="Potential concerns: ['unverified_claim', 'paywall', 'partisan_source', etc.]"
    )
```

### 3.2 Entity Extraction for Knowledge Graph

```python
class EntityRelationshipExtractor(dspy.Signature):
    """
    Extract structured entities and relationships for knowledge graph.
    Focus on background screening domain: agencies, laws, companies, events.
    """
    text: str = dspy.InputField()
    region: str = dspy.InputField(desc="Geographic context for entity disambiguation")

    # Structured entity extraction
    organizations: List[str] = dspy.OutputField(
        desc="Companies, agencies, regulators (e.g., 'EEOC', 'Sterling', 'UK ICO')"
    )

    regulations: List[str] = dspy.OutputField(
        desc="Laws, policies, standards (e.g., 'FCRA', 'GDPR', 'Ban the Box CA SB 731')"
    )

    jurisdictions: List[str] = dspy.OutputField(
        desc="Countries, states, cities with legal authority"
    )

    events: List[str] = dspy.OutputField(
        desc="Court cases, enforcement actions, conferences (e.g., 'Smith v. Acme Corp')"
    )

    # Relationships (subject-predicate-object triples)
    relationships: List[dict] = dspy.OutputField(
        desc="Entity relationships in JSON format: "
             "[{'subject': 'EEOC', 'predicate': 'enforces', 'object': 'Title VII'}, ...]"
    )

    temporal_markers: List[dict] = dspy.OutputField(
        desc="Time-bound events: "
             "[{'event': 'CA SB 731 effective', 'date': '2023-01-01'}, ...]"
    )
```

### 3.3 Query Intent with Graph-Aware Retrieval

```python
class GraphAwareQueryIntent(dspy.Signature):
    """
    Parse query into retrieval strategy that combines vectors + graph traversal.
    """
    query: str = dspy.InputField()

    # Intent classification
    intent_type: Literal[
        "entity_focused",      # "Tell me about FCRA updates"
        "regional_scan",       # "What's happening in APAC?"
        "temporal_trend",      # "How has Ban the Box evolved?"
        "comparative",         # "Compare US vs EU background check laws"
        "relationship_query"   # "Who regulates FCRA compliance?"
    ] = dspy.OutputField()

    # Retrieval parameters
    target_entities: List[str] = dspy.OutputField(
        desc="Entities to use as graph seeds (e.g., ['FCRA', 'CFPB'])"
    )
    target_regions: List[str] = dspy.OutputField()
    target_topics: List[str] = dspy.OutputField()
    time_scope: Literal["last_week", "last_month", "last_quarter", "last_year", "all"] = dspy.OutputField()

    # Graph traversal instructions
    graph_strategy: Literal["none", "1-hop", "2-hop", "authority-subgraph"] = dspy.OutputField(
        desc="none = vector-only, 1-hop = entity + direct neighbors, "
             "authority-subgraph = high-PageRank entities within topic"
    )

    # Vector search optimization
    reformulated_query: str = dspy.OutputField(
        desc="Optimized query for embedding search, using domain terminology"
    )

    boost_recent: bool = dspy.OutputField(
        desc="True if query implies recency (e.g., 'recent', 'latest', 'new')"
    )
```

### 3.4 Multi-Document Synthesis with Authority Weighting

```python
class AuthorityWeightedSynthesis(dspy.Signature):
    """
    Synthesize answer from multiple sources, weighting by source authority.
    Handle contradictions by citing most authoritative source.
    """
    question: str = dspy.InputField()
    retrieved_articles: str = dspy.InputField(
        desc="JSON array: [{'title': ..., 'text': ..., 'source': ..., 'authority': 0.0-1.0}, ...]"
    )
    intent_type: str = dspy.InputField()

    # Synthesized answer
    answer: str = dspy.OutputField(
        desc="Comprehensive 3-5 paragraph answer. "
             "Structure: overview, key developments, regional variations, implications. "
             "Cite sources inline: 'According to EEOC guidance [1], ...'"
    )

    # Source management
    sources_cited: List[dict] = dspy.OutputField(
        desc="[{'id': 1, 'title': ..., 'authority': 0.85}, ...]. Ordered by importance."
    )

    conflicting_info: List[dict] = dspy.OutputField(
        desc="If sources contradict, list conflicts: "
             "[{'claim': 'X', 'source_a': 'Y', 'source_b': 'Z', 'authority_winner': 'source_a'}, ...]"
    )

    confidence: float = dspy.OutputField(
        desc="Answer confidence based on source authority and coverage (0.0-1.0)"
    )

    coverage_gaps: List[str] = dspy.OutputField(
        desc="Aspects of question not fully addressed by available sources"
    )

    followup_suggestions: List[str] = dspy.OutputField(
        desc="2-3 related questions for deeper exploration"
    )
```

---

## 4. Module Implementation - Recommendation Engine

### 4.1 Editorial Recommendation Module

```python
class EditorialRecommendationEngine(dspy.Module):
    """
    Predicts editorial appeal using collaborative filtering signals.
    Learns what makes content 'newsletter-worthy' from feedback.
    """
    def __init__(self, source_reputation_db, historical_selections):
        super().__init__()
        self.appeal_predictor = dspy.ChainOfThought(EditorialAppealSignature)
        self.source_db = source_reputation_db
        self.history = historical_selections

    def forward(self, article: dict) -> dict:
        """
        Score article for editorial appeal.
        Returns: Enhanced article dict with appeal_score, novelty, reasoning.
        """
        # Get source authority from reputation engine
        source_auth = self.source_db.get(article['source'], 0.5)

        # Find similar historical articles for context
        historical_context = self._get_similar_selections(article)

        # Compute recency
        from datetime import datetime
        pub_date = datetime.fromisoformat(article['published_date'])
        days_old = (datetime.now() - pub_date).days

        # Predict appeal
        prediction = self.appeal_predictor(
            title=article['title'],
            content_snippet=article['content'][:1000],
            source_name=article['source'],
            source_authority=source_auth,
            historical_context=historical_context,
            recency_days=days_old
        )

        # Enrich article with scores
        article['appeal'] = {
            'score': prediction.appeal_score,
            'novelty': prediction.novelty_score,
            'urgency': prediction.urgency_tier,
            'confidence': prediction.confidence,
            'reasoning': prediction.reasoning,
            'risk_flags': prediction.risk_flags
        }

        # Composite final score (for ranking)
        article['final_score'] = self._compute_composite_score(
            appeal=prediction.appeal_score,
            novelty=prediction.novelty_score,
            authority=source_auth,
            recency_days=days_old,
            urgency=prediction.urgency_tier
        )

        return article

    def _get_similar_selections(self, article):
        """
        Retrieve 3-5 similar articles from historical selections.
        Use for collaborative filtering signals.
        """
        # Simplified: would use embedding similarity in production
        similar = [
            a for a in self.history
            if any(t in article.get('topics', []) for t in a.get('topics', []))
        ][:5]

        context = "\n".join([
            f"- {a['title']} (selected, score: {a.get('score', 'N/A')})"
            for a in similar
        ])
        return context or "No similar historical selections found."

    def _compute_composite_score(self, appeal, novelty, authority, recency_days, urgency):
        """
        Weighted composite: appeal + novelty + authority + recency + urgency.
        """
        # Recency decay: exponential with half-life of 14 days
        import math
        recency_factor = math.exp(-0.05 * recency_days)

        # Urgency boost
        urgency_boost = {
            'breaking': 1.3,
            'important': 1.1,
            'routine': 1.0,
            'fyi': 0.9
        }.get(urgency, 1.0)

        # Weighted formula
        composite = (
            0.40 * appeal +
            0.20 * novelty +
            0.15 * authority +
            0.15 * recency_factor +
            0.10 * urgency_boost
        )

        return min(composite, 1.0)
```

### 4.2 Knowledge Graph Integration

```python
import networkx as nx
from datetime import datetime

class KnowledgeGraphEngine:
    """
    Build and query knowledge graph of articles, entities, sources.
    Enables graph-scoped retrieval and authority scoring.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_types = {}  # entity_name → type
        self.article_index = {}  # article_id → metadata

    def add_article(self, article_id, metadata, entities):
        """
        Add article node and entity relationships.

        Args:
            article_id: Unique ID
            metadata: {title, source, date, region, topics, score}
            entities: {organizations: [], regulations: [], jurisdictions: [], ...}
        """
        # Add article node
        self.graph.add_node(
            f"article_{article_id}",
            type='article',
            **metadata
        )
        self.article_index[article_id] = metadata

        # Add source node (if new)
        source_id = f"source_{metadata['source']}"
        if source_id not in self.graph:
            self.graph.add_node(source_id, type='source', name=metadata['source'])

        # Edge: article → published_by → source
        self.graph.add_edge(f"article_{article_id}", source_id, relation='published_by')

        # Add entity nodes and edges
        for entity_type, entity_list in entities.items():
            for entity_name in entity_list:
                entity_id = f"entity_{entity_name}"

                # Add entity node (if new)
                if entity_id not in self.graph:
                    self.graph.add_node(
                        entity_id,
                        type=entity_type,  # 'organization', 'regulation', etc.
                        name=entity_name
                    )
                    self.entity_types[entity_name] = entity_type

                # Edge: article → mentions → entity
                self.graph.add_edge(
                    f"article_{article_id}",
                    entity_id,
                    relation='mentions'
                )

    def add_entity_relationship(self, subject, predicate, object_entity):
        """
        Add inter-entity relationship (e.g., EEOC enforces FCRA).
        """
        sub_id = f"entity_{subject}"
        obj_id = f"entity_{object_entity}"

        if sub_id in self.graph and obj_id in self.graph:
            self.graph.add_edge(sub_id, obj_id, relation=predicate)

    def compute_authority_scores(self):
        """
        Run PageRank on source nodes to compute authority.
        Returns: {source_name: authority_score}
        """
        # Extract source subgraph
        source_nodes = [n for n,d in self.graph.nodes(data=True) if d.get('type')=='source']
        source_subgraph = self.graph.subgraph(source_nodes)

        # Run PageRank
        if len(source_subgraph) > 0:
            scores = nx.pagerank(source_subgraph, alpha=0.85)
            return {
                self.graph.nodes[node]['name']: score
                for node, score in scores.items()
            }
        return {}

    def graph_scoped_search(self, entity_seeds, max_hops=2):
        """
        Retrieve articles connected to seed entities via graph traversal.

        Args:
            entity_seeds: List of entity names (e.g., ['FCRA', 'CFPB'])
            max_hops: Max distance from seed entities

        Returns:
            List of article_ids
        """
        # Convert seeds to entity_ids
        seed_ids = [f"entity_{e}" for e in entity_seeds if f"entity_{e}" in self.graph]

        # BFS expansion
        visited = set()
        for seed in seed_ids:
            if seed in self.graph:
                # Get all nodes within max_hops
                for node in nx.single_source_shortest_path_length(
                    self.graph, seed, cutoff=max_hops
                ).keys():
                    visited.add(node)

        # Filter for article nodes
        article_nodes = [
            n for n in visited
            if self.graph.nodes.get(n, {}).get('type') == 'article'
        ]

        # Extract article IDs
        article_ids = [int(n.replace('article_', '')) for n in article_nodes]
        return article_ids

    def get_entity_centrality(self, top_n=20):
        """
        Identify most central entities (highest betweenness centrality).
        Useful for finding key regulations/organizations.
        """
        entity_nodes = [n for n,d in self.graph.nodes(data=True) if n.startswith('entity_')]
        subgraph = self.graph.subgraph(entity_nodes)

        centrality = nx.betweenness_centrality(subgraph)
        top_entities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]

        return [
            {
                'entity': self.graph.nodes[node]['name'],
                'type': self.graph.nodes[node]['type'],
                'centrality': score
            }
            for node, score in top_entities
        ]
```

### 4.3 Hybrid RAG with Graph Traversal

```python
class HybridGraphRAG(dspy.Module):
    """
    Retrieval-augmented generation combining:
    1. Vector similarity (ChromaDB)
    2. Graph traversal (NetworkX)
    3. Source authority weighting
    """
    def __init__(
        self,
        vector_retriever: ChromaDBRetriever,
        knowledge_graph: KnowledgeGraphEngine,
        k: int = 10
    ):
        super().__init__()
        self.vector_retriever = vector_retriever
        self.graph = knowledge_graph
        self.k = k

        # DSPy modules
        self.intent_analyzer = dspy.TypedPredictor(GraphAwareQueryIntent)
        self.answer_generator = dspy.ChainOfThought(AuthorityWeightedSynthesis)

    def forward(self, question: str) -> dspy.Prediction:
        """
        Answer question using hybrid retrieval strategy.
        """
        # Step 1: Analyze intent and determine retrieval strategy
        intent = self.intent_analyzer(query=question)

        # Step 2: Hybrid retrieval
        candidates = self._hybrid_retrieve(intent)

        # Step 3: Rerank by authority + relevance
        ranked_articles = self._rerank_by_authority(candidates, question)

        # Step 4: Format context for LLM
        context = self._format_context_with_authority(ranked_articles[:self.k])

        # Step 5: Generate answer with authority weighting
        response = self.answer_generator(
            question=question,
            retrieved_articles=context,
            intent_type=intent.intent_type
        )

        return response

    def _hybrid_retrieve(self, intent):
        """
        Combine vector search and graph traversal based on intent.
        """
        candidates = []

        # Strategy 1: Vector search (always)
        vector_results = self.vector_retriever(
            query=intent.reformulated_query,
            k=20,  # Over-fetch for reranking
            region=intent.target_regions[0] if intent.target_regions else None,
            topics=intent.target_topics
        )
        candidates.extend(vector_results.passages)

        # Strategy 2: Graph traversal (if entity-focused)
        if intent.graph_strategy in ['1-hop', '2-hop', 'authority-subgraph']:
            max_hops = 1 if intent.graph_strategy == '1-hop' else 2
            graph_article_ids = self.graph.graph_scoped_search(
                entity_seeds=intent.target_entities,
                max_hops=max_hops
            )

            # Fetch article metadata
            for article_id in graph_article_ids[:10]:
                if article_id in self.graph.article_index:
                    candidates.append(self.graph.article_index[article_id])

        # Deduplicate by article_id
        seen = set()
        unique_candidates = []
        for c in candidates:
            aid = c.get('article_id')
            if aid not in seen:
                seen.add(aid)
                unique_candidates.append(c)

        return unique_candidates

    def _rerank_by_authority(self, candidates, query):
        """
        Rerank by: 0.5 * vector_similarity + 0.3 * source_authority + 0.2 * recency
        """
        import math
        from datetime import datetime

        scored_candidates = []
        for article in candidates:
            # Vector similarity (already in article['similarity'])
            vector_score = article.get('similarity', 0.5)

            # Source authority (from graph PageRank)
            source_auth = self.graph.compute_authority_scores().get(
                article.get('source'), 0.5
            )

            # Recency (exponential decay)
            pub_date = datetime.fromisoformat(article.get('published_date', datetime.now().isoformat()))
            days_old = (datetime.now() - pub_date).days
            recency_score = math.exp(-0.05 * days_old)

            # Composite score
            composite = 0.5 * vector_score + 0.3 * source_auth + 0.2 * recency_score

            scored_candidates.append({
                **article,
                'composite_score': composite,
                'source_authority': source_auth
            })

        # Sort by composite score
        scored_candidates.sort(key=lambda x: x['composite_score'], reverse=True)
        return scored_candidates

    def _format_context_with_authority(self, articles):
        """
        Format articles as JSON with authority metadata for LLM.
        """
        import json
        context_items = [
            {
                'title': a['title'],
                'text': a.get('text', '')[:800],
                'source': a['source'],
                'authority': a.get('source_authority', 0.5),
                'date': a['published_date'],
                'url': a.get('url', '')
            }
            for a in articles
        ]
        return json.dumps(context_items, indent=2)
```

---

## 5. Optimization Strategy - Active Learning Focus

### 5.1 Smart Labeling with Active Learning

```python
class ActiveLabelingOrchestrator:
    """
    Minimize labeling effort by selecting most informative examples.
    Target: 200 labeled examples achieve 75% F1 (vs 1000+ with random sampling)
    """
    def __init__(self, classifier_module, uncertainty_threshold=0.6):
        self.classifier = classifier_module
        self.threshold = uncertainty_threshold
        self.labeled_pool = []
        self.unlabeled_pool = []

    def select_batch_for_labeling(self, budget=50):
        """
        Select top-N articles for human labeling based on:
        1. Classification uncertainty (entropy)
        2. Novelty (distance to training set)
        3. Source drift (changing editorial patterns)
        4. Strategic coverage (underrepresented regions/topics)
        """
        candidates = []

        for article in self.unlabeled_pool:
            # Score 1: Uncertainty
            uncertainty = self._compute_entropy(article)

            # Score 2: Novelty
            novelty = self._compute_novelty(article)

            # Score 3: Source drift
            drift = self._detect_source_drift(article['source'])

            # Score 4: Coverage gap
            coverage_gap = self._coverage_gap_score(article)

            # Combined priority
            priority = (
                0.35 * uncertainty +
                0.25 * novelty +
                0.20 * drift +
                0.20 * coverage_gap
            )

            candidates.append({
                'article': article,
                'priority': priority,
                'reasons': {
                    'uncertainty': uncertainty,
                    'novelty': novelty,
                    'drift': drift,
                    'coverage_gap': coverage_gap
                }
            })

        # Sort by priority, return top budget
        candidates.sort(key=lambda x: x['priority'], reverse=True)
        return candidates[:budget]

    def _compute_entropy(self, article):
        """
        Classification probability entropy.
        High entropy = model is uncertain.
        """
        # Get classification probabilities from model
        # (Simplified: assume we have topic probabilities)
        probs = article.get('classification_probabilities', [0.5, 0.5])

        import numpy as np
        return -np.sum(probs * np.log(probs + 1e-10))

    def _compute_novelty(self, article):
        """
        Distance to nearest neighbor in labeled pool.
        Large distance = novel content.
        """
        if len(self.labeled_pool) == 0:
            return 1.0

        # Simplified: would use embedding distance in production
        # For now, check topic overlap
        article_topics = set(article.get('topics', []))
        min_distance = 1.0

        for labeled in self.labeled_pool:
            labeled_topics = set(labeled.get('topics', []))
            jaccard = len(article_topics & labeled_topics) / (len(article_topics | labeled_topics) + 1e-6)
            distance = 1 - jaccard
            min_distance = min(min_distance, distance)

        return min_distance

    def _detect_source_drift(self, source):
        """
        Has source's editorial acceptance rate changed recently?
        Drift indicates need for retraining on that source.
        """
        # Get recent (30-day) and historical acceptance rates
        recent_rate = self._get_acceptance_rate(source, days=30)
        historical_rate = self._get_acceptance_rate(source, days=180)

        return abs(recent_rate - historical_rate)

    def _coverage_gap_score(self, article):
        """
        Is this article from an underrepresented region/topic in training?
        """
        # Count labels by region/topic
        from collections import Counter
        region_counts = Counter(a['region'] for a in self.labeled_pool)
        topic_counts = Counter(t for a in self.labeled_pool for t in a.get('topics', []))

        # Score inversely proportional to count (underrepresented = high score)
        article_region = article.get('region')
        article_topics = article.get('topics', [])

        region_score = 1.0 / (region_counts.get(article_region, 0) + 1)
        topic_score = np.mean([1.0 / (topic_counts.get(t, 0) + 1) for t in article_topics])

        return 0.5 * region_score + 0.5 * topic_score

    def _get_acceptance_rate(self, source, days):
        """
        Fraction of articles from source that were editorially approved.
        """
        # Placeholder: query database for editorial decisions
        # Filter by source and date range, compute accept / total
        return 0.5  # Placeholder
```

### 5.2 Incremental Optimization Strategy

```python
class IncrementalOptimizer:
    """
    Weekly optimization with new labeled examples.
    Avoids full retraining cost while incorporating feedback.
    """
    def __init__(self, base_classifier):
        self.classifier = base_classifier
        self.optimization_history = []

    def weekly_optimization(self, new_labeled_examples):
        """
        Run BootstrapFewShot with incremental examples.
        Cost: ~$2-5 per run vs ~$15 for full MIPROv2.
        """
        from dspy.teleprompt import BootstrapFewShot

        # Combine new examples with stratified sample of old
        training_set = self._build_stratified_trainset(new_labeled_examples)

        # Optimize with BootstrapFewShot (fast, cheap)
        optimizer = BootstrapFewShot(
            metric=self.classification_metric,
            max_bootstrapped_demos=8,
            max_labeled_demos=16,
            max_rounds=2  # Reduce rounds for cost control
        )

        optimized = optimizer.compile(
            self.classifier,
            trainset=training_set
        )

        # Evaluate on held-out validation set
        val_score = self._evaluate(optimized)

        # Only deploy if improvement
        if val_score > self._get_current_score():
            self.classifier = optimized
            self.save_checkpoint(optimized, val_score)
            return {"status": "deployed", "score": val_score}
        else:
            return {"status": "kept_previous", "score": val_score}

    def _build_stratified_trainset(self, new_examples, total_size=200):
        """
        Stratified sampling: ensure region/topic balance.
        """
        # Implementation: sample proportionally from each region/topic
        # to maintain balanced training set
        pass

    def classification_metric(self, example, pred, trace=None):
        """
        F1 metric: 30% region + 70% topics
        """
        if pred is None:
            return 0.0

        region_match = 1.0 if pred.region == example.region else 0.0

        pred_topics = set(pred.topics)
        true_topics = set(example.topics)
        if len(pred_topics | true_topics) == 0:
            topic_f1 = 1.0
        else:
            precision = len(pred_topics & true_topics) / len(pred_topics) if pred_topics else 0
            recall = len(pred_topics & true_topics) / len(true_topics) if true_topics else 0
            topic_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return 0.3 * region_match + 0.7 * topic_f1
```

---

## 6. Phased Implementation - Rapid Prototyping

### Phase 1: Foundation (Weeks 1-2) - Prove the Concept

**Goal**: End-to-end pipeline with basic recommendation scoring

**Deliverables**:
1. RSS ingestion (5 pilot sources)
2. Two-stage classification (relevance filter + full classification)
3. SQLite storage + basic editorial appeal scoring
4. Label 100 articles manually
5. Run BootstrapFewShot optimization
6. Achieve 60%+ recall on test set

**Key Innovation**: Start with recommendation mindset from day 1
- Don't just classify—predict editorial appeal
- Track which sources get selected more often
- Use historical selections as "training signal"

**Success Metric**: 30-40 articles/day in review queue, 60%+ precision

---

### Phase 2: Knowledge Graph (Weeks 3-4) - Add Intelligence

**Goal**: Entity extraction + graph-based authority scoring

**Deliverables**:
1. Entity extraction module (organizations, laws, jurisdictions)
2. NetworkX knowledge graph initialization
3. Source authority scoring (PageRank)
4. Graph-scoped retrieval in queries
5. Label 100 more articles (active learning selection)
6. Retrain with 200 total examples

**Key Innovation**: Sources compete for attention
- Authority scores boost/penalize articles
- Citation tracking reveals "first to report"
- Entity relationships enable "related content" discovery

**Success Metric**: Source authority scores correlate with editorial decisions (ρ > 0.6)

---

### Phase 3: Active Learning (Weeks 5-6) - Minimize Labeling

**Goal**: Smart labeling strategy + continuous improvement

**Deliverables**:
1. Uncertainty sampling module
2. Coverage gap analysis (underrepresented regions/topics)
3. Source drift detection
4. Weekly optimization scheduler
5. A/B testing framework (old vs new classifier)
6. Feedback collection interface

**Key Innovation**: System identifies its own blind spots
- High uncertainty → human review
- Novel patterns → priority labeling
- Source drift → targeted retraining

**Success Metric**: 75%+ F1 with only 200 labeled examples

---

### Phase 4: RAG Enhancement (Weeks 7-8) - Production Polish

**Goal**: Hybrid retrieval + authority-weighted synthesis

**Deliverables**:
1. ChromaDB vector store
2. Hybrid retriever (vector + graph)
3. Query intent analysis
4. Multi-document synthesis with citations
5. Query caching layer
6. CLI interface with rich output

**Key Innovation**: Queries leverage full context
- Entity-focused queries use graph traversal
- Authority weighting in answer generation
- Conflicting sources handled gracefully

**Success Metric**: <5 sec query response, 80%+ user satisfaction

---

## 7. Risks & Creative Mitigations

### 7.1 Technical Risks

| Risk | Creative Mitigation |
|------|---------------------|
| **Graph becomes too large** | Implement "temporal pruning": archive nodes older than 6 months to separate cold-storage graph. Query both if needed. |
| **Authority scores amplify bias** | Track "diversity" separately—occasionally boost low-authority sources for exploration (ε-greedy strategy). |
| **Active learning selects non-diverse examples** | Add explicit diversity term to selection criterion (coverage gap score). |
| **Knowledge graph relationships are noisy** | Use confidence scoring on relationships; prune low-confidence edges monthly. |
| **Novel entity patterns overwhelm system** | Maintain "entity vocabulary" allowlist; flag completely new entities for human verification. |

### 7.2 Operational Risks

| Risk | Creative Mitigation |
|------|---------------------|
| **Editors disagree on article quality** | Multi-rater labeling for borderline cases; track inter-rater agreement; use ensemble of editor preferences. |
| **Source reputation creates filter bubble** | Monthly "source diversity audit": ensure low-authority sources get occasional exposure. |
| **Novelty scoring penalizes legitimate updates** | Distinguish "update to existing story" from "redundant coverage" via entity overlap analysis. |
| **Graph authority favors established sources** | New source "bootstrap period": neutral authority (0.5) for first 30 days, then adjust. |

---

## 8. Success Metrics - Recommendation System Style

### 8.1 Recommendation Quality Metrics

| Metric | Target | Netflix Equivalent |
|--------|--------|-------------------|
| **Precision@20** | >70% | Top 20 recommendations match user's watch history |
| **Recall@50** | >75% | Capture 75% of manually selected articles in top 50 |
| **Mean Reciprocal Rank (MRR)** | >0.6 | On average, first relevant article appears in top 2 |
| **Novelty Rate** | 20-30% | 20-30% of recommendations are genuinely new angles |
| **Source Diversity (Gini)** | <0.7 | Avoid over-concentration on top 3 sources |

### 8.2 Learning System Metrics

| Metric | Target | Interpretation |
|--------|--------|----------------|
| **Active Learning Efficiency** | 75% F1 with 200 examples | vs 65% F1 with random 200 |
| **Optimization ROI** | +10% F1 per $5 spent | Cost-benefit of weekly retraining |
| **Source Drift Detection Rate** | Flag 80%+ of quality shifts | Catch sources that change editorial stance |
| **Graph Query Speedup** | 2x faster than pure vector | Entity-focused queries benefit from graph |

### 8.3 Business Impact Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Time Savings** | >15 hours/week | Before/after newsletter creation time |
| **Miss Rate** | <5% | Important articles missed by system but found manually |
| **False Positive Rate** | <30% | Articles in review queue that editors reject |
| **Query Utility** | >80% | CLI queries rated "helpful" by staff |

---

## 9. Appendix - Implementation Code Snippets

### 9.1 Complete Classification Pipeline with Recommendation

```python
import dspy
from typing import List, Dict, Any

class RecommendationPipeline(dspy.Module):
    """
    End-to-end: Ingestion → Classification → Appeal Scoring → Graph Update
    """
    def __init__(
        self,
        source_reputation_db: dict,
        knowledge_graph: KnowledgeGraphEngine,
        active_learner: ActiveLabelingOrchestrator
    ):
        super().__init__()

        # Stage 1: Quick filter
        self.relevance_filter = dspy.Predict(RelevanceFilter)

        # Stage 2: Classification
        self.classifier = dspy.ChainOfThought(ArticleClassifier)

        # Stage 3: Entity extraction
        self.entity_extractor = dspy.TypedPredictor(EntityRelationshipExtractor)

        # Stage 4: Editorial appeal
        self.appeal_scorer = EditorialRecommendationEngine(
            source_reputation_db,
            historical_selections=[]
        )

        self.graph = knowledge_graph
        self.active_learner = active_learner

    def forward(self, articles: List[dict]) -> Dict[str, Any]:
        """
        Process batch of articles through full pipeline.
        Returns: Shortlist + analytics + active learning candidates.
        """
        results = {
            'shortlist': [],
            'review_queue': [],
            'active_learning_batch': [],
            'stats': {}
        }

        processed = 0
        filtered_out = 0

        for article in articles:
            # Stage 1: Quick relevance check
            filter_result = self.relevance_filter(
                title=article['title'],
                snippet=article['content'][:300]
            )

            if not filter_result.is_relevant or filter_result.confidence < 0.5:
                filtered_out += 1
                continue

            # Stage 2: Full classification
            classification = self.classifier(
                title=article['title'],
                content=article['content'][:1500]
            )

            article['region'] = classification.region
            article['country'] = classification.country
            article['topics'] = classification.topics

            # Stage 3: Entity extraction
            entities = self.entity_extractor(
                text=article['content'][:2000],
                region=classification.region
            )

            # Stage 4: Editorial appeal scoring
            article_with_appeal = self.appeal_scorer(article)

            # Stage 5: Update knowledge graph
            self.graph.add_article(
                article_id=article['id'],
                metadata={
                    'title': article['title'],
                    'source': article['source'],
                    'date': article['published_date'],
                    'region': classification.region,
                    'topics': classification.topics,
                    'score': article_with_appeal['final_score']
                },
                entities={
                    'organizations': entities.organizations,
                    'regulations': entities.regulations,
                    'jurisdictions': entities.jurisdictions
                }
            )

            # Add entity relationships to graph
            for rel in entities.relationships:
                self.graph.add_entity_relationship(
                    subject=rel['subject'],
                    predicate=rel['predicate'],
                    object_entity=rel['object']
                )

            # Routing logic
            if article_with_appeal['final_score'] > 0.7:
                results['shortlist'].append(article_with_appeal)
            elif article_with_appeal['final_score'] > 0.5:
                results['review_queue'].append(article_with_appeal)

            # Active learning: flag uncertain articles
            if article_with_appeal['appeal']['confidence'] < 0.6:
                self.active_learner.unlabeled_pool.append(article_with_appeal)

            processed += 1

        # Select batch for human labeling
        results['active_learning_batch'] = self.active_learner.select_batch_for_labeling(budget=20)

        # Compute source authority scores
        authority_scores = self.graph.compute_authority_scores()

        # Stats
        results['stats'] = {
            'total_ingested': len(articles),
            'filtered_out': filtered_out,
            'processed': processed,
            'shortlist_size': len(results['shortlist']),
            'review_queue_size': len(results['review_queue']),
            'active_learning_size': len(results['active_learning_batch']),
            'top_sources': sorted(
                authority_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }

        return results
```

---

## 10. Handoff Summary

### 10.1 What Makes This Architecture Different?

**Traditional Approach**:
```
Articles → Classify → Filter → Store → Query
```

**Our Recommendation Approach**:
```
Articles → Predict Editorial Appeal → Rank by Confidence →
  Update Source Reputation Graph → Learn from Feedback →
    Identify Blind Spots → Target Labeling → Retrain
```

**Key Innovations**:
1. **Recommendation mindset**: Predict "would editors select this?" vs "what category is this?"
2. **Source reputation layer**: PageRank-style authority scoring rewards quality sources
3. **Active learning**: System identifies its own training gaps (200 labels vs 1000+)
4. **Knowledge graph**: Enables "who regulates what?" and temporal relationship queries
5. **Continuous improvement**: Weekly optimization loops instead of static deployment

### 10.2 Critical Path for Success

**Week 1**: Prove recommendation scoring works
- Implement EditorialAppealSignature
- Score 100 historical articles
- Validate: appeal_score correlates with manual selections (ρ > 0.5)

**Week 3**: Prove graph adds value
- Build knowledge graph from 500 articles
- Run graph-scoped vs vector-only retrieval
- Validate: graph queries find 30%+ more relevant results

**Week 5**: Prove active learning reduces labeling burden
- Run uncertainty sampling on 1000 unlabeled articles
- Label top 50 vs random 50
- Validate: uncertainty batch improves F1 by 10+ points

**Week 7**: Integrate and polish
- Full pipeline end-to-end
- CLI with hybrid retrieval
- Performance tuning and caching

### 10.3 Open Questions for Stakeholders

1. **Editorial diversity**: Should we explicitly boost underrepresented sources even if authority is low?
2. **Novelty threshold**: What % of shortlist should be "familiar updates" vs "novel angles"?
3. **Feedback frequency**: Weekly labeling sessions (50 articles) or ad-hoc as needed?
4. **Graph scope**: Archive old graph nodes after 6 months, or keep indefinitely for historical queries?

---

**Document Status**: Design Exploration Complete
**Recommended Next Step**: Prototype Week 1 deliverables to validate recommendation approach
**Estimated PoC Timeline**: 8 weeks to production-ready system
**Estimated Cost**: <$50/month operational + $20-30 one-time optimization

---

*This creative exploration architecture demonstrates that newsletter curation is fundamentally a recommendation problem. By borrowing patterns from Netflix, Google PageRank, and active learning research, we build a system that doesn't just classify—it learns what makes content valuable and continuously improves its editorial judgment.*
