# DSPy Newsletter Research Tool - Architect Handoff Document

## Mission Statement

Design a DSPy-based architecture for PreEmploymentDirectory's automated newsletter research tool. This system will transform their manual article discovery process (LinkedIn searches, email subscriptions, Dropbox dumps, ChatGPT summaries) into an intelligent, automated pipeline serving 2,100+ background screening firms worldwide.

**Business Impact**: Replace 20+ hours/week of manual research with automated candidate identification, achieving 70%+ recall while maintaining quality signal-to-noise ratio.

---

## Problem Domain Summary

### Current State (Manual Process)
1. Staff search LinkedIn and email subscriptions
2. Dump articles in Dropbox
3. Use ChatGPT to summarize
4. Manual categorization and curation

### Target State (Automated Pipeline)
1. **Ingest**: Monitor 247+ sources via RSS/scraping/email parsing
2. **Classify**: Categorize by 6 regions × 8 topics
3. **Score**: Relevance scoring with keyword/semantic matching
4. **Store**: Vector embeddings for RAG retrieval
5. **Query**: CLI for natural language questions like "recent regulation changes in APAC"
6. **Output**: Daily shortlist of 20-50 candidate articles for human review

---

## Source Inventory (247+ Sources)

### Tier 1: High-Volume RSS (20 feeds, 500+ articles/day)
- JDSupra Labor/Employment
- Lexology (custom feeds, free registration)
- National Law Review
- HR Dive
- EDPB News (EU)
- Federal Register, CFPB, FTC, EEOC, DOL (US Federal)

### Tier 2: Company & Regional RSS (15 feeds)
- First Advantage, Sterling, HireRight blogs
- OPC Canada, Singapore PDPC
- Morgan Lewis Middle East, China Briefing

### Tier 3: Scraping Required (25+ sources)
- PBSA (Professional Background Screening Association)
- DPAs globally (Ireland, UK ICO, Australia OAIC, Hong Kong PCPD)
- State-level Ban the Box trackers
- E-Verify updates

### Tier 4: Email/Social Integration
- Newsletter subscriptions (OAIC, Nordic DPAs, NELP)
- Social media monitoring for DPA announcements

---

## Classification Taxonomy

### Regions (6)
1. Africa & Middle East
2. Asia Pacific
3. Europe
4. North America & Caribbean
5. South America
6. Worldwide (cross-regional)

### Topics (8)
1. Regulatory/legal changes
2. Criminal background check requirements
3. Education/credential verification
4. Immigration/right-to-work
5. Industry M&A/company news
6. Technology/product announcements
7. Conference/event news
8. Court cases/legal precedents

---

## Relevance Keywords

### Primary (High Signal)
- background check, background screening
- employment screening, pre-employment
- criminal record, criminal history
- right to work, work authorization
- Ban the Box, fair chance
- FCRA (US), GDPR (EU), data protection
- credential verification, education verification

### Secondary (Supporting)
- drug testing, drug screening
- reference check
- identity verification
- continuous monitoring
- adverse action
- consumer reporting agency

### Regional Triggers
- [Country] + "employment law"
- [Country] + "background check"
- Specific legislation (PIPEDA, POPIA, LGPD, DPDP, etc.)

---

## Technical Requirements

### Stack
- **Language**: Python
- **LLM**: OpenAI/Anthropic APIs
- **Framework**: DSPy for LM programming
- **Vector DB**: ChromaDB or ColBERTv2
- **Processing**: Daily batch (not real-time)
- **Storage**: Local JSON files for PoC

### DSPy Components to Consider
- **Signatures**: Define input/output schemas for classification, scoring, summarization
- **Modules**: Predict, ChainOfThought, TypedPredictor, ReAct
- **Optimizers**: BootstrapFewShot with labeled newsletter examples, MIPROv2
- **Retrieval**: dspy.Retrieve with ChromaDB/ColBERTv2

### Output Schema
```json
{
  "title": "Article title",
  "source": "Publication name",
  "url": "https://...",
  "published_date": "2025-01-10",
  "region": "Asia Pacific",
  "country": "Australia",
  "topics": ["regulatory", "criminal_records"],
  "relevance_score": 0.85,
  "summary": "2-3 sentence summary",
  "reasoning": "Why flagged as relevant"
}
```

### Success Criteria
- 70%+ recall vs manual process
- Manageable false positive rate
- 20-50 candidate articles per day
- CLI queries return relevant results in <5 seconds

---

## Architectural Constraints

1. **Cost**: PoC budget - minimize API calls
2. **Latency**: Daily batch acceptable, not real-time
3. **Scale**: 500+ articles/day ingestion
4. **Quality**: Prefer over-inclusion (humans filter) vs strict precision
5. **Deduplication**: Same story across sources should be handled
6. **Multilingual**: ~70% English, need strategy for non-English content

---

## Reference: DSPy Patterns

### Signatures
```python
class ArticleClassifier(dspy.Signature):
    """Classify article by region and topic."""
    title = dspy.InputField()
    content = dspy.InputField()
    region = dspy.OutputField(desc="one of: africa_me, apac, europe, north_america, south_america, worldwide")
    topics = dspy.OutputField(desc="list of applicable topics")
```

### Modules
- `dspy.Predict` - Fast, simple classification
- `dspy.ChainOfThought` - Reasoning for relevance scoring
- `dspy.TypedPredictor` - Structured JSON output
- `dspy.ReAct` - Agent with tools for complex queries

### Optimization
```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=relevance_accuracy,
    max_bootstrapped_demos=5
)
optimized_classifier = optimizer.compile(classifier, trainset=labeled_articles)
```

### RAG Pattern
```python
class NewsletterRAG(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=10)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)
```

---

## Design Questions to Address

1. **Pipeline Architecture**: Monolithic vs modular vs agent-based?
2. **Classification Strategy**: Two-stage (region, then topic) or single multi-label?
3. **Relevance Scoring**: Keyword-based, semantic, or hybrid?
4. **Deduplication**: Before classification or after?
5. **Embedding Strategy**: Per-article or chunked?
6. **Optimization**: Which DSPy optimizer and training data strategy?
7. **CLI Design**: Simple query or conversational agent?
8. **Phase Decomposition**: How to break into incremental deliverables?

---

## Evaluation Criteria for Solutions

1. **Architectural Soundness** (20%): Clear separation of concerns, extensibility
2. **DSPy Utilization** (20%): Effective use of signatures, modules, optimizers
3. **Practical Feasibility** (20%): PoC-appropriate, cost-conscious
4. **RAG Effectiveness** (15%): Query quality and retrieval precision
5. **Scalability Path** (10%): Room to grow beyond PoC
6. **Innovation** (10%): Novel approaches worth exploring
7. **Risk Mitigation** (5%): Handling edge cases and failures

---

## Architect Instructions

You are a solution architect designing this DSPy-based system. Your task:

1. **Propose a complete architecture** addressing all components
2. **Define DSPy signatures** for key operations
3. **Select appropriate modules** with justification
4. **Design the optimization strategy** with training data approach
5. **Specify the RAG implementation** for CLI queries
6. **Outline a phased implementation** (3-4 phases)
7. **Identify risks and mitigations**

**Use your assigned reasoning strategy** to approach the problem. Be specific and include code examples where helpful.

---

*End of Handoff Document*
