# TDD Testing Strategy for DSPy Newsletter Research Tool PoC

**Date**: 2026-01-12
**Version**: 1.0
**Status**: Active

---

## Overview

This document defines the Test-Driven Development (TDD) approach for the DSPy Newsletter Research Tool PoC. Each implementation step follows the **Red-Green-Refactor** cycle with independent validation via `validation-agent`.

### Core Principles

1. **RED**: Write failing tests FIRST - tests must fail before implementation
2. **GREEN**: Write minimal code to make tests pass
3. **REFACTOR**: Improve code while keeping tests green
4. **VALIDATE**: Use `validation-agent` for independent test execution and evaluation

---

## Test Hierarchy Aligned with Beads Tasks

| Beads Task | Test Module | Test Focus |
|------------|-------------|------------|
| `dspy-fx7` | `test_infrastructure.py` | Docker, Weaviate, environment |
| `dspy-4tp` | `test_models.py` | Pydantic models, taxonomy enums |
| `dspy-7eh` | `test_ingestion.py` | RSS parsing, article extraction |
| `dspy-50z` | `test_deduplication.py` | Hash + fuzzy matching |
| `dspy-2yy` | `test_prefilter.py` | Tiny LM relevance scoring |
| `dspy-0s9` | `test_classification.py` | Region, topic, scoring DSPy modules |
| `dspy-13p` | `test_storage.py` | Weaviate CRUD operations |
| `dspy-v72` | `test_query_agent.py` | QUIPLER retrieval, ReAct synthesis |
| `dspy-bot` | `test_cli.py` | Click commands, rich output |
| `dspy-4aa` | `test_optimization.py` | BootstrapFewShot, metrics |

---

## DSPy Module Testing Patterns

### Pattern 1: dspy.Predict Tests

**Used for**: Simple transformations without reasoning chain

```python
# tests/test_prefilter.py

import pytest
import dspy
from unittest.mock import Mock, patch
from src.prefilter import RelevancePreFilter, TinyLMRelevanceFilter

class TestRelevancePreFilterSignature:
    """RED: Define expected behavior BEFORE implementation"""

    def test_signature_has_required_input_fields(self):
        """Test signature defines expected inputs"""
        # RED: This test should FAIL until signature is implemented
        sig = RelevancePreFilter
        assert 'title' in sig.input_fields
        assert 'content_preview' in sig.input_fields
        assert 'source_category' in sig.input_fields

    def test_signature_has_required_output_fields(self):
        """Test signature defines expected outputs"""
        # RED: This test should FAIL until signature is implemented
        sig = RelevancePreFilter
        assert 'is_relevant' in sig.output_fields
        assert 'confidence' in sig.output_fields
        assert 'reason' in sig.output_fields

    def test_output_types_are_correct(self):
        """Test output field types"""
        sig = RelevancePreFilter
        # is_relevant should be bool
        assert sig.output_fields['is_relevant'].annotation == bool
        # confidence should be float
        assert sig.output_fields['confidence'].annotation == float


class TestTinyLMRelevanceFilter:
    """Integration tests for the filter module"""

    @pytest.fixture
    def mock_lm(self):
        """Mock the language model for deterministic testing"""
        with patch('dspy.settings.configure') as mock:
            yield mock

    def test_filter_returns_prediction_object(self, mock_lm):
        """Test filter returns dspy.Prediction with expected fields"""
        # RED: Should fail until TinyLMRelevanceFilter is implemented
        filter_module = TinyLMRelevanceFilter()
        result = filter_module(
            title="New FCRA Amendment Proposed",
            content_preview="The Fair Credit Reporting Act...",
            source_category="legal"
        )

        assert hasattr(result, 'is_relevant')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'reason')

    def test_relevant_article_scores_high(self, mock_lm):
        """Test relevant articles get high confidence"""
        filter_module = TinyLMRelevanceFilter()

        result = filter_module(
            title="Background Screening Industry Report 2026",
            content_preview="Employment background checks continue to evolve...",
            source_category="industry"
        )

        assert result.is_relevant == True
        assert result.confidence >= 0.7

    def test_irrelevant_article_scores_low(self, mock_lm):
        """Test irrelevant articles get low confidence"""
        filter_module = TinyLMRelevanceFilter()

        result = filter_module(
            title="Celebrity Gossip Daily",
            content_preview="Hollywood stars attended the award show...",
            source_category="entertainment"
        )

        assert result.is_relevant == False
        assert result.confidence <= 0.3
```

### Pattern 2: dspy.TypedPredictor Tests

**Used for**: Classification with structured Pydantic output

```python
# tests/test_classification.py

import pytest
from enum import Enum
from pydantic import BaseModel
import dspy
from src.classification import (
    ArticleClassification,
    RegionEnum,
    TopicEnum,
    ClassificationModule
)


class TestArticleClassificationModel:
    """RED: Test Pydantic model before implementation"""

    def test_classification_model_has_region(self):
        """Test model includes region field"""
        fields = ArticleClassification.model_fields
        assert 'region' in fields

    def test_classification_model_has_topics(self):
        """Test model includes topics field as list"""
        fields = ArticleClassification.model_fields
        assert 'topics' in fields
        # Should be a list of topics

    def test_classification_model_has_relevance_score(self):
        """Test model includes relevance_score"""
        fields = ArticleClassification.model_fields
        assert 'relevance_score' in fields


class TestRegionEnum:
    """Test region taxonomy"""

    def test_region_enum_has_required_values(self):
        """Test all required regions exist"""
        required_regions = [
            'AFRICA_ME', 'APAC', 'EUROPE',
            'N_AMERICA_CARIBBEAN', 'S_AMERICA', 'WORLDWIDE'
        ]
        for region in required_regions:
            assert hasattr(RegionEnum, region)


class TestTopicEnum:
    """Test topic taxonomy"""

    def test_topic_enum_has_required_values(self):
        """Test all required topics exist"""
        required_topics = [
            'REGULATORY', 'CRIMINAL_RECORDS', 'CREDENTIALS',
            'IMMIGRATION', 'M_AND_A', 'TECHNOLOGY',
            'EVENTS', 'COURT_CASES'
        ]
        for topic in required_topics:
            assert hasattr(TopicEnum, topic)


class TestClassificationModule:
    """Integration tests for TypedPredictor module"""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance"""
        return ClassificationModule()

    def test_classification_returns_valid_pydantic(self, classifier):
        """Test output is valid ArticleClassification"""
        result = classifier(
            title="New Ban the Box Law in California",
            content="California passed legislation restricting...",
            source_url="https://example.com/article"
        )

        # Should return ArticleClassification instance
        assert isinstance(result.classification, ArticleClassification)

    def test_classification_assigns_correct_region(self, classifier):
        """Test region assignment accuracy"""
        result = classifier(
            title="GDPR Enforcement Update in Germany",
            content="German regulators announced new enforcement...",
            source_url="https://example.com"
        )

        assert result.classification.region == RegionEnum.EUROPE

    def test_classification_assigns_multiple_topics(self, classifier):
        """Test multi-label topic assignment"""
        result = classifier(
            title="FCRA Class Action Settlement",
            content="A major background screening firm settled...",
            source_url="https://example.com"
        )

        # Should have both REGULATORY and COURT_CASES
        assert TopicEnum.REGULATORY in result.classification.topics
        assert TopicEnum.COURT_CASES in result.classification.topics
```

### Pattern 3: dspy.ChainOfThought Tests

**Used for**: Scoring with explainable reasoning

```python
# tests/test_scoring.py

import pytest
import dspy
from src.scoring import RelevanceScoringSignature, RelevanceScorer


class TestRelevanceScoringSignature:
    """Test ChainOfThought signature for scoring"""

    def test_signature_has_reasoning_output(self):
        """ChainOfThought should include reasoning"""
        sig = RelevanceScoringSignature
        # ChainOfThought adds 'rationale' automatically
        # Our signature should have the core fields
        assert 'title' in sig.input_fields
        assert 'content' in sig.input_fields
        assert 'relevance_score' in sig.output_fields

    def test_score_is_bounded(self):
        """Test score output is 0-100"""
        scorer = RelevanceScorer()
        result = scorer(
            title="Test Article",
            content="Some content about background checks..."
        )

        assert 0 <= result.relevance_score <= 100


class TestRelevanceScorer:
    """Integration tests for scoring module"""

    def test_scoring_provides_rationale(self):
        """Test ChainOfThought provides reasoning"""
        scorer = RelevanceScorer()
        result = scorer(
            title="FCRA Amendment Bill",
            content="Congress is considering amendments to FCRA..."
        )

        # ChainOfThought should provide rationale
        assert hasattr(result, 'rationale')
        assert len(result.rationale) > 0

    def test_high_relevance_content_scores_high(self):
        """Test highly relevant content gets high score"""
        scorer = RelevanceScorer()
        result = scorer(
            title="Background Check Industry Trends 2026",
            content="""
            The employment screening industry continues to evolve
            with new FCRA regulations, ban-the-box laws expanding
            across states, and increasing focus on credential verification.
            """
        )

        assert result.relevance_score >= 70

    def test_irrelevant_content_scores_low(self):
        """Test irrelevant content gets low score"""
        scorer = RelevanceScorer()
        result = scorer(
            title="Recipe of the Day",
            content="Today we're making chocolate chip cookies..."
        )

        assert result.relevance_score <= 20
```

### Pattern 4: dspy.ReAct Tests

**Used for**: Query agent with tool use

```python
# tests/test_query_agent.py

import pytest
from unittest.mock import Mock, patch, MagicMock
import dspy
from src.query_agent import (
    NewsletterQueryAgent,
    QueryResponseSignature,
    filter_by_date,
    filter_by_region,
    filter_by_topic
)


class TestQueryResponseSignature:
    """Test ReAct signature for query responses"""

    def test_signature_has_context_input(self):
        """Test signature accepts context from retrieval"""
        sig = QueryResponseSignature
        assert 'context' in sig.input_fields

    def test_signature_has_question_input(self):
        """Test signature accepts user question"""
        sig = QueryResponseSignature
        assert 'question' in sig.input_fields

    def test_signature_outputs_answer(self):
        """Test signature produces answer"""
        sig = QueryResponseSignature
        assert 'answer' in sig.output_fields

    def test_signature_outputs_sources(self):
        """Test signature includes sources"""
        sig = QueryResponseSignature
        assert 'sources' in sig.output_fields


class TestQueryTools:
    """Test ReAct tool functions"""

    def test_filter_by_date_tool_signature(self):
        """Test date filter tool has correct signature"""
        # Tool should accept start_date, end_date
        import inspect
        sig = inspect.signature(filter_by_date)
        params = list(sig.parameters.keys())
        assert 'start_date' in params
        assert 'end_date' in params

    def test_filter_by_region_returns_filtered_results(self):
        """Test region filter reduces results"""
        mock_articles = [
            {'id': '1', 'region': 'EUROPE'},
            {'id': '2', 'region': 'APAC'},
            {'id': '3', 'region': 'EUROPE'},
        ]

        result = filter_by_region(mock_articles, region='EUROPE')
        assert len(result) == 2
        assert all(a['region'] == 'EUROPE' for a in result)


class TestNewsletterQueryAgent:
    """Integration tests for ReAct query agent"""

    @pytest.fixture
    def mock_weaviate(self):
        """Mock Weaviate client"""
        with patch('weaviate.connect_to_local') as mock:
            yield mock

    @pytest.fixture
    def mock_cohere(self):
        """Mock Cohere client"""
        with patch('cohere.Client') as mock:
            yield mock

    @pytest.fixture
    def agent(self, mock_weaviate, mock_cohere):
        """Create query agent with mocked dependencies"""
        return NewsletterQueryAgent(
            weaviate_client=mock_weaviate,
            cohere_client=mock_cohere
        )

    def test_agent_uses_quipler_retriever(self, agent):
        """Test agent uses QUIPLER for retrieval"""
        # QUIPLER should be configured
        assert hasattr(agent, 'retriever')
        assert agent.retriever.__class__.__name__ == 'QUIPLER'

    def test_agent_uses_react_for_synthesis(self, agent):
        """Test agent uses ReAct for answer synthesis"""
        assert hasattr(agent, 'synthesize')
        # ReAct module check
        assert 'ReAct' in str(type(agent.synthesize))

    def test_agent_query_returns_answer_and_sources(self, agent):
        """Test query returns structured response"""
        result = agent(question="What are recent APAC regulations?")

        assert hasattr(result, 'answer')
        assert hasattr(result, 'sources')
        assert isinstance(result.sources, list)

    def test_agent_handles_complex_query(self, agent):
        """Test multi-hop reasoning for complex queries"""
        result = agent(
            question="How has GDPR enforcement evolved in the UK post-Brexit and what does this mean for US screening firms?"
        )

        # Should provide substantive answer
        assert len(result.answer) > 100
        # Should cite sources
        assert len(result.sources) > 0
```

---

## Validation-Agent Integration

### Workflow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Worker writes  │────▶│  Orchestrator    │────▶│ Validation-Agent│
│  implementation │     │  spawns test     │     │ runs tests      │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐              │
                        │  Report results  │◀─────────────┘
                        │  RED or GREEN    │
                        └──────────────────┘
```

### Spawning Validation-Agent

```python
# Orchestrator spawns validation-agent for test execution
Task(
    subagent_type="validation-agent",
    prompt="""
    Run TDD validation for task <bd-id>:

    --mode=implementation
    --task_id=<bd-id>
    --tdd_phase=RED|GREEN|REFACTOR

    PHASE INSTRUCTIONS:

    RED PHASE:
    1. Run: pytest tests/test_<module>.py -v
    2. Verify tests FAIL (no implementation yet)
    3. Report: "RED PHASE PASS - X tests failing as expected"
    4. If tests PASS: Report "RED PHASE FAIL - tests should not pass yet"

    GREEN PHASE:
    1. Run: pytest tests/test_<module>.py -v
    2. Verify tests PASS
    3. Report: "GREEN PHASE PASS - X tests passing"
    4. If tests FAIL: Report "GREEN PHASE FAIL - implementation incomplete"

    REFACTOR PHASE:
    1. Run: pytest tests/test_<module>.py -v --cov=src/<module>
    2. Verify tests still PASS
    3. Report coverage percentage
    4. Flag if coverage < 80%
    """
)
```

### TDD Checkpoint Commands

```bash
# RED checkpoint - verify tests fail
pytest tests/test_<module>.py -v --tb=short 2>&1 | tee test_output.txt
# Expected: All tests FAIL

# GREEN checkpoint - verify tests pass
pytest tests/test_<module>.py -v --tb=short 2>&1 | tee test_output.txt
# Expected: All tests PASS

# REFACTOR checkpoint - verify coverage
pytest tests/test_<module>.py -v --cov=src/<module> --cov-report=term-missing
# Expected: >80% coverage, all tests PASS
```

---

## Task-Specific Test Examples

### Task: dspy-fx7 (Infrastructure Setup)

```python
# tests/test_infrastructure.py

import pytest
import docker
import weaviate
import dspy


class TestDockerEnvironment:
    """RED: Test Docker environment before setup"""

    def test_docker_client_available(self):
        """Test Docker is running"""
        client = docker.from_env()
        assert client.ping()

    def test_weaviate_container_running(self):
        """Test Weaviate container exists and is running"""
        client = docker.from_env()
        containers = client.containers.list(filters={'name': 'weaviate'})
        assert len(containers) > 0
        assert containers[0].status == 'running'


class TestWeaviateConnection:
    """Test Weaviate connectivity"""

    def test_weaviate_healthcheck(self):
        """Test Weaviate is accessible"""
        client = weaviate.connect_to_local()
        assert client.is_ready()
        client.close()

    def test_weaviate_has_newsletter_collection(self):
        """Test NewsletterArticles collection exists"""
        client = weaviate.connect_to_local()
        collections = client.collections.list_all()
        assert 'NewsletterArticles' in [c.name for c in collections]
        client.close()


class TestDSPyConfiguration:
    """Test DSPy is properly configured"""

    def test_dspy_lm_configured(self):
        """Test language model is configured"""
        # Should have a configured LM
        assert dspy.settings.lm is not None

    def test_env_variables_loaded(self):
        """Test required environment variables exist"""
        import os
        required_vars = ['OPENAI_API_KEY', 'COHERE_API_KEY']
        for var in required_vars:
            assert os.getenv(var) is not None, f"Missing: {var}"
```

### Task: dspy-4tp (Data Models)

```python
# tests/test_models.py

import pytest
from datetime import datetime
from pydantic import ValidationError


class TestArticleModel:
    """RED: Test Article model before implementation"""

    def test_article_model_exists(self):
        """Test Article model can be imported"""
        from src.models import Article
        assert Article is not None

    def test_article_requires_title(self):
        """Test Article requires title field"""
        from src.models import Article
        with pytest.raises(ValidationError):
            Article(content="Some content")  # Missing title

    def test_article_requires_content(self):
        """Test Article requires content field"""
        from src.models import Article
        with pytest.raises(ValidationError):
            Article(title="Test")  # Missing content

    def test_article_has_metadata_fields(self):
        """Test Article has expected metadata"""
        from src.models import Article
        article = Article(
            title="Test",
            content="Content",
            source_url="https://example.com",
            published_date=datetime.now()
        )
        assert hasattr(article, 'source_url')
        assert hasattr(article, 'published_date')
        assert hasattr(article, 'region')
        assert hasattr(article, 'topics')


class TestProcessingLogModel:
    """Test ProcessingLog for audit trail"""

    def test_processing_log_tracks_stages(self):
        """Test log tracks processing stages"""
        from src.models import ProcessingLog
        log = ProcessingLog(article_id="123")

        assert hasattr(log, 'ingestion_timestamp')
        assert hasattr(log, 'prefilter_result')
        assert hasattr(log, 'classification_result')
        assert hasattr(log, 'storage_timestamp')
```

---

## Test File Structure

```
tests/
├── conftest.py                 # Shared fixtures
├── test_infrastructure.py      # dspy-fx7
├── test_models.py              # dspy-4tp
├── test_ingestion.py           # dspy-7eh
├── test_deduplication.py       # dspy-50z
├── test_prefilter.py           # dspy-2yy
├── test_classification.py      # dspy-0s9
├── test_storage.py             # dspy-13p
├── test_query_agent.py         # dspy-v72
├── test_cli.py                 # dspy-bot
├── test_optimization.py        # dspy-4aa
└── integration/
    ├── test_pipeline_e2e.py    # End-to-end pipeline
    └── test_query_e2e.py       # Query workflow E2E
```

### conftest.py (Shared Fixtures)

```python
# tests/conftest.py

import pytest
import dspy
from unittest.mock import Mock, patch
import weaviate
import os


@pytest.fixture(scope="session")
def mock_openai_key():
    """Ensure OpenAI key is available"""
    os.environ.setdefault('OPENAI_API_KEY', 'test-key-for-mocking')


@pytest.fixture
def mock_dspy_lm():
    """Mock DSPy language model for deterministic tests"""
    with patch.object(dspy.settings, 'lm') as mock_lm:
        mock_lm.return_value = Mock()
        yield mock_lm


@pytest.fixture
def sample_article():
    """Sample article for testing"""
    return {
        'title': 'FCRA Compliance Update 2026',
        'content': '''
        The Fair Credit Reporting Act continues to evolve with new
        amendments proposed for employment background screening. Key changes
        include enhanced consumer notification requirements and stricter
        accuracy standards for consumer reporting agencies.
        ''',
        'source_url': 'https://example.com/fcra-update',
        'published_date': '2026-01-10',
        'source_category': 'legal'
    }


@pytest.fixture
def irrelevant_article():
    """Irrelevant article for negative testing"""
    return {
        'title': 'Best Pizza Recipes',
        'content': 'Today we explore the art of making perfect pizza dough...',
        'source_url': 'https://example.com/pizza',
        'published_date': '2026-01-10',
        'source_category': 'lifestyle'
    }


@pytest.fixture
def weaviate_test_client():
    """Weaviate client for integration tests"""
    client = weaviate.connect_to_local()
    yield client
    client.close()
```

---

## Red-Green-Refactor Workflow per Task

### Example: Task dspy-2yy (Tiny LM Pre-Filter)

#### Step 1: RED Phase

```bash
# 1. Write failing tests first
# File: tests/test_prefilter.py (as shown above)

# 2. Verify tests fail
pytest tests/test_prefilter.py -v
# Expected output: FAILED (module 'src.prefilter' not found)

# 3. Validation-agent confirms RED
# Report: "RED PHASE PASS - 8 tests failing (ImportError: No module named 'src.prefilter')"
```

#### Step 2: GREEN Phase

```bash
# 1. Implement minimal code to pass tests
# File: src/prefilter.py

# 2. Run tests
pytest tests/test_prefilter.py -v
# Expected: All tests PASS

# 3. Validation-agent confirms GREEN
# Report: "GREEN PHASE PASS - 8 tests passing"
```

#### Step 3: REFACTOR Phase

```bash
# 1. Improve code quality, add docstrings, optimize
# 2. Verify tests still pass with coverage
pytest tests/test_prefilter.py -v --cov=src/prefilter --cov-report=term-missing

# 3. Validation-agent confirms REFACTOR
# Report: "REFACTOR PASS - 8 tests passing, 92% coverage"
```

---

## Validation-Agent Modes

| Mode | Phase | Success Criteria |
|------|-------|------------------|
| `--tdd_phase=RED` | Before implementation | Tests FAIL |
| `--tdd_phase=GREEN` | After implementation | Tests PASS |
| `--tdd_phase=REFACTOR` | After cleanup | Tests PASS + ≥80% coverage |

### Independent Evaluation Benefits

1. **Unbiased**: Validation-agent runs tests without implementation context
2. **Reproducible**: Same commands, consistent environment
3. **Documented**: Test output captured as evidence
4. **Quality Gate**: Blocks task closure until all phases pass

---

## Test Execution Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_prefilter.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only fast unit tests (no integration)
pytest tests/ -v -m "not integration"

# Run integration tests only
pytest tests/integration/ -v -m "integration"

# Generate JUnit XML for CI
pytest tests/ --junitxml=test-results.xml
```

---

## Acceptance Test (AT) Epic Structure

To follow the orchestrator pattern, we need a paired AT epic:

```
UBER-EPIC: dspy-2k6 (DSPy Newsletter Research Tool PoC)
│
├── [Existing Functional Tasks]
│   ├── dspy-fx7: Infrastructure Setup
│   ├── dspy-4tp: Data Models
│   └── ... (8 more tasks)
│
└── AT-EPIC: AT-DSPy Newsletter Tool (NEW)
    ├── AT-TASK: Unit tests for all modules
    ├── AT-TASK: Integration tests for pipeline
    └── AT-TASK: E2E query validation
```

---

## Summary

This TDD strategy ensures:

1. **Tests First**: Every feature starts with failing tests
2. **Incremental Progress**: Red → Green → Refactor cycle
3. **Independent Validation**: validation-agent executes tests without implementation bias
4. **Coverage Tracking**: Minimum 80% coverage required
5. **Traceability**: Each test file maps to a Beads task

**Next Steps**:
1. Create AT epic in Beads
2. Begin dspy-fx7 with test_infrastructure.py (RED phase)
3. Iterate through tasks using TDD workflow
