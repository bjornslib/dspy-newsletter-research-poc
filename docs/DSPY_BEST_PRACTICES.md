# DSPy Best Practices Guide

> **Source**: Compiled from DSPy official documentation (dspy.ai), Context7 research, and Stanford NLP resources.
> **Last Updated**: 2026-01-12

## Table of Contents

1. [Core Philosophy](#core-philosophy)
2. [Signature Best Practices](#signature-best-practices)
3. [Module Selection Guide](#module-selection-guide)
4. [Configuration & Setup](#configuration--setup)
5. [Evaluation & Metrics](#evaluation--metrics)
6. [Production Deployment](#production-deployment)
7. [Common Pitfalls to Avoid](#common-pitfalls-to-avoid)

---

## Core Philosophy

DSPy's core principle: **Program, don't prompt.** Instead of manually crafting prompts, you define:

1. **Signatures** - What the task should do (inputs → outputs)
2. **Modules** - How to accomplish it (reasoning patterns)
3. **Metrics** - What success looks like
4. **Optimizers** - Automatic improvement via data

### The DSPy Workflow

```
Define Signature → Choose Module → Create Training Data → Define Metric → Optimize → Deploy
```

### Key Insight

> "Instead of manually tuning prompts, define a metric and let DSPy optimize."
> — DSPy Documentation

---

## Signature Best Practices

### 1. Use Descriptive Class-Based Signatures

```python
# ❌ Bad: Vague inline signature
qa = dspy.Predict("input -> output")

# ✅ Good: Descriptive class signature
class SummarizeArticle(dspy.Signature):
    """Summarize news articles into 3-5 key points."""
    article = dspy.InputField(desc="full article text")
    summary = dspy.OutputField(desc="bullet points, 3-5 items")
```

### 2. Leverage Field Descriptions

Field descriptions guide the model on expectations:

```python
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often between 1 and 5 words")
```

### 3. Use Docstrings to Clarify Task Nature

The signature docstring becomes part of the prompt:

```python
class CraftRedactedRequest(dspy.Signature):
    """Given a private user query, create a privacy-preserving request
    for a powerful external LLM. The LLM may assist without learning
    private information about the user."""

    user_query = dspy.InputField()
    llm_request = dspy.OutputField()
```

### 4. Structured Outputs with Pydantic

For complex outputs, use Pydantic models:

```python
from pydantic import BaseModel, Field

class PersonInfo(BaseModel):
    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")
    occupation: str = Field(description="Current job")

class ExtractPerson(dspy.Signature):
    """Extract person information from text."""
    text = dspy.InputField()
    person: PersonInfo = dspy.OutputField()

extractor = dspy.Predict(ExtractPerson)
```

---

## Module Selection Guide

| Task Type | Recommended Module | Reason |
|-----------|-------------------|--------|
| Simple classification | `dspy.Predict` | Fast, direct |
| Math word problems | `dspy.ProgramOfThought` | Reliable calculations |
| Logical reasoning | `dspy.ChainOfThought` | Better with steps |
| Multi-step research | `dspy.ReAct` | Tool usage |
| High-stakes decisions | `dspy.MultiChainComparison` | Self-consistency |
| Structured extraction | `dspy.TypedPredictor` | Type safety |
| Ambiguous questions | `dspy.MultiChainComparison` | Multiple perspectives |

### Module Performance Characteristics

| Module | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| `Predict` | ⚡ Fast | Good | Direct predictions |
| `ChainOfThought` | ~2x slower | Better on reasoning | Complex logic |
| `ProgramOfThought` | Medium | Excellent for math | Calculations |
| `ReAct` | Slowest | Best for research | Tool-using agents |

### Start Simple, Iterate

```python
# Step 1: Start with Predict
qa = dspy.Predict("question -> answer")

# Step 2: Add reasoning if needed
qa = dspy.ChainOfThought("question -> answer")

# Step 3: Add optimization when you have data
optimized_qa = optimizer.compile(qa, trainset=data)
```

---

## Configuration & Setup

### Language Model Configuration

```python
import dspy

# Basic setup
lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_API_KEY')
dspy.configure(lm=lm)

# With generation parameters
lm = dspy.LM(
    'openai/gpt-4o-mini',
    temperature=0.7,
    max_tokens=1000,
    cache=True  # Default: enabled
)
```

### Using Multiple LMs

```python
# Different models for different tasks
cheap_lm = dspy.LM('openai/gpt-4o-mini')
strong_lm = dspy.LM('openai/gpt-4o')

# Context manager for temporary LM switch
with dspy.context(lm=cheap_lm):
    context = retriever(question)

with dspy.context(lm=strong_lm):
    answer = generator(context=context, question=question)
```

### Caching Configuration

```python
# Disable caching (for debugging/fresh results)
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

# Configure cache limits
dspy.configure_cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
    disk_size_limit_bytes=1_000_000_000,  # 1GB
    memory_max_entries=10000,
)

# Per-LM caching
lm = dspy.LM('openai/gpt-4o-mini', cache=False)
```

### Force Fresh Results with rollout_id

```python
# Bypass cache with unique rollout_id
lm("Say this is a test!", rollout_id=1, temperature=1.0)

# In Predict modules
predict = dspy.Predict("question -> answer")
predict(question="What is 1 + 52?", config={"rollout_id": 5, "temperature": 1.0})
```

---

## Evaluation & Metrics

### Basic Metric Pattern

```python
def metric(example, pred, trace=None):
    """
    Args:
        example: Ground truth from dataset
        pred: Model prediction
        trace: Optional trace for debugging/bootstrapping

    Returns:
        - Float (0.0-1.0) for evaluation/optimization
        - Boolean for bootstrapping demonstrations
    """
    return example.answer == pred.answer
```

### Metric with Trace for Bootstrapping

```python
def validate_context_and_answer(example, pred, trace=None):
    answer_match = example.answer.lower() == pred.answer.lower()
    context_match = any((pred.answer.lower() in c) for c in pred.context)

    if trace is None:  # Evaluation or optimization
        return (answer_match + context_match) / 2.0
    else:  # Bootstrapping demonstrations
        return answer_match and context_match
```

### LLM-as-Judge Metric

```python
class Assess(dspy.Signature):
    """Assess the quality of a text."""
    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer: bool = dspy.OutputField()

def metric(gold, pred, trace=None):
    question, answer, tweet = gold.question, gold.answer, pred.output

    engaging = "Does the assessed text make for a self-contained, engaging tweet?"
    correct = f"The text should answer `{question}` with `{answer}`. Does it?"

    correct_score = dspy.Predict(Assess)(assessed_text=tweet, assessment_question=correct)
    engaging_score = dspy.Predict(Assess)(assessed_text=tweet, assessment_question=engaging)

    score = (correct_score.assessment_answer + engaging_score.assessment_answer)

    if trace is not None:
        return score >= 2
    return score / 2.0
```

### Evaluation Setup

```python
from dspy import Evaluate

# Create evaluator
evaluate = dspy.Evaluate(
    devset=test_set,
    metric=your_metric,
    num_threads=16,
    display_progress=True,
    display_table=5  # Show top 5 results
)

# Run evaluation
score = evaluate(your_program)
print(f"Accuracy: {score}")
```

---

## Production Deployment

### Save and Load Optimized Programs

```python
# Save after optimization
optimized_program.save("./models/v1.json")

# Load for production
loaded_program = YourProgramClass()
loaded_program.load(path="./models/v1.json")
```

### Error Handling

```python
class RobustModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.process = dspy.ChainOfThought("input -> output")

    def forward(self, input):
        try:
            result = self.process(input=input)
            return result
        except Exception as e:
            print(f"Error processing {input}: {e}")
            return dspy.Prediction(output="Error: could not process input")
```

### Caching for Performance

```python
from functools import lru_cache

class CachedRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    @lru_cache(maxsize=1000)
    def forward(self, question):
        passages = self.retrieve(question).passages
        context = "\n".join(passages)
        return self.generate(context=context, question=question).answer
```

### Monitoring and Stats

```python
class MonitoredModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.process = dspy.ChainOfThought("input -> output")
        self.call_count = 0
        self.errors = 0

    def forward(self, input):
        self.call_count += 1
        try:
            return self.process(input=input)
        except Exception as e:
            self.errors += 1
            raise

    def get_stats(self):
        return {
            "calls": self.call_count,
            "errors": self.errors,
            "error_rate": self.errors / max(self.call_count, 1)
        }
```

---

## Common Pitfalls to Avoid

### 1. Overfitting to Training Data

```python
# ❌ Bad: Too many demonstrations
optimizer = BootstrapFewShot(max_bootstrapped_demos=20)  # Overfits!

# ✅ Good: Moderate demonstrations
optimizer = BootstrapFewShot(max_bootstrapped_demos=3)  # Typically 3-5
```

### 2. Metric Doesn't Match Task

```python
# ❌ Bad: Binary metric for nuanced task
def bad_metric(example, pred, trace=None):
    return example.answer == pred.answer  # Too strict!

# ✅ Good: Graded metric with partial credit
def good_metric(example, pred, trace=None):
    return f1_score(example.answer, pred.answer)
```

### 3. Insufficient Training Data

```python
# ❌ Bad: Too few examples
trainset = data[:5]  # Not enough!

# ✅ Good: Sufficient data
trainset = data[:50]  # Minimum 10-50 for BootstrapFewShot
```

### 4. No Proper Data Splits

```python
# ❌ Bad: Optimizing on test set
optimizer.compile(module, trainset=testset)  # Cheating!

# ✅ Good: Proper train/val/test splits
trainset = data[:100]   # 70%
valset = data[100:120]  # 15%
testset = data[120:]    # 15%

optimizer.compile(module, trainset=trainset, valset=valset)
evaluate(optimized, devset=testset)  # Final evaluation
```

### 5. Using Wrong Module for Task

```python
# ❌ Bad: ChainOfThought for simple extraction
cot = dspy.ChainOfThought("name -> greeting")  # Overkill

# ✅ Good: Predict for simple tasks
predict = dspy.Predict("name -> greeting")  # Faster, sufficient
```

### 6. Ignoring Trace in Metrics

```python
# ❌ Bad: Same behavior for eval and bootstrap
def metric(example, pred, trace=None):
    return example.answer in pred.answer

# ✅ Good: Different behavior based on context
def metric(example, pred, trace=None):
    score = f1_score(example.answer, pred.answer)
    if trace is not None:  # Bootstrapping
        return score >= 0.9  # Only accept high-quality demos
    return score  # Return float for evaluation
```

---

## Quick Reference

### Essential Imports

```python
import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2, COPRO
from dspy import Evaluate
from pydantic import BaseModel, Field
```

### Minimal Working Example

```python
import dspy

# Configure
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)

# Define signature
class QA(dspy.Signature):
    """Answer questions concisely."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="1-5 words")

# Create module
qa = dspy.ChainOfThought(QA)

# Use
result = qa(question="What is the capital of France?")
print(result.answer)  # "Paris"

# Optimize (with data)
from dspy.teleprompt import BootstrapFewShot

metric = lambda ex, pred, trace=None: ex.answer.lower() in pred.answer.lower()
optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=3)
optimized_qa = optimizer.compile(qa, trainset=trainset)

# Save
optimized_qa.save("qa_v1.json")
```

---

## Resources

- **Official Documentation**: https://dspy.ai
- **GitHub**: https://github.com/stanfordnlp/dspy (22k+ stars)
- **Discord**: https://discord.gg/XCGy2WDCQB
- **Paper**: "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"

---

*See also:*
- `DSPY_OPTIMIZERS_GUIDE.md` - Deep dive into optimization algorithms
- `DSPY_PATTERNS_COOKBOOK.md` - Real-world patterns and recipes
