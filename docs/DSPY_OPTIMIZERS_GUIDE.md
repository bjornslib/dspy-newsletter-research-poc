# DSPy Optimizers (Teleprompters) Guide

> **Source**: Compiled from DSPy official documentation (dspy.ai) and Context7 research.
> **Last Updated**: 2026-01-12

## Table of Contents

1. [What Are Optimizers?](#what-are-optimizers)
2. [Optimizer Selection Matrix](#optimizer-selection-matrix)
3. [BootstrapFewShot](#bootstrapfewshot)
4. [MIPROv2](#miprov2)
5. [COPRO](#copro)
6. [GEPA](#gepa)
7. [BootstrapFinetune](#bootstrapfinetune)
8. [KNNFewShot](#knnfewshot)
9. [Writing Effective Metrics](#writing-effective-metrics)
10. [Optimization Workflow](#optimization-workflow)
11. [Advanced Patterns](#advanced-patterns)

---

## What Are Optimizers?

DSPy optimizers (called "teleprompters") automatically improve your modules by:

- **Synthesizing few-shot examples** from training data
- **Proposing better instructions** through search
- **Fine-tuning model weights** (optional)

### Key Insight

> "Instead of manually tuning prompts, define a metric and let DSPy optimize."

### How Optimization Works

1. Takes your training examples
2. Uses your module to generate predictions
3. Evaluates predictions against your metric
4. Selects high-quality examples as demonstrations
5. Iteratively improves instructions (for advanced optimizers)

---

## Optimizer Selection Matrix

| Optimizer | Best For | Speed | Quality | Data Needed | Complexity |
|-----------|----------|-------|---------|-------------|------------|
| **BootstrapFewShot** | General purpose | ⚡ Fast | Good | 10-50 examples | Low |
| **MIPROv2** | State-of-the-art | Medium | Excellent | 50-200 examples | Medium |
| **COPRO** | Prompt optimization | Medium | Good | 20-100 examples | Medium |
| **GEPA** | Iterative refinement | Slow | Excellent | 50-200 examples | Medium |
| **BootstrapFinetune** | Fine-tuning | Slow | Excellent | 100+ examples | High |
| **KNNFewShot** | Quick baseline | ⚡⚡ Very fast | Fair | 10+ examples | Very Low |

### Decision Tree

```
Do you have 100+ examples?
├── Yes → Consider BootstrapFinetune (if latency critical)
│         Otherwise → MIPROv2 or GEPA
│
└── No → Do you have 50+ examples?
         ├── Yes → MIPROv2 (best quality)
         │         COPRO (faster alternative)
         │
         └── No → BootstrapFewShot (10-50 examples)
                  KNNFewShot (quick baseline)
```

---

## BootstrapFewShot

**The most popular optimizer** - Generates few-shot demonstrations from training data.

### How It Works

1. Takes your training examples
2. Uses your module to generate predictions
3. Selects high-quality predictions (based on metric)
4. Uses these as few-shot examples in future prompts

### Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `metric` | Function that scores predictions | Required | - |
| `max_bootstrapped_demos` | Max demonstrations to generate | 4 | 3-5 |
| `max_labeled_demos` | Max labeled examples to use | 16 | 8-16 |
| `max_rounds` | Optimization iterations | 1 | 1-3 |
| `max_errors` | Max errors before stopping | 10 | 5-10 |
| `teacher_settings` | LM settings for teacher | None | Use stronger LM |

### Basic Usage

```python
from dspy.teleprompt import BootstrapFewShot

# Define metric
def validate_answer(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# Training data
trainset = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="What is 3+5?", answer="8").with_inputs("question"),
    # ... more examples (10-50 total)
]

# Create optimizer
optimizer = BootstrapFewShot(
    metric=validate_answer,
    max_bootstrapped_demos=3,
    max_rounds=2
)

# Optimize
optimized_qa = optimizer.compile(qa, trainset=trainset)
```

### With Teacher Model

Use a stronger model to generate better demonstrations:

```python
gpt4 = dspy.LM('openai/gpt-4')

optimizer = BootstrapFewShot(
    metric=validate_answer,
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
    teacher_settings=dict(lm=gpt4)
)

optimized = optimizer.compile(student=qa, trainset=trainset)
```

### Best Practices

- ✅ Start with 10-50 training examples
- ✅ Use diverse examples covering edge cases
- ✅ Set `max_bootstrapped_demos=3-5` for most tasks
- ✅ Increase `max_rounds=2-3` for better quality
- ✅ Use a stronger teacher model when possible
- ❌ Don't use more than 10 bootstrapped demos (overfitting risk)

---

## MIPROv2

**State-of-the-art optimizer** - Iteratively searches for better instructions.

### How It Works

1. Generates candidate instructions
2. Tests each on validation set
3. Selects best-performing instructions
4. Iterates to refine further

### Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `metric` | Evaluation metric | Required | - |
| `auto` | Optimization level | "medium" | "light", "medium", "heavy" |
| `num_candidates` | Instructions per iteration | 10 | 10-20 |
| `teacher_settings` | Settings for teacher model | None | Use GPT-4 |
| `prompt_model` | Model for prompt generation | None | Use fast model |

### Optimization Levels

| Level | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| `"light"` | Fast | Good | Quick iteration |
| `"medium"` | Medium | Better | Most use cases |
| `"heavy"` | Slow | Best | Production deployment |

### Basic Usage

```python
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(
    metric=your_metric,
    auto="medium"
)

optimized = optimizer.compile(
    dspy.ChainOfThought("question -> answer"),
    trainset=trainset
)

# Save for production
optimized.save("optimized.json")
```

### Advanced Configuration

```python
gpt4o = dspy.LM('openai/gpt-4o')
gpt4o_mini = dspy.LM('openai/gpt-4o-mini')

optimizer = MIPROv2(
    metric=your_metric,
    auto="medium",
    num_threads=16,
    teacher_settings=dict(lm=gpt4o),
    prompt_model=gpt4o_mini
)

optimized = optimizer.compile(
    module,
    trainset=trainset,
    max_bootstrapped_demos=4,
    max_labeled_demos=4
)
```

### Best Practices

- ✅ Use 50-200 training examples
- ✅ Separate validation set (20-50 examples)
- ✅ Use `auto="medium"` for most cases
- ✅ Use stronger teacher model (GPT-4) for better quality
- ✅ Takes 10-30 minutes typically
- ❌ Don't use with fewer than 50 examples

---

## COPRO

**Coordinate Prompt Optimization** - Gradient-free prompt search.

### How It Works

1. Generates prompt variants
2. Evaluates each variant on training data
3. Selects best prompts
4. Iterates to refine

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `prompt_model` | Model for generating prompts | Required |
| `metric` | Evaluation metric | Required |
| `breadth` | Candidates per iteration | 10 |
| `depth` | Optimization rounds | 3 |
| `init_temperature` | Sampling temperature | 1.0 |

### Usage

```python
from dspy.teleprompt import COPRO

eval_kwargs = dict(num_threads=16, display_progress=True, display_table=0)

optimizer = COPRO(
    prompt_model=dspy.LM('openai/gpt-4o'),
    metric=your_metric,
    breadth=10,  # Candidates per iteration
    depth=3,     # Optimization rounds
    init_temperature=1.0
)

optimized = optimizer.compile(
    your_program,
    trainset=trainset,
    eval_kwargs=eval_kwargs
)
```

### When to Use

- ✅ Want prompt optimization without heavy compute
- ✅ Have 20-100 examples
- ✅ MIPROv2 is too slow
- ❌ Don't use for complex multi-stage pipelines

---

## GEPA

**Guided Example Program Adaptation** - Iterative refinement with detailed feedback.

### How It Works

1. Trains across multiple iterations
2. Evaluates classification/task accuracy
3. Progressively refines the module
4. Shows performance metrics per iteration

### Usage

```python
from dspy.teleprompt.gepa import GEPA

optimizer = GEPA(
    metric=accuracy_metric,
    max_iterations=27
)

optimized = optimizer.compile(
    student=YourModule(),
    trainset=training_examples,
    valset=validation_examples
)
```

### Example Output

```
Iteration 25: score 0.7207, accuracy 77.8%
Iteration 26: score improved, accuracy 88.9%
Iteration 27: score 0.8328, accuracy 96.7%
Average Metric: 2.9 / 3 (96.7%)
```

### When to Use

- ✅ Classification tasks
- ✅ Want detailed iteration feedback
- ✅ Have enough compute budget
- ❌ Slow for simple tasks

---

## BootstrapFinetune

**Fine-tune model weights** - Creates training dataset for fine-tuning.

### How It Works

1. Generates synthetic training data using bootstrapping
2. Exports data in fine-tuning format
3. You fine-tune model with your LM provider
4. Load fine-tuned model back into DSPy

### Usage

```python
from dspy.teleprompt import BootstrapFinetune

optimizer = BootstrapFinetune(metric=validate_answer)
optimized = optimizer.compile(qa, trainset=trainset)

# Exports training data to file
# Fine-tune using your LM provider's API

# After fine-tuning, load your model:
finetuned_lm = dspy.LM("openai/ft:gpt-3.5-turbo:your-model-id")
dspy.configure(lm=finetuned_lm)
```

### When to Use

- ✅ Have 100+ training examples
- ✅ Latency is critical (fine-tuned models are faster)
- ✅ Task is narrow and well-defined
- ✅ Prompt optimization isn't sufficient
- ❌ Don't use for rapidly changing tasks
- ❌ Don't use if you need to switch models frequently

---

## KNNFewShot

**K-Nearest Neighbors** - Selects similar examples per query.

### How It Works

1. Embeds all training examples
2. For each query, finds k most similar examples
3. Uses these as few-shot demonstrations

### Usage

```python
from dspy.teleprompt import KNNFewShot

optimizer = KNNFewShot(k=3)
optimized = optimizer.compile(qa, trainset=trainset)
```

### When to Use

- ✅ Quick baseline
- ✅ Diverse training examples available
- ✅ Similarity is good proxy for helpfulness
- ❌ Doesn't optimize instructions

---

## Writing Effective Metrics

### Binary Metrics

```python
def exact_match(example, pred, trace=None):
    """Return True if prediction exactly matches gold."""
    return example.answer == pred.answer

def contains_answer(example, pred, trace=None):
    """Return True if prediction contains gold answer."""
    return example.answer.lower() in pred.answer.lower()
```

### Continuous Metrics (Better for Optimization)

```python
def f1_score(example, pred, trace=None):
    """F1 score between prediction and gold."""
    pred_tokens = set(pred.answer.lower().split())
    gold_tokens = set(example.answer.lower().split())

    if not pred_tokens or not gold_tokens:
        return 0.0

    precision = len(pred_tokens & gold_tokens) / len(pred_tokens)
    recall = len(pred_tokens & gold_tokens) / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)
```

### Metrics with Trace (For Bootstrapping)

```python
def metric_with_trace(example, pred, trace=None):
    """Different behavior for eval vs bootstrapping."""
    score = f1_score(example.answer, pred.answer)

    if trace is not None:  # Bootstrapping: accept only high-quality demos
        return score >= 0.9
    return score  # Evaluation: return continuous score
```

### Multi-Factor Metrics

```python
def comprehensive_metric(example, pred, trace=None):
    """Combine multiple quality factors."""
    score = 0.0

    # Correctness (50%)
    if example.answer.lower() in pred.answer.lower():
        score += 0.5

    # Conciseness (25%)
    if len(pred.answer.split()) <= 20:
        score += 0.25

    # Has citation (25%)
    if "source:" in pred.answer.lower():
        score += 0.25

    if trace is not None:
        return score >= 0.75
    return score
```

---

## Optimization Workflow

### 1. Start with Baseline

```python
# No optimization
baseline = dspy.ChainOfThought("question -> answer")
baseline_score = evaluate(baseline, devset=testset)
print(f"Baseline: {baseline_score:.2%}")
```

### 2. Try BootstrapFewShot

```python
fewshot = BootstrapFewShot(metric=metric, max_bootstrapped_demos=3)
optimized = fewshot.compile(baseline, trainset=trainset)
fewshot_score = evaluate(optimized, devset=testset)
print(f"Few-shot: {fewshot_score:.2%} (+{fewshot_score - baseline_score:.2%})")
```

### 3. Try MIPROv2 If More Data Available

```python
mipro = MIPROv2(metric=metric, auto="medium")
optimized_mipro = mipro.compile(baseline, trainset=trainset)
mipro_score = evaluate(optimized_mipro, devset=testset)
print(f"MIPROv2: {mipro_score:.2%} (+{mipro_score - baseline_score:.2%})")
```

### 4. Save Best Model

```python
if mipro_score > fewshot_score:
    optimized_mipro.save("models/best_model.json")
    print("Saved MIPROv2 model")
else:
    optimized.save("models/best_model.json")
    print("Saved BootstrapFewShot model")
```

### 5. Production Loading

```python
# Load optimized model
production_model = YourModuleClass()
production_model.load("models/best_model.json")
```

---

## Advanced Patterns

### Multi-Stage Optimization

```python
# Stage 1: Bootstrap few-shot
stage1 = BootstrapFewShot(metric=metric, max_bootstrapped_demos=3)
optimized1 = stage1.compile(module, trainset=trainset)

# Stage 2: Instruction tuning
stage2 = MIPROv2(metric=metric, auto="light")
optimized2 = stage2.compile(optimized1, trainset=trainset)

# Final optimized module
final_module = optimized2
```

### Ensemble Optimization

```python
class EnsembleModule(dspy.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules = modules

    def forward(self, question):
        predictions = [m(question=question).answer for m in self.modules]
        # Majority vote
        return dspy.Prediction(answer=max(set(predictions), key=predictions.count))

# Optimize with different optimizers
opt1 = BootstrapFewShot(metric=metric).compile(module, trainset=trainset)
opt2 = MIPROv2(metric=metric, auto="light").compile(module, trainset=trainset)
opt3 = COPRO(prompt_model=lm, metric=metric).compile(module, trainset=trainset)

# Ensemble
ensemble = EnsembleModule([opt1, opt2, opt3])
```

### Cross-Validation

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5)
scores = []

for train_idx, val_idx in kfold.split(data):
    trainset = [data[i] for i in train_idx]
    valset = [data[i] for i in val_idx]

    optimized = optimizer.compile(module, trainset=trainset)
    score = evaluate(optimized, devset=valset)
    scores.append(score)

print(f"Average score: {sum(scores) / len(scores):.2%}")
```

---

## Quick Reference

### Optimizer Imports

```python
from dspy.teleprompt import (
    BootstrapFewShot,
    MIPROv2,
    COPRO,
    BootstrapFinetune,
    KNNFewShot,
)
from dspy.teleprompt.gepa import GEPA
```

### Minimal Optimization Example

```python
import dspy
from dspy.teleprompt import BootstrapFewShot

# Configure
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

# Data
trainset = [
    dspy.Example(question="...", answer="...").with_inputs("question")
    for q, a in your_data
]

# Metric
metric = lambda ex, pred, trace=None: ex.answer.lower() in pred.answer.lower()

# Module
qa = dspy.ChainOfThought("question -> answer")

# Optimize
optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=3)
optimized = optimizer.compile(qa, trainset=trainset)

# Save
optimized.save("qa_optimized.json")
```

---

*See also:*
- `DSPY_BEST_PRACTICES.md` - Core concepts and configuration
- `DSPY_PATTERNS_COOKBOOK.md` - Real-world patterns and recipes
