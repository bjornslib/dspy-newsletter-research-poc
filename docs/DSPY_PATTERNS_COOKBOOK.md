# DSPy Patterns Cookbook

> **Source**: Compiled from DSPy official documentation (dspy.ai), Context7 research, and real-world examples.
> **Last Updated**: 2026-01-12

## Table of Contents

1. [RAG Patterns](#rag-patterns)
2. [Agent Patterns](#agent-patterns)
3. [Classification Patterns](#classification-patterns)
4. [Data Processing Patterns](#data-processing-patterns)
5. [Multi-Stage Pipeline Patterns](#multi-stage-pipeline-patterns)
6. [Quality Control Patterns](#quality-control-patterns)
7. [Production Patterns](#production-patterns)

---

## RAG Patterns

### Basic RAG

The foundational retrieval-augmented generation pattern.

```python
import dspy

class BasicRAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        passages = self.retrieve(question).passages
        context = "\n\n".join(passages)
        return self.generate(context=context, question=question)

# Configure retriever
from dspy.retrieve.chromadb_rm import ChromadbRM

retriever = ChromadbRM(
    collection_name="my_docs",
    persist_directory="./chroma_db",
    k=3
)
dspy.configure(rm=retriever)

# Use
rag = BasicRAG()
result = rag(question="What is DSPy?")
```

### RAG with ColBERTv2

```python
def search(query: str) -> list[str]:
    """Retrieves abstracts from Wikipedia."""
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

rag = dspy.ChainOfThought('context, question -> response')

question = "What's the name of the castle that David Gregory inherited?"
result = rag(context=search(question), question=question)
```

### Multi-Hop RAG

For questions requiring multiple retrieval steps.

```python
class MultiHopRAG(dspy.Module):
    """RAG that follows chains of reasoning across documents."""

    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.generate_query = dspy.ChainOfThought("question -> search_query")
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        # First retrieval
        query1 = self.generate_query(question=question).search_query
        passages1 = self.retrieve(query1).passages

        # Generate follow-up query based on first results
        context1 = "\n".join(passages1)
        query2 = self.generate_query(
            question=f"Based on: {context1}\nFollow-up: {question}"
        ).search_query

        # Second retrieval
        passages2 = self.retrieve(query2).passages

        # Combine all context
        all_context = "\n\n".join(passages1 + passages2)

        # Generate final answer
        return self.generate_answer(context=all_context, question=question)
```

### RAG with Reranking

Retrieve more, rerank, use best.

```python
class RerankedRAG(dspy.Module):
    """RAG with learned reranking of retrieved passages."""

    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=10)  # Get more candidates
        self.rerank = dspy.Predict("question, passage -> relevance_score: float")
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        # Retrieve candidates
        passages = self.retrieve(question).passages

        # Rerank passages
        scored_passages = []
        for passage in passages:
            score = float(self.rerank(
                question=question,
                passage=passage
            ).relevance_score)
            scored_passages.append((score, passage))

        # Take top 3 after reranking
        top_passages = [p for _, p in sorted(scored_passages, reverse=True)[:3]]
        context = "\n\n".join(top_passages)

        # Generate answer from reranked context
        return self.answer(context=context, question=question)
```

### Multi-Stage RAG with Query Generation

```python
class QueryGenerator(dspy.Signature):
    """Generate a query based on question to fetch relevant context"""
    question: str = dspy.InputField()
    query: str = dspy.OutputField()

def search_wikipedia(query: str) -> list[str]:
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=1)
    return [x["text"] for x in results]

class RAG(dspy.Module):
    def __init__(self):
        self.query_generator = dspy.Predict(QueryGenerator)
        self.answer_generator = dspy.ChainOfThought("question, context -> answer")

    def forward(self, question, **kwargs):
        query = self.query_generator(question=question).query
        context = search_wikipedia(query)[0]
        return self.answer_generator(question=question, context=context).answer
```

---

## Agent Patterns

### Basic ReAct Agent

```python
from dspy import ReAct

# Define tools
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 75°F"

def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': [relevant information...]"

# Create agent
react_agent = dspy.ReAct(
    signature="question -> answer",
    tools=[get_weather, search_web],
    max_iters=5
)

# Use
result = react_agent(question="What's the weather like in Tokyo?")
print(result.answer)
print("Tool calls made:", result.trajectory)
```

### ReAct with ColBERTv2 Retrieval

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
colbert = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
dspy.configure(lm=lm)

def retrieve(query: str):
    """Retrieve top 3 relevant information from ColBert"""
    results = colbert(query, k=3)
    return [x["text"] for x in results]

agent = dspy.ReAct("question -> answer", tools=[retrieve], max_iters=3)
result = agent(question="When was Python first released?")
```

### Multi-Agent System with Router

```python
class MultiAgentSystem(dspy.Module):
    """System with specialized agents for different tasks."""

    def __init__(self):
        super().__init__()

        # Router agent
        self.router = dspy.Predict("question -> agent_type: str")

        # Specialized agents
        self.research_agent = dspy.ReAct(
            "question -> answer",
            tools=[search_wikipedia, search_web]
        )
        self.math_agent = dspy.ProgramOfThought("problem -> answer")
        self.reasoning_agent = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        # Route to appropriate agent
        agent_type = self.router(question=question).agent_type

        if agent_type == "research":
            return self.research_agent(question=question)
        elif agent_type == "math":
            return self.math_agent(problem=question)
        else:
            return self.reasoning_agent(question=question)
```

### Agent with Custom Signature

```python
class ResearchAgent(dspy.Signature):
    """Answer questions using available tools."""
    question = dspy.InputField()
    answer = dspy.OutputField()

def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information."""
    import wikipedia
    try:
        return wikipedia.summary(query, sentences=3)
    except:
        return "No results found"

def calculate(expression: str) -> str:
    """Evaluate mathematical expression safely."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except:
        return "Invalid expression"

agent = dspy.ReAct(ResearchAgent, tools=[search_wikipedia, calculate])
result = agent(question="How old was Einstein when he published special relativity?")
```

---

## Classification Patterns

### Binary Classification

```python
class SentimentClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict("text -> sentiment: str")

    def forward(self, text):
        return self.classify(text=text)

# Training and optimization
trainset = [
    dspy.Example(text="I love this!", sentiment="positive").with_inputs("text"),
    dspy.Example(text="Terrible experience", sentiment="negative").with_inputs("text"),
]

def accuracy(example, pred, trace=None):
    return example.sentiment == pred.sentiment

optimizer = BootstrapFewShot(metric=accuracy, max_bootstrapped_demos=5)
classifier = SentimentClassifier()
optimized_classifier = optimizer.compile(classifier, trainset=trainset)
```

### Multi-Class Classification with Confidence

```python
class TopicSignature(dspy.Signature):
    """Classify text into one of: technology, sports, politics, entertainment."""
    text = dspy.InputField()
    category = dspy.OutputField(desc="one of: technology, sports, politics, entertainment")
    confidence = dspy.OutputField(desc="0.0 to 1.0")

classifier = dspy.ChainOfThought(TopicSignature)
result = classifier(text="The Lakers won the championship")
print(result.category)    # "sports"
print(result.confidence)  # 0.95
```

### Hierarchical Classification

```python
class HierarchicalClassifier(dspy.Module):
    """Two-stage classification: coarse then fine-grained."""

    def __init__(self):
        super().__init__()
        self.coarse = dspy.Predict("text -> broad_category: str")
        self.fine_tech = dspy.Predict("text -> tech_subcategory: str")
        self.fine_sports = dspy.Predict("text -> sports_subcategory: str")

    def forward(self, text):
        # Stage 1: Broad category
        broad = self.coarse(text=text).broad_category

        # Stage 2: Fine-grained based on broad
        if broad == "technology":
            fine = self.fine_tech(text=text).tech_subcategory
        elif broad == "sports":
            fine = self.fine_sports(text=text).sports_subcategory
        else:
            fine = "other"

        return dspy.Prediction(broad_category=broad, fine_category=fine)
```

---

## Data Processing Patterns

### Information Extraction with Pydantic

```python
from pydantic import BaseModel, Field

class PersonInfo(BaseModel):
    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")
    occupation: str = Field(description="Job title")
    location: str = Field(description="City and country")

class ExtractPerson(dspy.Signature):
    """Extract person information from text."""
    text = dspy.InputField()
    person: PersonInfo = dspy.OutputField()

extractor = dspy.TypedPredictor(ExtractPerson)

text = "Dr. Jane Smith, 42, is a neuroscientist at Stanford in Palo Alto, CA."
result = extractor(text=text)

print(result.person.name)       # "Dr. Jane Smith"
print(result.person.age)        # 42
print(result.person.occupation) # "neuroscientist"
print(result.person.location)   # "Palo Alto, California"
```

### Adaptive Summarization

```python
class AdaptiveSummarizer(dspy.Module):
    """Summarizes text to target length."""

    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought("text, target_length -> summary")

    def forward(self, text, target_length="3 sentences"):
        return self.summarize(text=text, target_length=target_length)

summarizer = AdaptiveSummarizer()
long_text = "..."  # Long article

short_summary = summarizer(long_text, target_length="1 sentence")
medium_summary = summarizer(long_text, target_length="3 sentences")
detailed_summary = summarizer(long_text, target_length="1 paragraph")
```

### Batch Processing

```python
class BatchProcessor(dspy.Module):
    """Process large datasets efficiently."""

    def __init__(self):
        super().__init__()
        self.process = dspy.Predict("text -> processed_text")

    def forward(self, texts):
        # Batch processing for efficiency
        return self.process.batch([{"text": t} for t in texts])

# Process 1000 documents
processor = BatchProcessor()
results = processor(texts=large_dataset)
```

---

## Multi-Stage Pipeline Patterns

### Document Processing Pipeline

```python
class DocumentPipeline(dspy.Module):
    """Multi-stage document processing."""

    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict("document -> key_points")
        self.classify = dspy.Predict("key_points -> category")
        self.summarize = dspy.ChainOfThought("key_points, category -> summary")
        self.tag = dspy.Predict("summary -> tags")

    def forward(self, document):
        # Stage 1: Extract key points
        key_points = self.extract(document=document).key_points

        # Stage 2: Classify
        category = self.classify(key_points=key_points).category

        # Stage 3: Summarize
        summary = self.summarize(
            key_points=key_points,
            category=category
        ).summary

        # Stage 4: Generate tags
        tags = self.tag(summary=summary).tags

        return dspy.Prediction(
            key_points=key_points,
            category=category,
            summary=summary,
            tags=tags
        )
```

### Sequential Pipeline

```python
class Pipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = dspy.Predict("input -> intermediate")
        self.stage2 = dspy.ChainOfThought("intermediate -> output")

    def forward(self, input):
        intermediate = self.stage1(input=input).intermediate
        output = self.stage2(intermediate=intermediate).output
        return dspy.Prediction(output=output)
```

### Conditional Pipeline

```python
class ConditionalModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.router = dspy.Predict("question -> complexity: str")
        self.simple_qa = dspy.Predict("question -> answer")
        self.complex_qa = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        complexity = self.router(question=question).complexity

        if complexity == "simple":
            return self.simple_qa(question=question)
        else:
            return self.complex_qa(question=question)
```

### Parallel Pipeline

```python
class ParallelModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.approach1 = dspy.ChainOfThought("question -> answer")
        self.approach2 = dspy.ProgramOfThought("question -> answer")

    def forward(self, question):
        # Run both approaches
        answer1 = self.approach1(question=question).answer
        answer2 = self.approach2(question=question).answer

        # Compare or combine results
        if answer1 == answer2:
            return dspy.Prediction(answer=answer1, confidence="high")
        else:
            return dspy.Prediction(answer=answer1, confidence="low")
```

---

## Quality Control Patterns

### Self-Refinement with Reward Function

```python
import dspy

qa = dspy.ChainOfThought("question -> answer")

def one_word_answer(args, pred):
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

best_of_3 = dspy.Refine(
    module=qa,
    N=3,
    reward_fn=one_word_answer,
    threshold=1.0
)

result = best_of_3(question="What is the capital of Belgium?")
print(result.answer)  # "Brussels"
```

### Quality Control with Verification

```python
class QualityControlPipeline(dspy.Module):
    """Generate output and verify quality."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought("prompt -> output")
        self.verify = dspy.Predict("output -> is_valid: bool, issues: str")
        self.improve = dspy.ChainOfThought("output, issues -> improved_output")

    def forward(self, prompt, max_iterations=3):
        output = self.generate(prompt=prompt).output

        for i in range(max_iterations):
            # Verify output
            verification = self.verify(output=output)

            if verification.is_valid:
                return dspy.Prediction(output=output, iterations=i + 1)

            # Improve based on issues
            output = self.improve(
                output=output,
                issues=verification.issues
            ).improved_output

        return dspy.Prediction(output=output, iterations=max_iterations)
```

### Verification Loop

```python
class VerifiedQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.answer = dspy.ChainOfThought("question -> answer")
        self.verify = dspy.Predict("question, answer -> is_correct: bool")

    def forward(self, question, max_attempts=3):
        for _ in range(max_attempts):
            answer = self.answer(question=question).answer
            is_correct = self.verify(question=question, answer=answer).is_correct

            if is_correct:
                return dspy.Prediction(answer=answer)

        return dspy.Prediction(answer="Unable to verify answer")
```

### Self-Consistency (Multiple Samples)

```python
from collections import Counter

class ConsistentQA(dspy.Module):
    def __init__(self, num_samples=5):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
        self.num_samples = num_samples

    def forward(self, question):
        # Generate multiple answers
        answers = []
        for _ in range(self.num_samples):
            result = self.qa(question=question)
            answers.append(result.answer)

        # Return most common answer
        most_common = Counter(answers).most_common(1)[0][0]
        return dspy.Prediction(answer=most_common)
```

---

## Production Patterns

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
            result = self.process(input=input)
            return result
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

### A/B Testing

```python
import random

class ABTestModule(dspy.Module):
    """Run two variants and compare."""

    def __init__(self, variant_a, variant_b):
        super().__init__()
        self.variant_a = variant_a
        self.variant_b = variant_b
        self.a_calls = 0
        self.b_calls = 0

    def forward(self, input, variant=None):
        if variant is None:
            variant = "a" if random.random() < 0.5 else "b"

        if variant == "a":
            self.a_calls += 1
            return self.variant_a(input=input)
        else:
            self.b_calls += 1
            return self.variant_b(input=input)

# Compare two optimizers
baseline = dspy.ChainOfThought("question -> answer")
optimized = BootstrapFewShot(metric=metric).compile(baseline, trainset=trainset)

ab_test = ABTestModule(variant_a=baseline, variant_b=optimized)
```

### Complete Customer Support Bot

```python
class CustomerSupportBot(dspy.Module):
    """Complete customer support system."""

    def __init__(self):
        super().__init__()

        # Classify intent
        self.classify_intent = dspy.Predict("message -> intent: str")

        # Specialized handlers
        self.technical_handler = dspy.ChainOfThought("message, history -> response")
        self.billing_handler = dspy.ChainOfThought("message, history -> response")
        self.general_handler = dspy.Predict("message, history -> response")

        # Retrieve relevant docs
        self.retrieve = dspy.Retrieve(k=3)

        # Conversation history
        self.history = []

    def forward(self, message):
        # Classify intent
        intent = self.classify_intent(message=message).intent

        # Retrieve relevant documentation
        docs = self.retrieve(message).passages
        context = "\n".join(docs)

        # Add context to history
        history_str = "\n".join(self.history)
        full_message = f"Context: {context}\n\nMessage: {message}"

        # Route to appropriate handler
        if intent == "technical":
            response = self.technical_handler(
                message=full_message,
                history=history_str
            ).response
        elif intent == "billing":
            response = self.billing_handler(
                message=full_message,
                history=history_str
            ).response
        else:
            response = self.general_handler(
                message=full_message,
                history=history_str
            ).response

        # Update history
        self.history.append(f"User: {message}")
        self.history.append(f"Bot: {response}")

        return dspy.Prediction(response=response, intent=intent)
```

---

## Quick Reference

### Pattern Selection Guide

| Use Case | Recommended Pattern |
|----------|-------------------|
| Simple Q&A | BasicRAG |
| Complex research | MultiHopRAG + ReAct |
| Text classification | Binary/Multi-class Classifier |
| Data extraction | TypedPredictor + Pydantic |
| Document processing | Multi-stage Pipeline |
| Production deployment | Cached + Monitored Module |
| High reliability | Self-Consistency or Refine |
| A/B testing | ABTestModule |

### Essential Imports

```python
import dspy
from dspy import ReAct, Retrieve, Evaluate
from dspy.teleprompt import BootstrapFewShot, MIPROv2
from pydantic import BaseModel, Field
from functools import lru_cache
from collections import Counter
```

---

## Resources

- **Official Documentation**: https://dspy.ai
- **Examples Repo**: https://github.com/stanfordnlp/dspy/tree/main/examples
- **Discord**: https://discord.gg/XCGy2WDCQB

---

*See also:*
- `DSPY_BEST_PRACTICES.md` - Core concepts and configuration
- `DSPY_OPTIMIZERS_GUIDE.md` - Deep dive into optimization algorithms
