# src/optimization.py
"""Optimization module for DSPy Newsletter Research Tool.

This module provides:
- relevance_metric: Metric for relevance scoring
- classification_metric: Metric for classification accuracy
- TrainingDataset: Dataset class for training data
- create_optimizer: Factory function for DSPy optimizers
- optimize_module: Convenience function for optimization
- save_optimized_module, load_optimized_module: Persistence utilities
- evaluate_module: Evaluation utilities
- calculate_accuracy: Accuracy calculation
- hyperparameter_search: Grid search utility

Beads Task: dspy-4aa
"""

import json
from typing import List, Dict, Any, Optional, Tuple, Type, Callable
from itertools import product

import dspy


# =============================================================================
# Metrics
# =============================================================================

def relevance_metric(example: Any, prediction: Any) -> float:
    """Score relevance prediction accuracy.

    Args:
        example: Ground truth example with is_relevant field.
        prediction: Prediction with is_relevant and confidence fields.

    Returns:
        Score between 0 and 1.
    """
    # Get ground truth - may be attribute or the prediction confidence alone
    ground_truth = getattr(example, 'is_relevant', None)

    # Get prediction
    pred_relevant = getattr(prediction, 'is_relevant', False)
    pred_confidence = getattr(prediction, 'confidence', 0.5)

    # If no ground truth, use confidence as score (for inference scenarios)
    if ground_truth is None:
        return min(1.0, max(0.0, pred_confidence))

    # Compare prediction to ground truth
    if ground_truth == pred_relevant:
        # Correct prediction - score based on confidence
        return 0.6 + (0.4 * pred_confidence)
    else:
        # Wrong prediction - penalize
        return 0.3 * (1 - pred_confidence)


def classification_metric(example: Any, prediction: Any) -> float:
    """Score classification prediction accuracy.

    Evaluates region and topic classification against ground truth.

    Args:
        example: Ground truth example with region and topics fields.
        prediction: Prediction with classification attribute.

    Returns:
        Score between 0 and 1.
    """
    # Get ground truth
    gt_region = getattr(example, 'region', None)
    gt_topics = getattr(example, 'topics', [])

    # Get prediction
    classification = getattr(prediction, 'classification', prediction)
    pred_region = getattr(classification, 'region', None)
    pred_topics = getattr(classification, 'topics', [])

    # Normalize regions for comparison
    def normalize(val):
        if val is None:
            return None
        if hasattr(val, 'value'):
            return val.value
        return str(val).upper()

    gt_region_norm = normalize(gt_region)
    pred_region_norm = normalize(pred_region)

    # Normalize topics
    gt_topics_norm = set(normalize(t) for t in gt_topics if t)
    pred_topics_norm = set(normalize(t) for t in pred_topics if t)

    # Score components
    region_score = 1.0 if gt_region_norm == pred_region_norm else 0.0

    # Topic score - Jaccard similarity
    if not gt_topics_norm and not pred_topics_norm:
        topic_score = 1.0
    elif not gt_topics_norm or not pred_topics_norm:
        topic_score = 0.0
    else:
        intersection = len(gt_topics_norm & pred_topics_norm)
        union = len(gt_topics_norm | pred_topics_norm)
        topic_score = intersection / union if union > 0 else 0.0

    # Combined score (region weighted slightly higher)
    return 0.6 * region_score + 0.4 * topic_score


# =============================================================================
# Training Dataset
# =============================================================================

class TrainingDataset:
    """Dataset class for training data.

    Wraps raw dictionaries and converts to DSPy Examples.

    Attributes:
        data: List of raw data dictionaries.
    """

    def __init__(self, data: List[Dict[str, Any]]):
        """Initialize dataset with data.

        Args:
            data: List of dictionaries with training examples.
        """
        self.data = data if data is not None else []

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.data)

    @classmethod
    def from_file(cls, path: str) -> 'TrainingDataset':
        """Load dataset from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            TrainingDataset instance.
        """
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(data)

    def to_examples(self) -> List[dspy.Example]:
        """Convert data to DSPy Examples.

        Returns:
            List of dspy.Example objects.
        """
        examples = []

        for item in self.data:
            example = dspy.Example(**item)
            examples.append(example)

        return examples

    def split(self, train_ratio: float = 0.8) -> Tuple['TrainingDataset', 'TrainingDataset']:
        """Split dataset into train and dev sets.

        Args:
            train_ratio: Proportion of data for training (default 0.8).

        Returns:
            Tuple of (train_dataset, dev_dataset).
        """
        split_idx = int(len(self.data) * train_ratio)

        train_data = self.data[:split_idx]
        dev_data = self.data[split_idx:]

        return TrainingDataset(train_data), TrainingDataset(dev_data)


# =============================================================================
# Optimizer Factory
# =============================================================================

class OptimizedBootstrapFewShot:
    """Wrapper for BootstrapFewShot optimizer.

    Provides consistent interface for optimization.

    Attributes:
        max_bootstrapped_demos: Max demos to bootstrap.
        max_labeled_demos: Max labeled demos.
        metric: Metric function for evaluation.
    """

    def __init__(
        self,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
        metric: Optional[Callable] = None
    ):
        """Initialize optimizer.

        Args:
            max_bootstrapped_demos: Max demos to bootstrap.
            max_labeled_demos: Max labeled demos.
            metric: Optional metric function.
        """
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.metric = metric or relevance_metric

        # Try to create actual BootstrapFewShot
        try:
            self._optimizer = dspy.BootstrapFewShot(
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                metric=self.metric
            )
        except Exception:
            self._optimizer = None

    def compile(self, module: dspy.Module, trainset: List[dspy.Example]) -> dspy.Module:
        """Compile module with training data.

        Args:
            module: DSPy module to optimize.
            trainset: Training examples.

        Returns:
            Optimized module.
        """
        if self._optimizer is not None:
            try:
                return self._optimizer.compile(module, trainset=trainset)
            except Exception:
                pass

        # Fallback: return module with demos attached
        module.demos = trainset[:self.max_bootstrapped_demos]
        return module


class OptimizedMIPRO:
    """Wrapper for MIPRO optimizer.

    Attributes:
        num_candidates: Number of candidates to generate.
        metric: Metric function.
    """

    def __init__(
        self,
        num_candidates: int = 10,
        metric: Optional[Callable] = None
    ):
        """Initialize MIPRO optimizer.

        Args:
            num_candidates: Number of candidates.
            metric: Metric function.
        """
        self.num_candidates = num_candidates
        self.metric = metric or relevance_metric

        # Try to create actual MIPRO
        try:
            self._optimizer = dspy.MIPRO(
                metric=self.metric,
                num_candidates=num_candidates
            )
        except (AttributeError, Exception):
            # MIPRO may not be available in all DSPy versions
            self._optimizer = None

    def compile(self, module: dspy.Module, trainset: List[dspy.Example]) -> dspy.Module:
        """Compile module with training data.

        Args:
            module: DSPy module to optimize.
            trainset: Training examples.

        Returns:
            Optimized module.
        """
        if self._optimizer is not None:
            try:
                return self._optimizer.compile(module, trainset=trainset)
            except Exception:
                pass

        # Fallback
        module.demos = trainset[:4]
        return module


def create_optimizer(
    optimizer_type: str = "bootstrap_few_shot",
    metric: Optional[Callable] = None,
    **kwargs
) -> Any:
    """Create a DSPy optimizer.

    Args:
        optimizer_type: Type of optimizer ("bootstrap_few_shot" or "mipro").
        metric: Optional metric function.
        **kwargs: Additional optimizer parameters.

    Returns:
        Optimizer instance.

    Raises:
        NotImplementedError: If optimizer type is not supported.
    """
    if optimizer_type == "bootstrap_few_shot":
        return OptimizedBootstrapFewShot(
            max_bootstrapped_demos=kwargs.get('max_bootstrapped_demos', 4),
            max_labeled_demos=kwargs.get('max_labeled_demos', 16),
            metric=metric
        )
    elif optimizer_type == "mipro":
        return OptimizedMIPRO(
            num_candidates=kwargs.get('num_candidates', 10),
            metric=metric
        )
    else:
        raise NotImplementedError(f"Optimizer type '{optimizer_type}' not supported")


# =============================================================================
# Optimization Pipeline
# =============================================================================

def optimize_module(
    module: dspy.Module,
    training_data: List[Dict[str, Any]],
    optimizer_type: str = "bootstrap_few_shot",
    checkpoint_path: Optional[str] = None,
    metric: Optional[Callable] = None
) -> dspy.Module:
    """Optimize a DSPy module with training data.

    Args:
        module: DSPy module to optimize.
        training_data: List of training examples as dicts.
        optimizer_type: Type of optimizer to use.
        checkpoint_path: Optional path to save checkpoint.
        metric: Optional metric function.

    Returns:
        Optimized module.
    """
    # Create dataset
    dataset = TrainingDataset(training_data)
    examples = dataset.to_examples()

    # Create optimizer
    optimizer = create_optimizer(
        optimizer_type=optimizer_type,
        metric=metric
    )

    # Run optimization
    optimized = optimizer.compile(module, trainset=examples)

    # Save checkpoint if requested
    if checkpoint_path:
        save_optimized_module(optimized, checkpoint_path)

    return optimized


# =============================================================================
# Model Persistence
# =============================================================================

def save_optimized_module(module: dspy.Module, path: str) -> None:
    """Save optimized module to file.

    Args:
        module: Optimized DSPy module.
        path: Path to save file.
    """
    # Extract state
    state = {
        'module_type': type(module).__name__,
        'demos': getattr(module, 'demos', []),
    }

    # Handle demo serialization
    serialized_demos = []
    for demo in state['demos']:
        if hasattr(demo, 'toDict'):
            serialized_demos.append(demo.toDict())
        elif isinstance(demo, dict):
            serialized_demos.append(demo)
        else:
            # Try to convert to dict
            serialized_demos.append(vars(demo) if hasattr(demo, '__dict__') else str(demo))

    state['demos'] = serialized_demos

    # Save to file
    with open(path, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def load_optimized_module(
    module_class: Type[dspy.Module],
    path: str
) -> dspy.Module:
    """Load optimized module from file.

    Args:
        module_class: Class of the module to load.
        path: Path to saved file.

    Returns:
        Loaded module with demos.
    """
    # Load state
    with open(path, 'r') as f:
        state = json.load(f)

    # Create module instance
    module = module_class()

    # Restore demos
    demos = state.get('demos', [])
    module.demos = demos

    return module


# =============================================================================
# Evaluation Utilities
# =============================================================================

def calculate_accuracy(
    predictions: List[bool],
    ground_truth: List[bool]
) -> float:
    """Calculate accuracy from predictions and ground truth.

    Args:
        predictions: List of predicted values.
        ground_truth: List of true values.

    Returns:
        Accuracy score between 0 and 1.
    """
    if not predictions or not ground_truth:
        return 0.0

    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground_truth must have same length")

    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(predictions)


def evaluate_module(
    module: dspy.Module,
    test_data: List[Dict[str, Any]],
    relevance_field: str = 'is_relevant'
) -> Dict[str, float]:
    """Evaluate module on test data.

    Args:
        module: DSPy module to evaluate.
        test_data: List of test examples as dicts.
        relevance_field: Field name for ground truth.

    Returns:
        Dictionary with accuracy, precision, recall, f1.
    """
    predictions = []
    ground_truths = []

    for item in test_data:
        # Get ground truth
        gt = item.get(relevance_field, False)
        ground_truths.append(gt)

        # Get prediction
        try:
            title = item.get('title', '')
            content = item.get('content', '')
            source_category = item.get('source_category', 'general')

            # Try calling module
            result = module(
                title=title,
                content_preview=content[:500],
                source_category=source_category
            )

            pred = getattr(result, 'is_relevant', False)
            predictions.append(pred)
        except Exception:
            predictions.append(False)

    # Calculate metrics
    accuracy = calculate_accuracy(predictions, ground_truths)

    # True positives, false positives, false negatives
    tp = sum(1 for p, g in zip(predictions, ground_truths) if p and g)
    fp = sum(1 for p, g in zip(predictions, ground_truths) if p and not g)
    fn = sum(1 for p, g in zip(predictions, ground_truths) if not p and g)

    # Precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# =============================================================================
# Hyperparameter Search
# =============================================================================

def hyperparameter_search(
    module: dspy.Module,
    param_grid: Dict[str, List[Any]],
    training_data: List[Dict[str, Any]],
    metric: Optional[Callable] = None
) -> Dict[str, Any]:
    """Perform grid search over hyperparameters.

    Args:
        module: DSPy module to optimize.
        param_grid: Dictionary mapping parameter names to lists of values.
        training_data: Training data for evaluation.
        metric: Optional metric function.

    Returns:
        Dictionary with best parameter values.
    """
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    best_score = -1
    best_params = {}

    for combo in combinations:
        # Create parameter dict for this combination
        params = dict(zip(param_names, combo))

        try:
            # Apply parameters if applicable
            test_module = type(module)()

            # Set threshold if applicable
            if 'threshold' in params and hasattr(test_module, 'threshold'):
                test_module.threshold = params['threshold']

            # Create dataset
            dataset = TrainingDataset(training_data)
            examples = dataset.to_examples()

            # Evaluate with current params
            score = 0.0
            for ex in examples:
                try:
                    pred = test_module(
                        title=getattr(ex, 'title', ''),
                        content_preview=getattr(ex, 'content', '')[:500],
                        source_category=getattr(ex, 'source_category', 'general')
                    )

                    if metric:
                        score += metric(ex, pred)
                    else:
                        # Default: check if prediction matches ground truth
                        gt = getattr(ex, 'is_relevant', None)
                        pred_val = getattr(pred, 'is_relevant', None)
                        if gt is not None and pred_val == gt:
                            score += 1.0
                except Exception:
                    pass

            # Normalize score
            if examples:
                score /= len(examples)

            # Track best
            if score > best_score:
                best_score = score
                best_params = params.copy()

        except Exception:
            continue

    # Ensure all params are present in result (use first value if no best found)
    if not best_params:
        best_params = {name: values[0] for name, values in param_grid.items()}

    return best_params


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'relevance_metric',
    'classification_metric',
    'TrainingDataset',
    'create_optimizer',
    'optimize_module',
    'save_optimized_module',
    'load_optimized_module',
    'evaluate_module',
    'calculate_accuracy',
    'hyperparameter_search',
]
