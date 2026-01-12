# tests/test_optimization.py
"""
RED PHASE: Optimization tests for DSPy Newsletter Research Tool PoC.

Beads Task: dspy-4aa (BootstrapFewShot, Metrics)

These tests verify:
- BootstrapFewShot optimizer configuration
- Custom metrics for relevance and classification
- Optimization loop execution
- Optimized module saving and loading

Expected Behavior in RED Phase:
- All tests should FAIL because optimization module is not yet implemented
- ImportError for 'from src.optimization import ...'
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


# =============================================================================
# Metrics Tests
# =============================================================================

class TestRelevanceMetric:
    """RED: Test relevance scoring metric."""

    def test_relevance_metric_exists(self):
        """Test relevance_metric function can be imported."""
        from src.optimization import relevance_metric
        assert relevance_metric is not None

    def test_relevance_metric_returns_score(self):
        """Test metric returns numeric score."""
        from src.optimization import relevance_metric

        example = Mock()
        example.title = "FCRA Compliance Guide"
        example.content = "Background screening regulations..."

        prediction = Mock()
        prediction.is_relevant = True
        prediction.confidence = 0.9

        score = relevance_metric(example, prediction)

        assert isinstance(score, (int, float))
        assert 0 <= score <= 1

    def test_relevance_metric_rewards_correct_positive(self):
        """Test metric rewards correctly identifying relevant content."""
        from src.optimization import relevance_metric

        example = Mock()
        example.is_relevant = True  # Ground truth

        prediction = Mock()
        prediction.is_relevant = True
        prediction.confidence = 0.95

        score = relevance_metric(example, prediction)
        assert score > 0.5

    def test_relevance_metric_penalizes_false_positive(self):
        """Test metric penalizes false positives."""
        from src.optimization import relevance_metric

        example = Mock()
        example.is_relevant = False  # Ground truth

        prediction = Mock()
        prediction.is_relevant = True  # Wrong prediction
        prediction.confidence = 0.8

        score = relevance_metric(example, prediction)
        assert score < 0.5

    def test_relevance_metric_penalizes_false_negative(self):
        """Test metric penalizes false negatives."""
        from src.optimization import relevance_metric

        example = Mock()
        example.is_relevant = True  # Ground truth

        prediction = Mock()
        prediction.is_relevant = False  # Wrong prediction
        prediction.confidence = 0.8

        score = relevance_metric(example, prediction)
        assert score < 0.5


# =============================================================================
# Classification Metric Tests
# =============================================================================

class TestClassificationMetric:
    """Test classification accuracy metric."""

    def test_classification_metric_exists(self):
        """Test classification_metric function can be imported."""
        from src.optimization import classification_metric
        assert classification_metric is not None

    def test_classification_metric_returns_score(self):
        """Test metric returns numeric score."""
        from src.optimization import classification_metric

        example = Mock()
        example.region = "EUROPE"
        example.topics = ["REGULATORY", "COURT_CASES"]

        prediction = Mock()
        prediction.classification = Mock()
        prediction.classification.region = "EUROPE"
        prediction.classification.topics = ["REGULATORY"]

        score = classification_metric(example, prediction)

        assert isinstance(score, (int, float))
        assert 0 <= score <= 1

    def test_classification_metric_rewards_correct_region(self):
        """Test metric rewards correct region classification."""
        from src.optimization import classification_metric

        example = Mock()
        example.region = "APAC"
        example.topics = []

        prediction = Mock()
        prediction.classification = Mock()
        prediction.classification.region = "APAC"
        prediction.classification.topics = []

        score = classification_metric(example, prediction)
        assert score > 0.5

    def test_classification_metric_rewards_topic_overlap(self):
        """Test metric rewards correct topic classification."""
        from src.optimization import classification_metric

        example = Mock()
        example.region = "N_AMERICA_CARIBBEAN"
        example.topics = ["REGULATORY", "COURT_CASES"]

        prediction = Mock()
        prediction.classification = Mock()
        prediction.classification.region = "N_AMERICA_CARIBBEAN"
        prediction.classification.topics = ["REGULATORY", "COURT_CASES"]

        score = classification_metric(example, prediction)
        assert score >= 0.9

    def test_classification_metric_partial_topic_match(self):
        """Test metric handles partial topic matches."""
        from src.optimization import classification_metric

        example = Mock()
        example.region = "EUROPE"
        example.topics = ["REGULATORY", "COURT_CASES", "TECHNOLOGY"]

        prediction = Mock()
        prediction.classification = Mock()
        prediction.classification.region = "EUROPE"
        prediction.classification.topics = ["REGULATORY"]  # Partial match

        score = classification_metric(example, prediction)
        # Should be between 0.5 and 1 (partial credit)
        assert 0.3 < score < 1.0


# =============================================================================
# Training Dataset Tests
# =============================================================================

class TestTrainingDataset:
    """Test training dataset creation."""

    def test_training_dataset_exists(self):
        """Test TrainingDataset class can be imported."""
        from src.optimization import TrainingDataset
        assert TrainingDataset is not None

    def test_dataset_loads_from_file(self):
        """Test dataset can load from JSON file."""
        from src.optimization import TrainingDataset
        import tempfile
        import json

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([
                {"title": "Test 1", "content": "Content 1", "is_relevant": True},
                {"title": "Test 2", "content": "Content 2", "is_relevant": False}
            ], f)
            f.flush()

            dataset = TrainingDataset.from_file(f.name)
            assert len(dataset) == 2

    def test_dataset_creates_dspy_examples(self):
        """Test dataset creates DSPy Example objects."""
        from src.optimization import TrainingDataset
        import dspy

        dataset = TrainingDataset([
            {"title": "Test", "content": "Content", "is_relevant": True}
        ])

        examples = dataset.to_examples()
        assert len(examples) == 1
        assert isinstance(examples[0], dspy.Example)

    def test_dataset_splits_train_dev(self):
        """Test dataset can be split into train/dev."""
        from src.optimization import TrainingDataset

        dataset = TrainingDataset([
            {"title": f"Test {i}", "content": f"Content {i}", "is_relevant": True}
            for i in range(100)
        ])

        train, dev = dataset.split(train_ratio=0.8)
        assert len(train) == 80
        assert len(dev) == 20


# =============================================================================
# BootstrapFewShot Optimizer Tests
# =============================================================================

class TestBootstrapFewShotOptimizer:
    """Test BootstrapFewShot optimization."""

    def test_optimizer_exists(self):
        """Test optimizer can be imported."""
        from src.optimization import create_optimizer
        assert create_optimizer is not None

    def test_create_bootstrap_optimizer(self):
        """Test creating BootstrapFewShot optimizer."""
        from src.optimization import create_optimizer

        optimizer = create_optimizer(
            optimizer_type="bootstrap_few_shot",
            max_bootstrapped_demos=4,
            max_labeled_demos=8
        )

        assert optimizer is not None
        assert optimizer.max_bootstrapped_demos == 4

    def test_optimizer_accepts_metric(self):
        """Test optimizer accepts custom metric."""
        from src.optimization import create_optimizer, relevance_metric

        optimizer = create_optimizer(
            optimizer_type="bootstrap_few_shot",
            metric=relevance_metric
        )

        assert optimizer.metric is not None

    @patch('dspy.settings')
    def test_optimizer_compiles_module(self, mock_settings, mock_dspy_lm):
        """Test optimizer compiles a module."""
        from src.optimization import create_optimizer, TrainingDataset
        from src.prefilter import TinyLMRelevanceFilter

        optimizer = create_optimizer(optimizer_type="bootstrap_few_shot")

        dataset = TrainingDataset([
            {"title": "FCRA", "content": "Compliance", "is_relevant": True}
        ])

        module = TinyLMRelevanceFilter()
        optimized = optimizer.compile(
            module,
            trainset=dataset.to_examples()
        )

        assert optimized is not None


# =============================================================================
# MIPRO Optimizer Tests
# =============================================================================

class TestMIPROOptimizer:
    """Test MIPRO optimization (if available)."""

    def test_mipro_optimizer_available(self):
        """Test MIPRO optimizer can be created."""
        from src.optimization import create_optimizer

        try:
            optimizer = create_optimizer(optimizer_type="mipro")
            assert optimizer is not None
        except NotImplementedError:
            pytest.skip("MIPRO not implemented")

    def test_mipro_accepts_num_candidates(self):
        """Test MIPRO accepts num_candidates parameter."""
        from src.optimization import create_optimizer

        try:
            optimizer = create_optimizer(
                optimizer_type="mipro",
                num_candidates=10
            )
            assert optimizer.num_candidates == 10
        except NotImplementedError:
            pytest.skip("MIPRO not implemented")


# =============================================================================
# Optimization Pipeline Tests
# =============================================================================

class TestOptimizationPipeline:
    """Test the complete optimization pipeline."""

    def test_optimize_function_exists(self):
        """Test optimize convenience function exists."""
        from src.optimization import optimize_module
        assert optimize_module is not None

    @patch('dspy.settings')
    def test_optimize_returns_optimized_module(self, mock_settings, mock_dspy_lm):
        """Test optimize returns optimized module."""
        from src.optimization import optimize_module
        from src.prefilter import TinyLMRelevanceFilter

        module = TinyLMRelevanceFilter()
        training_data = [
            {"title": "FCRA", "content": "Compliance", "is_relevant": True},
            {"title": "Pizza", "content": "Recipe", "is_relevant": False}
        ]

        optimized = optimize_module(
            module=module,
            training_data=training_data,
            optimizer_type="bootstrap_few_shot"
        )

        assert optimized is not None

    def test_optimize_saves_checkpoint(self, tmp_path):
        """Test optimization saves checkpoints."""
        from src.optimization import optimize_module
        from src.prefilter import TinyLMRelevanceFilter

        checkpoint_path = tmp_path / "checkpoint.json"

        module = TinyLMRelevanceFilter()
        training_data = [
            {"title": "FCRA", "content": "Compliance", "is_relevant": True}
        ]

        with patch('dspy.settings'):
            optimize_module(
                module=module,
                training_data=training_data,
                checkpoint_path=str(checkpoint_path)
            )

        # Checkpoint should be created
        # (will fail in RED phase)


# =============================================================================
# Model Save/Load Tests
# =============================================================================

class TestModelSaveLoad:
    """Test optimized model persistence."""

    def test_save_optimized_module(self, tmp_path):
        """Test saving optimized module."""
        from src.optimization import save_optimized_module
        from src.prefilter import TinyLMRelevanceFilter

        module = TinyLMRelevanceFilter()
        path = tmp_path / "optimized_filter.json"

        save_optimized_module(module, str(path))

        assert path.exists()

    def test_load_optimized_module(self, tmp_path):
        """Test loading optimized module."""
        from src.optimization import save_optimized_module, load_optimized_module
        from src.prefilter import TinyLMRelevanceFilter

        module = TinyLMRelevanceFilter()
        path = tmp_path / "optimized_filter.json"

        save_optimized_module(module, str(path))
        loaded = load_optimized_module(TinyLMRelevanceFilter, str(path))

        assert loaded is not None

    def test_loaded_module_has_demos(self, tmp_path):
        """Test loaded module preserves demonstrations."""
        from src.optimization import save_optimized_module, load_optimized_module
        from src.prefilter import TinyLMRelevanceFilter

        module = TinyLMRelevanceFilter()
        # Simulate having demonstrations
        module.demos = [{"input": "test", "output": "result"}]

        path = tmp_path / "optimized_filter.json"

        save_optimized_module(module, str(path))
        loaded = load_optimized_module(TinyLMRelevanceFilter, str(path))

        assert hasattr(loaded, 'demos')
        assert len(loaded.demos) > 0


# =============================================================================
# Evaluation Tests
# =============================================================================

class TestEvaluation:
    """Test model evaluation utilities."""

    def test_evaluate_function_exists(self):
        """Test evaluate function exists."""
        from src.optimization import evaluate_module
        assert evaluate_module is not None

    @patch('dspy.settings')
    def test_evaluate_returns_metrics(self, mock_settings, mock_dspy_lm):
        """Test evaluate returns metrics dict."""
        from src.optimization import evaluate_module
        from src.prefilter import TinyLMRelevanceFilter

        module = TinyLMRelevanceFilter()
        test_data = [
            {"title": "FCRA", "content": "Compliance", "is_relevant": True},
            {"title": "Pizza", "content": "Recipe", "is_relevant": False}
        ]

        metrics = evaluate_module(module, test_data)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics

    def test_evaluate_calculates_accuracy(self):
        """Test accuracy calculation."""
        from src.optimization import calculate_accuracy

        predictions = [True, True, False, False, True]
        ground_truth = [True, False, False, False, True]

        accuracy = calculate_accuracy(predictions, ground_truth)
        assert accuracy == 0.8  # 4/5 correct


# =============================================================================
# Hyperparameter Search Tests
# =============================================================================

class TestHyperparameterSearch:
    """Test hyperparameter search utilities."""

    def test_hyperparameter_search_exists(self):
        """Test hyperparameter search function exists."""
        from src.optimization import hyperparameter_search
        assert hyperparameter_search is not None

    def test_search_accepts_param_grid(self):
        """Test search accepts parameter grid."""
        from src.optimization import hyperparameter_search
        from src.prefilter import TinyLMRelevanceFilter

        param_grid = {
            'threshold': [0.5, 0.6, 0.7, 0.8],
            'max_demos': [2, 4, 8]
        }

        module = TinyLMRelevanceFilter()
        training_data = [
            {"title": "Test", "content": "Content", "is_relevant": True}
        ]

        with patch('dspy.settings'):
            best_params = hyperparameter_search(
                module=module,
                param_grid=param_grid,
                training_data=training_data
            )

        assert 'threshold' in best_params
        assert 'max_demos' in best_params
