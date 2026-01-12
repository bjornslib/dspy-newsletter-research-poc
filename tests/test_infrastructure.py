# tests/test_infrastructure.py
"""
RED PHASE: Infrastructure tests for DSPy Newsletter Research Tool PoC.

Beads Task: dspy-fx7 (Infrastructure Setup)

These tests verify:
- Docker environment is properly configured
- Weaviate container is running and accessible
- DSPy is configured with appropriate language models
- Environment variables are loaded correctly

Expected Behavior in RED Phase:
- All tests should FAIL because infrastructure is not yet set up
- Import errors indicate modules don't exist yet
- Connection errors indicate services aren't running
"""

import pytest
import os


# =============================================================================
# Docker Environment Tests
# =============================================================================

class TestDockerEnvironment:
    """RED: Test Docker environment before setup."""

    def test_docker_client_available(self):
        """Test Docker is running and accessible."""
        import docker
        client = docker.from_env()
        assert client.ping(), "Docker daemon should be running"

    def test_weaviate_container_exists(self):
        """Test Weaviate container exists."""
        import docker
        client = docker.from_env()
        containers = client.containers.list(all=True, filters={'name': 'weaviate'})
        assert len(containers) > 0, "Weaviate container should exist"

    def test_weaviate_container_running(self):
        """Test Weaviate container is in running state."""
        import docker
        client = docker.from_env()
        containers = client.containers.list(filters={'name': 'weaviate'})
        assert len(containers) > 0, "Weaviate container should be running"
        assert containers[0].status == 'running', "Weaviate should be in running state"

    def test_docker_compose_file_exists(self):
        """Test docker-compose.yml exists in project root."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        compose_path = os.path.join(project_root, 'docker-compose.yml')
        assert os.path.exists(compose_path), "docker-compose.yml should exist"


# =============================================================================
# Weaviate Connection Tests
# =============================================================================

class TestWeaviateConnection:
    """Test Weaviate connectivity and configuration."""

    def test_weaviate_healthcheck(self):
        """Test Weaviate is accessible and healthy."""
        import weaviate
        client = weaviate.connect_to_local()
        try:
            assert client.is_ready(), "Weaviate should be ready"
        finally:
            client.close()

    def test_weaviate_has_newsletter_collection(self):
        """Test NewsletterArticles collection exists in Weaviate."""
        import weaviate
        client = weaviate.connect_to_local()
        try:
            collections = client.collections.list_all()
            # list_all() returns a dict where keys are collection names
            collection_names = list(collections.keys())
            assert 'NewsletterArticles' in collection_names, \
                "NewsletterArticles collection should exist"
        finally:
            client.close()

    def test_newsletter_collection_has_required_properties(self):
        """Test NewsletterArticles collection has expected schema."""
        import weaviate
        client = weaviate.connect_to_local()
        try:
            collection = client.collections.get('NewsletterArticles')
            config = collection.config.get()

            required_properties = [
                'title', 'content', 'source_url', 'published_date',
                'region', 'topics', 'relevance_score'
            ]

            property_names = [p.name for p in config.properties]
            for prop in required_properties:
                assert prop in property_names, \
                    f"Property '{prop}' should exist in NewsletterArticles schema"
        finally:
            client.close()

    def test_weaviate_vectorizer_configured(self):
        """Test Weaviate collection has vectorizer configured."""
        import weaviate
        client = weaviate.connect_to_local()
        try:
            collection = client.collections.get('NewsletterArticles')
            config = collection.config.get()
            assert config.vectorizer is not None, \
                "Vectorizer should be configured for NewsletterArticles"
        finally:
            client.close()


# =============================================================================
# DSPy Configuration Tests
# =============================================================================

class TestDSPyConfiguration:
    """Test DSPy is properly configured."""

    def test_dspy_can_be_imported(self):
        """Test DSPy package is installed and importable."""
        import dspy
        assert dspy is not None

    def test_dspy_lm_configured(self):
        """Test a language model is configured in DSPy settings."""
        import dspy
        # This requires the project's config to be loaded
        from src.config import configure_dspy
        configure_dspy()
        assert dspy.settings.lm is not None, "DSPy LM should be configured"

    def test_dspy_supports_typed_signatures(self):
        """Test DSPy Signature (typed predictor) is available.

        Note: In DSPy 3.x, TypedPredictor is replaced by using
        dspy.Signature with Pydantic models for typed outputs.
        """
        from dspy import Signature, Predict
        assert Signature is not None
        assert Predict is not None

    def test_dspy_supports_chain_of_thought(self):
        """Test DSPy ChainOfThought is available."""
        from dspy import ChainOfThought
        assert ChainOfThought is not None

    def test_dspy_supports_react(self):
        """Test DSPy ReAct module is available."""
        from dspy import ReAct
        assert ReAct is not None


# =============================================================================
# Environment Variables Tests
# =============================================================================

class TestEnvironmentVariables:
    """Test required environment variables are present."""

    def test_openai_api_key_exists(self):
        """Test OPENAI_API_KEY environment variable is set."""
        api_key = os.getenv('OPENAI_API_KEY')
        assert api_key is not None, "OPENAI_API_KEY should be set"
        assert len(api_key) > 0, "OPENAI_API_KEY should not be empty"

    def test_cohere_api_key_exists(self):
        """Test COHERE_API_KEY environment variable is set."""
        api_key = os.getenv('COHERE_API_KEY')
        assert api_key is not None, "COHERE_API_KEY should be set"
        assert len(api_key) > 0, "COHERE_API_KEY should not be empty"

    def test_weaviate_url_configured(self):
        """Test Weaviate URL is configured (default or env var)."""
        weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
        assert 'localhost' in weaviate_url or 'weaviate' in weaviate_url, \
            "Weaviate URL should point to local or container service"

    def test_env_file_exists(self):
        """Test .env file exists in project root."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(project_root, '.env')
        assert os.path.exists(env_path), ".env file should exist in project root"


# =============================================================================
# Project Structure Tests
# =============================================================================

class TestProjectStructure:
    """Test project directory structure is correct."""

    def test_src_directory_exists(self):
        """Test src/ directory exists."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src_path = os.path.join(project_root, 'src')
        assert os.path.isdir(src_path), "src/ directory should exist"

    def test_src_has_init_file(self):
        """Test src/__init__.py exists."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        init_path = os.path.join(project_root, 'src', '__init__.py')
        assert os.path.exists(init_path), "src/__init__.py should exist"

    def test_config_module_exists(self):
        """Test src/config.py exists."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, 'src', 'config.py')
        assert os.path.exists(config_path), "src/config.py should exist"

    def test_config_module_importable(self):
        """Test config module can be imported."""
        from src.config import configure_dspy
        assert callable(configure_dspy), "configure_dspy should be callable"
