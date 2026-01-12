#!/usr/bin/env python3
# scripts/health_check.py
"""Health check script for DSPy Newsletter Research Tool infrastructure.

Verifies:
- Docker daemon connectivity
- Weaviate container status
- Weaviate API accessibility
- NewsletterArticles collection existence
- DSPy configuration
- Environment variables

Usage:
    python scripts/health_check.py
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_docker() -> bool:
    """Check Docker daemon is running."""
    print("Checking Docker daemon...")
    try:
        import docker
        client = docker.from_env()
        client.ping()
        print("  Docker daemon: OK")
        return True
    except Exception as e:
        print(f"  Docker daemon: FAILED - {e}")
        return False


def check_weaviate_container() -> bool:
    """Check Weaviate container is running."""
    print("Checking Weaviate container...")
    try:
        import docker
        client = docker.from_env()
        containers = client.containers.list(filters={'name': 'weaviate'})
        if containers and containers[0].status == 'running':
            print("  Weaviate container: OK (running)")
            return True
        else:
            print("  Weaviate container: FAILED (not running)")
            return False
    except Exception as e:
        print(f"  Weaviate container: FAILED - {e}")
        return False


def check_weaviate_api() -> bool:
    """Check Weaviate API is accessible."""
    print("Checking Weaviate API...")
    try:
        import weaviate
        client = weaviate.connect_to_local()
        is_ready = client.is_ready()
        client.close()
        if is_ready:
            print("  Weaviate API: OK")
            return True
        else:
            print("  Weaviate API: FAILED (not ready)")
            return False
    except Exception as e:
        print(f"  Weaviate API: FAILED - {e}")
        return False


def check_newsletter_collection() -> bool:
    """Check NewsletterArticles collection exists."""
    print("Checking NewsletterArticles collection...")
    try:
        import weaviate
        client = weaviate.connect_to_local()
        exists = client.collections.exists("NewsletterArticles")
        client.close()
        if exists:
            print("  NewsletterArticles collection: OK")
            return True
        else:
            print("  NewsletterArticles collection: FAILED (not found)")
            return False
    except Exception as e:
        print(f"  NewsletterArticles collection: FAILED - {e}")
        return False


def check_collection_schema() -> bool:
    """Check NewsletterArticles has required properties."""
    print("Checking collection schema...")
    try:
        import weaviate
        client = weaviate.connect_to_local()
        collection = client.collections.get("NewsletterArticles")
        config = collection.config.get()

        required_props = [
            'title', 'content', 'source_url', 'published_date',
            'region', 'topics', 'relevance_score'
        ]
        property_names = [p.name for p in config.properties]

        missing = [p for p in required_props if p not in property_names]
        client.close()

        if not missing:
            print("  Collection schema: OK (all required properties present)")
            return True
        else:
            print(f"  Collection schema: FAILED (missing: {missing})")
            return False
    except Exception as e:
        print(f"  Collection schema: FAILED - {e}")
        return False


def check_dspy_config() -> bool:
    """Check DSPy can be configured."""
    print("Checking DSPy configuration...")
    try:
        from src.config import configure_dspy
        import dspy
        configure_dspy()
        if dspy.settings.lm is not None:
            print("  DSPy configuration: OK")
            return True
        else:
            print("  DSPy configuration: FAILED (LM not set)")
            return False
    except Exception as e:
        print(f"  DSPy configuration: FAILED - {e}")
        return False


def check_env_vars() -> bool:
    """Check required environment variables."""
    print("Checking environment variables...")
    from dotenv import load_dotenv
    load_dotenv(project_root / '.env')

    required = ['OPENAI_API_KEY', 'COHERE_API_KEY']
    missing = []

    for var in required:
        value = os.getenv(var)
        if not value:
            missing.append(var)
            print(f"  {var}: MISSING")
        else:
            print(f"  {var}: OK (set)")

    if not missing:
        return True
    else:
        print(f"  Environment variables: FAILED (missing: {missing})")
        return False


def main():
    """Run all health checks."""
    print("=" * 60)
    print("DSPy Newsletter Research Tool - Infrastructure Health Check")
    print("=" * 60)
    print()

    checks = [
        ("Docker", check_docker),
        ("Weaviate Container", check_weaviate_container),
        ("Weaviate API", check_weaviate_api),
        ("Newsletter Collection", check_newsletter_collection),
        ("Collection Schema", check_collection_schema),
        ("Environment Variables", check_env_vars),
        ("DSPy Configuration", check_dspy_config),
    ]

    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"  Unexpected error: {e}")
            results.append((name, False))
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    print()
    print(f"Total: {passed}/{total} checks passed")

    if passed == total:
        print("\nAll health checks PASSED!")
        sys.exit(0)
    else:
        print("\nSome health checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
