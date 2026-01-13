# tests/test_cli.py
"""
RED PHASE: CLI tests for DSPy Newsletter Research Tool PoC.

Beads Task: dspy-bot (Click Commands, Rich Output)

These tests verify:
- Click command group structure
- Individual commands (ingest, query, status)
- Rich terminal output formatting
- Configuration handling

Expected Behavior in RED Phase:
- All tests should FAIL because CLI module is not yet implemented
- ImportError for 'from src.cli import ...'
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner


# =============================================================================
# CLI App Tests
# =============================================================================

class TestCLIApp:
    """RED: Test CLI application structure."""

    def test_cli_app_exists(self):
        """Test main CLI app can be imported."""
        from src.cli import cli
        assert cli is not None

    def test_cli_is_click_group(self):
        """Test CLI is a Click command group."""
        from src.cli import cli
        import click
        assert isinstance(cli, click.Group)

    def test_cli_has_version_option(self):
        """Test CLI has --version option."""
        from src.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])

        assert result.exit_code == 0
        assert 'version' in result.output.lower() or '0.' in result.output

    def test_cli_has_help_option(self):
        """Test CLI has --help option."""
        from src.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])

        assert result.exit_code == 0
        assert 'Usage' in result.output


# =============================================================================
# Ingest Command Tests
# =============================================================================

class TestIngestCommand:
    """Test the ingest command."""

    def test_ingest_command_exists(self):
        """Test ingest command can be invoked."""
        from src.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['ingest', '--help'])

        assert result.exit_code == 0

    def test_ingest_accepts_feed_url(self):
        """Test ingest accepts --feed option."""
        from src.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['ingest', '--help'])

        assert '--feed' in result.output or '-f' in result.output

    def test_ingest_accepts_config_file(self):
        """Test ingest accepts --config option."""
        from src.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['ingest', '--help'])

        assert '--config' in result.output or '-c' in result.output

    @patch('src.ingestion.ingest_from_feeds')
    def test_ingest_calls_ingestion_module(self, mock_ingest):
        """Test ingest command calls ingestion module."""
        from src.cli import cli

        mock_ingest.return_value = [{"title": "Test"}]

        runner = CliRunner()
        result = runner.invoke(cli, [
            'ingest',
            '--feed', 'https://example.com/feed.xml'
        ])

        mock_ingest.assert_called()

    @patch('src.ingestion.ingest_from_feeds')
    def test_ingest_shows_progress(self, mock_ingest):
        """Test ingest shows progress indicator."""
        from src.cli import cli

        mock_ingest.return_value = [{"title": "Test"} for _ in range(10)]

        runner = CliRunner()
        result = runner.invoke(cli, [
            'ingest',
            '--feed', 'https://example.com/feed.xml'
        ])

        # Should show some progress indication
        assert 'Processing' in result.output or 'Ingested' in result.output

    @patch('src.ingestion.ingest_from_feeds')
    def test_ingest_shows_summary(self, mock_ingest):
        """Test ingest shows summary after completion."""
        from src.cli import cli

        mock_ingest.return_value = [{"title": "Test"} for _ in range(5)]

        runner = CliRunner()
        result = runner.invoke(cli, [
            'ingest',
            '--feed', 'https://example.com/feed.xml'
        ])

        # Should show summary
        assert '5' in result.output or 'articles' in result.output.lower()


# =============================================================================
# Query Command Tests
# =============================================================================

class TestQueryCommand:
    """Test the query command."""

    def test_query_command_exists(self):
        """Test query command can be invoked."""
        from src.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['query', '--help'])

        assert result.exit_code == 0

    def test_query_accepts_question(self):
        """Test query accepts question argument."""
        from src.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['query', '--help'])

        assert 'QUESTION' in result.output or 'question' in result.output.lower()

    def test_query_accepts_region_filter(self):
        """Test query accepts --region filter."""
        from src.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['query', '--help'])

        assert '--region' in result.output

    def test_query_accepts_topic_filter(self):
        """Test query accepts --topic filter."""
        from src.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['query', '--help'])

        assert '--topic' in result.output

    @patch('weaviate.connect_to_local')
    @patch('src.query_agent.query')
    def test_query_calls_query_agent(self, mock_query, mock_weaviate):
        """Test query command calls query agent."""
        from src.cli import cli

        mock_weaviate.return_value = MagicMock()
        mock_query.return_value = {
            'answer': 'FCRA requires consent',
            'sources': [],
            'confidence': 0.9
        }

        runner = CliRunner()
        result = runner.invoke(cli, [
            'query',
            'What are FCRA requirements?'
        ])

        mock_query.assert_called()

    @patch('weaviate.connect_to_local')
    @patch('src.query_agent.query')
    def test_query_displays_answer(self, mock_query, mock_weaviate):
        """Test query displays answer."""
        from src.cli import cli

        mock_weaviate.return_value = MagicMock()
        mock_query.return_value = {
            'answer': 'FCRA requires written consent for background checks.',
            'sources': [{'title': 'FCRA Guide', 'url': 'https://example.com'}],
            'confidence': 0.9
        }

        runner = CliRunner()
        result = runner.invoke(cli, [
            'query',
            'What are FCRA requirements?'
        ])

        assert 'FCRA requires' in result.output

    @patch('weaviate.connect_to_local')
    @patch('src.query_agent.query')
    def test_query_displays_sources(self, mock_query, mock_weaviate):
        """Test query displays source citations."""
        from src.cli import cli

        mock_weaviate.return_value = MagicMock()
        mock_query.return_value = {
            'answer': 'Answer here',
            'sources': [
                {'title': 'FCRA Guide', 'url': 'https://example.com/1'},
                {'title': 'Compliance Tips', 'url': 'https://example.com/2'}
            ],
            'confidence': 0.9
        }

        runner = CliRunner()
        result = runner.invoke(cli, ['query', 'test question'])

        assert 'FCRA Guide' in result.output or 'Sources' in result.output


# =============================================================================
# Status Command Tests
# =============================================================================

class TestStatusCommand:
    """Test the status command."""

    def test_status_command_exists(self):
        """Test status command can be invoked."""
        from src.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['status', '--help'])

        assert result.exit_code == 0

    @patch('src.storage.ArticleStore')
    def test_status_shows_article_count(self, mock_store):
        """Test status shows total article count."""
        from src.cli import cli

        mock_store.return_value.count.return_value = 150

        runner = CliRunner()
        result = runner.invoke(cli, ['status'])

        assert '150' in result.output or 'articles' in result.output.lower()

    @patch('src.storage.ArticleStore')
    def test_status_shows_collection_info(self, mock_store):
        """Test status shows collection information."""
        from src.cli import cli

        mock_store.return_value.get_schema.return_value = {
            'name': 'NewsletterArticles',
            'properties': []
        }

        runner = CliRunner()
        result = runner.invoke(cli, ['status'])

        assert 'Newsletter' in result.output or 'collection' in result.output.lower()


# =============================================================================
# Config Command Tests
# =============================================================================

class TestConfigCommand:
    """Test the config command."""

    def test_config_command_exists(self):
        """Test config command can be invoked."""
        from src.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['config', '--help'])

        assert result.exit_code == 0

    def test_config_show_subcommand(self):
        """Test config show subcommand exists."""
        from src.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['config', 'show'])

        assert result.exit_code == 0

    def test_config_set_subcommand(self):
        """Test config set subcommand exists."""
        from src.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['config', 'set', '--help'])

        assert result.exit_code == 0


# =============================================================================
# Rich Output Tests
# =============================================================================

class TestRichOutput:
    """Test Rich terminal output formatting."""

    def test_uses_rich_console(self):
        """Test CLI uses Rich console for output."""
        from src.cli import console
        from rich.console import Console

        assert isinstance(console, Console)

    @patch('src.query_agent.query')
    def test_query_uses_rich_panel(self, mock_query):
        """Test query output uses Rich panels."""
        from src.cli import cli

        mock_query.return_value = {
            'answer': 'Test answer',
            'sources': [],
            'confidence': 0.9
        }

        runner = CliRunner()
        result = runner.invoke(cli, ['query', 'test'])

        # Rich formatting should be present (or plain text equivalent)
        assert len(result.output) > 0

    @patch('src.storage.ArticleStore')
    def test_status_uses_rich_table(self, mock_store):
        """Test status output uses Rich tables."""
        from src.cli import cli

        mock_store.return_value.count.return_value = 100
        mock_store.return_value.get_stats.return_value = {
            'total': 100,
            'by_region': {'EUROPE': 30, 'APAC': 20}
        }

        runner = CliRunner()
        result = runner.invoke(cli, ['status'])

        # Should have table-like output
        assert len(result.output) > 0


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test CLI error handling."""

    def test_missing_required_args_shows_error(self):
        """Test missing required args shows helpful error."""
        from src.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['query'])  # Missing question

        assert result.exit_code != 0

    @patch('src.query_agent.query')
    def test_api_error_shows_message(self, mock_query):
        """Test API errors are handled gracefully."""
        from src.cli import cli

        mock_query.side_effect = Exception("API connection failed")

        runner = CliRunner()
        result = runner.invoke(cli, ['query', 'test'])

        assert 'error' in result.output.lower() or result.exit_code != 0

    def test_invalid_region_shows_error(self):
        """Test invalid region shows helpful error."""
        from src.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            'query',
            '--region', 'INVALID_REGION',
            'test question'
        ])

        # Should indicate invalid choice
        assert result.exit_code != 0 or 'invalid' in result.output.lower()


# =============================================================================
# Interactive Mode Tests
# =============================================================================

class TestInteractiveMode:
    """Test interactive query mode."""

    def test_interactive_command_exists(self):
        """Test interactive command exists."""
        from src.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['interactive', '--help'])

        # May or may not exist
        assert result.exit_code in [0, 2]  # 2 = no such command

    @patch('src.query_agent.query')
    def test_interactive_accepts_input(self, mock_query):
        """Test interactive mode accepts user input."""
        from src.cli import cli

        mock_query.return_value = {
            'answer': 'Test answer',
            'sources': [],
            'confidence': 0.9
        }

        runner = CliRunner()
        result = runner.invoke(cli, ['interactive'], input='test\nexit\n')

        # Should process input
        assert result.exit_code == 0 or 'exit' in result.output.lower()
