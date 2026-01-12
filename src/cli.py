# src/cli.py
"""CLI module for DSPy Newsletter Research Tool.

This module provides:
- cli: Main Click command group
- console: Rich console for output
- ingest: Command to ingest RSS feeds
- query: Command to query the database
- status: Command to show system status
- config: Command group for configuration
- interactive: Interactive query mode

Beads Task: dspy-bot
"""

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing import Optional, List

# Create Rich console for output
console = Console()


# =============================================================================
# CLI Configuration
# =============================================================================

VERSION = "0.1.0"

VALID_REGIONS = [
    'AFRICA_ME', 'APAC', 'EUROPE', 'N_AMERICA_CARIBBEAN', 'S_AMERICA', 'WORLDWIDE'
]

VALID_TOPICS = [
    'REGULATORY', 'CRIMINAL_RECORDS', 'CREDENTIALS', 'IMMIGRATION',
    'M_AND_A', 'TECHNOLOGY', 'EVENTS', 'COURT_CASES'
]


# =============================================================================
# Main CLI Group
# =============================================================================

@click.group()
@click.version_option(version=VERSION, prog_name='newsletter-research')
def cli():
    """DSPy Newsletter Research Tool.

    A tool for ingesting, classifying, and querying background screening
    industry news articles using DSPy.
    """
    pass


# =============================================================================
# Ingest Command
# =============================================================================

@cli.command()
@click.option('--feed', '-f', 'feeds', multiple=True, help='RSS feed URL to ingest')
@click.option('--config', '-c', 'config_file', type=click.Path(exists=True),
              help='Configuration file with feed list')
@click.option('--extract-content/--no-extract-content', default=False,
              help='Extract full article content')
def ingest(feeds: tuple, config_file: Optional[str], extract_content: bool):
    """Ingest articles from RSS feeds.

    Fetches articles from specified RSS feeds, filters for relevance,
    classifies by region and topic, and stores in the vector database.

    Examples:
        newsletter-research ingest --feed https://example.com/feed.xml
        newsletter-research ingest --config feeds.yaml
    """
    from src import ingestion

    # Collect feed URLs
    feed_urls = list(feeds)

    # Load from config file if provided
    if config_file:
        try:
            import yaml
            with open(config_file) as f:
                config = yaml.safe_load(f)
                if 'feeds' in config:
                    feed_urls.extend(config['feeds'])
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/red]")
            raise click.Abort()

    if not feed_urls:
        console.print("[yellow]No feeds specified. Use --feed or --config option.[/yellow]")
        raise click.Abort()

    # Ingest articles
    console.print(f"[blue]Processing {len(feed_urls)} feed(s)...[/blue]")

    try:
        articles = ingestion.ingest_from_feeds(
            feed_urls,
            extract_content=extract_content
        )

        # Show summary
        console.print(Panel(
            f"[green]Ingested {len(articles)} articles[/green]\n"
            f"From {len(feed_urls)} feed(s)",
            title="Ingestion Complete"
        ))

    except Exception as e:
        console.print(f"[red]Error during ingestion: {e}[/red]")
        raise click.Abort()


# =============================================================================
# Query Command
# =============================================================================

@cli.command()
@click.argument('question')
@click.option('--region', type=click.Choice(VALID_REGIONS, case_sensitive=False),
              help='Filter by geographic region')
@click.option('--topic', type=click.Choice(VALID_TOPICS, case_sensitive=False),
              help='Filter by topic category')
@click.option('--max-sources', default=5, help='Maximum number of sources to cite')
def query(question: str, region: Optional[str], topic: Optional[str], max_sources: int):
    """Query the newsletter database.

    Ask questions about background screening industry news and regulations.
    Results are synthesized from relevant articles in the database.

    Examples:
        newsletter-research query "What are the latest FCRA changes?"
        newsletter-research query "GDPR impact on screening" --region EUROPE
    """
    from src import query_agent

    # Build filters
    filters = {}
    if region:
        filters['region'] = region.upper()
    if topic:
        filters['topics'] = [topic.upper()]

    try:
        result = query_agent.query(
            question,
            filters=filters if filters else None,
            max_sources=max_sources
        )

        # Display answer
        console.print(Panel(
            result['answer'],
            title="Answer",
            border_style="green"
        ))

        # Display sources
        if result.get('sources'):
            console.print("\n[bold]Sources:[/bold]")
            for i, source in enumerate(result['sources'], 1):
                title = source.get('title', 'Unknown')
                url = source.get('url', '')
                console.print(f"  [{i}] {title}")
                if url:
                    console.print(f"      [dim]{url}[/dim]")

        # Display confidence
        confidence = result.get('confidence', 0)
        console.print(f"\n[dim]Confidence: {confidence:.0%}[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


# =============================================================================
# Status Command
# =============================================================================

@cli.command()
def status():
    """Show system status.

    Displays information about the newsletter database including
    article counts, collection schema, and connection status.
    """
    from src import storage

    try:
        store = storage.ArticleStore()

        # Get article count - ensure it's an int
        try:
            count = store.count()
            if not isinstance(count, (int, float)):
                count = 0
            count = int(count)
        except (TypeError, ValueError, Exception):
            count = 0

        # Get schema info - ensure it's a string
        try:
            schema = store.get_schema()
            if isinstance(schema, dict):
                collection_name = schema.get('name', 'NewsletterArticles')
            else:
                collection_name = 'NewsletterArticles'
            if not isinstance(collection_name, str):
                collection_name = 'NewsletterArticles'
        except Exception:
            collection_name = 'NewsletterArticles'
            schema = {}

        # Get stats if available
        try:
            stats = store.get_stats()
            if not isinstance(stats, dict):
                stats = {'total': count}
        except Exception:
            stats = {'total': count}

        # Display status table
        table = Table(title="Newsletter Research Tool Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Collection", str(collection_name))
        table.add_row("Total Articles", str(count))

        # Add region breakdown if available
        if isinstance(stats, dict) and 'by_region' in stats:
            by_region = stats['by_region']
            if isinstance(by_region, dict):
                for region, rcount in by_region.items():
                    table.add_row(f"  {region}", str(rcount))

        console.print(table)

        # Close store
        try:
            store.close()
        except Exception:
            pass

    except Exception as e:
        console.print(f"[red]Error getting status: {e}[/red]")
        console.print("[yellow]Make sure Weaviate is running and accessible.[/yellow]")


# =============================================================================
# Config Command Group
# =============================================================================

@cli.group()
def config():
    """Manage configuration settings.

    View and modify tool configuration including API keys,
    database settings, and default options.
    """
    pass


@config.command('show')
def config_show():
    """Show current configuration."""
    import os

    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Source", style="dim")

    # Check environment variables
    settings = [
        ('OPENAI_API_KEY', os.environ.get('OPENAI_API_KEY', '')),
        ('COHERE_API_KEY', os.environ.get('COHERE_API_KEY', '')),
        ('WEAVIATE_URL', os.environ.get('WEAVIATE_URL', 'http://localhost:8080')),
    ]

    for name, value in settings:
        # Mask API keys
        if 'KEY' in name and value:
            display_value = value[:8] + '...' if len(value) > 8 else '***'
            source = 'environment'
        elif value:
            display_value = value
            source = 'environment'
        else:
            display_value = '[not set]'
            source = '-'

        table.add_row(name, display_value, source)

    console.print(table)


@config.command('set')
@click.argument('key')
@click.argument('value')
def config_set(key: str, value: str):
    """Set a configuration value.

    Sets configuration values. Some values may require environment
    variables to be set permanently in your shell profile.
    """
    import os

    os.environ[key] = value
    console.print(f"[green]Set {key} for this session.[/green]")
    console.print("[dim]Note: To persist, add to your shell profile.[/dim]")


# =============================================================================
# Interactive Command
# =============================================================================

@cli.command()
def interactive():
    """Start interactive query mode.

    Enter an interactive session where you can ask multiple questions
    without restarting the tool. Type 'exit' or 'quit' to leave.
    """
    from src import query_agent

    console.print(Panel(
        "Welcome to Newsletter Research Interactive Mode\n"
        "Type your questions and press Enter.\n"
        "Type 'exit' or 'quit' to leave.",
        title="Interactive Mode"
    ))

    while True:
        try:
            question = console.input("[bold blue]> [/bold blue]")

            if question.lower() in ['exit', 'quit', 'q']:
                console.print("[dim]Goodbye![/dim]")
                break

            if not question.strip():
                continue

            # Process query
            try:
                result = query_agent.query(question)

                console.print(f"\n[green]{result['answer']}[/green]\n")

                if result.get('sources'):
                    console.print("[dim]Sources:[/dim]")
                    for source in result['sources'][:3]:
                        console.print(f"  - {source.get('title', 'Unknown')}")

                console.print()

            except Exception as e:
                console.print(f"[red]Error: {e}[/red]\n")

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    cli()
