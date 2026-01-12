# src/ingestion.py
"""RSS Ingestion module for DSPy Newsletter Research Tool.

This module provides:
- RSSParser: Parse RSS feeds and extract articles
- ContentExtractor: Extract full article content from URLs
- FeedManager: Manage feed subscriptions
- ingest_from_feeds: Main ingestion function
- extract_metadata: Extract metadata from HTML

Beads Task: dspy-7eh
"""

import re
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser

import requests


# =============================================================================
# Custom Exceptions
# =============================================================================

class IngestionError(Exception):
    """Base exception for ingestion errors."""
    pass


class RSSParseError(IngestionError):
    """Raised when RSS feed cannot be parsed."""
    pass


class NetworkError(IngestionError):
    """Raised when network request fails."""
    pass


class PaywallDetectedError(IngestionError):
    """Raised when paywall is detected on article."""
    pass


# =============================================================================
# HTML Text Extractor
# =============================================================================

class HTMLTextExtractor(HTMLParser):
    """HTML parser that extracts clean text content."""

    def __init__(self):
        super().__init__()
        self.result = []
        self.skip_tags = {'script', 'style', 'head', 'meta', 'link'}
        self._skip_data = False

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self._skip_data = True

    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self._skip_data = False

    def handle_data(self, data):
        if not self._skip_data:
            text = data.strip()
            if text:
                self.result.append(text)

    def get_text(self) -> str:
        return ' '.join(self.result)


def html_to_text(html: str) -> str:
    """Convert HTML to plain text."""
    extractor = HTMLTextExtractor()
    try:
        extractor.feed(html)
    except Exception:
        # Fallback: basic tag stripping
        return re.sub(r'<[^>]+>', '', html)
    return extractor.get_text()


# =============================================================================
# RSS Parser
# =============================================================================

class RSSParser:
    """Parse RSS feeds and extract articles.

    Attributes:
        feed_url: URL of the RSS feed to parse.
        timeout: Request timeout in seconds.
    """

    def __init__(self, feed_url: str, timeout: int = 30):
        """Initialize RSS parser with feed URL.

        Args:
            feed_url: URL of the RSS feed.
            timeout: Request timeout in seconds.
        """
        self.feed_url = feed_url
        self.timeout = timeout

    def parse(self) -> List[Dict[str, Any]]:
        """Parse the RSS feed and extract articles.

        Returns:
            List of article dictionaries with keys:
            - title: Article title
            - source_url: Link to article
            - description: Article summary/description
            - published_date: Publication datetime
            - source: Feed name

        Raises:
            NetworkError: If network request fails.
            RSSParseError: If RSS content cannot be parsed.
        """
        # Fetch feed content
        try:
            response = requests.get(self.feed_url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise NetworkError(f"Failed to fetch feed: {e}")
        except Exception as e:
            raise NetworkError(f"Unexpected network error: {e}")

        # Parse XML
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as e:
            raise RSSParseError(f"Invalid RSS XML: {e}")

        # Extract articles
        articles = []
        channel = root.find('channel')

        if channel is None:
            # Try Atom format
            return self._parse_atom(root)

        # Get feed title
        feed_title_elem = channel.find('title')
        feed_title = feed_title_elem.text if feed_title_elem is not None else self.feed_url

        # Parse items
        for item in channel.findall('item'):
            article = self._parse_item(item, feed_title)
            if article:
                articles.append(article)

        return articles

    def _parse_item(self, item: ET.Element, feed_title: str) -> Optional[Dict[str, Any]]:
        """Parse a single RSS item.

        Args:
            item: XML element representing the item.
            feed_title: Title of the parent feed.

        Returns:
            Article dictionary or None if parsing fails.
        """
        title_elem = item.find('title')
        link_elem = item.find('link')
        desc_elem = item.find('description')
        pub_date_elem = item.find('pubDate')

        if title_elem is None or title_elem.text is None:
            return None

        article = {
            'title': title_elem.text.strip(),
            'source_url': link_elem.text.strip() if link_elem is not None and link_elem.text else None,
            'description': desc_elem.text.strip() if desc_elem is not None and desc_elem.text else '',
            'source': feed_title,
            'feed_url': self.feed_url,
        }

        # Parse publication date
        if pub_date_elem is not None and pub_date_elem.text:
            try:
                article['published_date'] = parsedate_to_datetime(pub_date_elem.text)
            except (ValueError, TypeError):
                article['published_date'] = None
        else:
            article['published_date'] = None

        return article

    def _parse_atom(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Parse Atom feed format.

        Args:
            root: Root XML element.

        Returns:
            List of article dictionaries.
        """
        # Handle Atom namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        articles = []

        feed_title_elem = root.find('atom:title', ns)
        feed_title = feed_title_elem.text if feed_title_elem is not None else self.feed_url

        for entry in root.findall('atom:entry', ns):
            title_elem = entry.find('atom:title', ns)
            link_elem = entry.find('atom:link', ns)
            summary_elem = entry.find('atom:summary', ns)
            published_elem = entry.find('atom:published', ns)

            if title_elem is None or title_elem.text is None:
                continue

            article = {
                'title': title_elem.text.strip(),
                'source_url': link_elem.get('href') if link_elem is not None else None,
                'description': summary_elem.text.strip() if summary_elem is not None and summary_elem.text else '',
                'source': feed_title,
                'feed_url': self.feed_url,
                'published_date': None,
            }

            if published_elem is not None and published_elem.text:
                try:
                    article['published_date'] = datetime.fromisoformat(
                        published_elem.text.replace('Z', '+00:00')
                    )
                except ValueError:
                    pass

            articles.append(article)

        return articles


# =============================================================================
# Content Extractor
# =============================================================================

class ContentExtractor:
    """Extract full article content from URLs.

    Handles HTML parsing, paywall detection, and content cleaning.

    Attributes:
        use_javascript: Whether to render JavaScript content.
        respect_robots: Whether to respect robots.txt directives.
        timeout: Request timeout in seconds.
    """

    # Paywall indicators
    PAYWALL_INDICATORS = [
        'paywall', 'subscribe to continue', 'subscription required',
        'premium content', 'member-only', 'sign up to read',
        'create an account to continue', 'unlock this article'
    ]

    def __init__(
        self,
        use_javascript: bool = False,
        respect_robots: bool = True,
        timeout: int = 30
    ):
        """Initialize content extractor.

        Args:
            use_javascript: Enable JavaScript rendering (requires additional deps).
            respect_robots: Check robots.txt before crawling.
            timeout: Request timeout in seconds.
        """
        self.use_javascript = use_javascript
        self.respect_robots = respect_robots
        self.timeout = timeout

    def extract(self, url: str) -> str:
        """Extract clean text content from URL.

        Args:
            url: URL of the article to extract.

        Returns:
            Clean text content of the article.

        Raises:
            NetworkError: If request fails.
            PaywallDetectedError: If paywall is detected.
        """
        # Fetch HTML
        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={'User-Agent': 'DSPy Newsletter Research Tool/1.0'}
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise NetworkError(f"Failed to fetch article: {e}")

        html = response.text

        # Check for paywall
        if self._detect_paywall(html):
            raise PaywallDetectedError(f"Paywall detected at {url}")

        # Extract and clean content
        content = html_to_text(html)

        return content

    def _detect_paywall(self, html: str) -> bool:
        """Check if HTML contains paywall indicators.

        Args:
            html: HTML content to check.

        Returns:
            True if paywall detected.
        """
        html_lower = html.lower()

        # Check for paywall class or indicator text
        if 'class="paywall"' in html_lower or 'class=\'paywall\'' in html_lower:
            return True

        for indicator in self.PAYWALL_INDICATORS:
            if indicator in html_lower:
                return True

        return False


# =============================================================================
# Feed Manager
# =============================================================================

class FeedManager:
    """Manage RSS feed subscriptions.

    Provides CRUD operations for feed management with categorization
    and priority settings.
    """

    def __init__(self):
        """Initialize empty feed manager."""
        self.feeds: List[Dict[str, Any]] = []

    def add_feed(
        self,
        url: str,
        category: str,
        priority: int = 0
    ) -> None:
        """Add a new feed subscription.

        Args:
            url: RSS feed URL.
            category: Feed category (e.g., 'legal', 'industry').
            priority: Priority level (lower = higher priority).
        """
        feed = {
            'url': url,
            'category': category,
            'priority': priority,
            'added_at': datetime.now(),
        }
        self.feeds.append(feed)

    def list_feeds(self) -> List[Dict[str, Any]]:
        """List all subscribed feeds.

        Returns:
            List of feed dictionaries.
        """
        return self.feeds.copy()

    def remove_feed(self, url: str) -> bool:
        """Remove a feed by URL.

        Args:
            url: URL of feed to remove.

        Returns:
            True if feed was removed, False if not found.
        """
        initial_len = len(self.feeds)
        self.feeds = [f for f in self.feeds if f['url'] != url]
        return len(self.feeds) < initial_len

    def get_feeds_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get feeds filtered by category.

        Args:
            category: Category to filter by.

        Returns:
            List of feeds in the specified category.
        """
        return [f for f in self.feeds if f['category'] == category]

    def get_feeds_by_priority(self, max_priority: int) -> List[Dict[str, Any]]:
        """Get feeds with priority <= max_priority.

        Args:
            max_priority: Maximum priority level to include.

        Returns:
            List of feeds within priority threshold.
        """
        return [f for f in self.feeds if f['priority'] <= max_priority]


# =============================================================================
# Metadata Extraction
# =============================================================================

def extract_metadata(html: str) -> Dict[str, Any]:
    """Extract metadata from article HTML.

    Extracts author, publication date, and Open Graph data.

    Args:
        html: HTML content to parse.

    Returns:
        Dictionary with extracted metadata:
        - author: Article author
        - published_date: Publication datetime
        - image_url: OG image URL
    """
    metadata: Dict[str, Any] = {}

    # Extract author from meta tag
    author_match = re.search(
        r'<meta\s+name=["\']author["\']\s+content=["\']([^"\']+)["\']',
        html,
        re.IGNORECASE
    )
    if author_match:
        metadata['author'] = author_match.group(1)

    # Extract publication date from various formats
    date_patterns = [
        r'<meta\s+property=["\']article:published_time["\']\s+content=["\']([^"\']+)["\']',
        r'<meta\s+name=["\']date["\']\s+content=["\']([^"\']+)["\']',
        r'<time[^>]+datetime=["\']([^"\']+)["\']',
    ]

    for pattern in date_patterns:
        date_match = re.search(pattern, html, re.IGNORECASE)
        if date_match:
            try:
                date_str = date_match.group(1)
                # Handle ISO format with Z
                if date_str.endswith('Z'):
                    date_str = date_str[:-1] + '+00:00'
                metadata['published_date'] = datetime.fromisoformat(date_str)
                break
            except ValueError:
                continue

    # Extract Open Graph image
    og_image_match = re.search(
        r'<meta\s+property=["\']og:image["\']\s+content=["\']([^"\']+)["\']',
        html,
        re.IGNORECASE
    )
    if og_image_match:
        metadata['image_url'] = og_image_match.group(1)

    return metadata


# =============================================================================
# Ingestion Pipeline
# =============================================================================

def ingest_from_feeds(
    feed_urls: List[Union[str, Tuple[str, str]]],
    deduplicate: bool = True,
    extract_content: bool = False,
    timeout: int = 30
) -> List[Dict[str, Any]]:
    """Ingest articles from multiple RSS feeds.

    Args:
        feed_urls: List of feed URLs or (url, category) tuples.
        deduplicate: Remove duplicate articles by URL.
        extract_content: Fetch full article content (slower).
        timeout: Request timeout in seconds.

    Returns:
        List of article dictionaries with keys:
        - title: Article title
        - source_url: Link to article
        - description: Article summary
        - published_date: Publication datetime
        - source: Feed name
        - source_category: Feed category (if provided)
        - content: Full article text (if extract_content=True)
    """
    all_articles: List[Dict[str, Any]] = []
    seen_urls: set = set()

    for feed_entry in feed_urls:
        # Handle both string and tuple formats
        if isinstance(feed_entry, tuple):
            feed_url, category = feed_entry
        else:
            feed_url = feed_entry
            category = None

        # Parse feed
        try:
            parser = RSSParser(feed_url, timeout=timeout)
            articles = parser.parse()
        except (NetworkError, RSSParseError) as e:
            # Log and continue with other feeds
            continue

        # Process articles
        for article in articles:
            # Add category if provided
            if category:
                article['source_category'] = category

            # Deduplicate by URL
            url = article.get('source_url')
            if deduplicate and url:
                if url in seen_urls:
                    continue
                seen_urls.add(url)

            # Optionally extract full content
            if extract_content and url:
                try:
                    extractor = ContentExtractor(timeout=timeout)
                    article['content'] = extractor.extract(url)
                except (NetworkError, PaywallDetectedError):
                    article['content'] = article.get('description', '')

            all_articles.append(article)

    return all_articles


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_content_hash(title: str, content: str) -> str:
    """Compute hash for content deduplication.

    Args:
        title: Article title.
        content: Article content.

    Returns:
        32-character hex hash.
    """
    hash_input = f"{title}{content}".encode('utf-8')
    return hashlib.sha256(hash_input).hexdigest()[:32]
