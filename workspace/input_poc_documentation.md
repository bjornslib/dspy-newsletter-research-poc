# Coding Brief: Newsletter Article Research PoC

## Context

Building a proof of concept for PreEmploymentDirectory, a B2B publisher serving the background screening industry. They produce newsletters (The Global Background Screener, The Background Buzz) for 2,100+ screening firms worldwide.

Current process is manual: staff search LinkedIn and email subscriptions, dump articles in Dropbox, then use ChatGPT to summarise. We want to automate the research and candidate identification phase.

## What We’re Building

An article research tool that:

1. Monitors relevant sources for background screening industry news
1. Identifies candidate articles worth including in the newsletter
1. Categorises them by region and topic
1. Outputs a shortlist for human review

This is a PoC. Scrappy is fine. We want to test the concept, not build production infrastructure.

## Content Categories

The newsletter organises content into:

**By Region:**

- Africa & Middle East
- Asia Pacific
- Europe
- North America & Caribbean
- South America
- Worldwide (cross-regional or global news)

**By Topic:**

- Regulatory/legal changes (employment law, data protection, privacy)
- Criminal background check requirements
- Education and credential verification
- Immigration and right-to-work
- Industry M&A and company news
- Technology and product announcements
- Conference and event news
- Court cases and legal precedents

## Source Types to Monitor

**Legal and Regulatory:**

- SHRM (shrm.org)
- Law360
- Lexology
- JDSupra
- National Law Review
- Government gazette feeds (where available)
- Data protection authority announcements

**Industry-Specific:**

- PBSA (Professional Background Screening Association)
- Regional screening associations
- Background screening company blogs and press releases
- HR and talent acquisition publications

**General News (filtered for relevance):**

- Reuters, Bloomberg, AP (employment/HR beats)
- Regional business news for international coverage

**Social:**

- LinkedIn posts from industry thought leaders (harder to automate, lower priority)

## What Makes a Good Candidate Article

**Include:**

- Regulatory changes affecting background checks in any jurisdiction
- New laws or amendments to employment screening requirements
- Court decisions impacting screening practices
- Data protection/privacy changes affecting employee data
- Major company announcements (acquisitions, new products, leadership)
- Industry research or survey results
- Changes to criminal record disclosure rules
- Education verification or credential fraud stories
- Immigration and work authorisation updates

**Exclude:**

- Generic HR content not specific to screening
- Opinion pieces without news value
- Press releases that are purely promotional
- Content older than 30 days (unless major regulatory)
- Duplicate coverage of same story

## Suggested Approach

**Option 1: RSS + News API**

- Aggregate RSS feeds from legal/industry sites
- Use NewsAPI or similar for general news with keyword filtering
- Filter and rank by relevance

**Option 2: Web Scraping**

- Target specific high-value sources
- Extract article metadata (title, date, summary, URL)
- Classify by region and topic

**Option 3: Hybrid**

- RSS where available
- Scraping for sites without feeds
- LLM for relevance scoring and categorisation

## Keywords and Phrases

Use these for filtering and relevance scoring:

**Primary:**

- background check, background screening
- employment screening, pre-employment
- criminal record, criminal history
- right to work, work authorisation
- Ban the Box, fair chance
- FCRA (US), GDPR (EU), data protection
- credential verification, education verification

**Secondary:**

- drug testing, drug screening
- reference check
- identity verification
- continuous monitoring
- adverse action
- consumer reporting agency

**Regional Triggers:**

- Country names + “employment law”
- Country names + “background check”
- Specific legislation names (varies by jurisdiction)

## Output Format

For each candidate article:

```
{
  "title": "Article title",
  "source": "Publication name",
  "url": "https://...",
  "published_date": "2025-01-10",
  "region": "Asia Pacific",
  "country": "Australia",
  "topics": ["regulatory", "criminal_records"],
  "relevance_score": 0.85,
  "summary": "2-3 sentence summary of the article",
  "reasoning": "Why this was flagged as relevant"
}
```

Batch output as JSON. We’ll review and feed back on false positives/negatives to tune.

## Technical Constraints

- Python preferred
- Can use OpenAI/Anthropic APIs for relevance scoring and summarisation
- Keep costs reasonable (this is a PoC, not production)
- Don’t need real-time. Daily batch is fine.
- Store raw results somewhere we can inspect (local JSON files fine for now)

## What Success Looks Like

Run it against the past week of news. Get a shortlist of 20-50 candidate articles. Manually review against what actually appeared in their December newsletter.

If we’re catching 70%+ of what they would have found manually, and not drowning in irrelevant noise, we’re in good shape.

## Questions to Resolve

1. Which sources have RSS feeds vs need scraping?
1. Do we need authentication for any legal databases?
1. What’s the acceptable false positive rate? (Better to over-include and let humans filter, or be strict?)
1. Should we attempt to deduplicate coverage of the same story across sources?

## Next Steps After PoC

If this works:

- Add summarisation that matches their editorial style
- Add advertiser category tagging (for the “worth more” angle)
- Build a simple review interface
- Connect to their Dropbox or delivery workflow
