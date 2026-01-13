# Phase 1: Lead Enrichment

## Purpose

Gather the 4 dimensions of data needed to write Identity-First messages. This phase is about COLLECTING information, not writing messages yet.

## The 4 Enrichment Dimensions

### 1. Individual Profile (WHO they are)

**Primary Sources:**
- LinkedIn About section (their own words describing themselves)
- LinkedIn recommendations (what others say about them)
- Career trajectory (pivots that reveal values)
- Self-descriptions and philosophy statements

**What to Capture:**
- Their self-defined identity ("captain, coach and cheerleader")
- Values revealed by career choices
- Reputation themes from recommendations
- Unique positioning (only person doing X)

**Identity Signals Hierarchy (Best to Worst):**
1. **Their Own Words** - Quotes from About section
2. **Their Reputation** - What recommendations say
3. **Their Purpose** - Career pivots revealing values
4. **Unique Achievements** - Only they can claim
5. **Recent Activity** - Posts, articles, engagement

### 2. Recent Activity (CURRENT focus)

**What to Look For:**
- Posts in last 30-90 days
- Articles published
- Job changes
- Hiring activity
- Speaking engagements
- Award announcements

**Why This Matters:**
Current activity signals what's top of mind. A CEO posting about AI integration is more receptive than one posting about golf.

### 3. Company Context (WHERE they operate)

**What to Capture:**
- Company size and stage
- Industry challenges
- Recent news (acquisitions, expansions, challenges)
- Competitive positioning
- Geographic reach

**Avoid:**
- Stacking multiple company facts
- Generic industry observations
- Assumptions without evidence

### 4. AI Opportunity Mapping (BRIDGE to offer)

**The Question:**
"Given who they are and what they're dealing with, what AI opportunity would resonate?"

**Good AI Opportunities:**
- Specific to their domain
- Connected to their current challenges
- Framed as questions, not assertions
- Make them the hero who could bring it to life

**Bad AI Opportunities:**
- Generic ("improve efficiency")
- Assumes their role incorrectly
- Multiple stacked opportunities
- Marketing speak ("unlock", "leverage")

## Research Workflow

### Step 1: LinkedIn Profile Deep Read
```
For each lead:
1. Read About section completely
2. Note self-descriptions and philosophy
3. Check recommendations (3-5 most recent)
4. Note any unique positioning
```

### Step 2: Activity Scan
```
1. Check "Activity" tab for recent posts
2. Look for articles published
3. Note engagement patterns
4. Flag any hiring posts
```

### Step 3: Company Research
```
1. Company LinkedIn page
2. Recent news (Perplexity: "[Company] news 2025")
3. Size, industry, geographic focus
4. Any known challenges or opportunities
```

### Step 4: Opportunity Mapping
```
Given:
- Their identity signals
- Their current focus
- Their company context

Ask: "What AI opportunity would make them think 'they really get me'?"
```

## Subagent Delegation Pattern

For batch enrichment, use general-purpose agents:

```markdown
Research leads [X] through [Y] from the lead list.

For each lead, gather:
1. Individual Profile:
   - Self-description from About section
   - Key themes from recommendations
   - Career pivot insights

2. Recent Activity:
   - Posts/articles in last 90 days
   - Job changes or announcements

3. Company Context:
   - Company size and industry
   - Recent news or developments

4. AI Opportunity Ideas:
   - 2-3 potential angles based on findings

Save results incrementally to: output/batch_XXX/leads_enriched.json

Use this format:
{
  "firstName": "...",
  "lastName": "...",
  "company": "...",
  "title": "...",
  "enrichment": {
    "individualProfile": "...",
    "recentActivity": "...",
    "companyContext": "...",
    "aiOpportunities": ["...", "..."]
  }
}
```

## Validation Checklist

Before moving to Phase 2, verify each lead has:

- [ ] Identity signal (their own words, reputation, or unique achievement)
- [ ] At least one current or relevant data point
- [ ] Company context sufficient for AI opportunity
- [ ] At least one AI opportunity idea mapped

## Output Format

```json
{
  "firstName": "Greg",
  "lastName": "Barclay",
  "company": "Slater and Gordon Lawyers",
  "title": "Transformation Director",
  "enrichment": {
    "individualProfile": "10 professional certifications including Stanford Innovation; deep expertise in organizational transformation; recommended as 'change management leader'",
    "recentActivity": "Company underwent CEO transition March 2024; transformation initiatives ongoing",
    "companyContext": "Major Australian law firm; significant organizational change period",
    "aiOpportunities": [
      "Predict case complexity from intake for routing",
      "Analyze transformation readiness across departments",
      "Surface patterns in legal workflow bottlenecks"
    ]
  }
}
```

## Common Mistakes

1. **Stacking facts** - Pick ONE compelling fact, not three mediocre ones
2. **Company over person** - Facts should be about THEM, not their employer
3. **Boring facts** - "9 years as COO" is boring; "known for zero-turnover teams" is compelling
4. **Unverifiable claims** - If you can't easily deduce it from their profile, don't use it
5. **Multiple AI opportunities in message** - Map several, but use only ONE in the message

## Next Phase

Once enrichment is validated, proceed to **Phase 2: Message Crafting** where we apply Identity-First methodology to write the actual messages.
