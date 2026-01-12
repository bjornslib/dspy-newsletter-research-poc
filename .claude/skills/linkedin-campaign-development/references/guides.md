# Campaign Guides

Combined reference for Perfect Future Customer (PFC) criteria and CSV schema.

---

## Part 1: Perfect Future Customer (PFC) Criteria

### Inclusion Criteria

**Ideal Lead Profile:**
- COO, CEO, or senior executive (Director+) at SME/Mid-sized business
- Company size: 50-500 employees (typical)
- Australia/New Zealand based (primary market)
- Decision-making authority for operations or strategy
- Indicators of growth mindset or transformation interest

**Strong Signals:**
- Recent activity about AI, digital transformation, or innovation
- Hiring for operational or technical roles
- Speaking at industry events
- Published thought leadership
- Awards or recognition for innovation/leadership

### Exclusion Criteria

**Automatic Exclusions:**
- Consultants/advisors (not end customers)
- Competitors (AI consulting, digital agencies)
- Too junior (Manager level without decision authority)
- Too large (Enterprise 5000+ employees)
- Not target geography

**Yellow Flags (Review Carefully):**
- Unclear current role (may have moved)
- No recent activity (profile may be stale)
- Company in obvious distress (layoffs, restructuring)
- Industry misalignment with offer

### Industry Prioritization

**Tier 1 (High Priority):**
- Mining and Resources
- Healthcare and NDIS
- Financial Services
- Professional Services (Legal, Accounting)

**Tier 2 (Good Fit):**
- Manufacturing
- Logistics and Transport
- Retail (mid-sized chains)
- Real Estate/Property

**Tier 3 (Evaluate Case by Case):**
- Education
- Government
- Not-for-profit
- Agriculture

---

## Part 2: CSV Schema

### Primary Output File: campaign_research_audit.csv

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| firstName | string | Lead's first name | "Greg" |
| lastName | string | Lead's last name | "Barclay" |
| company | string | Current company name | "Slater and Gordon Lawyers" |
| title | string | Current job title | "Transformation Director" |
| linkedInUrl | string | Full LinkedIn profile URL | "https://linkedin.com/in/..." |
| personalisedMessage | string | Identity-First message | "Noticed your 10 transformation..." |
| messageRationale | string | Why this approach was chosen | "Led with transformation expertise..." |
| personalisationHooks | string | Enrichment data gathered | "10 certifications including Stanford..." |
| qaStatus | string | Quality assurance status | "approved" / "revision_required" |
| validationNotes | string | Notes from Perplexity/user review | "Fact confirmed, role aligned" |

### Status Tracking File: all_leads_master.csv

| Field | Type | Description | Values |
|-------|------|-------------|--------|
| firstName | string | Lead's first name | - |
| lastName | string | Lead's last name | - |
| company | string | Current company | - |
| status | string | Pipeline status | "new", "enriched", "drafted", "approved", "sent", "responded" |
| phase | string | Current workflow phase | "phase-1" through "phase-5" |
| batch | string | Batch assignment | "batch_001", etc. |
| lastUpdated | string | ISO timestamp | "2025-12-16T10:30:00Z" |
| notes | string | Any tracking notes | - |

### Batch File Format: batch_XXX/leads_enriched.json

```json
[
  {
    "firstName": "Greg",
    "lastName": "Barclay",
    "company": "Slater and Gordon Lawyers",
    "title": "Transformation Director",
    "linkedInUrl": "https://linkedin.com/in/gregbarclay",
    "enrichment": {
      "individualProfile": "10 professional certifications...",
      "recentActivity": "Company underwent CEO transition...",
      "companyContext": "Major Australian law firm...",
      "aiOpportunities": [
        "Predict case complexity from intake",
        "Analyze transformation readiness"
      ]
    }
  }
]
```

### Batch File Format: batch_XXX/messages_draft.json

```json
[
  {
    "firstName": "Greg",
    "lastName": "Barclay",
    "company": "Slater and Gordon Lawyers",
    "title": "Transformation Director",
    "linkedInUrl": "https://linkedin.com/in/gregbarclay",
    "personalisedMessage": "Noticed your 10 transformation certifications...",
    "messageRationale": "Identity-First: Led with professional development identity...",
    "personalisationHooks": "10 professional certifications including Stanford Innovation..."
  }
]
```

---

## Part 3: Field Guidelines

### personalisedMessage Field

**Character Limit:** 300 characters recommended (LinkedIn DM limit is higher, but brevity wins)

**Structure:**
```
{observation}; {AI opportunity question}
```

**DO:**
- Start with "Saw...", "Noticed...", "Congrats on..."
- Use semicolon to connect thoughts
- End with "what if..." or "do you think..." question
- Keep to 2-3 sentences maximum

**DON'T:**
- Use em-dashes (-)
- Stack multiple facts
- Use marketing speak
- Include TM, (R), superscript characters

### messageRationale Field

**Purpose:** Document WHY this message was crafted this way

**Structure:**
```
{approach}: {identity signal used}. {why AI opportunity chosen}. {any special considerations}.
```

**Example:**
```
Identity-First: Led with transformation expertise credentials (Stanford Innovation). AI opportunity aligned with legal workflow optimization during organizational change period. Role-appropriate for Transformation Director.
```

### personalisationHooks Field

**Purpose:** Store all enrichment data for reference and future messages

**Include:**
- All 4 dimensions from Phase 1 enrichment
- Specific facts that could be used
- URLs or sources if available
- Ideas for follow-up messages

**Format:** Pipe-separated or natural language paragraphs

---

## Part 4: Data Hygiene

### Encoding Standards

- Always save CSV as UTF-8
- Check for mojibake characters before export
- Common issues to fix:
  - `Ã¢Â€Â™` → `'`
  - `Ã¢Â€Â"` → `-`
  - `â€™` → `'`

### Name Formatting

- First name: Capitalize first letter
- Last name: Capitalize first letter (handle McName, O'Name correctly)
- No trailing spaces
- No titles (Dr., Mr., etc.) unless specifically required

### Company Formatting

- Use official company name
- Include legal suffix if commonly used (Pty Ltd, Limited)
- No quotes around name
- Handle ampersands consistently (& not "and")

### URL Formatting

- Full URL including https://
- No tracking parameters
- Verify URL resolves before including

---

## Part 5: Quality Gates

### Phase 1 → Phase 2 Gate

Lead MUST have:
- [ ] At least one identity signal captured
- [ ] Company context sufficient for AI opportunity
- [ ] Current/relevant data point

### Phase 2 → Phase 3 Gate

Message MUST have:
- [ ] Single fact (not stacked)
- [ ] Question ending
- [ ] No encoding issues
- [ ] Role-appropriate AI opportunity

### Phase 4 → Phase 5 Gate

Batch MUST have:
- [ ] 100% leads processed
- [ ] Spot-check passed
- [ ] No pattern drift

### Phase 5 → Export Gate

Campaign MUST have:
- [ ] User approval received
- [ ] Revisions applied
- [ ] Encoding verified
- [ ] Export format validated
