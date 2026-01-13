---
name: linkedin-campaign-development
description: This skill should be used when the user asks to "create a LinkedIn campaign", "develop personalized outreach", "write LinkedIn messages", "research leads for a campaign", "create Sales Navigator outreach", "build B2B outreach", "personalize messages for leads", or mentions LinkedIn campaign development, lead personalization, identity-first messaging, or Sales Navigator campaigns.
---

# LinkedIn Campaign Development

## Purpose

End-to-end LinkedIn campaign development: from lead enrichment through to campaign-ready personalized messages. This skill embeds Identity-First messaging as the core methodology, not an afterthought.

## The Identity-First Philosophy

Most outreach messages focus on WHAT people did (achievements, awards, milestones). These prove research was done. But messages that resonate deeply focus on WHO they are (values, mission, philosophy, the thread connecting their choices). These create CONNECTION.

**The Pattern:**
```
{identity insight}; {warm validation}; {identity-aligned AI opportunity}
```

This philosophy is embedded throughout the workflow, not applied as a final polish.

## 5-Phase Workflow Overview

| Phase | Focus | Output |
|-------|-------|--------|
| **1. Enrichment** | Gather 4 dimensions + validate with agent | Verified lead data |
| **2. Message Crafting** | Write Identity-First messages | Draft messages |
| **3. Quality Assurance** | Perplexity full-context check + pilot review | Approved approach |
| **4. Scale** | Apply approved approach to remaining leads | All messages drafted |
| **5. Finalize** | Merge + final user review partnership | Campaign-ready messages |

## Critical Rules

### File Storage (Mandatory)
```
output/
├── batch_001/
│   ├── leads_extracted.json
│   ├── leads_enriched.json
│   └── messages_draft.json
├── batch_002/
│   └── ...
├── campaign_research_audit.csv      ← Final merged output
└── all_leads_master.csv             ← Status tracking
```

### Incremental Save Protocol
After processing each lead, immediately save to prevent data loss:
1. Read existing batch file
2. Append new lead data
3. Write back to file

This ensures no data loss if context runs out mid-batch.

### Subagent Delegation
- Use `general-purpose` agents for batch enrichment research
- Use `haiku` model for validation tasks (faster, cheaper)
- Never process more than 5 leads without saving

## Phase Navigation

Load the appropriate phase reference file when working on that phase:

### Phase 1: Enrichment
**When:** Starting a new campaign or processing new leads
**Load:** `references/phases/phase-1-enrichment.md`
**Focus:** Gather 4 dimensions (Individual, Activity, Company, Opportunities) and validate

### Phase 2: Message Crafting
**When:** Enrichment complete, ready to write messages
**Load:** `references/phases/phase-2-message-crafting.md`
**Focus:** Write Identity-First messages using the core pattern

### Phase 3: Quality Assurance
**When:** Draft messages ready for validation
**Load:** `references/phases/phase-3-quality-assurance.md`
**Focus:** Perplexity sense-check with full context, pilot user review (~5 leads)

### Phase 4: Scale
**When:** Pilot approved, ready to process remaining leads
**Load:** `references/phases/phase-4-scale.md`
**Focus:** Apply approved approach to all remaining leads in batches

### Phase 5: Finalize
**When:** All leads processed, ready for final review
**Load:** `references/phases/phase-5-finalize.md`
**Focus:** Merge batches, final user review partnership, export

## Required Inputs

Before starting any campaign:

1. **Campaign Brief** (Mandatory)
   - Located in `campaigns/` directory
   - Contains: offer, target persona, value proposition, message templates
   - Example: `campaigns/faie-ai-accelerator-feb2026.md`

2. **Lead List**
   - From LinkedIn Sales Navigator or provided CSV
   - Minimum fields: firstName, lastName, company, title, linkedInUrl

3. **PFC Criteria**
   - Perfect Future Customer definition
   - See `references/guides.md` for exclusion/inclusion criteria

## Output Files

| File | Purpose |
|------|---------|
| `campaign_research_audit.csv` | Final merged output with all leads and messages |
| `all_leads_master.csv` | Status tracking across campaign lifecycle |
| `batch_XXX/*.json` | Intermediate batch files for recovery |

## Reference Files

### Phase Guides
- `references/phases/phase-1-enrichment.md` - 4-dimension data gathering
- `references/phases/phase-2-message-crafting.md` - Identity-First messaging (core method)
- `references/phases/phase-3-quality-assurance.md` - Perplexity + pilot review
- `references/phases/phase-4-scale.md` - Batch processing remaining leads
- `references/phases/phase-5-finalize.md` - Merge and final review

### Supporting Guides
- `references/guides.md` - PFC criteria + CSV schema
- `references/troubleshooting.md` - Common issues and solutions

### Scripts
- `scripts/batch_update_linkedin_urls.py` - Update CSV with LinkedIn URLs
- `scripts/csv_to_multilead_json.py` - Export to Multilead format
- `scripts/merge_csv_batches.sh` - Merge batch files

### Examples
- `examples/campaign-brief-template.md` - Campaign brief structure
- `examples/enrichment-example.json` - Sample enrichment data
- `examples/message-examples.md` - Identity-First message examples

## Quick Start

1. **Read campaign brief** from `campaigns/` directory
2. **Load Phase 1** (`references/phases/phase-1-enrichment.md`)
3. **Progress sequentially** through phases
4. **Save incrementally** after each lead
5. **User reviews at Phase 3** (pilot) and **Phase 5** (final)

## Resuming Mid-Campaign

If context was lost mid-campaign:
1. Check `output/` for latest batch files
2. Review `all_leads_master.csv` for status
3. Load appropriate phase reference
4. Continue from last saved state
