# Phase 5: Finalize

## Purpose

Merge all batch outputs into campaign-ready deliverables and conduct final user review partnership.

## Part 1: Merge Batches

### Merge Script

Use the batch merge script to combine all outputs:

```bash
# From project root
./scripts/merge_csv_batches.sh output/batch_*/messages_draft.json > output/campaign_research_audit.csv
```

Or manually with Python:

```python
import json
import csv
import glob

# Collect all batch files
batch_files = sorted(glob.glob("output/batch_*/messages_draft.json"))

all_leads = []
for batch_file in batch_files:
    with open(batch_file, 'r') as f:
        leads = json.load(f)
        all_leads.extend(leads)

# Write merged CSV
with open("output/campaign_research_audit.csv", 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'firstName', 'lastName', 'company', 'title', 'linkedInUrl',
        'personalisedMessage', 'messageRationale', 'personalisationHooks',
        'qaStatus', 'validationNotes'
    ])
    writer.writeheader()
    writer.writerows(all_leads)

print(f"Merged {len(all_leads)} leads to campaign_research_audit.csv")
```

### Post-Merge Validation

After merge, verify:

- [ ] Total lead count matches expected
- [ ] No duplicate entries
- [ ] All required fields populated
- [ ] No encoding issues (mojibake characters)

### Encoding Check

Scan for common encoding issues:

```python
import csv

issues = []
with open("output/campaign_research_audit.csv", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        message = row.get('personalisedMessage', '')
        # Check for mojibake patterns
        if any(pattern in message for pattern in ['Ã¢', 'â€', 'Â']):
            issues.append((row['firstName'], row['lastName']))

print(f"Found {len(issues)} leads with encoding issues")
```

Fix any encoding issues before proceeding.

## Part 2: Final User Review Partnership

### Review Presentation Format

Present the complete campaign to user in digestible format:

```markdown
## Campaign Review: [Campaign Name]

**Summary:**
- Total leads: [count]
- Industries covered: [list]
- Seniority mix: [CEO: X, COO: Y, Director: Z]

**Sample Messages by Category:**

### Mining/Resources
1. [Lead Name] - [Company]
   > [message]

2. [Lead Name] - [Company]
   > [message]

### Healthcare
1. [Lead Name] - [Company]
   > [message]

[Continue for each industry category]

**Questions for Review:**
1. Overall tone and approach - on target?
2. Any industries needing adjustment?
3. Specific leads to revise?
```

### Batch Review Strategy

For large campaigns (50+ leads):

1. **Category Sampling** - Review 2-3 from each industry category
2. **Edge Case Focus** - Review any flagged during Phase 4
3. **Random Spot-Check** - 5-10 random selections
4. **User's Picks** - Any specific leads user wants to see

### User Feedback Categories

| Feedback | Action |
|----------|--------|
| "All good" | Mark campaign ready |
| "Minor tweaks on X" | Apply edits, update CSV |
| "Industry Y needs different approach" | Revise that category |
| "These 3 leads need work" | Individual revisions |

### Revision Workflow

For any revisions needed:

1. **Document the issue** - What needs changing and why
2. **Apply the revision** - Update message in CSV
3. **Update rationale** - Add `[REVISED]` note to Message_Rationale
4. **Re-validate if significant** - Quick Perplexity check for major changes

```python
# Example revision update
def apply_revision(csv_path, first_name, last_name, new_message, revision_reason):
    # Read CSV
    # Find lead by name
    # Update personalisedMessage
    # Append to Message_Rationale: "\n\n[REVISED] {revision_reason}"
    # Write back
```

## Part 3: Export Options

### Export to Multilead Format

If sending via Multilead tool:

```bash
python scripts/csv_to_multilead_json.py \
  --input output/campaign_research_audit.csv \
  --output output/multilead_import.json
```

### Export for Sales Navigator

If manual sending:

```markdown
## Ready-to-Send Messages

### Lead 1: [Name] at [Company]
**LinkedIn URL**: [url]
**Message**:
[paste-ready message]

---

### Lead 2: [Name] at [Company]
...
```

### Archive for Future Reference

Create campaign archive:

```
output/
├── campaign_research_audit.csv      ← Main deliverable
├── all_leads_master.csv             ← Status tracking
├── multilead_import.json            ← Tool import format
├── batch_001/ through batch_XXX/    ← Original batches (keep for recovery)
└── campaign_metadata.json           ← Campaign summary
```

Campaign metadata:

```json
{
  "campaignName": "FAIE AI Accelerator Feb 2026",
  "createdDate": "2025-12-16",
  "totalLeads": 94,
  "approvalDate": "2025-12-16",
  "methodology": "Identity-First",
  "phases": ["enrichment", "message-crafting", "qa", "scale", "finalize"]
}
```

## Final Checklist

### Merge Complete
- [ ] All batches merged to single CSV
- [ ] Lead count verified
- [ ] No duplicates
- [ ] Encoding clean

### User Review Complete
- [ ] Sample messages reviewed by category
- [ ] User feedback incorporated
- [ ] Any revisions applied
- [ ] Final approval received

### Campaign Ready
- [ ] Export format prepared
- [ ] Archive created
- [ ] Campaign metadata documented
- [ ] Ready for send

## Post-Campaign Notes

### Learnings to Capture

After campaign sends, document:

1. **What worked** - Which message patterns got best response
2. **What didn't** - Any patterns that fell flat
3. **Industry insights** - Sector-specific learnings
4. **Process improvements** - Workflow refinements for next time

### Update Campaign Brief

If learnings are significant, update the campaign brief (`campaigns/[campaign-name].md`) with:

- New example messages that worked well
- Industry-specific adjustments
- Updated writing guidelines

### Memory Storage

Store campaign learnings for future reference:

```markdown
## Campaign: [Name] - Learnings

### What Worked
- Identity-First approach with [specific pattern]
- [Industry] responded well to [approach]

### What Didn't
- [Pattern] felt too [assessment]
- Avoid [specific phrasing] for [audience]

### Process Notes
- Batch size of [X] was optimal
- Perplexity validation caught [specific issue types]
```

## Campaign Complete

Congratulations! The campaign is ready to send. Key deliverables:

1. **`campaign_research_audit.csv`** - All leads with personalized messages
2. **`multilead_import.json`** (if applicable) - Tool-ready format
3. **Campaign archive** - Full documentation for future reference

Return to **SKILL.md** for next campaign or reference materials.
