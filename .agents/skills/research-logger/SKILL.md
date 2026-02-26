---
name: research-logger
description: Document research interactions automatically. Log paper searches, findings, decisions, and extracted knowledge to the project knowledge base.
---

# Research Logger

Automatically document every research interaction to build a persistent, searchable knowledge base.

## Knowledge Base Files

| File | Format | Purpose |
|---|---|---|
| `references/knowledge_base.jsonl` | JSONL (one entry per line) | Append-only log of ALL paper interactions |
| `references/kb_index.json` | JSON | Topic → paper keys → depth for fast lookup |

## When to Log

Log **every time** you:
- Search Zotero and find relevant results
- Read a paper at any layer (L1, L2, or L3)
- Make a decision about a paper (cite, implement, discard)
- Extract equations or factual evidence

## How to Log

### 1. Append to knowledge_base.jsonl

Use a shell command to append a single JSON line:

```bash
echo '{"timestamp":"<ISO-8601>","paper_key":"<key>","paper_title":"<title>","layer":"L1","query_context":"<why>","findings":{...}}' >> references/knowledge_base.jsonl
```

Or write using Python for complex entries:

```python
import json, datetime

entry = {
    "timestamp": datetime.datetime.now().isoformat(),
    "paper_key": "<key>",
    "paper_title": "<title>",
    "authors": "<authors>",
    "year": "<year>",
    "doi": "<doi>",
    "layer": "L1",  # or L2, L3
    "query_context": "<what you searched and why>",
    "findings": {
        # L1 fields:
        "relevance": "high",
        "key_claims": ["..."],
        "problem_addressed": "...",
        "proposed_approach": "...",
        "next_action": "proceed_L2",
        # L2 fields (add when layer=L2):
        "method_summary": "...",
        "architecture": "...",
        "results_table": [{"metric": "AUC", "value": "82.8%", "baseline": "81.8%"}],
        "key_findings": ["..."],
        "limitations": ["..."],
        # L3 fields (add when layer=L3):
        "equations": [{"name": "...", "latex": "...", "context": "...", "variables": {}}],
        "definitions": [{"term": "...", "definition": "...", "source_section": "..."}],
        "factual_supports": [{"claim": "...", "evidence": "...", "cited_in": "..."}]
    }
}

with open("references/knowledge_base.jsonl", "a") as f:
    f.write(json.dumps(entry) + "\n")
```

### 2. Update kb_index.json

After each KB entry, update the topic index:

```python
import json

index_path = "references/kb_index.json"
with open(index_path, "r") as f:
    index = json.load(f)

# Add paper under relevant topics
topics = ["attention_mechanism", "tabular_data"]
paper_ref = "NFWGUBEM:L1"

for topic in topics:
    if topic not in index:
        index[topic] = []
    # Update depth if already exists
    existing = [r for r in index[topic] if r.startswith("NFWGUBEM")]
    for old in existing:
        index[topic].remove(old)
    index[topic].append(paper_ref)

with open(index_path, "w") as f:
    json.dump(index, f, indent=2)
```

## Before Reading Any Paper

**Always check the KB first** to avoid duplicate work:

```bash
grep "<paper_key>" references/knowledge_base.jsonl
```

Or search by topic:
```python
import json
with open("references/kb_index.json") as f:
    index = json.load(f)
print(index.get("attention_mechanism", []))
```

If the paper is already at the depth you need, use the existing entry. Only go deeper if more detail is required.

## Research Decision Trail

When making a decision about a paper, document:
- **cite**: Paper supports our claims, add to bibliography
- **implement**: Paper's method should be coded, create a task
- **proceed_L2/L3**: Need more depth
- **discard**: Not relevant enough

This trail enables reproducible research decisions and avoids revisiting discarded papers.
