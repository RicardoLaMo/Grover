---
name: zotero-research
description: Search, retrieve, and cite papers from the Zotero library for the TopAttention project. Integrates with PI, MLE, and LaTeX reporter agents.
---

# Zotero Research Skill (TopAttention Project)

You are equipped to search, read, and cite academic papers from the Zotero library. This is the primary literature management tool for the project.

## Project Collection

All papers relevant to this project should be tagged with **`TopAttention`** in Zotero. When adding new papers, always apply this tag.

## Available Tools

| Tool | Purpose |
|---|---|
| `zotero_search_items` | Search by title, author, year, or full text |
| `zotero_item_metadata` | Get metadata (title, authors, DOI, URL, tags) |
| `zotero_item_fulltext` | Get the full text content of a paper |

## Progressive Reading Protocol (L1 → L2 → L3)

**Never read a full paper in one shot.** Follow the layered protocol to manage context and document insights incrementally.

### Layer 1: Scan (Abstract + Introduction)

**When:** Initial topic exploration, relevance filtering.

1. Search Zotero: `zotero_search_items(query="...", qmode="everything", limit=10)`
2. For each candidate, get metadata: `zotero_item_metadata(item_key="...")`
3. Get full text: `zotero_item_fulltext(item_key="...")`
4. Read **only** the Abstract and Introduction (stop at Section 2 / "Methods" / "Related Work").
5. Log to knowledge base:

```jsonl
{
  "timestamp": "<ISO-8601>",
  "paper_key": "<key>",
  "paper_title": "<title>",
  "authors": "<authors>",
  "year": "<year>",
  "doi": "<doi>",
  "layer": "L1",
  "query_context": "<what you searched for and why>",
  "findings": {
    "relevance": "high|medium|low",
    "key_claims": ["claim1", "claim2"],
    "problem_addressed": "<what problem does this paper solve>",
    "proposed_approach": "<one-line summary of their approach>",
    "next_action": "proceed_L2|cite_only|discard"
  }
}
```

**Decision gate:** Only proceed to L2 if relevance is "high" and the paper's approach is directly useful.

---

### Layer 2: Deep Read (Methodology + Results)

**When:** Paper confirmed relevant at L1, need method details or results.

1. Read the Methodology/Model section — understand the architecture, algorithm, or formulation.
2. Read the Results/Experiments section — extract key metrics and comparisons.
3. Read the Discussion section — identify limitations, future work, and connections to our project.
4. Append to knowledge base:

```jsonl
{
  "timestamp": "<ISO-8601>",
  "paper_key": "<key>",
  "layer": "L2",
  "query_context": "<why we're reading deeper>",
  "findings": {
    "method_summary": "<2-3 sentence description of the method>",
    "architecture": "<key components: encoder, attention, loss function, etc.>",
    "results_table": [
      {"metric": "AUC", "value": "82.8%", "baseline": "81.8% (MLP)", "dataset": "15 UCI datasets"}
    ],
    "key_findings": ["finding1", "finding2"],
    "limitations": ["limitation1"],
    "relevance_to_project": "<how this connects to TopAttention>",
    "next_action": "proceed_L3|implement|cite"
  }
}
```

---

### Layer 3: Extract (Equations + Facts)

**When:** Building on the paper's ideas, need exact formulations for implementation or citation.

1. Extract all relevant **mathematical equations** as LaTeX strings.
2. Extract **definitions** (formal definitions of terms, metrics, or concepts).
3. Extract **factual supports** (specific claims with their evidence and citations from the paper).
4. Append to knowledge base:

```jsonl
{
  "timestamp": "<ISO-8601>",
  "paper_key": "<key>",
  "layer": "L3",
  "query_context": "<what equations/facts we need>",
  "findings": {
    "equations": [
      {
        "name": "Self-Attention",
        "latex": "\\text{Attention}(K, Q, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{k}}\\right) \\cdot V",
        "context": "Core attention mechanism, Eq. 2",
        "variables": {"K": "Key matrix", "Q": "Query matrix", "V": "Value matrix", "k": "key dimension"}
      }
    ],
    "definitions": [
      {
        "term": "Contextual Embedding",
        "definition": "Embedding transformed through successive Transformer layers that aggregates context from other feature embeddings",
        "source_section": "Section 2"
      }
    ],
    "factual_supports": [
      {
        "claim": "TabTransformer outperforms MLP by 1.0% on mean AUC across 15 datasets",
        "evidence": "Table 1, Section 3.1",
        "cited_in": "huang_tabtransformer_2020"
      }
    ]
  }
}
```

---

## Knowledge Base Files

All research is persisted to these files:

| File | Purpose |
|---|---|
| `references/knowledge_base.jsonl` | Append-only log of ALL paper interactions (L1/L2/L3) |
| `references/kb_index.json` | Topic → paper key → depth index for fast lookup |

### Checking the Knowledge Base Before Reading

**Before reading any paper, ALWAYS check if it's already in the knowledge base:**

```bash
# Check if paper was already read
grep "<paper_key>" references/knowledge_base.jsonl
```

If found, read the existing entry instead of re-reading the paper. Only proceed to a deeper layer if needed.

### Updating the Topic Index

After logging a KB entry, update `references/kb_index.json`:

```json
{
  "attention_mechanism": ["NFWGUBEM:L3", "2DYNG9GB:L1"],
  "graph_neural_network": ["XYZ789:L2"],
  "tabular_data": ["NFWGUBEM:L2"]
}
```

Add the paper key under ALL relevant topic keys. Create new topic keys as needed.

---

## BibTeX Citation Extraction

After retrieving metadata, format a BibTeX entry:

```bibtex
@article{lastname_keyword_year,
  title  = {Paper Title},
  author = {Last, First and Last2, First2},
  journal = {Journal Name},
  year   = {2024},
  doi    = {10.xxxx/...},
}
```

**Key naming convention:** `{first_author_lastname}_{one_keyword}_{year}` (e.g., `huang_tabtransformer_2020`)

---

## Adding Papers by DOI (Code Execution)

```python
import sys
sys.path.append('/Volumes/MacMini/.gemini/antigravity/skills/zotero-mcp-code/scripts')
import setup_paths
from zotero_lib import SearchOrchestrator

orchestrator = SearchOrchestrator()
item_key = orchestrator.create_item_from_doi("10.xxxx/paper-doi")
print(f"Added paper with key: {item_key}")
```

## Comprehensive Literature Search (Code Execution)

```python
import sys
sys.path.append('/Volumes/MacMini/.gemini/antigravity/skills/zotero-mcp-code/scripts')
import setup_paths
from zotero_lib import SearchOrchestrator, format_results

orchestrator = SearchOrchestrator()
results = orchestrator.comprehensive_search("graph attention networks", max_results=20)
print(format_results(results, include_abstracts=True))
```

## Agent-Specific Instructions

### For the PI Agent
- **Always follow L1→L2→L3 protocol** — never skip layers.
- **Before proposing a new idea:** Search Zotero + check KB index for existing coverage.
- **When writing paper sections:** Use L2/L3 entries from the KB for method descriptions and citations.
- **When discovering new papers:** Add via DOI, tag `TopAttention`, start with L1.

### For the MLE Agent
- **Before implementing:** Check KB for L3 entries (equations, architecture details).
- **If L3 not available:** Read the paper through L1→L2→L3, extract the math, THEN implement.
- **Search GitHub** (via `github-code-researcher`) after reading the paper for reference code.

### For the LaTeX Reporter
- **Pull citations** from KB entries (L1 key_claims and L3 factual_supports).
- **Use equations** from L3 entries directly in `.tex` fragments.
- **Auto-generate BibTeX** from metadata when citing a new paper.
