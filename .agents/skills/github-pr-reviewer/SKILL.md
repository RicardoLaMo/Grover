---
name: github-pr-reviewer
description: Review, comment on, and resolve GitHub Pull Requests. Auto-trigger LaTeX updates when PRs contain experiment results.
---

# GitHub PR Reviewer

You are equipped to interact with GitHub Pull Requests via the `gh` CLI. Use this skill when the user asks you to check, review, or resolve a PR.

## Prerequisites

- **GitHub CLI**: `gh` (installed at `/opt/homebrew/bin/gh`)
- **Authentication**: Logged in as `RicardoLaMo` via `gh auth login`
- **Repository**: `RicardoLaMo/Top-Attention`

## Commands Reference

| Action | Command |
|---|---|
| List open PRs | `gh pr list` |
| View PR details | `gh pr view <number>` |
| View PR with comments | `gh pr view <number> --comments` |
| View PR diff | `gh pr diff <number>` |
| Check out PR locally | `gh pr checkout <number>` |
| Post review comment | `gh pr review <number> --comment --body "..."` |
| Approve PR | `gh pr review <number> --approve --body "..."` |
| Request changes | `gh pr review <number> --request-changes --body "..."` |
| Merge PR (squash) | `gh pr merge <number> --squash --delete-branch` |
| Create a new PR | `gh pr create --title "..." --body "..." --base main` |

## Review Workflow

When asked to review a PR:

### Step 1: Fetch & Understand
```bash
gh pr view <number>
gh pr diff <number>
```
Read the title, description, and full diff. Understand the intent.

### Step 2: Check Compliance
Verify these items and note any failures:
- [ ] **PRD exists**: Does `docs/prd/` contain a PRD linked to this change?
- [ ] **Experiment logged**: If code touches model/eval logic, was `artifacts/experiment_log.jsonl` updated?
- [ ] **Tests pass**: Does the code run without errors?
- [ ] **Clean commits**: Are commit messages following convention (`feat:`, `fix:`, etc.)?

### Step 3: Detect LaTeX Update Triggers
Check if the PR touches ANY of these:
- `artifacts/experiment_log.jsonl`
- `reports/experiments/*.tex`
- Files in `src/` related to model evaluation

**If triggered:** Use the `latex-experiment-reporter` skill to generate or update the corresponding `.tex` report fragment. Commit it to the same PR branch:
```bash
gh pr checkout <number>
# ... generate/update .tex file ...
git add reports/experiments/<name>.tex
git commit -m "docs: auto-generate LaTeX report for <experiment>"
git push
```

### Step 4: Post Review
Based on your analysis, take ONE of:

**If everything looks good:**
```bash
gh pr review <number> --approve --body "✅ LGTM. PRD linked, experiment logged, LaTeX report generated."
```

**If changes needed:**
```bash
gh pr review <number> --request-changes --body "❌ Issues found:
- [ ] Missing PRD in docs/prd/
- [ ] experiment_log.jsonl not updated
Please fix and re-request review."
```

**If just commenting (not blocking):**
```bash
gh pr review <number> --comment --body "💬 Looks good overall. Minor suggestion: ..."
```

### Step 5: Merge (if authorized)
Only merge after approval and all checks pass:
```bash
gh pr merge <number> --squash --delete-branch
```

## Resolving PR Issues

When asked to **resolve** issues on a PR:

1. Check out the PR branch: `gh pr checkout <number>`
2. Read the review comments: `gh pr view <number> --comments`
3. Make the required code fixes locally.
4. Commit and push: `git add -A && git commit -m "fix: resolve PR feedback" && git push`
5. Post a comment confirming the fix: `gh pr review <number> --comment --body "Fixed: [description of changes]"`

## Creating PRs

When the MLE or Orchestrator finishes work on a feature branch:
```bash
gh pr create --title "feat: <description>" --body "## Summary
<what changed>

## PRD
docs/prd/<feature>.md

## Experiment Results
<metrics if applicable>" --base main
```
