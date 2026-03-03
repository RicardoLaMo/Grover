# THORN Paper Revision Checklist (Critique → Action Plan)

This checklist converts the critiques into **paper-ready edits, experiments, and positioning changes** for the current THORN manuscript and proposed THORN++ extension.

---

## 0) Critique Triage

### Must change (high risk in review)
1. **Observer redundancy not demonstrated** (LID/ORC/LOF may be collinear)  
   - Add non-redundancy analysis + observer ablations + router attribution.
2. **Shared Q/K/V across views is a representational bottleneck** (“experts” are only adjacency, not projection)  
   - Implement view-specific projections (or adapters) as **THORN++** and report results.
3. **Log-barrier numerical fragility** under normalization/self-loops  
   - Replace “avoid normalization” guidance with a stabilized formulation (learned temperature + clamp).
4. **FGW offline caching weakens inductive/streaming story**  
   - Either (A) scope the claim to transductive/static settings **or** (B) add a scalable proxy/online approximation story.

### Can be scoped (lower risk if positioned correctly)
- “Paradigm shift” rhetoric: keep but tighten to **routing prior + view selection** with explicit limitations.
- Real-world benchmarks: ideal; if not possible now, explicitly position synthetic as **mechanistic validation** and make benchmark extension a stated next step.

---

## 1) Abstract & Contributions (Front Matter)

### Edits
- Add: “We study **mechanistic regimes** on a controlled benchmark; real-world evaluation is ongoing.”
- Add: “We provide **observer non-redundancy analysis** and **router attribution** to validate observer orthogonality.”
- Tighten: “Subsumes X/Y” → “**function-class reductions under parameter settings** (not optimization/training equivalence).”

### Deliverables
- Updated abstract paragraph
- Contribution bullets revised to match new tables/figures below

---

## 2) Section 3.2 Observer Features → Add **3.2.1 Observer Non-Redundancy**

### 3.2.1 Required components
**(i) Correlation + effective rank**
- Pairwise Pearson/Spearman across `{LID, ORC stats, LOF, degree stats, kNN stats}`
- PCA / effective rank summary (e.g., top-k PCs explain %)

**(ii) Router attribution**
- Permutation importance (or IG/SHAP) on routing logits / routing loss
- Report per-observer contribution to routing decisions (`π_{i,h,m}`)

**(iii) Observer ablation**
- Remove LID only; remove ORC only; remove LOF only; remove all three
- Report: PR-AUC (or task metric), routing entropy, specialization stability

### New tables
- **Table X:** Observer redundancy (correlation + effective rank)
- **Table Y:** Observer ablations (metric + routing diagnostics)

---

## 3) Section 3.4 Observer Routing (MoE-style) → Clarify Collapse Mitigation

### Edits
- Add paragraph: orthogonality/load-balancing are **soft priors**; task loss can still dominate.
- Report tuned weights (λ terms) and routing entropy behavior.

### New diagnostics
- Plot: routing entropy vs drift
- Plot: per-head view usage vs drift

---

## 4) Section 3.5 Log-Barrier Masking → Stabilize Numerics (Turn weakness into contribution)

### Replace Eq. (5) with temperature-scaled barrier
Use **learned per-head** temperature:
- `log( τ_h * Â_ij + ε )`, with `τ_h = softplus(s_h)` initialized to 1.

### Clamp to bound gradients near zero
- `Â ← clamp(Â, a_min, 1)` (e.g., `a_min = 1e-4`)

### New ablation table (required)
Compare under preprocessing variants:
- Baseline vs `+τ` vs `+τ+clamp`
- Under (a) none, (b) row norm, (c) sym norm, (d) self-loops

### Deliverables
- Updated equation and implementation note
- **Table Z:** Stability study

---

## 5) Section 3.6 Isolation vs Interaction → Make it Predictive (Strongest “theory” contribution)

### Refinements
- Add an estimator paragraph:
  - Define how to compute neighborhood overlap `ρ_i` (e.g., intersection/union of view neighborhoods)
  - Define an empirical SNR proxy (synthetic or label-consistency proxy)

### New predictive result
- Plot: `Δ(metric) = post-softmax − pre-softmax` vs `(median ρ, SNR gap)`
- Show correlation / monotonic trend → turns narrative into diagnostic theory

### Deliverables
- One new figure (tradeoff predictor)
- One new subsection: “Estimators and Predictive Diagnostics”

---

## 6) Section 3.7 FGW Alignment → Fix the Scope/Scalability Problem

Pick **one** approach and make it explicit.

### Option A — Scope it cleanly (fastest editorial fix)
- State: FGW is **training-time regularizer** computed offline; inference uses cached consensus.
- Add limitation: dynamic streaming requires periodic refresh (daily/weekly) if topology drifts.

### Option B — Add a scalable proxy (best for streaming vision)
- Add alignment proxy based on cross-view agreement:
  - Neighborhood Jaccard
  - Distance-rank correlation
  - MMD between neighbor feature distributions
  - Mutual-kNN consistency
- Keep FGW as small-scale “gold standard”

### Required experiment
- **Table:** No alignment vs FGW vs Proxy alignment (metric + compute cost)

---

## 7) Section 4 Theoretical Analysis → Reviewer-proof Reductions

### Edit (one sentence)
- “These are **function-class reductions under parameter settings**, not claims about optimization or training dynamics.”

### Deliverable
- One-line clarification after Theorem 2 (and similar places)

---

## 8) THORN++ (View-Specific Projections) — Must-Have Refinement

### Minimal viable implementation (best ROI)
- Keep shared `W_Q/W_K/W_V`
- Add **LoRA-style low-rank adapters** per view: `ΔW_Q^(m), ΔW_K^(m), ΔW_V^(m)`
- Use routing `π` to select/blend view-specific projections

### Heavier but clean baseline
- Full per-view projections: `W_Q^(m), W_K^(m), W_V^(m)`

### Required experiment
- **Table:** THORN vs THORN++ on drift sweep + representative drift
- Also report effect on pre-softmax vs post-softmax regimes

---

## 9) Non-Negotiable New Tables/Figures (Reviewer checklist)

1. Observer redundancy + observer ablation (**new**)  
2. Log-barrier stabilization study (**new**)  
3. Alignment study: none vs FGW vs proxy (**expanded**)  
4. THORN++ view-specific projection comparison (**new**)  
5. Routing diagnostics across drift (entropy + head specialization) (**expanded**)

---

## 10) Priority Order (Fastest path to “submit-ready”)

1. Implement **τ log-barrier stabilization** + rerun normalization/self-loop tests  
2. Implement **THORN++ view-specific Q/K/V** (adapters acceptable) + rerun drift sweep  
3. Add **observer redundancy/ablation** section + router attribution figure  
4. Fix **FGW scope/proxy** + alignment ablation  
5. Add at least one real benchmark if targeting top-tier venues; otherwise explicitly scope to mechanistic validation

---

## 11) Pasteable Positioning Paragraph (for Introduction)

> THORN’s novelty is not merely incorporating geometry, but using **geometric observers as a routing prior** over competing neighborhood definitions, enabling dynamic selection between interaction and isolation regimes. We provide a diagnostic theory predicting when post-softmax interaction is beneficial versus when pre-softmax isolation avoids noisy cross-view coupling. Transport-based alignment is presented as a principled regularizer; we scope its use to training-time or provide scalable proxies for dynamic settings.

