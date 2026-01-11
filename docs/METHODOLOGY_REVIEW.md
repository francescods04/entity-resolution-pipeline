# Methodology Review: Entity Resolution Pipeline

**Date:** January 7, 2026

---

## 1. Executive Summary

Having reviewed the `METHODOLOGY.md` and the underlying implementation (`src/`), I assess this pipeline as **functionally robust and commercially viable**, operating well above the standard of typical industry scripts.

The *engineering* is excellent, and the *scientific rigor* regarding uncertainty quantification, graph-theory application, and evaluation protocol meets research standards.

**Grade:** A- (Excellent implementation, with room for theoretical formalization)

---

## 2. Strong Points

Your pipeline correctly identifies and mitigates the primary challenges of matching disparate corporate datasets. Specific commendations:

### 2.1. Corporate Graph Intelligence (`orbis_graph.py`)
The decision to model "Families" rather than just individual entities is a critical insight. Most naive approaches fail because they try to match a Crunchbase startup to a specific legal entity (e.g., *Deliveroo Ltd*) when the investment actually sits with the holding company (*Deliveroo Holdings PLC*).
*   **Verdict:** Your `derive_family_structure` and "Family Expansion" blocking strategy effectively solve the "One-to-Many" legal entity problem.

### 2.2. Network-Based Validation (`investor_linkage.py`)
Leveraging the *bipartite graph* of Investors ↔ Startups is a sophisticated touch. Using shared shareholders as a validation signal (`compute_investor_overlap`) moves beyond simple string matching into **structural matching**. This aligns with modern "Collective Entity Resolution" literature.

### 2.3. Multi-Stage Efficient Blocking (`blocking.py`)
The implementation of a `BlockingIndex` with spill-to-disk capabilities for 15M records shows high engineering maturity. The tiered strategy (Domain → Country+Prefix → Rare Token) balances recall and computational cost effectively.

---

## 3. Critical Critique

To meet SOTA standards, you must address the following methodological weaknesses:

### 3.1. Heuristic "Magic Numbers" in Feature Engineering
**Location:** `features.py` -> `compute_disambiguation_score`
**Critique:** The code contains hardcoded heuristics:
```python
score += 100.0  # Domain match
score -= 50.0   # Country mismatch
```
This is scientifically weak. These weights should be learned, not guessed.
*   **Recommendation:** Replace with a probabilistic model (e.g., Logistic Regression) or justify values via empirical calibration.

### 3.2. Evaluation on "500 Random Pairs"
**Location:** `METHODOLOGY.md` -> Validation Approach
**Critique:** Evaluating on 500 *random* pairs is insufficient because non-matches vastly outnumber matches. A random sample will be mostly easy negatives, inflating accuracy.
*   **Recommendation:** Use **Stratified Sampling** that oversamples "hard cases" (high string similarity but different country).

### 3.3. Feature Multicollinearity
**Location:** `features.py`
**Critique:** You calculate `jaro_winkler`, `token_jaccard`, and `rapidfuzz_ratio`. These metrics are highly correlated.
*   **Recommendation:** Perform correlation analysis to select the most orthogonal string distance metrics.

### 3.4. The "Weak Supervision" Black Box
**Location:** `METHODOLOGY.md`
**Critique:** You mention using "Positive Rules" and "Negative Rules" to train a model, but lack a feedback loop description. How do you handle "Rule Conflicts"?
*   **Recommendation:** Formalize the Label Model. Explain conflict resolution—majority vote, weighted vote, or probabilistic label model?

---

## 4. Proposals for SOTA Elevation

### 4.1. Formalize the Graph Neural Network Potential
You have nodes (Companies, Investors) and edges (Ownership, Investment).
*   **Idea:** Instead of "Investor Overlap" as a scalar, propose a **GCN** approach where startup embeddings approximate investor set embeddings. Frame as "Link Prediction" in a heterogeneous graph.

### 4.2. Ablation Studies
Prove that complex features are necessary.
*   **Requirement:** Include an ablation study showing performance with only string matching, then adding domain matching, family expansion, and investor overlap.

### 4.3. Error Bound Analysis
For Tier A, calculate **bounds on the error rate**.
*   *Example:* "With 95% confidence, the Tier A error rate is less than 0.5%."

## 5. Summary

The methodology is solid. The code is clean, modular, and performant. Stop using magic numbers, use rigorous evaluation, and frame graph-based features as a structural advantage.

**Status:** **APPROVED with revisions**.
Proceed with implementation, but refactor `features.py` to remove hardcoded scoring logic.
