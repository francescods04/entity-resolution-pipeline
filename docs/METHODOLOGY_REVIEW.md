# PhD Methodology Review: Entity Resolution Pipeline

**Date:** January 7, 2026
**Reviewer:** Prof. [Antigravity]
**Subject:** Scientific Evaluation of Entity Resolution Methodology for Venture Capital Data

---

## 1. Executive Summary

Having reviewed the `METHODOLOGY.md` and the underlying implementation (`src/`), I assess this pipeline as **functionally robust and commercially viable**, operating well above the standard of typical industry scripts.

However, from a **PhD-level academic perspective**, while the *engineering* is excellent, the *scientific rigor* regarding uncertainty quantification, graph-theory application, and evaluation protocol requires elevation.

**Grade:** A- (Excellent implementation, but theoretical formalization can be deeper)

---

## 2. Strong Points (The "Good")

Your pipeline correctly identifies and mitigates the primary challenges of matching disparate corporate datasets. Specific commendations:

### 2.1. Corporate Graph Intelligence (`orbis_graph.py`)
The decision to model "Families" rather than just individual entities is a critical insight. Most naive approaches fail because they try to match a Crunchbase startup to a specific legal entity (e.g., *Deliveroo Ltd*) when the investment actually sits with the holding company (*Deliveroo Holdings PLC*).
*   **Verdict:** Your `derive_family_structure` and "Family Expansion" blocking strategy effectively solve the "One-to-Many" legal entity problem. This is a publication-worthy contribution if formalized correctly.

### 2.2. Network-Based Validation (`investor_linkage.py`)
Leveraging the *bipartite graph* of Investors ↔ Startups is a sophisticated touch. Using shared shareholders as a validation signal (`compute_investor_overlap`) moves beyond simple string matching into **structural matching**. This is robust and aligns with modern "Collective Entity Resolution" literature.

### 2.3. Multi-Stage Efficient Blocking (`blocking.py`)
The implementation of a `BlockingIndex` with spill-to-disk capabilities for 15M records shows high engineering maturity. The tiered strategy (Domain → Country+Prefix → Rare Token) balances recall and computational cost ($O(k \times n)$) effectively.

---

## 3. Critical Critique (The "Bad" & "Ugly")

To withstand a doctoral defense, you must address the following methodological weaknesses:

### 3.1. Heuristic "Magic Numbers" in Feature Engineering
**Location:** `features.py` -> `compute_disambiguation_score`
**Critique:** The code contains hardcoded heuristics:
```python
score += 100.0  # Domain match
score -= 50.0   # Country mismatch
score += city_sim * 20.0
```
This is **scientifically indefensible**. Why 100? Why 50? In a PhD context, these weights should be learned, not guessed.
*   **Recommendation:** Replace this with a proper probabilistic model (e.g., a lightweight Logistic Regression or Naive Bayes layer) even for the "disambiguation" step, or justify these values via an empirical calibration study.

### 3.2. Evaluation on "500 Random Pairs"
**Location:** `METHODOLOGY.md` -> Validation Approach
**Critique:** Evaluating on 500 *random* pairs is insufficient because non-matches vastly outnumber matches (class imbalance). A random sample will likely be 99% easy negatives, inflating your accuracy metrics.
*   **Recommendation:** usage of **Stratified Sampling**. You need to create a "Active Learning" test set that specifically over-samples "hard cases" (e.g., high string similarity but different country, or same family but different legal entity) to truly measure the pipeline's resolution power.

### 3.3. Feature Multicollinearity
**Location:** `features.py`
**Critique:** You calculate `jaro_winkler`, `token_jaccard`, and `rapidfuzz_ratio`. These features are highly correlated. While Gradient Boosting handles collinearity better than linear models, adding redundant features dilutes interpretability (feature importance analysis) and wastes compute.
*   **Recommendation:** Perform a PCA or correlation analysis to select the most orthogonal string distance metrics.

### 3.4. The "Weak Supervision" Black Box
**Location:** `METHODOLOGY.md`
**Critique:** You mention using "Positive Rules" and "Negative Rules" to train a model. This is excellent (Snorkel approach), but the methodology lacks a feedback loop description. How do you handle "Rule Conflicts" (where one rule says yes, another says no)?
*   **Recommendation:** Formalize the Label Model. Explain how you resolve conflicts—do you use majority vote, weighted vote, or a probabilistic label model?

---

## 4. Proposals for PhD-Level Elevation

To take this from "Great Project" to "PhD Thesis":

### 4.1. Formalize the Graph Neural Network (GNN) Potential
You have the nodes (Companies, Investors) and edges (Ownership, Investment).
*   **Idea:** Instead of just calculating "Investor Overlap" as a scalar feature, you could propose (even if you don't fully implement yet) a **Graph Convolutional Network (GCN)** approach where the embedding of a startup approximates the embedding of its investor set. This frames the problem as "Link Prediction" in a heterogeneous graph.

### 4.2. Ablation Studies
You must prove that your complex features are necessary.
*   **Requirement:** Include an "Ablation Study" section in your results:
    *   *Performance with ONLY String Matching*
    *   *Performance adding Domain Matching*
    *   *Performance adding Family Expansion* (Key contribution!)
    *   *Performance adding Investor Overlap*
    *   This quantifies the marginal gain of your novel contributions.

### 4.3. Error Bound Analysis
For the "Automatic Acceptance" (Tier A), standard precision/recall is not enough. You should calculate **bounds on the error rate**.
*   *Example:* "With 95% confidence, the error rate in Tier A is less than 0.5%."

## 5. Summary

The methodology is solid. The code is clean, modular, and performant. The "Science" needs to catch up to the "Engineering". Stop using magic numbers, rigorous-ize your evaluation set, and frame your graph-based features as a structural advantage over flat string matching.

**Approval Status:** **APPROVED with revisions**.
Proceed with the implementation, but schedule a refactor of `features.py` to remove hardcoded scoring logic in favor of the learned model.
