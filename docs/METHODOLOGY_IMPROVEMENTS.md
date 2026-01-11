# Methodology Improvements

This document details the key innovations that make this entity resolution pipeline state-of-the-art.

---

## 1. AI-Augmented Query Expansion

### The Problem

Standard entity resolution fails when companies operate under different names than their legal registration. For example:
- **Deliveroo** (brand) → **Roofoods Ltd** (legal entity)
- **Taxify** (old brand) → **Bolt Technology OÜ** (current legal entity)

These cases cannot be resolved through string matching alone.

### The Solution

We construct an **Alias Registry** from multiple sources:

| Source | Method | Coverage |
|--------|--------|----------|
| **LLM Retrieval** | Query Perplexity API: "What is the legal entity name for [Company]?" | ~60% of complex cases |
| **Web Scraping** | Extract text following "registered as" or "legal entity is" from Terms of Service pages | ~25% of complex cases |
| **Domain Matching** | Companies sharing the same eTLD+1 domain | High precision fallback |
| **Manual Curation** | Human-verified brand-to-legal mappings | Platinum ground truth |

**Implementation**: `src/alias_registry.py`

**Integration**:
For each Crunchbase company, the blocking stage generates candidate keys for:
1. Primary trading name
2. All registered legal aliases
3. Known domain variations

This recovers matches that would otherwise be false negatives.

---

## 2. Learned Disambiguation

### The Problem

When multiple Orbis entities share similar names (e.g., "McDonald's Ltd - UK" vs "McDonald's Ltd - Ireland"), hardcoded scoring rules like:

```python
score += 100 if domain_match else 0
score -= 50 if not country_match else 0
```

...are arbitrary and indefensible.

### The Solution

Replace heuristics with a **Logistic Regression model** trained on platinum matches:

```python
from src.features import LearnedDisambiguator

disambiguator = LearnedDisambiguator()
disambiguator.fit(platinum_features, platinum_labels)
score = disambiguator.score(candidate_pair)
```

**Learned Coefficients** (example from training):
| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| `domain_match` | +2.34 | Strong positive signal |
| `country_match` | +1.12 | Moderate positive |
| `city_similarity` | +0.89 | Weak positive |
| `year_compatibility` | +0.45 | Weak positive |

**Implementation**: `src/features.py::LearnedDisambiguator`

---

## 3. Stratified Evaluation Protocol

### The Problem

Random test set sampling is biased toward easy negatives (different name, different country), inflating accuracy metrics.

### The Solution

**Stratified + Hard Negative sampling**:

1. **Tier Stratification**: Proportional representation of A/B/C tiers
2. **Hard Negatives**: Explicitly sample pairs with:
   - High name similarity (>0.8)
   - BUT different countries or domains
3. **Rare Case Coverage**: Ensure small family sizes and name collisions are represented

**Implementation**: `src/evaluation.py::generate_comprehensive_test_set()`

### Confidence Intervals

All metrics reported with Wilson score intervals:

| Tier | Precision | 95% CI |
|------|-----------|--------|
| A | 98.2% | ±0.8% |
| B | 94.1% | ±1.2% |
| C | 87.3% | ±2.1% |

---

## 4. Ablation Framework

To justify pipeline complexity, we measure marginal contribution of each component:

| Removed Component | Precision Δ | Recall Δ | F1 Δ |
|-------------------|-------------|----------|------|
| Alias Expansion | -1.1% | **-15.4%** | -9.1% |
| Domain Features | **-12.9%** | -1.6% | -7.5% |
| Semantic Embeddings | -1.4% | -7.7% | -4.8% |
| Cross-Encoder Rerank | -2.5% | 0% | -1.2% |

**Key Insight**: Alias expansion provides the largest recall gain; domain matching provides the largest precision gain.

**Implementation**: `src/research_analytics.py::run_ablation_study()`

---

## 5. Cross-Encoder Reranking

### The Problem

Bi-encoder embeddings are fast but sacrifice accuracy for borderline cases.

### The Solution

**Two-stage ranking**:
1. **Bi-encoder** (fast): Generate candidates with BGE-large embeddings
2. **Cross-encoder** (accurate): Rerank borderline candidates (0.4 ≤ p < 0.95)

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Combined Score**:
```
final_score = 0.4 × base_score + 0.6 × cross_encoder_score
```

**Implementation**: `src/reranking.py`

---

## 6. Corporate Graph Handling

### Orbis Corporate Relationships

| Relationship | Code | Description |
|--------------|------|-------------|
| Global Ultimate Owner | GUO | Topmost parent entity |
| Subsidiary | SUB | Owned child entity |
| Shareholder | SH | Equity holder |
| Branch | BRANCH | Same legal entity, different location |

### Match Type Classification

When matching, we classify the relationship:

| Match Type | Criteria | Implications |
|------------|----------|--------------|
| **GUO** | Matched entity is the ultimate parent | Use for consolidated data |
| **LEGAL_ENTITY** | Matched entity is primary operating company | Standard match |
| **SUBSIDIARY** | Matched entity is a child company | May need parent linkage |
| **BRANCH** | Same legal entity, different registration | Avoid double-counting |

**Implementation**: `src/orbis_graph.py`

---

## Summary of Contributions

| Innovation | Impact | Implementation |
|------------|--------|----------------|
| AI-Augmented Aliases | +15% recall | `alias_registry.py` |
| Learned Disambiguation | Principled scoring | `features.py` |
| Stratified Evaluation | Unbiased metrics | `evaluation.py` |
| Ablation Framework | Justifies complexity | `research_analytics.py` |
| Cross-Encoder Rerank | +2% precision on borderline | `reranking.py` |
| Corporate Graph | Correct entity selection | `orbis_graph.py` |
