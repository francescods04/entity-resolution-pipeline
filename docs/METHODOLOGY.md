# Methodology

## Entity Resolution Approach

This project implements a **multi-stage entity resolution pipeline** designed to achieve PhD-level academic rigor in matching venture capital-backed European startups from Crunchbase with corporate entities in the Bureau van Dijk Orbis database.

## Theoretical Foundation

### Problem Definition

Entity resolution (also known as record linkage or deduplication) is the task of identifying records across heterogeneous data sources that refer to the same real-world entity. In our context:

- **Source A (Crunchbase)**: VC/startup ecosystem data with rich investor relationships
- **Source B (Orbis)**: Regulatory corporate data with ownership hierarchies

### Challenges

1. **Schema Heterogeneity**: Different naming conventions, missing fields
2. **Name Variations**: Legal suffixes, rebrands, local vs. international names
3. **Corporate Structures**: Parent-subsidiary relationships, holding companies
4. **Scale Imbalance**: ~19K startups vs. ~15M Orbis records (1:800 ratio)

---

## Pipeline Architecture

### Stage 1: Data Preparation

**Normalization Layer**
- Unicode normalization (NFKD decomposition)
- Legal suffix stripping (GmbH, S.p.A., Inc., Ltd.)
- Diacritic removal (é → e, ü → u)
- Tokenization and fingerprinting

**Domain Extraction**
- eTLD+1 extraction using `tldextract`
- Free email blacklist filtering
- Subdomain handling

### Stage 2: Candidate Generation (Blocking)

To avoid O(n×m) comparison complexity, we employ **multi-source blocking**:

| Method | Precision | Recall | Cost |
|--------|-----------|--------|------|
| Domain Exact Match | Very High | Medium | O(n) |
| Country + Prefix | Medium | High | O(n log m) |
| Rare Token Overlap | Medium | Medium | O(k×n) |
| ANN Name Embedding | Medium | Very High | O(n log m) |
| ANN Desc Embedding | Low-Medium | High | O(n log m) |

**Family Expansion**: For each Orbis candidate, we also consider:
- The Global Ultimate Owner (GUO)
- Top subsidiaries with matching domains

### Stage 3: Feature Engineering

Our feature set captures multiple similarity dimensions:

**String Similarity**
- Jaro-Winkler distance
- Token Jaccard coefficient
- RapidFuzz token_sort_ratio
- Prefix matching (8-character)

**Domain/Contact**
- Exact domain match (strongest signal)
- Email domain intersection
- Domain-to-family overlap

**Geographic**
- Country ISO match
- City fuzzy match

**Temporal**
- Founded year vs. incorporation year alignment

**Semantic**
- Cosine similarity of sentence embeddings (name)
- Cosine similarity of sentence embeddings (description/industry)

### Stage 4: Classification

**Weak Supervision Framework**

Rather than requiring extensive manual labeling, we use **programmatic labeling** with high-precision rules:

```
Positive Rules:
  - domain_exact AND country_match AND name_sim >= 0.85
  - domain_family_hit AND desc_embedding_cos >= 0.90

Negative Rules:
  - same_country AND different_domains AND name_sim < 0.50
```

**Model Choice**: Gradient Boosting (scikit-learn) with:
- Isotonic calibration for probability estimates
- Feature importance analysis for interpretability

### Stage 5: Decisioning

**Tiered Output**:
- **Tier A** (p ≥ 0.98): Auto-accept
- **Tier B** (0.93 ≤ p < 0.98): High confidence
- **Tier C** (0.75 ≤ p < 0.93): Human review
- **Reject** (p < 0.75): No match

---

## Corporate Graph Handling

### Orbis Relationship Types

| Field | Meaning |
|-------|---------|
| GUO | Global Ultimate Owner (topco) |
| SUB | Subsidiaries |
| SH | Shareholders |
| BRANCH | Branches (same legal entity) |

### Match Types

When a Crunchbase startup matches an Orbis record, we classify:

- **GUO**: Match is the ultimate parent
- **LEGAL_ENTITY**: Match is the primary operating entity
- **SUBSIDIARY**: Match is a child company
- **BRANCH**: Match is a branch (not separate legal entity)

---

## Validation Approach

1. **Stratified Sampling**: Test set generated via `generate_stratified_test_set()` ensuring balanced tier representation
2. **Hard Negative Mining**: `sample_hard_negatives()` specifically over-samples challenging cases (high name similarity, different country)
3. **Error Bound Analysis**: Wilson score confidence intervals per tier via `compute_error_bounds()`
4. **Weak Label Quality**: Formalized `LabelModel` with weighted conflict resolution
5. **Ablation Studies**: Feature group contribution measured via `run_ablation_study()`

---

## Feature Analysis

To address multicollinearity among string distance metrics:
- `feature_analysis.py` provides correlation analysis
- `select_orthogonal_features()` removes redundant features
- Recommended: Use `name_jw` + `name_token_jaccard` (drop `name_rapidfuzz_ratio`)

---

## References

- Fellegi, I. P., & Sunter, A. B. (1969). A theory for record linkage. *JASA*.
- Christen, P. (2012). *Data Matching*. Springer.
- Mudgal et al. (2018). Deep Learning for Entity Matching. *SIGMOD*.
- Ratner et al. (2017). Snorkel: Rapid Training Data Creation with Weak Supervision. *VLDB*.
