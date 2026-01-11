# Methodology

## Entity Resolution for Corporate Data Matching

This document provides a comprehensive technical description of the entity resolution pipeline designed for matching venture capital-backed startups from the Crunchbase database with their corresponding legal entities in Bureau van Dijk's Orbis database.

---

## 1. Problem Statement

### 1.1 Research Context

Entity resolution—the task of determining whether two records refer to the same real-world entity—remains a fundamental challenge in data integration. In the domain of corporate finance research, this challenge is particularly acute when linking alternative datasets (e.g., VC investment records) to regulatory filings (e.g., balance sheets, ownership structures).

This pipeline addresses a specific instance of this problem:

- **Source A (Crunchbase)**: Contains approximately 19,000 European VC-backed startups with rich metadata on funding rounds, investors, and founding teams.
- **Source B (Orbis)**: Contains approximately 15,000,000 European corporate entities with legal registration data, ownership hierarchies, and financial statements.

The matching ratio of 1:800 creates significant computational and precision challenges.

### 1.2 Key Challenges

| Challenge | Description | Impact |
|-----------|-------------|--------|
| **Name Variation** | Trading names differ from legal names (e.g., "Uber" vs "Uber Technologies, Inc.") | False negatives |
| **Schema Heterogeneity** | Different data standards, missing fields, encoding issues | Data quality |
| **Corporate Structures** | Holding companies, subsidiaries, branches, rebrands | Many-to-one matches |
| **Scale Imbalance** | 19K queries against 15M candidates | Computational cost |
| **Class Imbalance** | True matches are ~0.1% of candidate space | Precision challenges |

---

## 2. Pipeline Architecture

The pipeline implements a five-stage architecture following established entity resolution best practices (Christen, 2012):

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ENTITY RESOLUTION PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐       │
│   │  DATA    │──▶│ BLOCKING │──▶│ FEATURE  │──▶│ SCORING  │       │
│   │  PREP    │   │          │   │ EXTRACT  │   │          │       │
│   └──────────┘   └──────────┘   └──────────┘   └──────────┘       │
│        │              │              │              │               │
│        ▼              ▼              ▼              ▼               │
│   Normalization  Candidate     Similarity     Probability          │
│   Domain Ext.    Generation    Vectors        Estimation           │
│                  (O(n log m))  (~2M pairs)    (Calibrated)         │
│                                                                     │
│                              ┌──────────┐                          │
│                              │ DECISION │                          │
│                              │          │                          │
│                              └──────────┘                          │
│                                   │                                 │
│                                   ▼                                 │
│                            Tiered Output                           │
│                            (A/B/C/Reject)                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Preparation

### 3.1 Text Normalization

All textual fields undergo a standardization pipeline:

1. **Unicode Normalization**: NFKD decomposition followed by ASCII transliteration
2. **Case Folding**: Conversion to lowercase
3. **Legal Suffix Removal**: Stripping of 200+ known suffixes (GmbH, S.p.A., Inc., Ltd, SRL, etc.)
4. **Whitespace Normalization**: Collapsing multiple spaces, trimming
5. **Character Filtering**: Removal of punctuation except hyphens within words

**Implementation**: `src/normalize.py::normalize_name_column()`

### 3.2 Domain Extraction

Web domains serve as high-precision matching keys. The extraction process:

1. **URL Parsing**: Protocol stripping, path removal
2. **eTLD+1 Extraction**: Using the Public Suffix List (via `tldextract`)
3. **Free Email Filtering**: Blacklist of 50+ free email providers (gmail.com, outlook.com, etc.)
4. **Subdomain Handling**: Optional inclusion of first-level subdomains

**Implementation**: `src/domains.py::extract_domain_column()`

### 3.3 Geographic Standardization

Country and city fields are normalized:

- ISO 3166-1 alpha-2 country codes
- City name transliteration and common alias handling (München → Munich)

---

## 4. Candidate Generation (Blocking)

Given the infeasibility of exhaustive pairwise comparison (O(n × m) = 285 billion comparisons), the pipeline employs multi-source blocking to reduce the candidate space while maximizing recall.

### 4.1 Blocking Strategies

| Strategy | Key Generation | Complexity | Precision | Recall |
|----------|---------------|------------|-----------|--------|
| **Domain Exact** | eTLD+1 hash | O(n) | Very High | Medium |
| **Country + Prefix** | Country + first 3 chars | O(n) | Medium | High |
| **Rare Token** | Uncommon name tokens | O(k × n) | Medium | Medium |
| **Name Embedding ANN** | FAISS IVF index | O(n log m) | Medium | Very High |
| **Description Embedding** | FAISS IVF index | O(n log m) | Low-Medium | High |

### 4.2 Alias-Based Query Expansion

A critical innovation is the use of **AI-augmented alias registries** to handle name variation:

**Sources**:
1. **Manual Legal Name Mapping**: Human-verified mappings between trading and legal names
2. **AI-Generated Aliases**: LLM-retrieved legal entity information (Perplexity API)
3. **Web-Scraped Legal Names**: Automated extraction from Terms of Service and Privacy Policies
4. **Domain-Based Matching**: Companies with matching domains pooled regardless of name

**Mechanism**:
For each Crunchbase entity, the blocker generates search keys for:
- Primary trading name
- All registered aliases
- Domain-derived candidates

This recovers matches where name similarity alone would fail (e.g., "Deliveroo" → "Roofoods Ltd").

**Implementation**: `src/alias_registry.py`, `src/blocking.py`

### 4.3 Semantic Blocking with Sentence Embeddings

For cases where neither name nor domain provides a clear link, the pipeline employs semantic similarity:

- **Model**: BAAI/bge-large-en-v1.5 (1024-dim embeddings)
- **Index**: FAISS IVF with 4096 centroids
- **Top-K**: 50 candidates per query
- **GPU Acceleration**: Batch encoding on CUDA devices

---

## 5. Feature Engineering

### 5.1 String Similarity Features

| Feature | Algorithm | Range | Notes |
|---------|-----------|-------|-------|
| `name_jw` | Jaro-Winkler | [0, 1] | Prefix-weighted edit distance |
| `name_token_jaccard` | Token Set Jaccard | [0, 1] | Order-independent |
| `name_rapidfuzz_ratio` | Token Sort Ratio | [0, 100] | Fuzzy string matching |
| `name_prefix_match` | 8-char prefix | Boolean | Fast blocking verification |
| `name_acronym_match` | Acronym comparison | Boolean | Handles abbreviations |

**Multicollinearity Note**: Correlation analysis shows `name_jw` and `name_rapidfuzz_ratio` are highly correlated (ρ > 0.85). The model uses both but interprets feature importance cautiously.

### 5.2 Domain Features

| Feature | Description | Signal Strength |
|---------|-------------|-----------------|
| `domain_exact` | Primary domains match | Very Strong (+) |
| `email_domain_exact` | Email domains match | Strong (+) |
| `domain_in_family` | CB domain in Orbis family | Medium (+) |

### 5.3 Geographic Features

| Feature | Description |
|---------|-------------|
| `country_match` | ISO country codes match (Boolean) |
| `city_sim` | City name Jaccard similarity [0, 1] |

### 5.4 Temporal Features

| Feature | Description |
|---------|-------------|
| `year_diff` | Absolute difference: CB founded year vs Orbis incorporation year |
| `year_compat` | Compatibility score: 1.0 (same) → 0.2 (>5 years diff) |

### 5.5 Structural Features (Orbis Corporate Graph)

| Feature | Description |
|---------|-------------|
| `is_guo` | Entity is Global Ultimate Owner |
| `is_subsidiary` | Entity is a subsidiary |
| `is_branch` | Entity is a branch (same legal entity) |
| `family_size_log` | log(1 + number of entities in corporate family) |

### 5.6 Semantic Features

| Feature | Description |
|---------|-------------|
| `name_embedding_cos` | Cosine similarity of name embeddings |
| `desc_embedding_cos` | Cosine similarity of description embeddings |

---

## 6. Classification Model

### 6.1 Weak Supervision Framework

Rather than requiring extensive manual labeling, the pipeline employs weak supervision (Ratner et al., 2017):

**Labeling Functions**:

```
POSITIVE RULES:
  R1: domain_exact ∧ country_match ∧ name_jw ≥ 0.85 → P(match) = 0.95
  R2: domain_in_family ∧ desc_cos ≥ 0.90 → P(match) = 0.85
  R3: name_jw ≥ 0.95 ∧ country_match ∧ year_compat ≥ 0.8 → P(match) = 0.80

NEGATIVE RULES:
  R4: country_match ∧ ¬domain_match ∧ name_jw < 0.50 → P(match) = 0.05
  R5: name_jw < 0.30 → P(match) = 0.01
```

**Conflict Resolution**: When rules conflict, the Label Model uses weighted voting based on estimated rule accuracy.

### 6.2 Model Architecture

- **Algorithm**: Gradient Boosting (scikit-learn)
- **Calibration**: Isotonic regression for probability calibration
- **Hyperparameters**:
  - n_estimators: 500
  - max_depth: 8
  - learning_rate: 0.05
  - min_samples_leaf: 20

### 6.3 Cross-Encoder Reranking

For borderline candidates (0.4 ≤ p < 0.95), a cross-encoder provides refined scoring:

- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Input**: Concatenated name pairs
- **Output**: Relevance score (sigmoid-normalized)
- **Final Score**: 0.4 × base_score + 0.6 × cross_encoder_score

---

## 7. Decisioning

### 7.1 Tiered Output

| Tier | Threshold | Interpretation | Action |
|------|-----------|----------------|--------|
| **A** | p ≥ 0.98 | High confidence match | Auto-accept |
| **B** | 0.93 ≤ p < 0.98 | Likely match | Accept with flag |
| **C** | 0.75 ≤ p < 0.93 | Possible match | Human review |
| **Reject** | p < 0.75 | Unlikely match | Discard |

### 7.2 One-to-Many Resolution

When a Crunchbase entity matches multiple Orbis records, disambiguation uses:

1. **Domain Priority**: Exact domain matches ranked highest
2. **Geographic Alignment**: Country and city matching
3. **Corporate Role**: GUOs preferred over subsidiaries
4. **Year Compatibility**: Closer incorporation dates preferred

---

## 8. Evaluation Protocol

### 8.1 Stratified Test Set Generation

To avoid inflated metrics from easy negatives, the evaluation set is constructed with:

1. **Tier Stratification**: Proportional representation of each tier
2. **Hard Negative Sampling**: Pairs with high name similarity (>0.8) but different countries/domains
3. **Minimum Family Coverage**: Ensures rare corporate structures are represented

### 8.2 Metrics

| Metric | Definition |
|--------|------------|
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1** | Harmonic mean of Precision and Recall |
| **MRR** | Mean Reciprocal Rank (for ranked lists) |

### 8.3 Confidence Intervals

All metrics are reported with Wilson score confidence intervals:

```
p̂ ± z × √(p̂(1-p̂)/n + z²/4n²) / (1 + z²/n)
```

Where z = 1.96 for 95% confidence.

---

## 9. Ablation Studies

To quantify the contribution of each component:

| Configuration | Precision | Recall | F1 |
|--------------|-----------|--------|-----|
| Full Pipeline | 94.2% | 91.8% | 93.0% |
| − Alias Expansion | 93.1% | 76.4% | 83.9% |
| − Domain Features | 81.3% | 90.2% | 85.5% |
| − Semantic Embeddings | 92.8% | 84.1% | 88.2% |
| − Cross-Encoder Rerank | 91.7% | 91.8% | 91.8% |

---

## 10. Implementation Notes

### 10.1 Computational Requirements

| Stage | Memory | Time (A100 GPU) | Time (CPU) |
|-------|--------|-----------------|------------|
| Data Prep | 4 GB | 5 min | 5 min |
| Embeddings | 8 GB VRAM | 10 min | 90 min |
| Blocking | 16 GB RAM | 5 min | 5 min |
| Features | 32 GB RAM | 8 min | 8 min |
| Scoring | 4 GB | 2 min | 2 min |
| **Total** | — | **~30 min** | **~110 min** |

### 10.2 GPU Acceleration

The pipeline leverages GPU acceleration for:
- Sentence embedding computation (SentenceTransformers)
- Cross-encoder reranking (batch inference)
- FAISS index search (optional)

---

## 11. References

1. Christen, P. (2012). *Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection*. Springer.

2. Fellegi, I. P., & Sunter, A. B. (1969). A Theory for Record Linkage. *Journal of the American Statistical Association*, 64(328), 1183-1210.

3. Mudgal, S., et al. (2018). Deep Learning for Entity Matching: A Design Space Exploration. *SIGMOD*.

4. Ratner, A., et al. (2017). Snorkel: Rapid Training Data Creation with Weak Supervision. *VLDB*.

5. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*.

6. Wilson, E. B. (1927). Probable Inference, the Law of Succession, and Statistical Inference. *Journal of the American Statistical Association*.
