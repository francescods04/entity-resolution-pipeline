# Data Dictionary

This document provides a comprehensive schema reference for all data files used in the entity resolution pipeline.

---

## 1. Input Data Sources

### 1.1 Crunchbase Export

**Location**: `dati europe cb/`

**Files**:
- `organizations.csv` — Primary company data
- `funding_rounds.csv` — Investment round details
- `investors.csv` — Investor entities
- `acquisitions.csv` — M&A events

#### organizations.csv

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `uuid` | String | Unique identifier | `a1b2c3d4-...` |
| `name` | String | Trading/brand name | `Spotify` |
| `legal_name` | String | Registered legal name (often null) | `Spotify AB` |
| `domain` | String | Primary website domain | `spotify.com` |
| `homepage_url` | String | Full URL | `https://www.spotify.com` |
| `country_code` | String | ISO 3166-1 alpha-2 | `SE` |
| `city` | String | Headquarters city | `Stockholm` |
| `short_description` | String | Company tagline | `Music streaming service` |
| `founded_on` | Date | Founding date | `2006-04-23` |
| `employee_count` | String | Size bucket | `1001-5000` |
| `status` | String | Operating status | `operating` |
| `category_list` | String | Industry tags (comma-separated) | `Music,Streaming,Entertainment` |

**Notes**:
- Column names vary across export versions (e.g., `Organization Name` vs `name`)
- The data loader (`src/data_loader.py`) handles schema variations automatically

---

### 1.2 Orbis Export

**Location**: `new orbis/`

**Format**: Excel files (`.xlsx`), approximately 944 files with ~17,000 records each

#### Results Sheet Schema

| Column | Normalized Name | Type | Description |
|--------|-----------------|------|-------------|
| `BvD ID number` | `bvd_id` | String | Bureau van Dijk unique identifier |
| `Company name Latin alphabet` | `orbis_name` | String | Official registered name |
| `Country ISO code` | `orbis_country` | String | ISO 3166-1 alpha-2 |
| `City` | `orbis_city` | String | Registered address city |
| `Postcode` | `orbis_postcode` | String | Postal code |
| `Website address` | `orbis_website` | String | Company websites (pipe-separated if multiple) |
| `E-mail address` | `orbis_email` | String | Contact emails (pipe-separated) |
| `Phone number` | `orbis_phone` | String | Contact phone |
| `Date of incorporation` | `orbis_incorp_date` | Date/String | Registration date |
| `Trade description (English)` | `orbis_trade_desc` | String | Business description |
| `NACE Rev. 2 core code` | `orbis_nace` | String | Industry classification |
| `Standardised legal form` | `orbis_legal_form` | String | Legal entity type |

**Multi-Row Structure**:
Orbis exports use a multi-row format where:
- Row 1: Primary data
- Row 2+: Additional websites, emails, phone numbers

The ingestion process (`POLARS_FAST_INGEST.py`) aggregates these into pipe-separated values.

---

### 1.3 Alias Registry

**Location**: `database-done.xlsx`

This file contains human-verified and AI-generated mappings between Crunchbase brand names and Orbis legal entities.

#### Sheets

| Sheet | Purpose | Columns |
|-------|---------|---------|
| `companies base` | Core Crunchbase companies | `cb_id`, `company_name`, `domain` |
| `Matching ai` | AI-retrieved legal names | `company_name`, `legal_entity_orbis`, `source` |
| `Matching platinum` | Domain-matched high-confidence pairs | `cb_id`, `bvd_id`, `confidence` |
| `Matching manuale` | Human-verified matches | `cb_id`, `bvd_id`, `notes` |
| `vat_matches` | VAT code based matches | `cb_id`, `bvd_id`, `vat_code` |

**Alias Loading Priority**:
1. Platinum matches (domain + manual verification) — Highest confidence
2. VAT matches — Tax ID based
3. AI matches — LLM-generated
4. Manual matches — Human annotated

---

## 2. Intermediate Data

### 2.1 Normalized Crunchbase

**Path**: `data/interim/cb_clean/cb_clean.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `cb_id` | String | Original UUID |
| `cb_name` | String | Normalized trading name |
| `cb_name_clean` | String | Cleaned, tokenized name |
| `cb_domain` | String | eTLD+1 extracted domain |
| `cb_country_iso` | String | ISO country code |
| `cb_city_norm` | String | Normalized city name |
| `cb_founded_year` | Int | Extracted founding year |
| `cb_desc` | String | Short description |

### 2.2 Normalized Orbis

**Path**: `data/interim/orbis_clean/orbis_clean.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `bvd_id` | String | Bureau van Dijk ID |
| `orbis_name` | String | Original registered name |
| `orbis_name_clean` | String | Normalized name |
| `orbis_domain` | String | Primary website eTLD+1 |
| `orbis_country` | String | ISO country code |
| `orbis_city` | String | Registered city |
| `orbis_incorp_year` | Int | Incorporation year |
| `orbis_legal_form` | String | Legal entity type |

### 2.3 Embeddings

**Path**: `data/interim/embeddings/`

| File | Shape | Description |
|------|-------|-------------|
| `cb_name_emb.npy` | (N, 1024) | Crunchbase name embeddings |
| `cb_desc_emb.npy` | (N, 1024) | Crunchbase description embeddings |
| `orbis_name_emb.npy` | (M, 1024) | Orbis name embeddings |
| `*_index.parquet` | — | ID-to-row mapping |

**Model**: BAAI/bge-large-en-v1.5

### 2.4 Candidate Pairs

**Path**: `data/interim/blocking/candidates.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `cb_id` | String | Crunchbase identifier |
| `bvd_id` | String | Orbis identifier |
| `blocking_source` | String | How the pair was generated |
| `blocking_score` | Float | Initial similarity score |

**Blocking Sources**:
- `domain_exact` — Matching eTLD+1
- `country_prefix` — Same country + 3-char prefix
- `token_overlap` — Shared rare tokens
- `ann_name` — ANN embedding similarity
- `ann_desc` — Description embedding similarity
- `alias_registry` — From database-done.xlsx

### 2.5 Feature Matrix

**Path**: `data/interim/features/pair_features.parquet`

Contains all computed pairwise features (see Methodology §5 for complete feature list).

---

## 3. Output Data

### 3.1 Scored Candidates

**Path**: `data/outputs/scoring/scored_candidates.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `cb_id` | String | Crunchbase identifier |
| `bvd_id` | String | Orbis identifier |
| `p_match` | Float | Match probability [0, 1] |
| `tier` | String | Decision tier (A/B/C/Reject) |
| All feature columns | Various | For interpretability |

### 3.2 Final Matches

**Path**: `data/outputs/matches/matches_final.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `cb_id` | String | Crunchbase identifier |
| `cb_name` | String | Original CB name |
| `bvd_id` | String | Matched Orbis identifier |
| `orbis_name` | String | Matched Orbis name |
| `p_match` | Float | Final match probability |
| `tier` | String | Confidence tier |
| `match_type` | String | GUO/LEGAL_ENTITY/SUBSIDIARY/BRANCH |
| `evidence` | JSON | Match evidence summary |

### 3.3 Review Queue

**Path**: `data/outputs/matches/review_queue.csv`

Tier C matches requiring human review, formatted for manual inspection.

### 3.4 Run Manifest

**Path**: `data/outputs/run_manifest.json`

```json
{
  "run_id": "20260111_154530",
  "config": "configs/a100_turbo.yaml",
  "timestamps": {
    "start": "2026-01-11T15:45:30",
    "end": "2026-01-11T16:15:42"
  },
  "statistics": {
    "cb_count": 19234,
    "orbis_count": 15945484,
    "candidates_generated": 2134567,
    "matches_tier_a": 8934,
    "matches_tier_b": 2156,
    "matches_tier_c": 1045,
    "rejected": 2122432
  }
}
```

---

## 4. Configuration Files

### 4.1 Pipeline Configuration

**Path**: `configs/a100_turbo.yaml`

```yaml
paths:
  project_root: /content/local_pipeline/entity-resolution-pipeline
  raw_crunchbase: /content/local_pipeline/cb_data
  raw_orbis: /content/orbis_local

embeddings:
  model_name: BAAI/bge-large-en-v1.5
  batch_size: 4096
  device: cuda
  dtype: float16

blocking:
  MAX_CANDIDATES_PER_CB: 500
  ANN_TOPK_NAME: 50

features:
  parallel_workers: 12

model:
  n_estimators: 500
  max_depth: 8

tiers:
  A: 0.98
  B: 0.93
  C: 0.75
```

---

## 5. File Size Reference

| File | Typical Size | Records |
|------|--------------|---------|
| `orbis_raw.parquet` | 1.0 GB | 15.9M |
| `orbis_clean.parquet` | 1.2 GB | 15.9M |
| `cb_clean.parquet` | 15 MB | 19K |
| `candidates.parquet` | 200 MB | 2M |
| `pair_features.parquet` | 800 MB | 2M |
| `matches_final.parquet` | 5 MB | 12K |
