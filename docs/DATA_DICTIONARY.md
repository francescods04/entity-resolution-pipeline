# Data Dictionary & Schema Documentation

**Project**: Entity Resolution Pipeline (Crunchbase $\leftrightarrow$ Orbis)
**Date**: January 7, 2026

---

## 1. Primary Knowledge Base: `database-done.xlsx`

This file is the "Brain" of the pipeline, containing ground truth matches and AI-enriched aliases.

| Sheet Name | Description | Key Columns | Usage in Pipeline |
| :--- | :--- | :--- | :--- |
| **0. DB company crunchbase** | Base List of Crunchbase companies (Cleaned). | `Organization Name`, `Organization Website` | Reference list for CB entities. |
| **1. bvd id** | Mappings to BvD IDs (Orbis unique identifiers). | `company name` (Brand), `legal name`, `website` | Seed for alias registry. |
| **2. Matching 1 platinum...** | **Platinum Matches**: High-confidence pairs verified via strict domain matching. | `nome_df1` (Orbis), `nome_df2_match` (CB), `match_type` | **Training Data**: Used to train the logistic regression model (Supervised Learning). |
| **3. Matching 2** | **Web Scraping Results**: Legal names extracted from "Terms of Service" or footers. | `dominio`, `json_final_company_name` | **Blocking**: Expanded search keys. |
| **4. Matching manuale** | **Manual/AI Verification**: Difficult cases verified by human or AI. | `name ` (Brand), `company name` (Legal Desc) | **Blocking**: Expanded search keys. |
| **5. Matching ai** | **Perplexity API Results**: Generative AI-discovered legal entities. | `Name` (Brand), `legal_name` (AI Result) | **Blocking**: Expanded search keys. |

---

## 2. Input Data Sources

### A. Crunchbase Data (CSV Dump)
Collection of CSV files (`organizations.csv`, etc.). The robust loader normalizes these into a consistent schema.

| Normalized Field | Typ. Source Columns | Description |
| :--- | :--- | :--- |
| `cb_id` | `uuid`, `id` | Unique Crunchbase Identifier. |
| `cb_name` | `name`, `organization_name`, `company_name` | Commercial/Brand Name. |
| `cb_domain` | `domain`, `website`, `homepage_url` | Primary website domain. |
| `cb_country_iso` | `country_code`, `country_iso3` | ISO Country Code (e.g., USA, GBR). |
| `cb_city_norm` | `city`, `city_name` | Normalized city name. |
| `cb_founded_year` | `founded_on`, `founded_year` | Year of foundation. |

### B. Orbis Data (Excel Batch)
Consists of ~1,000 Excel files merged into a single Parquet file.

| Field Name | Description | Note |
| :--- | :--- | :--- |
| `bvd_id` | Bureau van Dijk ID | Unique Global Identifier. |
| `orbis_name` | Legal Entity Name | Often includes legal suffixes (Ltd, S.r.l). |
| `orbis_incorp_year` | Incorporation Year | Verified legal incorporation date. |
| `entity_role` | Corporate Hierarchy Role | `GUO` (Global Ultimate Owner), `SUBSIDIARY`, `BRANCH`. |
| `family_size` | Number of subsidiaries | Used as a feature (larger families = higher match prob). |

---

## 3. Internal Parquet Storage (`data/interim/`)

The pipeline converts "messy" raw inputs into optimized Parquet files for speed.

-   **`orbis_merged.parquet`**: All Orbis Excel files combined (10M+ rows).
-   **`cb_clean.parquet`**: Normalized Crunchbase data.
-   **`alias_registry.parquet`**: Flattened lookup table (Brand $\to$ Alias) derived from `database-done.xlsx`.
-   **`candidates.parquet`**: Generated potential matches (Pairs of `cb_id`, `bvd_id`).
-   **`features.parquet`**: Computed feature vectors for every candidate pair.
