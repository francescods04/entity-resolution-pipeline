# Entity Resolution Pipeline: Crunchbase ↔ Orbis

**PhD-Level Company Matching System for European VC-Backed Startups**

## Overview

This project implements a state-of-the-art entity resolution pipeline to match **Crunchbase startup data** with **Orbis corporate records** from Bureau van Dijk. The system uses multi-source candidate blocking, GPU-accelerated embeddings, and machine learning with weak supervision.

## Quick Start

### Prerequisites

- Python 3.10+
- Google Colab Pro (A100 GPU recommended)
- ~50GB storage for raw + processed data

### Installation

```bash
pip install -r requirements.txt
```

## Key Features

*   **PhD-Level Methodology**:
    *   **Learned Disambiguation**: Replaces heuristics with Logistic Regression trained on platinum matches.
    *   **Rigorous Evaluation**: Stratified sampling, hard-negative mining, and Wilson score confidence intervals.
    *   **Detailed Ablation Studies**: Automated framework to justify feature contributions.

*   **Advanced Matching**:
    *   **Multi-Source Blocking**: Uses `database-done.xlsx` aliases (Manual/AI) to find companies with different legal names.
    *   **Robust Data Ingestion**: Handles varying Crunchbase CSV schemas and "messy" Orbis batches.
    *   **Deep Semantic Matching**: (Optional) GPU-accelerated embedding search.

## Usage

### 1. Optimize Data
Combine dispersed Orbis Excel files into a high-performance Parquet file:
```bash
python process_orbis_batch.py --input "../new orbis" --output data/raw/orbis_merged.parquet
```

### 2. Run Pipeline
Execute the full matching pipeline. It automatically detects and uses your platinum matches for training.
```bash
python run_pipeline.py --config configs/colab_gpu.yaml
```

### 3. Analyze Results
Generate research-grade reports and ablation tables:
```bash
python run_pipeline.py --step analytics
```

## Documentation
*   [**PhD Methodology**](docs/PHD_METHODOLOGY_IMPROVEMENTS.md): Scientific justification of the pipeline's innovative features (Learned Disambiguation, AI-Augmented Indices).
*   [**Data Dictionary**](docs/DATA_DICTIONARY.md): Detailed schema of `database-done.xlsx` (Platinum Matches, AI Aliases) and input files.
*   [**Operations Guide**](docs/WORKFLOW_RUN_PIPELINE.md): Step-by-step commands to run the pipeline.

## Data Sources

| Source | Location | Records |
|--------|----------|---------|
| Crunchbase | `dati europe cb/` | ~19K companies |
| Orbis | `new orbis/` | ~15M companies (944 files × 16K each) |

## Key Features

- **Multi-Source Blocking**: Domain matching, ANN embeddings, token overlap
- **GPU Acceleration**: FAISS + SentenceTransformers on A100
- **Corporate Graph**: Orbis SUB/SH/GUO/BRANCH relationship handling
- **Tiered Decisioning**: A (auto) / B (high) / C (review) / Reject
- **Evidence Tracking**: JSON explanations for every match

## Output

| File | Description |
|------|-------------|
| `matches_final.parquet` | Final matches with tiers |
| `review_queue.csv` | Human review candidates |
| `quality_report.html` | Metrics dashboard |

## Documentation

- [Methodology](docs/METHODOLOGY.md) - Academic approach description
- [Data Dictionary](docs/DATA_DICTIONARY.md) - Schema documentation
- [Quality Metrics](docs/QUALITY_METRICS.md) - Precision/recall analysis

## Project Structure

```
entity-resolution-pipeline/
├── data/                    # Raw, interim, outputs
├── src/                     # Python modules
├── notebooks/               # Colab notebooks (00-09)
├── configs/                 # YAML configurations
└── docs/                    # Academic documentation
```

## License

For academic research purposes only.
