# Entity Resolution Pipeline

Matching Crunchbase startups to their Orbis corporate records.

## What This Does

If you've ever tried to connect startup data from Crunchbase with Bureau van Dijk's Orbis database, you know it's a nightmare. Company names don't match, legal entities differ from trading names, and there are millions of potential false positives.

This pipeline solves that. It takes ~19,000 European VC-backed companies from Crunchbase and finds their corresponding records among ~16 million Orbis companies. The result? Accurate matches with confidence scores, ready for your research.

## Getting Started

### On Google Colab (Recommended)

The fastest way to run this is on Colab with a GPU. Just paste this in a cell:

```python
!pip install -q polars[calamine] pyarrow tqdm sentence-transformers rapidfuzz faiss-cpu

from google.colab import drive
drive.mount('/content/drive')

# Point this to wherever you stored the pipeline
%run "/content/drive/MyDrive/entity-resolution-pipeline/COLAB_A100_TURBO.py"
```

It takes about 30-40 minutes on an A100.

### Running Locally

If you prefer running on your own machine:

```bash
pip install -r requirements.txt
python run_pipeline.py --config configs/local.yaml
```

Expect 60-90 minutes on an M1 Mac or similar.

## How It Works

The pipeline runs through several stages:

1. **Normalize** — Cleans company names, extracts domains, standardizes formats
2. **Block** — Finds candidate pairs using multiple strategies (domain match, name similarity, embeddings)
3. **Score** — Computes features and runs them through a trained model
4. **Decide** — Assigns confidence tiers (A/B/C) or rejects

The key insight is that we use your existing manual matches (from `database-done.xlsx`) to train the model. The more platinum matches you have, the better it performs.

## What You Need

**Data:**
- Crunchbase export (CSV files in `dati europe cb/`)
- Orbis batch exports (Excel files in `new orbis/`)
- Your manual matches (optional but recommended: `database-done.xlsx`)

**Hardware:**
- 16GB+ RAM for local runs
- GPU for faster embeddings (optional but helps a lot)

## Output

After running, you'll find:

| File | What's in it |
|------|-------------|
| `matches_final.parquet` | All matches with confidence tiers |
| `review_queue.csv` | Borderline cases that need human review |
| `run_manifest.json` | Stats and metadata from the run |

## Project Layout

```
entity-resolution-pipeline/
├── src/                 # All the matching logic
├── configs/             # Settings for local vs GPU runs
├── docs/                # Methodology notes
├── COLAB_A100_TURBO.py  # One-click Colab script
└── run_pipeline.py      # Main entry point
```

## Docs

If you want to dig deeper:

- [Methodology](docs/METHODOLOGY.md) — The academic approach behind this
- [Data Dictionary](docs/DATA_DICTIONARY.md) — What each column means
- [Running the Pipeline](docs/WORKFLOW_RUN_PIPELINE.md) — Step-by-step guide

## Questions?

This was built for academic research on European startups and corporate governance. If you're using it for something similar and run into issues, the code is reasonably well-commented.

---

*For academic research purposes.*
