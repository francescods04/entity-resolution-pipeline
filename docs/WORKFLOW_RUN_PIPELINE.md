# Run Entity Resolution Pipeline

This workflow handles the end-to-end execution of the entity resolution pipeline, ensuring robust data loading and platinum-match-based training.

## 1. Prepare Data Staging
Merge the distributed Orbis files into a single batch for efficient processing.

```bash
cd /Users/francescodelsesto/Downloads/ricerca/entity-resolution-pipeline
# Optimizes ~878 Excel files into one parquet
python process_orbis_batch.py --input "../new orbis" --output data/raw/orbis_merged.parquet
```

## 2. Run Pipeline

Choose **Local** (easier) or **Colab** (faster for embeddings).

### Option A: Local Execution (Recommended for Data Prep & Testing)
**Best for**: Avoiding file uploads, debugging, and standard matching.
**Config**: `configs/local.yaml` (Pre-configured for your Mac paths)

```bash
# Verify it works
python run_pipeline.py --config configs/local.yaml --step ingest

# Run full pipeline (CPU optimized)
python run_pipeline.py --config configs/local.yaml
```

### Option B: Google Colab Execution
**Best for**: Deep Learning Embeddings (GPU required) or if Local is too slow.
**Config**: `configs/colab_gpu.yaml`

1.  **Upload Data**: Copy `data/raw/orbis_merged.parquet` and `database-done.xlsx` to Google Drive `ricerca/`.
2.  **Run**:
    ```bash
    python run_pipeline.py --config configs/colab_gpu.yaml
    ```

## 3. Step-by-Step (Debugging)

If you need to restart or check a specific step locally:

```bash
# 1. Ingest
python run_pipeline.py --config configs/local.yaml --step ingest

# 2. Block & Index (Uses database-done.xlsx aliases)
python run_pipeline.py --config configs/local.yaml --step blocking

# 3. Features & Training
python run_pipeline.py --config configs/local.yaml --step train
```

## 4. Verify Results


Check the output `matches_final.parquet` and the logs to confirm platinum matches were used.
```bash
ls -lh data/models/company_match/
ls -lh data/matches/
```
