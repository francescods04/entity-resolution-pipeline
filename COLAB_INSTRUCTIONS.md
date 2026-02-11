# 🚀 CPU-Only Colab Optimized Run Guide

This pipeline is optimized to run on **Google Colab** using the **Split Workflow** strategy to minimize costs (Zero GPU quota required for 80% of the work).

## 📋 Prerequisite (One-Time Setup)

1.  **Mount Drive** and **Navigate to Repo**:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Navigate to your repo folder (adjust path if needed)
    %cd "/content/drive/MyDrive/ricerca/entity-resolution-pipeline"
    ```

2.  **Update Code** (Important!)
    Run this to ensure you have the latest `COLAB_UNIVERSAL.py`:
    ```python
    !git pull
    ```

---

## 🏎️ The Split Workflow (Cost-Optimized)

Follow this 3-step sequence to save GPU hours.

### 🟢 PHASE 1: PREP (Run on CPU)
**Runtime:** `CPU` (High-RAM recommended)
**Cost:** Free
**Time:** ~15 mins

In a cell, run:
```python
# 1. PREP: Ingest, Normalize, Alias, Index
%run COLAB_UNIVERSAL.py --phase prep
```

*What it does:*
- Copies data from `new orbis` and `new orbis 2`.
- Normalizes names and domains.
- Builds the Alias Registry.
- Creating the search index.
*Stops automatically when done.*

---

### 🟡 PHASE 2: EMBEDDINGS (Run on GPU)
**Runtime:** Change Runtime Type -> `T4` (Standard) or `L4` (Premium)
**Cost:** Paid (Low cost)
**Time:** ~15-25 mins (L4/T4)

In a cell, run:
```python
# 2. EMBEDDINGS ONLY
%run COLAB_UNIVERSAL.py --phase embeddings
```

*What it does:*
- Computes 2.2M embeddings using the GPU.
- Saves them to Drive.
*Stops automatically when done.*

---

### 🟢 PHASE 3: FINISH (Run on CPU)
**Runtime:** Change Runtime Type -> `CPU` (High-RAM recommended)
**Cost:** Free
**Time:** ~30-45 mins

In a cell, run:
```python
# 3. FINISH: Blocking, Matching, Decisioning
%run COLAB_UNIVERSAL.py --phase finish
```

*What it does:*
- Re-uses the embeddings from Phase 2.
- Generates candidates (Blocking).
- Computes ML features.
- Scores and Decides (A/B/C tiers).
- Generates `matches_final.parquet`.

---

## ⚡ Option B: The "Just Run It" (All-in-One)

If you have a powerful runtime (A100) or don't care about CPU slowness (High-RAM CPU overnight), just run:

```python
%run COLAB_UNIVERSAL.py
```
*Auto-detects your hardware and runs everything start to finish.*
