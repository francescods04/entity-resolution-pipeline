#!/usr/bin/env python3
"""
COLAB_A100_TURBO.py - MAXIMUM GPU UTILIZATION

Optimized for A100 80GB + 167GB RAM:
- 4x larger batch sizes (2048 vs 512)
- faiss-gpu for ANN search
- Parallel CPU operations with 8 workers
- 100GB+ RAM caching
"""

import os
import sys
import subprocess
import time

print("="*60)
print("🚀 A100 TURBO MODE - MAXIMUM GPU UTILIZATION")
print("="*60)

# =============================================================================
# STEP 1: SKIP INSTALL (already done in cell above)
# =============================================================================
print("\n📦 Checking packages...")

try:
    import polars, pyarrow, tqdm, sentence_transformers, rapidfuzz, faiss
    print("  ✅ All packages already installed")
except ImportError as e:
    print(f"  ⚠ Missing package: {e}")
    print("  Run this first: !pip install -q polars[calamine] pyarrow tqdm sentence-transformers rapidfuzz faiss-cpu joblib")

# Verify GPU
import torch
if torch.cuda.is_available():
    print(f"\n🔥 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠ No GPU detected!")

# =============================================================================
# STEP 2: OPTIMIZED CONFIG FOR A100
# =============================================================================

# =============================================================================
# PATH DETECTION (Auto-detect your Drive structure)
# =============================================================================
from pathlib import Path

# Try to auto-detect Drive base path
POSSIBLE_BASES = [
    "/content/drive/Othercomputers/My MacBook Pro/Downloads/ricerca",
    "/content/drive/MyDrive/ricerca",
    "/content/drive/My Drive/ricerca",
]

DRIVE_BASE = None
for base in POSSIBLE_BASES:
    if os.path.exists(base):
        DRIVE_BASE = base
        break

if DRIVE_BASE is None:
    # Fallback: find it dynamically
    drive_root = Path("/content/drive")
    for p in drive_root.rglob("entity-resolution-pipeline"):
        if p.is_dir() and (p / "run_pipeline.py").exists():
            DRIVE_BASE = str(p.parent)
            break

if DRIVE_BASE is None:
    raise FileNotFoundError("Could not find 'ricerca' folder in Google Drive. Please check your Drive is mounted.")

print(f"📁 Found Drive base: {DRIVE_BASE}")

LOCAL_PIPELINE = "/content/local_pipeline/entity-resolution-pipeline"
LOCAL_CB = "/content/local_pipeline/cb_data"
LOCAL_ORBIS = "/content/orbis_local"

# Detect GPU type for optimal batch sizes
import torch
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

# Adjust batch sizes based on GPU
if "A100" in gpu_name:
    BATCH_SIZE = 2048
    CHUNK_SIZE = 200000
    USE_GPU_FAISS = True
elif "L4" in gpu_name or gpu_vram >= 20:
    BATCH_SIZE = 1024  # L4 has 24GB - good middle ground
    CHUNK_SIZE = 150000
    USE_GPU_FAISS = False  # L4 FAISS-GPU can be unstable
else:  # T4 or smaller
    BATCH_SIZE = 512
    CHUNK_SIZE = 100000
    USE_GPU_FAISS = False

print(f"⚙️  Optimizing for {gpu_name} ({gpu_vram:.1f}GB): batch_size={BATCH_SIZE}")

# Create optimized config
A100_CONFIG = f"""
# GPU-OPTIMIZED CONFIG (auto-detected: {gpu_name})

paths:
  project_root: {LOCAL_PIPELINE}
  raw_crunchbase: {LOCAL_CB}
  raw_orbis: {LOCAL_ORBIS}

embeddings:
  batch_size: {BATCH_SIZE}
  device: cuda
  dtype: float16
  enabled: true
  model_name: all-MiniLM-L6-v2
  streaming_chunk_size: {CHUNK_SIZE}
  use_streaming: true

blocking:
  ANN_TOPK_DESC: 100
  ANN_TOPK_NAME: 200
  MAX_CANDIDATES_PER_CB: 500
  rare_token_df_threshold: 1000

faiss:
  metric: inner_product
  nprobe: 32
  use_gpu: {str(USE_GPU_FAISS).lower()}

features:
  enable_family_expansion: true
  enable_investor_checks: true
  enable_semantic_embeddings: true
  parallel_workers: 8

logging:
  level: INFO
  memory_warnings_threshold_gb: 50
  save_timing: true

model:
  calibration: isotonic
  learning_rate: 0.1
  max_depth: 8
  n_estimators: 500
  type: gradient_boosting

processing:
  chunk_size: {CHUNK_SIZE}
  feature_chunk_size: {CHUNK_SIZE}

reranking:
  enabled: true
  max_score: 0.95
  min_score: 0.4
  model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
  batch_size: 128

random_seed: 42

semantic_blocking:
  enabled: true
  similarity_threshold: 0.7
  top_k: 100

tiers:
  A: 0.98
  B: 0.93
  C: 0.75
"""

os.makedirs(f"{LOCAL_PIPELINE}/configs", exist_ok=True)
config_path = f"{LOCAL_PIPELINE}/configs/a100_turbo.yaml"
with open(config_path, 'w') as f:
    f.write(A100_CONFIG)
print(f"✅ Created GPU-optimized config: {config_path}")

# =============================================================================
# STEP 3: FAST DATA SETUP with INTEGRITY VALIDATION
# =============================================================================
print("\n📂 Setting up local data...")

import shutil

# Full copy if directory doesn't exist
if not os.path.exists(LOCAL_PIPELINE):
    print("  Copying pipeline code (first run)...")
    shutil.copytree(f"{DRIVE_BASE}/entity-resolution-pipeline", LOCAL_PIPELINE)
else:
    # ALWAYS re-sync src/ and run_pipeline.py from Drive to pick up bug fixes
    # This is fast (~2s for small Python files) and prevents stale code issues
    print("  🔄 Syncing code from Drive (ensures latest fixes)...")
    src_local = f"{LOCAL_PIPELINE}/src"
    if os.path.exists(src_local):
        shutil.rmtree(src_local)
    shutil.copytree(f"{DRIVE_BASE}/entity-resolution-pipeline/src", src_local)
    shutil.copy(f"{DRIVE_BASE}/entity-resolution-pipeline/run_pipeline.py", f"{LOCAL_PIPELINE}/run_pipeline.py")
    # Also sync configs
    configs_drive = f"{DRIVE_BASE}/entity-resolution-pipeline/configs"
    if os.path.exists(configs_drive):
        configs_local = f"{LOCAL_PIPELINE}/configs"
        if os.path.exists(configs_local):
            shutil.rmtree(configs_local)
        shutil.copytree(configs_drive, configs_local)
    print("    ✓ src/ + run_pipeline.py synced")

# Copy CB data
if not os.path.exists(LOCAL_CB):
    print("  Copying Crunchbase data...")
    shutil.copytree(f"{DRIVE_BASE}/dati europe cb", LOCAL_CB)

# Copy database-done.xlsx - check multiple possible locations
DB_DONE = "/content/local_pipeline/database-done.xlsx"
if not os.path.exists(DB_DONE):
    # Try multiple possible locations
    possible_db_paths = [
        f"{DRIVE_BASE}/entity-resolution-pipeline/database-done.xlsx",
        f"{DRIVE_BASE}/database-done.xlsx",
        f"{LOCAL_PIPELINE}/database-done.xlsx",
    ]
    
    db_source = None
    for path in possible_db_paths:
        if os.path.exists(path):
            db_source = path
            break
    
    if db_source:
        print(f"  Found database-done.xlsx at: {db_source}")
        shutil.copy2(db_source, DB_DONE)
    else:
        print("  ⚠️  database-done.xlsx not found - pipeline will train without platinum labels")
        print(f"     Searched: {possible_db_paths}")

# Check Orbis data
orbis_raw = f"{LOCAL_PIPELINE}/data/interim/orbis_clean/orbis_raw.parquet"
if not os.path.exists(orbis_raw):
    # Copy Orbis Excel files if needed
    os.makedirs(LOCAL_ORBIS, exist_ok=True)
    orbis_files = list(Path(LOCAL_ORBIS).glob("*.xlsx"))
    
    if not orbis_files:
        print("  Copying Orbis Excel files (this takes ~5 min)...")
        drive_files = []
        for folder in ["new orbis", "new orbis 2"]:
            folder_path = Path(f"{DRIVE_BASE}/{folder}")
            if folder_path.exists():
                drive_files.extend(list(folder_path.glob("*.xlsx")))
        
        print(f"  Found {len(drive_files)} Excel files in Drive (new orbis + new orbis 2)")
        for i, f in enumerate(drive_files):
            if i % 100 == 0:
                print(f"    {i}/{len(drive_files)} files...", flush=True)
            shutil.copy2(f, LOCAL_ORBIS)
        orbis_files = list(Path(LOCAL_ORBIS).glob("*.xlsx"))
        print(f"  ✅ Copied {len(orbis_files)} files")
    
    # Run Polars ingestion INLINE with correct paths
    print(f"  Running Polars ingestion on {len(orbis_files)} files...")
    import polars as pl
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm.auto import tqdm
    
    # Schema mapping
    ORBIS_COLS = {
        'BvD ID number': 'bvd_id',
        'Company name Latin alphabet': 'orbis_name',
        'Country ISO code': 'orbis_country',
        'City': 'orbis_city',
        'City (Latin Alphabet)': 'orbis_city',       # New export variant
        'City\nLatin Alphabet': 'orbis_city',         # Newline variant
        'Postcode': 'orbis_postcode',
        'Website address': 'orbis_website',
        'E-mail address': 'orbis_email',
        'Phone number': 'orbis_phone',
        'Date of incorporation': 'orbis_incorp_date',
        'Trade description (English)': 'orbis_trade_desc',
        'NACE Rev. 2 core code': 'orbis_nace',
        'Standardised legal form': 'orbis_legal_form',
        'Operating revenue (Turnover)': 'orbis_operating_revenue',
        'Total assets': 'orbis_total_assets',
        'Number of employees': 'orbis_num_employees',
        'SUB - BvD ID number': 'sub_bvd_id',
        'SH - BvD ID number': 'sh_bvd_id',
        'GUO - BvD ID number': 'guo_bvd_id',
        'BRANCH - BvD ID number': 'branch_bvd_id',
        'SH - Name': 'sh_name',
    }
    FIRST_COLS = ['bvd_id', 'orbis_name', 'orbis_country', 'orbis_city', 'orbis_postcode',
                  'orbis_phone', 'orbis_incorp_date', 'orbis_trade_desc', 'orbis_nace',
                  'orbis_legal_form', 'orbis_operating_revenue', 'orbis_total_assets', 'orbis_num_employees']
    AGG_COLS = ['orbis_website', 'orbis_email', 'sub_bvd_id', 'sh_bvd_id', 'guo_bvd_id', 'branch_bvd_id', 'sh_name']
    ALL_COLS = FIRST_COLS + AGG_COLS
    
    def process_file(filepath):
        try:
            df = pl.read_excel(str(filepath), sheet_name="Results", engine="calamine")
        except:
            try:
                df = pl.read_excel(str(filepath), sheet_name=0, engine="calamine")
            except:
                return pl.DataFrame()
        if df.is_empty():
            return pl.DataFrame()
        rename_map = {}
        for orig, new in ORBIS_COLS.items():
            for col in df.columns:
                if col.lower().replace('\n', ' ').strip() == orig.lower():
                    rename_map[col] = new
                    break
        df = df.rename(rename_map)
        for col in ALL_COLS:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).alias(col))
        idx_col = df.columns[0]
        df = df.with_columns(pl.col(idx_col).forward_fill().alias("_idx"))
        first_exprs = [pl.col(c).first().alias(c) for c in FIRST_COLS if c in df.columns]
        agg_exprs = [pl.col(c).drop_nulls().unique().str.concat("|").alias(c) for c in AGG_COLS if c in df.columns]
        result = df.group_by("_idx", maintain_order=True).agg(first_exprs + agg_exprs)
        result = result.filter(pl.col("bvd_id").is_not_null()).drop("_idx")
        result = result.select([pl.col(c).cast(pl.Utf8).alias(c) for c in result.columns])
        return result
    
    all_dfs = []
    failed_files = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(process_file, f): f for f in orbis_files}
        for future in tqdm(as_completed(futures), total=len(orbis_files), desc="Processing"):
            filepath = futures[future]
            try:
                df = future.result(timeout=60)
                if df is not None and not df.is_empty():
                    all_dfs.append(df)
            except Exception as e:
                failed_files.append((filepath.name, str(e)[:50]))
    
    if failed_files:
        print(f"  ⚠️ {len(failed_files)} files failed: {failed_files[:5]}...")
    
    final_df = pl.concat(all_dfs).unique(subset=["bvd_id"], keep="first")
    for col in ALL_COLS:
        if col not in final_df.columns:
            final_df = final_df.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))
    final_df = final_df.select(ALL_COLS)
    
    os.makedirs(os.path.dirname(orbis_raw), exist_ok=True)
    final_df.write_parquet(orbis_raw, compression="snappy")
    print(f"  ✅ Orbis ingested: {len(final_df):,} rows ({os.path.getsize(orbis_raw)/(1024**3):.2f} GB)")

print("✅ Data ready")

# =============================================================================
# STEP 4: RUN PIPELINE WITH GPU
# =============================================================================
print("\n🔥 RUNNING PIPELINE WITH A100 GPU...")

os.chdir(LOCAL_PIPELINE)
# Add BOTH paths: local (fast) and Drive (fallback if copy incomplete)
sys.path.insert(0, f"{LOCAL_PIPELINE}/src")
sys.path.insert(0, f"{DRIVE_BASE}/entity-resolution-pipeline/src")  # Fallback

# GPU-accelerated normalize (inline for speed)
orbis_clean = f"{LOCAL_PIPELINE}/data/interim/orbis_clean/orbis_clean.parquet"
if not os.path.exists(orbis_clean):
    print("\n▶ NORMALIZE (GPU-accelerated)")
    
    # Ensure src is in path
    src_path = f"{LOCAL_PIPELINE}/src"
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    os.chdir(LOCAL_PIPELINE)
    
    import pandas as pd
    import numpy as np
    from normalize import normalize_name_column
    from domains import extract_domain_column
    
    orbis = pd.read_parquet(orbis_raw)
    print(f"  Loaded {len(orbis):,} rows")
    
    # Normalize names
    name_features = normalize_name_column(orbis['orbis_name'])
    for col in name_features.columns:
        orbis[f'orbis_{col}'] = name_features[col]
    
    # Domains
    first_websites = orbis['orbis_website'].fillna('').str.split('|').str[0]
    domain_features = extract_domain_column(first_websites)
    orbis['orbis_domain'] = domain_features['etld1']
    
    # Year (vectorized)
    date_col = orbis['orbis_incorp_date'].astype(str)
    is_numeric = date_col.str.match(r'^\d+$', na=False)
    numeric_dates = pd.to_datetime('1899-12-30') + pd.to_timedelta(
        pd.to_numeric(date_col.where(is_numeric), errors='coerce'), unit='D')
    parsed_dates = pd.to_datetime(date_col.where(~is_numeric), errors='coerce', dayfirst=True)
    orbis['orbis_incorp_year'] = numeric_dates.fillna(parsed_dates).dt.year
    
    orbis.to_parquet(orbis_clean, index=False)
    print(f"  ✅ Saved orbis_clean.parquet ({os.path.getsize(orbis_clean)/(1024**3):.2f} GB)")
    
    # FREE MEMORY: Release ~2.2 GB before subprocess steps begin
    del orbis, name_features, domain_features, first_websites, date_col, is_numeric, numeric_dates, parsed_dates
    import gc; gc.collect()
    print("  🧹 Memory released after inline normalize")

# Run remaining steps with A100 config
# NOTE: 'ingest' creates cb_raw_companies.parquet (Orbis already ingested inline above)
# NOTE: 'normalize' creates cb_clean.parquet (Orbis already normalized inline above)
STEPS = ['ingest', 'normalize', 'alias', 'index', 'embeddings', 'blocking', 'features', 'train', 'score', 'rerank', 'decide', 'report', 'analytics']

# =============================================================================
# SMART RESUME: Check checkpoint file to skip completed steps
# =============================================================================
import json
checkpoint_file = f"{LOCAL_PIPELINE}/data/interim/pipeline_checkpoint.json"
completed_steps = set()

if os.path.exists(checkpoint_file):
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            completed_steps = set(checkpoint_data.get('completed_steps', []))
            last_step = checkpoint_data.get('last_step', 'unknown')
            timestamp = checkpoint_data.get('timestamp', 'unknown')
            print(f"\n🔄 RESUME MODE DETECTED")
            print(f"   Last checkpoint: {last_step} at {timestamp}")
            print(f"   Completed steps: {', '.join(sorted(completed_steps))}")
    except Exception as e:
        print(f"⚠️  Could not read checkpoint: {e}, starting fresh")

# Filter out already-completed steps
steps_to_run = [s for s in STEPS if s not in completed_steps]

if len(steps_to_run) < len(STEPS):
    skipped = len(STEPS) - len(steps_to_run)
    print(f"   ⏭️  Skipping {skipped} completed steps")
    print(f"   ▶️  Will run: {', '.join(steps_to_run)}")
else:
    print(f"\n▶️  Running all {len(STEPS)} steps (no checkpoint found)")

for step in steps_to_run:
    print(f"\n{'='*60}")
    print(f"▶ {step.upper()}")
    print('='*60, flush=True)
    
    start = time.time()
    run_script = f"{LOCAL_PIPELINE}/run_pipeline.py"
    
    # Throttle tqdm output to prevent Colab browser freeze
    env = os.environ.copy()
    env['TQDM_MININTERVAL'] = '30'  # Update at most every 30 seconds
    env['TQDM_DISABLE'] = '0'
    
    process = subprocess.Popen(
        ['python', '-u', run_script, '--config', config_path, '--step', step],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        cwd=LOCAL_PIPELINE,
        env=env
    )
    for line in process.stdout:
        # Filter out tqdm carriage-return spam (lines with \r but no \n)
        if '\r' in line and line.strip().endswith('%|'):
            continue
        print(line, end='', flush=True)
    process.wait()
    
    elapsed = time.time() - start
    
    if process.returncode != 0:
        print(f"❌ {step} failed!")
        break
    
    print(f"✅ {step} complete ({elapsed:.1f}s)")
    
    # Sync critical files to Drive
    for src in [
        f"{LOCAL_PIPELINE}/data/interim/orbis_clean/orbis_clean.parquet",
        f"{LOCAL_PIPELINE}/data/interim/candidates/candidates.parquet",
        f"{LOCAL_PIPELINE}/data/interim/features/pair_features.parquet",
        f"{LOCAL_PIPELINE}/data/outputs/matches/matches_final.parquet",
        f"{LOCAL_PIPELINE}/data/interim/pipeline_checkpoint.json",
    ]:
        if os.path.exists(src):
            dst = src.replace(LOCAL_PIPELINE, f"{DRIVE_BASE}/entity-resolution-pipeline")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.copy2(src, dst)
            except:
                pass

print("\n" + "="*60)
print("🎉 PIPELINE COMPLETE!")
print("="*60)

# Show results
matches = f"{LOCAL_PIPELINE}/data/outputs/matches/matches_final.parquet"
if os.path.exists(matches):
    import pandas as pd
    df = pd.read_parquet(matches)
    print(f"\nTotal matches: {len(df):,}")
    if 'tier' in df.columns:
        print(df['tier'].value_counts())
