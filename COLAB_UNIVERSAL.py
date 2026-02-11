#!/usr/bin/env python3
"""
COLAB_UNIVERSAL.py — ONE SCRIPT, ANY HARDWARE

Auto-detects what's available and runs ALL 13 pipeline steps:
  • T4 GPU  (15 GB VRAM) — embeddings ~25 min, total ~1h
  • L4 GPU  (24 GB VRAM) — embeddings ~15 min, total ~45 min
  • CPU-only (High-RAM)  — embeddings ~90-180 min, total ~2-4h

Use this when you're low on A100 quota.
For A100, use COLAB_A100_TURBO.py instead.

RUNTIME SETUP:
  Runtime → Change runtime type → T4 / L4 / None (CPU)
  (Enable 'High-RAM' if on CPU for best results)
"""

import os, sys, subprocess, time, shutil, json

print("=" * 60)
print("🔍 UNIVERSAL LAUNCHER — Auto-detecting hardware...")
print("=" * 60)

# =============================================================================
# 1. DETECT HARDWARE + INSTALL PACKAGES
# =============================================================================

# Detect GPU
HAS_GPU = False
GPU_NAME = "None (CPU-only)"
GPU_VRAM = 0
try:
    import torch
    if torch.cuda.is_available():
        HAS_GPU = True
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_VRAM = torch.cuda.get_device_properties(0).total_memory / 1e9
except ImportError:
    pass

print(f"  🖥️  GPU: {GPU_NAME}")
if HAS_GPU:
    print(f"  💽  VRAM: {GPU_VRAM:.1f} GB")

# Install packages based on hardware
print("\n📦 Installing packages...")
base_pkgs = "polars[calamine] pyarrow tqdm rapidfuzz pyyaml openpyxl joblib sentence-transformers scikit-learn xgboost tldextract"
if HAS_GPU:
    os.system(f"pip install -q {base_pkgs} faiss-gpu")
else:
    os.system(f"pip install -q {base_pkgs} faiss-cpu")

# Verify
try:
    import polars, pyarrow, tqdm, rapidfuzz, faiss, tldextract
    from sentence_transformers import SentenceTransformer
    faiss_type = "GPU" if HAS_GPU else "CPU"
    print(f"  ✅ All packages ready (faiss-{faiss_type.lower()}, sentence-transformers)")
except ImportError as e:
    print(f"  ❌ Missing: {e}")
    sys.exit(1)

# RAM detection
import psutil
ram_gb = psutil.virtual_memory().total / (1024**3)
print(f"  💾 RAM: {ram_gb:.1f} GB")

# Hardware tier summary
if "A100" in GPU_NAME:
    HW_TIER = "A100"
    print("\n⚡ A100 detected — consider using COLAB_A100_TURBO.py for max performance.")
    print("   Continuing with universal config...")
elif "L4" in GPU_NAME or GPU_VRAM >= 20:
    HW_TIER = "L4"
    print(f"\n🟢 {GPU_NAME} — fast embeddings, moderate batches")
elif HAS_GPU:
    HW_TIER = "T4"
    print(f"\n🟡 {GPU_NAME} — GPU embeddings, conservative batches")
else:
    HW_TIER = "CPU"
    if ram_gb >= 25:
        print("\n🔵 CPU + High-RAM — embeddings will be slower but everything works")
    else:
        print("\n🟠 CPU + Low-RAM — consider 'High-RAM' toggle in Runtime settings")

# =============================================================================
# 2. DRIVE PATH DETECTION
# =============================================================================
from pathlib import Path

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
    drive_root = Path("/content/drive")
    if drive_root.exists():
        for p in drive_root.rglob("entity-resolution-pipeline"):
            if p.is_dir() and (p / "run_pipeline.py").exists():
                DRIVE_BASE = str(p.parent)
                break

if DRIVE_BASE is None:
    raise FileNotFoundError(
        "Could not find 'ricerca' folder in Google Drive.\n"
        "Run this first: from google.colab import drive; drive.mount('/content/drive')"
    )

print(f"\n📁 Drive base: {DRIVE_BASE}")

DRIVE_PIPELINE = f"{DRIVE_BASE}/entity-resolution-pipeline"
LOCAL_PIPELINE = "/content/local_pipeline/entity-resolution-pipeline"
LOCAL_CB = "/content/local_pipeline/cb_data"
LOCAL_ORBIS = "/content/orbis_local"

# =============================================================================
# 3. DATA SETUP (copy code + data to local SSD for speed)
# =============================================================================
print("\n📂 Setting up data on local SSD...")

# --- Code sync (always re-sync to pick up latest bug fixes) ---
if not os.path.exists(LOCAL_PIPELINE):
    print("  Copying pipeline code (first run)...")
    os.makedirs(LOCAL_PIPELINE, exist_ok=True)
    shutil.copytree(f"{DRIVE_PIPELINE}/src", f"{LOCAL_PIPELINE}/src")
    if os.path.exists(f"{DRIVE_PIPELINE}/configs"):
        shutil.copytree(f"{DRIVE_PIPELINE}/configs", f"{LOCAL_PIPELINE}/configs", dirs_exist_ok=True)
    shutil.copy(f"{DRIVE_PIPELINE}/run_pipeline.py", f"{LOCAL_PIPELINE}/run_pipeline.py")
else:
    print("  🔄 Syncing code from Drive...")
    src_local = f"{LOCAL_PIPELINE}/src"
    if os.path.exists(src_local):
        shutil.rmtree(src_local)
    shutil.copytree(f"{DRIVE_PIPELINE}/src", src_local)
    shutil.copy(f"{DRIVE_PIPELINE}/run_pipeline.py", f"{LOCAL_PIPELINE}/run_pipeline.py")
    print("    ✓ Code synced")

# --- Checkpoint sync ---
drive_cp = f"{DRIVE_PIPELINE}/data/interim/pipeline_checkpoint.json"
local_cp = f"{LOCAL_PIPELINE}/data/interim/pipeline_checkpoint.json"
if os.path.exists(drive_cp):
    os.makedirs(os.path.dirname(local_cp), exist_ok=True)
    shutil.copy2(drive_cp, local_cp)
    print("    ✓ Checkpoint synced from Drive")

# --- Crunchbase data ---
if not os.path.exists(LOCAL_CB):
    print("  Copying Crunchbase data...")
    shutil.copytree(f"{DRIVE_BASE}/dati europe cb", LOCAL_CB)

# --- database-done.xlsx ---
DB_DONE = "/content/local_pipeline/database-done.xlsx"
if not os.path.exists(DB_DONE):
    for path in [f"{DRIVE_PIPELINE}/database-done.xlsx", f"{DRIVE_BASE}/database-done.xlsx"]:
        if os.path.exists(path):
            shutil.copy2(path, DB_DONE)
            print(f"    ✓ database-done.xlsx copied")
            break

# --- Orbis parquet (reuse if already processed) ---
orbis_raw = f"{LOCAL_PIPELINE}/data/interim/orbis_clean/orbis_raw.parquet"
drive_orbis_raw = f"{DRIVE_PIPELINE}/data/interim/orbis_clean/orbis_raw.parquet"

if os.path.exists(drive_orbis_raw) and not os.path.exists(orbis_raw):
    print("  Copying existing orbis_raw.parquet from Drive...")
    os.makedirs(os.path.dirname(orbis_raw), exist_ok=True)
    shutil.copy2(drive_orbis_raw, orbis_raw)
elif not os.path.exists(orbis_raw):
    # Full Orbis Excel → Parquet ingestion
    os.makedirs(LOCAL_ORBIS, exist_ok=True)
    orbis_files = list(Path(LOCAL_ORBIS).glob("*.xlsx"))

    if not orbis_files:
        print("  Copying Orbis Excel files (~5 min)...")
        drive_files = []
        for folder in ["new orbis", "new orbis 2"]:
            folder_path = Path(f"{DRIVE_BASE}/{folder}")
            if folder_path.exists():
                drive_files.extend(list(folder_path.glob("*.xlsx")))
        
        print(f"  Found {len(drive_files)} Excel files in Drive (new orbis + new orbis 2)")
        for i, f in enumerate(drive_files):
            if i % 100 == 0:
                print(f"    {i}/{len(drive_files)}...", flush=True)
            shutil.copy2(f, LOCAL_ORBIS)
        orbis_files = list(Path(LOCAL_ORBIS).glob("*.xlsx"))
        print(f"  ✅ Copied {len(orbis_files)} files")

    print(f"  Processing {len(orbis_files)} Orbis files with Polars...")
    import polars as pl
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm.auto import tqdm

    ORBIS_COLS = {
        'BvD ID number': 'bvd_id', 'Company name Latin alphabet': 'orbis_name',
        'Country ISO code': 'orbis_country', 'City': 'orbis_city',
        'City (Latin Alphabet)': 'orbis_city',       # New export variant
        'City\nLatin Alphabet': 'orbis_city',         # Newline variant
        'Postcode': 'orbis_postcode', 'Website address': 'orbis_website',
        'E-mail address': 'orbis_email', 'Phone number': 'orbis_phone',
        'Date of incorporation': 'orbis_incorp_date',
        'Trade description (English)': 'orbis_trade_desc',
        'NACE Rev. 2 core code': 'orbis_nace',
        'Standardised legal form': 'orbis_legal_form',
        'Operating revenue (Turnover)': 'orbis_operating_revenue',
        'Total assets': 'orbis_total_assets',
        'Number of employees': 'orbis_num_employees',
        'SUB - BvD ID number': 'sub_bvd_id', 'SH - BvD ID number': 'sh_bvd_id',
        'GUO - BvD ID number': 'guo_bvd_id', 'BRANCH - BvD ID number': 'branch_bvd_id',
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
    with ThreadPoolExecutor(max_workers=8) as ex:  # 8 workers (CPU-friendly)
        futures = {ex.submit(process_file, f): f for f in orbis_files}
        for future in tqdm(as_completed(futures), total=len(orbis_files), desc="Processing"):
            try:
                df = future.result(timeout=60)
                if df is not None and not df.is_empty():
                    all_dfs.append(df)
            except:
                pass

    final_df = pl.concat(all_dfs).unique(subset=["bvd_id"], keep="first")
    for col in ALL_COLS:
        if col not in final_df.columns:
            final_df = final_df.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))
    final_df = final_df.select(ALL_COLS)
    os.makedirs(os.path.dirname(orbis_raw), exist_ok=True)
    final_df.write_parquet(orbis_raw, compression="snappy")
    print(f"  ✅ Orbis: {len(final_df):,} rows")
    del all_dfs, final_df

# --- Inline Orbis normalize (if not done) ---
orbis_clean = f"{LOCAL_PIPELINE}/data/interim/orbis_clean/orbis_clean.parquet"
drive_orbis_clean = f"{DRIVE_PIPELINE}/data/interim/orbis_clean/orbis_clean.parquet"

if os.path.exists(drive_orbis_clean) and not os.path.exists(orbis_clean):
    print("  Copying existing orbis_clean.parquet from Drive...")
    shutil.copy2(drive_orbis_clean, orbis_clean)
elif not os.path.exists(orbis_clean):
    print("  Normalizing Orbis data...")
    import pandas as pd
    import numpy as np
    sys.path.insert(0, f"{LOCAL_PIPELINE}/src")
    os.chdir(LOCAL_PIPELINE)
    from normalize import normalize_name_column
    from domains import extract_domain_column

    orbis = pd.read_parquet(orbis_raw)
    print(f"    Loaded {len(orbis):,} rows")

    name_features = normalize_name_column(orbis['orbis_name'])
    for col in name_features.columns:
        orbis[f'orbis_{col}'] = name_features[col]

    first_websites = orbis['orbis_website'].fillna('').str.split('|').str[0]
    domain_features = extract_domain_column(first_websites)
    orbis['orbis_domain'] = domain_features['etld1']

    date_col = orbis['orbis_incorp_date'].astype(str)
    is_numeric = date_col.str.match(r'^\d+$', na=False)
    numeric_dates = pd.to_datetime('1899-12-30') + pd.to_timedelta(
        pd.to_numeric(date_col.where(is_numeric), errors='coerce'), unit='D')
    parsed_dates = pd.to_datetime(date_col.where(~is_numeric), errors='coerce', dayfirst=True)
    orbis['orbis_incorp_year'] = numeric_dates.fillna(parsed_dates).dt.year

    orbis.to_parquet(orbis_clean, index=False)
    print(f"  ✅ orbis_clean.parquet ({os.path.getsize(orbis_clean)/(1024**3):.2f} GB)")
    del orbis, name_features, domain_features, first_websites, date_col
    import gc; gc.collect()

# --- Sync ANY existing interim data from Drive (embeddings, indexes, etc.) ---
print("  Syncing existing interim data from Drive...")
SYNC_DIRS = [
    "data/interim/cb_clean",
    "data/interim/embeddings",
    "data/interim/indexes",
    "data/interim/models/company_match",
    "data/interim/features",
    "data/interim/candidates",
    "data/interim",
]
for d in SYNC_DIRS:
    drive_dir = f"{DRIVE_PIPELINE}/{d}"
    local_dir = f"{LOCAL_PIPELINE}/{d}"
    if os.path.exists(drive_dir):
        os.makedirs(local_dir, exist_ok=True)
        for f in Path(drive_dir).glob("*"):
            if f.is_file():
                dest = f"{local_dir}/{f.name}"
                src_size = f.stat().st_size
                needs_copy = not os.path.exists(dest)
                if not needs_copy:
                    dst_size = os.path.getsize(dest)
                    if dst_size != src_size:
                        needs_copy = True
                if needs_copy:
                    size_mb = src_size / (1024**2)
                    if size_mb > 100:
                        print(f"    📦 {f.name} ({size_mb:.0f} MB)...", flush=True)
                    shutil.copy2(str(f), dest)

print("✅ Data ready\n")

# =============================================================================
# 4. AUTO-TUNED CONFIG (adapts to detected hardware)
# =============================================================================
os.makedirs(f"{LOCAL_PIPELINE}/configs", exist_ok=True)
config_path = f"{LOCAL_PIPELINE}/configs/universal.yaml"

# Hardware-aware parameter tuning
HW_PROFILES = {
    'A100': {'emb_batch': 2048, 'emb_chunk': 200000, 'chunk': 200000, 'workers': 8,
             'device': 'cuda', 'dtype': 'float16', 'faiss_gpu': True, 'rerank_batch': 128},
    'L4':   {'emb_batch': 1024, 'emb_chunk': 150000, 'chunk': 150000, 'workers': 4,
             'device': 'cuda', 'dtype': 'float16', 'faiss_gpu': False, 'rerank_batch': 64},
    'T4':   {'emb_batch': 512,  'emb_chunk': 100000, 'chunk': 100000, 'workers': 4,
             'device': 'cuda', 'dtype': 'float16', 'faiss_gpu': False, 'rerank_batch': 32},
    'CPU':  {'emb_batch': 128,  'emb_chunk': 75000,  'chunk': 75000,  'workers': 2,
             'device': 'cpu',  'dtype': 'float32',  'faiss_gpu': False, 'rerank_batch': 16},
}

prof = HW_PROFILES[HW_TIER]

# Further adjust for available RAM
if ram_gb < 20 and HW_TIER == 'CPU':
    prof['chunk'] = 50000
    prof['emb_chunk'] = 50000
    prof['emb_batch'] = 64
    prof['workers'] = 1
elif ram_gb >= 40 and HW_TIER == 'CPU':
    prof['chunk'] = 100000
    prof['emb_chunk'] = 100000
    prof['emb_batch'] = 256
    prof['workers'] = 4

# Estimate time
time_estimates = {
    'A100': '~40 min total', 'L4': '~50 min total',
    'T4': '~1.5h total', 'CPU': '~2-4h total'
}

print(f"\n⚙️  Config: {HW_TIER} | device={prof['device']} | emb_batch={prof['emb_batch']} | chunk={prof['chunk']}")
print(f"⏱️  Estimated: {time_estimates[HW_TIER]}")

CONFIG = f"""
# UNIVERSAL CONFIG (auto: {HW_TIER}, {ram_gb:.0f} GB RAM, {GPU_NAME})

paths:
  project_root: {LOCAL_PIPELINE}
  raw_crunchbase: {LOCAL_CB}
  raw_orbis: {LOCAL_ORBIS}

embeddings:
  batch_size: {prof['emb_batch']}
  device: {prof['device']}
  dtype: {prof['dtype']}
  enabled: true
  model_name: all-MiniLM-L6-v2
  streaming_chunk_size: {prof['emb_chunk']}
  use_streaming: true

blocking:
  ANN_TOPK_DESC: 100
  ANN_TOPK_NAME: 200
  MAX_CANDIDATES_PER_CB: 500
  rare_token_df_threshold: 1000

faiss:
  metric: inner_product
  nprobe: 32
  use_gpu: {str(prof['faiss_gpu']).lower()}

features:
  enable_family_expansion: true
  enable_investor_checks: true
  enable_semantic_embeddings: true
  parallel_workers: {prof['workers']}

logging:
  level: INFO
  memory_warnings_threshold_gb: {int(ram_gb * 0.8)}
  save_timing: true

model:
  calibration: isotonic
  learning_rate: 0.1
  max_depth: 8
  n_estimators: 500
  type: gradient_boosting

processing:
  chunk_size: {prof['chunk']}
  feature_chunk_size: {prof['chunk']}

reranking:
  enabled: true
  max_score: 0.95
  min_score: 0.4
  model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
  batch_size: {prof['rerank_batch']}

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

with open(config_path, 'w') as f:
    f.write(CONFIG)

# =============================================================================
# 5. RUN ALL 13 STEPS
# =============================================================================
STEPS = ['ingest', 'normalize', 'alias', 'index', 'embeddings',
         'blocking', 'features', 'train', 'score', 'rerank',
         'decide', 'report', 'analytics']

os.chdir(LOCAL_PIPELINE)
sys.path.insert(0, f"{LOCAL_PIPELINE}/src")

# Check which steps already completed
completed_steps = set()
if os.path.exists(local_cp):
    try:
        with open(local_cp) as f:
            cp_data = json.load(f)
            completed_steps = set(cp_data.get('completed_steps', []))
    except:
        pass

steps_to_run = [s for s in STEPS if s not in completed_steps]

if not steps_to_run:
    print("✅ All steps already completed!")
else:
    skipped = len(STEPS) - len(steps_to_run)
    if skipped:
        print(f"⏭️  Skipping {skipped} completed: {', '.join(sorted(completed_steps))}")
    print(f"▶️  Running: {', '.join(steps_to_run)}")

    total_start = time.time()

    for step in steps_to_run:
        print(f"\n{'='*60}")

        # Show estimated time for embeddings
        if step == 'embeddings':
            emb_times = {'A100': '~5 min', 'L4': '~15 min', 'T4': '~25 min', 'CPU': '~90-180 min ☕'}
            print(f"▶ EMBEDDINGS ({HW_TIER}: {emb_times[HW_TIER]})")
        else:
            print(f"▶ {step.upper()}")
        print('='*60, flush=True)

        start = time.time()
        run_script = f"{LOCAL_PIPELINE}/run_pipeline.py"

        env = os.environ.copy()
        env['TQDM_MININTERVAL'] = '30'
        env['TQDM_DISABLE'] = '0'

        process = subprocess.Popen(
            ['python', '-u', run_script, '--config', config_path, '--step', step],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            cwd=LOCAL_PIPELINE, env=env
        )
        for line in process.stdout:
            if '\r' in line and line.strip().endswith('%|'):
                continue
            print(line, end='', flush=True)
        process.wait()

        elapsed = time.time() - start

        if process.returncode != 0:
            print(f"❌ {step} FAILED after {elapsed:.1f}s")
            print(f"\n💡 TIP: Re-run this cell to resume from '{step}'")
            break

        print(f"✅ {step} complete ({elapsed:.1f}s)")

        # Sync critical outputs to Drive after each step
        SYNC_OUTPUTS = [
            "data/interim/orbis_clean/orbis_clean.parquet",
            "data/interim/orbis_clean/orbis_raw.parquet",
            "data/interim/cb_clean/cb_clean.parquet",
            "data/interim/cb_clean/cb_raw_companies.parquet",
            "data/interim/candidates/candidates.parquet",
            "data/interim/features/pair_features.parquet",
            "data/interim/features/scored_candidates.parquet",
            "data/interim/features/reranked_candidates.parquet",
            "data/interim/pipeline_checkpoint.json",
            "data/outputs/matches/matches_final.parquet",
        ]
        for rel in SYNC_OUTPUTS:
            src = f"{LOCAL_PIPELINE}/{rel}"
            if os.path.exists(src):
                dst = src.replace(LOCAL_PIPELINE, DRIVE_PIPELINE)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    shutil.copy2(src, dst)
                except:
                    pass

        # Sync directories (embeddings, indexes, models)
        for sync_dir in ["data/interim/embeddings", "data/interim/indexes",
                         "data/interim/models/company_match"]:
            local_d = f"{LOCAL_PIPELINE}/{sync_dir}"
            if os.path.exists(local_d):
                drive_d = f"{DRIVE_PIPELINE}/{sync_dir}"
                os.makedirs(drive_d, exist_ok=True)
                for f_path in Path(local_d).glob("*"):
                    if f_path.is_file():
                        try:
                            shutil.copy2(str(f_path), f"{drive_d}/{f_path.name}")
                        except:
                            pass

    total_elapsed = time.time() - total_start
    hours = int(total_elapsed // 3600)
    mins = int((total_elapsed % 3600) // 60)
    print(f"\n⏱️  Total time: {hours}h {mins}m")

# =============================================================================
# 6. RESULTS
# =============================================================================
print("\n" + "=" * 60)
print("🎉 PIPELINE COMPLETE!")
print("=" * 60)

matches = f"{LOCAL_PIPELINE}/data/outputs/matches/matches_final.parquet"
if os.path.exists(matches):
    import pandas as pd
    df = pd.read_parquet(matches)
    print(f"\nTotal matches: {len(df):,}")
    if 'tier' in df.columns:
        print(df['tier'].value_counts())
    if 'blocking_source' in df.columns:
        print(f"\nBlocking sources:")
        print(df['blocking_source'].value_counts())
