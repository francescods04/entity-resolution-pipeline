#!/usr/bin/env python3
"""
COLAB_OPTIMIZED.py - Bulletproof Entity Resolution Pipeline for Google Colab

ULTRATHINK OPTIMIZED:
- SSD-first architecture: All processing on /content, only final sync to Drive
- Automatic checkpoint/resume support
- Live progress output with tqdm
- Handles all edge cases (Drive sync failures, session timeouts)

USAGE (in Colab cells):
-----------------------
Cell 1: !pip install -q polars[calamine] pyarrow tqdm sentence-transformers faiss-cpu rapidfuzz
Cell 2: from google.colab import drive; drive.mount('/content/drive')
Cell 3: %run "/content/drive/Othercomputers/My MacBook Pro/Downloads/ricerca/entity-resolution-pipeline/COLAB_OPTIMIZED.py"

EXPECTED RUNTIME:
- T4 GPU: ~80 minutes
- A100 GPU: ~40 minutes
"""

import os
import sys
import shutil
import subprocess
import time
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Source paths on Google Drive
DRIVE_BASE = "/content/drive/Othercomputers/My MacBook Pro/Downloads/ricerca"
DRIVE_PIPELINE = f"{DRIVE_BASE}/entity-resolution-pipeline"
DRIVE_CB = f"{DRIVE_BASE}/dati europe cb"
DRIVE_ORBIS = f"{DRIVE_BASE}/new orbis"
DRIVE_DB_DONE = f"{DRIVE_BASE}/database-done.xlsx"

# Local SSD paths (fast I/O)
LOCAL_BASE = "/content/local_pipeline"
LOCAL_PIPELINE = f"{LOCAL_BASE}/entity-resolution-pipeline"
LOCAL_CB = f"{LOCAL_BASE}/cb_data"
LOCAL_ORBIS = "/content/orbis_local"  # From previous Polars ingestion
LOCAL_DB_DONE = f"{LOCAL_BASE}/database-done.xlsx"

# Output paths
LOCAL_OUTPUT = f"{LOCAL_PIPELINE}/data"
DRIVE_OUTPUT = f"{DRIVE_PIPELINE}/data"

# Pipeline steps in order
PIPELINE_STEPS = [
    'normalize', 'alias', 'index', 'embeddings', 
    'blocking', 'features', 'train', 'score', 
    'rerank', 'decide', 'report', 'analytics'
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(text, char="="):
    """Print formatted header."""
    print(f"\n{char*60}")
    print(f"  {text}")
    print(f"{char*60}\n", flush=True)


def copy_with_progress(src, dst, desc="Copying"):
    """Copy directory with progress indication."""
    if os.path.isfile(src):
        print(f"  {desc}: {os.path.basename(src)}", flush=True)
        shutil.copy2(src, dst)
    elif os.path.isdir(src):
        print(f"  {desc}: {os.path.basename(src)}/", flush=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)


def run_step(step_name, config_path):
    """Run a single pipeline step with live output."""
    print(f"\n▶ STEP: {step_name.upper()}", flush=True)
    print("-" * 40, flush=True)
    
    start = time.time()
    
    process = subprocess.Popen(
        ['python', '-u', 'run_pipeline.py', '--config', config_path, '--step', step_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=LOCAL_PIPELINE
    )
    
    for line in process.stdout:
        print(line, end='', flush=True)
    
    process.wait()
    elapsed = time.time() - start
    
    if process.returncode != 0:
        print(f"\n❌ STEP {step_name} FAILED (exit code {process.returncode})")
        return False
    
    print(f"\n✅ {step_name} complete ({elapsed:.1f}s)", flush=True)
    return True


def incremental_sync(phase_name):
    """
    RESILIENCE: Sync critical files to Drive after each phase.
    Prevents data loss if Colab session disconnects.
    """
    print(f"  📤 Syncing {phase_name} to Drive...", flush=True)
    
    sync_files = [
        # Critical parquet files
        ("data/interim/orbis_clean/orbis_raw.parquet", "data/interim/orbis_clean/"),
        ("data/interim/orbis_clean/orbis_clean.parquet", "data/interim/orbis_clean/"),
        ("data/interim/cb_clean/cb_raw_companies.parquet", "data/interim/cb_clean/"),
        ("data/interim/cb_clean/cb_clean.parquet", "data/interim/cb_clean/"),
        ("data/interim/alias_registry.parquet", "data/interim/"),
        # Candidates and features (expensive to recompute)
        ("data/interim/candidates/candidates.parquet", "data/interim/candidates/"),
        ("data/interim/features/pair_features.parquet", "data/interim/features/"),
        ("data/interim/features/scored_candidates.parquet", "data/interim/features/"),
        # Final outputs
        ("data/outputs/matches/matches_final.parquet", "data/outputs/matches/"),
        ("data/outputs/reports/run_manifest.json", "data/outputs/reports/"),
    ]
    
    synced = 0
    for src_rel, dst_dir in sync_files:
        src = os.path.join(LOCAL_PIPELINE, src_rel)
        if os.path.exists(src):
            dst = os.path.join(DRIVE_PIPELINE, dst_dir)
            os.makedirs(dst, exist_ok=True)
            dst_file = os.path.join(dst, os.path.basename(src))
            try:
                shutil.copy2(src, dst_file)
                synced += 1
            except Exception as e:
                print(f"    ⚠ Could not sync {os.path.basename(src)}: {e}", flush=True)
    
    if synced > 0:
        print(f"  ✓ Synced {synced} files to Drive", flush=True)


def verify_file(path, min_size_mb=1):
    """Verify file exists and meets minimum size."""
    if not os.path.exists(path):
        return False, f"File not found: {path}"
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb < min_size_mb:
        return False, f"File too small: {path} ({size_mb:.1f} MB < {min_size_mb} MB)"
    return True, f"{path} ({size_mb:.1f} MB)"


# =============================================================================
# PHASE 1: SETUP LOCAL ENVIRONMENT
# =============================================================================

def phase1_setup():
    """Copy pipeline code to local SSD."""
    print_header("PHASE 1: SETUP LOCAL ENVIRONMENT")
    
    # Create local directories
    os.makedirs(LOCAL_BASE, exist_ok=True)
    os.makedirs(LOCAL_ORBIS, exist_ok=True)
    
    # Copy pipeline code (small, fast)
    if not os.path.exists(LOCAL_PIPELINE):
        print("Copying pipeline code to local SSD...", flush=True)
        shutil.copytree(DRIVE_PIPELINE, LOCAL_PIPELINE)
        print(f"  ✓ Pipeline code copied", flush=True)
    else:
        print("  ✓ Pipeline code already on local SSD", flush=True)
    
    # Copy Crunchbase data (small, ~100MB)
    if not os.path.exists(LOCAL_CB):
        print("Copying Crunchbase data...", flush=True)
        shutil.copytree(DRIVE_CB, LOCAL_CB)
        print(f"  ✓ Crunchbase data copied", flush=True)
    else:
        print("  ✓ Crunchbase data already local", flush=True)
    
    # Copy database-done.xlsx
    if not os.path.exists(LOCAL_DB_DONE):
        print("Copying database-done.xlsx...", flush=True)
        shutil.copy2(DRIVE_DB_DONE, LOCAL_DB_DONE)
        print(f"  ✓ database-done.xlsx copied", flush=True)
    else:
        print("  ✓ database-done.xlsx already local", flush=True)
    
    print("\n✅ Phase 1 complete: Local environment ready", flush=True)
    return True


# =============================================================================
# PHASE 2: DATA INGESTION
# =============================================================================

def phase2_ingest():
    """Run Polars ultra-fast ingestion if needed."""
    print_header("PHASE 2: DATA INGESTION")
    
    orbis_raw_path = f"{LOCAL_PIPELINE}/data/interim/orbis_clean/orbis_raw.parquet"
    
    # Check if Orbis already ingested
    if os.path.exists(orbis_raw_path):
        size_mb = os.path.getsize(orbis_raw_path) / (1024 * 1024)
        print(f"  ✓ Orbis already ingested: {size_mb:.1f} MB", flush=True)
    else:
        # Check if Orbis files on local SSD
        orbis_files = list(Path(LOCAL_ORBIS).glob("*.xlsx"))
        
        if not orbis_files:
            print("  ⚠ No Orbis files on local SSD. Copying from Drive...", flush=True)
            print("    (This may take 5-10 minutes for 944 Excel files)", flush=True)
            
            # Copy Orbis Excel files to local SSD
            drive_orbis_files = list(Path(DRIVE_ORBIS).glob("*.xlsx"))
            for i, f in enumerate(drive_orbis_files):
                if i % 100 == 0:
                    print(f"    Copied {i}/{len(drive_orbis_files)} files...", flush=True)
                shutil.copy2(f, LOCAL_ORBIS)
            print(f"    ✓ Copied {len(drive_orbis_files)} files", flush=True)
        
        # Run Polars ingestion
        print("  Running Polars ultra-fast ingestion...", flush=True)
        os.chdir(LOCAL_PIPELINE)
        subprocess.run(['python', '-u', 'POLARS_FAST_INGEST.py'], check=True)
    
    # Load Crunchbase
    cb_raw_path = f"{LOCAL_PIPELINE}/data/interim/cb_clean/cb_raw_companies.parquet"
    if not os.path.exists(cb_raw_path):
        print("  Loading Crunchbase...", flush=True)
        sys.path.insert(0, f"{LOCAL_PIPELINE}/src")
        from data_io import load_crunchbase_companies, save_to_parquet
        
        os.makedirs(os.path.dirname(cb_raw_path), exist_ok=True)
        save_to_parquet(load_crunchbase_companies(LOCAL_CB), cb_raw_path)
        print(f"  ✓ Crunchbase loaded", flush=True)
    else:
        print(f"  ✓ Crunchbase already loaded", flush=True)
    
    print("\n✅ Phase 2 complete: Data ingested", flush=True)
    return True


# =============================================================================
# PHASE 3: NORMALIZE (with local save verification)
# =============================================================================

def phase3_normalize():
    """Run normalize step with explicit file verification."""
    print_header("PHASE 3: NORMALIZATION")
    
    orbis_clean_path = f"{LOCAL_PIPELINE}/data/interim/orbis_clean/orbis_clean.parquet"
    
    # Check if already normalized
    if os.path.exists(orbis_clean_path):
        size_mb = os.path.getsize(orbis_clean_path) / (1024 * 1024)
        print(f"  ✓ Already normalized: {size_mb:.1f} MB", flush=True)
        return True
    
    # Run normalize inline (not subprocess) to catch errors
    print("  Running inline normalize (catches save errors)...", flush=True)
    
    os.chdir(LOCAL_PIPELINE)
    sys.path.insert(0, f"{LOCAL_PIPELINE}/src")
    
    import pandas as pd
    from normalize import normalize_name_column
    from domains import extract_domain_column
    
    # Load raw Orbis
    orbis_raw_path = f"{LOCAL_PIPELINE}/data/interim/orbis_clean/orbis_raw.parquet"
    print("  Loading orbis_raw.parquet...", flush=True)
    orbis = pd.read_parquet(orbis_raw_path)
    print(f"  Loaded {len(orbis):,} rows", flush=True)
    
    # Normalize names
    print("  Normalizing names...", flush=True)
    name_features = normalize_name_column(orbis['orbis_name'])
    for col in name_features.columns:
        orbis[f'orbis_{col}'] = name_features[col]
    
    # Extract domains
    print("  Extracting domains...", flush=True)
    first_websites = orbis['orbis_website'].fillna('').str.split('|').str[0]
    domain_features = extract_domain_column(first_websites)
    orbis['orbis_domain'] = domain_features['etld1']
    
    # Year - vectorized
    print("  Extracting years (vectorized)...", flush=True)
    date_col = orbis['orbis_incorp_date'].astype(str)
    is_numeric = date_col.str.match(r'^\d+$', na=False)
    import numpy as np
    numeric_dates = pd.to_datetime('1899-12-30') + pd.to_timedelta(
        pd.to_numeric(date_col.where(is_numeric), errors='coerce'), unit='D')
    parsed_dates = pd.to_datetime(date_col.where(~is_numeric), errors='coerce', dayfirst=True)
    orbis['orbis_incorp_year'] = numeric_dates.fillna(parsed_dates).dt.year
    
    # Save to LOCAL SSD
    print(f"  Saving to {orbis_clean_path}...", flush=True)
    orbis.to_parquet(orbis_clean_path, index=False)
    
    # Verify save
    ok, msg = verify_file(orbis_clean_path, min_size_mb=1000)  # Expect ~2GB
    if ok:
        print(f"  ✓ {msg}", flush=True)
    else:
        print(f"  ❌ {msg}", flush=True)
        return False
    
    print("\n✅ Phase 3 complete: Data normalized", flush=True)
    return True


# =============================================================================
# PHASE 4: RUN REMAINING PIPELINE STEPS
# =============================================================================

def phase4_pipeline():
    """Run remaining pipeline steps."""
    print_header("PHASE 4: PIPELINE EXECUTION")
    
    os.chdir(LOCAL_PIPELINE)
    
    # Create local config with SSD paths
    config_path = f"{LOCAL_PIPELINE}/configs/colab_local.yaml"
    
    with open(f"{LOCAL_PIPELINE}/configs/colab_gpu.yaml", 'r') as f:
        config = f.read()
    
    # Override paths to local SSD
    config = config.replace(
        "/content/drive/Othercomputers/My MacBook Pro/Downloads/ricerca/entity-resolution-pipeline",
        LOCAL_PIPELINE
    ).replace(
        "/content/drive/Othercomputers/My MacBook Pro/Downloads/ricerca/dati\n    europe cb",
        LOCAL_CB
    ).replace(
        "/content/drive/Othercomputers/My MacBook Pro/Downloads/ricerca/new orbis",
        LOCAL_ORBIS
    )
    
    with open(config_path, 'w') as f:
        f.write(config)
    
    print(f"  Created local config: {config_path}", flush=True)
    
    # Run remaining steps (normalize already done)
    remaining_steps = ['alias', 'index', 'embeddings', 'blocking', 'features', 
                       'train', 'score', 'rerank', 'decide', 'report', 'analytics']
    
    for step in remaining_steps:
        if not run_step(step, config_path):
            print(f"\n❌ Pipeline failed at step: {step}")
            return False
    
    print("\n✅ Phase 4 complete: Pipeline finished", flush=True)
    return True


# =============================================================================
# PHASE 5: SYNC RESULTS TO DRIVE
# =============================================================================

def phase5_sync():
    """Sync results back to Google Drive."""
    print_header("PHASE 5: SYNC TO DRIVE")
    
    # Paths to sync
    sync_paths = [
        (f"{LOCAL_PIPELINE}/data/outputs", f"{DRIVE_PIPELINE}/data/outputs"),
        (f"{LOCAL_PIPELINE}/data/interim/alias_registry.parquet", 
         f"{DRIVE_PIPELINE}/data/interim/alias_registry.parquet"),
    ]
    
    for src, dst in sync_paths:
        if os.path.exists(src):
            print(f"  Syncing: {os.path.basename(src)}", flush=True)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
    
    print("\n✅ Phase 5 complete: Results synced to Drive", flush=True)
    return True


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run full optimized pipeline."""
    print_header("🚀 COLAB OPTIMIZED ENTITY RESOLUTION PIPELINE", "═")
    print(f"Start time: {time.strftime('%H:%M:%S')}")
    print(f"Expected runtime: ~80 min (T4) / ~40 min (A100)")
    
    total_start = time.time()
    
    phases = [
        ("Setup", phase1_setup),
        ("Ingest", phase2_ingest),
        ("Normalize", phase3_normalize),
        ("Pipeline", phase4_pipeline),
        ("Sync", phase5_sync),
    ]
    
    for name, func in phases:
        try:
            if not func():
                print(f"\n❌ FAILED at phase: {name}")
                return False
            # RESILIENCE: Sync to Drive after each phase
            incremental_sync(name)
        except Exception as e:
            print(f"\n❌ ERROR in phase {name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    total_elapsed = time.time() - total_start
    
    print_header("🎉 PIPELINE COMPLETE!", "═")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Results saved to: {DRIVE_PIPELINE}/data/outputs/")
    
    # Quick stats
    matches_path = f"{LOCAL_PIPELINE}/data/outputs/matches/matches_final.parquet"
    if os.path.exists(matches_path):
        import pandas as pd
        matches = pd.read_parquet(matches_path)
        print(f"\nMatch Statistics:")
        print(f"  Total matches: {len(matches):,}")
        if 'tier' in matches.columns:
            print(f"  Tier distribution:")
            for tier, count in matches['tier'].value_counts().items():
                print(f"    {tier}: {count:,}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
