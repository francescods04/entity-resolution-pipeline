#!/usr/bin/env python3
"""
process_orbis_batch.py - FAST Parallel Orbis File Processor

Optimized for M1 Mac: Uses all CPU cores to process Excel files in parallel.

KEY FEATURES:
- Multiprocessing (ProcessPoolExecutor)
- Chunked writes to avoid memory bloat
- Automatic aggregation of chunks

USAGE:
python process_orbis_batch.py
"""

import argparse
import json
import logging
import os
import sys
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import from src (must be importable by workers)
from data_io import load_orbis_single_excel
from normalize import normalize_name

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input/Output paths matches USER PREFERENCE
ORBIS_DIR = Path('/Users/francescodelsesto/Downloads/ricerca/new orbis')
OUTPUT_DIR = Path('/Users/francescodelsesto/Downloads/ricerca/entity-resolution-pipeline/data/interim/orbis_clean')
CHUNKS_DIR = OUTPUT_DIR / 'temp_chunks'

# Final output
FINAL_PARQUET = OUTPUT_DIR / 'orbis_raw.parquet'

TRACKING_FILE = OUTPUT_DIR / 'processed_files.json'
COLLISION_REPORT = OUTPUT_DIR / 'name_collisions.json'


# =============================================================================
# WORKER FUNCTION (Must be top-level for pickling)
# =============================================================================

def process_single_file_task(file_path_str: str) -> Dict:
    """
    Worker task: Load Excel -> Save Chunk Parquet.
    Returns tracking info dict.
    """
    file_path = Path(file_path_str)
    try:
        # Load and aggregate
        # Suppress logging inside worker to avoid clutter
        df = load_orbis_single_excel(str(file_path), aggregate=True)
        
        if df is None or len(df) == 0:
            return {'status': 'empty', 'file': file_path.name}
        
        # Ensure BvD ID exists
        if 'bvd_id' not in df.columns:
            return {'status': 'error', 'file': file_path.name, 'error': 'No bvd_id column'}

        # Normalize types for Parquet
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).replace('nan', None).replace('None', None)
        
        # Generate clean chunk filename
        # Use file stem + hash or just stem if unique. Files are numbered like 1.xlsx, 1-1.xlsx
        safe_name = file_path.stem.replace(' ', '_')
        chunk_path = CHUNKS_DIR / f"chunk_{safe_name}.parquet"
        
        # Write chunk
        df.to_parquet(chunk_path, index=False)
        
        return {
            'status': 'success',
            'file': file_path.name,
            'count': len(df),
            'chunk_path': str(chunk_path)
        }
        
    except Exception as e:
        return {'status': 'error', 'file': file_path.name, 'error': str(e)}


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

def process_parallel(files: List[Path], max_workers: int = None) -> None:
    """Run parallel processing."""
    
    if not files:
        logger.info("No files to process.")
        return

    # Prepare chunks dir
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    logger.info(f"Starting parallel processing of {len(files)} files...")
    logger.info(f"Using {max_workers or os.cpu_count()} cores.")

    # Convert paths to strings for pickling
    file_paths = [str(f) for f in files]
    
    results = []
    
    # Run in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_file_task, f): f for f in file_paths}
        
        # Monitor progress
        with tqdm(total=len(files), desc="Processing (Multi-Core)") as pbar:
            for future in as_completed(future_to_file):
                res = future.result()
                results.append(res)
                pbar.update(1)
                
                # Update description with speed
                if res['status'] == 'error':
                    pbar.set_postfix_str(f"Last Error: {res.get('error')[:20]}...")
    
    # Analyze results
    successes = [r for r in results if r['status'] == 'success']
    errors = [r for r in results if r['status'] == 'error']
    empties = [r for r in results if r['status'] == 'empty']
    
    logger.info(f"Batch processed in {time.time()-start_time:.1f}s")
    logger.info(f"Success: {len(successes)}, Empty: {len(empties)}, Errors: {len(errors)}")
    
    if errors:
        logger.warning(f"First 5 errors: {[e['error'] for e in errors[:5]]}")

    # ==========================================
    # AGGREGATION PHASE
    # ==========================================
    logger.info("Aggregating chunks into final Parquet...")
    
    if not successes:
        logger.warning("No successful chunks to aggregate.")
        return

    chunk_files = [Path(r['chunk_path']) for r in successes]
    
    try:
        # Read all chunks efficiently
        # PyArrow Dataset is faster than pandas concat for reading
        final_table = pq.read_table([str(p) for p in chunk_files])
        
        # Convert to pandas for deduplication (Parquet doesn't enforce uniqueness)
        df_final = final_table.to_pandas()
        
        logger.info(f"Total rows before deduplication: {len(df_final):,}")
        
        # Dedup by bvd_id
        df_final = df_final.drop_duplicates(subset=['bvd_id'], keep='first')
        
        logger.info(f"Total unique companies: {len(df_final):,}")
        
        # Save Final
        FINAL_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(FINAL_PARQUET, index=False)
        logger.info(f"Saved merged database to: {FINAL_PARQUET}")
        
        # Cleanup chunks (Optional: user might want to keep? Nah, clean up to save space)
        # shutil.rmtree(CHUNKS_DIR) 
        # logger.info("Cleaned up temp chunks.")
        
    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
        raise

    # Update Tracking File
    processed_names = [Path(f).name for f in files]
    update_tracking(processed_names, len(df_final))


def load_tracking() -> Dict:
    if TRACKING_FILE.exists():
        with open(TRACKING_FILE, 'r') as f:
            return json.load(f)
    return {'processed_files': [], 'total_companies': 0}

def update_tracking(new_files: List[str], total_count: int):
    tracking = load_tracking()
    # Add new files, dedup
    existing = set(tracking.get('processed_files', []))
    existing.update(new_files)
    tracking['processed_files'] = list(existing)
    tracking['total_companies'] = total_count
    tracking['last_updated'] = datetime.now().isoformat()
    
    with open(TRACKING_FILE, 'w') as f:
        json.dump(tracking, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Fast Parallel Orbis Processor')
    parser.add_argument('--workers', type=int, help='Number of CPU cores to use')
    parser.add_argument('--reprocess', action='store_true', help='Ignore tracking and reprocess all')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("FAST ORBIS PROCESSOR (M1 OPTIMIZED)")
    logger.info("="*60)
    
    # Identify files
    all_files = sorted(ORBIS_DIR.glob('*.xlsx'))
    
    if args.reprocess:
        files_to_process = all_files
    else:
        tracking = load_tracking()
        processed_set = set(tracking.get('processed_files', []))
        files_to_process = [f for f in all_files if f.name not in processed_set]
    
    # Filter out temp/hidden files
    files_to_process = [f for f in files_to_process if not f.name.startswith('~$') and not f.name.endswith('.part')]
    
    if not files_to_process:
        logger.info("No new files to process.")
        return

    process_parallel(files_to_process, max_workers=args.workers)


if __name__ == '__main__':
    # Fix for macOS multiprocessing
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    main()
