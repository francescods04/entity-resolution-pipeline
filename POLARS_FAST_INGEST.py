# =============================================================================

# 🚀 ULTRA-FAST POLARS INGESTION - 5-10x FASTER
# =============================================================================
# Uses Polars instead of Pandas for Excel parsing
# Polars has native Rust-based calamine integration = MUCH faster
# =============================================================================

import os
import sys
import glob
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

# Install polars if needed (run this cell first in Colab):
# !pip install -q polars[calamine] pyarrow tqdm

try:
    import polars as pl
except ImportError:
    raise ImportError("Run: !pip install -q polars[calamine]")

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_PATH = "/content/drive/Othercomputers/My MacBook Pro/Downloads/ricerca"
PIPELINE_DIR = f"{BASE_PATH}/entity-resolution-pipeline"
ORBIS_DIR_LOCAL = "/content/orbis_local"
OUTPUT_DIR = f"{PIPELINE_DIR}/data/interim"
ORBIS_OUTPUT = f"{OUTPUT_DIR}/orbis_clean/orbis_raw.parquet"

NUM_WORKERS = 8  # ProcessPoolExecutor workers
BATCH_SIZE = 200  # Larger batches with Polars

os.makedirs(f"{OUTPUT_DIR}/orbis_clean", exist_ok=True)

# =============================================================================
# SCHEMA
# =============================================================================
ORBIS_COLS = {
    'BvD ID number': 'bvd_id',
    'Company name Latin alphabet': 'orbis_name',
    'Country ISO code': 'orbis_country',
    'City': 'orbis_city',
    'Postcode': 'orbis_postcode',
    'Website address': 'orbis_website',
    'E-mail address': 'orbis_email',
    'Phone number': 'orbis_phone',
    'Date of incorporation': 'orbis_incorp_date',
    'Trade description (English)': 'orbis_trade_desc',
    'NACE Rev. 2 core code': 'orbis_nace',
    'NACE Rev. 2 core code description': 'orbis_nace_desc',
    'Standardised legal form': 'orbis_legal_form',
    'Operating revenue (Turnover)': 'orbis_operating_revenue',
    'Total assets': 'orbis_total_assets',
    'Number of employees': 'orbis_num_employees',
    'SUB - BvD ID number': 'sub_bvd_id',
    'SH - BvD ID number': 'sh_bvd_id',
    'GUO - BvD ID number': 'guo_bvd_id',
    'BRANCH - BvD ID number': 'branch_bvd_id',
    'SH - Name': 'sh_name',
    'SH - Country ISO code': 'sh_country',
    'SH - Type': 'sh_type',
    'SH - Direct %': 'sh_direct_pct',
    'SH - Total %': 'sh_total_pct',
}

FIRST_COLS = ['bvd_id', 'orbis_name', 'orbis_country', 'orbis_city', 'orbis_postcode',
              'orbis_phone', 'orbis_incorp_date', 'orbis_trade_desc', 'orbis_nace',
              'orbis_nace_desc', 'orbis_legal_form', 'orbis_operating_revenue',
              'orbis_total_assets', 'orbis_num_employees']

AGG_COLS = ['orbis_website', 'orbis_email', 'sub_bvd_id', 'sh_bvd_id', 'guo_bvd_id',
            'branch_bvd_id', 'sh_name', 'sh_country', 'sh_type', 'sh_direct_pct', 'sh_total_pct']

ALL_COLS = FIRST_COLS + AGG_COLS

# =============================================================================
# FAST POLARS PROCESSOR
# =============================================================================
def process_file_polars(filepath: str) -> pl.DataFrame:
    """Process single Excel file with Polars - 5-10x faster than pandas."""
    try:
        # Polars reads Excel directly with Rust-based calamine
        df = pl.read_excel(filepath, sheet_name="Results", engine="calamine")
    except:
        try:
            df = pl.read_excel(filepath, sheet_name=0, engine="calamine")
        except:
            return pl.DataFrame()
    
    if df.is_empty():
        return pl.DataFrame()
    
    # Rename columns
    rename_map = {}
    for orig, new in ORBIS_COLS.items():
        # Find column (case-insensitive, handle whitespace)
        for col in df.columns:
            if col.lower().replace('\n', ' ').strip() == orig.lower():
                rename_map[col] = new
                break
    
    df = df.rename(rename_map)
    
    # Add missing columns
    for col in ALL_COLS:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))
    
    # Get first column for grouping
    idx_col = df.columns[0]
    df = df.with_columns(pl.col(idx_col).forward_fill().alias("_idx"))
    
    # Aggregate: first value for core cols, concat for multi-value cols
    first_exprs = [pl.col(c).first().alias(c) for c in FIRST_COLS if c in df.columns]
    agg_exprs = [pl.col(c).drop_nulls().unique().str.concat("|").alias(c) for c in AGG_COLS if c in df.columns]
    
    result = df.group_by("_idx", maintain_order=True).agg(first_exprs + agg_exprs)
    result = result.filter(pl.col("bvd_id").is_not_null())
    result = result.drop("_idx")
    
    # Cast ALL columns to String to prevent schema mismatch across files
    result = result.select([pl.col(c).cast(pl.Utf8).alias(c) for c in result.columns])
    
    return result

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def run_polars_ingestion():
    orbis_files = sorted(glob.glob(f"{ORBIS_DIR_LOCAL}/*.xlsx"))
    
    if not orbis_files:
        print("❌ No files found. Run the copy step first.")
        return
    
    print(f"🚀 POLARS ULTRA-FAST INGESTION")
    print(f"   Files: {len(orbis_files)}")
    print(f"   Workers: {NUM_WORKERS}")
    print("=" * 60)
    
    if os.path.exists(ORBIS_OUTPUT):
        os.remove(ORBIS_OUTPUT)
    
    start = time.time()
    all_dfs = []
    
    # Process with ProcessPoolExecutor for true parallelism
    for batch_start in range(0, len(orbis_files), BATCH_SIZE):
        batch = orbis_files[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(orbis_files) + BATCH_SIZE - 1) // BATCH_SIZE
        
        batch_dfs = []
        
        # ThreadPoolExecutor is more stable in Colab than ProcessPoolExecutor
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = {ex.submit(process_file_polars, f): f for f in batch}
            for future in tqdm(as_completed(futures), total=len(batch), desc=f"Batch {batch_num}/{total_batches}"):
                try:
                    df = future.result(timeout=30)
                    if df is not None and not df.is_empty():
                        batch_dfs.append(df)
                except Exception as e:
                    pass
        
        if batch_dfs:
            batch_df = pl.concat(batch_dfs).unique(subset=["bvd_id"], keep="first")
            all_dfs.append(batch_df)
            
            total_so_far = sum(len(d) for d in all_dfs)
            elapsed = time.time() - start
            rate = (batch_start + len(batch)) / elapsed * 60
            print(f"   ✓ {total_so_far:,} records | {rate:.0f} files/min")
    
    # Combine and save
    print("\n📦 Combining and saving...")
    final_df = pl.concat(all_dfs).unique(subset=["bvd_id"], keep="first")
    
    # Ensure all columns exist
    for col in ALL_COLS:
        if col not in final_df.columns:
            final_df = final_df.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))
    
    final_df = final_df.select(ALL_COLS)
    final_df.write_parquet(ORBIS_OUTPUT, compression="snappy")
    
    elapsed = time.time() - start
    size_mb = os.path.getsize(ORBIS_OUTPUT) / (1024*1024)
    
    print(f"\n✅ COMPLETE!")
    print(f"   Records: {len(final_df):,}")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"   Time: {elapsed/60:.1f} min")
    print(f"   Speed: {len(orbis_files) / elapsed * 60:.0f} files/min")

if __name__ == "__main__":
    run_polars_ingestion()
