# =============================================================================
# 🚀 COMPLETE ENTITY RESOLUTION PIPELINE - T4 OPTIMIZED
# =============================================================================
# Optimized for: T4 GPU | 50GB RAM | 15GB VRAM
# Key: Copies files to local SSD first for 10x faster I/O
# =============================================================================

import os
import sys
import glob
import time
import shutil
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_PATH = "/content/drive/Othercomputers/My MacBook Pro/Downloads/ricerca"
PIPELINE_DIR = f"{BASE_PATH}/entity-resolution-pipeline"
CB_DIR = f"{BASE_PATH}/dati europe cb"
ORBIS_DIR_DRIVE = f"{BASE_PATH}/new orbis"
ORBIS_DIR_LOCAL = "/content/orbis_local"
OUTPUT_DIR = f"{PIPELINE_DIR}/data/interim"

NUM_WORKERS = 16
BATCH_SIZE = 100

# Setup
sys.path.insert(0, PIPELINE_DIR)
os.chdir(PIPELINE_DIR)
os.makedirs(f"{OUTPUT_DIR}/cb_clean", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/orbis_clean", exist_ok=True)
os.makedirs(ORBIS_DIR_LOCAL, exist_ok=True)

print("✓ Configuration loaded")

# =============================================================================
# STEP 1: COPY FILES TO LOCAL SSD (10x FASTER)
# =============================================================================
print("\n" + "="*60)
print("📥 COPYING FILES TO LOCAL SSD")
print("="*60)

orbis_files_drive = sorted(glob.glob(f"{ORBIS_DIR_DRIVE}/*.xlsx"))
orbis_files_local = glob.glob(f"{ORBIS_DIR_LOCAL}/*.xlsx")

if len(orbis_files_local) >= len(orbis_files_drive):
    print(f"✓ Already copied ({len(orbis_files_local)} files)")
else:
    print(f"Copying {len(orbis_files_drive)} files...")
    for src in tqdm(orbis_files_drive, desc="Copying"):
        dst = f"{ORBIS_DIR_LOCAL}/{os.path.basename(src)}"
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    print("✓ Copy complete")

orbis_files = sorted(glob.glob(f"{ORBIS_DIR_LOCAL}/*.xlsx"))
print(f"📂 {len(orbis_files)} files ready for processing")

# =============================================================================
# ORBIS PROCESSING FUNCTIONS
# =============================================================================
ORBIS_SCHEMA = {
    'bvd_id': 'BvD ID number', 'orbis_name': 'Company name Latin alphabet',
    'orbis_name_latin': 'Company name Latin alphabet', 'orbis_country': 'Country ISO code',
    'orbis_city': 'City', 'orbis_city_latin': 'City Latin Alphabet',
    'orbis_postcode': 'Postcode', 'orbis_website': 'Website address',
    'orbis_email': 'E-mail address', 'orbis_phone': 'Phone number',
    'orbis_incorp_date': 'Date of incorporation', 'orbis_trade_desc': 'Trade description (English)',
    'orbis_trade_desc_orig': 'Trade description (Original language)',
    'orbis_nace': 'NACE Rev. 2 core code', 'orbis_nace_desc': 'NACE Rev. 2 core code description',
    'orbis_legal_form': 'Standardised legal form', 'orbis_operating_revenue': 'Operating revenue (Turnover)',
    'orbis_total_assets': 'Total assets', 'orbis_num_employees': 'Number of employees',
    'sub_bvd_id': 'SUB - BvD ID number', 'sh_bvd_id': 'SH - BvD ID number',
    'guo_bvd_id': 'GUO - BvD ID number', 'branch_bvd_id': 'BRANCH - BvD ID number',
    'sh_name': 'SH - Name', 'sh_country': 'SH - Country ISO code',
    'sh_type': 'SH - Type', 'sh_direct_pct': 'SH - Direct %', 'sh_total_pct': 'SH - Total %',
}

FIRST_VALUE_COLS = ['bvd_id', 'orbis_name', 'orbis_name_latin', 'orbis_country', 'orbis_city', 
    'orbis_city_latin', 'orbis_postcode', 'orbis_phone', 'orbis_incorp_date', 'orbis_trade_desc',
    'orbis_trade_desc_orig', 'orbis_nace', 'orbis_nace_desc', 'orbis_legal_form',
    'orbis_operating_revenue', 'orbis_total_assets', 'orbis_num_employees']

AGGREGATE_COLS = ['orbis_website', 'orbis_email', 'sub_bvd_id', 'sh_bvd_id', 'guo_bvd_id',
    'branch_bvd_id', 'sh_name', 'sh_country', 'sh_type', 'sh_direct_pct', 'sh_total_pct']

def normalize_col(col):
    return ' '.join(str(col).split()).strip().lower().replace('_', ' ').replace('-', ' ')

def find_column(cols, target):
    target_n = normalize_col(target)
    for c in cols:
        if normalize_col(c) == target_n:
            return c
    return None

def apply_schema(df, schema):
    result = {}
    for new, orig in schema.items():
        col = find_column(df.columns, orig)
        result[new] = df[col].copy() if col else pd.NA
    return pd.DataFrame(result)

def aggregate_fast(df):
    first_col = df.columns[0]
    df = df.copy()
    df['_idx'] = df[first_col].ffill()
    existing_first = [c for c in FIRST_VALUE_COLS if c in df.columns]
    first_vals = df.groupby('_idx', sort=False)[existing_first].first()
    existing_agg = [c for c in AGGREGATE_COLS if c in df.columns]
    if existing_agg:
        for col in existing_agg:
            df[col] = df[col].astype(str).replace({'nan': '', 'None': ''})
        multi_vals = df.groupby('_idx', sort=False)[existing_agg].agg(lambda x: '|'.join(filter(None, x.unique())))
        multi_vals = multi_vals.replace('', pd.NA)
        result = first_vals.join(multi_vals)
    else:
        result = first_vals
    result = result.reset_index(drop=True)
    return result[result['bvd_id'].notna()]

def process_file(fp):
    try:
        try:
            xls = pd.ExcelFile(fp, engine='calamine')
        except:
            xls = pd.ExcelFile(fp)
        sheet = 'Results' if 'Results' in xls.sheet_names else (xls.sheet_names[0] if xls.sheet_names else None)
        if not sheet:
            return pd.DataFrame()
        df = pd.read_excel(xls, sheet_name=sheet)
        if len(df) == 0:
            return pd.DataFrame()
        df_mapped = apply_schema(df, ORBIS_SCHEMA)
        df_mapped.insert(0, '_raw_idx', df.iloc[:, 0])
        return aggregate_fast(df_mapped)
    except:
        return pd.DataFrame()

# =============================================================================
# STEP 2: PROCESS ORBIS FILES
# =============================================================================
print("\n" + "="*60)
print("🚀 PROCESSING ORBIS FILES")
print("="*60)

ORBIS_OUTPUT = f"{OUTPUT_DIR}/orbis_clean/orbis_raw.parquet"
if os.path.exists(ORBIS_OUTPUT):
    os.remove(ORBIS_OUTPUT)

start = time.time()
writer = None
total_records = 0
SCHEMA_COLS = list(ORBIS_SCHEMA.keys())

try:
    for batch_start in range(0, len(orbis_files), BATCH_SIZE):
        batch = orbis_files[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(orbis_files) + BATCH_SIZE - 1) // BATCH_SIZE
        
        dfs = []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = {ex.submit(process_file, f): f for f in batch}
            for future in tqdm(as_completed(futures), total=len(batch), desc=f"Batch {batch_num}/{total_batches}"):
                try:
                    df = future.result(timeout=60)
                    if df is not None and len(df) > 0:
                        dfs.append(df)
                except:
                    pass
        
        if dfs:
            batch_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['bvd_id'], keep='first')
            for col in SCHEMA_COLS:
                if col not in batch_df.columns:
                    batch_df[col] = pd.NA
            batch_df = batch_df[SCHEMA_COLS].reset_index(drop=True)
            for col in batch_df.select_dtypes(include=['object']).columns:
                batch_df[col] = batch_df[col].astype('string')
            
            table = pa.Table.from_pandas(batch_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(ORBIS_OUTPUT, table.schema, compression='snappy')
            writer.write_table(table)
            total_records += len(batch_df)
            
            elapsed = time.time() - start
            print(f"   ✓ {total_records:,} records | {elapsed/60:.1f} min elapsed")
finally:
    if writer:
        writer.close()

print(f"\n✅ Orbis complete: {total_records:,} records in {(time.time()-start)/60:.1f} min")

# =============================================================================
# STEP 3: LOAD CRUNCHBASE
# =============================================================================
print("\n" + "="*60)
print("📥 LOADING CRUNCHBASE")
print("="*60)

from src.data_io import load_crunchbase_companies, load_crunchbase_funding, load_crunchbase_investors, load_company_lookup, save_to_parquet

cb_companies = load_crunchbase_companies(CB_DIR)
save_to_parquet(cb_companies, f"{OUTPUT_DIR}/cb_clean/cb_raw_companies.parquet")

cb_funding = load_crunchbase_funding(CB_DIR)
save_to_parquet(cb_funding, f"{OUTPUT_DIR}/cb_clean/cb_raw_funding.parquet")

cb_investors = load_crunchbase_investors(CB_DIR)
save_to_parquet(cb_investors, f"{OUTPUT_DIR}/cb_clean/cb_raw_investors.parquet")

cb_lookup = load_company_lookup(CB_DIR)
save_to_parquet(cb_lookup, f"{OUTPUT_DIR}/cb_clean/cb_domain_lookup.parquet")

print("✅ Crunchbase loaded")

# =============================================================================
# STEP 4: RUN REMAINING PIPELINE
# =============================================================================
print("\n" + "="*60)
print("🔧 RUNNING PIPELINE STEPS")
print("="*60)

import subprocess
for step in ['clean', 'block', 'match', 'score', 'resolve']:
    print(f"\n▶ Running: {step}")
    subprocess.run(['python', 'run_pipeline.py', '--config', 'configs/colab_gpu.yaml', '--step', step], check=True)

print("\n" + "="*60)
print("🎉 COMPLETE!")
print("="*60)
print(f"Total time: {(time.time()-start)/60:.1f} min")
