"""
embeddings.py - GPU-Accelerated Embeddings + FAISS Index

Computes sentence embeddings for company names and descriptions.
Builds FAISS indexes for efficient approximate nearest neighbor search.

DESIGNED FOR GOOGLE COLAB A100:
------------------------------
- batch_size=512 for GPU efficiency
- float16 storage for memory optimization
- GPU-accelerated FAISS when available

EMBEDDING TYPES:
---------------
1. name_embedding: on legal_stripped name
2. desc_embedding: on combined industries + description
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# EMBEDDING COMPUTATION
# =============================================================================

def compute_embeddings(
    texts: List[str],
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 512,
    device: str = 'auto',
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute sentence embeddings using SentenceTransformers.
    
    Args:
        texts: List of texts to embed
        model_name: HuggingFace model name
        batch_size: Batch size for encoding
        device: 'cuda', 'cpu', or 'auto' (detect GPU)
        show_progress: Show progress bar
    
    Returns:
        np.ndarray of shape (len(texts), embedding_dim)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        raise
    
    # Auto-detect device
    if device == 'auto':
        try:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
                logger.info("Apple Silicon GPU (MPS) detected and enabled! 🚀")
            else:
                device = 'cpu'
        except ImportError:
            device = 'cpu'
    
    logger.info(f"Computing embeddings for {len(texts)} texts on {device}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Batch size: {batch_size}")
    
    # Load model
    model = SentenceTransformer(model_name, device=device)
    
    # Clean texts (handle None/nan)
    clean_texts = [str(t) if pd.notna(t) and t else '' for t in texts]
    
    # Encode
    embeddings = model.encode(
        clean_texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    
    logger.info(f"Computed embeddings: shape {embeddings.shape}")
    
    return embeddings


def compute_embeddings_streaming(
    texts: List[str],
    output_path: str,
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 512,
    chunk_size: int = 50000,  # Write to disk every 50K texts
    device: str = 'auto',
) -> Tuple[str, int]:
    """
    MEMORY-EFFICIENT streaming embeddings that write to disk in chunks.
    
    Uses ~2-3 GB RAM instead of 20+ GB for 16M texts.
    
    Args:
        texts: List of texts to embed
        output_path: Path to save the memmapped numpy file
        model_name: Embedding model name
        batch_size: Batch size for GPU
        chunk_size: Texts per chunk written to disk
        device: Device to use
    
    Returns:
        (output_path, embedding_dim)
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError:
        raise ImportError("sentence-transformers and torch required")
    
    from tqdm import tqdm
    
    # Auto-detect device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    n_texts = len(texts)
    logger.info(f"Computing embeddings for {n_texts:,} texts on {device} (STREAMING MODE)")
    logger.info(f"  Chunk size: {chunk_size:,} (writes to disk every {chunk_size:,} texts)")
    
    # Load model to get embedding dimension
    model = SentenceTransformer(model_name, device=device)
    dim = model.get_sentence_embedding_dimension()
    
    # Create memory-mapped file for output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize memmap (this creates the file on disk)
    mmap_path = str(output_path)
    embeddings_mmap = np.memmap(mmap_path, dtype=np.float16, mode='w+', shape=(n_texts, dim))
    
    # Process in chunks
    n_chunks = (n_texts + chunk_size - 1) // chunk_size
    
    for chunk_idx in tqdm(range(n_chunks), desc="Embedding chunks"):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_texts)
        
        chunk_texts = texts[start_idx:end_idx]
        clean_chunk = [str(t) if pd.notna(t) and t else '' for t in chunk_texts]
        
        # Encode chunk
        chunk_embeddings = model.encode(
            clean_chunk,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        
        # Write directly to memmap (on disk)
        embeddings_mmap[start_idx:end_idx] = chunk_embeddings.astype(np.float16)
        
        # Flush to disk and free memory
        embeddings_mmap.flush()
        del chunk_embeddings
        
        # Force garbage collection
        import gc
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Finalize memmap
    del embeddings_mmap
    
    logger.info(f"Saved streaming embeddings to {output_path} (shape: {n_texts} x {dim})")
    
    return str(output_path), dim


def compute_embeddings_batched(
    df: pd.DataFrame,
    text_column: str,
    id_column: str,
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 512,
    output_dir: str = None,
    prefix: str = 'embeddings',
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute embeddings for a DataFrame column with checkpointing.
    
    Saves intermediate results for large datasets.
    
    Args:
        df: DataFrame with texts
        text_column: Column containing texts to embed
        id_column: Column containing IDs for mapping
        model_name: Embedding model
        batch_size: Batch size
        output_dir: Directory for checkpoints
        prefix: File prefix for outputs
    
    Returns:
        (embeddings, index_df) where index_df maps row index to ID
    """
    texts = df[text_column].tolist()
    ids = df[id_column].tolist()
    
    embeddings = compute_embeddings(
        texts,
        model_name=model_name,
        batch_size=batch_size,
    )
    
    # Create index DataFrame
    index_df = pd.DataFrame({
        'idx': range(len(ids)),
        'id': ids,
        'text_preview': [str(t)[:50] if pd.notna(t) else '' for t in texts],
    })
    
    # Save if output_dir specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings as float16 for space
        emb_path = output_dir / f'{prefix}.npy'
        np.save(emb_path, embeddings.astype(np.float16))
        logger.info(f"Saved embeddings to {emb_path}")
        
        # Save index
        idx_path = output_dir / f'{prefix}_index.parquet'
        index_df.to_parquet(idx_path, index=False)
        logger.info(f"Saved index to {idx_path}")
    
    return embeddings, index_df


# =============================================================================
# FAISS INDEX
# =============================================================================

def build_faiss_index(
    embeddings: np.ndarray,
    use_gpu: bool = True,
    index_type: str = 'flat',
) -> 'faiss.Index':
    """
    Build FAISS index for fast similarity search.
    
    Args:
        embeddings: Embeddings array (n_samples, dim)
        use_gpu: Use GPU-accelerated FAISS if available
        index_type: 'flat' (exact) or 'ivf' (approximate)
    
    Returns:
        faiss.Index
    """
    try:
        import faiss
    except ImportError:
        logger.error("faiss not installed. Run: pip install faiss-cpu or faiss-gpu")
        raise
    
    dim = embeddings.shape[1]
    embeddings = embeddings.astype(np.float32)
    
    # Normalize for cosine similarity (using inner product on normalized vectors)
    faiss.normalize_L2(embeddings)
    
    if index_type == 'flat':
        # Exact search
        index = faiss.IndexFlatIP(dim)
    else:
        # IVF for approximate search on large datasets
        nlist = min(100, len(embeddings) // 100)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
    
    # Move to GPU if available and requested
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            logger.info("Using GPU-accelerated FAISS")
        except Exception as e:
            logger.warning(f"GPU FAISS not available, using CPU: {e}")
    
    index.add(embeddings)
    logger.info(f"Built FAISS index with {index.ntotal} vectors")
    
    return index


def search_faiss(
    index: 'faiss.Index',
    query_embeddings: np.ndarray,
    k: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search FAISS index for nearest neighbors.
    
    Args:
        index: FAISS index
        query_embeddings: Query vectors
        k: Number of neighbors to return
    
    Returns:
        (distances, indices) arrays of shape (n_queries, k)
    """
    import faiss
    
    query_embeddings = query_embeddings.astype(np.float32)
    faiss.normalize_L2(query_embeddings)
    
    distances, indices = index.search(query_embeddings, k)
    
    return distances, indices


def save_faiss_index(index: 'faiss.Index', path: str) -> None:
    """Save FAISS index to disk."""
    import faiss
    
    # Move to CPU if on GPU
    if hasattr(index, 'getDevice'):
        index = faiss.index_gpu_to_cpu(index)
    
    faiss.write_index(index, path)
    logger.info(f"Saved FAISS index to {path}")


def load_faiss_index(path: str, use_gpu: bool = False) -> 'faiss.Index':
    """Load FAISS index from disk."""
    import faiss
    
    index = faiss.read_index(path)
    
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception as e:
            logger.warning(f"Could not load to GPU: {e}")
    
    return index


# =============================================================================
# EMBEDDING PIPELINE FOR ENTITY MATCHING
# =============================================================================

def compute_all_embeddings(
    cb_data: pd.DataFrame,
    orbis_data: pd.DataFrame,
    output_dir: str,
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 512,
    use_streaming: bool = True,  # NEW: Enable streaming for large datasets
    streaming_chunk_size: int = 50000,
) -> Dict[str, str]:
    """
    Compute all embeddings for entity matching.
    
    Uses STREAMING mode for Orbis (16M+ records) to avoid RAM exhaustion.
    
    Recommended models (by quality, with 15GB VRAM):
    - 'BAAI/bge-large-en-v1.5' (1.3GB, 1024-dim) ✅ Best for similarity
    - 'intfloat/e5-large-v2' (1.3GB, 1024-dim) ✅ Excellent retrieval
    - 'sentence-transformers/all-mpnet-base-v2' (420MB, 768-dim) Good balance
    - 'all-MiniLM-L6-v2' (80MB, 384-dim) Fast, lower quality
    
    Args:
        cb_data: Crunchbase data with normalized fields
        orbis_data: Orbis data with normalized fields
        output_dir: Directory to save outputs
        model_name: Embedding model name
        batch_size: Batch size for encoding
        use_streaming: Use streaming mode for Orbis (saves RAM)
        streaming_chunk_size: Chunk size for streaming
    
    Returns:
        Dict of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    outputs = {}
    
    # CB Name embeddings (small, use regular mode)
    logger.info("Computing CB name embeddings...")
    cb_name_col = 'cb_name_legal_stripped' if 'cb_name_legal_stripped' in cb_data.columns else 'cb_name'
    _, _ = compute_embeddings_batched(
        cb_data, cb_name_col, 'cb_id',
        model_name=model_name, batch_size=batch_size,
        output_dir=str(output_dir), prefix='cb_name_emb'
    )
    outputs['cb_name_emb'] = str(output_dir / 'cb_name_emb.npy')
    
    # CB Description embeddings (small)
    logger.info("Computing CB description embeddings...")
    cb_data = cb_data.copy()
    cb_data['_desc_combined'] = cb_data.apply(
        lambda r: ' '.join(filter(None, [
            str(r.get('cb_industries', '')),
            str(r.get('cb_industry_groups', '')),
            str(r.get('cb_description', '')),
        ])), axis=1
    )
    _, _ = compute_embeddings_batched(
        cb_data, '_desc_combined', 'cb_id',
        model_name=model_name, batch_size=batch_size,
        output_dir=str(output_dir), prefix='cb_desc_emb'
    )
    outputs['cb_desc_emb'] = str(output_dir / 'cb_desc_emb.npy')
    
    # =========================================================================
    # ORBIS EMBEDDINGS - Use STREAMING mode to save RAM
    # =========================================================================
    orbis_name_col = 'orbis_name_legal_stripped' if 'orbis_name_legal_stripped' in orbis_data.columns else 'orbis_name'
    orbis_desc_col = 'orbis_trade_desc' if 'orbis_trade_desc' in orbis_data.columns else 'orbis_name'
    
    if use_streaming and len(orbis_data) > 100000:
        logger.info(f"Using STREAMING mode for {len(orbis_data):,} Orbis records (saves ~20GB RAM)")
        
        # Orbis Name embeddings (streaming)
        logger.info("Computing Orbis name embeddings (streaming)...")
        texts = orbis_data[orbis_name_col].tolist()
        compute_embeddings_streaming(
            texts=texts,
            output_path=str(output_dir / 'orbis_name_emb.npy'),
            model_name=model_name,
            batch_size=batch_size,
            chunk_size=streaming_chunk_size,
        )
        outputs['orbis_name_emb'] = str(output_dir / 'orbis_name_emb.npy')
        
        # Save index separately
        idx_df = pd.DataFrame({'idx': range(len(orbis_data)), 'id': orbis_data['bvd_id']})
        idx_df.to_parquet(output_dir / 'orbis_name_emb_index.parquet', index=False)
        
        # Orbis Description embeddings (streaming)
        logger.info("Computing Orbis description embeddings (streaming)...")
        texts = orbis_data[orbis_desc_col].tolist()
        compute_embeddings_streaming(
            texts=texts,
            output_path=str(output_dir / 'orbis_desc_emb.npy'),
            model_name=model_name,
            batch_size=batch_size,
            chunk_size=streaming_chunk_size,
        )
        outputs['orbis_desc_emb'] = str(output_dir / 'orbis_desc_emb.npy')
        
        # Save index
        idx_df.to_parquet(output_dir / 'orbis_desc_emb_index.parquet', index=False)
        
    else:
        # Small dataset, use regular batched mode
        logger.info("Computing Orbis name embeddings...")
        _, _ = compute_embeddings_batched(
            orbis_data, orbis_name_col, 'bvd_id',
            model_name=model_name, batch_size=batch_size,
            output_dir=str(output_dir), prefix='orbis_name_emb'
        )
        outputs['orbis_name_emb'] = str(output_dir / 'orbis_name_emb.npy')
        
        logger.info("Computing Orbis description embeddings...")
        _, _ = compute_embeddings_batched(
            orbis_data, orbis_desc_col, 'bvd_id',
            model_name=model_name, batch_size=batch_size,
            output_dir=str(output_dir), prefix='orbis_desc_emb'
        )
        outputs['orbis_desc_emb'] = str(output_dir / 'orbis_desc_emb.npy')
    
    logger.info(f"All embeddings saved to {output_dir}")
    
    return outputs


def build_all_faiss_indexes(
    embeddings_dir: str,
    use_gpu: bool = True,
) -> Dict[str, str]:
    """
    Build FAISS indexes for all Orbis embeddings.
    
    Args:
        embeddings_dir: Directory containing embeddings
        use_gpu: Use GPU FAISS
    
    Returns:
        Dict of index file paths
    """
    embeddings_dir = Path(embeddings_dir)
    outputs = {}
    
    for emb_type in ['orbis_name_emb', 'orbis_desc_emb']:
        emb_path = embeddings_dir / f'{emb_type}.npy'
        if emb_path.exists():
            logger.info(f"Building FAISS index for {emb_type}...")
            embeddings = np.load(emb_path).astype(np.float32)
            index = build_faiss_index(embeddings, use_gpu=use_gpu)
            
            index_path = embeddings_dir / f'{emb_type}_faiss.index'
            save_faiss_index(index, str(index_path))
            outputs[emb_type] = str(index_path)
    
    return outputs


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("Embeddings module loaded.")
    print("Key functions:")
    print("  - compute_embeddings(texts, model_name)")
    print("  - build_faiss_index(embeddings)")
    print("  - compute_all_embeddings(cb_data, orbis_data, output_dir)")
    print("  - build_all_faiss_indexes(embeddings_dir)")
    print("\nNote: GPU operations require: sentence-transformers, faiss-gpu")
