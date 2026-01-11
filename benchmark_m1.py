
import time
import torch
import pandas as pd
import numpy as np

def test_mps():
    print("\n--- 1. MPS (GPU) Check ---")
    try:
        if torch.backends.mps.is_available():
            print("✅ MPS is available! Using Apple Neural Engine.")
            x = torch.ones(5, device='mps')
            print(f"Tensor on device: {x.device}")
        else:
            print("❌ MPS not detected. Is this an M1/M2 Mac?")
    except Exception as e:
        print(f"❌ Error checking MPS: {e}")

def test_vectorization():
    print("\n--- 2. Vectorization Speed Test ---")
    size = 100_000
    df = pd.DataFrame({'a': np.random.rand(size), 'b': np.random.rand(size)})
    
    # Slow loop
    print(f"Benchmarking iteration vs vectorization on {size:,} rows...")
    start = time.time()
    res = [x > y for x, y in zip(df.a, df.b)]
    loop_time = time.time() - start
    
    # Fast vector
    start = time.time()
    res = df.a > df.b
    vec_time = time.time() - start
    
    print(f"Loop time: {loop_time:.4f}s")
    print(f"Vector time: {vec_time:.4f}s")
    print(f"Speedup: {loop_time/vec_time:.1f}x 🚀")

if __name__ == "__main__":
    test_mps()
    test_vectorization()
