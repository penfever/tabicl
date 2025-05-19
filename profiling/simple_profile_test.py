"""
Simple profiling script to analyze test_class_separability.py performance
"""

import time
import subprocess
import sys
import os
import tempfile
import shutil
import psutil
import numpy as np

def measure_step_time(description, func):
    """Measure time and memory for a step"""
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    start_time = time.time()
    result = func()
    elapsed = time.time() - start_time
    
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"{description}:")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Memory: {mem_before:.1f}MB -> {mem_after:.1f}MB (delta: {mem_after - mem_before:.1f}MB)")
    
    return result, elapsed

def profile_generation_subprocess():
    """Profile the generation subprocess directly"""
    print("=== Profiling dataset generation subprocess ===\n")
    
    temp_dir = tempfile.mkdtemp(prefix="profile_")
    
    # Test parameters from test_class_separability.py
    cmd = [
        sys.executable,
        "generate_data.py",
        "--n_datasets", "1",
        "--prior", "deterministic_tree_scm",
        "--num_gpus", "0",
        "--min_features", "10",
        "--max_features", "25", 
        "--min_seq", "2000",
        "--max_seq", "2001",
        "--min_classes", "10",
        "--max_classes", "10",
        "--class_separability", "5.0",
        "--max_imbalance_ratio", "2.0",
        "--out_dir", temp_dir,
        "--inner_bsz", "32",
        "--no_causal",
        "--num_layers", "1",
        "--min_swap_prob", "0.0",
        "--max_swap_prob", "0.0",
        "--transform_type", "polynomial",
        "--noise_type", "swap",
        "--noise_std", "0.001"
    ]
    
    # Change to tabicl directory
    tabicl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(tabicl_dir)
    
    # Run with timing
    def run_generation():
        return subprocess.run(cmd, capture_output=True, text=True)
    
    result, generation_time = measure_step_time("Dataset generation", run_generation)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return
    
    # Load the generated dataset
    def load_data():
        files = os.listdir(temp_dir)
        X_file = [f for f in files if f.endswith("_X.npy")][0]
        y_file = [f for f in files if f.endswith("_y.npy")][0]
        X = np.load(os.path.join(temp_dir, X_file))
        y = np.load(os.path.join(temp_dir, y_file))
        return X, y
    
    (X, y), load_time = measure_step_time("Loading dataset", load_data)
    print(f"  Shape: X={X.shape}, y={y.shape}")
    
    # Clean up
    shutil.rmtree(temp_dir)
    
    return generation_time, load_time

def profile_simple_generation():
    """Profile a simpler generation approach to compare"""
    print("\n=== Profiling simple sklearn make_classification ===\n")
    
    from sklearn.datasets import make_classification
    
    def generate_sklearn():
        return make_classification(
            n_samples=2000,
            n_features=20,
            n_informative=10,
            n_redundant=2,
            n_classes=10,
            n_clusters_per_class=2,
            class_sep=5.0,
            random_state=42
        )
    
    (X, y), sklearn_time = measure_step_time("sklearn make_classification", generate_sklearn)
    print(f"  Shape: X={X.shape}, y={y.shape}")
    
    return sklearn_time

def profile_parts():
    """Profile individual parts of the generation"""
    print("\n=== Profiling individual components ===\n")
    
    import torch
    from tabicl.prior.utils import apply_class_separability
    
    # Generate test data
    def create_test_data():
        X = torch.randn(2000, 20)
        y = torch.randint(0, 10, (2000,))
        return X, y
    
    X, y = create_test_data()
    
    # Profile apply_class_separability
    def apply_sep():
        return apply_class_separability(X, y, 5.0)
    
    _, sep_time = measure_step_time("apply_class_separability", apply_sep)
    
    # Profile correlation computation specifically
    def compute_correlations():
        X_cpu = X.cpu().numpy()
        y_cpu = y.cpu().numpy()
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X_cpu[:, i], y_cpu)[0, 1]
            correlations.append(abs(corr))
        return correlations
    
    _, corr_time = measure_step_time("Correlation computation", compute_correlations)
    
    return sep_time, corr_time

def main():
    """Run all profiling tests"""
    print("Profiling TabICL Dataset Generation Performance")
    print("=" * 50 + "\n")
    
    # Profile the full generation subprocess
    gen_time, load_time = profile_generation_subprocess()
    
    # Profile sklearn for comparison
    sklearn_time = profile_simple_generation()
    
    # Profile individual components
    sep_time, corr_time = profile_parts()
    
    # Summary
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Full dataset generation: {gen_time:.2f}s")
    print(f"Dataset loading: {load_time:.2f}s")
    print(f"sklearn make_classification: {sklearn_time:.2f}s (baseline)")
    print(f"apply_class_separability: {sep_time:.4f}s")
    print(f"Correlation computation: {corr_time:.4f}s")
    print(f"\nSlowdown vs sklearn: {gen_time/sklearn_time:.1f}x")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("1. Consider caching correlation computations")
    print("2. Use vectorized operations where possible")
    print("3. Profile tree model fitting separately")
    print("4. Consider parallel generation for multiple datasets")

if __name__ == "__main__":
    main()