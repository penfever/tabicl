"""
Profiling script to analyze performance of generate_data.py calls from test_class_separability.py
"""

import cProfile
import pstats
import io
import sys
import os
import time
import tracemalloc
import numpy as np
import tempfile
from contextlib import contextmanager

# Add the parent directory to Python path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tabicl.prior.dataset import PriorDataset

@contextmanager
def profile_block(name):
    """Context manager for profiling a code block"""
    pr = cProfile.Profile()
    tracemalloc.start()
    
    start_time = time.time()
    pr.enable()
    
    yield
    
    pr.disable()
    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\n=== Profiling results for {name} ===")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Current memory: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
    
    # Get the stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Print top 20 functions
    print(s.getvalue())


def profile_dataset_generation():
    """Profile the dataset generation process"""
    print("Profiling dataset generation with different parameters...")
    
    # Parameters similar to test_class_separability.py
    test_params = {
        "batch_size": 32,
        "batch_size_per_gp": 1,
        "batch_size_per_subgp": 1,
        "prior_type": "deterministic_tree_scm",
        "min_features": 10,
        "max_features": 25,
        "min_classes": 10,
        "max_classes": 10,
        "min_seq_len": 2000,
        "max_seq_len": 2200,
        "replay_small": False,
        "seq_len_per_gp": False,
        "n_jobs": 1,
        "num_threads_per_generate": 2,
        "device": "cpu",
        "min_imbalance_ratio": 1.0,
        "max_imbalance_ratio": 2.0,
    }
    
    # Test with different class_separability values
    separability_values = [1.0, 5.0, 10.0]
    
    for sep_value in separability_values:
        print(f"\n\n{'='*60}")
        print(f"Testing with class_separability={sep_value}")
        print('='*60)
        
        # Add SCM parameters
        scm_fixed_hp = {
            'is_causal': False,
            'num_layers': 1,
            'min_swap_prob': 0.0,
            'max_swap_prob': 0.0,
            'transform_type': 'polynomial',
            'noise_type': 'swap',
            'noise_std': 0.001,
            'class_separability': sep_value
        }
        
        scm_sampled_hp = {}
        
        # Profile dataset creation
        with profile_block(f"Dataset creation (sep={sep_value})"):
            ds = PriorDataset(
                **test_params,
                scm_fixed_hp=scm_fixed_hp,
                scm_sampled_hp=scm_sampled_hp
            )
        
        # Profile getting first batch
        with profile_block(f"First batch generation (sep={sep_value})"):
            batch = next(iter(ds))
            n_samples = len(batch[2])  # Number of samples in y
            n_features = batch[0].shape[1]  # Number of features
            print(f"Batch shape: X={batch[0].shape}, y={batch[2].shape}")
        
        # Profile generating multiple batches
        n_batches = 10
        with profile_block(f"Generating {n_batches} batches (sep={sep_value})"):
            for i in range(n_batches):
                batch = next(iter(ds))
        
        del ds  # Clean up


def profile_specific_functions():
    """Profile specific functions that might be bottlenecks"""
    print("\n\n=== Profiling specific functions ===")
    
    # Create a simple dataset
    test_params = {
        "batch_size": 32,
        "batch_size_per_gp": 1,
        "batch_size_per_subgp": 1,
        "prior_type": "deterministic_tree_scm",
        "min_features": 20,
        "max_features": 20,
        "min_classes": 10,
        "max_classes": 10,
        "min_seq_len": 1000,
        "max_seq_len": 1000,
        "n_jobs": 1,
        "device": "cpu",
        "scm_fixed_hp": {
            'is_causal': False,
            'num_layers': 1,
            'min_swap_prob': 0.0,
            'max_swap_prob': 0.0,
            'class_separability': 5.0
        },
        "scm_sampled_hp": {}
    }
    
    ds = PriorDataset(**test_params)
    
    # Profile the apply_class_separability function directly
    from src.tabicl.prior.utils import apply_class_separability
    
    # Generate some test data
    X = np.random.randn(1000, 20)
    y = np.random.randint(0, 10, 1000)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    with profile_block("apply_class_separability function"):
        for _ in range(10):  # Run multiple times to get better stats
            X_scaled = apply_class_separability(X_tensor, y_tensor, 5.0)
    
    del ds


def analyze_deterministic_tree_scm():
    """Analyze the DeterministicTreeSCM specifically"""
    print("\n\n=== Analyzing DeterministicTreeSCM ===")
    
    from src.tabicl.prior.deterministic_tree_scm import DeterministicTreeSCM
    import torch
    
    # Create model with profiling
    with profile_block("DeterministicTreeSCM initialization"):
        model = DeterministicTreeSCM(
            seq_len=1000,
            num_features=20,
            num_outputs=1,
            num_layers=1,
            class_separability=5.0,
            device="cpu"
        )
    
    # Profile forward pass
    with profile_block("DeterministicTreeSCM forward pass"):
        for _ in range(5):
            X, y = model()
            print(f"  Generated: X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    # Run all profiling
    profile_dataset_generation()
    profile_specific_functions()
    analyze_deterministic_tree_scm()
    
    print("\n\nProfiling complete. Key insights:")
    print("1. Check cumulative time for expensive operations")
    print("2. Look for memory spikes")
    print("3. Identify bottlenecks in the generation pipeline")