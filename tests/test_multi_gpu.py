"""Test script for multi-GPU data generation."""

import subprocess
import tempfile
import shutil
import time
import torch
from pathlib import Path


def test_multi_gpu_generation():
    """Test multi-GPU data generation with different configurations."""
    
    print("Testing multi-GPU data generation...")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Create temporary directory for test outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Test configurations
        test_configs = [
            # (n_datasets, num_gpus, description)
            (10, 0, "CPU only"),
            (10, 1, "Single GPU"),
            (20, 2, "Two GPUs"),
            (40, -1, "All available GPUs"),
        ]
        
        for n_datasets, num_gpus, desc in test_configs:
            # Skip multi-GPU tests if not enough GPUs
            if num_gpus > torch.cuda.device_count():
                print(f"Skipping {desc} test (not enough GPUs)")
                continue
                
            print(f"\nTesting {desc}...")
            output_dir = tmpdir / f"test_{desc.replace(' ', '_')}"
            
            cmd = [
                "python", "generate_data.py",
                "--n_datasets", str(n_datasets),
                "--out_dir", str(output_dir),
                "--num_gpus", str(num_gpus),
                "--prior", "dummy",  # Use dummy for faster testing
                "--inner_bsz", "2"
            ]
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            end_time = time.time()
            
            if result.returncode == 0:
                # Check that files were created
                num_files = len(list(output_dir.glob("*.npy")))
                expected_files = n_datasets * 2  # X and y for each dataset
                
                if num_files == expected_files:
                    print(f"✓ Success: Generated {n_datasets} datasets in {end_time - start_time:.2f}s")
                else:
                    print(f"✗ Error: Expected {expected_files} files, found {num_files}")
            else:
                print(f"✗ Error: Command failed")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
    
    print("\nMulti-GPU tests completed!")


def benchmark_multi_gpu():
    """Benchmark multi-GPU performance."""
    
    if torch.cuda.device_count() < 2:
        print("Need at least 2 GPUs for benchmarking")
        return
    
    print("\nBenchmarking multi-GPU performance...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        n_datasets = 1000
        
        # Test different GPU configurations
        gpu_configs = [1, 2, 4, 8, -1]
        
        results = []
        for num_gpus in gpu_configs:
            if num_gpus > torch.cuda.device_count():
                continue
                
            actual_gpus = num_gpus if num_gpus > 0 else torch.cuda.device_count()
            print(f"\nBenchmarking with {actual_gpus} GPU(s)...")
            
            output_dir = tmpdir / f"bench_{num_gpus}gpu"
            
            cmd = [
                "python", "generate_data.py",
                "--n_datasets", str(n_datasets),
                "--out_dir", str(output_dir),
                "--num_gpus", str(num_gpus),
                "--prior", "mix_scm",
                "--inner_bsz", "64"
            ]
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            end_time = time.time()
            
            if result.returncode == 0:
                elapsed = end_time - start_time
                rate = n_datasets / elapsed
                results.append((actual_gpus, elapsed, rate))
                print(f"Completed in {elapsed:.2f}s ({rate:.1f} datasets/sec)")
            else:
                print(f"Failed: {result.stderr}")
        
        # Print summary
        print("\n" + "="*50)
        print("Benchmark Summary")
        print("="*50)
        print(f"Dataset count: {n_datasets}")
        print(f"{'GPUs':<10} {'Time (s)':<15} {'Rate (ds/s)':<15} {'Speedup':<10}")
        print("-"*50)
        
        if results:
            base_time = results[0][1]  # Single GPU time
            for gpus, elapsed, rate in results:
                speedup = base_time / elapsed
                print(f"{gpus:<10} {elapsed:<15.2f} {rate:<15.1f} {speedup:<10.2f}x")


if __name__ == "__main__":
    test_multi_gpu_generation()
    benchmark_multi_gpu()