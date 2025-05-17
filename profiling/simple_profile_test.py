#!/usr/bin/env python3
"""
Simple profiling test for noise injection methods
"""
import numpy as np
import time
import cProfile
import pstats

# Simulate the noise injection methods
def profile_noise_methods():
    n_samples = 5000
    n_dims = 10
    y = np.random.randn(n_samples, n_dims)
    
    # Test corrupt noise
    print("Testing corrupt noise...")
    start = time.time()
    swap_prob = 0.3
    n_corrupt = int(swap_prob * n_samples)
    corrupt_indices = np.random.choice(n_samples, n_corrupt, replace=False)
    
    y_corrupt = y.copy()
    for idx in corrupt_indices:
        noise_scale = np.std(y) * np.random.uniform(0.5, 2.0)
        y_corrupt[idx] += np.random.randn(*y[idx].shape) * noise_scale
    
    corrupt_time = time.time() - start
    print(f"Corrupt time: {corrupt_time:.3f}s")
    
    # Test boundary blur
    print("\nTesting boundary blur noise...")
    start = time.time()
    y_blur = y.copy()
    
    for dim in range(y.shape[1]):
        sorted_indices = np.argsort(y[:, dim])
        middle_start = int(0.3 * n_samples)
        middle_end = int(0.7 * n_samples)
        boundary_indices = sorted_indices[middle_start:middle_end]
        
        n_blur = int(swap_prob * len(boundary_indices))
        blur_indices = np.random.choice(boundary_indices, n_blur, replace=False)
        
        noise_scale = np.std(y[:, dim]) * 0.3
        y_blur[blur_indices, dim] += np.random.randn(n_blur) * noise_scale
    
    blur_time = time.time() - start
    print(f"Boundary blur time: {blur_time:.3f}s")
    
    # Optimized corrupt noise
    print("\nTesting optimized corrupt noise...")
    start = time.time()
    n_corrupt = int(swap_prob * n_samples)
    corrupt_indices = np.random.choice(n_samples, n_corrupt, replace=False)
    
    y_corrupt_opt = y.copy()
    # Vectorized operation instead of loop
    noise_scales = np.std(y) * np.random.uniform(0.5, 2.0, size=(n_corrupt, 1))
    noise = np.random.randn(n_corrupt, n_dims) * noise_scales
    y_corrupt_opt[corrupt_indices] += noise
    
    corrupt_opt_time = time.time() - start
    print(f"Optimized corrupt time: {corrupt_opt_time:.3f}s")
    print(f"Speedup: {corrupt_time/corrupt_opt_time:.2f}x")
    
    # Optimized boundary blur
    print("\nTesting optimized boundary blur...")
    start = time.time()
    y_blur_opt = y.copy()
    
    for dim in range(y.shape[1]):
        sorted_indices = np.argsort(y[:, dim])
        middle_start = int(0.3 * n_samples)
        middle_end = int(0.7 * n_samples)
        boundary_indices = sorted_indices[middle_start:middle_end]
        
        n_blur = int(swap_prob * len(boundary_indices))
        blur_indices = np.random.choice(boundary_indices, n_blur, replace=False)
        
        # Vectorized noise generation
        noise_scale = np.std(y[:, dim]) * 0.3
        y_blur_opt[blur_indices, dim] += np.random.randn(n_blur) * noise_scale
    
    blur_opt_time = time.time() - start
    print(f"Optimized boundary blur time: {blur_opt_time:.3f}s")
    
    # The bottleneck is likely the loop over dimensions, let's try a different approach
    print("\nTesting fully vectorized boundary blur...")
    start = time.time()
    y_blur_vec = y.copy()
    
    # Process all dimensions at once
    n_blur_per_dim = int(swap_prob * n_samples * 0.4)  # 40% of samples are in boundary
    all_blur_indices = []
    all_dims = []
    
    for dim in range(y.shape[1]):
        sorted_indices = np.argsort(y[:, dim])
        middle_start = int(0.3 * n_samples)
        middle_end = int(0.7 * n_samples)
        boundary_indices = sorted_indices[middle_start:middle_end]
        
        if len(boundary_indices) > n_blur_per_dim:
            blur_indices = np.random.choice(boundary_indices, n_blur_per_dim, replace=False)
        else:
            blur_indices = boundary_indices
        
        all_blur_indices.extend(blur_indices)
        all_dims.extend([dim] * len(blur_indices))
    
    # Apply noise in one vectorized operation
    noise_scale = np.std(y) * 0.3
    noise = np.random.randn(len(all_blur_indices)) * noise_scale
    y_blur_vec[all_blur_indices, all_dims] += noise
    
    blur_vec_time = time.time() - start
    print(f"Fully vectorized boundary blur time: {blur_vec_time:.3f}s")
    print(f"Speedup over original: {blur_time/blur_vec_time:.2f}x")


if __name__ == "__main__":
    profile_noise_methods()