"""Optimized polynomial transform for DeterministicTreeSCM."""

import numpy as np
import torch
from numba import njit, prange


@njit(parallel=True)
def compute_polynomial_features(X, feat_indices, out_dim):
    """Numba-optimized polynomial feature computation."""
    n_samples = X.shape[0]
    y = np.zeros((n_samples, out_dim))
    
    for j in prange(out_dim):
        indices = feat_indices[j]
        n_features = len(indices)
        
        if n_features > 0:
            # Squared terms
            for i in range(n_samples):
                squared_sum = 0.0
                for k in range(n_features):
                    squared_sum += X[i, indices[k]] ** 2
                y[i, j] = squared_sum
            
            # Cross terms
            if n_features > 1:
                for i in range(n_samples):
                    y[i, j] += X[i, indices[0]] * X[i, indices[1]]
    
    return y


def optimized_polynomial_transform(X_np, weights, out_dim):
    """Optimized polynomial transformation using vectorization and numba."""
    n_samples, n_features = X_np.shape
    
    # Convert feat_indices to numpy array for numba
    feat_indices_array = np.zeros((out_dim, 3), dtype=np.int32)  # Max 3 features per output
    for j in range(out_dim):
        indices = weights['feat_indices'][j]
        for k in range(min(len(indices), 3)):
            feat_indices_array[j, k] = indices[k]
    
    # Use numba-compiled function
    y = compute_polynomial_features(X_np, feat_indices_array, out_dim)
    
    # Normalize to prevent extreme values
    y_mean = np.mean(y, axis=0, keepdims=True)
    y_std = np.std(y, axis=0, keepdims=True) + 1e-8
    y = (y - y_mean) / y_std
    
    return y


def vectorized_polynomial_transform(X_np, weights, out_dim):
    """Fully vectorized polynomial transformation without loops."""
    n_samples, n_features = X_np.shape
    
    # Pre-allocate output
    y = np.zeros((n_samples, out_dim))
    
    # Group operations by feature index patterns
    feat_indices_list = weights['feat_indices']
    
    # Process all squared terms at once
    for j in range(out_dim):
        indices = feat_indices_list[j]
        if len(indices) > 0:
            # Squared terms - fully vectorized
            X_selected = X_np[:, indices]
            y[:, j] = np.sum(X_selected ** 2, axis=1)
            
            # Cross terms - also vectorized
            if len(indices) > 1:
                y[:, j] += X_selected[:, 0] * X_selected[:, 1]
    
    # Normalize all at once
    y_mean = np.mean(y, axis=0, keepdims=True)
    y_std = np.std(y, axis=0, keepdims=True) + 1e-8
    y = (y - y_mean) / y_std
    
    return y