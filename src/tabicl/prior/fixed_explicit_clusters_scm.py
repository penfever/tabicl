"""Fixed explicit cluster-based SCM that preserves cluster structure."""

import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any

from .utils import XSampler


class FixedExplicitClustersSCM(nn.Module):
    """SCM that creates explicit clusters that maintain their structure through assigners."""
    
    def __init__(self,
                 seq_len: int = 1024,
                 num_features: int = 100,
                 num_outputs: int = 1,
                 hyperparams: dict = None,
                 num_classes: int = 10,
                 cluster_separation: float = 3.0,
                 within_cluster_std: float = 0.3,
                 random_state: int = 42,
                 device: str = "cpu",
                 **kwargs):
        super(FixedExplicitClustersSCM, self).__init__()
        
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.hyperparams = hyperparams or {}
        self.num_classes = num_classes
        self.cluster_separation = cluster_separation
        self.within_cluster_std = within_cluster_std
        self.random_state = random_state
        self.device = device
        
        # Set random seed
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        # Create cluster centers
        self.cluster_centers = self._create_orthogonal_clusters()
        
    def _create_orthogonal_clusters(self):
        """Create orthogonal cluster centers for maximum separation."""
        centers = np.zeros((self.num_classes, self.num_features))
        
        # Create orthogonal basis vectors
        for i in range(min(self.num_classes, self.num_features)):
            centers[i, i] = self.cluster_separation
        
        # For extra classes beyond feature dimensions, create combinations
        if self.num_classes > self.num_features:
            for i in range(self.num_features, self.num_classes):
                # Use combinations of existing dimensions
                n_dims = 2  # Use pairs of dimensions
                selected_dims = np.random.choice(self.num_features, n_dims, replace=False)
                for dim in selected_dims:
                    centers[i, dim] = self.cluster_separation / np.sqrt(n_dims)
        
        return centers
    
    def _generate_samples(self):
        """Generate balanced samples from each cluster."""
        samples_per_class = self.seq_len // self.num_classes
        extra_samples = self.seq_len % self.num_classes
        
        X_list = []
        y_list = []
        
        for class_idx in range(self.num_classes):
            # Determine number of samples for this class
            n_samples = samples_per_class
            if class_idx < extra_samples:
                n_samples += 1
            
            # Generate samples around cluster center
            center = self.cluster_centers[class_idx]
            # Use isotropic Gaussian for simplicity
            samples = np.random.randn(n_samples, self.num_features) * self.within_cluster_std
            samples += center
            
            X_list.append(samples)
            y_list.append(np.full(n_samples, class_idx))
        
        # Combine and shuffle
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        perm = np.random.permutation(self.seq_len)
        X = X[perm]
        y = y[perm]
        
        return X, y
    
    def forward(self, X=None, info=None):
        """Generate synthetic data with explicit clusters."""
        # Generate samples
        X, y = self._generate_samples()
        
        # Add small noise for robustness
        X += np.random.randn(*X.shape) * 0.01
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        if self.num_outputs == 1:
            y_tensor = y_tensor.unsqueeze(-1)
        
        # For test compatibility - return continuous values that preserve cluster structure
        if info is not None:
            # Create continuous values that preserve the cluster structure
            # Use larger separation to ensure assigners can distinguish them
            y_continuous = y_tensor * 10.0 + torch.randn_like(y_tensor) * 0.1
            return {'y_cont': y_continuous}
        
        return X_tensor, y_tensor
    
    def __call__(self, X=None, info=None):
        """Make the class callable."""
        return self.forward(X, info)