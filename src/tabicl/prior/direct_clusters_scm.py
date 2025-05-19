"""Direct cluster-based SCM that creates pre-labeled clusters without transformation."""

import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional

from .utils import XSampler


class DirectClustersSCM(nn.Module):
    """SCM that directly generates labeled clusters without continuous transformation."""
    
    def __init__(self,
                 seq_len: int = 1024,
                 num_features: int = 100,
                 num_outputs: int = 1,
                 hyperparams: dict = None,
                 num_classes: int = 10,
                 cluster_separation: float = 3.0,
                 within_cluster_std: float = 0.2,
                 random_state: int = 42,
                 device: str = "cpu",
                 **kwargs):
        super(DirectClustersSCM, self).__init__()
        
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
        self.cluster_centers = self._create_cluster_centers()
        
    def _create_cluster_centers(self):
        """Create well-separated cluster centers."""
        centers = []
        
        for i in range(self.num_classes):
            if i == 0:
                # First center at origin
                center = np.zeros(self.num_features)
            else:
                # Each subsequent center is maximally separated
                # Create orthogonal unit vectors
                center = np.zeros(self.num_features)
                if i <= self.num_features:
                    center[i-1] = self.cluster_separation
                else:
                    # When we have more classes than features, create combinations
                    # Use multiple dimensions
                    dims = []
                    remaining = i - 1
                    dim_idx = 0
                    while remaining > 0 and dim_idx < self.num_features:
                        dims.append(dim_idx)
                        remaining -= 1
                        dim_idx += 1
                    
                    for d in dims:
                        center[d] = self.cluster_separation / np.sqrt(len(dims))
            
            centers.append(center)
        
        return np.array(centers)
    
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
        """Generate synthetic data with direct clusters."""
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
        
        # For test compatibility - return continuous labels directly
        if info is not None:
            # Return the actual class labels as continuous values
            # No transformation needed - let the test handle it
            return {'y_cont': y_tensor}
        
        return X_tensor, y_tensor
    
    def __call__(self, X=None, info=None):
        """Make the class callable."""
        return self.forward(X, info)