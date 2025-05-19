"""GMM-based SCM that uses Gaussian Mixture Models for optimal cluster generation."""

import numpy as np
import torch
from torch import nn
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, Any, Tuple

from .utils import XSampler


class GMMClustersSCM(nn.Module):
    """SCM that uses GMM to create well-separated, balanced clusters."""
    
    def __init__(self,
                 seq_len: int = 1024,
                 num_features: int = 100,
                 num_outputs: int = 1,
                 hyperparams: dict = None,
                 num_classes: int = 10,
                 separation_strength: float = 10.0,
                 balance_strength: float = 0.9,
                 use_pca: bool = True,
                 pca_components: int = 50,
                 random_state: int = 42,
                 device: str = "cpu",
                 **kwargs):
        super(GMMClustersSCM, self).__init__()
        
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.hyperparams = hyperparams or {}
        self.num_classes = kwargs.get('num_classes', num_classes)
        self.separation_strength = kwargs.get('separation_strength', separation_strength)
        self.balance_strength = kwargs.get('balance_strength', balance_strength)
        self.use_pca = use_pca
        self.pca_components = min(pca_components, num_features)
        self.random_state = random_state
        self.device = device
        
        # Set random seed
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        # Initialize GMM parameters
        self._initialize_gmm_parameters()
    
    def _initialize_gmm_parameters(self):
        """Initialize GMM parameters for well-separated clusters."""
        # Create means on a hypersphere for optimal separation
        self.means = np.zeros((self.num_classes, self.num_features))
        
        # Create well-spaced samples in [0, 1]^d
        d = min(self.num_features, 3)
        samples = np.zeros((self.num_classes, d))
        
        # Simple grid-based spacing
        if self.num_classes <= 8:
            # For small number of classes, use corners of hypercube
            for i in range(self.num_classes):
                binary = format(i, f'0{d}b')[-d:]
                samples[i] = [int(b) for b in binary]
        else:
            # For larger number, use random spacing
            samples = np.random.rand(self.num_classes, d)
        
        # Scale and center the samples
        samples = (samples - 0.5) * 2 * self.separation_strength
        
        # Assign to means
        for i in range(self.num_classes):
            if self.num_features >= 3:
                self.means[i, :3] = samples[i]
                # Small random values for remaining dimensions
                if self.num_features > 3:
                    self.means[i, 3:] = np.random.randn(self.num_features - 3) * 0.1
            else:
                self.means[i, :self.num_features] = samples[i, :self.num_features]
        
        # Create covariance matrices (spherical with slight elongation)
        self.covariances = []
        
        # Adjust covariance based on separation strength - less separation = more variance
        base_variance = 0.5 * (10.0 / max(self.separation_strength, 1.0))
        
        for i in range(self.num_classes):
            # Start with identity
            cov = np.eye(self.num_features) * base_variance
            
            # Add some random structure
            if self.num_features > 1:
                # Create random rotation
                theta = np.random.rand() * 2 * np.pi
                rotation = np.eye(self.num_features)
                rotation[0, 0] = np.cos(theta)
                rotation[0, 1] = -np.sin(theta)
                rotation[1, 0] = np.sin(theta)
                rotation[1, 1] = np.cos(theta)
                
                # Apply slight elongation
                scale = np.eye(self.num_features)
                scale[0, 0] = 1.5  # Elongate in one direction
                
                cov = rotation @ scale @ cov @ scale.T @ rotation.T
            
            self.covariances.append(cov)
        
        # Equal weights for balance
        self.weights = np.ones(self.num_classes) / self.num_classes
    
    def _apply_balance_constraint(self, assignments):
        """Apply balancing to ensure roughly equal class sizes."""
        n_samples = len(assignments)
        ideal_size = n_samples // self.num_classes
        
        # Count current assignments
        counts = np.bincount(assignments, minlength=self.num_classes)
        
        # Identify over and under-represented classes
        over_represented = np.where(counts > ideal_size * 1.2)[0]
        under_represented = np.where(counts < ideal_size * 0.8)[0]
        
        if len(over_represented) > 0 and len(under_represented) > 0:
            # Reassign some samples from over to under-represented classes
            for over_class in over_represented:
                # Get indices of samples in this class
                indices = np.where(assignments == over_class)[0]
                
                # Number to reassign
                n_reassign = min(
                    len(indices) // 4,  # Don't reassign too many
                    counts[over_class] - ideal_size
                )
                
                if n_reassign > 0:
                    # Choose samples to reassign (preferably boundary samples)
                    reassign_indices = np.random.choice(indices, n_reassign, replace=False)
                    
                    # Assign to under-represented classes
                    for idx in reassign_indices:
                        target_class = np.random.choice(under_represented)
                        assignments[idx] = target_class
                        counts[over_class] -= 1
                        counts[target_class] += 1
                        
                        # Update under-represented list
                        if counts[target_class] >= ideal_size * 0.8:
                            under_represented = under_represented[under_represented != target_class]
                            if len(under_represented) == 0:
                                break
        
        return assignments
    
    def _generate_samples(self):
        """Generate samples from the GMM."""
        X_list = []
        y_list = []
        
        # Calculate samples per class
        base_samples = self.seq_len // self.num_classes
        remainder = self.seq_len % self.num_classes
        
        for i in range(self.num_classes):
            # Determine number of samples for this class
            n_samples = base_samples + (1 if i < remainder else 0)
            
            # Sample from multivariate normal
            samples = np.random.multivariate_normal(
                mean=self.means[i],
                cov=self.covariances[i],
                size=n_samples
            )
            
            X_list.append(samples)
            y_list.append(np.full(n_samples, i))
        
        # Combine and shuffle
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        # Shuffle
        perm = np.random.permutation(self.seq_len)
        X = X[perm]
        y = y[perm]
        
        return X, y
    
    def _add_complexity(self, X, y):
        """Add complexity to the features while preserving cluster structure."""
        n_samples, n_features = X.shape
        
        # Add polynomial interactions
        if n_features >= 2:
            # Select random pairs of features
            n_interactions = min(20, n_features)
            for _ in range(n_interactions):
                i, j = np.random.choice(n_features, 2, replace=False)
                interaction = X[:, i] * X[:, j]
                X = np.column_stack([X, interaction])
        
        # Add RBF transformations based on class centers
        n_rbf = min(10, self.num_classes)
        for i in range(n_rbf):
            center = self.means[i][:n_features]
            distances = np.sum((X[:, :n_features] - center) ** 2, axis=1)
            rbf_feature = np.exp(-distances / (2 * self.separation_strength))
            X = np.column_stack([X, rbf_feature])
        
        # Apply PCA if requested
        if self.use_pca and X.shape[1] > self.pca_components:
            pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            X = pca.fit_transform(X)
        
        # Ensure we have exactly num_features
        if X.shape[1] > self.num_features:
            # Project to num_features dimensions
            projection = np.random.randn(X.shape[1], self.num_features)
            projection = np.linalg.qr(projection)[0]  # Orthogonalize
            X = X @ projection
        elif X.shape[1] < self.num_features:
            # Pad with correlated noise
            n_pad = self.num_features - X.shape[1]
            padding = np.random.randn(n_samples, n_pad) * 0.1
            # Make padding slightly correlated with existing features
            for i in range(n_pad):
                if i < X.shape[1]:
                    padding[:, i] += 0.3 * X[:, i % X.shape[1]]
            X = np.column_stack([X, padding])
        
        # Final normalization
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Scale back up for better separation
        X *= 2.0
        
        return X
    
    def forward(self, X=None, info=None):
        """Generate synthetic data with GMM-based clusters."""
        # Generate samples from GMM
        X, y = self._generate_samples()
        
        # Add complexity
        X = self._add_complexity(X, y)
        
        # Apply balance constraint if needed
        if self.balance_strength > np.random.rand():
            y = self._apply_balance_constraint(y)
        
        # Add small noise
        X += np.random.randn(*X.shape) * 0.1
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        if self.num_outputs == 1:
            y_tensor = y_tensor.unsqueeze(-1)
        
        # Return in expected format
        if info is not None:
            # Create continuous values with clear separation
            y_continuous = y_tensor * 10.0 + torch.randn_like(y_tensor) * 0.3
            # Add class-specific offsets to ensure separation
            for i in range(self.num_classes):
                mask = (y_tensor.squeeze() == i)
                y_continuous[mask] += i * 0.5
            return {'y_cont': y_continuous}
        
        return X_tensor, y_tensor