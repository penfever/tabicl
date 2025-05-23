from __future__ import annotations

import random
import numpy as np
import torch
from torch import nn
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from typing import Optional, Tuple, Dict, Any, Union, List, Callable
import time

from .utils import GaussianNoise, XSampler, apply_class_separability


class DeterministicTreeLayer(nn.Module):
    """Optimized version of DeterministicTreeLayer with performance improvements."""
    
    def __init__(self, 
                 tree_model: str, 
                 max_depth: int, 
                 n_estimators: int, 
                 out_dim: int,
                 swap_prob: float = 0.1,
                 transform_type: str = "polynomial",
                 device: str = "cpu",
                 noise_type: str = "swap",
                 n_jobs: int = 4,  # Control parallelism explicitly
                 class_separability: float = 1.0,
                 hyperparams: dict = None):
        super(DeterministicTreeLayer, self).__init__()
        self.out_dim = out_dim
        self.swap_prob = swap_prob
        self.transform_type = transform_type
        self.device = device
        self.noise_type = noise_type
        self.n_jobs = n_jobs  # Control parallelism to avoid excessive overhead
        self.class_separability = class_separability
        self.hyperparams = hyperparams or {}
        
        # Cache for deterministic transformations
        self._transform_cache = {}
        
        # Adaptive complexity based on swap probability
        if swap_prob > 0.15:
            # Use simpler model for more noisy data
            self.max_depth = min(max_depth, 3)
            self.n_estimators = min(n_estimators, 50)
        else:
            self.max_depth = max_depth
            self.n_estimators = n_estimators
        
        # Efficient model initialization with optimized parameters
        if tree_model == "decision_tree":
            base_model = DecisionTreeRegressor(
                max_depth=self.max_depth,
                # Add presort=False for modern sklearn versions
                splitter="best",  # "random" is faster but less accurate
            )
            self.model = MultiOutputRegressor(base_model, n_jobs=self.n_jobs)
        elif tree_model == "extra_trees":
            base_model = ExtraTreesRegressor(
                n_estimators=self.n_estimators, 
                max_depth=self.max_depth,
                bootstrap=False,  # Faster without bootstrap
                n_jobs=1,  # Let MultiOutputRegressor handle parallelism
                criterion="squared_error",
                warm_start=True  # Reuse previous tree structure when possible
            )
            self.model = MultiOutputRegressor(base_model, n_jobs=self.n_jobs)
        elif tree_model == "random_forest":
            base_model = RandomForestRegressor(
                n_estimators=self.n_estimators, 
                max_depth=self.max_depth,
                bootstrap=True,
                n_jobs=1,  # Let MultiOutputRegressor handle parallelism
                criterion="squared_error", 
                warm_start=True  # Reuse previous tree structure when possible
            )
            self.model = MultiOutputRegressor(base_model, n_jobs=self.n_jobs)
        elif tree_model == "xgboost":
            # Adjust parameters based on noise level
            if self.swap_prob > 0.3:
                # For high noise, use more aggressive regularization and simpler models
                self.model = XGBRegressor(
                    n_estimators=min(self.n_estimators, 30),
                    max_depth=min(self.max_depth, 2),
                    tree_method="hist",
                    multi_strategy="multi_output_tree",
                    n_jobs=self.n_jobs,
                    learning_rate=0.5,
                    verbosity=0,
                    subsample=0.6,
                    colsample_bytree=0.6,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    max_bin=128,  # Reduce histogram bins for speed
                    predictor="cpu_predictor",
                )
            else:
                self.model = XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    tree_method="hist",
                    multi_strategy="multi_output_tree",
                    n_jobs=self.n_jobs,
                    learning_rate=0.3,
                    verbosity=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.01,
                    reg_lambda=0.01,
                    max_bin=256,
                    predictor="cpu_predictor",
                )
        else:
            raise ValueError(f"Invalid tree model: {tree_model}")
        
        # Pre-compute transformation weights for polynomial transformations
        self._precompute_transform_weights()
    
    def _create_well_separated_centers(self, num_centers, n_features):
        """Create cluster centers that are well-separated in feature space."""
        centers = np.zeros((num_centers, n_features))
        
        # Create centers on a hypersphere with good separation
        for i in range(num_centers):
            # Use spherical coordinates for better separation
            angles = np.random.uniform(0, 2*np.pi, size=n_features-1)
            radius = 3.0 + np.random.uniform(-0.5, 0.5)  # Some variation in radius
            
            # Convert to Cartesian coordinates
            center = np.zeros(n_features)
            center[0] = radius * np.prod(np.cos(angles))
            for j in range(1, n_features):
                if j < n_features - 1:
                    center[j] = radius * np.sin(angles[j-1]) * np.prod(np.cos(angles[j:]))
                else:
                    center[j] = radius * np.sin(angles[j-1])
            
            # Add some noise for variety
            center += np.random.randn(n_features) * 0.2
            centers[i] = center
            
        return centers
    
    def _create_balanced_weights(self, num_centers, out_dim):
        """Create weights that promote balanced activation across centers."""
        # Use orthogonal initialization for better distribution
        weights = np.random.randn(num_centers, out_dim)
        U, _, Vt = np.linalg.svd(weights, full_matrices=False)
        weights = U @ Vt
        return weights * 2.0  # Scale up for better separation
    
    def _create_multi_modal_centers(self, num_clusters, n_features):
        """Create centers for multi-modal distribution with better separation."""
        centers = np.zeros((num_clusters, n_features))
        
        # Create centers in a regular simplex for maximum separation
        if n_features >= num_clusters - 1:
            # Use simplex vertices for first dimensions
            for i in range(num_clusters):
                for j in range(min(num_clusters - 1, n_features)):
                    if i == j:
                        centers[i, j] = np.sqrt(2 * (j + 1) * j / ((j + 1) * (j + 2))) * 5.0
                    elif i == j + 1:
                        centers[i, j] = -1.0 / np.sqrt((j + 1) * (j + 2)) * 5.0
                    elif i > j + 1:
                        centers[i, j] = 0
                # Add random variation in remaining dimensions
                if n_features > num_clusters - 1:
                    centers[i, num_clusters-1:] = np.random.randn(n_features - num_clusters + 1) * 0.5
        else:
            # Fall back to circular arrangement if not enough dimensions
            for i in range(num_clusters):
                angle = 2 * np.pi * i / num_clusters
                centers[i, 0] = np.cos(angle) * 5.0
                if n_features > 1:
                    centers[i, 1] = np.sin(angle) * 5.0
                if n_features > 2:
                    centers[i, 2:] = np.random.randn(n_features - 2) * 0.5
        
        return centers
    
    def _create_balanced_cluster_centers(self, num_clusters, n_features):
        """Create cluster centers optimized for balanced class distribution."""
        # Use k-means++ style initialization for better spread
        centers = np.zeros((num_clusters, n_features))
        
        # First center is random
        centers[0] = np.random.randn(n_features) * 2.0
        
        # Each subsequent center is chosen to be far from existing centers
        for i in range(1, num_clusters):
            # Calculate distances to all existing centers
            distances = np.zeros(1000)  # Sample many points
            for j in range(1000):
                candidate = np.random.randn(n_features) * 2.0
                min_dist = float('inf')
                for k in range(i):
                    dist = np.linalg.norm(candidate - centers[k])
                    min_dist = min(min_dist, dist)
                distances[j] = min_dist
            
            # Choose point with maximum minimum distance
            best_idx = np.argmax(distances)
            np.random.seed(best_idx)  # Reproducible
            centers[i] = np.random.randn(n_features) * 2.0
        
        return centers * 3.0  # Scale up for better separation
    
    def _create_mixture_components(self, num_components, n_features):
        """Create diverse mixture components with enhanced separability."""
        components = []
        
        # Use different strategies for different components
        for i in range(num_components):
            if i % 3 == 0:
                # RBF component with distinct center
                center = np.zeros(n_features)
                # Place centers on a hypersphere for better separation
                angle = 2 * np.pi * i / num_components
                center[0] = np.cos(angle) * 5.0
                center[1 % n_features] = np.sin(angle) * 5.0
                # Add random variation in other dimensions
                if n_features > 2:
                    center[2:] = np.random.randn(n_features - 2) * 0.5
                
                component = {
                    'type': 'rbf',
                    'center': center,
                    'scale': 1.0,
                    'gamma': 0.3  # Consistent gamma for better control
                }
            elif i % 3 == 1:
                # Polynomial component with offset
                component = {
                    'type': 'polynomial',
                    'center': np.random.randn(n_features) * 2.0,
                    'scale': 0.5,
                    'gamma': 0.5,
                    'degree': 2
                }
            else:
                # Trigonometric component for variety
                component = {
                    'type': 'trigonometric',
                    'center': np.random.randn(n_features) * 1.5,
                    'scale': 1.0,
                    'gamma': 0.8,
                    'frequency': np.pi * (i // 3 + 1)
                }
            
            components.append(component)
        
        return components
    
    def _create_optimal_centers(self, num_centers, n_features):
        """Create optimally separated centers using quasi-random sequences."""
        centers = np.zeros((num_centers, n_features))
        
        if n_features >= 3:
            # Use 3D sphere packing for first 3 dimensions
            for i in range(num_centers):
                # Fibonacci sphere algorithm for uniform distribution
                golden_angle = np.pi * (3.0 - np.sqrt(5.0))
                theta = golden_angle * i
                z = 1.0 - (i / float(num_centers - 1)) * 2.0
                radius = np.sqrt(1.0 - z * z)
                
                centers[i, 0] = radius * np.cos(theta) * 5.0
                centers[i, 1] = radius * np.sin(theta) * 5.0
                centers[i, 2] = z * 5.0
                
                # Random for remaining dimensions
                if n_features > 3:
                    centers[i, 3:] = np.random.randn(n_features - 3) * 0.5
        else:
            # For low dimensions, use regular grid
            for i in range(num_centers):
                angle = 2 * np.pi * i / num_centers
                centers[i, 0] = np.cos(angle) * 5.0
                if n_features > 1:
                    centers[i, 1] = np.sin(angle) * 5.0
        
        return centers
    
    def _create_enhanced_mixture_components(self, num_components, n_features):
        """Create enhanced mixture components with optimal properties."""
        centers = self._create_optimal_centers(num_components, n_features)
        components = []
        
        for i in range(num_components):
            # All components are RBF for consistency and control
            component = {
                'type': 'rbf',
                'center': centers[i],
                'scale': 1.0,
                'gamma': 0.2,  # Tighter clusters
                'weight': 1.0 / num_components  # Equal weights
            }
            components.append(component)
        
        return components
    
    def _precompute_transform_weights(self):
        """Pre-compute weights for deterministic transformations to avoid redundant computations."""
        self._transform_weights = {}
        
        # We don't know the feature dimension yet, so we'll create a factory function
        # that will generate weights when needed
        if self.transform_type == "polynomial":
            self._transform_weights_factory = lambda n_features: {
                'feat_indices': [
                    np.random.choice(n_features, size=min(5, n_features), replace=False)
                    for _ in range(self.out_dim)
                ]
            }
        elif self.transform_type == "rbf":
            # Create well-separated cluster centers
            num_rbf_centers = self.hyperparams.get('num_classes', 5) * 2  # More centers than classes
            self._transform_weights_factory = lambda n_features: {
                'centers': self._create_well_separated_centers(num_rbf_centers, n_features),
                'gamma': 0.25 / (self.class_separability if hasattr(self, 'class_separability') else 1.0),
                'weights': self._create_balanced_weights(num_rbf_centers, self.out_dim),
                'normalize': True
            }
        elif self.transform_type == "multi_modal":
            # Create explicit cluster centers matching desired number of classes
            num_clusters = self.hyperparams.get('num_classes', 5)
            self._transform_weights_factory = lambda n_features: {
                'centers': self._create_multi_modal_centers(num_clusters, n_features),
                'gamma': 0.5 / (self.class_separability if hasattr(self, 'class_separability') else 1.0),
                'weights': self._create_balanced_weights(num_clusters, self.out_dim),
                'cluster_probs': np.ones(num_clusters) / num_clusters,
                'normalize': True
            }
        elif self.transform_type == "trigonometric":
            self._transform_weights_factory = lambda n_features: {
                'feat_indices': [
                    np.random.choice(n_features, size=min(3, n_features), replace=False)
                    for _ in range(self.out_dim)
                ]
            }
        elif self.transform_type == "exponential":
            self._transform_weights_factory = lambda n_features: {
                'feat_indices': [
                    np.random.choice(n_features, size=min(2, n_features), replace=False)
                    for _ in range(self.out_dim)
                ]
            }
        elif self.transform_type == "mixed":
            self._transform_weights_factory = lambda n_features: {
                'transforms': np.random.choice(["polynomial", "trigonometric", "exponential"], size=self.out_dim),
                'feat_indices': [
                    np.random.choice(n_features, size=min(3, n_features), replace=False)
                    for _ in range(self.out_dim)
                ]
            }
        elif self.transform_type == "mixture":
            # Mixture of different transformations to create natural clusters
            num_components = self.hyperparams.get('num_classes', 5)
            self._transform_weights_factory = lambda n_features: {
                'component_weights': self._create_mixture_components(num_components, n_features),
                'mixture_weights': np.ones(num_components) / num_components,  # Equal weights for balance
                'separation_factor': 5.0,  # Increase separation between components
                'normalize': True
            }
        elif self.transform_type == "balanced_clusters":
            # Specifically designed for balanced class generation
            num_clusters = self.hyperparams.get('num_classes', 5)
            self._transform_weights_factory = lambda n_features: {
                'centers': self._create_balanced_cluster_centers(num_clusters, n_features),
                'gamma': 0.3,  # Fixed gamma for consistent cluster sizes
                'weights': np.eye(num_clusters, self.out_dim),  # Direct mapping to clusters
                'balance_factor': 2.0,
                'normalize': True
            }
        elif self.transform_type == "enhanced_mixture":
            # Enhanced mixture with better separability and balance
            num_clusters = self.hyperparams.get('num_classes', 5)
            self._transform_weights_factory = lambda n_features: {
                'centers': self._create_optimal_centers(num_clusters, n_features),
                'component_weights': self._create_enhanced_mixture_components(num_clusters, n_features),
                'mixture_weights': np.ones(num_clusters) / num_clusters,
                'gamma': 0.2,  # Tighter clusters
                'separation_factor': 8.0,  # Stronger separation
                'repulsion_factor': 2.0,  # Add repulsion between clusters
                'normalize': True
            }
        else:
            # Default to linear combination
            self._transform_weights_factory = lambda n_features: {
                'weights': np.random.randn(n_features, self.out_dim)
            }
    
    def _get_transform_weights(self, n_features):
        """Get cached or create new transform weights for a given feature dimension."""
        if n_features not in self._transform_weights:
            self._transform_weights[n_features] = self._transform_weights_factory(n_features)
        return self._transform_weights[n_features]
    
    def _generate_deterministic_targets(self, X: torch.Tensor) -> np.ndarray:
        """Generate deterministic targets based on input features with performance optimizations."""
        X_np = X.cpu().detach().numpy()  # Detach to reduce memory usage
        n_samples, n_features = X_np.shape
        
        # Get cached or create new weights
        weights = self._get_transform_weights(n_features)
        
        if self.transform_type == "polynomial":
            # Optimized polynomial transformation
            y = np.zeros((n_samples, self.out_dim))
            feat_indices_list = weights['feat_indices']
            
            # Batch process all features for better performance
            all_feature_indices = []
            for indices in feat_indices_list:
                all_feature_indices.extend(indices)
            
            if all_feature_indices:
                # Get unique indices and pre-compute squares
                unique_indices = sorted(set(all_feature_indices))
                X_selected = X_np[:, unique_indices]
                X_squared = X_selected ** 2
                
                # Create mapping for fast lookup
                index_map = {idx: i for i, idx in enumerate(unique_indices)}
                
                # Process each output dimension using pre-computed values
                for j in range(self.out_dim):
                    feat_indices = feat_indices_list[j]
                    if feat_indices:
                        # Map to pre-computed indices
                        mapped_indices = [index_map[idx] for idx in feat_indices]
                        
                        # Sum squared terms
                        y[:, j] = np.sum(X_squared[:, mapped_indices], axis=1)
                        
                        # Add cross terms if applicable
                        if len(feat_indices) > 1:
                            y[:, j] += X_np[:, feat_indices[0]] * X_np[:, feat_indices[1]]
            
            # Normalize to prevent extreme values
            y_mean = np.mean(y, axis=0, keepdims=True)
            y_std = np.std(y, axis=0, keepdims=True) + 1e-8
            y = (y - y_mean) / y_std
                    
        elif self.transform_type == "rbf":
            # Radial Basis Function transformation
            centers = weights['centers']
            gamma = weights['gamma']
            rbf_weights = weights['weights']
            
            # Compute RBF activations: exp(-gamma * ||x - center||^2)
            # Using broadcasting for efficiency
            X_expanded = X_np[:, np.newaxis, :]  # (n_samples, 1, n_features)
            centers_expanded = centers[np.newaxis, :, :]  # (1, n_centers, n_features)
            distances_squared = np.sum((X_expanded - centers_expanded) ** 2, axis=2)  # (n_samples, n_centers)
            
            # Apply RBF kernel
            rbf_activations = np.exp(-gamma * distances_squared)  # (n_samples, n_centers)
            
            # Combine RBF activations with weights to produce outputs
            y = rbf_activations @ rbf_weights  # (n_samples, out_dim)
            
            # Normalize outputs if specified
            if weights.get('normalize', False):
                y_mean = np.mean(y, axis=0, keepdims=True)
                y_std = np.std(y, axis=0, keepdims=True) + 1e-8
                y = (y - y_mean) / y_std
            
        elif self.transform_type == "multi_modal":
            # Generate multi-modal outputs for better natural clustering
            centers = weights['centers']
            gamma = weights['gamma'] 
            weights_matrix = weights['weights']
            
            # Calculate distances to all centers
            X_expanded = X_np[:, np.newaxis, :]
            centers_expanded = centers[np.newaxis, :, :]
            distances_squared = np.sum((X_expanded - centers_expanded) ** 2, axis=2)
            
            # Apply RBF kernel with cluster-specific scaling
            rbf_activations = np.exp(-gamma * distances_squared)
            
            # Weight the activations to create distinct modes
            y = rbf_activations @ weights_matrix
            
            # Add some between-cluster separation
            if self.hyperparams.get('add_cluster_separation', True):
                num_clusters = self.hyperparams.get('num_classes', 5)
                cluster_assignments = np.argmax(rbf_activations[:, :num_clusters], axis=1)
                for i in range(num_clusters):
                    mask = cluster_assignments == i
                    if np.any(mask):
                        y[mask] += i * 2.0  # Smaller offset after normalization
            
            # Normalize outputs if specified
            if weights.get('normalize', False):
                y_mean = np.mean(y, axis=0, keepdims=True)
                y_std = np.std(y, axis=0, keepdims=True) + 1e-8
                y = (y - y_mean) / y_std
                    
        elif self.transform_type == "trigonometric":
            # Vectorized trigonometric transformation
            y = np.zeros((n_samples, self.out_dim))
            for j in range(self.out_dim):
                feat_indices = weights['feat_indices'][j]
                y[:, j] = np.sin(X_np[:, feat_indices[0]] * np.pi)
                if len(feat_indices) > 1:
                    y[:, j] += np.cos(X_np[:, feat_indices[1]] * np.pi / 2)
                    
        elif self.transform_type == "exponential":
            # Vectorized exponential transformation
            y = np.zeros((n_samples, self.out_dim))
            for j in range(self.out_dim):
                feat_indices = weights['feat_indices'][j]
                y[:, j] = np.exp(np.clip(X_np[:, feat_indices[0]], -5, 5))
                
        elif self.transform_type == "mixed":
            # Vectorized mixed transformation
            y = np.zeros((n_samples, self.out_dim))
            transforms = weights['transforms']
            
            # Process each transform type in batches for vectorization
            poly_indices = np.where(transforms == "polynomial")[0]
            if len(poly_indices) > 0:
                for j in poly_indices:
                    feat_indices = weights['feat_indices'][j]
                    y[:, j] = np.sum(X_np[:, feat_indices] ** 2, axis=1)
            
            trig_indices = np.where(transforms == "trigonometric")[0]
            if len(trig_indices) > 0:
                for j in trig_indices:
                    feat_indices = weights['feat_indices'][j]
                    y[:, j] = np.sin(X_np[:, feat_indices[0]] * np.pi)
            
            exp_indices = np.where(transforms == "exponential")[0]
            if len(exp_indices) > 0:
                for j in exp_indices:
                    feat_indices = weights['feat_indices'][j]
                    y[:, j] = np.exp(np.clip(X_np[:, feat_indices[0]], -5, 5))
                    
        elif self.transform_type == "mixture":
            # Mixture of transformations to create natural clusters
            component_weights = weights['component_weights']
            mixture_weights = weights['mixture_weights']
            
            y = np.zeros((n_samples, self.out_dim))
            component_outputs = []
            
            for i, comp in enumerate(component_weights):
                comp_output = np.zeros((n_samples, self.out_dim))
                
                if comp['type'] == 'rbf':
                    # RBF around component center
                    center = comp['center']
                    gamma = comp['gamma']
                    distances_squared = np.sum((X_np - center) ** 2, axis=1)
                    comp_output[:, :] = np.exp(-gamma * distances_squared)[:, np.newaxis] * comp['scale']
                    
                elif comp['type'] == 'polynomial':
                    # Polynomial centered at component
                    shifted_X = X_np - comp['center']
                    comp_output[:, :] = np.sum(shifted_X ** 2, axis=1)[:, np.newaxis] * comp['scale']
                    
                elif comp['type'] == 'trigonometric':
                    # Trigonometric with phase shift
                    shifted_X = X_np - comp['center']
                    frequency = comp.get('frequency', np.pi)
                    comp_output[:, :] = np.sin(np.sum(shifted_X, axis=1) * frequency * comp['scale'])[:, np.newaxis]
                
                component_outputs.append(comp_output * mixture_weights[i])
            
            # Combine all components
            y = np.sum(component_outputs, axis=0)
            
            # Add separation between mixture components
            if self.hyperparams.get('add_component_separation', True):
                # Assign each sample to closest component
                distances_to_centers = np.array([
                    np.sum((X_np - comp['center']) ** 2, axis=1) 
                    for comp in component_weights
                ])
                component_assignments = np.argmin(distances_to_centers, axis=0)
                
                # Add stronger component-specific offset with separation factor
                separation_factor = weights.get('separation_factor', 3.0)
                for i in range(len(component_weights)):
                    mask = component_assignments == i
                    if np.any(mask):
                        y[mask] += i * separation_factor
            
            # Normalize to prevent extreme values
            y_mean = np.mean(y, axis=0, keepdims=True)
            y_std = np.std(y, axis=0, keepdims=True) + 1e-8
            y = (y - y_mean) / y_std
                        
        elif self.transform_type == "balanced_clusters":
            # Balanced cluster generation
            centers = weights['centers']
            gamma = weights['gamma']
            balance_factor = weights['balance_factor']
            
            # Calculate distances to all centers
            distances = np.zeros((n_samples, len(centers)))
            for i, center in enumerate(centers):
                distances[:, i] = np.sum((X_np - center) ** 2, axis=1)
            
            # Apply RBF kernel
            rbf_activations = np.exp(-gamma * distances)
            
            # Apply balancing factor to encourage equal cluster sizes
            cluster_assignments = np.argmax(rbf_activations, axis=1)
            cluster_counts = np.bincount(cluster_assignments, minlength=len(centers))
            
            # Adjust activations based on current cluster sizes
            for i in range(len(centers)):
                if cluster_counts[i] > 0:
                    # Reduce activation for over-represented clusters
                    size_factor = len(centers) / (cluster_counts[i] * len(centers) / n_samples + 1)
                    rbf_activations[:, i] *= size_factor ** balance_factor
            
            # Re-calculate assignments after balancing
            cluster_assignments = np.argmax(rbf_activations, axis=1)
            
            # Create output based on cluster assignment
            y = np.zeros((n_samples, weights['weights'].shape[1]))
            for i in range(len(centers)):
                mask = cluster_assignments == i
                if np.any(mask):
                    y[mask] = i
            
            # Add some continuous variation within clusters
            y += np.random.randn(n_samples, y.shape[1]) * 0.1
            
            # Normalize if specified
            if weights.get('normalize', False):
                y_mean = np.mean(y, axis=0, keepdims=True)
                y_std = np.std(y, axis=0, keepdims=True) + 1e-8
                y = (y - y_mean) / y_std
        elif self.transform_type == "enhanced_mixture":
            # Enhanced mixture with repulsion and optimal separation
            centers = weights['centers']
            gamma = weights['gamma']
            repulsion_factor = weights.get('repulsion_factor', 2.0)
            
            # Calculate RBF activations for each center
            activations = np.zeros((n_samples, len(centers)))
            for i, center in enumerate(centers):
                distances_squared = np.sum((X_np - center) ** 2, axis=1)
                activations[:, i] = np.exp(-gamma * distances_squared)
            
            # Add repulsion between clusters
            if repulsion_factor > 0:
                # Calculate pairwise repulsion
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        center_dist = np.linalg.norm(centers[i] - centers[j])
                        if center_dist < 10.0:  # Only repel nearby clusters
                            repulsion_strength = np.exp(-center_dist / 2.0) * repulsion_factor
                            # Reduce activation overlap
                            overlap = activations[:, i] * activations[:, j]
                            activations[:, i] -= overlap * repulsion_strength
                            activations[:, j] -= overlap * repulsion_strength
            
            # Normalize activations
            activations = np.maximum(activations, 0)  # Remove negative values from repulsion
            row_sums = activations.sum(axis=1, keepdims=True) + 1e-8
            activations = activations / row_sums
            
            # Create output based on strongest activation
            y = np.zeros((n_samples, 1))
            cluster_assignments = np.argmax(activations, axis=1)
            
            for i in range(len(centers)):
                mask = cluster_assignments == i
                if np.any(mask):
                    y[mask] = i
            
            # Add small continuous variation within clusters
            y += np.random.randn(n_samples, 1) * 0.05
            
            # Normalize if specified
            if weights.get('normalize', False):
                y_mean = np.mean(y, axis=0, keepdims=True)
                y_std = np.std(y, axis=0, keepdims=True) + 1e-8
                y = (y - y_mean) / y_std
                        
        else:
            # Linear combination (matrix multiplication is faster)
            y = X_np @ weights['weights']
            
        return y
    
    def _inject_noise(self, y: np.ndarray) -> np.ndarray:
        """Inject various types of noise into targets with performance optimizations."""
        if self.swap_prob == 0.0:
            return y
            
        y_noisy = y.copy()
        n_samples = y.shape[0]
        
        if self.noise_type == "swap":
            # Vectorized swap implementation
            percentile = self.swap_prob * 100
            num_samples_to_swap = int(percentile * n_samples / 100)
            
            if num_samples_to_swap == 0:
                return y
            
            # Ensure we have an even number for pairwise swapping
            num_samples_to_swap = (num_samples_to_swap // 2) * 2
            
            if num_samples_to_swap > 0:
                # Generate random permutation only once
                y_permuted = np.random.randn(n_samples, y.shape[1])
                
                # Use vectorized operations for swapping
                indices_to_swap = np.random.choice(n_samples, num_samples_to_swap, replace=False)
                idx1 = indices_to_swap[:num_samples_to_swap//2]
                idx2 = indices_to_swap[num_samples_to_swap//2:]
                
                # Swap in one vectorized operation
                y_noisy[idx1] = y_permuted[idx2]
                y_noisy[idx2] = y_permuted[idx1]
                
        elif self.noise_type == "corrupt":
            # Vectorized corruption
            n_corrupt = int(self.swap_prob * n_samples)
            if n_corrupt > 0:
                corrupt_indices = np.random.choice(n_samples, n_corrupt, replace=False)
                
                # Single vectorized operation for noise generation
                y_std = np.std(y, axis=0, keepdims=True)
                noise_scales = np.random.uniform(0.5, 2.0, size=(n_corrupt, 1)) * y_std
                noise = np.random.randn(n_corrupt, y.shape[1]) * noise_scales
                y_noisy[corrupt_indices] += noise
                
        elif self.noise_type == "boundary_blur":
            # Optimized boundary blurring
            # Instead of processing each dimension separately, process in batches
            n_dims = y.shape[1]
            
            # Pre-calculate percentiles for all dimensions at once
            if n_samples > 1000:
                # For large datasets, approximate with sampling
                sample_size = min(1000, n_samples)
                sample_indices = np.random.choice(n_samples, sample_size, replace=False)
                y_sample = y[sample_indices]
                
                # Approximate percentiles from sample
                low_percentiles = np.percentile(y_sample, 30, axis=0)
                high_percentiles = np.percentile(y_sample, 70, axis=0)
            else:
                # For smaller datasets, calculate exact percentiles
                low_percentiles = np.percentile(y, 30, axis=0)
                high_percentiles = np.percentile(y, 70, axis=0)
            
            # Find indices in the boundary region for all dimensions at once
            in_boundary = np.logical_and(
                y >= low_percentiles,
                y <= high_percentiles
            )
            
            # Number of elements to blur
            n_blur = int(self.swap_prob * np.count_nonzero(in_boundary) / n_dims)
            
            if n_blur > 0:
                # Convert to flat indices
                boundary_indices = np.where(in_boundary)
                if len(boundary_indices[0]) > 0:
                    # Randomly select from boundary_indices
                    selection_idx = np.random.choice(len(boundary_indices[0]), min(n_blur, len(boundary_indices[0])), replace=False)
                    
                    # Apply noise to selected indices
                    row_indices = boundary_indices[0][selection_idx]
                    col_indices = boundary_indices[1][selection_idx]
                    
                    # Calculate noise scale per dimension
                    y_std = np.std(y, axis=0)
                    noise_scales = y_std[col_indices] * 0.3
                    
                    # Add noise
                    y_noisy[row_indices, col_indices] += np.random.randn(len(selection_idx)) * noise_scales
                
        elif self.noise_type == "mixed":
            # Optimized mixed noise approach
            # Apply noise types in a single pass with vectorized operations
            
            # Calculate how many samples to affect with each noise type
            n_samples_swap = int(self.swap_prob * 0.3 * n_samples)
            n_samples_corrupt = int(self.swap_prob * 0.3 * n_samples)
            n_samples_blur = int(self.swap_prob * 0.4 * n_samples)
            
            # Make sure we don't overlap indices
            all_indices = np.random.permutation(n_samples)
            swap_indices = all_indices[:n_samples_swap]
            corrupt_indices = all_indices[n_samples_swap:n_samples_swap+n_samples_corrupt]
            blur_indices = all_indices[n_samples_swap+n_samples_corrupt:n_samples_swap+n_samples_corrupt+n_samples_blur]
            
            # Apply swaps (if any)
            if n_samples_swap >= 2:
                n_pairs = n_samples_swap // 2
                idx1 = swap_indices[:n_pairs]
                idx2 = swap_indices[n_pairs:2*n_pairs]
                
                # Generate random values only for the samples we're swapping
                y_random = np.random.randn(n_pairs, y.shape[1])
                
                # Swap using temporary array to avoid race conditions
                temp = y_noisy[idx1].copy()
                y_noisy[idx1] = y_random
                y_noisy[idx2] = temp
            
            # Apply corruption (if any)
            if n_samples_corrupt > 0:
                y_std = np.std(y, axis=0, keepdims=True)
                noise_scales = np.random.uniform(0.5, 2.0, size=(len(corrupt_indices), 1)) * y_std
                noise = np.random.randn(len(corrupt_indices), y.shape[1]) * noise_scales
                y_noisy[corrupt_indices] += noise
            
            # Apply blur (if any) - simplified version for performance
            if n_samples_blur > 0:
                y_std = np.std(y, axis=0)
                for j in range(min(3, y.shape[1])):  # Limit to 3 dimensions max for performance
                    noise_scale = y_std[j] * 0.2
                    noise = np.random.randn(len(blur_indices)) * noise_scale
                    y_noisy[blur_indices, j] += noise
            
        return y_noisy
    
    def forward(self, X):
        """Applies the tree-based transformation with optimized performance."""
        # Handle tensor conversion efficiently
        if isinstance(X, torch.Tensor):
            X_tensor = X.nan_to_num(0.0)
            X_np = X_tensor.cpu().numpy()
        else:
            X_np = X
            X_tensor = torch.tensor(X, dtype=torch.float, device=self.device)
        
        # Generate deterministic targets
        y_deterministic = self._generate_deterministic_targets(X_tensor)
        
        # Apply controlled noise injection
        y_targets = self._inject_noise(y_deterministic)
        
        # Fit tree model with early stopping for faster training
        self.model.fit(X_np, y_targets)
        
        # Use batch prediction for better performance
        y = self.model.predict(X_np)
        y = torch.tensor(y, dtype=torch.float, device=self.device)
        
        if self.out_dim == 1:
            y = y.view(-1, 1)
            
        return y


class DeterministicTreeSCM(nn.Module):
    """Optimized version of DeterministicTreeSCM with performance improvements."""
    
    def __init__(self,
                 seq_len: int = 1024,
                 num_features: int = 100,
                 num_outputs: int = 1,
                 hyperparams: dict = None,
                 is_causal: bool = False,
                 num_causes: int = 10,
                 y_is_effect: bool = True,
                 in_clique: bool = False,
                 sort_features: bool = True,
                 num_layers: int = 2,
                 hidden_dim: int = 10,
                 tree_model: str = "random_forest",
                 max_depth: int = 4,
                 n_estimators: int = 10,
                 min_swap_prob: float = 0.0,
                 max_swap_prob: float = 0.2,
                 transform_type: str = "polynomial",
                 noise_type: str = "swap",
                 sampling: str = "normal",
                 pre_sample_cause_stats: bool = False,
                 noise_std: float = 0.001,
                 pre_sample_noise_std: bool = False,
                 class_separability: float = 1.0,
                 device: str = "cpu",
                 n_jobs: int = 4,  # Control parallelism explicitly
                 **kwargs):
        super(DeterministicTreeSCM, self).__init__()
        
        # Store parameters
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.is_causal = is_causal
        self.num_causes = num_causes
        self.y_is_effect = y_is_effect
        self.in_clique = in_clique
        self.sort_features = sort_features
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.tree_model = tree_model
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.min_swap_prob = min_swap_prob
        self.max_swap_prob = max_swap_prob
        self.transform_type = transform_type
        self.noise_type = noise_type
        self.sampling = sampling
        self.pre_sample_cause_stats = pre_sample_cause_stats
        self.noise_std = noise_std
        self.pre_sample_noise_std = pre_sample_noise_std
        self.class_separability = class_separability
        self.device = device
        self.n_jobs = n_jobs
        self.hyperparams = hyperparams or {}
        
        # Cache for reusable components
        self._output_cache = None
        self._causes_cache = None
        
        if self.is_causal:
            # Ensure enough intermediate variables for sampling X and y
            self.hidden_dim = max(self.hidden_dim, self.num_outputs + 2 * self.num_features)
        else:
            # In non-causal mode, features are the causes
            self.num_causes = self.num_features
        
        # Define the input sampler
        self.xsampler = XSampler(
            seq_len=self.seq_len,
            num_features=self.num_causes,
            pre_stats=self.pre_sample_cause_stats,
            sampling=self.sampling,
            device=self.device,
        )
        
        # Build layers
        if self.num_layers == 1:
            # Single layer: directly from causes to outputs
            self.layers = nn.ModuleList([self._make_layer(self.num_causes, self.num_outputs)])
        else:
            # Multiple layers: causes -> hidden -> ... -> outputs
            self.layers = nn.ModuleList([self._make_layer(self.num_causes, self.hidden_dim)])
            for _ in range(self.num_layers - 2):
                self.layers.append(self._make_layer(self.hidden_dim, self.hidden_dim))
            self.layers.append(self._make_layer(self.hidden_dim, self.num_outputs))
            
        # Pre-compute random permutations for efficiency
        self._precompute_permutations()
    
    def _precompute_permutations(self):
        """Pre-compute random permutations for output handling."""
        if self.is_causal:
            # For causal mode, we need permutations for output sampling
            flat_dim = self.num_layers * self.hidden_dim
            if self.in_clique:
                # Pre-compute a set of starting points
                max_start = flat_dim - self.num_outputs - self.num_features
                if max_start > 0:
                    self._start_indices = np.random.randint(0, max_start + 1, size=10)
                else:
                    self._start_indices = np.zeros(10, dtype=int)
            else:
                # Pre-compute a set of permutations
                self._permutations = [torch.randperm(flat_dim, device=self.device) for _ in range(5)]
    
    def _make_layer(self, in_dim: int, out_dim: int) -> nn.Module:
        """Create a tree layer with optimized configuration."""
        # Sample swap probability for this layer
        swap_prob = np.random.uniform(self.min_swap_prob, self.max_swap_prob)
        
        tree_layer = DeterministicTreeLayer(
            tree_model=self.tree_model,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            out_dim=out_dim,
            swap_prob=swap_prob,
            transform_type=self.transform_type,
            device=self.device,
            noise_type=self.noise_type,
            n_jobs=self.n_jobs,  # Pass explicit parallelism control
            class_separability=self.class_separability,
            hyperparams=self.hyperparams,
        )
        
        if self.pre_sample_noise_std:
            noise_std = torch.abs(
                torch.normal(torch.zeros(size=(1, out_dim), device=self.device), float(self.noise_std))
            )
        else:
            noise_std = self.noise_std
        noise_layer = GaussianNoise(noise_std)
        
        return nn.Sequential(tree_layer, noise_layer)
    
    def forward(self, X=None, info=None):
        """Generates synthetic data with enhanced performance."""
        # If X is provided, use it directly as the causes
        if X is not None:
            causes = X
            # Update the number of causes based on input
            if causes.shape[1] != self.num_causes:
                self.num_causes = causes.shape[1]
        else:
            # Sample causes or use cached values if available
            if self._causes_cache is None:
                causes = self.xsampler.sample()  # (seq_len, num_causes)
                self._causes_cache = causes
            else:
                causes = self._causes_cache
        
        # Generate outputs through tree layers
        if self._output_cache is None:
            outputs = [causes]
            for layer in self.layers:
                outputs.append(layer(outputs[-1]))
            
            # Remove the first element (initial causes) to get only layer outputs
            outputs = outputs[1:]
            self._output_cache = outputs
        else:
            outputs = self._output_cache
        
        # Handle outputs based on causality
        X, y = self.handle_outputs(causes, outputs)
        
        # Apply class separability scaling to increase separation between different classes
        if self.class_separability != 1.0 and self.num_outputs == 1:
            X = apply_class_separability(X, y, self.class_separability)
        
        # Check for NaNs and handle them
        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)):
            X = torch.zeros_like(X)
            y = torch.full_like(y, -100.0)
        
        if self.num_outputs == 1:
            y = y.squeeze(-1)
            
        # Clear cache after usage to avoid memory leaks
        self._causes_cache = None
        self._output_cache = None
            
        # If info dict is provided, return in expected format for testing
        if info is not None:
            return {'y_cont': y.unsqueeze(-1) if y.ndim == 1 else y}
        
        return X, y
    
    def handle_outputs(self, causes, outputs):
        """Optimized output handling based on causality mode."""
        if self.is_causal:
            # More efficient concatenation
            outputs_flat = torch.cat(outputs, dim=-1)
            
            if self.in_clique:
                # Use pre-computed start index
                start_idx = np.random.choice(self._start_indices)
                start = min(start_idx, outputs_flat.shape[-1] - self.num_outputs - self.num_features)
                random_perm = start + torch.arange(self.num_outputs + self.num_features, device=self.device)
            else:
                # Use pre-computed permutation
                random_perm = self._permutations[np.random.randint(0, len(self._permutations))]
            
            indices_X = random_perm[self.num_outputs : self.num_outputs + self.num_features]
            if self.y_is_effect:
                # Take from final outputs (more efficient indexing)
                indices_y = torch.arange(-self.num_outputs, 0, device=self.device)
            else:
                # Take from permuted list
                indices_y = random_perm[: self.num_outputs]
            
            if self.sort_features:
                indices_X, _ = torch.sort(indices_X)
            
            # Use .index_select for more efficient indexing
            X = torch.index_select(outputs_flat, 1, indices_X)
            y = torch.index_select(outputs_flat, 1, indices_y)
        else:
            # Non-causal mode: direct mapping
            X = causes
            y = outputs[-1]
            
        return X, y
