"""
Deterministic Tree-based SCM with controlled target swapping for generating learnable synthetic datasets.
"""

from __future__ import annotations

import random
import numpy as np
import torch
from torch import nn
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from typing import Optional, Tuple

from .utils import GaussianNoise, XSampler


class DeterministicTreeLayer(nn.Module):
    """A layer that transforms input features using a tree-based model with controlled randomness.
    
    Instead of fitting to completely random targets, this layer creates a deterministic
    transformation that can be optionally corrupted by swapping target pairs with
    probability p.
    
    Parameters
    ----------
    tree_model : str
        The type of tree-based model to use. Options are "decision_tree",
        "extra_trees", "random_forest", "xgboost".
    
    max_depth : int
        The maximum depth allowed for the individual trees in the model.
    
    n_estimators : int
        The number of trees in the ensemble.
    
    out_dim : int
        The desired output dimension for the transformed features.
    
    swap_prob : float
        Probability of swapping any pair of targets (0.0 = deterministic, 1.0 = very noisy)
    
    transform_type : str
        Type of deterministic transformation to apply before swapping.
        Options: "polynomial", "trigonometric", "exponential", "mixed"
    
    device : str or torch.device
        The device ('cpu' or 'cuda') on which to place the output tensor.
    """
    
    def __init__(self, 
                 tree_model: str, 
                 max_depth: int, 
                 n_estimators: int, 
                 out_dim: int,
                 swap_prob: float = 0.1,
                 transform_type: str = "polynomial",
                 device: str = "cpu"):
        super(DeterministicTreeLayer, self).__init__()
        self.out_dim = out_dim
        self.swap_prob = swap_prob
        self.transform_type = transform_type
        self.device = device
        
        # Adaptive complexity based on swap probability
        if swap_prob > 0.15:
            # Use simpler model for more noisy data
            self.max_depth = min(max_depth, 3)
            self.n_estimators = min(n_estimators, 50)
        else:
            self.max_depth = max_depth
            self.n_estimators = n_estimators
        
        if tree_model == "decision_tree":
            self.model = MultiOutputRegressor(DecisionTreeRegressor(max_depth=self.max_depth), n_jobs=-1)
        elif tree_model == "extra_trees":
            self.model = MultiOutputRegressor(
                ExtraTreesRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth), n_jobs=-1
            )
        elif tree_model == "random_forest":
            self.model = MultiOutputRegressor(
                RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth), n_jobs=-1
            )
        elif tree_model == "xgboost":
            self.model = XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                tree_method="hist",
                multi_strategy="multi_output_tree",
                n_jobs=-1,
                learning_rate=0.3,
                verbosity=0,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.01,
                reg_lambda=0.01,
            )
        else:
            raise ValueError(f"Invalid tree model: {tree_model}")
    
    def _generate_deterministic_targets(self, X: torch.Tensor) -> np.ndarray:
        """Generate deterministic targets based on input features."""
        X_np = X.cpu().numpy()
        n_samples, n_features = X_np.shape
        
        if self.transform_type == "polynomial":
            # Polynomial transformation: y = sum(x_i^2) + cross terms
            y = np.zeros((n_samples, self.out_dim))
            for j in range(self.out_dim):
                # Use different feature combinations for each output
                feat_indices = np.random.choice(n_features, size=min(5, n_features), replace=False)
                y[:, j] = np.sum(X_np[:, feat_indices] ** 2, axis=1)
                # Add some cross terms
                if len(feat_indices) > 1:
                    y[:, j] += X_np[:, feat_indices[0]] * X_np[:, feat_indices[1]]
                    
        elif self.transform_type == "trigonometric":
            # Trigonometric transformation
            y = np.zeros((n_samples, self.out_dim))
            for j in range(self.out_dim):
                feat_indices = np.random.choice(n_features, size=min(3, n_features), replace=False)
                y[:, j] = np.sin(X_np[:, feat_indices[0]] * np.pi)
                if len(feat_indices) > 1:
                    y[:, j] += np.cos(X_np[:, feat_indices[1]] * np.pi / 2)
                    
        elif self.transform_type == "exponential":
            # Exponential transformation (capped to avoid overflow)
            y = np.zeros((n_samples, self.out_dim))
            for j in range(self.out_dim):
                feat_indices = np.random.choice(n_features, size=min(2, n_features), replace=False)
                y[:, j] = np.exp(np.clip(X_np[:, feat_indices[0]], -5, 5))
                
        elif self.transform_type == "mixed":
            # Mix of different transformations
            y = np.zeros((n_samples, self.out_dim))
            transforms = ["polynomial", "trigonometric", "exponential"]
            for j in range(self.out_dim):
                transform = np.random.choice(transforms)
                feat_indices = np.random.choice(n_features, size=min(3, n_features), replace=False)
                
                if transform == "polynomial":
                    y[:, j] = np.sum(X_np[:, feat_indices] ** 2, axis=1)
                elif transform == "trigonometric":
                    y[:, j] = np.sin(X_np[:, feat_indices[0]] * np.pi)
                else:  # exponential
                    y[:, j] = np.exp(np.clip(X_np[:, feat_indices[0]], -5, 5))
        else:
            # Default to linear combination
            weights = np.random.randn(n_features, self.out_dim)
            y = X_np @ weights
            
        return y
    
    def _swap_targets(self, y: np.ndarray) -> np.ndarray:
        """Randomly swap pairs of targets with probability swap_prob, limited to n_classes swaps."""
        if self.swap_prob == 0.0:
            return y
            
        y_swapped = y.copy()
        n_samples = y.shape[0]
        
        # For regression targets, estimate n_classes as square root of n_samples
        # For classification, would use actual unique values
        n_classes = int(np.sqrt(n_samples))
        
        # Calculate the expected number of swaps based on swap_prob
        # Limited to n_classes maximum
        expected_swaps = min(int(self.swap_prob * n_samples / 2), n_classes)
        
        if expected_swaps == 0:
            return y
            
        # Generate random pairs of indices to swap
        np.random.seed(None)  # Use random seed for true randomness
        
        # Randomly select pairs to swap
        all_indices = np.arange(n_samples)
        np.random.shuffle(all_indices)
        
        # Perform the limited number of swaps
        swaps_made = 0
        for i in range(0, min(n_samples - 1, expected_swaps * 2), 2):
            if swaps_made >= expected_swaps:
                break
                
            idx1, idx2 = all_indices[i], all_indices[i + 1]
            y_swapped[idx1], y_swapped[idx2] = y_swapped[idx2].copy(), y_swapped[idx1].copy()
            swaps_made += 1
            
        return y_swapped
    
    def forward(self, X):
        """Applies the deterministic tree-based transformation with controlled noise."""
        # Handle tensor conversion
        if isinstance(X, torch.Tensor):
            X_tensor = X.nan_to_num(0.0)
            X_np = X_tensor.cpu().numpy()
        else:
            X_np = X
            X_tensor = torch.tensor(X, dtype=torch.float, device=self.device)
        
        # Generate deterministic targets
        y_deterministic = self._generate_deterministic_targets(X_tensor)
        
        # Apply controlled swapping
        y_targets = self._swap_targets(y_deterministic)
        
        # Fit tree model
        self.model.fit(X_np, y_targets)
        
        # Predict (this will learn the potentially swapped patterns)
        y = self.model.predict(X_np)
        y = torch.tensor(y, dtype=torch.float, device=self.device)
        
        if self.out_dim == 1:
            y = y.view(-1, 1)
            
        return y


class DeterministicTreeSCM(nn.Module):
    """A Deterministic Tree-based Structural Causal Model for generating learnable synthetic datasets.
    
    This version creates datasets with controllable difficulty by adjusting the target swapping
    probability rather than using completely random targets.
    
    Parameters
    ----------
    seq_len : int, default=1024
        The number of samples (rows) to generate for the dataset.
        
    num_features : int, default=100
        The number of features.
        
    num_outputs : int, default=1
        The number of outputs.
        
    is_causal : bool, default=False
        Whether to use causal mode (sampling from intermediate outputs).
        
    num_causes : int, default=10
        The number of initial root 'cause' variables.
        
    y_is_effect : bool, default=True
        Specifies how the target `y` is selected when `is_causal=True`.
        
    in_clique : bool, default=False
        Controls how features `X` and targets `y` are sampled.
        
    sort_features : bool, default=True
        Whether to sort features by their indices.
        
    num_layers : int, default=2
        Number of tree transformation layers.
        
    hidden_dim : int, default=10
        Output dimension size for intermediate tree transformations.
        
    tree_model : str, default="random_forest"
        Type of tree model to use.
        
    max_depth : int, default=4
        Maximum depth for tree models.
        
    n_estimators : int, default=10
        Number of estimators for ensemble models.
        
    min_swap_prob : float, default=0.0
        Minimum probability of swapping target pairs.
        
    max_swap_prob : float, default=0.2
        Maximum probability of swapping target pairs.
        
    transform_type : str, default="polynomial"
        Type of deterministic transformation to use.
        
    sampling : str, default="normal"
        The method used by `XSampler` to generate the initial 'cause' variables.
        
    pre_sample_cause_stats : bool, default=False
        If `True`, pre-sample statistics for cause variables.
        
    noise_std : float, default=0.001
        Standard deviation for Gaussian noise.
        
    pre_sample_noise_std : bool, default=False
        Whether to pre-sample noise standard deviation.
        
    device : str, default="cpu"
        The computing device.
        
    **kwargs : dict
        Additional unused hyperparameters.
    """
    
    def __init__(self,
                 seq_len: int = 1024,
                 num_features: int = 100,
                 num_outputs: int = 1,
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
                 sampling: str = "normal",
                 pre_sample_cause_stats: bool = False,
                 noise_std: float = 0.001,
                 pre_sample_noise_std: bool = False,
                 device: str = "cpu",
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
        self.sampling = sampling
        self.pre_sample_cause_stats = pre_sample_cause_stats
        self.noise_std = noise_std
        self.pre_sample_noise_std = pre_sample_noise_std
        self.device = device
        
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
    
    def _make_layer(self, in_dim: int, out_dim: int) -> nn.Module:
        """Create a tree layer with noise."""
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
        )
        
        if self.pre_sample_noise_std:
            noise_std = torch.abs(
                torch.normal(torch.zeros(size=(1, out_dim), device=self.device), float(self.noise_std))
            )
        else:
            noise_std = self.noise_std
        noise_layer = GaussianNoise(noise_std)
        
        return nn.Sequential(tree_layer, noise_layer)
    
    def forward(self):
        """Generates synthetic data by sampling input features and applying tree-based transformations."""
        causes = self.xsampler.sample()  # (seq_len, num_causes)
        
        # Generate outputs through tree layers
        outputs = [causes]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        
        # Remove the first element (initial causes) to get only layer outputs
        outputs = outputs[1:]
        
        # Handle outputs based on causality
        X, y = self.handle_outputs(causes, outputs)
        
        # Check for NaNs and handle them
        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)):
            X[:] = 0.0
            y[:] = -100.0
        
        if self.num_outputs == 1:
            y = y.squeeze(-1)
            
        return X, y
    
    def handle_outputs(self, causes, outputs):
        """
        Handles outputs based on whether causal or not.
        
        Parameters
        ----------
        causes : torch.Tensor
            Causes of shape (seq_len, num_causes)
            
        outputs : list of torch.Tensor
            List of output tensors from tree layers
            
        Returns
        -------
        X : torch.Tensor
            Input features (seq_len, num_features)
            
        y : torch.Tensor
            Target (seq_len, num_outputs)
        """
        if self.is_causal:
            outputs_flat = torch.cat(outputs, dim=-1)
            if self.in_clique:
                # Sample from contiguous block
                start = random.randint(0, outputs_flat.shape[-1] - self.num_outputs - self.num_features)
                random_perm = start + torch.randperm(self.num_outputs + self.num_features, device=self.device)
            else:
                # Random sampling
                random_perm = torch.randperm(outputs_flat.shape[-1], device=self.device)
            
            indices_X = random_perm[self.num_outputs : self.num_outputs + self.num_features]
            if self.y_is_effect:
                # Take from final outputs
                indices_y = list(range(-self.num_outputs, 0))
            else:
                # Take from permuted list
                indices_y = random_perm[: self.num_outputs]
            
            if self.sort_features:
                indices_X, _ = torch.sort(indices_X)
            
            X = outputs_flat[:, indices_X]
            y = outputs_flat[:, indices_y]
        else:
            # Non-causal mode: direct mapping
            X = causes
            y = outputs[-1]
            
        return X, y