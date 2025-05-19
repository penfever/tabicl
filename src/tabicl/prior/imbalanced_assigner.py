"""Module for creating imbalanced class distributions in synthetic datasets."""

import torch
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F


class ImbalancedMulticlassAssigner(nn.Module):
    """Transforms continuous targets into discrete classes with controlled imbalance.
    
    This class creates imbalanced class distributions by adjusting the boundaries
    between classes based on the desired imbalance ratio.
    
    Parameters
    ----------
    num_classes : int
        The target number of discrete classes to output.
        
    imbalance_ratio : float, default=1.0
        The ratio between the largest and smallest class sizes.
        1.0 means balanced classes, higher values create more imbalance.
        
    mode : str, default="rank"
        The method used to determine class boundaries:
        - "rank": Boundaries are set based on data quantiles
        - "value": Boundaries are randomly sampled from a normal distribution
    """
    
    def __init__(self, num_classes: int, imbalance_ratio: float = 1.0, mode: str = "rank"):
        super().__init__()
        if num_classes < 2:
            raise ValueError("The number of classes must be at least 2 for ImbalancedMulticlassAssigner.")
            
        self.num_classes = num_classes
        self.imbalance_ratio = imbalance_ratio
        self.mode = mode
        
    def forward(self, input: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input : Tensor
            Input of shape (T,).
            
        Returns
        -------
        Tensor
            Class labels of shape (T,) with integer values [0, num_classes-1].
        """
        T = input.shape[0]
        device = input.device
        
        # Calculate the desired class proportions based on imbalance ratio
        proportions = self._calculate_imbalanced_proportions()
        
        if self.mode == "rank":
            # Sort the input to get rank-based boundaries
            sorted_input, _ = torch.sort(input)
            
            # Calculate cumulative proportions to determine boundaries
            cumulative_props = torch.cumsum(torch.tensor(proportions[:-1], device=device), dim=0)
            boundary_indices = (cumulative_props * T).long()
            boundaries = sorted_input[boundary_indices]
            
        elif self.mode == "value":
            # For value mode, we still need to adjust boundaries to achieve imbalance
            # This is more challenging and may not guarantee exact proportions
            boundaries = torch.randn(self.num_classes - 1, device=device)
            # Adjust boundaries based on desired proportions
            # This is a simplified approach
            mean = torch.mean(input)
            std = torch.std(input)
            quantiles = torch.cumsum(torch.tensor(proportions[:-1], device=device), dim=0)
            boundaries = mean + std * torch.erfinv(2 * quantiles - 1) * np.sqrt(2)
            
        # Compare input tensor with boundaries and sum across the boundary dimension to get classes
        classes = (input.unsqueeze(-1) > boundaries.unsqueeze(0)).sum(dim=1)
        
        return classes
    
    def _calculate_imbalanced_proportions(self):
        """Calculate class proportions for the desired imbalance ratio.
        
        Returns
        -------
        list
            List of proportions for each class, summing to 1.0
        """
        if self.imbalance_ratio == 1.0:
            # Balanced case
            return [1.0 / self.num_classes] * self.num_classes
        
        # Create a geometric progression for imbalanced distribution
        # The ratio between consecutive classes is r = imbalance_ratio^(1/(num_classes-1))
        r = self.imbalance_ratio ** (1 / (self.num_classes - 1))
        
        # Calculate proportions for each class
        proportions = []
        for i in range(self.num_classes):
            proportions.append(r ** i)
            
        # Normalize so they sum to 1
        total = sum(proportions)
        proportions = [p / total for p in proportions]
        
        # Reverse so the largest class comes first (more intuitive)
        proportions.reverse()
        
        return proportions


class PiecewiseConstantAssigner(nn.Module):
    """Creates piecewise constant regions with random class assignments.
    
    Similar to TICL's MulticlassSteps approach.
    """
    
    def __init__(self, num_classes: int, max_steps: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.max_steps = max_steps
    
    def forward(self, input: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input : Tensor
            Input of shape (T,).
            
        Returns
        -------
        Tensor
            Class labels of shape (T,) with integer values [0, num_classes-1].
        """
        T = input.shape[0]
        device = input.device
        
        # Determine number of steps (regions)
        num_steps = np.random.randint(1, min(self.max_steps, T // 2) + 1)
        
        # Pick random boundaries from the data
        boundary_indices = torch.randint(0, T, (num_steps - 1,), device=device)
        boundaries = input[boundary_indices].sort()[0]
        
        # Assign each sample to a region
        regions = torch.searchsorted(boundaries, input)
        
        # Randomly map regions to classes
        class_mapping = torch.randint(0, self.num_classes, (num_steps,), device=device)
        
        return class_mapping[regions]


class RandomRegionAssigner(nn.Module):
    """Creates random non-overlapping regions and maps them to classes."""
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, input: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input : Tensor
            Input of shape (T,).
            
        Returns
        -------
        Tensor
            Class labels of shape (T,) with integer values [0, num_classes-1].
        """
        device = input.device
        
        # Create more regions than classes for variety
        num_regions = self.num_classes * np.random.randint(1, 4)
        
        # Generate random thresholds within the data range
        data_min, data_max = input.min(), input.max()
        thresholds = torch.sort(
            torch.rand(num_regions - 1, device=device) * (data_max - data_min) + data_min
        )[0]
        
        # Assign samples to regions
        regions = torch.searchsorted(thresholds, input)
        
        # Randomly map regions to classes
        class_mapping = torch.randint(0, self.num_classes, (num_regions,), device=device)
        
        return class_mapping[regions]


class StepFunctionAssigner(nn.Module):
    """Creates step function boundaries similar to TICL's approach."""
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, input: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input : Tensor
            Input of shape (T,) or (T, H).
            
        Returns
        -------
        Tensor
            Class labels of shape (T,) with integer values [0, num_classes-1].
        """
        device = input.device
        
        # If multi-dimensional, select a random feature
        if input.dim() > 1:
            feature_idx = torch.randint(0, input.shape[1], (1,), device=device)
            x_selected = input[:, feature_idx].squeeze()
        else:
            x_selected = input
            
        # Create evenly spaced thresholds in the data range
        data_min, data_max = x_selected.min(), x_selected.max()
        
        # Add small epsilon to avoid edge cases
        eps = 1e-6
        thresholds = torch.linspace(
            data_min + eps, 
            data_max - eps, 
            self.num_classes - 1, 
            device=device
        )
        
        # Optionally add some randomness to thresholds
        if np.random.random() > 0.5:
            noise = torch.randn_like(thresholds) * (data_max - data_min) * 0.1
            thresholds = (thresholds + noise).sort()[0]
        
        # Count how many thresholds each sample exceeds
        return (x_selected.unsqueeze(-1) > thresholds).sum(dim=1)


class BooleanLogicAssigner(nn.Module):
    """Creates boolean logic boundaries for classification."""
    
    def __init__(self, num_classes: int, max_terms: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.max_terms = max_terms
    
    def forward(self, input: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input : Tensor
            Input of shape (T,) or (T, H).
            
        Returns
        -------
        Tensor
            Class labels of shape (T,) with integer values [0, num_classes-1].
        """
        device = input.device
        T = input.shape[0]
        
        # Handle 1D input by expanding dimensions
        if input.dim() == 1:
            input = input.unsqueeze(-1)
        
        H = input.shape[1]
        
        # Binarize features using median
        x_binary = input > input.median(dim=0, keepdim=True)[0]
        
        # Initialize all samples to class 0
        outputs = torch.zeros(T, dtype=torch.long, device=device)
        
        # Create random boolean formulas for each class
        for class_idx in range(1, self.num_classes):
            # Number of AND terms for this class
            num_terms = np.random.randint(1, min(self.max_terms, H) + 1)
            
            # Select random features for each term
            term_size = np.random.randint(1, min(3, H) + 1)
            selected_features = torch.randint(0, H, (term_size,), device=device)
            
            # Random signs (True/False) for each feature
            signs = torch.randint(0, 2, (term_size,), device=device).bool()
            
            # Evaluate the boolean formula
            matches = (x_binary[:, selected_features] == signs).all(dim=1)
            
            # Assign matching samples to this class (later classes can override earlier ones)
            outputs[matches] = class_idx
            
            # Early stopping if we've assigned enough samples
            if (outputs == class_idx).sum() > T // (2 * self.num_classes):
                break
        
        return outputs