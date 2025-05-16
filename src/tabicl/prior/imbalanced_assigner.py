"""Module for creating imbalanced class distributions in synthetic datasets."""

import torch
import numpy as np
from torch import nn, Tensor


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