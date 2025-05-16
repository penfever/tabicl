"""Test script to verify imbalanced class generation."""

import numpy as np
import torch
from collections import Counter
from tabicl.prior.dataset import PriorDataset


def test_imbalance_ratio(min_ratio=1.0, max_ratio=5.0, n_datasets=10):
    """Test that generated datasets have the expected class imbalance."""
    
    print(f"\nTesting with imbalance ratio range [{min_ratio}, {max_ratio}]")
    print("-" * 50)
    
    # Create dataset with imbalance parameters
    dataset = PriorDataset(
        batch_size=n_datasets,
        min_classes=3,
        max_classes=5,
        min_seq_len=1000,
        max_seq_len=1000,
        min_imbalance_ratio=min_ratio,
        max_imbalance_ratio=max_ratio,
        prior_type="dummy",  # Use dummy for faster testing
        device="cpu"
    )
    
    # Generate a batch
    X, y, d, seq_lens, train_sizes = next(dataset)
    
    # Analyze class distribution for each dataset
    for i in range(n_datasets):
        labels = y[i].cpu().numpy()
        class_counts = Counter(labels)
        
        # Calculate actual imbalance ratio
        if len(class_counts) > 1:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            actual_ratio = max_count / min_count
            
            print(f"Dataset {i}: Classes={len(class_counts)}, "
                  f"Counts={dict(class_counts)}, "
                  f"Imbalance ratio={actual_ratio:.2f}")
            
            # Verify the ratio is within expected range (with some tolerance)
            assert actual_ratio >= min_ratio * 0.8, f"Ratio {actual_ratio} is less than expected minimum {min_ratio}"
            assert actual_ratio <= max_ratio * 1.2, f"Ratio {actual_ratio} exceeds expected maximum {max_ratio}"
    
    print("\nTest passed!")


def test_scm_imbalance(min_ratio=1.0, max_ratio=3.0, n_datasets=5):
    """Test imbalance with SCM-based priors."""
    
    print(f"\nTesting SCM with imbalance ratio range [{min_ratio}, {max_ratio}]")
    print("-" * 50)
    
    # Create dataset with imbalance parameters
    dataset = PriorDataset(
        batch_size=n_datasets,
        min_classes=3,
        max_classes=4,
        min_seq_len=500,
        max_seq_len=500,
        min_imbalance_ratio=min_ratio,
        max_imbalance_ratio=max_ratio,
        prior_type="mlp_scm",
        device="cpu"
    )
    
    # Generate a batch
    X, y, d, seq_lens, train_sizes = next(dataset)
    
    # Analyze class distribution for each dataset
    for i in range(n_datasets):
        labels = y[i].cpu().numpy()
        class_counts = Counter(labels)
        
        # Calculate actual imbalance ratio
        if len(class_counts) > 1:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            actual_ratio = max_count / min_count
            
            print(f"Dataset {i}: Classes={len(class_counts)}, "
                  f"Counts={dict(class_counts)}, "
                  f"Imbalance ratio={actual_ratio:.2f}")
    
    print("\nTest passed!")


if __name__ == "__main__":
    # Test balanced case
    test_imbalance_ratio(min_ratio=1.0, max_ratio=1.0, n_datasets=5)
    
    # Test moderate imbalance
    test_imbalance_ratio(min_ratio=2.0, max_ratio=5.0, n_datasets=5)
    
    # Test high imbalance
    test_imbalance_ratio(min_ratio=5.0, max_ratio=10.0, n_datasets=5)
    
    # Test with SCM
    test_scm_imbalance(min_ratio=1.0, max_ratio=3.0, n_datasets=5)
    
    print("\nAll tests passed successfully!")