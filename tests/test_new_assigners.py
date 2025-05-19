"""Test the new class assigners for improved boundary generation."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tabicl.prior.imbalanced_assigner import (
    ImbalancedMulticlassAssigner, PiecewiseConstantAssigner,
    RandomRegionAssigner, StepFunctionAssigner, BooleanLogicAssigner
)
from src.tabicl.prior.mlp_scm import MLP
from src.tabicl.prior.utils import XSampler


def visualize_boundaries(y_orig, y_assigned, assigner_name):
    """Visualize how continuous values are mapped to classes."""
    plt.figure(figsize=(12, 4))
    
    # Sort by original values for visualization
    sort_idx = np.argsort(y_orig)
    y_orig_sorted = y_orig[sort_idx]
    y_assigned_sorted = y_assigned[sort_idx]
    
    # Plot 1: Original continuous values
    plt.subplot(1, 3, 1)
    plt.plot(y_orig_sorted, 'b-', alpha=0.7)
    plt.title(f'Original Continuous Values\n({assigner_name})')
    plt.xlabel('Sample Index (sorted)')
    plt.ylabel('Value')
    
    # Plot 2: Assigned classes
    plt.subplot(1, 3, 2)
    plt.scatter(range(len(y_assigned_sorted)), y_assigned_sorted, 
                c=y_assigned_sorted, cmap='viridis', alpha=0.6, s=10)
    plt.title('Assigned Classes')
    plt.xlabel('Sample Index (sorted by value)')
    plt.ylabel('Class')
    
    # Plot 3: Class boundaries on original values
    plt.subplot(1, 3, 3)
    for class_val in np.unique(y_assigned_sorted):
        mask = y_assigned_sorted == class_val
        plt.scatter(y_orig_sorted[mask], y_assigned_sorted[mask], 
                   label=f'Class {int(class_val)}', alpha=0.7, s=20)
    plt.xlabel('Original Value')
    plt.ylabel('Assigned Class')
    plt.title('Class Boundaries')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{assigner_name.lower()}_boundaries.png', dpi=150, bbox_inches='tight')
    plt.close()


def test_assigner(assigner, assigner_name, X, y_continuous):
    """Test a single assigner."""
    print(f"\n=== Testing {assigner_name} ===")
    
    # Convert continuous to classes
    y_classes = assigner(y_continuous).numpy()
    
    print(f"Number of unique classes: {len(np.unique(y_classes))}")
    print(f"Class distribution: {np.bincount(y_classes.astype(int))}")
    
    # Visualize boundaries
    visualize_boundaries(y_continuous.numpy(), y_classes, assigner_name)
    
    # Test classification performance
    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), y_classes, test_size=0.2, random_state=42
    )
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, rf.predict(X_test))
    
    print(f"Random Forest accuracy: {accuracy:.4f}")
    
    return accuracy


def test_all_assigners():
    """Test all new assigners with synthetic data."""
    print("Testing New Class Assigners for TabICL")
    print("=" * 40)
    
    # Generate synthetic data
    n_samples = 1000
    n_features = 10
    num_classes = 5
    
    # Initialize data sampler
    x_sampler = XSampler(
        seq_len=n_samples,
        num_features=n_features,
        sampling="normal",
        device="cpu"
    )
    
    # Generate features
    X = x_sampler.sample()
    
    # Generate continuous targets (similar to what MLP would produce)
    # Using a simple polynomial transformation for testing
    weights = torch.randn(n_features)
    y_continuous = X @ weights + torch.randn(n_samples) * 0.5
    
    # Normalize y to [0, 1] range for easier comparison
    y_continuous = (y_continuous - y_continuous.min()) / (y_continuous.max() - y_continuous.min())
    
    # Test each assigner
    assigners = {
        "Original_Rank": ImbalancedMulticlassAssigner(num_classes, mode="rank"),
        "Original_Value": ImbalancedMulticlassAssigner(num_classes, mode="value"),
        "PiecewiseConstant": PiecewiseConstantAssigner(num_classes, max_steps=7),
        "RandomRegion": RandomRegionAssigner(num_classes),
        "StepFunction": StepFunctionAssigner(num_classes),
    }
    
    # Special case for BooleanLogic - needs multi-dimensional input
    if n_features > 1:
        assigners["BooleanLogic"] = BooleanLogicAssigner(num_classes, max_terms=3)
        # For boolean logic, we'll use X directly instead of y_continuous
        
    results = {}
    
    for name, assigner in assigners.items():
        if name == "BooleanLogic":
            # Boolean logic uses features directly
            y_classes = assigner(X).numpy()
            accuracy = test_classification(X.numpy(), y_classes, name)
            results[name] = accuracy
        else:
            accuracy = test_assigner(assigner, name, X, y_continuous)
            results[name] = accuracy
    
    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    for name, accuracy in results.items():
        print(f"{name:20} Accuracy: {accuracy:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    accuracies = list(results.values())
    colors = ['gray', 'gray', 'blue', 'green', 'orange', 'red'][:len(names)]
    
    bars = plt.bar(names, accuracies, color=colors, alpha=0.7)
    plt.ylabel('Classification Accuracy')
    plt.title('Performance Comparison of Class Assigners')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('assigner_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization files saved:")
    print("- assigner_comparison.png")
    for name in assigners.keys():
        print(f"- {name.lower()}_boundaries.png")


def test_classification(X, y, name):
    """Test classification performance for boolean logic assigner."""
    print(f"\n=== Testing {name} ===")
    print(f"Number of unique classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, rf.predict(X_test))
    
    print(f"Random Forest accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    test_all_assigners()