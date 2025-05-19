"""Test only the explicit clusters approach."""

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

from tabicl.prior.explicit_clusters_scm import ExplicitClustersSCM
from tabicl.prior.imbalanced_assigner import ImbalancedMulticlassAssigner


def test_explicit_clusters():
    """Test the explicit clusters approach."""
    print("Testing Explicit Clusters Approach")
    print("=" * 40)
    
    n_samples = 2000
    n_features = 10
    num_classes = 10
    
    # Create the explicit clusters SCM
    scm = ExplicitClustersSCM(
        seq_len=n_samples,
        num_features=n_features,
        num_outputs=1,
        num_classes=num_classes,
        min_samples_per_class=n_samples // num_classes,
        cluster_separation=5.0,
        within_cluster_std=0.5,
        device="cpu"
    )
    
    # Test 1: Direct generation
    print("\nTest 1: Direct Generation")
    print("-" * 30)
    X, y = scm()
    X_np = X.numpy()
    y_np = y.numpy().squeeze().astype(int)
    
    # Check class distribution
    class_distribution = np.bincount(y_np)
    print(f"Class distribution: {class_distribution}")
    print(f"Min samples per class: {np.min(class_distribution)}")
    print(f"Max samples per class: {np.max(class_distribution)}")
    print(f"Imbalance ratio: {np.max(class_distribution) / np.min(class_distribution):.4f}")
    
    # Test classification
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42
    )
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, rf.predict(X_test))
    print(f"Random Forest accuracy: {accuracy:.4f}")
    
    # Test 2: With continuous output and assigner
    print("\n\nTest 2: With Continuous Output and Assigner")
    print("-" * 30)
    info = {}
    continuous_outputs = scm(info=info)
    y_continuous = continuous_outputs['y_cont'].numpy().squeeze()
    
    print(f"Continuous values shape: {y_continuous.shape}")
    print(f"Continuous values range: [{y_continuous.min():.3f}, {y_continuous.max():.3f}]")
    
    # Test with rank assigner
    assigner = ImbalancedMulticlassAssigner(num_classes, mode="rank")
    y_tensor = torch.from_numpy(y_continuous).unsqueeze(-1)
    y_assigned = assigner(y_tensor).numpy()
    
    # Check assigned class distribution
    assigned_distribution = np.bincount(y_assigned.astype(int))
    print(f"Assigned class distribution: {assigned_distribution}")
    print(f"Imbalance ratio: {np.max(assigned_distribution) / np.min(assigned_distribution):.4f}")
    
    # Test classification with assigned classes
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_assigned, test_size=0.2, random_state=42
    )
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, rf.predict(X_test))
    print(f"Random Forest accuracy with assigner: {accuracy:.4f}")
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Direct generation class distribution
    plt.subplot(1, 3, 1)
    plt.bar(range(len(class_distribution)), class_distribution, color='skyblue', alpha=0.7)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Direct Generation')
    
    # Plot 2: Assigned class distribution
    plt.subplot(1, 3, 2)
    plt.bar(range(len(assigned_distribution)), assigned_distribution, color='lightgreen', alpha=0.7)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('After Rank Assigner')
    
    # Plot 3: Continuous values histogram
    plt.subplot(1, 3, 3)
    plt.hist(y_continuous, bins=50, color='salmon', alpha=0.7)
    plt.xlabel('Continuous Value')
    plt.ylabel('Frequency')
    plt.title('Continuous Output Distribution')
    
    plt.tight_layout()
    plt.savefig('explicit_clusters_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization saved as 'explicit_clusters_test.png'")
    
    return accuracy


if __name__ == "__main__":
    accuracy = test_explicit_clusters()
    print(f"\nFinal accuracy: {accuracy:.4f}")