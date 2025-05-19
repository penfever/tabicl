"""
Direct test of RBF implementation in deterministic_tree_scm
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sys
import os

# Add parent directory to path to import tabicl modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tabicl.prior.deterministic_tree_scm import DeterministicTreeSCM


def test_rbf_transformation():
    """Test the RBF transformation directly"""
    print("\n=== Testing RBF Transformation ===")
    
    # Create a small dataset to test
    n_samples = 1000
    n_features = 10
    n_classes = 5
    
    # Initialize the deterministic tree SCM with RBF
    model = DeterministicTreeSCM(
        seq_len=n_samples,
        num_features=n_features,
        num_outputs=1,
        is_causal=False,
        num_layers=1,
        transform_type="rbf",
        class_separability=4.0,
        noise_std=0.001,
        min_swap_prob=0.0,
        max_swap_prob=0.0
    )
    
    # Generate data
    X, y = model()
    
    # Convert to numpy
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()
    
    # Convert continuous y to discrete classes
    y_classes = np.digitize(y_np, bins=np.percentile(y_np, [20, 40, 60, 80])) 
    
    # Analyze the data
    print(f"Dataset shape: {X_np.shape}")
    print(f"Number of unique classes: {len(np.unique(y_classes))}")
    print(f"Feature statistics:")
    print(f"  Mean: {np.mean(X_np):.4f}, Std: {np.std(X_np):.4f}")
    print(f"  Min: {np.min(X_np):.4f}, Max: {np.max(X_np):.4f}")
    
    # Test classification
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_classes, test_size=0.2, random_state=42
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, rf.predict(X_test))
    print(f"Random Forest accuracy: {accuracy:.4f}")
    
    # Visualize with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_np)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_classes, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('RBF Transformation - PCA visualization')
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.savefig('rbf_test_pca.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return X_np, y_classes, accuracy


def compare_with_polynomial():
    """Compare RBF with polynomial transformation"""
    print("\n=== Comparing RBF vs Polynomial ===")
    
    n_samples = 1000
    n_features = 10
    n_classes = 5
    
    results = {}
    
    for transform_type in ["polynomial", "rbf"]:
        print(f"\nTesting {transform_type} transformation...")
        
        model = DeterministicTreeSCM(
            seq_len=n_samples,
            num_features=n_features,
            num_outputs=1,
            is_causal=False,
            num_layers=1,
            transform_type=transform_type,
            class_separability=4.0,
            noise_std=0.001,
            min_swap_prob=0.0,
            max_swap_prob=0.0
        )
        
        # Generate data
        X, y = model()
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Convert to classes
        y_classes = np.digitize(y_np, bins=np.percentile(y_np, [20, 40, 60, 80]))
        
        # Test classification
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_classes, test_size=0.2, random_state=42
        )
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, rf.predict(X_test))
        
        results[transform_type] = accuracy
        print(f"Accuracy: {accuracy:.4f}")
    
    return results


def main():
    """Run the tests"""
    print("Testing RBF Implementation in DeterministicTreeSCM")
    print("=" * 50)
    
    # Test RBF transformation
    X_rbf, y_rbf, acc_rbf = test_rbf_transformation()
    
    # Compare with polynomial
    results = compare_with_polynomial()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("Transformation accuracies:")
    for transform, acc in results.items():
        print(f"  {transform}: {acc:.4f}")
    
    print("\nVisualization saved to: rbf_test_pca.png")


if __name__ == "__main__":
    main()