"""Simple test for ExplicitClustersSCM to verify it generates balanced, separable classes."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tabicl.prior.explicit_clusters_scm import ExplicitClustersSCM
from tabicl.prior.imbalanced_assigner import RankBasedAssigner

def test_explicit_clusters():
    """Test ExplicitClustersSCM directly without assigners."""
    print("Testing ExplicitClustersSCM...")
    
    # Generate data
    scm = ExplicitClustersSCM(
        seq_len=2000,
        num_features=20,
        num_outputs=1,
        num_classes=10,
        cluster_separation=2.0,
        within_cluster_std=0.3
    )
    
    # Generate samples - use forward method without info to get actual class labels
    X, y = scm.forward()
    
    # Convert to numpy
    X_np = X.numpy()
    y_np = y.numpy().squeeze().astype(int)
    
    # Check class distribution
    unique, counts = np.unique(y_np, return_counts=True)
    print(f"Classes present: {len(unique)} (expected: 10)")
    print(f"Class distribution: {dict(zip(unique, counts))}")
    print(f"Balance ratio: {max(counts) / min(counts):.2f}")
    
    # Train classifier
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification accuracy: {accuracy:.4f}")
    
    # Visualize with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_np)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_np, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'ExplicitClustersSCM PCA Visualization\nAccuracy: {accuracy:.3f}, Classes: {len(unique)}')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('explicit_clusters_direct_pca.png', dpi=300)
    plt.close()
    
    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred))
    

def test_with_rank_assigner():
    """Test ExplicitClustersSCM with RankBasedAssigner."""
    print("\n\nTesting with RankBasedAssigner...")
    
    # Generate data with continuous outputs
    scm = ExplicitClustersSCM(
        seq_len=2000,
        num_features=20,
        num_outputs=1,
        num_classes=10
    )
    
    # Generate continuous outputs
    result = scm.forward(info={'type': 'test'})
    y_cont = result['y_cont']
    
    # Also generate the features
    X, _ = scm.forward()
    
    # Apply rank-based assigner
    assigner = RankBasedAssigner(num_classes=10)
    y_assigned = assigner(y_cont).numpy().squeeze().astype(int)
    
    # Check class distribution
    unique, counts = np.unique(y_assigned, return_counts=True)
    print(f"Classes after rank assignment: {len(unique)} (expected: 10)")
    print(f"Class distribution: {dict(zip(unique, counts))}")
    print(f"Balance ratio: {max(counts) / min(counts):.2f}")
    
    # Train classifier
    X_np = X.numpy()
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_assigned, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification accuracy: {accuracy:.4f}")
    
    # Visualize
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_np)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_assigned, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'ExplicitClustersSCM with RankAssigner\nAccuracy: {accuracy:.3f}, Classes: {len(unique)}')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('explicit_clusters_rank_pca.png', dpi=300)
    plt.close()
    
    # Check continuous value distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_cont.numpy().flatten(), bins=50, alpha=0.7)
    plt.title('Distribution of Continuous Values')
    plt.xlabel('Continuous Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('explicit_clusters_continuous_dist.png', dpi=300)
    plt.close()
    
    print("\nContinuous value stats:")
    print(f"Mean: {y_cont.mean():.4f}")
    print(f"Std: {y_cont.std():.4f}")
    print(f"Min: {y_cont.min():.4f}")
    print(f"Max: {y_cont.max():.4f}")


if __name__ == "__main__":
    test_explicit_clusters()
    test_with_rank_assigner()
    print("\nTest complete. Check the PNG files for visualizations.")