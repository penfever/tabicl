"""Debug script to understand why explicit clusters are failing."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tabicl.prior.explicit_clusters_scm import ExplicitClustersSCM
from tabicl.prior.fixed_explicit_clusters_scm import FixedExplicitClustersSCM
from tabicl.prior.imbalanced_assigner import ImbalancedMulticlassAssigner


def visualize_continuous_vs_assigned():
    """Visualize what happens to continuous values through the assigner."""
    
    # Create two SCMs
    scm_original = ExplicitClustersSCM(
        seq_len=2000,
        num_features=50,
        num_outputs=1,
        num_classes=10,
        cluster_separation=2.0,
        within_cluster_std=0.3
    )
    
    scm_fixed = FixedExplicitClustersSCM(
        seq_len=2000,
        num_features=50,
        num_outputs=1,
        num_classes=10,
        cluster_separation=3.0,
        within_cluster_std=0.3
    )
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for idx, (scm, name) in enumerate([(scm_original, 'Original'), (scm_fixed, 'Fixed')]):
        # Generate data with continuous labels
        result = scm.forward(info={'type': 'test'})
        y_cont = result['y_cont'].numpy().squeeze()
        
        # Also get the actual clusters
        X, y_true = scm.forward()
        X = X.numpy()
        y_true = y_true.numpy().squeeze().astype(int)
        
        # Apply rank-based assigner
        assigner = ImbalancedMulticlassAssigner(num_classes=10, imbalance_ratio=1.0, mode="rank")
        y_assigned = assigner(result['y_cont']).numpy().squeeze().astype(int)
        
        # Plot continuous values
        ax1 = axes[idx, 0]
        ax1.hist(y_cont, bins=50, alpha=0.7)
        ax1.set_title(f'{name}: Continuous Values Distribution')
        ax1.set_xlabel('Continuous Value')
        ax1.set_ylabel('Frequency')
        
        # Plot true vs assigned labels
        ax2 = axes[idx, 1]
        scatter = ax2.scatter(y_true, y_assigned, alpha=0.5)
        ax2.set_title(f'{name}: True vs Assigned Labels')
        ax2.set_xlabel('True Label')
        ax2.set_ylabel('Assigned Label')
        ax2.plot([0, 9], [0, 9], 'r--', alpha=0.5)  # Perfect correlation line
        
        # PCA visualization with true labels
        ax3 = axes[idx, 2]
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='tab10', alpha=0.6)
        ax3.set_title(f'{name}: PCA with True Labels')
        ax3.set_xlabel('PC1')
        ax3.set_ylabel('PC2')
        plt.colorbar(scatter, ax=ax3)
        
        # Print statistics
        print(f"\n{name} Statistics:")
        print(f"Continuous values - min: {y_cont.min():.3f}, max: {y_cont.max():.3f}, std: {y_cont.std():.3f}")
        print(f"True labels unique: {np.unique(y_true)}")
        print(f"Assigned labels unique: {np.unique(y_assigned)}")
        
        # Check correlation between true and assigned
        correlation = np.corrcoef(y_true, y_assigned)[0, 1]
        print(f"Correlation between true and assigned: {correlation:.3f}")
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_assigned)
        print("Confusion matrix:")
        print(cm)
    
    plt.tight_layout()
    plt.savefig('debug_explicit_clusters.png', dpi=300)
    plt.close()
    
    # Additional plot: continuous values colored by true labels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (scm, name) in enumerate([(scm_original, 'Original'), (scm_fixed, 'Fixed')]):
        # Generate data
        result = scm.forward(info={'type': 'test'})
        y_cont = result['y_cont'].numpy().squeeze()
        X, y_true = scm.forward()
        y_true = y_true.numpy().squeeze().astype(int)
        
        ax = axes[idx]
        for label in range(10):
            mask = y_true == label
            ax.scatter(np.arange(len(y_cont))[mask], y_cont[mask], 
                      alpha=0.6, label=f'Class {label}', s=10)
        
        ax.set_title(f'{name}: Continuous Values by True Class')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Continuous Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('debug_continuous_by_class.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    visualize_continuous_vs_assigned()
    print("\nDebug visualizations saved.")