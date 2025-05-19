"""Test the new multi-modal and mixture transformations for better natural clustering."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tabicl.prior.deterministic_tree_scm import DeterministicTreeSCM
from src.tabicl.prior.imbalanced_assigner import ImbalancedMulticlassAssigner


def visualize_clustering(X, y_continuous, y_classes, transform_type):
    """Visualize how data clusters with different transformations."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{transform_type} Transformation', fontsize=16)
    
    # PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Plot 1: PCA colored by continuous values
    ax = axes[0, 0]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_continuous, 
                        cmap='viridis', alpha=0.6, s=30)
    ax.set_title('PCA - Continuous Values')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.colorbar(scatter, ax=ax)
    
    # Plot 2: PCA colored by assigned classes
    ax = axes[0, 1]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_classes, 
                        cmap='tab10', alpha=0.6, s=30)
    ax.set_title('PCA - Assigned Classes')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    
    # Plot 3: t-SNE visualization
    ax = axes[0, 2]
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X[:500])  # Use subset for speed
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_classes[:500], 
                        cmap='tab10', alpha=0.6, s=30)
    ax.set_title('t-SNE - Assigned Classes')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    # Plot 4: Distribution of continuous values
    ax = axes[1, 0]
    ax.hist(y_continuous, bins=50, alpha=0.7, color='blue')
    ax.set_title('Distribution of Continuous Values')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    
    # Plot 5: Distribution of continuous values by class
    ax = axes[1, 1]
    for class_id in np.unique(y_classes):
        mask = y_classes == class_id
        class_values = y_continuous[mask]
        if len(class_values) > 0 and np.std(class_values) > 0:
            # Use adaptive binning to avoid errors
            try:
                ax.hist(class_values, bins='auto', alpha=0.5, label=f'Class {int(class_id)}')
            except ValueError:
                # Fallback to fewer bins if needed
                ax.hist(class_values, bins=10, alpha=0.5, label=f'Class {int(class_id)}')
    ax.set_title('Continuous Values by Class')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    # Plot 6: Class size distribution
    ax = axes[1, 2]
    class_counts = np.bincount(y_classes.astype(int))
    ax.bar(range(len(class_counts)), class_counts, alpha=0.7)
    ax.set_title('Class Size Distribution')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'{transform_type.lower()}_clustering.png', dpi=150, bbox_inches='tight')
    plt.close()


def test_transform_clustering(transform_type, n_samples=2000, n_features=10, num_classes=5):
    """Test a specific transformation type."""
    print(f"\n=== Testing {transform_type} Transformation ===")
    
    # Create hyperparameters
    hyperparams = {
        'num_features': n_features,
        'num_classes': num_classes,
        'task': 'classification',
        'add_cluster_separation': True,
        'add_component_separation': True,
    }
    
    # Initialize the SCM
    scm = DeterministicTreeSCM(
        seq_len=n_samples,
        num_features=n_features,
        num_outputs=1,
        hyperparams=hyperparams,
        transform_type=transform_type,
        num_causes=n_features // 2,
        max_depth=6,
        tree_model="random_forest",  # Changed from "sklearn" to "random_forest"
        device="cpu"
    )
    
    # Generate data
    X_sampler = lambda: torch.randn(n_samples, n_features) * 2.0
    X = X_sampler()
    
    # Get continuous outputs
    info = {}
    continuous_outputs = scm(X, info)
    y_continuous = continuous_outputs['y_cont'].numpy()
    
    # Convert to classes using rank-based assigner
    assigner = ImbalancedMulticlassAssigner(num_classes, mode="rank")
    y_classes = assigner(continuous_outputs['y_cont']).numpy()
    
    # Debug shapes
    print(f"y_continuous shape: {y_continuous.shape}")
    print(f"y_classes shape: {y_classes.shape}")
    
    # Ensure y_classes is 1D
    if y_classes.ndim > 1:
        y_classes = y_classes.squeeze()
    
    print(f"Continuous values range: [{y_continuous.min():.3f}, {y_continuous.max():.3f}]")
    print(f"Number of unique classes: {len(np.unique(y_classes))}")
    print(f"Class distribution: {np.bincount(y_classes.astype(int))}")
    
    # Ensure proper shapes for visualization
    if y_continuous.ndim > 1:
        y_continuous_squeezed = y_continuous.squeeze()
    else:
        y_continuous_squeezed = y_continuous
        
    # Visualize clustering
    visualize_clustering(X.numpy(), y_continuous_squeezed, y_classes, transform_type)
    
    return y_continuous, y_classes


def main():
    """Test various transformation types for natural clustering."""
    print("Testing Natural Clustering with Different Transformations")
    print("=" * 50)
    
    # Test different transformation types
    transform_types = ['polynomial', 'rbf', 'multi_modal', 'mixture', 'balanced_clusters', 'enhanced_mixture']
    n_samples = 2000
    n_features = 10
    num_classes = 5
    
    results = {}
    
    for transform_type in transform_types:
        try:
            y_cont, y_classes = test_transform_clustering(
                transform_type, n_samples, n_features, num_classes
            )
            results[transform_type] = {
                'y_continuous': y_cont,
                'y_classes': y_classes,
                'class_counts': np.bincount(y_classes.astype(int))
            }
        except Exception as e:
            print(f"Error testing {transform_type}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Comparison of Transformation Types', fontsize=16)
    
    for i, (transform_type, ax) in enumerate(zip(transform_types, axes.flat)):
        if transform_type in results:
            y_cont = results[transform_type]['y_continuous']
            y_classes = results[transform_type]['y_classes']
            
            # Create 2D visualization using first two features
            X_sample = torch.randn(500, n_features)
            scm = DeterministicTreeSCM(
                seq_len=500,
                num_features=n_features,
                num_outputs=1,
                hyperparams={'num_features': n_features, 'num_classes': num_classes},
                transform_type=transform_type,
                num_causes=n_features // 2,
                tree_model="random_forest",
                device="cpu"
            )
            
            info = {}
            outputs = scm(X_sample, info)
            y_cont_sample = outputs['y_cont'].numpy()
            
            # Plot scatter
            ax.scatter(X_sample[:, 0], X_sample[:, 1], 
                      c=y_cont_sample.squeeze(), cmap='viridis', 
                      alpha=0.6, s=30)
            ax.set_title(f'{transform_type}')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('transformation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization files saved:")
    for transform_type in transform_types:
        print(f"- {transform_type.lower()}_clustering.png")
    print("- transformation_comparison.png")


if __name__ == "__main__":
    main()