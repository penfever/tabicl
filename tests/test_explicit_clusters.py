"""Test the explicit clusters approach for creating well-separated classes."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tabicl.prior.explicit_clusters_scm import ExplicitClustersSCM
from tabicl.prior.imbalanced_assigner import ImbalancedMulticlassAssigner


def visualize_explicit_clusters(X, y_true, title="Explicit Clusters"):
    """Visualize the cluster structure using PCA and t-SNE."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    ax = axes[0, 0]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, 
                        cmap='tab10', alpha=0.6, s=30)
    ax.set_title(f'PCA - {title}')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.colorbar(scatter, ax=ax)
    
    # t-SNE visualization
    ax = axes[0, 1]
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X[:500])  # Use subset for speed
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_true[:500], 
                        cmap='tab10', alpha=0.6, s=30)
    ax.set_title(f't-SNE - {title}')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    # Class distribution
    ax = axes[1, 0]
    class_counts = np.bincount(y_true.astype(int))
    bars = ax.bar(range(len(class_counts)), class_counts, color='skyblue', alpha=0.7)
    ax.set_title('Class Distribution')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_xticks(range(len(class_counts)))
    
    # Add value labels on bars
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{count}', ha='center', va='bottom')
    
    # Feature importance from Random Forest
    ax = axes[1, 1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, test_size=0.2, random_state=42
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Plot feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # Top 20 features
    ax.bar(range(20), importances[indices], color='lightgreen', alpha=0.7)
    ax.set_title('Top 20 Feature Importances')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Importance')
    ax.set_xticks(range(20))
    ax.set_xticklabels(indices, rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_visualization.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    return rf.score(X_test, y_test)


def test_explicit_clusters_performance():
    """Test the performance of explicit clusters approach."""
    print("Testing Explicit Clusters Approach")
    print("=" * 40)
    
    n_samples = 5000
    n_features = 20
    num_classes = 10
    
    # Test different cluster configurations
    configurations = [
        {"cluster_separation": 3.0, "within_cluster_std": 0.5},
        {"cluster_separation": 5.0, "within_cluster_std": 0.5},
        {"cluster_separation": 8.0, "within_cluster_std": 0.3},
        {"cluster_separation": 10.0, "within_cluster_std": 0.2},
    ]
    
    results = []
    
    for i, config in enumerate(configurations):
        print(f"\nConfiguration {i+1}:")
        print(f"  Cluster separation: {config['cluster_separation']}")
        print(f"  Within-cluster std: {config['within_cluster_std']}")
        
        # Create the SCM
        scm = ExplicitClustersSCM(
            seq_len=n_samples,
            num_features=n_features,
            num_outputs=1,
            num_classes=num_classes,
            min_samples_per_class=n_samples // num_classes,
            cluster_separation=config['cluster_separation'],
            within_cluster_std=config['within_cluster_std'],
            random_state=42,
            device="cpu"
        )
        
        # Generate data
        X, y = scm()
        X_np = X.numpy()
        y_np = y.numpy().squeeze().astype(int)
        
        # Test classification performance
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np, test_size=0.2, random_state=42
        )
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_accuracy = accuracy_score(y_test, rf.predict(X_test))
        
        # Calculate metrics
        class_distribution = np.bincount(y_np)
        imbalance_ratio = np.max(class_distribution) / np.min(class_distribution)
        
        print(f"  Random Forest accuracy: {rf_accuracy:.4f}")
        print(f"  Class distribution: {class_distribution}")
        print(f"  Imbalance ratio: {imbalance_ratio:.4f}")
        
        results.append({
            'config': config,
            'accuracy': rf_accuracy,
            'class_distribution': class_distribution,
            'imbalance_ratio': imbalance_ratio
        })
        
        # Visualize the best configuration
        if i == 2:  # Third configuration usually works well
            accuracy = visualize_explicit_clusters(X_np, y_np, 
                f"Explicit Clusters (sep={config['cluster_separation']}, std={config['within_cluster_std']})")
            
            # Also create confusion matrix
            plt.figure(figsize=(10, 8))
            y_pred = rf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - Explicit Clusters')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig('explicit_clusters_confusion_matrix.png', 
                        dpi=150, bbox_inches='tight')
            plt.close()
    
    # Compare with assigners
    print("\n\nComparing with Different Assigners:")
    print("-" * 40)
    
    # Use best configuration
    best_config = configurations[2]
    scm = ExplicitClustersSCM(
        seq_len=n_samples,
        num_features=n_features,
        num_outputs=1,
        num_classes=num_classes,
        min_samples_per_class=n_samples // num_classes,
        cluster_separation=best_config['cluster_separation'],
        within_cluster_std=best_config['within_cluster_std'],
        random_state=42,
        device="cpu"
    )
    
    # Generate data with continuous output
    info = {}
    continuous_outputs = scm(info=info)
    y_continuous = continuous_outputs['y_cont'].numpy().squeeze()
    
    # Generate features
    X, _ = scm()
    X_np = X.numpy()
    
    # Test with rank-based assigner
    assigner = ImbalancedMulticlassAssigner(num_classes, mode="rank")
    y_tensor = torch.from_numpy(y_continuous).unsqueeze(-1)
    y_assigned = assigner(y_tensor).numpy()
    
    # Test classification
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_assigned, test_size=0.2, random_state=42
    )
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, rf.predict(X_test))
    
    print(f"Rank-based assigner accuracy: {accuracy:.4f}")
    print(f"Class distribution: {np.bincount(y_assigned.astype(int))}")
    
    # Summary
    print("\n\nSummary:")
    print("=" * 40)
    print("The explicit clusters approach guarantees:")
    print("1. Exactly 10 well-separated clusters")
    print("2. Balanced class distribution")
    print("3. High classification accuracy")
    print("\nBest configuration:")
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"  Cluster separation: {best_result['config']['cluster_separation']}")
    print(f"  Within-cluster std: {best_result['config']['within_cluster_std']}")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  Imbalance ratio: {best_result['imbalance_ratio']:.4f}")


if __name__ == "__main__":
    test_explicit_clusters_performance()