"""Final comprehensive test for all clustering approaches with detailed visualization."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import pandas as pd
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tabicl.prior.deterministic_tree_scm import DeterministicTreeSCM
from tabicl.prior.explicit_clusters_scm import ExplicitClustersSCM
from tabicl.prior.direct_clusters_scm import DirectClustersSCM
from tabicl.prior.gmm_clusters_scm import GMMClustersSCM
from tabicl.prior.imbalanced_assigner import RankBasedAssigner


def create_visualization(X, y, method_name, accuracy, num_classes_found):
    """Create a comprehensive visualization for each method."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'{method_name} - Accuracy: {accuracy:.3f}, Classes Found: {num_classes_found}/10', fontsize=16)
    
    # 1. PCA plot
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    ax1 = axes[0, 0]
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6)
    ax1.set_title('PCA Visualization')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    plt.colorbar(scatter, ax=ax1)
    
    # 2. TSNE plot
    if X.shape[0] > 50:  # TSNE is slow for large datasets
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0]-1))
        X_tsne = tsne.fit_transform(X[:1000])  # Limit to 1000 samples
        y_subset = y[:1000]
    else:
        X_tsne = X_pca
        y_subset = y
    
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_subset, cmap='tab10', alpha=0.6)
    ax2.set_title('t-SNE Visualization')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    plt.colorbar(scatter2, ax=ax2)
    
    # 3. Class distribution
    ax3 = axes[0, 2]
    unique, counts = np.unique(y, return_counts=True)
    bars = ax3.bar(unique, counts, color=plt.cm.tab10(unique/10))
    ax3.set_title('Class Distribution')
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Count')
    ax3.set_xticks(unique)
    
    # 4. Feature correlation heatmap (first 10 features)
    ax4 = axes[1, 0]
    corr_matrix = np.corrcoef(X[:, :min(10, X.shape[1])].T)
    sns.heatmap(corr_matrix, ax=ax4, cmap='coolwarm', center=0)
    ax4.set_title('Feature Correlation (first 10)')
    
    # 5. Feature importance
    ax5 = axes[1, 1]
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_[:20]  # First 20 features
    ax5.bar(range(len(importances)), importances)
    ax5.set_title('Feature Importance (first 20)')
    ax5.set_xlabel('Feature Index')
    ax5.set_ylabel('Importance')
    
    # 6. Confusion matrix (subsample)
    ax6 = axes[1, 2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    # Normalize to percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues', ax=ax6)
    ax6.set_title('Normalized Confusion Matrix')
    ax6.set_xlabel('Predicted')
    ax6.set_ylabel('True')
    
    plt.tight_layout()
    
    # Save to specific directory
    output_dir = Path('/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/tabular-fm-llm/data/clustering_expts')
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / f'{method_name}_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def test_method(method_name, scm_class, scm_kwargs=None, use_rank_assigner=True):
    """Test a specific method and return metrics."""
    print(f"\nTesting {method_name}...")
    
    if scm_kwargs is None:
        scm_kwargs = {}
    
    # Create SCM
    scm = scm_class(
        seq_len=2000,
        num_features=50,
        num_outputs=1,
        num_classes=10,
        **scm_kwargs
    )
    
    # Generate data
    if use_rank_assigner:
        # For methods that return continuous values
        result = scm.forward(info={'type': 'test'})
        y_cont = result['y_cont']
        X, _ = scm.forward()
        
        # Apply rank-based assigner
        assigner = RankBasedAssigner(num_classes=10)
        y = assigner(y_cont).numpy().squeeze().astype(int)
    else:
        # For methods that return discrete labels directly
        X, y = scm.forward()
        y = y.numpy().squeeze().astype(int)
    
    X = X.numpy()
    
    # Compute metrics
    unique, counts = np.unique(y, return_counts=True)
    num_classes_found = len(unique)
    balance_ratio = max(counts) / min(counts) if len(counts) > 0 else float('inf')
    
    # Train classifier
    if len(unique) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    else:
        accuracy = 0.0
    
    # Create visualization
    create_visualization(X, y, method_name, accuracy, num_classes_found)
    
    # Print results
    print(f"  Classes found: {num_classes_found}/10")
    print(f"  Balance ratio: {balance_ratio:.2f}")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Class distribution: {dict(zip(unique, counts))}")
    
    return {
        'method': method_name,
        'accuracy': accuracy,
        'num_classes': num_classes_found,
        'balance_ratio': balance_ratio,
        'class_distribution': dict(zip(unique, counts))
    }


def main():
    """Test all methods and compile results."""
    results = []
    
    # Test different methods
    
    # 1. Direct Clusters (new method)
    results.append(test_method(
        'direct_clusters',
        DirectClustersSCM,
        {'cluster_separation': 3.0, 'within_cluster_std': 0.2},
        use_rank_assigner=False
    ))
    
    # 2. Explicit Clusters (modified)
    results.append(test_method(
        'explicit_clusters_v2',
        ExplicitClustersSCM,
        {'cluster_separation': 2.0, 'within_cluster_std': 0.3},
        use_rank_assigner=True
    ))
    
    # 3. DeterministicTreeSCM with RBF
    results.append(test_method(
        'tree_rbf',
        DeterministicTreeSCM,
        {
            'transform_type': 'rbf',
            'class_separability': 2.0,
            'hyperparams': {'num_classes': 10}
        },
        use_rank_assigner=True
    ))
    
    # 4. DeterministicTreeSCM with multi_modal
    results.append(test_method(
        'tree_multi_modal',
        DeterministicTreeSCM,
        {
            'transform_type': 'multi_modal',
            'class_separability': 2.0,
            'hyperparams': {'num_classes': 10}
        },
        use_rank_assigner=True
    ))
    
    # 5. DeterministicTreeSCM with balanced_clusters
    results.append(test_method(
        'tree_balanced_clusters',
        DeterministicTreeSCM,
        {
            'transform_type': 'balanced_clusters',
            'class_separability': 2.0,
            'hyperparams': {'num_classes': 10}
        },
        use_rank_assigner=True
    ))
    
    # 6. GMM Clusters
    results.append(test_method(
        'gmm_clusters',
        GMMClustersSCM,
        {
            'n_components': 10,
            'covariance_type': 'full',
            'balanced': True
        },
        use_rank_assigner=False
    ))
    
    # Create summary plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = [r['method'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    num_classes = [r['num_classes'] for r in results]
    balance_ratios = [np.log10(r['balance_ratio']) if r['balance_ratio'] < 1000 else 3 for r in results]
    
    # Accuracy plot
    axes[0].bar(methods, accuracies)
    axes[0].set_title('Classification Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xticklabels(methods, rotation=45)
    
    # Number of classes plot
    axes[1].bar(methods, num_classes)
    axes[1].axhline(10, color='red', linestyle='--', label='Target')
    axes[1].set_title('Number of Classes Found')
    axes[1].set_ylabel('Classes')
    axes[1].set_xticklabels(methods, rotation=45)
    axes[1].legend()
    
    # Balance ratio plot (log scale)
    axes[2].bar(methods, balance_ratios)
    axes[2].set_title('Log Balance Ratio')
    axes[2].set_ylabel('Log10(Max/Min)')
    axes[2].set_xticklabels(methods, rotation=45)
    
    plt.tight_layout()
    
    output_dir = Path('/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/tabular-fm-llm/data/clustering_expts')
    plt.savefig(output_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'final_results.csv', index=False)
    
    print("\n\nSummary:")
    print(df.to_string())
    
    print("\nVisualization files saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()