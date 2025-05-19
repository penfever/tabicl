"""
Comparative analysis between deterministic_tree_scm and scikit-learn's make_classification
to understand why synthetic datasets from deterministic_tree_scm are harder to classify.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import tempfile
import subprocess
import os
import sys


def analyze_sklearn_make_classification():
    """Analyze datasets from scikit-learn's make_classification"""
    print("\n=== Analyzing scikit-learn's make_classification ===")
    
    # Parameters similar to your example
    complex_params = {
        'n_samples': 3000,
        'n_features': 25,
        'n_informative': 10,
        'n_redundant': 2,
        'n_repeated': 0,
        'n_classes': 10,
        'n_clusters_per_class': 2,
        'class_sep': 4.0,
        'random_state': 42
    }
    
    X, y = make_classification(**complex_params)
    
    # Analyze features
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Feature statistics:")
    print(f"  Mean: {np.mean(X):.4f}, Std: {np.std(X):.4f}")
    print(f"  Min: {np.min(X):.4f}, Max: {np.max(X):.4f}")
    
    # Test with Random Forest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, rf.predict(X_test))
    print(f"Random Forest accuracy: {accuracy:.4f}")
    
    # Visualize with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('make_classification - PCA visualization')
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.savefig('make_classification_pca.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return X, y, accuracy


def analyze_deterministic_tree_scm(transform_type="polynomial", separability=4.0):
    """Analyze datasets from deterministic_tree_scm"""
    print(f"\n=== Analyzing deterministic_tree_scm ({transform_type}) ===")
    
    # Generate a dataset
    temp_dir = tempfile.mkdtemp(prefix="analysis_")
    
    cmd = [
        sys.executable,
        "generate_data.py",
        "--n_datasets", "1",
        "--prior", "deterministic_tree_scm",
        "--num_gpus", "0",
        "--min_features", "24",
        "--max_features", "25",
        "--min_seq", "2999",
        "--max_seq", "3000",
        "--min_classes", "10",
        "--max_classes", "10",
        "--class_separability", str(separability),
        "--max_imbalance_ratio", "1.0",
        "--out_dir", temp_dir,
        "--inner_bsz", "32",
        "--no_causal",  # Add this to match the original test
        "--num_layers", "1",
        "--min_swap_prob", "0.0",
        "--max_swap_prob", "0.0",
        "--transform_type", transform_type,
        "--noise_type", "swap",
        "--noise_std", "0.001"
    ]
    
    # Run from tabicl directory
    tabicl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(cmd, cwd=tabicl_dir, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Command failed with error:")
        print(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    
    # Load the generated dataset
    files = os.listdir(temp_dir)
    X_file = [f for f in files if f.endswith("_X.npy")][0]
    y_file = [f for f in files if f.endswith("_y.npy")][0]
    
    X = np.load(os.path.join(temp_dir, X_file))
    y = np.load(os.path.join(temp_dir, y_file))
    
    # Analyze features
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Feature statistics:")
    print(f"  Mean: {np.mean(X):.4f}, Std: {np.std(X):.4f}")
    print(f"  Min: {np.min(X):.4f}, Max: {np.max(X):.4f}")
    
    # Test with Random Forest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, rf.predict(X_test))
    print(f"Random Forest accuracy: {accuracy:.4f}")
    
    # Visualize with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f'deterministic_tree_scm ({transform_type}) - PCA visualization')
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.savefig(f'deterministic_tree_scm_{transform_type}_pca.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    
    return X, y, accuracy


def compare_feature_distributions(X_sklearn, y_sklearn, X_tree_scm, y_tree_scm):
    """Compare feature distributions between the two methods"""
    print("\n=== Comparing Feature Distributions ===")
    
    # Feature correlations with target
    from scipy.stats import spearmanr
    
    print("\nFeature-target correlations:")
    sklearn_corrs = []
    tree_scm_corrs = []
    
    for i in range(min(X_sklearn.shape[1], X_tree_scm.shape[1])):
        sklearn_corr = abs(spearmanr(X_sklearn[:, i], y_sklearn)[0])
        tree_scm_corr = abs(spearmanr(X_tree_scm[:, i], y_tree_scm)[0])
        sklearn_corrs.append(sklearn_corr)
        tree_scm_corrs.append(tree_scm_corr)
    
    print(f"make_classification - Mean correlation: {np.mean(sklearn_corrs):.4f}")
    print(f"deterministic_tree_scm - Mean correlation: {np.mean(tree_scm_corrs):.4f}")
    
    # Plot feature distributions for first few features
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i in range(3):
        # Sklearn features
        axes[0, i].hist(X_sklearn[:, i], bins=50, alpha=0.7, label='make_classification')
        axes[0, i].set_title(f'Feature {i} - make_classification')
        axes[0, i].set_xlabel('Value')
        axes[0, i].set_ylabel('Frequency')
        
        # Tree SCM features
        axes[1, i].hist(X_tree_scm[:, i], bins=50, alpha=0.7, label='tree_scm', color='orange')
        axes[1, i].set_title(f'Feature {i} - deterministic_tree_scm')
        axes[1, i].set_xlabel('Value')
        axes[1, i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot correlation histograms
    plt.figure(figsize=(10, 6))
    plt.hist(sklearn_corrs, bins=20, alpha=0.7, label='make_classification')
    plt.hist(tree_scm_corrs, bins=20, alpha=0.7, label='deterministic_tree_scm')
    plt.xlabel('Feature-target correlation')
    plt.ylabel('Number of features')
    plt.title('Distribution of feature-target correlations')
    plt.legend()
    plt.savefig('correlation_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()


def analyze_class_overlap(X, y, method_name):
    """Analyze class overlap in feature space"""
    from sklearn.neighbors import NearestNeighbors
    
    print(f"\n=== Analyzing class overlap for {method_name} ===")
    
    # For each class, find k-nearest neighbors and check their labels
    n_neighbors = 20
    overlap_scores = []
    
    for class_label in range(10):
        # Get samples from this class
        class_mask = y == class_label
        X_class = X[class_mask]
        
        if len(X_class) < n_neighbors:
            continue
            
        # Find nearest neighbors for each sample in this class
        nn = NearestNeighbors(n_neighbors=n_neighbors+1)  # +1 because it includes itself
        nn.fit(X)
        
        for sample in X_class[:50]:  # Check first 50 samples of each class
            distances, indices = nn.kneighbors([sample])
            neighbor_labels = y[indices[0][1:]]  # Exclude the sample itself
            
            # Calculate what fraction of neighbors are from the same class
            same_class_ratio = np.mean(neighbor_labels == class_label)
            overlap_scores.append(same_class_ratio)
    
    mean_overlap = np.mean(overlap_scores)
    print(f"Mean same-class neighbor ratio: {mean_overlap:.4f}")
    print(f"(Higher values indicate better class separation)")
    
    return mean_overlap


def main():
    """Run the comparative analysis"""
    print("Comparative Analysis: make_classification vs deterministic_tree_scm")
    print("=" * 60)
    
    # Analyze make_classification
    X_sklearn, y_sklearn, acc_sklearn = analyze_sklearn_make_classification()
    
    # Analyze deterministic_tree_scm with polynomial transformation
    X_tree_scm_poly, y_tree_scm_poly, acc_tree_scm_poly = analyze_deterministic_tree_scm(
        transform_type="polynomial", separability=4.0)
    
    # Analyze deterministic_tree_scm with RBF transformation
    X_tree_scm_rbf, y_tree_scm_rbf, acc_tree_scm_rbf = analyze_deterministic_tree_scm(
        transform_type="rbf", separability=4.0)
    
    # Compare distributions
    compare_feature_distributions(X_sklearn, y_sklearn, X_tree_scm_poly, y_tree_scm_poly)
    
    # Analyze class overlap
    overlap_sklearn = analyze_class_overlap(X_sklearn, y_sklearn, "make_classification")
    overlap_tree_scm_poly = analyze_class_overlap(X_tree_scm_poly, y_tree_scm_poly, "deterministic_tree_scm (polynomial)")
    overlap_tree_scm_rbf = analyze_class_overlap(X_tree_scm_rbf, y_tree_scm_rbf, "deterministic_tree_scm (rbf)")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"make_classification:")
    print(f"  Accuracy: {acc_sklearn:.4f}")
    print(f"  Class separation: {overlap_sklearn:.4f}")
    print(f"\ndeterministic_tree_scm (polynomial):")
    print(f"  Accuracy: {acc_tree_scm_poly:.4f}")
    print(f"  Class separation: {overlap_tree_scm_poly:.4f}")
    print(f"\ndeterministic_tree_scm (rbf):")
    print(f"  Accuracy: {acc_tree_scm_rbf:.4f}")
    print(f"  Class separation: {overlap_tree_scm_rbf:.4f}")
    
    print("\nKey Insights:")
    print("1. Feature-target correlations")
    print("2. Class overlap in feature space")
    print("3. Feature distributions")
    print("\nVisualization files saved:")
    print("- make_classification_pca.png")
    print("- deterministic_tree_scm_polynomial_pca.png")
    print("- deterministic_tree_scm_rbf_pca.png")
    print("- feature_distributions.png")
    print("- correlation_distributions.png")


if __name__ == "__main__":
    main()