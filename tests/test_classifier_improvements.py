"""Test improvements in classification with new assigners in the full TabICL pipeline."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import tempfile
import subprocess
import os
import sys


def generate_and_test_dataset(assigner_type="rank", transform_type="polynomial"):
    """Generate a dataset with specified assigner and test classification."""
    print(f"\n=== Testing {assigner_type} assigner with {transform_type} transform ===")
    
    # Generate a dataset
    temp_dir = tempfile.mkdtemp(prefix="assigner_test_")
    
    cmd = [
        sys.executable,
        "generate_data.py",
        "--n_datasets", "1",
        "--prior", "deterministic_tree_scm",
        "--num_gpus", "0",
        "--min_features", "19",
        "--max_features", "20",
        "--min_seq", "999",
        "--max_seq", "1000",
        "--min_classes", "5",
        "--max_classes", "5",
        "--class_separability", "4.0",
        "--max_imbalance_ratio", "1.0",
        "--out_dir", temp_dir,
        "--inner_bsz", "1",
        "--no_causal",
        "--num_layers", "2",
        "--min_swap_prob", "0.0",
        "--max_swap_prob", "0.0",
        "--transform_type", transform_type,
        "--noise_type", "swap",
        "--noise_std", "0.01",
        "--assigner_type", assigner_type  # New parameter for assigner type
    ]
    
    # Run from tabicl directory
    tabicl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # First, update generate_data.py to support assigner_type
    add_assigner_support()
    
    result = subprocess.run(cmd, cwd=tabicl_dir, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Command failed with error:")
        print(result.stderr)
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        return None, None
    
    # Load the generated dataset
    files = os.listdir(temp_dir)
    X_file = [f for f in files if f.endswith("_X.npy")][0]
    y_file = [f for f in files if f.endswith("_y.npy")][0]
    
    X = np.load(os.path.join(temp_dir, X_file))
    y = np.load(os.path.join(temp_dir, y_file))
    
    # Test with Random Forest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, rf.predict(X_test))
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Random Forest accuracy: {accuracy:.4f}")
    
    # Visualize with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f'{assigner_type} assigner - {transform_type} transform')
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.savefig(f'{assigner_type}_{transform_type}_pca.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    
    return accuracy, (X, y)


def add_assigner_support():
    """Add assigner_type support to generate_data.py if not already present."""
    generate_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "generate_data.py"
    )
    
    # Read the file
    with open(generate_file, 'r') as f:
        content = f.read()
    
    # Check if assigner_type is already there
    if "--assigner_type" in content:
        return
    
    # Add the argument after --class_separability
    lines = content.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        new_lines.append(line)
        if "--class_separability" in line and "add_argument" in line:
            # Find the end of this argument block (usually 2-3 lines)
            j = i
            while j < len(lines) and not lines[j].strip().startswith('ap.add_argument'):
                j += 1
                if j < len(lines):
                    new_lines.append(lines[j])
                i = j
            # Add our new argument
            new_lines.append('    ap.add_argument("--assigner_type", type=str, default="rank",')
            new_lines.append('                    choices=["rank", "value", "piecewise", "random_region", "step_function", "boolean_logic"],')
            new_lines.append('                    help="type of class assigner for regression to classification conversion")')
            break
    
    # Also add it to the hyperparameter overrides section
    for i, line in enumerate(new_lines):
        if "if args.class_separability" in line:
            new_lines.insert(i + 2, "    if args.assigner_type is not None:")
            new_lines.insert(i + 3, "        hp_overrides['assigner_type'] = args.assigner_type")
            break
    
    # Write back
    with open(generate_file, 'w') as f:
        f.write('\n'.join(new_lines))


def main():
    """Compare different assigners in the full pipeline."""
    print("Testing Class Assigner Improvements in TabICL")
    print("=" * 50)
    
    assigners = ["rank", "piecewise", "random_region", "step_function"]
    transforms = ["polynomial", "rbf"]  # Test with both transforms
    
    results = {}
    
    for transform in transforms:
        for assigner in assigners:
            key = f"{assigner}_{transform}"
            accuracy, data = generate_and_test_dataset(assigner, transform)
            if accuracy is not None:
                results[key] = accuracy
    
    # Summary plot
    if results:
        plt.figure(figsize=(12, 8))
        
        # Group by transform type
        transform_groups = {}
        for key, acc in results.items():
            assigner, transform = key.rsplit('_', 1)
            if transform not in transform_groups:
                transform_groups[transform] = {}
            transform_groups[transform][assigner] = acc
        
        # Plot grouped bars
        assigners_list = list(set(key.rsplit('_', 1)[0] for key in results.keys()))
        x = np.arange(len(assigners_list))
        width = 0.35
        
        for i, (transform, data) in enumerate(transform_groups.items()):
            accuracies = [data.get(assigner, 0) for assigner in assigners_list]
            offset = width * (i - 0.5)
            bars = plt.bar(x + offset, accuracies, width, label=f'{transform} transform', alpha=0.8)
            
            # Add value labels
            for bar, acc in zip(bars, accuracies):
                if acc > 0:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{acc:.3f}', ha='center', va='bottom')
        
        plt.xlabel('Assigner Type')
        plt.ylabel('Classification Accuracy')
        plt.title('Performance Comparison: Assigner Types with Different Transforms')
        plt.xticks(x, assigners_list)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('assigner_transform_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print summary
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"{'Configuration':30} {'Accuracy':>10}")
        print("-" * 42)
        for key, accuracy in sorted(results.items()):
            print(f"{key:30} {accuracy:>10.4f}")
        
        print("\nVisualization files saved:")
        print("- assigner_transform_comparison.png")
        for key in results.keys():
            print(f"- {key}_pca.png")


if __name__ == "__main__":
    main()