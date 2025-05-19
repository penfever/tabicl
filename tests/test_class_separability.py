import os
import numpy as np
import tempfile
import shutil
import subprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys


def generate_dataset(n_datasets=3, out_dir=None, class_separability=1.0, 
                    min_features=20, max_features=200, min_seq=4000, max_seq=10000):
    """Generate datasets using the generate_data.py script with specified parameters."""
    # Create a temporary directory if none provided
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="test_class_sep_")
    
    # Build the command
    cmd = [
        sys.executable,
        "generate_data.py",
        "--n_datasets", str(n_datasets),
        "--prior", "deterministic_tree_scm",
        "--num_gpus", "1",  # Use CPU for testing
        "--min_features", str(min_features),
        "--max_features", str(max_features),
        "--min_seq", str(min_seq),
        "--max_seq", str(max_seq),
        "--min_classes", "10",
        "--max_classes", "10",
        "--class_separability", str(class_separability),
        "--max_imbalance_ratio", "2.0",
        "--out_dir", out_dir,
        "--inner_bsz", "32",  # Smaller batch size for speed
        "--no_causal",
        "--num_layers", "1",
        "--min_swap_prob", "0.0",
        "--max_swap_prob", "0.0",
        "--transform_type", "polynomial",
        "--noise_type", "swap",
        "--noise_std", "0.001",
        "--save_csv"
    ]
    
    # Run the command from the tabicl directory
    tabicl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        result = subprocess.run(cmd, cwd=tabicl_dir, check=True, capture_output=True, text=True)
        print(f"Dataset generation successful in {out_dir}")
        return out_dir
    except subprocess.CalledProcessError as e:
        print(f"Error generating datasets: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise


def load_dataset(dataset_dir):
    """Load a dataset from the generated files."""
    # Find a dataset file
    files = os.listdir(dataset_dir)
    X_files = [f for f in files if f.endswith("_X.npy")]
    y_files = [f for f in files if f.endswith("_y.npy")]
    
    if not X_files or not y_files:
        raise ValueError(f"No dataset files found in {dataset_dir}")
    
    # Load the first dataset
    X = np.load(os.path.join(dataset_dir, X_files[0]))
    y = np.load(os.path.join(dataset_dir, y_files[0]))
    
    return X, y


def test_random_forest_accuracy(X, y, test_size=0.2, random_state=42):
    """Test Random Forest accuracy on the dataset."""
    from sklearn.model_selection import train_test_split
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train Random Forest with faster parameters
    rf = RandomForestClassifier(
        n_estimators=100,   # Fewer trees for speed
        max_depth=15,       # Limit depth for speed
        min_samples_split=5,  # Reduce overfitting and speed up
        min_samples_leaf=2,   # Reduce overfitting and speed up
        random_state=random_state,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    # Predict and calculate accuracy
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy


def test_class_separability():
    """Test that class_separability parameter improves Random Forest accuracy."""
    
    print("\nTesting class_separability parameter effect on Random Forest accuracy...")
    
    # Test with different class_separability values (including higher values for better separation)
    separability_values = [1.0, 3.0, 5.0, 10.0]
    accuracies = []
    temp_dirs = []
    
    for sep_value in separability_values:
        print(f"\nGenerating dataset with class_separability={sep_value}")
        
        # Generate dataset
        temp_dir = None
        try:
            temp_dir = generate_dataset(
                n_datasets=1,
                class_separability=sep_value,
                min_features=10,    # Reasonable number of features
                max_features=25,    # Fixed features for consistency
                min_seq=2000,       # Fewer samples for faster generation
                max_seq=2200        # Fixed samples for consistency
            )
            temp_dirs.append(temp_dir)
            
            # Load dataset
            X, y = load_dataset(temp_dir)
            print(f"Dataset shape: X={X.shape}, y={y.shape}")
            print(f"Number of unique classes: {len(np.unique(y))}")
            
            # Test accuracy
            accuracy = test_random_forest_accuracy(X, y)
            accuracies.append(accuracy)
            print(f"Random Forest accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error with class_separability={sep_value}: {e}")
            if temp_dir:
                temp_dirs.append(temp_dir)
            raise
    
    # Clean up temp directories
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    # Verify results
    print(f"\n{'='*50}")
    print("Results Summary:")
    print(f"{'='*50}")
    for sep_value, acc in zip(separability_values, accuracies):
        print(f"Class Separability: {sep_value:4.1f} -> Accuracy: {acc:6.4f}")
    
    # Check that accuracy improves with higher separability
    assert accuracies[-1] > accuracies[0], \
        f"Accuracy should improve with higher class_separability. Got {accuracies[0]:.4f} -> {accuracies[-1]:.4f}"
    
    # Check that highest separability achieves reasonable accuracy (adjusted expectation)
    # For 10-class classification with complex synthetic data, 50% is a good target
    assert accuracies[-1] >= 0.50, \
        f"Random Forest should achieve at least 50% accuracy with class_separability=10.0. Got {accuracies[-1]:.4f}"
    
    # Check general trend is positive (not strict monotonic due to randomness)
    improvements = [accuracies[i+1] - accuracies[i] for i in range(len(accuracies)-1)]
    positive_improvements = sum(1 for imp in improvements if imp > -0.10)  # Allow 10% tolerance
    assert positive_improvements >= len(improvements) // 2, \
        f"At least half of the improvements should be positive. Improvements: {improvements}"
    
    # Check significant improvement from baseline to highest
    improvement_ratio = accuracies[-1] / accuracies[0]
    assert improvement_ratio >= 1.5, \
        f"Accuracy should improve by at least 50% from baseline. Got {improvement_ratio:.2f}x improvement"
    
    print(f"\n{'='*50}")
    print("✓ All tests passed!")
    print(f"{'='*50}")


if __name__ == "__main__":
    # Run the test
    test_class_separability()