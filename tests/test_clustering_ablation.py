"""Test the effect of different cluster generating strategies on class separability."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import sys
import os
from itertools import product

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tabicl.prior.imbalanced_assigner import (
    ImbalancedMulticlassAssigner, PiecewiseConstantAssigner,
    RandomRegionAssigner, StepFunctionAssigner, BooleanLogicAssigner
)
from src.tabicl.prior.deterministic_tree_scm import DeterministicTreeSCM
from src.tabicl.prior.utils import XSampler


def visualize_clustering_strategy(X, y_continuous, y_classes, strategy_name, transform_type):
    """Visualize how different strategies create clusters."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{strategy_name} - {transform_type}', fontsize=14)
    
    # Use first two features for visualization
    X_2d = X[:, :2]
    
    # Plot 1: Original continuous values
    ax = axes[0]
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_continuous, 
                        cmap='viridis', alpha=0.6, s=20)
    ax.set_title('Continuous Values')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax)
    
    # Plot 2: Assigned classes
    ax = axes[1]
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_classes, 
                        cmap='tab10', alpha=0.6, s=20)
    ax.set_title('Assigned Classes')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    # Plot 3: Class distribution
    ax = axes[2]
    class_counts = np.bincount(y_classes.astype(int))
    bars = ax.bar(range(len(class_counts)), class_counts, alpha=0.7)
    ax.set_title('Class Distribution')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    
    # Add accuracy annotation
    if hasattr(fig, '_accuracy'):
        fig.text(0.99, 0.01, f'Accuracy: {fig._accuracy:.3f}', 
                ha='right', va='bottom', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{strategy_name.lower()}_{transform_type}_strategy.png', 
                dpi=150, bbox_inches='tight')
    plt.close()


def test_combination(transform_type, assigner_type, n_samples=1000, n_features=10, num_classes=5):
    """Test a specific combination of transformation and assigner."""
    # Create hyperparameters
    hyperparams = {
        'num_features': n_features,
        'num_classes': num_classes,
        'task': 'classification',
        'assigner_type': assigner_type,
        'add_cluster_separation': True,
        'add_component_separation': True,
    }
    
    # Initialize the SCM
    scm = DeterministicTreeSCM(
        hyperparams=hyperparams,
        transform_type=transform_type,
        num_causes=n_features // 2,
        max_depth=6,
        out_dim=1,
        tree_model="sklearn"
    )
    
    # Generate data
    X = torch.randn(n_samples, n_features) * 2.0
    info = {}
    continuous_outputs = scm(X, info)
    y_continuous = continuous_outputs['y_cont'].numpy().squeeze()
    
    # Assign classes based on assigner type
    if assigner_type == "rank":
        assigner = ImbalancedMulticlassAssigner(num_classes, mode="rank")
    elif assigner_type == "value":
        assigner = ImbalancedMulticlassAssigner(num_classes, mode="value")
    elif assigner_type == "piecewise":
        assigner = PiecewiseConstantAssigner(num_classes, max_steps=7)
    elif assigner_type == "random_region":
        assigner = RandomRegionAssigner(num_classes)
    elif assigner_type == "step_function":
        assigner = StepFunctionAssigner(num_classes)
    elif assigner_type == "boolean_logic":
        assigner = BooleanLogicAssigner(num_classes, max_terms=3)
        # For boolean logic, use features directly
        y_classes = assigner(X).numpy()
    else:
        raise ValueError(f"Unknown assigner type: {assigner_type}")
    
    if assigner_type != "boolean_logic":
        y_tensor = torch.from_numpy(y_continuous)
        y_classes = assigner(y_tensor).numpy()
    
    # Test classification
    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), y_classes, test_size=0.2, random_state=42
    )
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, rf.predict(X_test))
    
    return X.numpy(), y_continuous, y_classes, accuracy


def test_all_combinations():
    """Test all combinations of transform types and assigners."""
    print("Testing Cluster Generating Strategy Ablation")
    print("=" * 45)
    
    # Define strategies
    transform_types = ['polynomial', 'rbf', 'multi_modal', 'mixture']
    assigner_types = ['rank', 'value', 'piecewise', 'random_region', 'step_function']
    
    results = {}
    
    # Test all combinations
    for transform_type, assigner_type in product(transform_types, assigner_types):
        combo_name = f"{transform_type}-{assigner_type}"
        print(f"\nTesting {combo_name}...")
        
        try:
            X, y_cont, y_classes, accuracy = test_combination(
                transform_type, assigner_type
            )
            
            results[combo_name] = {
                'accuracy': accuracy,
                'class_distribution': np.bincount(y_classes.astype(int)),
                'num_classes': len(np.unique(y_classes))
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Classes: {results[combo_name]['num_classes']}")
            print(f"  Distribution: {results[combo_name]['class_distribution']}")
            
            # Visualize this combination
            fig = plt.figure()
            fig._accuracy = accuracy  # Attach accuracy for visualization
            visualize_clustering_strategy(X, y_cont, y_classes, combo_name, transform_type)
            
        except Exception as e:
            print(f"  Error: {e}")
            results[combo_name] = {'accuracy': 0.0, 'error': str(e)}
    
    # Create summary heatmap
    plt.figure(figsize=(10, 8))
    
    # Prepare data for heatmap
    accuracy_matrix = np.zeros((len(transform_types), len(assigner_types)))
    for i, transform in enumerate(transform_types):
        for j, assigner in enumerate(assigner_types):
            combo = f"{transform}-{assigner}"
            if combo in results and 'accuracy' in results[combo]:
                accuracy_matrix[i, j] = results[combo]['accuracy']
    
    # Plot heatmap
    im = plt.imshow(accuracy_matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label='Accuracy')
    
    # Set labels
    plt.xticks(range(len(assigner_types)), assigner_types, rotation=45, ha='right')
    plt.yticks(range(len(transform_types)), transform_types)
    plt.xlabel('Assigner Type')
    plt.ylabel('Transform Type')
    plt.title('Classification Accuracy: Transform × Assigner')
    
    # Add text annotations
    for i in range(len(transform_types)):
        for j in range(len(assigner_types)):
            text = plt.text(j, i, f'{accuracy_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black")
    
    plt.tight_layout()
    plt.savefig('clustering_strategy_ablation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create class balance analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Class Balance Analysis by Strategy', fontsize=16)
    
    for idx, transform in enumerate(transform_types):
        ax = axes[idx // 2, idx % 2]
        
        balance_scores = []
        labels = []
        
        for assigner in assigner_types:
            combo = f"{transform}-{assigner}"
            if combo in results and 'class_distribution' in results[combo]:
                dist = results[combo]['class_distribution']
                if len(dist) > 0 and np.min(dist) > 0:
                    # Calculate imbalance ratio (max/min)
                    balance_score = np.max(dist) / np.min(dist)
                else:
                    balance_score = float('inf')
                
                balance_scores.append(balance_score)
                labels.append(assigner)
        
        # Plot balance scores
        bars = ax.bar(range(len(labels)), balance_scores, alpha=0.7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Imbalance Ratio (max/min)')
        ax.set_title(f'{transform} Transform')
        ax.set_ylim(0, min(20, max(balance_scores) * 1.1))
        
        # Color bars by accuracy
        for bar, label in zip(bars, labels):
            combo = f"{transform}-{label}"
            if combo in results:
                accuracy = results[combo].get('accuracy', 0)
                bar.set_color(plt.cm.viridis(accuracy))
    
    plt.tight_layout()
    plt.savefig('class_balance_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "=" * 45)
    print("SUMMARY")
    print("=" * 45)
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), 
                          key=lambda x: x[1].get('accuracy', 0), 
                          reverse=True)
    
    print("\nTop 5 Combinations:")
    for i, (combo, result) in enumerate(sorted_results[:5]):
        if 'accuracy' in result:
            print(f"{i+1}. {combo:25} Accuracy: {result['accuracy']:.4f}")
    
    print("\nBottom 5 Combinations:")
    for i, (combo, result) in enumerate(sorted_results[-5:]):
        if 'accuracy' in result:
            print(f"{i+1}. {combo:25} Accuracy: {result['accuracy']:.4f}")
    
    # Analysis by transform type
    print("\nAverage Accuracy by Transform Type:")
    for transform in transform_types:
        accuracies = [results[f"{transform}-{a}"].get('accuracy', 0) 
                     for a in assigner_types 
                     if f"{transform}-{a}" in results]
        if accuracies:
            print(f"{transform:15} {np.mean(accuracies):.4f}")
    
    # Analysis by assigner type
    print("\nAverage Accuracy by Assigner Type:")
    for assigner in assigner_types:
        accuracies = [results[f"{t}-{assigner}"].get('accuracy', 0) 
                     for t in transform_types 
                     if f"{t}-{assigner}" in results]
        if accuracies:
            print(f"{assigner:15} {np.mean(accuracies):.4f}")
    
    print("\nVisualization files saved:")
    print("- clustering_strategy_ablation.png")
    print("- class_balance_analysis.png")
    print("- Individual combination plots")


if __name__ == "__main__":
    test_all_combinations()