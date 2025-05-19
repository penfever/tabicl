"""Comprehensive test comparing class balance, accuracy, and speed for different clustering strategies."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import time
import sys
import os
from itertools import product
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tabicl.prior.imbalanced_assigner import (
    ImbalancedMulticlassAssigner, PiecewiseConstantAssigner,
    RandomRegionAssigner, StepFunctionAssigner, BooleanLogicAssigner
)
from src.tabicl.prior.deterministic_tree_scm import DeterministicTreeSCM


def calculate_imbalance_ratio(class_distribution):
    """Calculate the imbalance ratio (max/min) for a class distribution."""
    if len(class_distribution) == 0 or np.min(class_distribution) == 0:
        return float('inf')
    return np.max(class_distribution) / np.min(class_distribution)


def calculate_gini_coefficient(class_distribution):
    """Calculate Gini coefficient for class balance (0=perfect balance, 1=perfect imbalance)."""
    sorted_dist = np.sort(class_distribution)
    n = len(sorted_dist)
    cumsum = np.cumsum(sorted_dist)
    return (n + 1 - 2 * np.sum((n - np.arange(n)) * sorted_dist) / cumsum[-1]) / n


def test_configuration(transform_type, assigner_type, n_samples=2000, n_features=10, num_classes=5):
    """Test a specific configuration and return metrics."""
    print(f"  Testing {transform_type}-{assigner_type}...", end='')
    
    try:
        # Time the generation process
        start_time = time.time()
        
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
            seq_len=n_samples,
            num_features=n_features,
            num_outputs=1,
            hyperparams=hyperparams,
            transform_type=transform_type,
            num_causes=n_features // 2,
            max_depth=6,
            tree_model="random_forest",
            device="cpu"
        )
        
        # Generate data
        X = torch.randn(n_samples, n_features) * 2.0
        info = {}
        continuous_outputs = scm(X, info)
        y_continuous = continuous_outputs['y_cont'].numpy()
        
        generation_time = time.time() - start_time
        
        # Ensure proper shape
        if y_continuous.ndim > 1:
            y_continuous = y_continuous.squeeze()
        
        # Assign classes based on assigner type
        start_time = time.time()
        
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
            y_classes = assigner(X).numpy().astype(int)
        else:
            raise ValueError(f"Unknown assigner type: {assigner_type}")
        
        if assigner_type != "boolean_logic":
            y_tensor = torch.from_numpy(y_continuous)
            if y_tensor.ndim == 1:
                y_tensor = y_tensor.unsqueeze(-1)
            y_classes = assigner(y_tensor).numpy()
            # Ensure y_classes is 1D and integer
            if y_classes.ndim > 1:
                y_classes = y_classes.squeeze()
            y_classes = y_classes.astype(int)
        
        assignment_time = time.time() - start_time
        
        # Final shape check
        if y_classes.ndim > 1:
            y_classes = y_classes.squeeze()
            
        # Calculate metrics
        class_distribution = np.bincount(y_classes.astype(int))
        imbalance_ratio = calculate_imbalance_ratio(class_distribution)
        gini_coefficient = calculate_gini_coefficient(class_distribution)
        
        # Test classification accuracy
        start_time = time.time()
        X_train, X_test, y_train, y_test = train_test_split(
            X.numpy(), y_classes, test_size=0.2, random_state=42
        )
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, rf.predict(X_test))
        
        classification_time = time.time() - start_time
        
        # Calculate continuous value statistics
        value_range = y_continuous.max() - y_continuous.min()
        value_std = np.std(y_continuous)
        
        print(" Done!")
        
        return {
            'transform': transform_type,
            'assigner': assigner_type,
            'accuracy': accuracy,
            'imbalance_ratio': imbalance_ratio,
            'gini_coefficient': gini_coefficient,
            'class_distribution': class_distribution,
            'num_classes': len(np.unique(y_classes)),
            'generation_time': generation_time,
            'assignment_time': assignment_time,
            'classification_time': classification_time,
            'total_time': generation_time + assignment_time + classification_time,
            'value_range': value_range,
            'value_std': value_std,
        }
        
    except Exception as e:
        print(f" Error: {e}")
        return {
            'transform': transform_type,
            'assigner': assigner_type,
            'error': str(e),
            'accuracy': 0.0,
            'imbalance_ratio': float('inf'),
            'gini_coefficient': 1.0,
        }


def create_visualizations(results_df):
    """Create comprehensive visualizations of results."""
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Accuracy heatmap
    ax1 = plt.subplot(3, 3, 1)
    pivot_accuracy = results_df.pivot(index='transform', columns='assigner', values='accuracy')
    im1 = ax1.imshow(pivot_accuracy, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(pivot_accuracy.columns)))
    ax1.set_xticklabels(pivot_accuracy.columns, rotation=45, ha='right')
    ax1.set_yticks(range(len(pivot_accuracy.index)))
    ax1.set_yticklabels(pivot_accuracy.index)
    ax1.set_title('Classification Accuracy')
    plt.colorbar(im1, ax=ax1)
    
    # Add text annotations
    for i in range(len(pivot_accuracy.index)):
        for j in range(len(pivot_accuracy.columns)):
            text = ax1.text(j, i, f'{pivot_accuracy.iloc[i, j]:.3f}',
                           ha="center", va="center", color="black")
    
    # 2. Imbalance ratio heatmap
    ax2 = plt.subplot(3, 3, 2)
    pivot_imbalance = results_df.pivot(index='transform', columns='assigner', values='imbalance_ratio')
    # Cap infinite values for visualization
    pivot_imbalance = pivot_imbalance.fillna(10).clip(upper=10)
    im2 = ax2.imshow(pivot_imbalance, cmap='YlOrRd_r', aspect='auto')
    ax2.set_xticks(range(len(pivot_imbalance.columns)))
    ax2.set_xticklabels(pivot_imbalance.columns, rotation=45, ha='right')
    ax2.set_yticks(range(len(pivot_imbalance.index)))
    ax2.set_yticklabels(pivot_imbalance.index)
    ax2.set_title('Imbalance Ratio (lower is better)')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Gini coefficient heatmap
    ax3 = plt.subplot(3, 3, 3)
    pivot_gini = results_df.pivot(index='transform', columns='assigner', values='gini_coefficient')
    im3 = ax3.imshow(pivot_gini, cmap='YlOrRd_r', aspect='auto')
    ax3.set_xticks(range(len(pivot_gini.columns)))
    ax3.set_xticklabels(pivot_gini.columns, rotation=45, ha='right')
    ax3.set_yticks(range(len(pivot_gini.index)))
    ax3.set_yticklabels(pivot_gini.index)
    ax3.set_title('Gini Coefficient (lower is better)')
    plt.colorbar(im3, ax=ax3)
    
    # 4. Speed comparison
    ax4 = plt.subplot(3, 3, 4)
    pivot_time = results_df.pivot(index='transform', columns='assigner', values='total_time')
    im4 = ax4.imshow(pivot_time, cmap='viridis', aspect='auto')
    ax4.set_xticks(range(len(pivot_time.columns)))
    ax4.set_xticklabels(pivot_time.columns, rotation=45, ha='right')
    ax4.set_yticks(range(len(pivot_time.index)))
    ax4.set_yticklabels(pivot_time.index)
    ax4.set_title('Total Time (seconds)')
    plt.colorbar(im4, ax=ax4)
    
    # 5. Best accuracy by transform
    ax5 = plt.subplot(3, 3, 5)
    best_by_transform = results_df.groupby('transform')['accuracy'].max()
    best_by_transform.plot(kind='bar', ax=ax5, color='skyblue')
    ax5.set_title('Best Accuracy by Transform')
    ax5.set_ylabel('Accuracy')
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
    
    # 6. Best balance by transform (lowest imbalance ratio)
    ax6 = plt.subplot(3, 3, 6)
    # Filter out infinite values
    valid_imbalance = results_df[results_df['imbalance_ratio'] != float('inf')]
    if not valid_imbalance.empty:
        best_balance = valid_imbalance.groupby('transform')['imbalance_ratio'].min()
        best_balance.plot(kind='bar', ax=ax6, color='lightgreen')
    ax6.set_title('Best Balance by Transform')
    ax6.set_ylabel('Imbalance Ratio')
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
    
    # 7. Scatter plot: Accuracy vs Balance
    ax7 = plt.subplot(3, 3, 7)
    valid_data = results_df[results_df['imbalance_ratio'] != float('inf')]
    for transform in valid_data['transform'].unique():
        data = valid_data[valid_data['transform'] == transform]
        ax7.scatter(data['imbalance_ratio'], data['accuracy'], 
                   label=transform, alpha=0.7, s=100)
    ax7.set_xlabel('Imbalance Ratio')
    ax7.set_ylabel('Accuracy')
    ax7.set_title('Accuracy vs Balance Trade-off')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Time breakdown
    ax8 = plt.subplot(3, 3, 8)
    time_data = results_df.groupby('transform')[['generation_time', 'assignment_time', 'classification_time']].mean()
    time_data.plot(kind='bar', stacked=True, ax=ax8)
    ax8.set_title('Average Time Breakdown')
    ax8.set_ylabel('Time (seconds)')
    ax8.set_xticklabels(ax8.get_xticklabels(), rotation=45, ha='right')
    ax8.legend(['Generation', 'Assignment', 'Classification'])
    
    # 9. Combined score (accuracy * (1/imbalance_ratio))
    ax9 = plt.subplot(3, 3, 9)
    # Calculate combined score
    valid_results = results_df[results_df['imbalance_ratio'] != float('inf')].copy()
    valid_results['combined_score'] = valid_results['accuracy'] / valid_results['imbalance_ratio']
    pivot_combined = valid_results.pivot(index='transform', columns='assigner', values='combined_score')
    im9 = ax9.imshow(pivot_combined, cmap='viridis', aspect='auto')
    ax9.set_xticks(range(len(pivot_combined.columns)))
    ax9.set_xticklabels(pivot_combined.columns, rotation=45, ha='right')
    ax9.set_yticks(range(len(pivot_combined.index)))
    ax9.set_yticklabels(pivot_combined.index)
    ax9.set_title('Combined Score (Accuracy/Imbalance)')
    plt.colorbar(im9, ax=ax9)
    
    plt.tight_layout()
    plt.savefig('comprehensive_clustering_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create a second figure for class distributions
    fig2, axes = plt.subplots(len(results_df['transform'].unique()), 
                              len(results_df['assigner'].unique()),
                              figsize=(20, 16))
    
    transforms = results_df['transform'].unique()
    assigners = results_df['assigner'].unique()
    
    for i, transform in enumerate(transforms):
        for j, assigner in enumerate(assigners):
            ax = axes[i, j] if len(transforms) > 1 else axes[j]
            
            result = results_df[(results_df['transform'] == transform) & 
                               (results_df['assigner'] == assigner)]
            
            if not result.empty and 'class_distribution' in result.columns:
                class_dist = result.iloc[0]['class_distribution']
                if isinstance(class_dist, np.ndarray):
                    ax.bar(range(len(class_dist)), class_dist, alpha=0.7)
                    ax.set_title(f'{transform}-{assigner}')
                    ax.set_xlabel('Class')
                    ax.set_ylabel('Count')
                    
                    # Add imbalance ratio text
                    imbalance = result.iloc[0]['imbalance_ratio']
                    accuracy = result.iloc[0]['accuracy']
                    ax.text(0.02, 0.98, f'Imb: {imbalance:.2f}\nAcc: {accuracy:.3f}',
                           transform=ax.transAxes, ha='left', va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('class_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()


def run_comprehensive_test():
    """Run the comprehensive test comparing all strategies."""
    print("Comprehensive Clustering Strategy Test")
    print("=" * 40)
    
    # Define test configurations
    transform_types = ['polynomial', 'rbf', 'multi_modal', 'mixture', 'balanced_clusters']
    assigner_types = ['rank', 'value', 'piecewise', 'random_region', 'step_function']
    
    # Storage for results
    all_results = []
    
    print("\nTesting all combinations...")
    print("-" * 40)
    
    # Test all combinations
    for transform, assigner in product(transform_types, assigner_types):
        result = test_configuration(transform, assigner)
        all_results.append(result)
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    # Display summary statistics
    print("\n" + "=" * 40)
    print("SUMMARY STATISTICS")
    print("=" * 40)
    
    # Best accuracy
    best_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
    print(f"\nBest Accuracy: {best_accuracy['accuracy']:.4f}")
    print(f"  Configuration: {best_accuracy['transform']}-{best_accuracy['assigner']}")
    
    # Best balance (lowest imbalance ratio)
    valid_imbalance = results_df[results_df['imbalance_ratio'] != float('inf')]
    if not valid_imbalance.empty:
        best_balance = valid_imbalance.loc[valid_imbalance['imbalance_ratio'].idxmin()]
        print(f"\nBest Balance: {best_balance['imbalance_ratio']:.4f}")
        print(f"  Configuration: {best_balance['transform']}-{best_balance['assigner']}")
    
    # Fastest configuration
    fastest = results_df.loc[results_df['total_time'].idxmin()]
    print(f"\nFastest: {fastest['total_time']:.4f} seconds")
    print(f"  Configuration: {fastest['transform']}-{fastest['assigner']}")
    
    # Best combined (accuracy and balance)
    if not valid_imbalance.empty:
        valid_imbalance['combined_score'] = valid_imbalance['accuracy'] / valid_imbalance['imbalance_ratio']
        best_combined = valid_imbalance.loc[valid_imbalance['combined_score'].idxmax()]
        print(f"\nBest Combined Score: {best_combined['combined_score']:.4f}")
        print(f"  Configuration: {best_combined['transform']}-{best_combined['assigner']}")
        print(f"  Accuracy: {best_combined['accuracy']:.4f}, Imbalance: {best_combined['imbalance_ratio']:.4f}")
    
    # Average statistics by transform
    print("\n" + "-" * 40)
    print("Average Performance by Transform Type")
    print("-" * 40)
    
    transform_stats = results_df.groupby('transform').agg({
        'accuracy': 'mean',
        'imbalance_ratio': lambda x: x[x != float('inf')].mean() if len(x[x != float('inf')]) > 0 else float('inf'),
        'gini_coefficient': 'mean',
        'total_time': 'mean'
    }).round(4)
    print(transform_stats)
    
    # Average statistics by assigner
    print("\n" + "-" * 40)
    print("Average Performance by Assigner Type")
    print("-" * 40)
    
    assigner_stats = results_df.groupby('assigner').agg({
        'accuracy': 'mean',
        'imbalance_ratio': lambda x: x[x != float('inf')].mean() if len(x[x != float('inf')]) > 0 else float('inf'),
        'gini_coefficient': 'mean',
        'total_time': 'mean'
    }).round(4)
    print(assigner_stats)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results_df)
    
    # Save detailed results
    results_df.to_csv('comprehensive_clustering_results.csv', index=False)
    
    print("\nResults saved:")
    print("  - comprehensive_clustering_results.csv")
    print("  - comprehensive_clustering_results.png")
    print("  - class_distributions.png")
    
    return results_df


if __name__ == "__main__":
    results = run_comprehensive_test()