#!/bin/bash
# Generate synthetic data with mixed clustering approaches - OPTIMIZED VERSION
# This script generates data more efficiently by reducing complexity for slow methods

# cd tabicl
# pip install -e .
# chmod +x scripts/gen_data_mixed_clusters_fast.sh
# ./scripts/gen_data_mixed_clusters_fast.sh

# Function to generate each type of data
generate_dataset() {
    local dataset_type=$1
    local output_dir=$2
    local num_datasets=$3
    local min_classes=$4
    local extra_args="${@:5}"
    
    echo "Generating ${dataset_type} data..."
    echo "Output directory: ${output_dir}"
    echo "Number of datasets: ${num_datasets}"
    echo "Min classes: ${min_classes}"
    echo "Extra arguments: ${extra_args}"
    
    python generate_data.py \
           --n_datasets ${num_datasets} \
           --num_gpus 1 \
           --min_features 20 \
           --max_features 100 \
           --min_seq 5000 \
           --max_seq 20000 \
           --min_classes ${min_classes} \
           --max_classes 10 \
           --replay_small \
           --out_dir ${output_dir} \
           --inner_bsz 256 \
           --save_csv \
           ${extra_args}
}

# 1. Generate GMM clusters data (~85-90% accuracy) - FAST
generate_dataset "GMM_85" \
    "../synth/gmm_clusters_85" \
    250 \
    10 \
    "--prior gmm_clusters_scm \
     --separation_strength 6.5 \
     --balance_strength 0.75"

generate_dataset "GMM_90" \
    "../synth/gmm_clusters_90" \
    250 \
    10 \
    "--prior gmm_clusters_scm \
     --separation_strength 7.5 \
     --balance_strength 0.8"

# 2. Generate explicit clusters data with 10% label noise - FAST
generate_dataset "Explicit_Clusters_Noisy" \
    "../synth/explicit_clusters_noisy" \
    250 \
    10 \
    "--prior real_explicit_clusters_scm \
     --cluster_separation 3.0 \
     --within_cluster_std 0.3 \
     --label_noise 0.1"

# 3. Generate unbalanced multimodal data - OPTIMIZED (reduced layers)
generate_dataset "Multimodal_Piecewise" \
    "../synth/multimodal_piecewise" \
    100 \
    2 \
    "--prior deterministic_tree_scm \
     --no_causal \
     --num_layers 1 \
     --min_swap_prob 0.2 \
     --max_swap_prob 0.35 \
     --transform_type multi_modal \
     --noise_type mixed \
     --noise_std 0.001 \
     --assigner_type piecewise \
     --min_imbalance_ratio 2.5 \
     --max_imbalance_ratio 3.0 \
     --n_estimators 50 \
     --max_depth 3"

# 4. Generate unbalanced polynomial data - OPTIMIZED (reduced complexity)
generate_dataset "Polynomial_Piecewise" \
    "../synth/polynomial_piecewise" \
    100 \
    2 \
    "--prior deterministic_tree_scm \
     --no_causal \
     --num_layers 2 \
     --min_swap_prob 0.15 \
     --max_swap_prob 0.3 \
     --transform_type polynomial \
     --noise_type swap \
     --noise_std 0.005 \
     --assigner_type piecewise \
     --min_imbalance_ratio 2.5 \
     --max_imbalance_ratio 3.0 \
     --n_estimators 30 \
     --max_depth 3"

# 5. Generate balanced polynomial data - OPTIMIZED
generate_dataset "Polynomial_Balanced" \
    "../synth/polynomial_balanced" \
    100 \
    10 \
    "--prior deterministic_tree_scm \
     --no_causal \
     --num_layers 2 \
     --min_swap_prob 0.15 \
     --max_swap_prob 0.3 \
     --transform_type polynomial \
     --noise_type swap \
     --noise_std 0.005 \
     --assigner_type rank \
     --min_imbalance_ratio 1.0 \
     --max_imbalance_ratio 1.0 \
     --n_estimators 30 \
     --max_depth 3"

# 6. Generate mixed transform data - OPTIMIZED
generate_dataset "Mixed_StepFunction" \
    "../synth/mixed_stepfunction" \
    100 \
    5 \
    "--prior deterministic_tree_scm \
     --no_causal \
     --num_layers 1 \
     --min_swap_prob 0.2 \
     --max_swap_prob 0.35 \
     --transform_type mixed \
     --noise_type mixed \
     --noise_std 0.001 \
     --assigner_type step_function \
     --min_imbalance_ratio 1.8 \
     --max_imbalance_ratio 2.0 \
     --n_estimators 50 \
     --max_depth 3"

# Create a summary script to blend datasets
cat > ../synth/blend_datasets.py << 'EOF'
#!/usr/bin/env python3
"""Blend multiple synthetic datasets together."""

import os
import shutil
import random
import glob
from pathlib import Path

def blend_datasets(source_dirs, output_dir, datasets_per_source):
    """Blend datasets from multiple sources."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    all_files = []
    
    for source_dir, num_datasets in zip(source_dirs, datasets_per_source):
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"Warning: {source_dir} does not exist, skipping...")
            continue
            
        # Find all .csv files
        csv_files = list(source_path.glob("*.csv"))
        random.shuffle(csv_files)
        
        # Take the requested number of files
        selected_files = csv_files[:num_datasets]
        
        print(f"Selected {len(selected_files)} files from {source_dir}")
        all_files.extend(selected_files)
    
    # Copy files to output directory with new names
    random.shuffle(all_files)
    
    for idx, source_file in enumerate(all_files):
        new_name = f"dataset_{idx:04d}.csv"
        dest_file = output_path / new_name
        shutil.copy2(source_file, dest_file)
        
        # Also copy H5 file if it exists
        h5_file = source_file.with_suffix('.h5')
        if h5_file.exists():
            dest_h5 = dest_file.with_suffix('.h5')
            shutil.copy2(h5_file, dest_h5)
        
        # Also copy metadata file if it exists
        meta_file = source_file.with_suffix('.meta.json')
        if meta_file.exists():
            dest_meta = dest_file.with_suffix('.meta.json')
            shutil.copy2(meta_file, dest_meta)
    
    print(f"Blended {len(all_files)} datasets into {output_dir}")

if __name__ == "__main__":
    # Configure which datasets to blend and how many from each
    source_dirs = [
        "../synth/gmm_clusters_85",
        "../synth/gmm_clusters_90", 
        "../synth/explicit_clusters_noisy",
        "../synth/multimodal_piecewise",
        "../synth/polynomial_piecewise",
        "../synth/polynomial_balanced",
        "../synth/mixed_stepfunction"
    ]
    
    # Number of datasets to take from each source
    datasets_per_source = [
        200,  # GMM 85%
        200,  # GMM 90%
        200,  # Explicit clusters
        100,  # Multimodal piecewise
        100,  # Polynomial piecewise
        100,  # Polynomial balanced
        100   # Mixed step function
    ]
    
    output_dir = "../synth/blended_clusters_final"
    
    blend_datasets(source_dirs, output_dir, datasets_per_source)
EOF

chmod +x ../synth/blend_datasets.py

echo "All dataset generation scripts created."
echo "To generate all datasets, run this script."
echo "To blend datasets after generation, run: python ../synth/blend_datasets.py"
echo ""
echo "OPTIMIZATIONS APPLIED:"
echo "  - Reduced num_layers for deterministic_tree_scm methods"
echo "  - Added n_estimators and max_depth parameters to limit tree complexity"
echo "  - Reduced dataset count for slow methods to 100 (from 250)"
echo "  - GMM and explicit clusters remain at 250 (they're fast)"
echo ""
echo "Generated datasets will be in:"
echo "  - ../synth/gmm_clusters_85 (250 datasets)"
echo "  - ../synth/gmm_clusters_90 (250 datasets)"
echo "  - ../synth/explicit_clusters_noisy (250 datasets)"
echo "  - ../synth/multimodal_piecewise (100 datasets)"
echo "  - ../synth/polynomial_piecewise (100 datasets)"
echo "  - ../synth/polynomial_balanced (100 datasets)"
echo "  - ../synth/mixed_stepfunction (100 datasets)"
echo ""
echo "Blended dataset will be in:"
echo "  - ../synth/blended_clusters_final (1000 total datasets)"