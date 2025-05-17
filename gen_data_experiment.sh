#!/bin/bash
# Experimental data generation with varying noise levels and causal structures
# Use this to test which settings produce the most learnable data

# cd tabicl
# pip install -e .
# chmod +x gen_data_experiment.sh
# ./gen_data_experiment.sh

# Configuration 1: Direct mapping with minimal noise
echo "Config 1: Direct mapping, minimal noise..."
python generate_data.py \
       --n_datasets 5 \
       --prior mlp_scm \
       --min_features 20 \
       --max_features 100 \
       --min_seq 1000 \
       --max_seq 5000 \
       --max_classes 5 \
       --out_dir ../synth/experiment/direct_low_noise \
       --inner_bsz 64 \
       --no_causal \
       --y_is_effect \
       --in_clique \
       --num_layers 2 \
       --num_causes 10 \
       --noise_std 0.0001 \
       --no_pre_sample_noise_std

# Configuration 2: Causal with strong dependencies
echo "Config 2: Causal mode, strong dependencies..."
python generate_data.py \
       --n_datasets 5 \
       --prior mlp_scm \
       --min_features 20 \
       --max_features 100 \
       --min_seq 1000 \
       --max_seq 5000 \
       --max_classes 5 \
       --out_dir ../synth/experiment/causal_strong_deps \
       --inner_bsz 64 \
       --is_causal \
       --y_is_effect \
       --in_clique \
       --num_layers 3 \
       --num_causes 15 \
       --noise_std 0.001 \
       --no_pre_sample_noise_std

# Configuration 3: Moderate noise, fewer layers
echo "Config 3: Moderate noise, simple structure..."
python generate_data.py \
       --n_datasets 5 \
       --prior mlp_scm \
       --min_features 20 \
       --max_features 100 \
       --min_seq 1000 \
       --max_seq 5000 \
       --max_classes 5 \
       --out_dir ../synth/experiment/moderate_simple \
       --inner_bsz 64 \
       --is_causal \
       --y_is_effect \
       --in_clique \
       --num_layers 2 \
       --num_causes 20 \
       --noise_std 0.005 \
       --no_pre_sample_noise_std

# Configuration 4: Tree-based with low noise
echo "Config 4: Tree-based, low noise..."
python generate_data.py \
       --n_datasets 5 \
       --prior tree_scm \
       --min_features 20 \
       --max_features 100 \
       --min_seq 1000 \
       --max_seq 5000 \
       --max_classes 5 \
       --out_dir ../synth/experiment/tree_low_noise \
       --inner_bsz 64 \
       --is_causal \
       --y_is_effect \
       --in_clique \
       --num_layers 2 \
       --num_causes 15 \
       --noise_std 0.001 \
       --no_pre_sample_noise_std

# Configuration 5: No noise baseline (sanity check)
echo "Config 5: No noise baseline..."
python generate_data.py \
       --n_datasets 5 \
       --prior mlp_scm \
       --min_features 20 \
       --max_features 50 \
       --min_seq 1000 \
       --max_seq 2000 \
       --max_classes 3 \
       --out_dir ../synth/experiment/no_noise \
       --inner_bsz 64 \
       --no_causal \
       --y_is_effect \
       --in_clique \
       --num_layers 1 \
       --num_causes 10 \
       --noise_std 0.0 \
       --no_pre_sample_noise_std

echo "Experiment generation complete!"
echo "Test each configuration to see which produces the most learnable data."