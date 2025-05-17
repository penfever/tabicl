#!/bin/bash
# Test script for the new deterministic tree SCM with varying noise levels

echo "Testing Deterministic Tree SCM with different swap probabilities..."

# Very clean data (no swapping)
echo "Test 1: No swapping (0% noise)"
python generate_data.py \
    --n_datasets 10 \
    --num_gpus 1 \
    --prior deterministic_tree_scm \
    --min_features 20 \
    --max_features 50 \
    --min_seq 1000 \
    --max_seq 2000 \
    --min_classes 2 \
    --max_classes 3 \
    --out_dir ./synth/deterministic_test/no_swap \
    --inner_bsz 32 \
    --no_causal \
    --num_layers 1 \
    --min_swap_prob 0.0 \
    --max_swap_prob 0.0 \
    --transform_type polynomial \
    --noise_std 0.001 \
    --save_csv

# Low noise (5% swapping)
echo "Test 2: Low noise (5% swapping)"
python generate_data.py \
    --n_datasets 10 \
    --num_gpus 1 \
    --prior deterministic_tree_scm \
    --min_features 20 \
    --max_features 50 \
    --min_seq 1000 \
    --max_seq 2000 \
    --min_classes 2 \
    --max_classes 3 \
    --out_dir ./synth/deterministic_test/low_swap \
    --inner_bsz 32 \
    --no_causal \
    --num_layers 1 \
    --min_swap_prob 0.0 \
    --max_swap_prob 0.05 \
    --transform_type polynomial \
    --noise_std 0.001 \
    --save_csv

# Moderate noise (10-20% swapping)
echo "Test 3: Moderate noise (10-20% swapping)"
python generate_data.py \
    --n_datasets 10 \
    --num_gpus 1 \
    --prior deterministic_tree_scm \
    --min_features 20 \
    --max_features 50 \
    --min_seq 1000 \
    --max_seq 2000 \
    --min_classes 2 \
    --max_classes 3 \
    --out_dir ./synth/deterministic_test/moderate_swap \
    --inner_bsz 32 \
    --no_causal \
    --num_layers 2 \
    --min_swap_prob 0.1 \
    --max_swap_prob 0.2 \
    --transform_type mixed \
    --noise_std 0.001 \
    --save_csv

# High noise (40-60% swapping)
echo "Test 3: High noise (40-60% swapping)"
python generate_data.py \
    --n_datasets 10 \
    --num_gpus 1 \
    --prior deterministic_tree_scm \
    --min_features 20 \
    --max_features 50 \
    --min_seq 1000 \
    --max_seq 2000 \
    --min_classes 2 \
    --max_classes 3 \
    --out_dir ./synth/deterministic_test/high_swap \
    --inner_bsz 32 \
    --no_causal \
    --num_layers 2 \
    --min_swap_prob 0.4 \
    --max_swap_prob 0.6 \
    --transform_type mixed \
    --noise_std 0.001 \
    --save_csv

echo "Generation complete!"
echo "Now test these datasets with a simple classifier to verify learnability:"
echo "python evaluate_on_dataset.py --baselines_only --model_id random_forest --data_dir ./synth/deterministic_test/no_swap"
echo "python evaluate_on_dataset.py --baselines_only --model_id random_forest --data_dir ./synth/deterministic_test/low_swap"
echo "python evaluate_on_dataset.py --baselines_only --model_id random_forest --data_dir ./synth/deterministic_test/moderate_swap"
echo "python evaluate_on_dataset.py --baselines_only --model_id random_forest --data_dir ./synth/deterministic_test/high_swap"