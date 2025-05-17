#!/bin/bash
# Test script for the new deterministic tree SCM with varying noise levels

echo "Testing Deterministic Tree SCM with different swap probabilities..."

# Very clean data (no noise)
echo "Test 1: No noise (0%)"
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
    --out_dir ./synth/deterministic_test/no_noise \
    --inner_bsz 32 \
    --no_causal \
    --num_layers 1 \
    --min_swap_prob 0.0 \
    --max_swap_prob 0.0 \
    --transform_type polynomial \
    --noise_type swap \
    --noise_std 0.001 \
    --save_csv

# Low noise with corruption
echo "Test 2: Low corruption (15%)"
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
    --out_dir ./synth/deterministic_test/low_corrupt \
    --inner_bsz 32 \
    --no_causal \
    --num_layers 1 \
    --min_swap_prob 0.1 \
    --max_swap_prob 0.15 \
    --transform_type polynomial \
    --noise_type corrupt \
    --noise_std 0.001 \
    --save_csv

# Moderate mixed noise
echo "Test 3: Moderate mixed noise (20-35%)"
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
    --out_dir ./synth/deterministic_test/moderate_mixed \
    --inner_bsz 32 \
    --no_causal \
    --num_layers 2 \
    --min_swap_prob 0.2 \
    --max_swap_prob 0.35 \
    --transform_type mixed \
    --noise_type mixed \
    --noise_std 0.001 \
    --save_csv

# High boundary blur noise
echo "Test 4: High boundary blur (50-70%)"
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
    --out_dir ./synth/deterministic_test/high_blur \
    --inner_bsz 32 \
    --no_causal \
    --num_layers 3 \
    --min_swap_prob 0.5 \
    --max_swap_prob 0.7 \
    --transform_type mixed \
    --noise_type boundary_blur \
    --noise_std 0.001 \
    --save_csv

echo "Generation complete!"
echo "Now test these datasets with a simple classifier to verify learnability:"
echo "python evaluate_on_dataset.py --baselines_only --model_id random_forest --data_dir ./synth/deterministic_test/no_noise"
echo "python evaluate_on_dataset.py --baselines_only --model_id random_forest --data_dir ./synth/deterministic_test/low_corrupt"
echo "python evaluate_on_dataset.py --baselines_only --model_id random_forest --data_dir ./synth/deterministic_test/moderate_mixed"
echo "python evaluate_on_dataset.py --baselines_only --model_id random_forest --data_dir ./synth/deterministic_test/high_blur"