#!/bin/bash
# Generate synthetic data with strong causal relationships for better learnability
# Recommended settings to improve model performance

# cd tabicl
# pip install -e .
# chmod +x gen_data_strongcause.sh
# ./gen_data_strongcause.sh

echo "Generating MLP-SCM data with strong causal relationships..."
python generate_data.py \
       --n_datasets 10 \
       --prior mlp_scm \
       --min_features 20 \
       --max_features 1000 \
       --min_seq 10000 \
       --max_seq 50000 \
       --max_classes 10 \
       --replay_small \
       --out_dir ../synth/mlp_scm_strongcause \
       --inner_bsz 256 \
       --no_causal \
       --y_is_effect \
       --in_clique \
       --num_layers 3 \
       --num_causes 10 \
       --noise_std 0.001 \
       --no_pre_sample_noise_std \
       --save_csv

echo "Generating Tree-SCM data with strong causal relationships..."
python generate_data.py \
       --n_datasets 10 \
       --prior tree_scm \
       --min_features 20 \
       --max_features 1000 \
       --min_seq 10000 \
       --max_seq 50000 \
       --max_classes 10 \
       --replay_small \
       --out_dir ../synth/tree_scm_strongcause \
       --inner_bsz 256 \
       --is_causal \
       --y_is_effect \
       --in_clique \
       --num_layers 3 \
       --num_causes 15 \
       --noise_std 0.0005 \
       --no_pre_sample_noise_std \
       --save_csv

echo "Data generation complete!"

# Optional: Test CSV generation with small dataset
echo "Testing CSV generation with a small dataset..."
python generate_data.py \
       --n_datasets 1 \
       --prior mlp_scm \
       --min_features 10 \
       --max_features 20 \
       --min_seq 100 \
       --max_seq 200 \
       --max_classes 3 \
       --out_dir ../synth/test_csv \
       --inner_bsz 1 \
       --no_causal \
       --y_is_effect \
       --in_clique \
       --num_layers 2 \
       --num_causes 5 \
       --noise_std 0.0001 \
       --save_csv