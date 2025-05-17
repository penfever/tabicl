#!/bin/bash
# Generate synthetic data with strong causal relationships for better learnability
# Recommended settings to improve model performance

# cd tabicl
# pip install -e .
# chmod +x gen_data_strongcause.sh
# ./gen_data_strongcause.sh

echo "Generating SCM data with strong causal relationships..."
python generate_data.py \
       --n_datasets 1000 \
       --num_gpus 1 \
       --prior tree_scm \
       --min_features 20 \
       --max_features 400 \
       --min_seq 5000 \
       --max_seq 20000 \
       --min_classes 10 \
       --max_classes 10 \
       --replay_small \
       --out_dir ../synth/tree_scm_strongcause_v2 \
       --inner_bsz 256 \
       --no_causal \
       --y_is_effect \
       --in_clique \
       --num_layers 3 \
       --num_causes 10 \
       --max_imbalance_ratio 2.5 \
       --noise_std 0.001 \
       --no_pre_sample_noise_std \
       --save_csv

# echo "Generating Tree-SCM data with strong causal relationships..."
# python generate_data.py \
#        --n_datasets 1000 \
#        --num_gpus 1 \
#        --prior tree_scm \
#        --min_features 20 \
#        --max_features 200 \
#        --min_seq 2000 \
#        --max_seq 20000 \
#        --min_classes 10 \
#        --max_classes 10 \
#        --replay_small \
#        --out_dir ../synth/tree_scm_strongcause \
#        --inner_bsz 256 \
#        --is_causal \
#        --y_is_effect \
#        --in_clique \
#        --num_layers 3 \
#        --num_causes 15 \
#        --noise_std 0.0005 \
#        --no_pre_sample_noise_std \
#        --save_csv

echo "Data generation complete!"

# Optional: Generate test datasets with extreme settings for comparison
# echo "Generating test data with very low noise..."
# python generate_data.py \
#        --n_datasets 5 \
#        --prior mlp_scm \
#        --min_features 10 \
#        --max_features 50 \
#        --min_seq 1000 \
#        --max_seq 5000 \
#        --max_classes 5 \
#        --out_dir ../synth/test_low_noise \
#        --inner_bsz 64 \
#        --no_causal \
#        --y_is_effect \
#        --in_clique \
#        --num_layers 2 \
#        --num_causes 10 \
#        --noise_std 0.0001