#!/bin/bash
# Generate synthetic data with strong causal relationships for better learnability
# Recommended settings to improve model performance

# cd tabicl
# pip install -e .
# chmod +x gen_data_strongcause.sh
# ./gen_data_strongcause.sh

echo "Generating SCM data with strong causal relationships..."
cd ..

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
       --prior deterministic_tree_scm \
       --min_features 20 \
       --max_features 400 \
       --min_seq 5000 \
       --max_seq 20000 \
       --min_classes 10 \
       --max_classes 10 \
       --replay_small \
       --out_dir ../synth/tree_scm_strongcause_v3 \
       --inner_bsz 256 \
       --no_causal \
       --num_layers 2 \
       --min_swap_prob 0.2 \
       --max_swap_prob 0.35 \
       --transform_type mixed \
       --noise_type mixed \
       --noise_std 0.001 \
       # --y_is_effect \
       --in_clique \
       # --num_causes 10 \
       --max_imbalance_ratio 2.5 \
       # --no_pre_sample_noise_std \
       --save_csv