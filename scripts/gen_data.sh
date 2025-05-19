# cd tabicl
# pip install -e .
# chmod +x gen_data.sh
# ./gen_data.sh

python generate_data.py \
       --n_datasets 10 \
       --prior deterministic_tree_scm \
       --num_gpus 1 \
       --min_features 20 \
       --max_features 200 \
       --min_seq 4000 \
       --max_seq 10000 \
       --min_classes 10 \
       --max_classes 10 \
       --class_separability 3.5 \
       --max_imbalance_ratio 2.0 \
       --replay_small \
       --out_dir ../synth/deterministic_tree_scm_classep_v1 \
       --inner_bsz 32 \
       --no_causal \
       --num_layers 1 \
       --min_swap_prob 0.0 \
       --max_swap_prob 0.0 \
       --transform_type polynomial \
       --noise_type swap \
       --noise_std 0.001 \
       --save_csv

# python generate_data.py \
#        --n_datasets 1000 \
#        --prior tree_scm \
#        --min_features 20 \
#        --max_features 1000 \
#        --min_seq 10000 \
#        --max_seq 50000 \
#        --max_classes 10 \
#        --replay_small \
#        --out_dir ../synth/tree_scm \
#        --inner_bsz 256
       