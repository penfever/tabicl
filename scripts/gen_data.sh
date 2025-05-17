# cd tabicl
# pip install -e .
# chmod +x gen_data.sh
# ./gen_data.sh

python generate_data.py \
       --n_datasets 10 \
       --prior mlp_scm \
       --min_features 20 \
       --max_features 1000 \
       --min_seq 10000 \
       --max_seq 50000 \
       --max_classes 10 \
       --replay_small \
       --out_dir ../synth/mlp_scm \
       --inner_bsz 256

python generate_data.py \
       --n_datasets 1000 \
       --prior tree_scm \
       --min_features 20 \
       --max_features 1000 \
       --min_seq 10000 \
       --max_seq 50000 \
       --max_classes 10 \
       --replay_small \
       --out_dir ../synth/tree_scm \
       --inner_bsz 256
       