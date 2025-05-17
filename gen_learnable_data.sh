#!/bin/bash
# Generate truly learnable synthetic datasets by disabling all randomization

echo "Generating learnable synthetic datasets..."

# Create a custom configuration file to override default randomization
cat > custom_fixed_hp.py << EOF
LEARNABLE_FIXED_HP = {
    # TreeSCM specific
    "tree_model": "random_forest",
    "tree_depth_lambda": 0.5,
    "tree_n_estimators_lambda": 0.5,
    
    # Reg2Cls - DISABLE ALL RANDOMIZATION
    "balanced": True,              # Use balanced classes
    "multiclass_ordered_prob": 1.0,  # Keep natural order
    "cat_prob": 0.0,              # No random categorical conversion
    "max_categories": 10,
    "scale_by_max_features": False,
    "permute_features": False,     # Don't permute features!
    "permute_labels": False,       # Don't permute labels!
    
    # Use deterministic transformation
    "multiclass_type": "rank",     # Use rank-based classes (more stable)
}
EOF

# Generate datasets with deterministic tree SCM
echo "Generating deterministic datasets..."
python -c "
import sys
sys.path.append('.')
from custom_fixed_hp import LEARNABLE_FIXED_HP
from tabicl.prior.dataset import PriorDataset
from tabicl.prior.prior_config import DEFAULT_SAMPLED_HP
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Override sampled hyperparameters to be deterministic
sampled_hp = DEFAULT_SAMPLED_HP.copy()
sampled_hp['multiclass_type'] = {'distribution': 'meta_choice', 'choice_values': ['rank']}

# Create dataset generator
ds = PriorDataset(
    batch_size=1,
    prior_type='deterministic_tree_scm',
    min_features=20,
    max_features=50,
    min_classes=2,
    max_classes=2,  # Binary for simplicity
    min_seq_len=1000,
    max_seq_len=2000,
    scm_fixed_hp=LEARNABLE_FIXED_HP,
    scm_sampled_hp=sampled_hp,
    min_imbalance_ratio=1.0,
    max_imbalance_ratio=1.0,
    device='cpu'
)

# Generate datasets
out_dir = Path('./synth/learnable_data')
out_dir.mkdir(parents=True, exist_ok=True)

n_datasets = 10
pbar = tqdm(total=n_datasets, desc='Generating learnable datasets')

for i in range(n_datasets):
    batch = next(iter(ds))
    X, y = batch[0].squeeze(0), batch[1].squeeze(0)
    
    # Save as numpy files
    np.save(out_dir / f'episode_{i+1:06d}_X.npy', X.numpy())
    np.save(out_dir / f'episode_{i+1:06d}_y.npy', y.numpy())
    
    # Also save as CSV for inspection
    np.savetxt(out_dir / f'episode_{i+1:06d}_X.csv', X.numpy(), delimiter=',', fmt='%.6f')
    np.savetxt(out_dir / f'episode_{i+1:06d}_y.csv', y.numpy(), delimiter=',', fmt='%d')
    
    pbar.update(1)

pbar.close()
print(f'Generated {n_datasets} learnable datasets in {out_dir}')
"

echo "Testing dataset learnability..."
python evaluate_on_dataset.py \
    --baselines_only \
    --model_id random_forest \
    --data_dir ./synth/learnable_data \
    --output_dir ./synth/learnable_data/results

echo "Complete! Check the results in ./synth/learnable_data/results/"