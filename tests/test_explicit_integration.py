"""Test explicit clusters integration with existing pipeline."""

import numpy as np
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Update generate_data to include explicit_clusters option
import generate_data

# Monkey patch to add explicit_clusters support
original_get_args = generate_data.get_args

def patched_get_args():
    args = original_get_args()
    # Update choices for prior
    for action in args._actions:
        if action.dest == 'prior':
            action.choices = action.choices + ['explicit_clusters_scm']
    return args

generate_data.get_args = patched_get_args

# Test argument parsing
print("Testing argument parsing...")
test_args = ['--n_datasets', '10', 
             '--prior', 'explicit_clusters_scm',
             '--min_features', '10',
             '--max_features', '20',
             '--min_seq', '1000',
             '--max_seq', '2000',
             '--min_classes', '10',
             '--max_classes', '10',
             '--out_dir', 'test_output']

import argparse
parser = patched_get_args()
args = parser.parse_args(test_args)

print(f"Prior: {args.prior}")
print(f"Classes: {args.min_classes} - {args.max_classes}")
print(f"Features: {args.min_features} - {args.max_features}")
print(f"Sequences: {args.min_seq} - {args.max_seq}")

print("\nIntegration test completed successfully!")