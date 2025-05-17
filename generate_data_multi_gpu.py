#!/usr/bin/env python3
"""
Multi-GPU data generation script for TabICL.

This script allows parallel generation of synthetic datasets across multiple GPUs,
significantly improving generation speed for large dataset creation.
"""

import argparse
import pathlib
import os
import time
from typing import List, Tuple
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from tqdm.auto import tqdm
from tabicl.prior.dataset import PriorDataset


def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """Initialize the distributed environment."""
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up the distributed environment."""
    destroy_process_group()


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().numpy()


def get_dataset_split(total_datasets: int, rank: int, world_size: int) -> Tuple[int, int]:
    """Calculate the dataset range for this rank."""
    datasets_per_rank = total_datasets // world_size
    remainder = total_datasets % world_size
    
    if rank < remainder:
        start = rank * (datasets_per_rank + 1)
        count = datasets_per_rank + 1
    else:
        start = rank * datasets_per_rank + remainder
        count = datasets_per_rank
    
    return start, count


def generate_worker(rank: int, world_size: int, args, start_idx: int):
    """Worker function for each GPU process."""
    # Set up distributed environment
    setup_distributed(rank, world_size)
    
    # Each worker gets its own dataset generator on its GPU
    device = f"cuda:{rank}"
    
    # Calculate this worker's dataset range
    _, num_datasets = get_dataset_split(args.n_datasets, rank, world_size)
    
    if num_datasets == 0:
        cleanup_distributed()
        return
    
    # Build hyperparameter overrides
    hp_overrides = {}
    if args.is_causal is not None:
        hp_overrides['is_causal'] = args.is_causal
    if args.y_is_effect is not None:
        hp_overrides['y_is_effect'] = args.y_is_effect
    if args.in_clique is not None:
        hp_overrides['in_clique'] = args.in_clique
    if args.num_layers is not None:
        hp_overrides['num_layers'] = args.num_layers
    if args.num_causes is not None:
        hp_overrides['num_causes'] = args.num_causes
    if args.noise_std is not None:
        hp_overrides['noise_std'] = args.noise_std
    if args.pre_sample_noise_std is not None:
        hp_overrides['pre_sample_noise_std'] = args.pre_sample_noise_std
    
    # Merge overrides with default fixed_hp
    from tabicl.prior.prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP
    fixed_hp = DEFAULT_FIXED_HP.copy()
    
    # Convert sampled values to fixed values by overriding them in fixed_hp
    # and adding distributions with single values to sampled_hp
    sampled_hp = DEFAULT_SAMPLED_HP.copy()
    for key, value in hp_overrides.items():
        # Add to fixed_hp so it won't be sampled
        fixed_hp[key] = value
        # Override the sampled distribution with a constant
        if key in sampled_hp:
            sampled_hp[key] = {"distribution": "meta_choice", "choice_values": [value]}
    
    # Create dataset generator
    ds = PriorDataset(
        batch_size=args.inner_bsz,
        batch_size_per_gp=1,
        batch_size_per_subgp=1,
        prior_type=args.prior,
        min_features=args.min_features,
        max_features=args.max_features,
        min_classes=args.min_classes,
        max_classes=args.max_classes,
        min_seq_len=args.min_seq,
        max_seq_len=args.max_seq,
        log_seq_len=args.log_seq,
        replay_small=args.replay_small,
        seq_len_per_gp=args.seq_len_per_gp,
        n_jobs=1,
        num_threads_per_generate=2,
        device=device,
        min_imbalance_ratio=args.min_imbalance_ratio,
        max_imbalance_ratio=args.max_imbalance_ratio,
        scm_fixed_hp=fixed_hp,
        scm_sampled_hp=sampled_hp,
    )
    
    if rank == 0:
        print(f"Worker {rank} ready on {device}. Generating {num_datasets} datasets...")
    
    # Progress bar only for rank 0
    pbar = None
    if rank == 0:
        pbar = tqdm(total=args.n_datasets, unit="ep", desc=f"Generating (all GPUs)")
    
    # Generate datasets
    produced = 0
    global_idx = start_idx
    
    for batch in ds:
        X_batch, y_batch = batch[0], batch[1]
        
        # Convert nested tensors to lists
        if hasattr(X_batch, "unbind"):
            X_list = [tensor_to_numpy(t) for t in X_batch.unbind()]
            y_list = [tensor_to_numpy(t) for t in y_batch.unbind()]
        else:
            X_list = [tensor_to_numpy(t) for t in X_batch]
            y_list = [tensor_to_numpy(t) for t in y_batch]
        
        # Save each dataset
        for Xi, yi in zip(X_list, y_list):
            if produced >= num_datasets:
                break
                
            ep_id = f"{global_idx:06}"
            base = args.out_dir / f"episode_{ep_id}"
            
            # Save numpy files
            np.save(base.parent / f"{base.name}_X.npy", Xi, allow_pickle=False)
            np.save(base.parent / f"{base.name}_y.npy", yi, allow_pickle=False)
            
            # Save CSV files if requested
            if args.save_csv:
                np.savetxt(base.parent / f"{base.name}_X.csv", Xi, delimiter=",", fmt="%s")
                np.savetxt(base.parent / f"{base.name}_y.csv", yi, delimiter=",", fmt="%s")
            
            produced += 1
            global_idx += 1
            
            # Update progress bar on rank 0
            if rank == 0 and pbar is not None:
                # Estimate total progress across all ranks
                estimated_total = produced * world_size
                pbar.n = min(estimated_total, args.n_datasets)
                pbar.refresh()
        
        if produced >= num_datasets:
            break
    
    if pbar is not None:
        pbar.close()
    
    cleanup_distributed()


def generate_multi_gpu(args):
    """Main multi-GPU generation function."""
    # Determine number of GPUs
    world_size = args.num_gpus
    if world_size == -1:
        world_size = torch.cuda.device_count()
    
    if world_size == 0:
        print("No GPUs available. Falling back to CPU generation.")
        # Fall back to single device generation
        import tabicl.generate_data as single_gpu
        return single_gpu.generate(args)
    
    print(f"Using {world_size} GPUs for data generation")
    
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Calculate starting indices for each rank
    processes = []
    for rank in range(world_size):
        start_idx, _ = get_dataset_split(args.n_datasets, rank, world_size)
        start_idx += 1  # 1-indexed
        
        p = mp.Process(
            target=generate_worker,
            args=(rank, world_size, args, start_idx)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()


def get_args():
    ap = argparse.ArgumentParser(
        description="Multi-GPU synthetic data generation for TabICL"
    )
    ap.add_argument("--n_datasets", type=int, required=True,
                    help="number of synthetic episodes to generate")
    ap.add_argument("--prior", default="mix_scm",
                    choices=["mlp_scm", "tree_scm", "mix_scm", "dummy"],
                    help="'mix_scm' resamples the family episode‑wise")
    ap.add_argument("--min_features", type=int, default=5)
    ap.add_argument("--max_features", type=int, default=120)
    ap.add_argument("--min_seq", type=int, default=200)
    ap.add_argument("--max_seq", type=int, default=60_000)
    ap.add_argument("--log_seq", action="store_true")
    ap.add_argument("--min_classes", type=int, default=2)
    ap.add_argument("--max_classes", type=int, default=10)
    ap.add_argument("--replay_small", action="store_true")
    ap.add_argument("--seq_len_per_gp", action="store_true")
    ap.add_argument("--inner_bsz", type=int, default=256,
                    help="episodes produced per generator call")
    ap.add_argument("--out_dir", type=pathlib.Path, required=True)
    ap.add_argument("--min_imbalance_ratio", type=float, default=1.0,
                    help="minimum ratio between largest and smallest class sizes (1.0 = balanced). "
                         "E.g., 2.0 means the largest class is at least 2x the smallest.")
    ap.add_argument("--max_imbalance_ratio", type=float, default=1.0,
                    help="maximum ratio between largest and smallest class sizes. "
                         "E.g., 10.0 means the largest class can be up to 10x the smallest.")
    
    # SCM causal structure parameters
    ap.add_argument("--is_causal", action="store_true", default=None,
                    help="use causal mode (sampling from intermediate SCM outputs)")
    ap.add_argument("--no_causal", dest="is_causal", action="store_false",
                    help="disable causal mode (direct mapping from causes to effects)")
    ap.add_argument("--y_is_effect", action="store_true", default=None,
                    help="sample targets from final layer (effects) rather than early layers")
    ap.add_argument("--no_y_is_effect", dest="y_is_effect", action="store_false",
                    help="sample targets from early layers (closer to causes)")
    ap.add_argument("--in_clique", action="store_true", default=None,
                    help="sample X and y from contiguous block (stronger dependencies)")
    ap.add_argument("--no_in_clique", dest="in_clique", action="store_false",
                    help="sample X and y independently (weaker dependencies)")
    ap.add_argument("--num_layers", type=int, default=None,
                    help="number of transformation layers in SCM")
    ap.add_argument("--num_causes", type=int, default=None,
                    help="number of initial cause variables")
    
    # Noise control parameters
    ap.add_argument("--noise_std", type=float, default=None,
                    help="base standard deviation for Gaussian noise")
    ap.add_argument("--pre_sample_noise_std", action="store_true", default=None,
                    help="pre-sample noise std for each layer")
    ap.add_argument("--no_pre_sample_noise_std", dest="pre_sample_noise_std", action="store_false",
                    help="use fixed noise std for all layers")
    
    # Multi-GPU specific arguments
    ap.add_argument("--num_gpus", type=int, default=-1,
                    help="number of GPUs to use (-1 for all available)")
    ap.add_argument("--save_csv", action="store_true",
                    help="also save data as CSV files (slower but more accessible)")
    ap.add_argument("--master_port", type=str, default="29500",
                    help="master port for distributed training")
    
    return ap.parse_args()


if __name__ == "__main__":
    args = get_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up distributed environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port
    
    start_time = time.time()
    generate_multi_gpu(args)
    end_time = time.time()
    
    print(f"\nFinished generating {args.n_datasets} datasets in {end_time - start_time:.2f} seconds")
    print(f"Average time per dataset: {(end_time - start_time) / args.n_datasets:.4f} seconds")