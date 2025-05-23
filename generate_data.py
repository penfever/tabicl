import argparse
import pathlib
import os
import time
import logging
from typing import Tuple
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from tqdm.auto import tqdm
from tabicl.prior.dataset import PriorDataset


def to_npy(arr: np.ndarray, path: pathlib.Path):
    np.save(path, arr, allow_pickle=False)

def to_csv(arr: np.ndarray, path: pathlib.Path):
    np.savetxt(path, arr, delimiter=",", fmt="%s")

def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().numpy()


def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """Initialize the distributed environment."""
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up the distributed environment."""
    import torch.distributed as dist
    if dist.is_initialized():
        try:
            dist.barrier()  # Final synchronization before cleanup
            destroy_process_group()
        except Exception as e:
            print(f"Warning: Cleanup error (can be ignored): {e}")
            pass


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

def generate_single_gpu(args):
    """Single-GPU generation function (original behavior)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    
    # Deterministic Tree SCM specific parameters
    if args.min_swap_prob is not None:
        hp_overrides['min_swap_prob'] = args.min_swap_prob
    if args.max_swap_prob is not None:
        hp_overrides['max_swap_prob'] = args.max_swap_prob
    if args.transform_type is not None:
        hp_overrides['transform_type'] = args.transform_type
    if args.noise_type is not None:
        hp_overrides['noise_type'] = args.noise_type
    if args.class_separability is not None and args.class_separability != 1.0:
        hp_overrides['class_separability'] = args.class_separability
    if args.assigner_type is not None:
        hp_overrides['assigner_type'] = args.assigner_type
    if args.n_estimators is not None:
        hp_overrides['n_estimators'] = args.n_estimators
    if args.max_depth is not None:
        hp_overrides['max_depth'] = args.max_depth
    
    # GMM clusters specific parameters
    if args.separation_strength is not None:
        hp_overrides['separation_strength'] = args.separation_strength
    if args.balance_strength is not None:
        hp_overrides['balance_strength'] = args.balance_strength
    
    # Explicit clusters specific parameters
    if args.cluster_separation is not None:
        hp_overrides['cluster_separation'] = args.cluster_separation
    if args.within_cluster_std is not None:
        hp_overrides['within_cluster_std'] = args.within_cluster_std
    if args.label_noise is not None:
        hp_overrides['label_noise'] = args.label_noise
    
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
    print(f"PriorDataset ready (prior={args.prior}). "
          f"Requesting first batch …")

    produced = 0
    uid_gen  = (f"{i:06}" for i in range(1, args.n_datasets + 1))
    pbar = tqdm(total=args.n_datasets, unit="ep", desc="Generating")

    for batch in ds:  # PriorDataset is already an iterable
        # batch is a tuple: (X, y, _, seq_len, train_size)
        X_batch, y_batch = batch[0], batch[1]

        # NestedTensor → list of Tensors; ordinary Tensor → split on dim‑0
        if hasattr(X_batch, "unbind"):
            X_list = [tensor_to_numpy(t) for t in X_batch.unbind()]
            y_list = [tensor_to_numpy(t) for t in y_batch.unbind()]
        else:
            X_list = [tensor_to_numpy(t) for t in X_batch]
            y_list = [tensor_to_numpy(t) for t in y_batch]

        for Xi, yi in zip(X_list, y_list):
            ep_id = next(uid_gen)
            base = args.out_dir / f"episode_{ep_id}"
            
            # Save numpy files
            np.save(base.parent / f"{base.name}_X.npy", Xi, allow_pickle=False)
            np.save(base.parent / f"{base.name}_y.npy", yi, allow_pickle=False)
            
            # Save CSV files if requested
            if args.save_csv:
                np.savetxt(base.parent / f"{base.name}_X.csv", Xi, delimiter=",", fmt="%.6f")
                np.savetxt(base.parent / f"{base.name}_y.csv", yi, delimiter=",", fmt="%d")

            produced += 1
            pbar.update(1)
            if produced == 1:
                print("First episode written — generation confirmed")
            if produced == args.n_datasets:
                pbar.close()
                return


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
    
    # Deterministic Tree SCM specific parameters
    if args.min_swap_prob is not None:
        hp_overrides['min_swap_prob'] = args.min_swap_prob
    if args.max_swap_prob is not None:
        hp_overrides['max_swap_prob'] = args.max_swap_prob
    if args.transform_type is not None:
        hp_overrides['transform_type'] = args.transform_type
    if args.noise_type is not None:
        hp_overrides['noise_type'] = args.noise_type
    if args.class_separability is not None and args.class_separability != 1.0:
        hp_overrides['class_separability'] = args.class_separability
    if args.assigner_type is not None:
        hp_overrides['assigner_type'] = args.assigner_type
    if args.n_estimators is not None:
        hp_overrides['n_estimators'] = args.n_estimators
    if args.max_depth is not None:
        hp_overrides['max_depth'] = args.max_depth
    
    # GMM clusters specific parameters
    if args.separation_strength is not None:
        hp_overrides['separation_strength'] = args.separation_strength
    if args.balance_strength is not None:
        hp_overrides['balance_strength'] = args.balance_strength
    
    # Explicit clusters specific parameters
    if args.cluster_separation is not None:
        hp_overrides['cluster_separation'] = args.cluster_separation
    if args.within_cluster_std is not None:
        hp_overrides['within_cluster_std'] = args.within_cluster_std
    if args.label_noise is not None:
        hp_overrides['label_noise'] = args.label_noise
    
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
                np.savetxt(base.parent / f"{base.name}_X.csv", Xi, delimiter=",", fmt="%.6f")
                np.savetxt(base.parent / f"{base.name}_y.csv", yi, delimiter=",", fmt="%d")
            
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
    
    # Synchronize all ranks before cleanup
    if world_size > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
    
    cleanup_distributed()


def generate_multi_gpu(args):
    """Main multi-GPU generation function."""
    # Determine number of GPUs
    world_size = args.num_gpus
    if world_size == -1:
        world_size = torch.cuda.device_count()
    
    if world_size == 0:
        print("No GPUs available. Falling back to CPU generation.")
        return generate_single_gpu(args)
    
    if world_size == 1:
        print("Single GPU detected. Using single-GPU generation.")
        return generate_single_gpu(args)
    
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
    
    # Ensure all ranks have finished before the main process exits
    torch.cuda.synchronize() if torch.cuda.is_available() else None


def generate(args):
    """Main generation function that dispatches to single or multi-GPU."""
    if args.num_gpus == 0:
        # Force single-GPU/CPU mode
        return generate_single_gpu(args)
    else:
        # Use multi-GPU if available
        return generate_multi_gpu(args)

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_datasets", type=int, required=True,
                    help="number of synthetic episodes to generate")
    ap.add_argument("--prior", default="mix_scm",
                    choices=["mlp_scm", "tree_scm", "mix_scm", "dummy", "deterministic_tree_scm", 
                             "gmm_clusters_scm", "real_explicit_clusters_scm"],
                    help="'mix_scm' resamples the family episode‑wise, 'deterministic_tree_scm' creates learnable datasets")
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
    
    # Deterministic Tree SCM specific parameters
    ap.add_argument("--min_swap_prob", type=float, default=0.0,
                    help="minimum probability of swapping target pairs in deterministic tree SCM")
    ap.add_argument("--max_swap_prob", type=float, default=0.2,
                    help="maximum probability of swapping target pairs in deterministic tree SCM")
    ap.add_argument("--transform_type", type=str, default="polynomial",
                    choices=["polynomial", "trigonometric", "exponential", "mixed", "rbf", "multi_modal", "mixture", "balanced_clusters", "enhanced_mixture"],
                    help="type of deterministic transformation for deterministic tree SCM")
    ap.add_argument("--noise_type", type=str, default="swap",
                    choices=["swap", "corrupt", "boundary_blur", "mixed"],
                    help="type of noise injection for deterministic tree SCM")
    ap.add_argument("--n_estimators", type=int, default=100,
                    help="number of trees in ensemble for tree-based models (default=100)")
    ap.add_argument("--max_depth", type=int, default=5,
                    help="maximum depth of trees for tree-based models (default=5)")
    
    # Class separability parameter
    ap.add_argument("--class_separability", type=float, default=1.0,
                    help="multiplier to scale informative features to increase class separation (default: 1.0)")
    
    # Assigner type for regression to classification conversion
    ap.add_argument("--assigner_type", type=str, default="rank",
                    choices=["rank", "value", "piecewise", "random_region", "step_function", "boolean_logic"],
                    help="type of class assigner for regression to classification conversion")
    
    # GMM clusters parameters
    ap.add_argument("--separation_strength", type=float, default=10.0,
                    help="separation strength for GMM clusters (lower = harder, default=10.0)")
    ap.add_argument("--balance_strength", type=float, default=0.9,
                    help="balance strength for GMM clusters (lower = more imbalance, default=0.9)")
    
    # Explicit clusters parameters
    ap.add_argument("--cluster_separation", type=float, default=3.0,
                    help="separation between explicit cluster centers (default=3.0)")
    ap.add_argument("--within_cluster_std", type=float, default=0.3,
                    help="standard deviation within each cluster (default=0.3)")
    ap.add_argument("--label_noise", type=float, default=0.0,
                    help="fraction of labels to randomly flip for noise (default=0.0)")
    
    # Multi-GPU specific arguments
    ap.add_argument("--num_gpus", type=int, default=-1,
                    help="number of GPUs to use (-1 for all available, 0 for CPU only)")
    ap.add_argument("--save_csv", action="store_true",
                    help="also save data as CSV files (slower but more accessible)")
    ap.add_argument("--master_port", type=str, default="29500",
                    help="master port for distributed training")
    ap.add_argument("--log_level", type=str, default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="logging level")
    
    return ap.parse_args()

if __name__ == "__main__":
    args = get_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.out_dir / 'generate_data.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Set up distributed environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port
    
    logger.info(f"Starting dataset generation with args: {args}")
    
    start_time = time.time()
    generate(args)
    end_time = time.time()
    
    print(f"\nFinished generating {args.n_datasets} datasets in {end_time - start_time:.2f} seconds")
    print(f"Average time per dataset: {(end_time - start_time) / args.n_datasets:.4f} seconds")
    logger.info(f"Finished generating {args.n_datasets} datasets in {end_time - start_time:.2f} seconds")
