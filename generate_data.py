import argparse, pathlib, io, uuid
import numpy as np
import torch
from tqdm.auto import tqdm
from tabicl.prior.dataset import PriorDataset


def to_npy(arr: np.ndarray, path: pathlib.Path):
    np.save(path, arr, allow_pickle=False)

def to_csv(arr: np.ndarray, path: pathlib.Path):
    np.savetxt(path, arr, delimiter=",", fmt="%s")

def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().numpy()

def generate(args):
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
        device="cuda" if torch.cuda.is_available() else "cpu",
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
            np.save(base.parent / f"{base.name}_X.npy", Xi, allow_pickle=False)
            np.save(base.parent / f"{base.name}_y.npy", yi, allow_pickle=False)
            np.savetxt(base.parent / f"{base.name}_X.csv", Xi, delimiter=",", fmt="%s")
            np.savetxt(base.parent / f"{base.name}_y.csv", yi, delimiter=",", fmt="%s")

            produced += 1
            pbar.update(1)
            if produced == 1:
                print("First episode written — generation confirmed")
            if produced == args.n_datasets:
                pbar.close()
                return

def get_args():
    ap = argparse.ArgumentParser()
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
    return ap.parse_args()

if __name__ == "__main__":
    args = get_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    generate(args)
    print("Finished.")
