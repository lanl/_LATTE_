#!/usr/bin/env python3
from __future__ import annotations

import os
import glob
import argparse
import numpy as np


def scalar_str(x):
    a = np.asarray(x)
    if a.shape == ():
        return str(a.item())
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttc_rollout_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split_prefix", default="train")
    ap.add_argument("--n_embed", type=int, default=2048)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.ttc_rollout_dir, "*.npz")))
    if not files:
        raise RuntimeError(f"No npz files found in {args.ttc_rollout_dir}")

    print(f"input files: {len(files)}")
    print(f"out_dir: {args.out_dir}")

    for fp in files:
        z = np.load(fp, allow_pickle=True)
        if "pred_codes" not in z:
            raise RuntimeError(f"{fp}: missing pred_codes")

        tokens = np.asarray(z["pred_codes"]).astype(np.int64)
        if tokens.ndim != 3:
            raise RuntimeError(f"{fp}: pred_codes expected (T,Hq,Wq), got {tokens.shape}")

        if "run_id" in z:
            run_id = scalar_str(z["run_id"])
        else:
            base = os.path.basename(fp)
            run_id = base.replace("_ttc_mass_select_rollout.npz", "")

        out_fp = os.path.join(args.out_dir, f"{args.split_prefix}_{run_id}_tokens.npz")

        np.savez_compressed(
            out_fp,
            tokens=tokens,
            n_embed=np.asarray(args.n_embed, dtype=np.int64),
            run_id=str(run_id),
            key=str(run_id),
            source_rollout=str(fp),
        )

        print(f"saved {out_fp} {tokens.shape}")

    print("done")


if __name__ == "__main__":
    main()
