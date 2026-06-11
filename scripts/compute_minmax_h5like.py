#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
import numpy as np
import h5py

def reduce_minmax(block, layout: str):
    """
    block is a numpy array chunk of the variable.
    Returns (mn, mx) arrays of shape (C,).
    Supported layouts:
      - NTCHW: (N, T, C, H, W)
      - NTHWC: (N, T, H, W, C)
      - NTHW : (N, T, H, W) -> treated as C=1
    """
    if layout == "NTCHW":
        mn = np.nanmin(block, axis=(0, 1, 3, 4))  # -> (C,)
        mx = np.nanmax(block, axis=(0, 1, 3, 4))
        return mn, mx
    if layout == "NTHWC":
        mn = np.nanmin(block, axis=(0, 1, 2, 3))  # -> (C,)
        mx = np.nanmax(block, axis=(0, 1, 2, 3))
        return mn, mx
    if layout == "NTHW":
        mn = np.nanmin(block, axis=(0, 1, 2, 3))  # scalar
        mx = np.nanmax(block, axis=(0, 1, 2, 3))
        return np.asarray([mn], dtype=np.float64), np.asarray([mx], dtype=np.float64)
    raise ValueError(f"Unsupported layout={layout}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to .nc/.h5/.hdf5 (must be HDF5-readable).")
    ap.add_argument("--key", required=True, help="Dataset key inside file (e.g., data or velocity).")
    ap.add_argument("--layout", default="NTCHW", choices=["NTCHW", "NTHWC", "NTHW"])
    ap.add_argument("--traj_start", type=int, default=0)
    ap.add_argument("--traj_count", type=int, required=True, help="Number of trajectories to include.")
    ap.add_argument("--time_start", type=int, default=0)
    ap.add_argument("--time_count", type=int, default=None, help="Number of timesteps (default: all).")

    ap.add_argument("--traj_block", type=int, default=32, help="How many trajectories per read.")
    ap.add_argument("--time_block", type=int, default=None, help="Optional time blocking (usually unnecessary).")

    ap.add_argument("--out_json", required=True)
    ap.add_argument("--float64_accum", action="store_true", help="Accumulate min/max in float64.")
    args = ap.parse_args()

    with h5py.File(args.path, "r") as f:
        if args.key not in f:
            raise KeyError(f"Key '{args.key}' not found. Keys={list(f.keys())}")
        d = f[args.key]
        shp = d.shape

        if args.layout == "NTCHW":
            N, T, C = int(shp[0]), int(shp[1]), int(shp[2])
        elif args.layout == "NTHWC":
            N, T, C = int(shp[0]), int(shp[1]), int(shp[4])
        else:  # NTHW
            N, T, C = int(shp[0]), int(shp[1]), 1

        traj_start = int(args.traj_start)
        traj_end = min(N, traj_start + int(args.traj_count))
        if traj_start < 0 or traj_start >= N:
            raise ValueError(f"traj_start {traj_start} out of range [0,{N})")
        if traj_end <= traj_start:
            raise ValueError("Empty traj range.")

        time_start = int(args.time_start)
        time_end = T if args.time_count is None else min(T, time_start + int(args.time_count))
        if time_start < 0 or time_start >= T:
            raise ValueError(f"time_start {time_start} out of range [0,{T})")
        if time_end <= time_start:
            raise ValueError("Empty time range.")

        acc_dtype = np.float64 if args.float64_accum else np.float32
        vmin = np.full((C,), np.inf, dtype=acc_dtype)
        vmax = np.full((C,), -np.inf, dtype=acc_dtype)

        # iterate
        for i0 in range(traj_start, traj_end, int(args.traj_block)):
            i1 = min(traj_end, i0 + int(args.traj_block))

            if args.time_block is None:
                if args.layout == "NTCHW":
                    block = np.asarray(d[i0:i1, time_start:time_end, :, :, :], dtype=np.float32)
                elif args.layout == "NTHWC":
                    block = np.asarray(d[i0:i1, time_start:time_end, :, :, :], dtype=np.float32)
                else:  # NTHW
                    block = np.asarray(d[i0:i1, time_start:time_end, :, :], dtype=np.float32)

                mn, mx = reduce_minmax(block, args.layout)
                vmin = np.minimum(vmin, mn.astype(acc_dtype))
                vmax = np.maximum(vmax, mx.astype(acc_dtype))
            else:
                # time-blocked mode
                tb = int(args.time_block)
                for t0 in range(time_start, time_end, tb):
                    t1 = min(time_end, t0 + tb)
                    if args.layout == "NTCHW":
                        block = np.asarray(d[i0:i1, t0:t1, :, :, :], dtype=np.float32)
                    elif args.layout == "NTHWC":
                        block = np.asarray(d[i0:i1, t0:t1, :, :, :], dtype=np.float32)
                    else:
                        block = np.asarray(d[i0:i1, t0:t1, :, :], dtype=np.float32)

                    mn, mx = reduce_minmax(block, args.layout)
                    vmin = np.minimum(vmin, mn.astype(acc_dtype))
                    vmax = np.maximum(vmax, mx.astype(acc_dtype))

            if (i0 - traj_start) // int(args.traj_block) % 20 == 0:
                print(f"[minmax] traj {i0}:{i1} / {traj_start}:{traj_end}", flush=True)

    out = dict(
        path=args.path,
        key=args.key,
        layout=args.layout,
        traj_start=traj_start,
        traj_end=traj_end,
        time_start=time_start,
        time_end=time_end,
        min=[float(x) for x in vmin.tolist()],
        max=[float(x) for x in vmax.tolist()],
    )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f">> wrote {args.out_json}")
    print(">> min:", out["min"])
    print(">> max:", out["max"])

if __name__ == "__main__":
    main()

