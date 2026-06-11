#!/usr/bin/env python3
import argparse, json, os
import numpy as np
import h5py
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_index_json", required=True)
    ap.add_argument("--splits_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--chunk_t", type=int, default=8)
    args = ap.parse_args()

    with open(args.frame_index_json, "r") as f:
        idx = json.load(f)
    with open(args.splits_json, "r") as f:
        splits = json.load(f)

    train_set = set(map(int, splits["train"]))

    # traj_id -> frames list
    traj_to_frames = {}
    for e in idx["entries"]:
        tid = int(e["traj_id"])
        if not e["is_complete"]:
            continue
        if tid not in train_set:
            continue
        traj_to_frames[tid] = e["frames"]  # list of {"t","shard","idx"}

    # cache open files per path
    files = {}
    def get_file(path):
        f = files.get(path)
        if f is None:
            f = h5py.File(path, "r")
            files[path] = f
        return f

    s = 0.0
    ss = 0.0
    cnt = 0
    gmin = None
    gmax = None

    try:
        for tid, frames in traj_to_frames.items():
            # frames are in time order (0..100)
            # read in small chunks over t
            for a in range(0, len(frames), args.chunk_t):
                chunk = frames[a:a+args.chunk_t]
                # group by shard to minimize random I/O
                by_shard = defaultdict(list)
                for rec in chunk:
                    by_shard[rec["shard"]].append(rec["idx"])

                for sp, idxs in by_shard.items():
                    f = get_file(sp)
                    d = f["av_density"]
                    # read each idx; (len, H, W)
                    x = np.stack([d[i] for i in idxs], axis=0).astype(np.float64)
                    s += x.sum()
                    ss += (x*x).sum()
                    cnt += x.size
                    mn = float(x.min()); mx = float(x.max())
                    gmin = mn if gmin is None else min(gmin, mn)
                    gmax = mx if gmax is None else max(gmax, mx)

    finally:
        for f in files.values():
            try: f.close()
            except: pass

    mean = s / cnt
    var = max(ss / cnt - mean*mean, 0.0)
    std = float(np.sqrt(var))

    out = {
        "field": "av_density",
        "mean": float(mean),
        "std": std,
        "min": float(gmin),
        "max": float(gmax),
        "count": int(cnt),
        "source": "train_split_complete_only",
        "frame_index_json": args.frame_index_json,
        "splits_json": args.splits_json,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote:", args.out_json)
    print(out)

if __name__ == "__main__":
    main()
