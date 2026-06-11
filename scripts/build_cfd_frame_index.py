#!/usr/bin/env python3
import argparse, glob, json, os
import numpy as np
import h5py
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_glob", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--T", type=int, default=101)
    args = ap.parse_args()

    shards = sorted(glob.glob(args.shard_glob))
    assert shards, f"No shards matched: {args.shard_glob}"

    # traj_id -> list of (time_id, shard_path, frame_index)
    traj_map = defaultdict(list)

    total_frames = 0
    for sp in shards:
        with h5py.File(sp, "r") as f:
            traj_id = f["traj_id"][:]
            time_id = f["time_id"][:]
            assert traj_id.shape == time_id.shape
            n = traj_id.shape[0]
            total_frames += n
            for i in range(n):
                tid = int(traj_id[i])
                t = int(time_id[i])
                traj_map[tid].append((t, sp, int(i)))

    # Build per-trajectory entries; keep only complete trajectories by default
    entries = []
    num_complete = 0
    num_incomplete = 0
    for tid, lst in traj_map.items():
        # sort by time_id, then shard name to stabilize
        lst_sorted = sorted(lst, key=lambda x: (x[0], x[1], x[2]))

        # if duplicates exist for same time_id, keep first and note it
        time_to_rec = {}
        dup_times = []
        for t, sp, idx in lst_sorted:
            if t in time_to_rec:
                dup_times.append(t)
                continue
            time_to_rec[t] = (sp, idx)

        present = sorted(time_to_rec.keys())
        is_complete = (len(present) == args.T) and (present[0] == 0) and (present[-1] == args.T - 1)

        entry = {
            "traj_id": tid,
            "is_complete": bool(is_complete),
            "num_frames": int(len(present)),
            "missing_times": [t for t in range(args.T) if t not in time_to_rec],
            "duplicate_times": sorted(set(dup_times)),
            # records: list indexed by time t: (shard, idx)
            "frames": [{"t": t, "shard": time_to_rec[t][0], "idx": time_to_rec[t][1]}
                       for t in range(args.T) if t in time_to_rec],
        }
        entries.append(entry)
        if is_complete:
            num_complete += 1
        else:
            num_incomplete += 1

    out = {
        "field": "av_density",
        "T": args.T,
        "H": 560,
        "W": 200,
        "shard_glob": args.shard_glob,
        "num_shards": len(shards),
        "total_frames_scanned": total_frames,
        "num_traj_total": len(entries),
        "num_traj_complete": num_complete,
        "num_traj_incomplete": num_incomplete,
        "entries": sorted(entries, key=lambda e: e["traj_id"]),
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out_json}")
    print(f"Traj total={out['num_traj_total']} complete={out['num_traj_complete']} incomplete={out['num_traj_incomplete']}")

if __name__ == "__main__":
    main()
