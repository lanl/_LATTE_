#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json, csv, zlib
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------
# Deck helpers (copied style)
# ---------------------------

def _detect_delim(path: str) -> str:
    with open(path, "r", newline="") as f:
        head = f.readline()
    return "\t" if ("\t" in head) else ","


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    s = str(x).strip()
    if s == "":
        return True
    if s.lower() in ("nan", "none", "null"):
        return True
    return False


def _try_float(x: Any) -> Optional[float]:
    try:
        v = float(str(x).strip())
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def load_deck_csv_stripped(deck_csv: str, id_col: str) -> Dict[str, Dict[str, str]]:
    delim = _detect_delim(deck_csv)
    out: Dict[str, Dict[str, str]] = {}

    with open(deck_csv, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        if reader.fieldnames is None:
            raise RuntimeError(f"deck_csv={deck_csv} has no header.")

        fns = [("" if fn is None else str(fn).strip()) for fn in reader.fieldnames]
        rename = {orig: new for orig, new in zip(reader.fieldnames, fns)}

        id_col = str(id_col).strip()
        if id_col not in fns:
            raise RuntimeError(f"deck_csv={deck_csv} missing id_col='{id_col}'. cols={fns}")

        for row in reader:
            row2 = {rename.get(k, str(k).strip()): ("" if v is None else str(v).strip())
                    for k, v in row.items()}
            rid = row2.get(id_col, "").strip()
            if rid:
                out[rid] = row2

    if not out:
        raise RuntimeError(f"deck_csv={deck_csv} produced 0 rows.")
    return out


def deck_tokens_from_row(
    row: Dict[str, str],
    deck_keys: List[str],
    key_stats: Dict[str, Tuple[str, Optional[float], Optional[float]]],
    deck_bins: int,
    deck_base: int,
) -> np.ndarray:
    """
    Same scheme as your GPT deck tokenization:
      token_id = deck_base + key_index*(deck_bins+1) + bin
      bin 0 = missing, bins 1..deck_bins used for values.
    """
    tokens = np.empty((len(deck_keys),), dtype=np.int64)

    for ki, k in enumerate(deck_keys):
        mode, lo, hi = key_stats.get(k, ("cat", None, None))
        v = row.get(k, "")

        if _is_missing(v):
            b = 0
        else:
            if mode == "num":
                fv = _try_float(v)
                if fv is None:
                    b = 0
                else:
                    if lo is None or hi is None or float(hi) <= float(lo):
                        b = 1
                    else:
                        x = (float(fv) - float(lo)) / (float(hi) - float(lo))
                        x = min(1.0, max(0.0, x))
                        b = 1 + int(round(x * (deck_bins - 1)))
            else:
                s = str(v).strip()
                h = zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF
                b = 1 + (h % deck_bins)

        tokens[ki] = int(deck_base) + ki * (deck_bins + 1) + int(b)

    return tokens


# ---------------------------
# CFD dataset
# ---------------------------

class CFDAVDensityH5Dataset(Dataset):
    """
    Dataset over *complete* CFD trajectories, reconstructed across shards by (traj_id, time_id).

    Inputs:
      frame_index_json: output of build_cfd_frame_index.py
      splits_json:      output of split_cfd_trajs.py
      norm_json:        {"mean":..., "std":...} for av_density

    Returns (dict):
      mode="traj":
        x: (T,1,H,W)
      mode="frame":
        x: (1,H,W)
      mode="pair":
        x_t:   (1,H,W)
        x_tp1: (1,H,W)

      deck: (deck_len,) int64 or empty
      traj_id: str
      time_id: int (frame/pair)
    """
    def __init__(
        self,
        frame_index_json: str,
        splits_json: str,
        split: str,                       # "train"|"val"|"test"
        norm_json: str,

        mode: str = "traj",              # "traj"|"frame"|"pair"
        field: str = "av_density",

        # sampling controls
        seed: int = 1337,

        # deck options (optional)
        deck_csv: Optional[str] = None,
        deck_id_col: str = "traj_id",     # must match CSV
        use_deck: bool = False,

        deck_keys: Optional[List[str]] = None,
        key_stats: Optional[Dict[str, Tuple[str, Optional[float], Optional[float]]]] = None,
        deck_bins: int = 64,

        missing_deck_policy: str = "error",   # error|drop|zeros (drop drops trajectory)
        # If you already have GPT n_embed, pass it so deck_base matches later;
        # otherwise we set deck_base=0 and still emit deck tokens in a consistent range.
        deck_base: int = 0,
    ):
        super().__init__()
        self.mode = str(mode)
        if self.mode not in ("traj", "frame", "pair"):
            raise ValueError("mode must be one of: traj, frame, pair")
        self.field = str(field)
        self.rng = np.random.default_rng(int(seed))

        with open(frame_index_json, "r") as f:
            idx = json.load(f)
        with open(splits_json, "r") as f:
            splits = json.load(f)
        with open(norm_json, "r") as f:
            norm = json.load(f)

        self.T = int(idx["T"])
        self.H = int(idx["H"])
        self.W = int(idx["W"])

        self.mean = float(norm["mean"])
        self.std = float(norm["std"]) if float(norm["std"]) > 0 else 1.0

        split = str(split)
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train|val|test")
        split_ids = set(map(int, splits[split]))

        # Keep only complete trajectories in the requested split
        entries = []
        for e in idx["entries"]:
            if not e.get("is_complete", False):
                continue
            tid = int(e["traj_id"])
            if tid in split_ids:
                entries.append(e)

        if not entries:
            raise RuntimeError(f"No complete trajectories found for split={split}.")

        self.entries = entries  # each has e["frames"] list length T

        # --- deck ---
        self.use_deck = bool(use_deck) and (deck_csv is not None)
        self.deck_bins = int(deck_bins)
        self.deck_base = int(deck_base)
        self.deck_map = None

        if self.use_deck:
            self.deck_map = load_deck_csv_stripped(str(deck_csv), str(deck_id_col))

            if deck_keys is not None:
                self.deck_keys = list(deck_keys)
            else:
                # default: all cols except id_col
                ex = next(iter(self.deck_map.values()))
                self.deck_keys = [k for k in ex.keys() if k != str(deck_id_col).strip()]

            if key_stats is None:
                self.key_stats = {k: ("cat", None, None) for k in self.deck_keys}
            else:
                self.key_stats = dict(key_stats)

            self.deck_len = len(self.deck_keys)
        else:
            self.deck_keys = []
            self.key_stats = {}
            self.deck_len = 0

        missing_deck_policy = str(missing_deck_policy).lower()
        if missing_deck_policy not in ("error", "drop", "zeros"):
            raise ValueError("missing_deck_policy must be error|drop|zeros")

        # Precompute deck tokens aligned with entries
        if self.deck_len > 0:
            keep = []
            deck_all = []
            missing = []
            for e in self.entries:
                tid = str(int(e["traj_id"]))
                row = self.deck_map.get(tid, None) if self.deck_map is not None else None
                if row is None:
                    missing.append(tid)
                    if missing_deck_policy == "drop":
                        continue
                    if missing_deck_policy == "zeros":
                        row = {}
                    if missing_deck_policy == "error":
                        row = {}
                dtok = deck_tokens_from_row(row, self.deck_keys, self.key_stats,
                                            self.deck_bins, self.deck_base)
                keep.append(e)
                deck_all.append(dtok.astype(np.int64))
            if missing:
                msg = f">> deck_csv missing {len(missing)} traj_ids (e.g. {missing[:5]}) policy={missing_deck_policy}"
                print(msg, flush=True)
                if missing_deck_policy == "error":
                    raise RuntimeError(msg)
            self.entries = keep
            if not self.entries:
                raise RuntimeError("All trajectories were dropped due to missing deck.")
            self.deck_tokens = np.stack(deck_all, axis=0).astype(np.int64)
        else:
            self.deck_tokens = None

        # H5 file cache (per-process)
        self._h5 = {}

    def __len__(self) -> int:
        return len(self.entries)

    def _get_h5(self, path: str):
        import h5py  # local import to avoid issues in some launchers
        f = self._h5.get(path, None)
        if f is None:
            f = h5py.File(path, "r")
            self._h5[path] = f
        return f

    def _read_frame(self, shard: str, idx: int) -> np.ndarray:
        f = self._get_h5(shard)
        d = f[self.field]  # (N,H,W)
        x = np.asarray(d[int(idx)], dtype=np.float32)  # (H,W)
        return x

    def _norm(self, x: np.ndarray) -> np.ndarray:
        # normalize to roughly zero-mean unit-std
        return (x - self.mean) / self.std

    def __getitem__(self, i: int):
        e = self.entries[i]
        tid_int = int(e["traj_id"])
        tid = str(tid_int)

        deck = np.empty((0,), dtype=np.int64)
        if self.deck_tokens is not None:
            deck = self.deck_tokens[i]

        frames = e["frames"]  # list of {"t","shard","idx"} length T

        if self.mode == "traj":
            # read all frames in order
            xs = np.empty((self.T, self.H, self.W), dtype=np.float32)
            for t, rec in enumerate(frames):
                x = self._read_frame(rec["shard"], rec["idx"])
                xs[t] = self._norm(x)
            xs = xs[:, None, :, :]  # (T,1,H,W)
            return {
                "traj_id": tid,
                "x": torch.from_numpy(xs),
                "deck": torch.from_numpy(deck),
                "src": frames,
            }

        if self.mode == "frame":
            t = int(self.rng.integers(0, self.T))
            rec = frames[t]
            x = self._norm(self._read_frame(rec["shard"], rec["idx"]))[None, :, :]  # (1,H,W)
            return {
                "traj_id": tid,
                "time_id": int(rec["t"]),
                "x": torch.from_numpy(x.astype(np.float32)),
                "deck": torch.from_numpy(deck),
                "src": rec,
            }

        # mode == "pair"
        t = int(self.rng.integers(0, self.T - 1))
        rec0 = frames[t]
        rec1 = frames[t + 1]
        x0 = self._norm(self._read_frame(rec0["shard"], rec0["idx"]))[None, :, :]
        x1 = self._norm(self._read_frame(rec1["shard"], rec1["idx"]))[None, :, :]
        return {
            "traj_id": tid,
            "time_id": int(rec0["t"]),
            "x_t": torch.from_numpy(x0.astype(np.float32)),
            "x_tp1": torch.from_numpy(x1.astype(np.float32)),
            "deck": torch.from_numpy(deck),
            "src": (rec0, rec1),
        }
