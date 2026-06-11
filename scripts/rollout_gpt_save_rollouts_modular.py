#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage A (modular, deterministic): Greedy rollout for ONE GPT checkpoint, save predicted token grids.

- Uses dataset registry JSON (same pattern as transformer.py)
- Uses a TRAJECTORY dataset (per-run NPZs) that returns:
    {"run_id", "tokens"(T,Hq,Wq), "deck"(deck_len,), "src_tokens_npz"}
- Loads GPT ckpt; passes deck_keys/key_stats/deck_bins into dataset to match training tokenization.
- Multi-GPU via multiprocessing, one proc per GPU.
"""

from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")

import argparse, json, math, time, zlib
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp

from src.data.registry import build_dataset_from_registry


# =============================================================================
# Determinism
# =============================================================================
def set_full_determinism(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f">> WARNING: torch.use_deterministic_algorithms(True) failed: {e}")

    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


# =============================================================================
# Model (EncDecGPT) minimal
# =============================================================================
class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, use_sdpa: bool, causal: bool):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.use_sdpa = bool(use_sdpa)
        self.causal = bool(causal)

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            dp = self.attn_drop.p if self.training else 0.0
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dp, is_causal=self.causal)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.causal:
                causal = torch.tril(torch.ones((T, T), device=att.device, dtype=torch.bool))
                att = att.masked_fill(~causal, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

class CrossAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, use_sdpa: bool):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.use_sdpa = bool(use_sdpa)

        self.q = nn.Linear(n_embd, n_embd, bias=False)
        self.kv = nn.Linear(n_embd, 2 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, mem):
        B, T, C = x.shape
        S = mem.size(1)

        q = self.q(x)
        kv = self.kv(mem)
        k, v = kv.split(C, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            dp = self.attn_drop.p if self.training else 0.0
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dp, is_causal=False)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

class EncoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout, use_sdpa):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, dropout, use_sdpa=use_sdpa, causal=False)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout, use_sdpa):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.self_attn = SelfAttention(n_embd, n_head, dropout, use_sdpa=use_sdpa, causal=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.cross_attn = CrossAttention(n_embd, n_head, dropout, use_sdpa=use_sdpa)
        self.ln3 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)
    def forward(self, x, mem):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), mem)
        x = x + self.mlp(self.ln3(x))
        return x

class EncDecGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        Hq: int,
        Wq: int,
        deck_len: int,
        dec_block_size: int,
        n_layer_enc: int,
        n_layer_dec: int,
        n_head: int,
        n_embd: int,
        dropout: float,
        pad_id: int,
        use_sdpa: bool,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n_embed = int(n_embed)
        self.Hq, self.Wq = int(Hq), int(Wq)
        self.enc_len_frame = self.Hq * self.Wq
        self.deck_len = int(deck_len)
        self.dec_block_size = int(dec_block_size)
        self.pad_id = int(pad_id)
        self.use_sdpa = bool(use_sdpa)

        self.tok_emb = nn.Embedding(self.vocab_size, n_embd)

        self.row_emb = nn.Embedding(self.Hq, n_embd)
        self.col_emb = nn.Embedding(self.Wq, n_embd)

        self.deck_pos_emb = nn.Embedding(max(1, self.deck_len), n_embd)
        self.deck_type = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.frame_type = nn.Parameter(torch.zeros(1, 1, n_embd))

        self.dec_pos_emb = nn.Embedding(self.dec_block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.enc_blocks = nn.ModuleList([EncoderBlock(n_embd, n_head, dropout, use_sdpa=use_sdpa) for _ in range(n_layer_enc)])
        self.dec_blocks = nn.ModuleList([DecoderBlock(n_embd, n_head, dropout, use_sdpa=use_sdpa) for _ in range(n_layer_dec)])

        self.enc_ln = nn.LayerNorm(n_embd)
        self.dec_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, self.vocab_size, bias=False)

    def encode(self, enc_tokens: torch.Tensor, deck_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L = enc_tokens.shape
        assert L == self.enc_len_frame

        xf = self.tok_emb(enc_tokens)
        r = torch.arange(self.Hq, device=enc_tokens.device).view(self.Hq, 1).expand(self.Hq, self.Wq).reshape(-1)
        c = torch.arange(self.Wq, device=enc_tokens.device).view(1, self.Wq).expand(self.Hq, self.Wq).reshape(-1)
        pos2d = self.row_emb(r)[None, :, :] + self.col_emb(c)[None, :, :]
        xf = xf + pos2d + self.frame_type

        if deck_tokens is not None and self.deck_len > 0:
            xd = self.tok_emb(deck_tokens)
            p = torch.arange(self.deck_len, device=deck_tokens.device)
            xd = xd + self.deck_pos_emb(p)[None, :, :] + self.deck_type
            x = torch.cat([xd, xf], dim=1)
        else:
            x = xf

        x = self.drop(x)
        for blk in self.enc_blocks:
            x = blk(x)
        return self.enc_ln(x)

    def decode(self, dec_in: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        B, T = dec_in.shape
        if T > self.dec_block_size:
            raise ValueError("decoder input too long")
        x = self.tok_emb(dec_in)
        pos = torch.arange(T, device=dec_in.device)
        x = self.drop(x + self.dec_pos_emb(pos)[None, :, :])
        for blk in self.dec_blocks:
            x = blk(x, mem)
        x = self.dec_ln(x)
        return self.lm_head(x)


def infer_vocab_size_from_state(sd: dict) -> Optional[int]:
    for k in ("tok_emb.weight", "module.tok_emb.weight"):
        if k in sd and isinstance(sd[k], torch.Tensor):
            return int(sd[k].shape[0])
    for k, v in sd.items():
        if k.endswith("tok_emb.weight") and isinstance(v, torch.Tensor):
            return int(v.shape[0])
    return None

def infer_dec_block_from_state(sd: dict) -> Optional[int]:
    for k in ("dec_pos_emb.weight", "module.dec_pos_emb.weight"):
        if k in sd and isinstance(sd[k], torch.Tensor):
            return int(sd[k].shape[0])
    for k, v in sd.items():
        if k.endswith("dec_pos_emb.weight") and isinstance(v, torch.Tensor):
            return int(v.shape[0])
    return None


@torch.inference_mode()
def sample_next_tokens_rowwise_greedy(
    model: EncDecGPT,
    enc_tokens_1d: np.ndarray,
    deck_tokens_1d: Optional[np.ndarray],
    *,
    Hq: int,
    Wq: int,
    n_embed: int,
    bos_id: int,
    eos_id: int,
    row_id: int,
    device: torch.device,
    max_gen_len: int,
) -> np.ndarray:
    model.eval()
    enc = torch.from_numpy(enc_tokens_1d.astype(np.int64))[None, :].to(device)

    deck = None
    if deck_tokens_1d is not None:
        deck = torch.from_numpy(deck_tokens_1d.astype(np.int64))[None, :].to(device)

    mem = model.encode(enc, deck_tokens=deck)

    out = torch.tensor([[int(bos_id)]], device=device, dtype=torch.long)
    for _ in range(max_gen_len - 1):
        ctx = out[:, -model.dec_block_size:] if out.size(1) > model.dec_block_size else out
        logits = model.decode(ctx, mem)[:, -1, :]
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
        if int(nxt.item()) == eos_id:
            break

    seq = out[0].detach().cpu().numpy().astype(np.int64)

    codes = []
    for tok in seq[1:]:
        if tok == eos_id:
            break
        if tok == row_id:
            continue
        if 0 <= tok < n_embed:
            codes.append(int(tok))
        if len(codes) >= Hq * Wq:
            break

    need = Hq * Wq
    if len(codes) < need:
        fill = codes[-1] if len(codes) > 0 else 0
        codes.extend([fill] * (need - len(codes)))

    return np.asarray(codes[:need], dtype=np.int64)


def _stable_int_from_str(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def load_gpt_bundle(ckpt_path: str, device: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu")
    if "model" not in ck:
        raise RuntimeError(f"GPT checkpoint missing 'model'. Keys: {list(ck.keys())}")
    sd = ck["model"]
    cfg = ck.get("cfg", {})

    vocab_size = int(ck.get("vocab_size", infer_vocab_size_from_state(sd)))
    dec_block = int(ck.get("dec_block", infer_dec_block_from_state(sd)))
    if vocab_size is None or dec_block is None:
        raise RuntimeError("Could not infer vocab_size/dec_block from checkpoint.")

    n_embed = int(ck["n_embed"])
    Hq = int(ck["Hq"])
    Wq = int(ck["Wq"])

    bos_id = int(ck.get("bos_id", n_embed))
    eos_id = int(ck.get("eos_id", n_embed + 1))
    row_id = int(ck.get("row_id", n_embed + 2))
    pad_id = int(ck.get("pad_id", n_embed + 3))

    deck_len = int(ck.get("deck_len", 0))
    deck_bins = int(ck.get("deck_bins", 0))
    deck_keys = list(ck.get("deck_keys", []))
    deck_key_stats = ck.get("deck_key_stats", ck.get("deck_key_stats_saved", ck.get("deck_key_stats", None)))
    if deck_key_stats is None:
        deck_key_stats = ck.get("deck_key_stats", {})
    # training saved it as dict k -> ("num", lo, hi) or ("cat", None, None)
    key_stats = {k: tuple(v) for k, v in (deck_key_stats or {}).items()}

    model = EncDecGPT(
        vocab_size=vocab_size,
        n_embed=n_embed,
        Hq=Hq, Wq=Wq,
        deck_len=deck_len,
        dec_block_size=dec_block,
        n_layer_enc=int(cfg.get("n_layer_enc", 6)),
        n_layer_dec=int(cfg.get("n_layer_dec", 12)),
        n_head=int(cfg.get("n_head", 8)),
        n_embd=int(cfg.get("n_embd", 768)),
        dropout=float(cfg.get("dropout", 0.0)),
        pad_id=pad_id,
        use_sdpa=bool(cfg.get("use_sdpa", False)),
    ).to(device).eval()

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) or len(unexpected):
        print(f">> WARNING load_state_dict: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    meta = dict(
        ckpt_path=ckpt_path,
        cfg=cfg,
        vocab_size=vocab_size,
        n_embed=n_embed,
        Hq=Hq, Wq=Wq,
        bos_id=bos_id, eos_id=eos_id, row_id=row_id, pad_id=pad_id,
        dec_block=dec_block,
        deck_len=deck_len,
        deck_bins=deck_bins,
        deck_keys=deck_keys,
        key_stats=key_stats,
    )
    return model, meta


def worker(rank: int, gpu_id: int, indices: List[int], out_dir: str, args_dict: dict):
    args = argparse.Namespace(**args_dict)

    if gpu_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    set_full_determinism(int(args.seed) + int(rank))

    gpt, meta = load_gpt_bundle(args.gpt_ckpt, device=device)

    # Build dataset inside each worker (cheap, avoids pickling issues)
    with open(args.datasets_json, "r") as f:
        reg = json.load(f)

    extra = {"seed": int(args.seed), "max_runs": int(args.max_runs) if int(args.max_runs) > 0 else None}

    # If ckpt has deck, enforce ckpt deck schema into dataset so rollouts match training
    if meta["deck_len"] > 0:
        extra.update(dict(
            use_deck=True,
            deck_keys=meta["deck_keys"][: meta["deck_len"]],
            key_stats=meta["key_stats"],
            deck_bins=int(meta["deck_bins"]),
        ))
    else:
        extra.update(dict(use_deck=False))

    ds = build_dataset_from_registry(reg, name=args.dataset_name, split=args.split, extra_kwargs=extra)

    # sanity match
    if int(getattr(ds, "Hq")) != int(meta["Hq"]) or int(getattr(ds, "Wq")) != int(meta["Wq"]) or int(getattr(ds, "n_embed")) != int(meta["n_embed"]):
        raise RuntimeError(f"Dataset (Hq,Wq,n_embed)=({ds.Hq},{ds.Wq},{ds.n_embed}) != ckpt=({meta['Hq']},{meta['Wq']},{meta['n_embed']})")

    T = int(getattr(ds, "T"))
    Hq = int(meta["Hq"]); Wq = int(meta["Wq"]); n_embed = int(meta["n_embed"])

    steps = int(args.steps)
    if steps <= 0:
        steps = T - 1
    steps = min(steps, T - 1)

    max_gen_len = int(args.max_gen_len) if int(args.max_gen_len) > 0 else int(meta["dec_block"]) + 1

    outputs = []
    for di in indices:
        ex = ds[di]
        run_id = str(ex["run_id"])
        src_fp = str(ex.get("src_tokens_npz", ""))

        tokens = ex["tokens"].detach().cpu().numpy().astype(np.int64)  # (T,Hq,Wq)
        deck = ex["deck"].detach().cpu().numpy().astype(np.int64)
        deck_tok = None if deck.size == 0 else deck

        out_fp = os.path.join(out_dir, f"{args.split}_{run_id}_pred.npz")
        if args.resume and os.path.exists(out_fp):
            continue

        pred = np.empty((steps + 1, Hq, Wq), dtype=np.int16)
        pred[0] = tokens[0].astype(np.int16)
        cur = tokens[0].reshape(-1).copy()

        for t in range(1, steps + 1):
            nxt = sample_next_tokens_rowwise_greedy(
                gpt, cur, deck_tok,
                Hq=Hq, Wq=Wq, n_embed=n_embed,
                bos_id=meta["bos_id"], eos_id=meta["eos_id"], row_id=meta["row_id"],
                device=device, max_gen_len=max_gen_len,
            )
            cur = nxt
            pred[t] = cur.reshape(Hq, Wq).astype(np.int16)

        np.savez_compressed(
            out_fp,
            pred_tokens=pred,
            run_id=np.array(run_id),
            src_tokens_npz=np.array(src_fp),
            steps=np.int32(steps),
            Hq=np.int32(Hq), Wq=np.int32(Wq), n_embed=np.int32(n_embed),
            gpt_ckpt=np.array(args.gpt_ckpt),
            deck_used=np.int32(1 if deck_tok is not None else 0),
            seed=np.int32(args.seed),
        )
        outputs.append(out_fp)
        print(f">> [rank{rank}] saved {out_fp}", flush=True)

    man = os.path.join(out_dir, f"_manifest_rank{rank}.json")
    with open(man, "w") as f:
        json.dump(dict(rank=rank, gpu_id=gpu_id, outputs=outputs), f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_json", required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--gpt_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--gpus", default="", help="Comma-separated GPU ids to use, e.g. '0,1,2,3'. Empty => cuda:0 if available.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_runs", type=int, default=-1, help="Limit number of runs (trajectory items). -1=all.")
    ap.add_argument("--resume", action="store_true", help="Skip outputs already present.")

    ap.add_argument("--steps", type=int, default=0, help="Number of rollout steps (0 => T-1). Saves steps+1 frames starting at t0=0.")
    ap.add_argument("--max_gen_len", type=int, default=0, help="Override generation length. 0 => dec_block+1 (recommended).")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # decide GPUs
    if args.gpus.strip():
        gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    else:
        gpu_ids = [0] if torch.cuda.is_available() else [-1]

    # build dataset once on CPU to get N and index list
    with open(args.datasets_json, "r") as f:
        reg = json.load(f)

    # build ckpt to enforce deck schema
    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, meta = load_gpt_bundle(args.gpt_ckpt, device=device0)

    extra = {"seed": int(args.seed), "max_runs": int(args.max_runs) if int(args.max_runs) > 0 else None}
    if meta["deck_len"] > 0:
        extra.update(dict(
            use_deck=True,
            deck_keys=meta["deck_keys"][: meta["deck_len"]],
            key_stats=meta["key_stats"],
            deck_bins=int(meta["deck_bins"]),
        ))
    else:
        extra.update(dict(use_deck=False))

    ds0 = build_dataset_from_registry(reg, name=args.dataset_name, split=args.split, extra_kwargs=extra)
    N = len(ds0)
    idxs = list(range(N))
    print(f">> Dataset={args.dataset_name} split={args.split} runs={N} using {len(gpu_ids)} proc(s): {gpu_ids}", flush=True)

    # shard indices across workers
    shards = [[] for _ in range(len(gpu_ids))]
    for i, di in enumerate(idxs):
        shards[i % len(gpu_ids)].append(di)

    # save meta
    meta_fp = os.path.join(args.out_dir, "rollout_meta.json")
    with open(meta_fp, "w") as f:
        json.dump(vars(args), f, indent=2)

    # spawn workers
    ctx = mp.get_context("spawn")
    procs = []
    args_dict = vars(args)

    for rank in range(len(gpu_ids)):
        p = ctx.Process(target=worker, args=(rank, gpu_ids[rank], shards[rank], args.out_dir, args_dict))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"A worker exited with code {p.exitcode}")

    # merge manifests
    out_files = []
    for rank in range(len(gpu_ids)):
        man = os.path.join(args.out_dir, f"_manifest_rank{rank}.json")
        with open(man, "r") as f:
            out_files.extend(json.load(f)["outputs"])
    out_files = sorted(set(out_files))

    index_fp = os.path.join(args.out_dir, "rollouts_index.json")
    with open(index_fp, "w") as f:
        json.dump(dict(outputs=out_files, meta=vars(args)), f, indent=2)

    print(f">> Done. Wrote index: {index_fp}", flush=True)


if __name__ == "__main__":
    main()
