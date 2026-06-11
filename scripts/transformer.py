#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic Enc-Dec Transformer over VQ tokens: predict frame t -> t+1.

Dataset interface (must return 4-tuple):
  deck:       (deck_len,) int64   (empty tensor if no deck)
  enc_tokens: (Hq*Wq,)    int64
  dec_in:     (Ldec-1,)   int64
  dec_tgt:    (Ldec-1,)   int64

Dataset must expose attributes (at least):
  n_embed, Hq, Wq, dec_block, vocab_size, deck_len, pad_id, bos_id, eos_id, row_id
And (if deck enabled):
  deck_base, deck_vocab_size, deck_bins, deck_keys, key_stats  (optional but recommended)

This script selects datasets using a registry JSON (like your VQ-VAE trainer).
"""

from __future__ import annotations

import os, argparse, math, time, random, json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler

from src.data.transformer_registry import build_transformer_splits_from_registry


# -----------------------------
# DDP helpers
# -----------------------------
def ddp_is_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def ddp_setup():
    if not ddp_is_enabled():
        return 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def ddp_cleanup():
    if ddp_is_enabled() and dist.is_initialized():
        dist.destroy_process_group()

def is_rank0(rank: int) -> bool:
    return rank == 0

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# AMP helpers
# -----------------------------
def make_scaler(enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=enabled)

def autocast_ctx(enabled: bool):
    return torch.amp.autocast("cuda", enabled=enabled)


# -----------------------------
# Transformer blocks
# -----------------------------
class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

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
        y = self.resid_drop(self.proj(y))
        return y

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
        y = self.resid_drop(self.proj(y))
        return y

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

        # Encoder 2D pos (frame)
        self.row_emb = nn.Embedding(self.Hq, n_embd)
        self.col_emb = nn.Embedding(self.Wq, n_embd)

        # Encoder deck pos (1D)
        self.deck_pos_emb = nn.Embedding(max(1, self.deck_len), n_embd)

        # Segment/type embeddings (learned)
        self.deck_type = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.frame_type = nn.Parameter(torch.zeros(1, 1, n_embd))

        # Decoder 1D pos
        self.dec_pos_emb = nn.Embedding(self.dec_block_size, n_embd)

        self.drop = nn.Dropout(dropout)

        self.enc_blocks = nn.ModuleList([EncoderBlock(n_embd, n_head, dropout, use_sdpa=use_sdpa) for _ in range(n_layer_enc)])
        self.dec_blocks = nn.ModuleList([DecoderBlock(n_embd, n_head, dropout, use_sdpa=use_sdpa) for _ in range(n_layer_dec)])

        self.enc_ln = nn.LayerNorm(n_embd)
        self.dec_ln = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, self.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def encode(self, enc_tokens: torch.Tensor, deck_tokens: torch.Tensor | None = None) -> torch.Tensor:
        B, L = enc_tokens.shape
        assert L == self.enc_len_frame, f"enc length {L} != Hq*Wq {self.enc_len_frame}"

        xf = self.tok_emb(enc_tokens)  # (B,L,C)

        r = torch.arange(self.Hq, device=enc_tokens.device).view(self.Hq, 1).expand(self.Hq, self.Wq).reshape(-1)
        c = torch.arange(self.Wq, device=enc_tokens.device).view(1, self.Wq).expand(self.Hq, self.Wq).reshape(-1)
        pos2d = self.row_emb(r)[None, :, :] + self.col_emb(c)[None, :, :]
        xf = xf + pos2d + self.frame_type

        if deck_tokens is not None and self.deck_len > 0:
            Bd, D = deck_tokens.shape
            assert Bd == B and D == self.deck_len, f"deck shape {deck_tokens.shape} expected {(B, self.deck_len)}"
            xd = self.tok_emb(deck_tokens)
            p = torch.arange(self.deck_len, device=deck_tokens.device)
            xd = xd + self.deck_pos_emb(p)[None, :, :] + self.deck_type
            x = torch.cat([xd, xf], dim=1)  # (B, D+L, C)
        else:
            x = xf

        x = self.drop(x)
        for blk in self.enc_blocks:
            x = blk(x)
        x = self.enc_ln(x)
        return x

    def decode(self, dec_in: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        B, T = dec_in.shape
        if T > self.dec_block_size:
            raise ValueError(f"Decoder length {T} > dec_block_size {self.dec_block_size}")

        x = self.tok_emb(dec_in)
        pos = torch.arange(T, device=dec_in.device)
        x = self.drop(x + self.dec_pos_emb(pos)[None, :, :])

        for blk in self.dec_blocks:
            x = blk(x, mem)
        x = self.dec_ln(x)
        logits = self.lm_head(x)
        return logits

    def forward(self, enc_tokens: torch.Tensor, dec_in: torch.Tensor, dec_tgt: torch.Tensor, deck_tokens: torch.Tensor | None = None):
        mem = self.encode(enc_tokens, deck_tokens=deck_tokens)
        logits = self.decode(dec_in, mem)
        logits_f = logits.float().reshape(-1, logits.size(-1))
        tgt_f = dec_tgt.reshape(-1)
        loss = F.cross_entropy(logits_f, tgt_f, ignore_index=self.pad_id)
        return logits, loss


class SubsetWithAttrs(Subset):
    """Keep dataset attributes visible through Subset (for Hq/Wq/vocab_size/etc)."""
    def __getattr__(self, name):
        return getattr(self.dataset, name)


# -----------------------------
# LR schedule
# -----------------------------
def lr_at_step(base_lr, step, warmup_steps, total_steps, min_lr_ratio=0.1):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos)

def save_ckpt(path, model, opt, scaler, step, best_val, args, extra):
    tmp = path + ".tmp"
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(
        {
            "model": state,
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": int(step),
            "best_val": float(best_val),
            "cfg": vars(args),
            **extra,
        },
        tmp,
    )
    os.replace(tmp, path)

def load_partial_init(target_model: nn.Module, init_ckpt_path: str, *, n_embed: int):
    ck = torch.load(init_ckpt_path, map_location="cpu")
    sd = ck.get("model", ck.get("state_dict", None))
    if sd is None:
        raise RuntimeError(f"{init_ckpt_path}: missing 'model' state_dict.")

    tgt = target_model

    tgt_sd = tgt.state_dict()
    new_sd = {}

    copied = []
    skipped = []

    base_rows = int(n_embed) + 4  # codes + BOS/EOS/ROW/PAD

    for k, v in sd.items():
        if k not in tgt_sd:
            skipped.append((k, "missing_in_target"))
            continue

        tv = tgt_sd[k]

        # Special-case token embedding and lm_head: copy only base rows
        if k.endswith("tok_emb.weight") or k.endswith("lm_head.weight"):
            if v.ndim == 2 and tv.ndim == 2 and v.shape[1] == tv.shape[1]:
                rows = min(base_rows, v.shape[0], tv.shape[0])
                out = tv.clone()
                out[:rows] = v[:rows]
                new_sd[k] = out
                copied.append((k, f"partial_rows={rows}/{tv.shape[0]}"))
            else:
                skipped.append((k, f"shape_mismatch {tuple(v.shape)} -> {tuple(tv.shape)}"))
            continue

        # Everything else: exact shape match only
        if tuple(v.shape) == tuple(tv.shape):
            new_sd[k] = v
            copied.append((k, "full"))
        else:
            skipped.append((k, f"shape_mismatch {tuple(v.shape)} -> {tuple(tv.shape)}"))

    # Load with strict=False so missing keys are fine
    missing, unexpected = tgt.load_state_dict(new_sd, strict=False)

    print(f">> init_from={init_ckpt_path}")
    print(f">> copied: {len(copied)} params  skipped: {len(skipped)}")
    print(f">> load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    # Optional: print a few informative skips
    for k, why in skipped[:10]:
        print(f"   skip: {k} ({why})")


def main():
    ap = argparse.ArgumentParser()

    # --- dataset selection ---
    ap.add_argument("--datasets_json", required=True, help="Registry JSON for transformer token datasets.")
    ap.add_argument("--dataset_name", required=True, help="Dataset key inside the registry JSON.")

    # --- output/resume ---
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--resume", type=str, default=None)

    # --- dataloader ---
    ap.add_argument("--batch_size", type=int, default=12, help="Per-GPU batch size (DDP).")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_pairs", type=int, default=None, help="Optional cap on training (run,t) pairs (dataset-dependent).")

    # --- training ---
    ap.add_argument("--max_steps", type=int, default=200000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=2000)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=3000)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # --- model ---
    ap.add_argument("--n_layer_enc", type=int, default=6)
    ap.add_argument("--n_layer_dec", type=int, default=12)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--n_embd", type=int, default=768)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_sdpa", action="store_true")

    # --- deck controls ---
    ap.add_argument("--no_deck_tokens", action="store_true", help="Disable deck conditioning (baseline).")
    ap.add_argument("--deck_ablation", type=str, default="none", choices=["none", "random", "pad"],
                    help="Deck ablations when deck conditioning enabled.")
    ap.add_argument("--deck_dropout", type=float, default=0.0, help="Train-time only: with prob p, replace deck conditioning with PAD tokens (classifier-free deck dropout).")

    # --- sanity split ---
    ap.add_argument("--in_dist_val", action="store_true",
                    help="Make val set by splitting train pairs (sanity check). Overrides registry val split.")
    ap.add_argument("--in_dist_val_frac", type=float, default=0.05,
                    help="Fraction of train pairs to reserve for in-distribution val when --in_dist_val.")
        

    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--init_from", type=str, default=None,
                    help="Initialize weights from another GPT checkpoint (weights-only partial load). Useful for cross-dataset finetuning.")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rank, world_size, local_rank = ddp_setup()

    if args.no_deck_tokens and args.deck_ablation != "none":
        raise ValueError("--deck_ablation requires deck conditioning; do not use with --no_deck_tokens.")

    try:
        seed_all(args.seed + rank)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        use_amp = torch.cuda.is_available()
        scaler = make_scaler(enabled=use_amp)

        # ----------------------------
        # Build datasets from registry
        # ----------------------------
        with open(args.datasets_json, "r") as f:
            reg = json.load(f)

        extra_train = {"seed": args.seed}
        extra_val = {"seed": args.seed}

        if args.max_pairs is not None:
            extra_train["max_pairs"] = int(args.max_pairs)

        if args.no_deck_tokens:
            extra_train["use_deck"] = False
            extra_val["use_deck"] = False

        ds_tr, ds_va, vocab_spec = build_transformer_splits_from_registry(
            reg, name=args.dataset_name,
            extra_train_kwargs=extra_train,
            extra_val_kwargs=extra_val,
        )

        # Optional in-dist val split (sanity)
        if args.in_dist_val:
            n_total = len(ds_tr)
            n_val = max(1, int(round(args.in_dist_val_frac * n_total)))
            n_train = n_total - n_val
            g = torch.Generator().manual_seed(args.seed)
            tr_subset, va_subset = random_split(ds_tr, [n_train, n_val], generator=g)
            ds_tr = SubsetWithAttrs(ds_tr, tr_subset.indices)
            ds_va = SubsetWithAttrs(ds_tr.dataset, va_subset.indices)  # keep attrs
            if is_rank0(rank):
                print(f">> in_dist_val split: train_pairs={len(ds_tr)} val_pairs={len(ds_va)}", flush=True)

        # Deck sanity
        deck_len_model = int(getattr(ds_tr, "deck_len", 0))
        if deck_len_model == 0 and args.deck_ablation != "none":
            raise ValueError("deck_ablation requested but dataset has deck_len=0. Enable deck or set --deck_ablation none.")

        # Vocab comes from dataset
        vocab_size = int(getattr(ds_tr, "vocab_size"))
        n_embed = int(getattr(ds_tr, "n_embed"))
        Hq = int(getattr(ds_tr, "Hq"))
        Wq = int(getattr(ds_tr, "Wq"))
        dec_block = int(getattr(ds_tr, "dec_block"))
        pad_id = int(getattr(ds_tr, "pad_id"))

        # Dataloaders
        tr_sampler = DistributedSampler(ds_tr, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if ddp_is_enabled() else None
        dl_tr = DataLoader(
            ds_tr,
            batch_size=args.batch_size,
            sampler=tr_sampler,
            shuffle=(tr_sampler is None),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=2,
        )

        dl_va = None
        if is_rank0(rank):
            dl_va = DataLoader(
                ds_va,
                batch_size=max(1, args.batch_size),
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
                persistent_workers=(args.num_workers > 0),
                prefetch_factor=2,
            )

        # Model
        model = EncDecGPT(
            vocab_size=vocab_size,
            n_embed=n_embed,
            Hq=Hq,
            Wq=Wq,
            deck_len=deck_len_model,
            dec_block_size=dec_block,
            n_layer_enc=args.n_layer_enc,
            n_layer_dec=args.n_layer_dec,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            pad_id=pad_id,
            use_sdpa=args.use_sdpa,
        ).to(device)

        if is_rank0(rank):
            n_params = sum(p.numel() for p in model.parameters())
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f">> params total={n_params:,} ({n_params/1e6:.2f}M)  trainable={n_trainable:,} ({n_trainable/1e6:.2f}M)", flush=True)

        if args.init_from is not None and args.resume is None:
            # cross-dataset init (do this BEFORE DDP wrapping)
            load_partial_init(model, args.init_from, n_embed=ds_tr.n_embed)

        opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

        ddp_find_unused = (deck_len_model == 0)
        if ddp_is_enabled():
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=ddp_find_unused,
            )

        def make_deck_input(deck_from_loader: torch.Tensor, *, batch_size: int, train_mode: bool):
            if deck_len_model == 0:
                return None

            # Train-time classifier-free deck dropout: replace with PAD deck
            if train_mode and args.deck_dropout > 0.0 and args.deck_ablation == "none":
                if torch.rand((), device=device).item() < float(args.deck_dropout):
                    return torch.full(
                        (batch_size, deck_len_model),
                        fill_value=int(getattr(ds_tr, "pad_id")),
                        device=device,
                        dtype=torch.long,
                    )

            # Usual paths
            if args.deck_ablation == "none":
                return deck_from_loader.to(device, non_blocking=True)

            if args.deck_ablation == "pad":
                return torch.full(
                    (batch_size, deck_len_model),
                    fill_value=int(getattr(ds_tr, "pad_id")),
                    device=device,
                    dtype=torch.long,
                )

            if args.deck_ablation == "random":
                low = int(getattr(ds_tr, "deck_base"))
                high = int(getattr(ds_tr, "deck_base") + getattr(ds_tr, "deck_vocab_size"))
                return torch.randint(low=low, high=high, size=(batch_size, deck_len_model), device=device, dtype=torch.long)

            raise RuntimeError(f"Unknown deck_ablation: {args.deck_ablation}")

        # Resume
        best_val = float("inf")
        step = 0
        micro_step = 0

        if args.resume is not None:
            ck = torch.load(args.resume, map_location="cpu")
            target = model.module if hasattr(model, "module") else model
            missing, unexpected = target.load_state_dict(ck["model"], strict=False)
            if is_rank0(rank):
                print(f">> RESUME load_state_dict: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

            if ck.get("optimizer") is not None:
                opt.load_state_dict(ck["optimizer"])
            if use_amp and ck.get("scaler") is not None:
                scaler.load_state_dict(ck["scaler"])

            step = int(ck.get("step", 0))
            micro_step = step * args.grad_accum
            best_val = float(ck.get("best_val", best_val))

            # Basic compat checks
            if int(ck.get("vocab_size", vocab_size)) != int(vocab_size):
                raise RuntimeError("Resume vocab_size mismatch.")
            if int(ck.get("Hq", Hq)) != int(Hq) or int(ck.get("Wq", Wq)) != int(Wq):
                raise RuntimeError("Resume Hq/Wq mismatch.")
            if int(ck.get("dec_block", dec_block)) != int(dec_block):
                raise RuntimeError("Resume dec_block mismatch.")
            if int(ck.get("deck_len", deck_len_model)) != int(deck_len_model):
                raise RuntimeError("Resume deck_len mismatch.")

            if is_rank0(rank):
                print(f">> RESUME {args.resume}", flush=True)
                print(f">> start_step={step} best_val={best_val:.6f}", flush=True)

        if is_rank0(rank):
            print(f">> Dataset: {args.dataset_name} from {args.datasets_json}", flush=True)
            print(f">> vocab_size={vocab_size}  n_embed={n_embed}  Hq={Hq} Wq={Wq}  deck_len={deck_len_model}", flush=True)
            if hasattr(ds_tr, "deck_bins"):
                print(f">> deck_bins={int(getattr(ds_tr, 'deck_bins'))}", flush=True)
            if hasattr(ds_tr, "deck_keys") and deck_len_model > 0:
                print(f">> deck_keys[:6]={list(getattr(ds_tr, 'deck_keys'))[:6]}", flush=True)
            print(f">> DDP world={world_size} rank={rank} local_rank={local_rank} amp={use_amp} use_sdpa={args.use_sdpa}", flush=True)
            print(f">> effective_batch = batch({args.batch_size}) * world({world_size}) * accum({args.grad_accum})", flush=True)
            print(f">> deck_ablation={args.deck_ablation}  no_deck_tokens={args.no_deck_tokens}", flush=True)

        # Eval
        def evaluate(max_batches=200):
            if not is_rank0(rank):
                return None
            model.eval()
            tot, n = 0.0, 0
            with torch.no_grad():
                for deck, enc, din, dtgt in dl_va:
                    enc = enc.to(device, non_blocking=True)
                    din = din.to(device, non_blocking=True)
                    dtgt = dtgt.to(device, non_blocking=True)

                    deck_in = make_deck_input(deck, batch_size=enc.size(0), train_mode=True)

                    with autocast_ctx(enabled=use_amp):
                        _, loss = model(enc, din, dtgt, deck_tokens=deck_in)
                    tot += float(loss.item()); n += 1
                    if n >= max_batches:
                        break
            model.train()
            return tot / max(1, n)

        # Train loop
        model.train()
        t0 = time.time()
        epoch = step
        if tr_sampler is not None:
            tr_sampler.set_epoch(epoch)

        while step < args.max_steps:
            if tr_sampler is not None:
                tr_sampler.set_epoch(epoch)

            for deck, enc, din, dtgt in dl_tr:
                enc = enc.to(device, non_blocking=True)
                din = din.to(device, non_blocking=True)
                dtgt = dtgt.to(device, non_blocking=True)

                deck_in = make_deck_input(deck, batch_size=enc.size(0), train_mode=False)

                lr = lr_at_step(args.lr, step, args.warmup_steps, args.max_steps, min_lr_ratio=args.min_lr_ratio)
                for pg in opt.param_groups:
                    pg["lr"] = lr

                with autocast_ctx(enabled=use_amp):
                    _, loss = model(enc, din, dtgt, deck_tokens=deck_in)
                    loss = loss / args.grad_accum

                scaler.scale(loss).backward()
                micro_step += 1

                if (micro_step % args.grad_accum) == 0:
                    scaler.unscale_(opt)
                    if args.grad_clip and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                    if (step % args.eval_every) == 0:
                        if ddp_is_enabled():
                            dist.barrier()
                        val = evaluate()
                        if ddp_is_enabled():
                            dist.barrier()

                        if is_rank0(rank):
                            train_loss = float(loss.item() * args.grad_accum)
                            dt = (time.time() - t0) / 60.0
                            print(f"[{step:>7}] lr={lr:.3e} train_loss={train_loss:.4f} val_loss={val:.4f} ({dt:.1f} min)", flush=True)

                            extra = dict(
                                dataset_name=str(args.dataset_name),
                                datasets_json=str(args.datasets_json),
                                vocab_size=int(vocab_size),
                                n_embed=int(n_embed),
                                Hq=int(Hq),
                                Wq=int(Wq),
                                dec_block=int(dec_block),
                                bos_id=int(getattr(ds_tr, "bos_id")),
                                eos_id=int(getattr(ds_tr, "eos_id")),
                                row_id=int(getattr(ds_tr, "row_id")),
                                pad_id=int(getattr(ds_tr, "pad_id")),
                                deck_len=int(deck_len_model),
                                deck_ablation=str(args.deck_ablation),
                            )

                            # optional deck vocab state
                            if hasattr(ds_tr, "deck_bins"):
                                extra["deck_bins"] = int(getattr(ds_tr, "deck_bins"))
                            if hasattr(ds_tr, "deck_vocab_size"):
                                extra["deck_vocab_size"] = int(getattr(ds_tr, "deck_vocab_size"))
                            if hasattr(ds_tr, "deck_keys"):
                                extra["deck_keys"] = list(getattr(ds_tr, "deck_keys"))
                            if hasattr(ds_tr, "key_stats"):
                                extra["deck_key_stats"] = dict(getattr(ds_tr, "key_stats"))

                            if val < best_val:
                                best_val = val
                                save_ckpt(os.path.join(args.out_dir, "gpt_best.pt"),
                                          model, opt, scaler, step, best_val, args, extra)
                                print(">> saved gpt_best.pt", flush=True)

                    if (step % args.save_every) == 0 and step > 0:
                        if ddp_is_enabled():
                            dist.barrier()
                        if is_rank0(rank):
                            extra = dict(
                                dataset_name=str(args.dataset_name),
                                datasets_json=str(args.datasets_json),
                                vocab_size=int(vocab_size),
                                n_embed=int(n_embed),
                                Hq=int(Hq),
                                Wq=int(Wq),
                                dec_block=int(dec_block),
                                bos_id=int(getattr(ds_tr, "bos_id")),
                                eos_id=int(getattr(ds_tr, "eos_id")),
                                row_id=int(getattr(ds_tr, "row_id")),
                                pad_id=int(getattr(ds_tr, "pad_id")),
                                deck_len=int(deck_len_model),
                                deck_ablation=str(args.deck_ablation),
                            )
                            save_ckpt(os.path.join(args.out_dir, f"gpt_step{step}.pt"),
                                      model, opt, scaler, step, best_val, args, extra)
                            print(f">> saved gpt_step{step}.pt", flush=True)

                    step += 1
                    if step >= args.max_steps:
                        break

            epoch += 1
            if step >= args.max_steps:
                break

        if ddp_is_enabled():
            dist.barrier()

        if is_rank0(rank):
            extra = dict(
                dataset_name=str(args.dataset_name),
                datasets_json=str(args.datasets_json),
                vocab_size=int(vocab_size),
                n_embed=int(n_embed),
                Hq=int(Hq),
                Wq=int(Wq),
                dec_block=int(dec_block),
                bos_id=int(getattr(ds_tr, "bos_id")),
                eos_id=int(getattr(ds_tr, "eos_id")),
                row_id=int(getattr(ds_tr, "row_id")),
                pad_id=int(getattr(ds_tr, "pad_id")),
                deck_len=int(deck_len_model),
                deck_ablation=str(args.deck_ablation),
            )
            save_ckpt(os.path.join(args.out_dir, "gpt_final.pt"),
                      model, opt, scaler, step, best_val, args, extra)
            print(">> training complete", flush=True)

    finally:
        ddp_cleanup()


if __name__ == "__main__":
    main()
