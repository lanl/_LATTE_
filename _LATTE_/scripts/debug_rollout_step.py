#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug one step of decoding to see whether the model is trying to output special tokens
(ROW/EOS/PAD/etc.) at positions where we assume it's outputting codes.

Usage:
  python scripts/debug_rollout_step.py \
    --gpt_ckpt /.../gpt_best.pt \
    --tokens_npz /path/to/test_example_tokens.npz \
    --t 0
"""

from __future__ import annotations
import os, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Model (must match your Stage A)
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
    def __init__(self, vocab_size, n_embed, Hq, Wq, deck_len, dec_block_size,
                 n_layer_enc, n_layer_dec, n_head, n_embd, dropout, pad_id, use_sdpa):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n_embed = int(n_embed)
        self.Hq, self.Wq = int(Hq), int(Wq)
        self.enc_len_frame = self.Hq * self.Wq
        self.deck_len = int(deck_len)
        self.dec_block_size = int(dec_block_size)
        self.pad_id = int(pad_id)

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

    def encode(self, enc_tokens, deck_tokens=None):
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

    def decode(self, dec_in, mem):
        B, T = dec_in.shape
        if T > self.dec_block_size:
            raise ValueError(f"Decoder length {T} > dec_block_size {self.dec_block_size}")
        x = self.tok_emb(dec_in)
        pos = torch.arange(T, device=dec_in.device)
        x = self.drop(x + self.dec_pos_emb(pos)[None, :, :])
        for blk in self.dec_blocks:
            x = blk(x, mem)
        x = self.dec_ln(x)
        return self.lm_head(x)

def infer_vocab_size_from_state(sd: dict):
    for k in ("tok_emb.weight", "module.tok_emb.weight"):
        if k in sd and isinstance(sd[k], torch.Tensor):
            return int(sd[k].shape[0])
    for k, v in sd.items():
        if k.endswith("tok_emb.weight") and isinstance(v, torch.Tensor):
            return int(v.shape[0])
    return None

def infer_dec_block_from_state(sd: dict):
    for k in ("dec_pos_emb.weight", "module.dec_pos_emb.weight"):
        if k in sd and isinstance(sd[k], torch.Tensor):
            return int(sd[k].shape[0])
    for k, v in sd.items():
        if k.endswith("dec_pos_emb.weight") and isinstance(v, torch.Tensor):
            return int(v.shape[0])
    return None


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt_ckpt", required=True)
    ap.add_argument("--tokens_npz", required=True)
    ap.add_argument("--t", type=int, default=0, help="condition on GT[t] and probe predicting GT[t+1]")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    z = np.load(args.tokens_npz, allow_pickle=True)
    tokens = z["tokens"].astype(np.int64)  # (T,Hq,Wq)
    T, Hq, Wq = tokens.shape
    n_embed = int(z["n_embed"])
    assert args.t >= 0 and args.t < T-1, f"t must be in [0,{T-2}]"

    ck = torch.load(args.gpt_ckpt, map_location="cpu")
    sd = ck["model"]
    cfg = ck.get("cfg", {}) or {}

    vocab_size = int(ck.get("vocab_size", infer_vocab_size_from_state(sd)))
    dec_block = int(infer_dec_block_from_state(sd))

    bos_id = int(ck.get("bos_id", n_embed))
    eos_id = int(ck.get("eos_id", n_embed + 1))
    row_id = int(ck.get("row_id", n_embed + 2))
    pad_id = int(ck.get("pad_id", n_embed + 3))

    model = EncDecGPT(
        vocab_size=vocab_size,
        n_embed=n_embed,
        Hq=Hq, Wq=Wq,
        deck_len=int(ck.get("deck_len", 0)),
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
    if missing or unexpected:
        print("missing:", missing[:10])
        print("unexpected:", unexpected[:10])
        raise RuntimeError("state_dict mismatch")

    # Build encoder input from GT[t]
    enc_codes = tokens[args.t].reshape(-1)
    enc = torch.from_numpy(enc_codes[None, :]).to(device)

    mem = model.encode(enc, deck_tokens=None)

    # Build the *exact* decoder prefix you assume during rollout:
    # BOS, then codes, inserting ROW between rows (no EOS)
    prefix = [bos_id]
    gt_next_codes = tokens[args.t + 1].astype(np.int64)  # (Hq,Wq)
    for r in range(Hq):
        for c in range(Wq):
            prefix.append(int(gt_next_codes[r, c]))
        if r != Hq - 1:
            prefix.append(row_id)

    # Now we will walk the prefix and measure what the model thinks the next token is.
    # At code positions we EXPECT a code. At row-boundary positions we EXPECT ROW.
    prefix_t = torch.tensor(prefix[:-1], device=device, dtype=torch.long)[None, :]  # exclude final token
    logits = model.decode(prefix_t[:, -model.dec_block_size:], mem)  # (1,L,V)
    # logits correspond to each position predicting the next token; we want the last-step for each prefix position
    # easiest: shift compare token-by-token
    pred = torch.argmax(logits[0], dim=-1).detach().cpu().numpy()  # (L,)

    target = np.array(prefix[1:], dtype=np.int64)  # next-token targets aligned with pred

    # Stats
    acc = (pred == target).mean()
    # How often full-vocab argmax is a code when target is a code?
    is_code_target = (target >= 0) & (target < n_embed)
    code_top1_is_code = ((pred >= 0) & (pred < n_embed))[is_code_target].mean() if is_code_target.any() else float("nan")
    # How often full-vocab argmax matches ROW where target is ROW?
    is_row_target = (target == row_id)
    row_acc = (pred[is_row_target] == row_id).mean() if is_row_target.any() else float("nan")

    print(f"T,Hq,Wq={T},{Hq},{Wq} n_embed={n_embed} vocab_size={vocab_size} dec_block={dec_block}")
    print(f"bos/eos/row/pad = {bos_id}/{eos_id}/{row_id}/{pad_id}")
    print(f"Teacher-forced next-token accuracy over the *assumed* format: {acc:.4f}")
    print(f"At CODE targets: fraction where model top1 is actually a CODE: {code_top1_is_code:.4f}")
    print(f"At ROW targets:  accuracy predicting ROW: {row_acc:.4f}")

    # If the model is trying to output special tokens at code positions, show which.
    if is_code_target.any():
        bad = pred[is_code_target]
        specials = bad[(bad < 0) | (bad >= n_embed)]
        if specials.size > 0:
            uniq, cnt = np.unique(specials, return_counts=True)
            print("Top special-token argmaxes at CODE positions (token_id: count):")
            for u, c in zip(uniq.tolist(), cnt.tolist()):
                print(f"  {u}: {c}")
        else:
            print("Model top1 is always a code at code positions (good sign).")

if __name__ == "__main__":
    main()
