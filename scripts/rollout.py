#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic decode-and-visualize autoregressive rollout for *token NPZ + GPT + VQ*.

Works for any dataset as long as you have:
- --run_tokens_npz containing:
    tokens: (T,Hq,Wq) int
    n_embed: int
    (optional) run_dir / deck_tokens / time_start / time_count (not required for no-deck GPT)
- --gpt_ckpt from your EncDecGPT trainer (gpt_best.pt / gpt_step*.pt)
- --vq_ckpt Lightning LitVQVAE .ckpt used to produce those tokens

Output:
- A PNG grid at milestones showing:
    GT(decoded from GT codes) | Pred(decoded from GPT codes) | |Err|
"""

from __future__ import annotations

import os, argparse, math, json, zlib
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.lit_vqvae import LitVQVAE


# =============================================================================
# Optional input-deck parsing.
# =============================================================================
def _try_float(x: str):
    try:
        return float(x)
    except Exception:
        return None

def parse_clover_in(path: str) -> dict:
    out = {}
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("*"):
                continue
            parts = line.split()

            if len(parts) >= 2 and parts[0].lower() == "state":
                st = parts[1]
                prefix = f"state{st}_"
                for tok in parts[2:]:
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        out[prefix + k.strip()] = v.strip()
                continue

            any_kv = False
            for tok in parts:
                if "=" in tok:
                    any_kv = True
                    k, v = tok.split("=", 1)
                    out[k.strip()] = v.strip()

            if (not any_kv) and len(parts) == 2:
                out[parts[0].strip()] = parts[1].strip()
    return out

def deck_tokens_for_run(run_dir: str, deck_keys, key_stats, deck_bins: int, deck_base: int) -> np.ndarray:
    kv = parse_clover_in(os.path.join(run_dir, "clover.in"))
    tokens = np.empty((len(deck_keys),), dtype=np.int64)

    for ki, k in enumerate(deck_keys):
        mode, lo, hi = key_stats.get(k, ("cat", None, None))
        v = kv.get(k, None)

        if v is None:
            b = 0
        else:
            if mode == "num":
                fv = _try_float(v)
                if fv is None or (not np.isfinite(fv)):
                    b = 0
                else:
                    if lo is None or hi is None or float(hi) <= float(lo):
                        b = 1
                    else:
                        x = (float(fv) - float(lo)) / (float(hi) - float(lo))
                        x = min(1.0, max(0.0, x))
                        b = 1 + int(round(x * (deck_bins - 1)))
            else:
                h = zlib.crc32(v.encode("utf-8")) & 0xFFFFFFFF
                b = 1 + (h % deck_bins)

        tokens[ki] = int(deck_base) + ki * (deck_bins + 1) + b

    return tokens

def build_key_stats_from_gpt_ckpt(ck: dict) -> Tuple[dict, bool]:
    """
    Training script saves deck_key_stats as dict(key -> (mode, lo, hi)) or similar.
    If missing, fall back to categorical for all keys.
    """
    for key_name in ("deck_key_stats", "key_stats"):
        if key_name in ck and isinstance(ck[key_name], dict):
            ks = ck[key_name]
            out = {}
            for k, v in ks.items():
                # v might be list/tuple like ("num", lo, hi)
                if isinstance(v, (list, tuple)) and len(v) >= 3:
                    out[k] = (v[0], v[1], v[2])
            return out, True

    deck_keys = ck.get("deck_keys", [])
    out = {k: ("cat", None, None) for k in deck_keys}
    return out, False


# =============================================================================
# EncDecGPT: must match your training-time module/key layout
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
        assert L == self.enc_len_frame, f"enc length {L} != Hq*Wq {self.enc_len_frame}"

        xf = self.tok_emb(enc_tokens)

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
            raise ValueError(f"Decoder length {T} > dec_block_size {self.dec_block_size}")

        x = self.tok_emb(dec_in)
        pos = torch.arange(T, device=dec_in.device)
        x = self.drop(x + self.dec_pos_emb(pos)[None, :, :])

        for blk in self.dec_blocks:
            x = blk(x, mem)
        x = self.dec_ln(x)
        return self.lm_head(x)


# =============================================================================
# GPT sampling: rowwise separators + EOS (+ optional fixed pause tokens after BOS)
# =============================================================================
@torch.inference_mode()
def sample_next_tokens_rowwise(
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
    pause_tokens: int,
    pause_id: Optional[int],
    temperature: float,
    top_k: int,
    greedy: bool,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    enc = torch.from_numpy(enc_tokens_1d.astype(np.int64))[None, :].to(device)
    deck = None
    if deck_tokens_1d is not None:
        deck = torch.from_numpy(deck_tokens_1d.astype(np.int64))[None, :].to(device)

    mem = model.encode(enc, deck_tokens=deck)

    max_len = 1 + pause_tokens + (Hq * Wq) + (Hq - 1) + 1  # BOS + PAUSE*K + codes + ROW + EOS

    # Start sequence: BOS + fixed pause tokens (if enabled)
    if pause_tokens > 0:
        if pause_id is None:
            raise RuntimeError("pause_tokens>0 but pause_id is None.")
        out = torch.full((1, 1 + pause_tokens), int(pause_id), device=device, dtype=torch.long)
        out[0, 0] = int(bos_id)
    else:
        out = torch.tensor([[int(bos_id)]], device=device, dtype=torch.long)

    for _ in range(max_len - out.size(1)):
        ctx = out[:, -model.dec_block_size:] if out.size(1) > model.dec_block_size else out
        logits = model.decode(ctx, mem)[:, -1, :]  # (1,V)

        temp = float(temperature)
        if temp <= 0:
            greedy = True
            temp = 1.0
            top_k = 0
        logits = logits / temp

        if top_k and top_k > 0:
            k = min(int(top_k), logits.size(-1))
            v, ix = torch.topk(logits, k=k)
            logits2 = torch.full_like(logits, float("-inf"))
            logits2.scatter_(1, ix, v)
            logits = logits2

        if greedy:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)

        out = torch.cat([out, nxt], dim=1)
        if int(nxt.item()) == int(eos_id):
            break

    seq = out[0].detach().cpu().numpy().astype(np.int64)

    # Convert sequence to exactly Hq*Wq code tokens
    # Skip BOS, skip PAUSE tokens, ignore ROW, stop at EOS.
    codes = []
    for tok in seq[1:]:
        if pause_id is not None and tok == int(pause_id):
            continue
        if tok == int(eos_id):
            break
        if tok == int(row_id):
            continue
        if 0 <= tok < int(n_embed):
            codes.append(int(tok))
        if len(codes) >= Hq * Wq:
            break

    need = Hq * Wq
    if len(codes) < need:
        fill = codes[-1] if len(codes) > 0 else 0
        codes.extend([fill] * (need - len(codes)))

    return np.asarray(codes[:need], dtype=np.int64)


# =============================================================================
# VQ decode helpers (generic)
# =============================================================================
def _get_nested_attr(obj: Any, path: str):
    cur = obj
    for p in path.split("."):
        if cur is None or not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur

def _find_codebook_weight(vq: nn.Module) -> Optional[torch.Tensor]:
    # Most common: quantize.embedding.weight
    candidates = [
        "quantize.embedding.weight",
        "model.quantize.embedding.weight",
        "quantizer.embedding.weight",
        "model.quantizer.embedding.weight",
    ]
    for c in candidates:
        w = _get_nested_attr(vq, c)
        if isinstance(w, torch.Tensor):
            return w

    candidates2 = [
        "quantize.embedding",
        "model.quantize.embedding",
        "quantizer.embedding",
        "model.quantizer.embedding",
    ]
    for c in candidates2:
        emb = _get_nested_attr(vq, c)
        if emb is not None and hasattr(emb, "weight") and isinstance(emb.weight, torch.Tensor):
            return emb.weight
    return None

def _vq_decode_latents(vq: nn.Module, z_bchw: torch.Tensor) -> torch.Tensor:
    # Prefer vq.decode()
    for fn_path in ("decode", "model.decode"):
        fn = _get_nested_attr(vq, fn_path)
        if callable(fn):
            out = fn(z_bchw)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out

    # Otherwise use decoder (+ optional post_quant_conv)
    for base in (vq, getattr(vq, "model", None)):
        if base is None:
            continue
        decoder = getattr(base, "decoder", None)
        if callable(decoder):
            z = z_bchw
            pqc = getattr(base, "post_quant_conv", None)
            if callable(pqc):
                z = pqc(z)
            out = decoder(z)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out

    raise RuntimeError("Could not find a decode path on LitVQVAE (no .decode() and no .decoder()).")

def copy_ema_to_base_if_present(model: torch.nn.Module, state_dict: dict):
    model_keys = set(model.state_dict().keys())
    ema_sd = {}
    for k, v in state_dict.items():
        if k.startswith("model_ema."):
            base_k = k.replace("model_ema.", "", 1)
            if base_k in model_keys:
                ema_sd[base_k] = v
    if not ema_sd:
        print(">> VQ EMA keys not found/mappable; using raw model weights.")
        return
    missing, unexpected = model.load_state_dict(ema_sd, strict=False)
    print(f">> VQ copied EMA->base. loaded={len(ema_sd)} missing={len(missing)} unexpected={len(unexpected)}")

@torch.inference_mode()
def decode_codes_to_fields(vq: LitVQVAE, codes_hw: np.ndarray, device: torch.device) -> np.ndarray:
    """
    codes_hw: (Hq,Wq) int64 codes
    returns:  (C,H,W) float32 (in VQ's normalized space, typically [-1,1])
    """
    codes = torch.from_numpy(codes_hw.astype(np.int64)).to(device)

    # If model exposes decode_code, prefer it
    for fn_path in ("decode_code", "model.decode_code"):
        fn = _get_nested_attr(vq, fn_path)
        if callable(fn):
            out = fn(codes[None, ...])  # (1,C,H,W)
            if isinstance(out, (tuple, list)):
                out = out[0]
            x = out
            if getattr(vq, "output_clamp", False):
                x = x.clamp(-1, 1)
            return x[0].detach().float().cpu().numpy()

    w = _find_codebook_weight(vq)
    if w is None:
        raise RuntimeError("Could not locate VQ codebook embedding weight.")

    Hq, Wq = codes.shape
    z = w[codes.reshape(-1)]                             # (Hq*Wq, embed_dim)
    z = z.view(Hq, Wq, -1).permute(2, 0, 1)[None, ...]   # (1, embed_dim, Hq, Wq)

    x = _vq_decode_latents(vq, z)
    if getattr(vq, "output_clamp", False):
        x = x.clamp(-1, 1)
    return x[0].detach().float().cpu().numpy()


# =============================================================================
# Plotting
# =============================================================================
def compute_channel_clims(arrs: List[np.ndarray], channels: List[int], mode: str,
                          fixed_vmin: Optional[float], fixed_vmax: Optional[float]):
    if fixed_vmin is not None and fixed_vmax is not None:
        return {ch: (float(fixed_vmin), float(fixed_vmax)) for ch in channels}
    if mode == "per_panel":
        return {ch: (None, None) for ch in channels}
    clims = {}
    for ch in channels:
        vv = np.concatenate([a[ch].reshape(-1) for a in arrs], axis=0)
        vmin = float(vv.min())
        vmax = float(vv.max())
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        clims[ch] = (vmin, vmax)
    return clims

def save_png_grid(
    gt_true_list: List[np.ndarray],  # (C,H,W)
    gt_dec_list:  List[np.ndarray],  # (C,H,W) = decoded from GT codes
    pr_dec_list:  List[np.ndarray],  # (C,H,W)
    milestones: List[int],
    out_png: str,
    *,
    channels: List[int],
    cmap: str,
    clim_mode: str,
    fixed_vmin: Optional[float],
    fixed_vmax: Optional[float],
    err_vmax: float,
):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    M = len(milestones)
    K = len(channels)

    clims = compute_channel_clims(gt_true_list + gt_dec_list + pr_dec_list, channels,
                                 clim_mode, fixed_vmin, fixed_vmax)

    # per milestone, per channel MSE vs TRUE GT
    mse_chan = np.zeros((M, K), dtype=np.float64)
    mse_all  = np.zeros((M,), dtype=np.float64)
    for mi in range(M):
        gt = gt_true_list[mi]
        pr = pr_dec_list[mi]
        for kj, ch in enumerate(channels):
            d = (pr[ch] - gt[ch]).astype(np.float64)
            mse_chan[mi, kj] = np.mean(d * d)
        d_all = (pr[channels] - gt[channels]).astype(np.float64)
        mse_all[mi] = np.mean(d_all * d_all)

    # 4 rows per milestone: GT(real) | GT(dec) | Pred | Err
    fig_h = max(3.0, 1.6 * M * 4)
    fig_w = max(6.0, 2.2 * K)
    fig, axs = plt.subplots(M * 4, K, figsize=(fig_w, fig_h), constrained_layout=True)

    if (M * 4) == 1 and K == 1:
        axs = np.array([[axs]])
    elif (M * 4) == 1:
        axs = np.array([axs])
    elif K == 1:
        axs = np.array([[a] for a in axs])

    for mi, t in enumerate(milestones):
        gt_true = gt_true_list[mi]
        gt_dec  = gt_dec_list[mi]
        pr      = pr_dec_list[mi]

        for kj, ch in enumerate(channels):
            gtT = gt_true[ch]
            gtD = gt_dec[ch]
            prH = pr[ch]
            err = np.abs(prH - gtT)

            vmin, vmax = clims[ch]
            r0 = mi * 4

            ax_t = axs[r0 + 0, kj]
            ax_d = axs[r0 + 1, kj]
            ax_p = axs[r0 + 2, kj]
            ax_e = axs[r0 + 3, kj]

            ax_t.imshow(gtT, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax_d.imshow(gtD, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax_p.imshow(prH, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax_e.imshow(err, origin="lower", cmap="gray", vmin=0.0, vmax=float(err_vmax), interpolation="nearest")

            if kj == 0:
                ax_t.set_ylabel(f"t={t}\nGT(real)", rotation=0, labelpad=40, va="center", ha="right")
                ax_d.set_ylabel("GT(dec)\nfrom codes", rotation=0, labelpad=40, va="center", ha="right")
                ax_p.set_ylabel(f"Pred\nMSE={mse_all[mi]:.4e}", rotation=0, labelpad=40, va="center", ha="right")
                ax_e.set_ylabel("|Err|", rotation=0, labelpad=40, va="center", ha="right")

            if mi == 0:
                ax_t.set_title(f"ch={ch}")

            ax_p.text(
                0.01, 0.99, f"MSE={mse_chan[mi, kj]:.3e}",
                transform=ax_p.transAxes, ha="left", va="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2.0),
            )

            for ax in (ax_t, ax_d, ax_p, ax_e):
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)

    fig.suptitle("Decoded rollout: GT(real) | GT(dec from codes) | Pred | |Pred-GT(real)|", y=1.01)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# =============================================================================
# Loading helpers
# =============================================================================
def infer_vocab_size_from_state(sd: dict) -> Optional[int]:
    for k in ("tok_emb.weight", "module.tok_emb.weight"):
        if k in sd and isinstance(sd[k], torch.Tensor):
            return int(sd[k].shape[0])
    for k, v in sd.items():
        if k.endswith("tok_emb.weight") and isinstance(v, torch.Tensor):
            return int(v.shape[0])
    return None

def load_gpt_any(gpt_ckpt: str, device: torch.device,
                 Hq: int, Wq: int, n_embed: int) -> Tuple[EncDecGPT, dict]:
    ck = torch.load(gpt_ckpt, map_location="cpu")
    if "model" not in ck:
        raise RuntimeError(f"{gpt_ckpt} missing 'model'. Keys={list(ck.keys())}")
    sd = ck["model"]
    cfg = ck.get("cfg", {}) or {}

    ck_Hq = int(ck.get("Hq", Hq))
    ck_Wq = int(ck.get("Wq", Wq))
    ck_ne = int(ck.get("n_embed", n_embed))
    if (ck_Hq, ck_Wq, ck_ne) != (int(Hq), int(Wq), int(n_embed)):
        raise RuntimeError(
            f"GPT/run mismatch: run(Hq,Wq,n_embed)=({Hq},{Wq},{n_embed}) "
            f"ckpt=({ck_Hq},{ck_Wq},{ck_ne})"
        )

    vocab_size = int(ck.get("vocab_size", infer_vocab_size_from_state(sd) or 0))
    if vocab_size <= 0:
        raise RuntimeError("Could not determine vocab_size from GPT ckpt.")

    dec_block = int(ck.get("dec_block", 1 + (Hq * Wq) + (Hq - 1) + 0))  # pause tokens handled separately
    deck_len = int(ck.get("deck_len", 0))

    gpt = EncDecGPT(
        vocab_size=vocab_size,
        n_embed=n_embed,
        Hq=Hq,
        Wq=Wq,
        deck_len=deck_len,
        dec_block_size=dec_block,
        n_layer_enc=int(cfg.get("n_layer_enc", 6)),
        n_layer_dec=int(cfg.get("n_layer_dec", 12)),
        n_head=int(cfg.get("n_head", 8)),
        n_embd=int(cfg.get("n_embd", 768)),
        dropout=float(cfg.get("dropout", 0.0)),
        pad_id=int(ck.get("pad_id", n_embed + 3)),
        use_sdpa=bool(cfg.get("use_sdpa", False)),
    ).to(device).eval()

    missing, unexpected = gpt.load_state_dict(sd, strict=False)
    print(f">> Loaded GPT: missing={len(missing)} unexpected={len(unexpected)}")
    return gpt, ck

def load_vq_any(vq_ckpt: str, device: torch.device, use_ema_copy: bool) -> LitVQVAE:
    ck = torch.load(vq_ckpt, map_location="cpu")
    sd = ck.get("state_dict", ck)
    hparams = ck.get("hyper_parameters", {}) or {}

    if "ddconfig" not in hparams:
        raise RuntimeError(
            "VQ checkpoint is missing hyper_parameters['ddconfig'].\n"
            "Fix: in src/models/lit_vqvae.py, change:\n"
            "  self.save_hyperparameters(ignore=['ddconfig'])\n"
            "to:\n"
            "  self.save_hyperparameters()\n"
            "Then retrain / resave a checkpoint."
        )

    ddconfig = hparams["ddconfig"]
    n_embed = int(hparams.get("n_embed"))
    embed_dim = int(hparams.get("embed_dim"))
    image_key = str(hparams.get("image_key", "image"))

    # Recreate model with same init signature
    vq = LitVQVAE(
        ddconfig=ddconfig,
        n_embed=n_embed,
        embed_dim=embed_dim,
        learning_rate=float(hparams.get("learning_rate", 2e-4)),
        beta=float(hparams.get("beta", 0.25)),
        vq_loss_weight=float(hparams.get("vq_loss_weight", 1.0)),
        recon_l2_weight=float(hparams.get("recon_l2_weight", 0.0)),
        grad_loss_weight=float(hparams.get("grad_loss_weight", 0.0)),
        l2_normalize_codebook=bool(hparams.get("l2_normalize_codebook", False)),
        legacy_beta_bug=bool(hparams.get("legacy_beta_bug", False)),
        output_clamp=bool(hparams.get("output_clamp", True)),
        image_key=image_key,
        usage_entropy_weight=float(hparams.get("usage_entropy_weight", 0.0)),
        dead_code_reset_every=int(hparams.get("dead_code_reset_every", 0)),
        dead_code_reset_threshold=float(hparams.get("dead_code_reset_threshold", 1.0)),
        no_quant=bool(hparams.get("no_quant", False)),
        warmup_steps=int(hparams.get("warmup_steps", 0)),
        total_steps=int(hparams.get("total_steps", 0)),
        min_lr=float(hparams.get("min_lr", 1e-6)),
    )

    missing, unexpected = vq.load_state_dict(sd, strict=False)
    print(f">> Loaded VQ: missing={len(missing)} unexpected={len(unexpected)}")

    if use_ema_copy:
        copy_ema_to_base_if_present(vq, sd)

    return vq.to(device).eval()

def load_gt_frames_npz(path: str, key: Optional[str] = None, expected_hw: Optional[Tuple[int,int]] = None) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    if key is None:
        for k in ("frames", "gt", "images", "x", "data"):
            if k in z:
                key = k
                break
    if key is None or key not in z:
        raise RuntimeError(f"{path} does not contain a GT array. Keys={list(z.keys())}. "
                           f"Provide --gt_key.")
    arr = z[key]
    arr = np.asarray(arr)

    # Normalize shapes to (T,C,H,W)
    if arr.ndim == 3:
        # arr could be (T,H,W) OR (T,W,H). Use expected_hw (H,W) to disambiguate.
        T, A, B = arr.shape
        if expected_hw is not None:
            expH, expW = expected_hw
            if (A, B) == (expH, expW):
                arr = arr[:, None, :, :]              # (T,1,H,W)
            elif (A, B) == (expW, expH):
                arr = arr.transpose(0, 2, 1)          # (T,H,W)
                arr = arr[:, None, :, :]              # (T,1,H,W)
            else:
                # fall back: assume (T,H,W)
                arr = arr[:, None, :, :]
        else:
            # fall back: assume (T,H,W)
            arr = arr[:, None, :, :]
    elif arr.ndim == 4:
        # (T,H,W,C) -> (T,C,H,W)
        if arr.shape[-1] <= 16 and arr.shape[1] > 16 and arr.shape[2] > 16:
            arr = arr.transpose(0, 3, 1, 2)
        # else assume already (T,C,H,W)
    else:
        raise RuntimeError(f"Unsupported GT shape {arr.shape} in {path}")

    return arr.astype(np.float32)

def load_minmax_json(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r") as f:
        mm = json.load(f)

    if not isinstance(mm, dict):
        raise RuntimeError(f"minmax json must be a dict, got {type(mm)}")

    # Case 1: scalar/json with percentile-clipped upper bound
    if "min" in mm and ("p_hi" in mm or "max" in mm):
        vmin = mm["min"]
        vmax = mm["p_hi"] if "p_hi" in mm else mm["max"]
        return np.asarray(vmin, dtype=np.float32), np.asarray(vmax, dtype=np.float32)

    # Case 2: channelwise arrays
    if "ch_min" in mm and "ch_max" in mm:
        return np.asarray(mm["ch_min"], dtype=np.float32), np.asarray(mm["ch_max"], dtype=np.float32)

    # Case 3: nested dict
    if len(mm) == 1 and isinstance(next(iter(mm.values())), dict):
        mm2 = next(iter(mm.values()))
        if "min" in mm2 and ("p_hi" in mm2 or "max" in mm2):
            vmin = mm2["min"]
            vmax = mm2["p_hi"] if "p_hi" in mm2 else mm2["max"]
            return np.asarray(vmin, dtype=np.float32), np.asarray(vmax, dtype=np.float32)
        if "ch_min" in mm2 and "ch_max" in mm2:
            return np.asarray(mm2["ch_min"], dtype=np.float32), np.asarray(mm2["ch_max"], dtype=np.float32)

    raise RuntimeError(f"Unrecognized minmax json format in {path}. Keys={list(mm.keys())}")

def normalize_to_m11(x_tchw: np.ndarray, vmin: np.ndarray, vmax: np.ndarray, clamp: bool = True) -> np.ndarray:
    vmin = np.asarray(vmin, dtype=np.float32).reshape(-1)
    vmax = np.asarray(vmax, dtype=np.float32).reshape(-1)

    C = x_tchw.shape[1]
    if vmin.size == 1 and C > 1:
        vmin = np.repeat(vmin, C)
        vmax = np.repeat(vmax, C)

    if vmin.size != C or vmax.size != C:
        raise RuntimeError(f"min/max have C={vmin.size} but GT has C={C}")

    den = (vmax - vmin)
    den = np.where(np.abs(den) < 1e-12, 1.0, den)

    x01 = (x_tchw - vmin[None, :, None, None]) / den[None, :, None, None]
    x11 = x01 * 2.0 - 1.0
    if clamp:
        x11 = np.clip(x11, -1.0, 1.0)
    return x11.astype(np.float32)



# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--run_tokens_npz", default="${LATTE_WORK_ROOT}/tokens/example/test_example_tokens.npz", help="Token NPZ with tokens(T,Hq,Wq) and n_embed.")
    ap.add_argument("--gpt_ckpt", required=True, help="gpt_best.pt / gpt_step*.pt from your GPT trainer.")
    ap.add_argument("--vq_ckpt", required=True, help="LitVQVAE lightning .ckpt used to produce those tokens.")
    ap.add_argument("--out_png", required=True)

    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--steps", type=int, default=56)
    ap.add_argument("--milestones", type=str, default="10,25,50",
                    help="Comma-separated relative steps from t0. Empty '' => only final frame.")

    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--greedy", action="store_true")

    ap.add_argument("--no_deck_tokens", action="store_true",
                    help="Force disable deck conditioning even if ckpt expects it.")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--use_vq_ema_copy", action="store_true")

    ap.add_argument("--channels", type=str, default="0")
    ap.add_argument("--cmap", type=str, default="viridis")
    ap.add_argument("--clim_mode", type=str, default="global", choices=["global", "per_panel"])
    ap.add_argument("--fixed_vmin", type=float, default=None)
    ap.add_argument("--fixed_vmax", type=float, default=None)
    ap.add_argument("--err_vmax", type=float, default=0.15)

    ap.add_argument("--gt_frames_npz", type=str, default="${LATTE_WORK_ROOT}/data/example/test_example_frames.npz",
                help="Optional NPZ with real-valued frames. Must align with token time axis. "
                     "Supported shapes: (T,H,W), (T,H,W,C), (T,C,H,W). Key can be frames/gt/images/x.")
    ap.add_argument("--gt_key", type=str, default=None,
                help="Optional key name inside gt_frames_npz. If not set, tries common keys.")

    ap.add_argument("--minmax_json", type=str, default="${LATTE_DATA_ROOT}/example_minmax.json",
                help="Optional min/max json used during VQ training to normalize GT into [-1,1].")
    ap.add_argument("--gt_already_normalized", action="store_true",
                help="If set, skip min/max normalization even if --minmax_json is provided.")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    # ---- load run tokens ----
    z = np.load(args.run_tokens_npz, allow_pickle=True)
    if "tokens" not in z:
        raise RuntimeError(f"{args.run_tokens_npz} missing 'tokens'. Keys={list(z.keys())}")

    tokens = z["tokens"].astype(np.int64)
    if tokens.ndim != 3:
        raise RuntimeError(f"Expected tokens (T,Hq,Wq); got {tokens.shape}")

    T, Hq, Wq = tokens.shape
    if "n_embed" not in z:
        raise RuntimeError(f"{args.run_tokens_npz} missing n_embed.")
    n_embed = int(z["n_embed"])

    if args.t0 < 0 or args.t0 >= (T - 1):
        raise ValueError(f"t0 out of range: t0={args.t0}, T={T}")
    if args.t0 + args.steps >= T:
        raise ValueError(f"Need GT up to t0+steps. t0={args.t0} steps={args.steps} T={T}")

    # milestones absolute
    if str(args.milestones).strip():
        ms_rel = sorted(set(int(x) for x in args.milestones.split(",") if x.strip()))
        if any(m <= 0 for m in ms_rel):
            raise ValueError("All milestones must be > 0 relative to t0.")
        milestones_abs = [args.t0 + m for m in ms_rel]
        if milestones_abs[-1] > args.t0 + args.steps:
            raise ValueError("A milestone exceeds t0+steps. Increase --steps or reduce --milestones.")
    else:
        milestones_abs = [args.t0 + args.steps]

    channels = [int(x) for x in args.channels.split(",") if x.strip()]

    # ---- load GPT ----
    gpt, gpt_ck = load_gpt_any(args.gpt_ckpt, device=device, Hq=Hq, Wq=Wq, n_embed=n_embed)

    bos_id = int(gpt_ck.get("bos_id", n_embed))
    eos_id = int(gpt_ck.get("eos_id", n_embed + 1))
    row_id = int(gpt_ck.get("row_id", n_embed + 2))
    pad_id = int(gpt_ck.get("pad_id", n_embed + 3))

    deck_len = int(gpt_ck.get("deck_len", 0))
    deck_bins = int(gpt_ck.get("deck_bins", 0))
    deck_keys = list(gpt_ck.get("deck_keys", []))

    pause_tokens = int(gpt_ck.get("pause_tokens", 0))
    pause_id = None
    if pause_tokens > 0:
        pid = gpt_ck.get("pause_id", None)
        if pid is None:
            raise RuntimeError("GPT ckpt indicates pause_tokens>0 but has no pause_id saved.")
        pause_id = int(pid)

    # ---- build deck tokens if needed ----
    deck_tokens = None
    if (not args.no_deck_tokens) and deck_len > 0:
        # best case: token NPZ already stores deck tokens
        if "deck_tokens" in z:
            dt = z["deck_tokens"].astype(np.int64)
            if dt.ndim == 1 and dt.shape[0] == deck_len:
                deck_tokens = dt
            else:
                raise RuntimeError(f"deck_tokens in NPZ has shape {dt.shape}, expected ({deck_len},)")

        # next best: if run_dir has clover.in and ckpt has enough deck info
        elif "run_dir" in z:
            run_dir = str(z["run_dir"])
            cin = os.path.join(run_dir, "clover.in")
            if os.path.isfile(cin) and deck_bins > 0:
                key_stats, has_stats = build_key_stats_from_gpt_ckpt(gpt_ck)
                if not has_stats:
                    print(">> WARNING: GPT ckpt has no numeric deck_key_stats; using categorical hashing for all keys.")
                if len(deck_keys) != deck_len:
                    deck_keys = (deck_keys[:deck_len] + [f"_missing_key_{i}" for i in range(max(0, deck_len - len(deck_keys)))])
                for k in deck_keys:
                    if k not in key_stats:
                        key_stats[k] = ("cat", None, None)
                deck_base = n_embed + 4
                deck_tokens = deck_tokens_for_run(run_dir, deck_keys, key_stats, deck_bins, deck_base)
            else:
                raise RuntimeError(
                    "GPT expects deck tokens, but token NPZ has no deck_tokens and run_dir lacks clover.in.\n"
                    "Fix: (1) run with --no_deck_tokens, or (2) store deck_tokens inside token NPZ during export."
                )
        else:
            raise RuntimeError(
                "GPT expects deck tokens, but token NPZ has neither deck_tokens nor run_dir.\n"
                "Fix: run with --no_deck_tokens, or store deck_tokens in NPZ at export time."
            )

    print(f">> Run NPZ: {args.run_tokens_npz}")
    print(f">> tokens: T={T} Hq={Hq} Wq={Wq} n_embed={n_embed}")
    print(f">> rollout: t0={args.t0} steps={args.steps} milestones={milestones_abs}")
    print(f">> deck: {'ON' if deck_tokens is not None else 'OFF'} (deck_len={deck_len})")
    print(f">> pause_tokens={pause_tokens} pause_id={pause_id}")
    print(f">> sampling: temperature={args.temperature} top_k={args.top_k} greedy={args.greedy}")

    # ---- rollout in token space ----
    cur = tokens[args.t0].reshape(-1).copy()
    preds = [tokens[args.t0].copy()]  # include t0

    for s in range(1, args.steps + 1):
        nxt = sample_next_tokens_rowwise(
            gpt,
            enc_tokens_1d=cur,
            deck_tokens_1d=deck_tokens,
            Hq=Hq, Wq=Wq,
            n_embed=n_embed,
            bos_id=bos_id, eos_id=eos_id, row_id=row_id,
            pause_tokens=pause_tokens, pause_id=pause_id,
            temperature=args.temperature,
            top_k=args.top_k,
            greedy=args.greedy,
            device=device,
        )
        cur = nxt
        preds.append(cur.reshape(Hq, Wq).copy())
        if (s % 10) == 0:
            print(f"  step {s}/{args.steps}", flush=True)

    preds = np.stack(preds, axis=0)  # (steps+1,Hq,Wq)

    # ---- load VQ ----
    vq = load_vq_any(args.vq_ckpt, device=device, use_ema_copy=args.use_vq_ema_copy)

    # sanity check: VQ n_embed should match token n_embed
    try:
        vq_ne = int(getattr(vq, "quantize").n_e)
    except Exception:
        vq_ne = None
    if vq_ne is not None and vq_ne != n_embed:
        raise RuntimeError(f"VQ n_embed={vq_ne} != token n_embed={n_embed}")

    gt_frames = None
    if args.gt_frames_npz is not None:
        gt_frames = load_gt_frames_npz(args.gt_frames_npz, key=args.gt_key)
        if gt_frames.shape[0] != T:
            raise RuntimeError(f"GT frames T={gt_frames.shape[0]} but token T={T}. Must match.")

        if (args.minmax_json is not None) and (not args.gt_already_normalized):
            vmin, vmax = load_minmax_json(args.minmax_json)
            gt_frames = normalize_to_m11(gt_frames, vmin, vmax, clamp=True)

    # Make GT match decoded spatial shape (handle swapped H/W)
    # Decode a reference frame to know what VQ outputs spatially.
    _ref_dec = decode_codes_to_fields(vq, tokens[args.t0], device=device)  # (C,H,W)
    C_ref, H_ref, W_ref = _ref_dec.shape

    # gt_frames is (T,C,H,W)
    if gt_frames.shape[1] != C_ref:
        raise RuntimeError(f"GT C={gt_frames.shape[1]} but VQ decoded C={C_ref}. Wrong gt_key or wrong data.")

    H_gt, W_gt = gt_frames.shape[2], gt_frames.shape[3]
    if (H_gt, W_gt) == (W_ref, H_ref):
        print(f">> NOTE: GT appears transposed ({H_gt},{W_gt}) vs VQ ({H_ref},{W_ref}); swapping axes.", flush=True)
        gt_frames = gt_frames.transpose(0, 1, 3, 2).copy()

    if (gt_frames.shape[2], gt_frames.shape[3]) != (H_ref, W_ref):
        raise RuntimeError(f"GT spatial {(gt_frames.shape[2],gt_frames.shape[3])} != VQ spatial {(H_ref,W_ref)}")


    # after vq is loaded and gt_frames normalized
    # after loading vq and (optionally) loading/normalizing gt_frames

    if gt_frames is not None:
        _ref_dec = decode_codes_to_fields(vq, tokens[args.t0], device=device).astype(np.float32)  # (C,H,W)

        if gt_frames.shape[1] != _ref_dec.shape[0]:
            raise RuntimeError(f"GT(real) C={gt_frames.shape[1]} but decoded C={_ref_dec.shape[0]}")

        if gt_frames.shape[2:] != _ref_dec.shape[1:]:
            if gt_frames.shape[2:] == (_ref_dec.shape[2], _ref_dec.shape[1]):
                gt_frames = gt_frames.transpose(0, 1, 3, 2)
                print(">> Transposed GT(real) H/W to match decoded.")
            else:
                raise RuntimeError(f"GT(real) HW={gt_frames.shape[2:]} but decoded HW={_ref_dec.shape[1:]}")


    # ---- decode only at milestones ----
    gt_true_list: List[np.ndarray] = []
    gt_dec_list:  List[np.ndarray] = []
    pr_dec_list:  List[np.ndarray] = []

    if gt_frames is None:
        raise RuntimeError("To plot GT(real) you must provide --gt_frames_npz (tokens alone only give GT(dec)).")

    for t_abs in milestones_abs:
        rel = t_abs - args.t0

        gt_codes = tokens[t_abs]
        pr_codes = preds[rel]

        gt_true = gt_frames[t_abs]  # (C,H,W) already
        gt_dec  = decode_codes_to_fields(vq, gt_codes, device=device).astype(np.float32)
        pr_dec  = decode_codes_to_fields(vq, pr_codes, device=device).astype(np.float32)

        gt_true_list.append(gt_true)
        gt_dec_list.append(gt_dec)
        pr_dec_list.append(pr_dec)


    C0 = gt_dec_list[0].shape[0]
    for ch in channels:
        if ch < 0 or ch >= C0:
            raise ValueError(f"--channels includes {ch} but decoded has C={C0}")

    save_png_grid(
        gt_true_list, gt_dec_list, pr_dec_list, milestones_abs, args.out_png,
        channels=channels, cmap=args.cmap, clim_mode=args.clim_mode,
        fixed_vmin=args.fixed_vmin, fixed_vmax=args.fixed_vmax, err_vmax=args.err_vmax,
    )


    print(f">> Saved: {args.out_png}")


if __name__ == "__main__":
    main()



