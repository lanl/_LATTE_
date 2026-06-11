#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import math
import argparse
from pathlib import Path
from typing import Optional, Any, Tuple, List

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# GPT model
# =============================================================================
class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, use_sdpa: bool, causal: bool):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, n_embd: int, n_head: int, dropout: float, use_sdpa: bool):
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

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, n_embd: int, n_head: int, dropout: float, use_sdpa: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, dropout, use_sdpa=use_sdpa, causal=False)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, use_sdpa: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.self_attn = SelfAttention(n_embd, n_head, dropout, use_sdpa=use_sdpa, causal=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.cross_attn = CrossAttention(n_embd, n_head, dropout, use_sdpa=use_sdpa)
        self.ln3 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
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

        self.tok_emb = nn.Embedding(self.vocab_size, n_embd)
        self.row_emb = nn.Embedding(self.Hq, n_embd)
        self.col_emb = nn.Embedding(self.Wq, n_embd)
        self.deck_pos_emb = nn.Embedding(max(1, self.deck_len), n_embd)
        self.deck_type = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.frame_type = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.dec_pos_emb = nn.Embedding(self.dec_block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.enc_blocks = nn.ModuleList([EncoderBlock(n_embd, n_head, dropout, use_sdpa) for _ in range(n_layer_enc)])
        self.dec_blocks = nn.ModuleList([DecoderBlock(n_embd, n_head, dropout, use_sdpa) for _ in range(n_layer_dec)])
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
        x = self.tok_emb(dec_in)
        pos = torch.arange(T, device=dec_in.device)
        x = self.drop(x + self.dec_pos_emb(pos)[None, :, :])
        for blk in self.dec_blocks:
            x = blk(x, mem)
        x = self.dec_ln(x)
        return self.lm_head(x)


# =============================================================================
# VQ helpers
# =============================================================================
from src.models.lit_vqvae import LitVQVAE


def _get_nested_attr(obj: Any, path: str):
    cur = obj
    for p in path.split("."):
        if cur is None or not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur


def _find_codebook_weight(vq: nn.Module) -> Optional[torch.Tensor]:
    for c in [
        "quantize.embedding.weight",
        "model.quantize.embedding.weight",
        "quantizer.embedding.weight",
        "model.quantizer.embedding.weight",
    ]:
        w = _get_nested_attr(vq, c)
        if isinstance(w, torch.Tensor):
            return w
    return None


def _vq_decode_latents(vq: nn.Module, z_bchw: torch.Tensor) -> torch.Tensor:
    for fn_path in ("decode", "model.decode"):
        fn = _get_nested_attr(vq, fn_path)
        if callable(fn):
            out = fn(z_bchw)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out

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

    raise RuntimeError("Could not find a VQ decode path.")


@torch.inference_mode()
def decode_codes_to_fields(vq: LitVQVAE, codes_hw: np.ndarray, device: torch.device) -> np.ndarray:
    codes = torch.from_numpy(codes_hw.astype(np.int64)).to(device)

    for fn_path in ("decode_code", "model.decode_code"):
        fn = _get_nested_attr(vq, fn_path)
        if callable(fn):
            out = fn(codes[None, ...])
            if isinstance(out, (tuple, list)):
                out = out[0]
            x = out
            if getattr(vq, "output_clamp", False):
                x = x.clamp(-1, 1)
            return x[0].detach().float().cpu().numpy()

    w = _find_codebook_weight(vq)
    if w is None:
        raise RuntimeError("Could not locate VQ codebook weight.")

    Hq, Wq = codes.shape
    z = w[codes.reshape(-1)]
    z = z.view(Hq, Wq, -1).permute(2, 0, 1)[None, ...]
    x = _vq_decode_latents(vq, z)
    if getattr(vq, "output_clamp", False):
        x = x.clamp(-1, 1)
    return x[0].detach().float().cpu().numpy()


@torch.inference_mode()
def encode_first_frame_to_codes(vq: LitVQVAE, x_chw: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(x_chw.astype(np.float32))[None, ...].to(device)
    z, _, info = vq.encode(x)
    inds = info[-1]
    if inds is None:
        raise RuntimeError("VQ encode returned no indices.")
    inds = torch.as_tensor(inds, device=device, dtype=torch.long)
    Hq = int(z.shape[-2])
    Wq = int(z.shape[-1])
    if inds.numel() != Hq * Wq:
        raise RuntimeError(f"Unexpected encoded size {inds.numel()} vs {Hq*Wq}.")
    return inds.view(Hq, Wq).detach().cpu().numpy().astype(np.int64)


def load_vq_any(vq_ckpt: str, device: torch.device) -> LitVQVAE:
    ck = torch.load(vq_ckpt, map_location="cpu")
    sd = ck.get("state_dict", ck)
    hparams = ck.get("hyper_parameters", {}) or {}
    ddconfig = hparams["ddconfig"]

    vq = LitVQVAE(
        ddconfig=ddconfig,
        n_embed=int(hparams["n_embed"]),
        embed_dim=int(hparams["embed_dim"]),
        learning_rate=float(hparams.get("learning_rate", 2e-4)),
        beta=float(hparams.get("beta", 0.1)),
        vq_loss_weight=float(hparams.get("vq_loss_weight", 5.0)),
        recon_l2_weight=float(hparams.get("recon_l2_weight", 0.0)),
        grad_loss_weight=float(hparams.get("grad_loss_weight", 0.0)),
        l2_normalize_codebook=bool(hparams.get("l2_normalize_codebook", False)),
        legacy_beta_bug=bool(hparams.get("legacy_beta_bug", False)),
        output_clamp=bool(hparams.get("output_clamp", False)),
        image_key=str(hparams.get("image_key", "image")),
        no_quant=bool(hparams.get("no_quant", False)),
        warmup_steps=0,
        total_steps=0,
        min_lr=float(hparams.get("min_lr", 1e-6)),
    )
    missing, unexpected = vq.load_state_dict(sd, strict=False)
    print(f">> Loaded VQ: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return vq.to(device).eval()


# =============================================================================
# GPT helpers
# =============================================================================
def infer_vocab_size_from_state(sd: dict) -> Optional[int]:
    for k in ("tok_emb.weight", "module.tok_emb.weight"):
        if k in sd and isinstance(sd[k], torch.Tensor):
            return int(sd[k].shape[0])
    return None


def infer_dec_block_from_state(sd: dict) -> Optional[int]:
    for k in ("dec_pos_emb.weight", "module.dec_pos_emb.weight"):
        if k in sd and isinstance(sd[k], torch.Tensor):
            return int(sd[k].shape[0])
    return None


def load_gpt_bundle(ckpt_path: str, device: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["model"]
    cfg = ck.get("cfg", {})

    vocab_size = int(ck.get("vocab_size", infer_vocab_size_from_state(sd)))
    dec_block = int(ck.get("dec_block", infer_dec_block_from_state(sd)))
    n_embed = int(ck["n_embed"])
    Hq = int(ck["Hq"])
    Wq = int(ck["Wq"])
    bos_id = int(ck.get("bos_id", n_embed))
    eos_id = int(ck.get("eos_id", n_embed + 1))
    row_id = int(ck.get("row_id", n_embed + 2))
    pad_id = int(ck.get("pad_id", n_embed + 3))
    deck_len = int(ck.get("deck_len", 0))

    model = EncDecGPT(
        vocab_size=vocab_size,
        n_embed=n_embed,
        Hq=Hq,
        Wq=Wq,
        deck_len=deck_len,
        dec_block_size=dec_block,
        n_layer_enc=int(cfg.get("n_layer_enc", 4)),
        n_layer_dec=int(cfg.get("n_layer_dec", 8)),
        n_head=int(cfg.get("n_head", 8)),
        n_embd=int(cfg.get("n_embd", 768)),
        dropout=float(cfg.get("dropout", 0.1)),
        pad_id=pad_id,
        use_sdpa=bool(cfg.get("use_sdpa", False)),
    ).to(device).eval()

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f">> Loaded GPT: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    meta = dict(
        n_embed=n_embed,
        Hq=Hq,
        Wq=Wq,
        bos_id=bos_id,
        eos_id=eos_id,
        row_id=row_id,
        pad_id=pad_id,
        dec_block=dec_block,
        deck_len=deck_len,
    )
    return model, meta


@torch.inference_mode()
def sample_next_tokens_rowwise_greedy(
    model: EncDecGPT,
    enc_tokens_1d: np.ndarray,
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
    mem = model.encode(enc, deck_tokens=None)

    out = torch.tensor([[int(bos_id)]], device=device, dtype=torch.long)
    for _ in range(max_gen_len - 1):
        ctx = out[:, -model.dec_block_size:] if out.size(1) > model.dec_block_size else out
        logits = model.decode(ctx, mem)[:, -1, :]
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
        if int(nxt.item()) == eos_id:
            break

    seq = out[0].detach().cpu().numpy().astype(np.int64)

    codes: List[int] = []
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


# =============================================================================
# Data helpers
# =============================================================================
def load_minmax_json(path: str) -> Tuple[float, float]:
    with open(path, "r") as f:
        mm = json.load(f)

    if not isinstance(mm, dict):
        raise RuntimeError("minmax json must be a dict")

    if "min" in mm and ("p_hi" in mm or "max" in mm):
        vmin = float(mm["min"])
        vmax = float(mm["p_hi"] if "p_hi" in mm else mm["max"])
        return vmin, vmax

    raise RuntimeError(f"Unsupported minmax format in {path}")


def normalize_to_m11(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    x = np.clip(x, vmin, vmax)
    x = 2.0 * (x - vmin) / (vmax - vmin) - 1.0
    return x.astype(np.float32)


def denormalize_from_m11(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    x = 0.5 * (x + 1.0) * (vmax - vmin) + vmin
    x = np.clip(x, vmin, vmax)
    return x.astype(np.float32)


def load_h5_frames(h5_path: str, field: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        x = f[field][...]
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 3:
        raise RuntimeError(f"Expected (T,H,W), got {x.shape} from {h5_path}:{field}")
    return x


def read_paths_list(txt_path: str) -> List[str]:
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_list_txt", required=True, help="Text file with one H5 path per line.")
    ap.add_argument("--field", default="t0_fields/av_density")
    ap.add_argument("--gpt_ckpt", required=True)
    ap.add_argument("--vq_ckpt", required=True)
    ap.add_argument("--minmax_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true", help="Skip runs whose output already exists.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    vmin, vmax = load_minmax_json(args.minmax_json)

    gpt, gmeta = load_gpt_bundle(args.gpt_ckpt, device=device)
    vq = load_vq_any(args.vq_ckpt, device=device)

    h5_paths = read_paths_list(args.test_list_txt)
    print(f">> test trajectories: {len(h5_paths)}", flush=True)

    for i, h5_path in enumerate(h5_paths):
        run_id = Path(h5_path).stem
        out_fp = os.path.join(args.out_dir, f"{run_id}_rollout.npz")
        if args.resume and os.path.exists(out_fp):
            print(f">> skipping existing {out_fp}", flush=True)
            continue

        print(f">> [{i+1}/{len(h5_paths)}] {run_id}", flush=True)

        gt_phys = load_h5_frames(h5_path, args.field)           # (T,H,W)
        T, H, W = gt_phys.shape

        gt_norm = normalize_to_m11(gt_phys, vmin, vmax)        # (T,H,W)
        x0 = gt_norm[0][None, :, :]                            # (1,H,W) as CHW

        init_codes = encode_first_frame_to_codes(vq, x0, device=device)  # (Hq,Wq)

        steps = min(int(args.steps), T - 1)
        pred_codes = np.empty((steps + 1, gmeta["Hq"], gmeta["Wq"]), dtype=np.int16)
        pred_codes[0] = init_codes.astype(np.int16)

        cur = init_codes.reshape(-1).copy()
        max_gen_len = int(gmeta["dec_block"]) + 1

        for t in range(1, steps + 1):
            nxt = sample_next_tokens_rowwise_greedy(
                gpt,
                cur,
                Hq=gmeta["Hq"],
                Wq=gmeta["Wq"],
                n_embed=gmeta["n_embed"],
                bos_id=gmeta["bos_id"],
                eos_id=gmeta["eos_id"],
                row_id=gmeta["row_id"],
                device=device,
                max_gen_len=max_gen_len,
            )
            cur = nxt
            pred_codes[t] = cur.reshape(gmeta["Hq"], gmeta["Wq"]).astype(np.int16)

        pred_norm = np.empty((steps + 1, H, W), dtype=np.float32)
        for t in range(steps + 1):
            dec = decode_codes_to_fields(vq, pred_codes[t].astype(np.int64), device=device)  # (C,H,W)
            pred_norm[t] = dec[0]

        pred_phys = denormalize_from_m11(pred_norm, vmin, vmax)

        np.savez_compressed(
            out_fp,
            run_id=np.array(run_id),
            src_h5=np.array(h5_path),
            pred_tokens=pred_codes,
            pred_norm=pred_norm,
            pred_phys=pred_phys,
            gt_phys=gt_phys[: steps + 1],
            gt_norm=gt_norm[: steps + 1],
            steps=np.int32(steps),
            Hq=np.int32(gmeta["Hq"]),
            Wq=np.int32(gmeta["Wq"]),
            n_embed=np.int32(gmeta["n_embed"]),
            field=np.array(args.field),
            min_val=np.float32(vmin),
            max_val=np.float32(vmax),
        )
        print(f">> saved {out_fp}", flush=True)


if __name__ == "__main__":
    main()
