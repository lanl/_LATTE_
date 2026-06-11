#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import argparse
from typing import Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from PIL import Image

from scripts.custom.transformer import EncDecGPT
from scripts.custom.export_vq_tokens import (
    decode_codes_to_x_m11,
    load_vqvae_from_ckpt,
    infer_codebook_from_state_dict,
    df_to_ch_mult,
)


def load_tokens_npz(path: str):
    z = np.load(path, allow_pickle=True)
    tokens = z["tokens"]

    if tokens.ndim == 3:
        tokens = tokens[:, None, :, :]
    elif tokens.ndim != 4:
        raise RuntimeError(f"Expected tokens ndim 3 or 4, got {tokens.shape}")

    time_count = int(z["time_count"])
    n_embed = int(z["n_embed"])
    dataset = str(z["dataset"]) if "dataset" in z else os.path.basename(path)
    split = str(z["split"]) if "split" in z else "unknown"

    N, C, Hq, Wq = tokens.shape
    if N % time_count != 0:
        raise RuntimeError(f"N={N} not divisible by time_count={time_count}")

    n_traj = N // time_count

    print(
        f">> loaded tokens: dataset={dataset} split={split} "
        f"shape={tokens.shape} time_count={time_count} n_traj={n_traj} n_embed={n_embed}",
        flush=True,
    )

    return tokens, time_count, n_embed, dataset, split


def load_encdec_from_ckpt(path: str, device: torch.device) -> Tuple[EncDecGPT, dict]:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("cfg", {})

    required = [
        "vocab_size",
        "n_embed",
        "Hq",
        "Wq",
        "dec_block",
        "pad_id",
        "deck_len",
    ]
    for k in required:
        if k not in ckpt:
            raise RuntimeError(f"Checkpoint missing key {k!r}")

    model = EncDecGPT(
        vocab_size=int(ckpt["vocab_size"]),
        n_embed=int(ckpt["n_embed"]),
        Hq=int(ckpt["Hq"]),
        Wq=int(ckpt["Wq"]),
        deck_len=int(ckpt["deck_len"]),
        dec_block_size=int(ckpt["dec_block"]),
        n_layer_enc=int(cfg.get("n_layer_enc", 6)),
        n_layer_dec=int(cfg.get("n_layer_dec", 12)),
        n_head=int(cfg.get("n_head", 12)),
        n_embd=int(cfg.get("n_embd", 768)),
        dropout=float(cfg.get("dropout", 0.0)),
        pad_id=int(ckpt["pad_id"]),
        use_sdpa=bool(cfg.get("use_sdpa", True)),
    )

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f">> loaded encdec: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    model = model.to(device).eval()
    return model, ckpt


@torch.inference_mode()
def greedy_predict_next_tokens(
    model: EncDecGPT,
    enc_tokens_hw: torch.Tensor,
    *,
    bos_id: int,
    eos_id: int,
) -> torch.Tensor:
    """
    enc_tokens_hw: [Hq,Wq] long on device
    returns predicted next tokens [Hq,Wq] long on device

    Greedy autoregressive decode:
      dec_in starts with BOS
      append argmax tokens until Hq*Wq tokens are generated
    """
    device = enc_tokens_hw.device
    Hq, Wq = enc_tokens_hw.shape
    L = Hq * Wq

    enc = enc_tokens_hw.reshape(1, L).long()

    mem = model.encode(enc, deck_tokens=None)

    dec = torch.tensor([[int(bos_id)]], device=device, dtype=torch.long)
    pred_tokens = []

    for _ in range(L):
        logits = model.decode(dec, mem)
        next_logits = logits[:, -1, :]

        # Only allow real VQ code IDs, not BOS/EOS/ROW/PAD.
        next_logits = next_logits[:, : model.n_embed]

        nxt = torch.argmax(next_logits, dim=-1)  # [1]
        pred_tokens.append(nxt)

        dec = torch.cat([dec, nxt[:, None]], dim=1)

    pred = torch.cat(pred_tokens, dim=0).reshape(Hq, Wq).long()
    return pred


def build_vq_fallback_ddconfig(
    vqvae_ckpt: str,
    *,
    H: int,
    W: int,
    down_factor: int,
    base_ch: int,
    num_res_blocks: int,
    dropout: float,
):
    ckpt = torch.load(vqvae_ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    _, embed_dim = infer_codebook_from_state_dict(sd)

    ch_mult = df_to_ch_mult(int(down_factor))
    latent_h = int(H) // int(down_factor)
    latent_w = int(W) // int(down_factor)

    return dict(
        double_z=False,
        z_channels=int(embed_dim),
        resolution=max(int(H), int(W)),
        in_channels=1,
        out_ch=1,
        ch=int(base_ch),
        ch_mult=list(ch_mult),
        num_res_blocks=int(num_res_blocks),
        attn_resolutions=[int(min(latent_h, latent_w))],
        dropout=float(dropout),
    )


@torch.inference_mode()
def decode_token_grid(vqvae, tokens_hq_wq: torch.Tensor) -> torch.Tensor:
    device = next(vqvae.parameters()).device
    inds = tokens_hq_wq.to(device=device, dtype=torch.long).unsqueeze(0)
    x = decode_codes_to_x_m11(vqvae, inds)
    x = x.clamp(-1.0, 1.0)
    return x[0, 0].detach().cpu().float()


def to_color(x_hw: torch.Tensor, vmin: float, vmax: float, cmap_name: str = "viridis") -> Image.Image:
    a = x_hw.detach().cpu().float().numpy()
    a = (a - float(vmin)) / (float(vmax) - float(vmin) + 1e-12)
    a = np.clip(a, 0.0, 1.0)

    try:
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
    except Exception:
        cmap = cm.get_cmap(cmap_name)

    rgb = (cmap(a)[..., :3] * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def to_err(err_hw: torch.Tensor, err_vmax: float) -> Image.Image:
    e = err_hw.detach().cpu().float()
    e = torch.clamp(e / float(err_vmax), 0.0, 1.0)
    e = (e.numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(e, mode="L").convert("RGB")


def save_rollout_grid(
    gt_imgs,
    pred_imgs,
    out_png: str,
    *,
    viz_vmin: float,
    viz_vmax: float,
    err_vmax: float,
    every: int = 1,
):
    idxs = list(range(0, len(gt_imgs), max(1, int(every))))
    if idxs[-1] != len(gt_imgs) - 1:
        idxs.append(len(gt_imgs) - 1)

    gt_tiles = [to_color(gt_imgs[i], viz_vmin, viz_vmax) for i in idxs]
    pr_tiles = [to_color(pred_imgs[i], viz_vmin, viz_vmax) for i in idxs]
    er_tiles = [to_err((gt_imgs[i] - pred_imgs[i]).abs(), err_vmax) for i in idxs]

    tile_w, tile_h = gt_tiles[0].size
    n = len(idxs)

    grid = Image.new("RGB", (tile_w * n, tile_h * 3))

    for j in range(n):
        grid.paste(gt_tiles[j], (j * tile_w, 0))
        grid.paste(pr_tiles[j], (j * tile_w, tile_h))
        grid.paste(er_tiles[j], (j * tile_w, tile_h * 2))

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    grid.save(out_png)

    print(
        f">> saved rollout PNG: {out_png} "
        f"columns={idxs} rows=[gt,pred,abs_error] tile={tile_w}x{tile_h}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--transformer_ckpt", required=True)
    ap.add_argument("--vqvae_ckpt", required=True)
    ap.add_argument("--tokens_npz", required=True)
    ap.add_argument("--out_png", required=True)

    ap.add_argument("--channel_idx", type=int, default=0)
    ap.add_argument("--traj_idx", type=int, default=0)
    ap.add_argument("--start_t", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument(
        "--teacher_forced",
        action="store_true",
        help="Feed true tokens_t at each step. Otherwise feed previous predicted tokens.",
    )

    ap.add_argument("--down_factor", type=int, default=8)
    ap.add_argument("--base_ch", type=int, default=128)
    ap.add_argument("--num_res_blocks", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--viz_vmin", type=float, default=-1.0)
    ap.add_argument("--viz_vmax", type=float, default=1.0)
    ap.add_argument("--viz_err_vmax", type=float, default=0.2)
    ap.add_argument("--save_every_frame", type=int, default=1)

    ap.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    tokens, time_count, n_embed, dataset, split = load_tokens_npz(args.tokens_npz)

    N, C, Hq, Wq = tokens.shape
    n_traj = N // time_count

    if args.traj_idx < 0 or args.traj_idx >= n_traj:
        raise ValueError(f"traj_idx={args.traj_idx} out of range [0,{n_traj})")
    if args.channel_idx < 0 or args.channel_idx >= C:
        raise ValueError(f"channel_idx={args.channel_idx} out of range [0,{C})")
    if args.start_t < 0 or args.start_t >= time_count - 1:
        raise ValueError(f"start_t={args.start_t} invalid for time_count={time_count}")
    if args.start_t + args.steps >= time_count:
        raise ValueError(
            f"Need start_t + steps < time_count. Got start_t={args.start_t}, "
            f"steps={args.steps}, time_count={time_count}"
        )

    model, ckpt = load_encdec_from_ckpt(args.transformer_ckpt, device=device)

    if int(ckpt["n_embed"]) != int(n_embed):
        raise RuntimeError(f"ckpt n_embed={ckpt['n_embed']} but token file n_embed={n_embed}")
    if int(ckpt["Hq"]) != int(Hq) or int(ckpt["Wq"]) != int(Wq):
        raise RuntimeError(
            f"ckpt Hq,Wq=({ckpt['Hq']},{ckpt['Wq']}) but token file Hq,Wq=({Hq},{Wq})"
        )

    H_img = Hq * int(args.down_factor)
    W_img = Wq * int(args.down_factor)

    ddconfig_fallback = build_vq_fallback_ddconfig(
        args.vqvae_ckpt,
        H=H_img,
        W=W_img,
        down_factor=args.down_factor,
        base_ch=args.base_ch,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
    )

    vqvae = load_vqvae_from_ckpt(
        args.vqvae_ckpt,
        ddconfig_fallback=ddconfig_fallback,
        use_ema_copy=False,
    ).to(device).eval()

    traj_start = int(args.traj_idx) * int(time_count)

    gt_token_seq = []
    for tt in range(args.start_t, args.start_t + args.steps + 1):
        arr = tokens[traj_start + tt, args.channel_idx]
        gt_token_seq.append(torch.from_numpy(arr.astype(np.int64, copy=False)).long())

    pred_token_seq = [gt_token_seq[0].clone().to(device)]

    cur = pred_token_seq[0]
    for s in range(args.steps):
        if args.teacher_forced:
            model_input = gt_token_seq[s].to(device)
        else:
            model_input = cur

        nxt = greedy_predict_next_tokens(
            model,
            model_input,
            bos_id=int(ckpt["bos_id"]),
            eos_id=int(ckpt["eos_id"]),
        )

        pred_token_seq.append(nxt)
        cur = nxt

    gt_imgs = []
    pred_imgs = []

    for i in range(args.steps + 1):
        gt_img = decode_token_grid(vqvae, gt_token_seq[i])
        pr_img = decode_token_grid(vqvae, pred_token_seq[i])
        gt_imgs.append(gt_img)
        pred_imgs.append(pr_img)

    for i in range(args.steps + 1):
        token_acc = (
            pred_token_seq[i].detach().cpu().reshape(-1)
            == gt_token_seq[i].detach().cpu().reshape(-1)
        ).float().mean().item()

        img_l1 = torch.mean((gt_imgs[i] - pred_imgs[i]).abs()).item()
        img_mse = torch.mean((gt_imgs[i] - pred_imgs[i]) ** 2).item()

        if i == 0:
            gt_change = 0.0
            pred_change = 0.0
        else:
            gt_change = (
                gt_token_seq[i].detach().cpu().reshape(-1)
                != gt_token_seq[i - 1].detach().cpu().reshape(-1)
            ).float().mean().item()
            pred_change = (
                pred_token_seq[i].detach().cpu().reshape(-1)
                != pred_token_seq[i - 1].detach().cpu().reshape(-1)
            ).float().mean().item()

        print(
            f">> step={i:02d} t={args.start_t+i:03d} "
            f"token_acc={token_acc:.4f} gt_change={gt_change:.4f} "
            f"pred_change={pred_change:.4f} img_l1={img_l1:.6f} img_mse={img_mse:.6e}",
            flush=True,
        )

    save_rollout_grid(
        gt_imgs,
        pred_imgs,
        args.out_png,
        viz_vmin=args.viz_vmin,
        viz_vmax=args.viz_vmax,
        err_vmax=args.viz_err_vmax,
        every=args.save_every_frame,
    )


if __name__ == "__main__":
    main()
