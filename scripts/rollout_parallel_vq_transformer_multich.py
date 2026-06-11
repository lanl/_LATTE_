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

from scripts.custom.train_parallel_vq_transformer_multich import ParallelVQTransformerMultiCh
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


def load_parallel_multich_from_ckpt(path: str, device: torch.device) -> Tuple[ParallelVQTransformerMultiCh, dict]:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("cfg", {})

    required = [
        "vocab_size",
        "n_embed",
        "pad_id",
        "max_Hq",
        "max_Wq",
        "num_datasets",
        "max_channels",
    ]
    for k in required:
        if k not in ckpt:
            raise RuntimeError(f"Checkpoint missing key {k!r}")

    model = ParallelVQTransformerMultiCh(
        vocab_size=int(ckpt["vocab_size"]),
        pad_id=int(ckpt["pad_id"]),
        max_Hq=int(ckpt["max_Hq"]),
        max_Wq=int(ckpt["max_Wq"]),
        num_datasets=int(ckpt["num_datasets"]),
        max_channels=int(ckpt["max_channels"]),
        n_layer=int(cfg.get("n_layer", 8)),
        n_head=int(cfg.get("n_head", 8)),
        n_embd=int(cfg.get("n_embd", 512)),
        dropout=float(cfg.get("dropout", 0.0)),
        use_dataset_emb=not bool(cfg.get("no_dataset_emb", False)),
        use_channel_emb=not bool(cfg.get("no_channel_emb", False)),
    )

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f">> loaded multich parallel: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    model = model.to(device).eval()
    return model, ckpt


def make_position_tensors(C: int, Hq: int, Wq: int, device: torch.device):
    rows = torch.arange(Hq, dtype=torch.long, device=device).view(1, Hq, 1).expand(C, Hq, Wq)
    cols = torch.arange(Wq, dtype=torch.long, device=device).view(1, 1, Wq).expand(C, Hq, Wq)
    channels = torch.arange(C, dtype=torch.long, device=device).view(C, 1, 1).expand(C, Hq, Wq)

    rows = rows.reshape(1, C * Hq * Wq)
    cols = cols.reshape(1, C * Hq * Wq)
    channels = channels.reshape(1, C * Hq * Wq)
    valid = torch.ones((1, C * Hq * Wq), dtype=torch.bool, device=device)

    return rows, cols, channels, valid


@torch.inference_mode()
def predict_next_multich(
    model: ParallelVQTransformerMultiCh,
    cur_chw: torch.Tensor,
    *,
    dataset_id: int,
    temperature: float = 0.0,
    top_k: int = 0,
) -> torch.Tensor:
    """
    cur_chw: [C,Hq,Wq] long on device
    returns: [C,Hq,Wq] long on device
    """
    device = cur_chw.device
    C, Hq, Wq = cur_chw.shape
    L = C * Hq * Wq

    tokens_in = cur_chw.reshape(1, L).long()
    rows, cols, channels, valid = make_position_tensors(C, Hq, Wq, device)

    dataset_id_t = torch.tensor([int(dataset_id)], dtype=torch.long, device=device)

    logits = model(
        tokens_in,
        rows,
        cols,
        valid,
        dataset_id=dataset_id_t,
        channels=channels,
    )

    # Only allow real VQ code IDs, not special tokens.
    logits = logits[:, :, : model.vocab_size - 4]

    if temperature is None or float(temperature) <= 0.0:
        pred = torch.argmax(logits, dim=-1)
    else:
        logits_s = logits / float(temperature)

        if int(top_k) > 0:
            k = min(int(top_k), logits_s.size(-1))
            vals, idx = torch.topk(logits_s, k=k, dim=-1)
            probs = torch.softmax(vals, dim=-1)
            pick = torch.multinomial(probs.reshape(-1, k), num_samples=1).view(1, L, 1)
            pred = idx.gather(-1, pick).squeeze(-1)
        else:
            probs = torch.softmax(logits_s, dim=-1)
            pred = torch.multinomial(probs.reshape(-1, probs.size(-1)), num_samples=1).view(1, L)

    return pred.reshape(C, Hq, Wq).long()


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
def decode_one_channel(vqvae, tokens_hq_wq: torch.Tensor) -> torch.Tensor:
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


def save_multich_rollout_grid(
    gt_imgs,       # list length T, each [C] tensors
    pred_imgs,     # list length T, each [C] tensors
    out_png: str,
    *,
    viz_vmin: float,
    viz_vmax: float,
    err_vmax: float,
    every: int = 1,
):
    """
    Grid layout:
      rows for each channel:
        GT ch0
        Pred ch0
        Err ch0
        GT ch1
        Pred ch1
        Err ch1
        ...
      columns are time steps.
    """
    idxs = list(range(0, len(gt_imgs), max(1, int(every))))
    if idxs[-1] != len(gt_imgs) - 1:
        idxs.append(len(gt_imgs) - 1)

    C = len(gt_imgs[0])
    first = to_color(gt_imgs[0][0], viz_vmin, viz_vmax)
    tile_w, tile_h = first.size

    n_cols = len(idxs)
    n_rows = 3 * C

    grid = Image.new("RGB", (tile_w * n_cols, tile_h * n_rows))

    for col_j, t_idx in enumerate(idxs):
        for ch in range(C):
            gt_tile = to_color(gt_imgs[t_idx][ch], viz_vmin, viz_vmax)
            pr_tile = to_color(pred_imgs[t_idx][ch], viz_vmin, viz_vmax)
            er_tile = to_err((gt_imgs[t_idx][ch] - pred_imgs[t_idx][ch]).abs(), err_vmax)

            row_base = 3 * ch
            x0 = col_j * tile_w
            grid.paste(gt_tile, (x0, (row_base + 0) * tile_h))
            grid.paste(pr_tile, (x0, (row_base + 1) * tile_h))
            grid.paste(er_tile, (x0, (row_base + 2) * tile_h))

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    grid.save(out_png)

    print(
        f">> saved rollout PNG: {out_png} "
        f"columns={idxs} rows_per_channel=[gt,pred,abs_error] "
        f"C={C} tile={tile_w}x{tile_h}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--transformer_ckpt", required=True)
    ap.add_argument("--vqvae_ckpt", required=True)
    ap.add_argument("--tokens_npz", required=True)
    ap.add_argument("--out_png", required=True)

    ap.add_argument("--dataset_id", type=int, default=0)
    ap.add_argument("--traj_idx", type=int, default=0)
    ap.add_argument("--start_t", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument(
        "--teacher_forced",
        action="store_true",
        help="Feed true tokens_t at each step. Otherwise feed previous predicted tokens.",
    )

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=0)

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
    if args.start_t < 0 or args.start_t >= time_count - 1:
        raise ValueError(f"start_t={args.start_t} invalid for time_count={time_count}")
    if args.start_t + args.steps >= time_count:
        raise ValueError(
            f"Need start_t + steps < time_count. Got start_t={args.start_t}, "
            f"steps={args.steps}, time_count={time_count}"
        )

    model, ckpt = load_parallel_multich_from_ckpt(args.transformer_ckpt, device=device)

    if int(ckpt["n_embed"]) != int(n_embed):
        raise RuntimeError(f"ckpt n_embed={ckpt['n_embed']} but token file n_embed={n_embed}")
    if int(ckpt["max_Hq"]) < int(Hq) or int(ckpt["max_Wq"]) < int(Wq):
        raise RuntimeError(
            f"ckpt max_Hq,max_Wq=({ckpt['max_Hq']},{ckpt['max_Wq']}) "
            f"but token file Hq,Wq=({Hq},{Wq})"
        )
    if int(ckpt["max_channels"]) < int(C):
        raise RuntimeError(f"ckpt max_channels={ckpt['max_channels']} but token file C={C}")

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
        arr = tokens[traj_start + tt]  # [C,Hq,Wq]
        gt_token_seq.append(torch.from_numpy(arr.astype(np.int64, copy=False)).long())

    pred_token_seq = [gt_token_seq[0].clone().to(device)]

    cur = pred_token_seq[0]
    for s in range(args.steps):
        if args.teacher_forced:
            model_input = gt_token_seq[s].to(device)
        else:
            model_input = cur

        nxt = predict_next_multich(
            model,
            model_input,
            dataset_id=int(args.dataset_id),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
        )

        pred_token_seq.append(nxt)
        cur = nxt

    gt_imgs = []
    pred_imgs = []

    for i in range(args.steps + 1):
        gt_ch_imgs = []
        pr_ch_imgs = []

        for ch in range(C):
            gt_img = decode_one_channel(vqvae, gt_token_seq[i][ch])
            pr_img = decode_one_channel(vqvae, pred_token_seq[i][ch])
            gt_ch_imgs.append(gt_img)
            pr_ch_imgs.append(pr_img)

        gt_imgs.append(gt_ch_imgs)
        pred_imgs.append(pr_ch_imgs)

    for i in range(args.steps + 1):
        token_acc = (
            pred_token_seq[i].detach().cpu().reshape(-1)
            == gt_token_seq[i].detach().cpu().reshape(-1)
        ).float().mean().item()

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

        img_l1_vals = []
        img_mse_vals = []
        for ch in range(C):
            img_l1_vals.append(torch.mean((gt_imgs[i][ch] - pred_imgs[i][ch]).abs()).item())
            img_mse_vals.append(torch.mean((gt_imgs[i][ch] - pred_imgs[i][ch]) ** 2).item())

        print(
            f">> step={i:02d} t={args.start_t+i:03d} "
            f"token_acc={token_acc:.4f} "
            f"gt_change={gt_change:.4f} pred_change={pred_change:.4f} "
            f"img_l1_mean={float(np.mean(img_l1_vals)):.6f} "
            f"img_mse_mean={float(np.mean(img_mse_vals)):.6e}",
            flush=True,
        )

    save_multich_rollout_grid(
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
