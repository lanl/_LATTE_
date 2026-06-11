#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from PIL import Image

from src.models.lit_vqvae import LitVQVAE
from src.data.registry import build_dataset_from_registry


def parse_names(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def infer_hwc_or_chw(x: torch.Tensor):
    if x.ndim != 3:
        raise RuntimeError(f"Expected 3D image tensor, got {tuple(x.shape)}")

    a, b, c = x.shape

    # CHW if first dim looks like channels.
    if a <= 64 and b >= 16 and c >= 16:
        return "CHW"

    return "HWC"


def select_scalar_channel(image: torch.Tensor, channel_idx: int) -> torch.Tensor:
    """
    Returns HWC scalar image: [H, W, 1]
    """
    layout = infer_hwc_or_chw(image)

    if layout == "HWC":
        h, w, c = image.shape
        if channel_idx >= c:
            raise RuntimeError(f"channel_idx={channel_idx} but image has C={c}")
        x = image[..., channel_idx : channel_idx + 1].contiguous()
        return x.float()

    c, h, w = image.shape
    if channel_idx >= c:
        raise RuntimeError(f"channel_idx={channel_idx} but image has C={c}")
    x = image[channel_idx : channel_idx + 1].permute(1, 2, 0).contiguous()
    return x.float()


def pad_to_multiple_bchw(x: torch.Tensor, multiple: int, mode: str = "replicate"):
    """
    x: [B, C, H, W]
    returns padded_x, original_h, original_w
    """
    _, _, h, w = x.shape

    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple

    if pad_h == 0 and pad_w == 0:
        return x, h, w

    # pad format: left, right, top, bottom
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x_pad, h, w


def to_color(x_hw: torch.Tensor, vmin: float, vmax: float, cmap_name: str = "viridis") -> Image.Image:
    a = x_hw.detach().float().cpu().numpy()
    a = (a - vmin) / (vmax - vmin + 1e-12)
    a = np.clip(a, 0.0, 1.0)
    cmap = cm.get_cmap(cmap_name)
    rgb = (cmap(a)[..., :3] * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def to_error(err_hw: torch.Tensor, err_vmax: float) -> Image.Image:
    e = err_hw.detach().float().cpu()
    e = torch.clamp(e / float(err_vmax), 0.0, 1.0)
    e = (e * 255.0).to(torch.uint8).numpy()
    return Image.fromarray(e, mode="L").convert("RGB")


def save_triplet_png(
    x: torch.Tensor,
    x_rec: torch.Tensor,
    out_path: str,
    viz_vmin: float,
    viz_vmax: float,
    err_vmax: float,
):
    """
    x, x_rec: [1, 1, H, W]
    Saves horizontal grid: GT | recon | abs error
    """
    x0 = x[0, 0].detach().cpu()
    r0 = x_rec[0, 0].detach().cpu()
    e0 = (x0 - r0).abs()

    gt_im = to_color(x0, viz_vmin, viz_vmax)
    rc_im = to_color(r0, viz_vmin, viz_vmax)
    er_im = to_error(e0, err_vmax)

    w, h = gt_im.size
    grid = Image.new("RGB", (w * 3, h))
    grid.paste(gt_im, (0, 0))
    grid.paste(rc_im, (w, 0))
    grid.paste(er_im, (2 * w, 0))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.save(out_path)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--datasets_json", required=True)
    ap.add_argument("--dataset_names", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--indices", type=str, default="0,1,2,5,10")
    ap.add_argument("--channel_idx", type=int, default=0)

    ap.add_argument("--down_factor", type=int, default=8)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--viz_vmin", type=float, default=-1.0)
    ap.add_argument("--viz_vmax", type=float, default=0.7)
    ap.add_argument("--viz_err_vmax", type=float, default=0.1)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.datasets_json, "r") as f:
        reg = json.load(f)

    dataset_names = parse_names(args.dataset_names)
    indices = parse_ints(args.indices)

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )

    print(f">> Loading checkpoint: {args.ckpt}", flush=True)
    model = LitVQVAE.load_from_checkpoint(args.ckpt, map_location="cpu")
    model.eval()
    model.to(device)

    for name in dataset_names:
        print(f">> Building dataset: {name} split={args.split}", flush=True)
        ds = build_dataset_from_registry(reg, name=name, split=args.split)

        print(f">> Dataset length: {len(ds)}", flush=True)

        for idx in indices:
            if idx < 0 or idx >= len(ds):
                print(f">> Skipping idx={idx}; out of range len={len(ds)}", flush=True)
                continue

            item = ds[idx]
            if not isinstance(item, dict) or "image" not in item:
                raise RuntimeError(f"Dataset item must be dict with key 'image', got {type(item)}")

            image = item["image"]
            if not torch.is_tensor(image):
                raise RuntimeError(f"item['image'] must be tensor, got {type(image)}")

            # [H,W,C] scalar
            x_hwc = select_scalar_channel(image, args.channel_idx).clamp(-1.0, 1.0)

            # [H,W,1] -> [1,1,H,W]
            x = x_hwc.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)

            x_pad, orig_h, orig_w = pad_to_multiple_bchw(x, args.down_factor)

            with torch.no_grad():
                x_rec_pad, qloss, info = model(x_pad)
                x_rec_pad = x_rec_pad.clamp(-1.0, 1.0)

            # Crop back if padding was applied.
            x_rec = x_rec_pad[..., :orig_h, :orig_w]
            x_eval = x[..., :orig_h, :orig_w]

            recon_l1 = F.l1_loss(x_rec, x_eval).item()
            recon_l2 = F.mse_loss(x_rec, x_eval).item()

            inds = info[-1]
            if inds is not None:
                uniq = inds.reshape(-1).unique().numel()
                inds_shape = tuple(inds.shape)
            else:
                uniq = -1
                inds_shape = None

            out_name = (
                f"{name}__split-{args.split}"
                f"__idx-{idx:06d}"
                f"__ch-{args.channel_idx:02d}"
                f"__H-{orig_h}__W-{orig_w}.png"
            )
            out_path = os.path.join(args.out_dir, out_name)

            save_triplet_png(
                x=x_eval.detach().cpu(),
                x_rec=x_rec.detach().cpu(),
                out_path=out_path,
                viz_vmin=args.viz_vmin,
                viz_vmax=args.viz_vmax,
                err_vmax=args.viz_err_vmax,
            )

            print(
                f">> Saved {out_path} "
                f"shape=({orig_h},{orig_w}) "
                f"recon_l1={recon_l1:.6g} recon_l2={recon_l2:.6g} "
                f"qloss={float(qloss):.6g} unique_codes={uniq} inds_shape={inds_shape}",
                flush=True,
            )


if __name__ == "__main__":
    main()
