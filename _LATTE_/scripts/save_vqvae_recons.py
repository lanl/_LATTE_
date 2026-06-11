#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import Dict, Any, List

import numpy as np
import torch
import torch.utils.data as tud

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from PIL import Image

from src.models.lit_vqvae import LitVQVAE
from src.data.registry import build_dataset_from_registry
from src.data.datasets.scalar_channel_wrapper import ScalarChannelDataset


def infer_real_num_channels_from_cfg(cfg: Dict[str, Any]) -> int:
    kwargs = cfg.get("kwargs", {})

    if "field_keys" in kwargs:
        return len(kwargs["field_keys"])

    if "channels_in" in kwargs:
        return int(kwargs["channels_in"])

    if "model_channels" in kwargs:
        return int(kwargs["model_channels"])

    return 1


def parse_names(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_ints(s: str) -> List[int]:
    if s is None or str(s).strip() == "":
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


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

    ap.add_argument("--scalar_crop_size", type=int, default=128)
    ap.add_argument("--samples_per_dataset", type=int, default=1000)
    ap.add_argument("--scalar_seed", type=int, default=123)

    ap.add_argument(
        "--local_indices",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated local scalar indices to save for each dataset.",
    )

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--viz_vmin", type=float, default=-1.0)
    ap.add_argument("--viz_vmax", type=float, default=0.7)
    ap.add_argument("--viz_err_vmax", type=float, default=0.1)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.datasets_json, "r") as f:
        reg = json.load(f)

    datasets_cfg = reg["datasets"] if "datasets" in reg else reg
    dataset_names = parse_names(args.dataset_names)
    local_indices = parse_ints(args.local_indices)

    if len(local_indices) == 0:
        raise ValueError("--local_indices must contain at least one integer")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    print(f">> Loading checkpoint: {args.ckpt}", flush=True)
    model = LitVQVAE.load_from_checkpoint(args.ckpt, map_location="cpu")
    model.eval()
    model.to(device)

    for dataset_i, name in enumerate(dataset_names):
        print(f">> Building {args.split} dataset: {name}", flush=True)

        base = build_dataset_from_registry(reg, name=name, split=args.split)

        cfg = datasets_cfg[name]
        num_channels = infer_real_num_channels_from_cfg(cfg)

        scalar_ds = ScalarChannelDataset(
            base_dataset=base,
            dataset_name=name,
            num_channels=num_channels,
            samples_per_dataset=args.samples_per_dataset,
            crop_size=args.scalar_crop_size,
            seed=args.scalar_seed + 99991 + 1009 * dataset_i,
            resize_if_smaller=True,
            return_channels_last=True,
            min_crop_std=0.0,
            crop_trials=1,
        )

        for local_idx in local_indices:
            if local_idx < 0 or local_idx >= len(scalar_ds):
                print(f">> Skipping {name} local_idx={local_idx}; out of range len={len(scalar_ds)}", flush=True)
                continue

            item = scalar_ds[local_idx]
            batch = {
                k: (v.unsqueeze(0) if torch.is_tensor(v) else v)
                for k, v in item.items()
            }

            x = model.get_input(batch).to(device)

            with torch.no_grad():
                x_rec, qloss, info = model(x)
                x_rec = x_rec.clamp(-1.0, 1.0)

            channel_idx = int(item.get("channel_idx", -1))
            base_idx = int(item.get("base_idx", -1))

            out_name = (
                f"{name}__split-{args.split}"
                f"__local-{local_idx:05d}"
                f"__base-{base_idx:07d}"
                f"__ch-{channel_idx:02d}.png"
            )
            out_path = os.path.join(args.out_dir, out_name)

            save_triplet_png(
                x=x.detach().cpu(),
                x_rec=x_rec.detach().cpu(),
                out_path=out_path,
                viz_vmin=args.viz_vmin,
                viz_vmax=args.viz_vmax,
                err_vmax=args.viz_err_vmax,
            )

            recon_l1 = torch.nn.functional.l1_loss(x_rec, x).item()
            recon_l2 = torch.nn.functional.mse_loss(x_rec, x).item()

            inds = info[-1]
            if inds is not None:
                uniq = inds.reshape(-1).unique().numel()
            else:
                uniq = -1

            print(
                f">> Saved {out_path} "
                f"dataset={name} local_idx={local_idx} base_idx={base_idx} "
                f"channel={channel_idx} recon_l1={recon_l1:.6g} "
                f"recon_l2={recon_l2:.6g} qloss={float(qloss):.6g} "
                f"unique_codes={uniq}",
                flush=True,
            )


if __name__ == "__main__":
    main()
