#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic VQ-VAE trainer.

- Dataset-agnostic: datasets live in src/data/datasets/*.py
- Dataset selection comes from a registry JSON (configs/datasets_*.json)
- Each dataset must return {"image": torch.float32 tensor} in [-1,1], layout HWC or CHW.

Expected registry schema (per dataset):
{
  "class": "src.data.datasets.hdf5_multi_key_field:HDF5MultiKeyFieldDataset",
  "kwargs": {... dataset-wide kwargs ...},
  "splits": {
    "train": {"traj_start":0, "traj_count":..., ...},
    "val":   {"traj_start":0, "traj_count":..., ...}
  }
}
"""

import os, json, argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.utils.data as tud

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

from PIL import Image

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.utilities import rank_zero_only

from src.models.lit_vqvae import LitVQVAE
from src.data.registry import build_dataset_from_registry  
from src.data.datasets.scalar_channel_wrapper import ScalarChannelDataset
torch.set_float32_matmul_precision("high")

# -----------------------------
# Helpers
# -----------------------------
def infer_real_num_channels_from_cfg(cfg: Dict[str, Any]) -> int:
    """
    Infer the number of real physical/scalar channels.

    Prefer true input channels / field keys over model_channels, because
    model_channels may include padding/truncation for some datasets.
    """
    kwargs = cfg.get("kwargs", {})

    if "field_keys" in kwargs:
        return len(kwargs["field_keys"])

    if "channels_in" in kwargs:
        return int(kwargs["channels_in"])

    if "model_channels" in kwargs:
        return int(kwargs["model_channels"])

    return 1

def parse_overrides(s: Optional[str]) -> Dict[str, int]:
    """
    "CE-CRP=1000,NS-Gauss=5000" -> {"CE-CRP":1000, "NS-Gauss":5000}
    """
    if s is None or str(s).strip() == "":
        return {}
    out = {}
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad override '{p}'. Expected NAME=N.")
        k, v = p.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def _infer_chw(x: torch.Tensor) -> Tuple[int, int, int]:
    """
    Accepts a single example tensor either HWC or CHW (no batch).
    Returns (C,H,W).
    """
    if x.dim() != 3:
        raise RuntimeError(f"Expected 3D tensor (HWC or CHW) but got {tuple(x.shape)}")

    a, b, c = x.shape
    # Heuristic: channels usually small (<=64). If first dim small -> CHW, else HWC.
    if a <= 64 and b >= 16 and c >= 16:
        C, H, W = a, b, c
    else:
        H, W, C = a, b, c
    return int(C), int(H), int(W)


def _peek_first_image(ds: tud.Dataset) -> torch.Tensor:
    ex = ds[0]
    if not isinstance(ex, dict) or "image" not in ex:
        raise RuntimeError("Dataset must return a dict with key 'image'")
    x = ex["image"]
    if not torch.is_tensor(x):
        raise RuntimeError("'image' must be a torch.Tensor")
    return x


# -----------------------------
# Callbacks (kept in-script for now)
# -----------------------------
class RunningCodeUsageCallback(Callback):
    def __init__(self, every_steps: int = 200, window_steps: int = 20):
        super().__init__()
        self.every_steps = max(int(every_steps), 1)
        self.window_steps = max(int(window_steps), 1)
        self._recent = []
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step % self.every_steps) != 0:
            return

        with torch.no_grad():
            x = pl_module.get_input(batch).to(pl_module.device, non_blocking=True)
            _, _, info = pl_module.encode(x)
            inds = info[-1]
            if inds is None:
                return

            inds = inds.reshape(-1).detach().cpu()
            self._recent.append(inds)

            if len(self._recent) > self.window_steps:
                self._recent.pop(0)

            all_inds = torch.cat(self._recent, dim=0)
            uniq = all_inds.unique().numel()
            ntok = all_inds.numel()
            ratio = float(uniq) / float(max(ntok, 1))

            pl_module.log(
                "train/codebook_unique_window",
                float(uniq),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
            )
            pl_module.log(
                "train/codebook_unique_ratio_window",
                ratio,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )

            if trainer.is_global_zero:
                trainer.print(
                    f">> [code-window] step={trainer.global_step} "
                    f"last_logged_batches={len(self._recent)} "
                    f"uniq={uniq}/{ntok} ratio={ratio:.4f}"
                )
class PrintValMetrics(Callback):
    def on_validation_end(self, trainer, pl_module):
        # Called after *each* validation run (including mid-epoch val_check_interval runs)
        if not trainer.is_global_zero:
            return
        if trainer.sanity_checking:
            return

        m = trainer.callback_metrics

        # Prefer "val/..." but fall back to anything that begins with "val"
        keys = [k for k in m.keys() if str(k).startswith("val/")]
        if not keys:
            keys = [k for k in m.keys() if str(k).startswith("val")]

        if not keys:
            trainer.print(">> [val] no val metrics found. callback_metrics keys = " +
                          ", ".join(sorted(map(str, m.keys()))))
            return

        # Current LR
        lr = None
        try:
            opt = trainer.optimizers[0]
            lr = opt.param_groups[0]["lr"]
        except Exception:
            pass

        line = "  ".join(f"{k}={float(m[k]):.6g}" for k in sorted(keys))
        if lr is not None:
            line += f"  lr={lr:.6e}"
        else:
            line += "  lr=NA"

        trainer.print(f">> [val] epoch={trainer.current_epoch} step={trainer.global_step}  {line}")

class SaveReconCallback(Callback):
    """
    Saves a 3xC grid PNG at the start of specified epochs:
      Row 0: GT channels
      Row 1: Recon channels
      Row 2: Abs error channels

    Dataset-agnostic:
      - Works for any channel count
      - Uses fixed mapping from [-1,1] for GT and Recon by default
    """
    def __init__(
        self,
        val_dataloader,
        epoch_list,
        out_dir: str,
        fname_prefix: str = "recon",
        max_channels: int = 8,
        sample_idx: int = 0,
        viz_vmin: float = -1.0,
        viz_vmax: float = 1.0,
        err_vmax: float = 0.1,
        cmap_name: str = "viridis",
    ):
        super().__init__()
        self.val_dataloader = val_dataloader
        self.epochs_to_save = set(int(e) for e in epoch_list if int(e) >= 0)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.fname_prefix = fname_prefix

        self.max_channels = int(max_channels)
        self.sample_idx = int(sample_idx)

        self.viz_vmin = float(viz_vmin)
        self.viz_vmax = float(viz_vmax)
        self.err_vmax = float(err_vmax)
        self.cmap_name = str(cmap_name)

        self._done_epochs = set()

    def _extract_chw_first(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x0 = x[0]
        elif x.dim() == 3:
            x0 = x
        else:
            raise RuntimeError(f"Unexpected tensor shape: {tuple(x.shape)}")

        C, H, W = _infer_chw(x0)
        if x0.shape[0] == C:
            return x0.contiguous()
        return x0.permute(2, 0, 1).contiguous()


    def _to_color_fixed(self, x_hw: torch.Tensor) -> Image.Image:
        a = x_hw.detach().float().cpu().numpy()
        a = (a - self.viz_vmin) / (self.viz_vmax - self.viz_vmin + 1e-12)
        a = np.clip(a, 0.0, 1.0)
        cmap = cm.get_cmap(self.cmap_name)
        rgb = (cmap(a)[..., :3] * 255.0).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")

    def _to_gray_err(self, err_hw: torch.Tensor) -> Image.Image:
        e = err_hw.detach().float().cpu()
        e = torch.clamp(e / float(self.err_vmax), 0.0, 1.0)
        e = (e * 255.0).to(torch.uint8).numpy()
        return Image.fromarray(e, mode="L")

    @rank_zero_only
    def _save_grid(self, gt: torch.Tensor, recon: torch.Tensor, epoch: int):
        gt_chw = self._extract_chw_first(gt)
        rc_chw = self._extract_chw_first(recon)

        C = min(self.max_channels, gt_chw.shape[0], rc_chw.shape[0])
        gt_tiles = [self._to_color_fixed(gt_chw[c]) for c in range(C)]
        rc_tiles = [self._to_color_fixed(rc_chw[c]) for c in range(C)]
        err = (gt_chw[:C] - rc_chw[:C]).abs()
        err_tiles = [self._to_gray_err(err[c]) for c in range(C)]

        tile_w, tile_h = gt_tiles[0].size
        grid = Image.new("RGB", (tile_w * C, tile_h * 3))

        for c in range(C):
            grid.paste(gt_tiles[c], (c * tile_w, 0))
            grid.paste(rc_tiles[c], (c * tile_w, tile_h))
            grid.paste(err_tiles[c].convert("RGB"), (c * tile_w, tile_h * 2))

        path = os.path.join(self.out_dir, f"{self.fname_prefix}_epoch{epoch+1:04d}.png")
        grid.save(path)
        print(f">> Saved recon PNG: {path} (C={C}, tile={tile_w}x{tile_h})", flush=True)

    def on_train_epoch_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        ce = trainer.current_epoch
        if ce not in self.epochs_to_save or ce in self._done_epochs:
            return

        ds = self.val_dataloader.dataset
        idx = max(0, min(self.sample_idx, len(ds) - 1))
        batch = ds[idx]
        batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in batch.items()}

        x = pl_module.get_input(batch).to(pl_module.device, non_blocking=True)
        with torch.no_grad():
            was_training = pl_module.training
            pl_module.eval()
            x_rec, _, _ = pl_module(x)
            x_rec = x_rec.clamp(-1, 1)
            self._save_grid(x.detach().cpu(), x_rec.detach().cpu(), ce)
            self._done_epochs.add(ce)
            if was_training:
                pl_module.train()

class TrajectoryLocalSampler(tud.Sampler[int]):
    """
    Assumes frame dataset indexing is:
      traj 0 -> indices [0, T)
      traj 1 -> indices [T, 2T)
      ...
    """
    def __init__(self, dataset, shuffle_frames_within_traj: bool = True, seed: int = 1337):
        self.dataset = dataset
        self.T = int(dataset.T)
        self.n_traj = len(dataset.base)
        self.shuffle_frames_within_traj = bool(shuffle_frames_within_traj)
        self.seed = int(seed)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        traj_ids = np.arange(self.n_traj)
        rng.shuffle(traj_ids)

        for traj_i in traj_ids:
            start = int(traj_i) * self.T
            inds = np.arange(start, start + self.T)
            if self.shuffle_frames_within_traj:
                rng.shuffle(inds)
            for idx in inds:
                yield int(idx)

    def __len__(self):
        return self.n_traj * self.T

def df_to_ch_mult(df: int):
    # Each extra entry beyond the first adds one 2x downsample.
    # This matches the style you used already.
    if df == 4:   return [1, 2, 2]          # 2 downsamples -> /4
    if df == 8:   return [1, 2, 2, 2]       # 3 downsamples -> /8
    if df == 16:  return [1, 2, 2, 2, 2]    # 4 downsamples -> /16
    raise ValueError(f"Unsupported down_factor={df}")

def _summarize_image_tensor(x: torch.Tensor, name: str = "image") -> None:
    """
    x: single example tensor, HWC or CHW (no batch)
    Prints shape + stats and checks for NaNs/Infs.
    """
    if not torch.is_tensor(x):
        raise RuntimeError(f"{name} is not a torch.Tensor")
    xf = x.detach().float()
    mn = float(xf.min())
    mx = float(xf.max())
    mean = float(xf.mean())
    std = float(xf.std())
    bad = (not torch.isfinite(xf).all().item())
    print(f">> [{name}] shape={tuple(x.shape)} dtype={x.dtype} min={mn:.6g} max={mx:.6g} mean={mean:.6g} std={std:.6g} finite={not bad}", flush=True)
    if bad:
        raise RuntimeError(f"{name} contains NaN/Inf")

def _assert_in_m11(x: torch.Tensor, name: str, tol: float = 1.05) -> None:
    xf = x.detach().float()
    mn = float(xf.min()); mx = float(xf.max())
    if mn < -tol or mx > tol:
        raise RuntimeError(
            f"{name} not in [-1,1] (tol={tol}). min={mn:.6g} max={mx:.6g}. "
            f"Fix normalization inside the dataset class (preferred)."
        )


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--datasets_json", required=True, help="Registry JSON with datasets (class/kwargs/splits).")
    ap.add_argument("--train_datasets", required=True, help="Comma-separated dataset names to use for TRAIN.")
    ap.add_argument("--val_datasets", default=None, help="Comma-separated dataset names for VAL (default: same as train).")

    ap.add_argument("--train_traj_override", default=None, help="Per-dataset overrides: NAME=N,NAME=N (applies within train split).")
    ap.add_argument("--val_traj_override", default=None, help="Per-dataset overrides: NAME=N,NAME=N (applies within val split).")

    # Optional global overrides pushed into dataset kwargs
    ap.add_argument("--time_start", type=int, default=None)
    ap.add_argument("--time_count", type=int, default=None)
    ap.add_argument("--res_h", type=int, default=None)
    ap.add_argument("--res_w", type=int, default=None)
    ap.add_argument("--model_channels", type=int, default=None)

    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--num_workers", type=int, default=8)

    # Scalar universal VQ-VAE data mode
    ap.add_argument(
        "--scalarize_channels",
        action="store_true",
        help="Wrap each dataset to emit one scalar channel image [H,W,1] instead of all channels.",
    )
    ap.add_argument(
        "--samples_per_dataset",
        type=int,
        default=10000,
        help="When --scalarize_channels, number of scalar training samples per dataset per epoch.",
    )
    ap.add_argument(
        "--val_samples_per_dataset",
        type=int,
        default=1000,
        help="When --scalarize_channels, number of scalar validation samples per dataset.",
    )
    ap.add_argument(
        "--scalar_crop_size",
        type=int,
        default=128,
        help="When --scalarize_channels, crop/resize scalar samples to this square size.",
    )
    ap.add_argument(
        "--scalar_seed",
        type=int,
        default=123,
        help="Seed for scalar frame/channel/crop sampling.",
    )
    ap.add_argument(
        "--min_crop_std",
        type=float,
        default=0.0,
        help="When --scalarize_channels, prefer random crops with std >= this value.",
    )
    ap.add_argument(
        "--crop_trials",
        type=int,
        default=1,
        help="When --scalarize_channels, number of random crop attempts before using best crop.",
    )

    ap.add_argument("--traj_local_sampling", action="store_true",
                help="Use trajectory-local sampling instead of global shuffle for frame datasets.")
    ap.add_argument("--shuffle_frames_within_traj", action="store_true",
                help="When using trajectory-local sampling, shuffle frame order within each trajectory.")

    # VQ params
    ap.add_argument("--n_embed", type=int, default=2048)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument(
        "--vq_loss_weight",
        type=float,
        default=5.0,
        help="Multiplier on VQ/codebook loss.",
    )
    ap.add_argument(
        "--grad_loss_weight",
        type=float,
        default=0.05,
        help="Weight for finite-difference gradient reconstruction loss.",
    )
    ap.add_argument(
        "--usage_entropy_weight",
        type=float,
        default=0.0,
        help="Weight for codebook usage entropy regularization. Use 0.0 for first 8192-code runs.",
    )
    ap.add_argument(
        "--dead_code_reset_every",
        type=int,
        default=0,
        help="Reset dead codes every N steps. Use 0 to disable.",
    )
    ap.add_argument(
        "--dead_code_reset_warmup_steps",
        type=int,
        default=2000,
        help="Do not reset dead codes before this global step.",
    )
    ap.add_argument(
        "--dead_code_reset_threshold",
        type=float,
        default=1.0,
        help="Codes with usage count <= this threshold are considered dead for reset.",
    )
    ap.add_argument(
        "--max_dead_code_resets",
        type=int,
        default=512,
        help="Maximum number of dead codes to reset at one reset event.",
    )

    # AE/UNet knobs (make resolution-agnostic)
    ap.add_argument("--base_ch", type=int, default=128)
    ap.add_argument("--num_res_blocks", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)

    # trainer
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--devices", type=int, default=4)
    ap.add_argument("--num_nodes", type=int, default=1)
    ap.add_argument("--precision", default="32", help="16-mixed, 32, 16")
    ap.add_argument("--max_epochs", type=int, default=50)
    ap.add_argument("--resume_ckpt", type=str, default=None, help="Lightning .ckpt to fully resume.")
    ap.add_argument("--ckpt_every_n_train_steps", type=int, default=100,
                    help="Save a full checkpoint every N training steps.")
    ap.add_argument("--log_every_n_steps", type=int, default=50)
    ap.add_argument("--val_check_interval", type=float, default=0.5)
    ap.add_argument("--disable_pbar", action="store_true")
    ap.add_argument("--code_usage_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no_quant", action="store_true")

    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--lr_schedule", type=str, default="cosine", choices=["none", "cosine"])

    ap.add_argument("--resume_weights_only", type=str, default=None,
                    help="Path to .ckpt to load model weights only (no optimizer/scheduler).")

    ap.add_argument("--dry_run", action="store_true", help="Load data + model, run 1 forward, then exit.")

    # recon viz
    ap.add_argument("--save_png", type=str, default="",
                    help="Comma-separated epoch numbers (1-based) to save recon PNGs, e.g. '5,10,15'.")
    ap.add_argument("--viz_channels", type=int, default=8)
    ap.add_argument("--viz_vmin", type=float, default=-1.0)
    ap.add_argument("--viz_vmax", type=float, default=0.7)
    ap.add_argument("--viz_err_vmax", type=float, default=0.1)
    ap.add_argument("--viz_sample_idx", type=int, default=600)
    ap.add_argument("--down_factor", type=int, default=8, choices=[4, 8, 16],
                help="Total spatial downsampling of the VQ encoder/decoder. Controls latent grid size.")
    ap.add_argument("--ch_mult", type=str, default="",
        help="Override ddconfig ch_mult as comma list, e.g. '1,1,2,2'. If empty, uses --down_factor mapping.")
    ap.add_argument("--dry_run_png", type=str, default="",
        help="If set and --dry_run, save a recon grid PNG to this path.")


    args = ap.parse_args()
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.datasets_json, "r") as f:
        reg = json.load(f)

    train_names = [s.strip() for s in args.train_datasets.split(",") if s.strip()]
    val_names = train_names if args.val_datasets is None else [s.strip() for s in args.val_datasets.split(",") if s.strip()]

    tr_over = parse_overrides(args.train_traj_override)
    va_over = parse_overrides(args.val_traj_override)

    # Build datasets
    train_sets: List[tud.Dataset] = []
    val_sets: List[tud.Dataset] = []

    # Extra kwargs pushed into dataset constructors (only if provided)
    global_kw: Dict[str, Any] = {}
    for k in ["time_start", "time_count", "res_h", "res_w"]:
        v = getattr(args, k)
        if v is not None:
            global_kw[k] = v

    if (not args.scalarize_channels) and args.model_channels is not None:
        global_kw["model_channels"] = args.model_channels

    def _build(name: str, split: str, over: Dict[str, int]) -> tud.Dataset:
        # Optionally override traj_count without editing JSON
        extra = dict(global_kw)
        # Look up base split config so we can clamp traj_count
        cfg = reg["datasets"][name] if "datasets" in reg else reg[name]
        sp = cfg.get("splits", {}).get(split, {})
        if "traj_count" in sp and name in over:
            extra["traj_count"] = min(int(sp["traj_count"]), int(over[name]))
        elif name in over:
            # If dataset doesn't use traj_count, ignore safely
            pass
        return build_dataset_from_registry(reg, name=name, split=split, extra_kwargs=extra)

    print(">> Building TRAIN datasets:", train_names, flush=True)
    for n in train_names:
        ds = _build(n, "train", tr_over)
        print(f"   - {n}: frames={len(ds)}", flush=True)

        # --- SANITY: base dataset must return {"image": ...} in [-1,1]
        x0n = _peek_first_image(ds)
        _summarize_image_tensor(x0n, name=f"{n}/train image0")
        _assert_in_m11(x0n, name=f"{n}/train image0")

        if args.scalarize_channels:
            cfg = reg["datasets"][n] if "datasets" in reg else reg[n]
            num_channels = infer_real_num_channels_from_cfg(cfg)

            ds = ScalarChannelDataset(
                base_dataset=ds,
                dataset_name=n,
                num_channels=num_channels,
                samples_per_dataset=args.samples_per_dataset,
                crop_size=args.scalar_crop_size,
                seed=args.scalar_seed + 1009 * len(train_sets),
                resize_if_smaller=True,
                return_channels_last=True,
                min_crop_std=args.min_crop_std,
                crop_trials=args.crop_trials,
            )

            x0s = _peek_first_image(ds)
            _summarize_image_tensor(x0s, name=f"{n}/train scalar image0")
            _assert_in_m11(x0s, name=f"{n}/train scalar image0")

            print(
                f"   -> scalarized train: samples={len(ds)} "
                f"channels={num_channels} crop={args.scalar_crop_size}",
                flush=True,
            )

        train_sets.append(ds)

    print(">> Building VAL datasets:", val_names, flush=True)
    for n in val_names:
        ds = _build(n, "val", va_over)
        print(f"   - {n}: frames={len(ds)}", flush=True)

        x0n = _peek_first_image(ds)
        _summarize_image_tensor(x0n, name=f"{n}/val image0")
        _assert_in_m11(x0n, name=f"{n}/val image0")

        if args.scalarize_channels:
            cfg = reg["datasets"][n] if "datasets" in reg else reg[n]
            num_channels = infer_real_num_channels_from_cfg(cfg)

            ds = ScalarChannelDataset(
                base_dataset=ds,
                dataset_name=n,
                num_channels=num_channels,
                samples_per_dataset=args.val_samples_per_dataset,
                crop_size=args.scalar_crop_size,
                seed=args.scalar_seed + 99991 + 1009 * len(val_sets),
                resize_if_smaller=True,
                return_channels_last=True,
                min_crop_std=0.0,
                crop_trials=1,
            )

            x0s = _peek_first_image(ds)
            _summarize_image_tensor(x0s, name=f"{n}/val scalar image0")
            _assert_in_m11(x0s, name=f"{n}/val scalar image0")

            print(
                f"   -> scalarized val: samples={len(ds)} "
                f"channels={num_channels} crop={args.scalar_crop_size}",
                flush=True,
            )

        val_sets.append(ds)

    ds_tr = tud.ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
    ds_va = tud.ConcatDataset(val_sets) if len(val_sets) > 1 else val_sets[0]

    print(f">> TRAIN total samples/epoch: {len(ds_tr)}", flush=True)
    print(f">> VAL total samples: {len(ds_va)}", flush=True)

    train_sampler = None
    train_shuffle = True

    if args.scalarize_channels and args.traj_local_sampling:
        raise ValueError("--traj_local_sampling is not compatible with --scalarize_channels.")

    if args.traj_local_sampling:
        if not hasattr(ds_tr, "T") or not hasattr(ds_tr, "base"):
            raise ValueError("Trajectory-local sampling requires a frame dataset with attributes T and base.")
        train_sampler = TrajectoryLocalSampler(
            ds_tr,
            shuffle_frames_within_traj=args.shuffle_frames_within_traj,
            seed=args.seed,
        )
        train_shuffle = False

    dl_tr = tud.DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    dl_va = tud.DataLoader(
        ds_va, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=(args.num_workers > 0)
    )

    # Infer model input shape if not explicitly provided
    x0 = _peek_first_image(ds_tr)
    C, H, W = _infer_chw(x0)

    # Effective model geometry.
    # In scalar mode, the wrapper emits [H,W,1] with fixed scalar_crop_size.
    C_eff, H_eff, W_eff = C, H, W

    if args.scalarize_channels:
        C_eff = 1
        H_eff = int(args.scalar_crop_size)
        W_eff = int(args.scalar_crop_size)
    else:
        if args.model_channels is not None:
            C_eff = int(args.model_channels)
        if args.res_h is not None:
            H_eff = int(args.res_h)
        if args.res_w is not None:
            W_eff = int(args.res_w)


    if args.scalarize_channels:
        print(
            f">> Scalar mode active: forcing model geometry "
            f"C={C_eff}, H={H_eff}, W={W_eff}",
            flush=True,
        )
    else:
        if args.model_channels is not None and args.model_channels != C:
            print(
                f">> NOTE: dataset sample has C={C} but --model_channels={args.model_channels}. "
                f"Ensure your dataset pads/truncates to {args.model_channels}.",
                flush=True,
            )

        if args.res_h is not None and args.res_h != H:
            print(
                f">> NOTE: dataset sample has H={H} but --res_h={args.res_h}. "
                f"Ensure dataset outputs res_h.",
                flush=True,
            )

        if args.res_w is not None and args.res_w != W:
            print(
                f">> NOTE: dataset sample has W={W} but --res_w={args.res_w}. "
                f"Ensure dataset outputs res_w.",
                flush=True,
            )

    # Schedule steps
    steps_per_epoch = len(dl_tr)
    total_steps = steps_per_epoch * args.max_epochs

    if args.ch_mult.strip():
        ch_mult = [int(x.strip()) for x in args.ch_mult.split(",") if x.strip()]
    else:
        ch_mult = df_to_ch_mult(args.down_factor)

    # sanity: down_factor must match number of downsamples implied by ch_mult
    implied_df = 2 ** (len(ch_mult) - 1)

    if implied_df != int(args.down_factor):
        print(f">> NOTE: overriding down_factor {args.down_factor} -> {implied_df} to match ch_mult={ch_mult}", flush=True)
        args.down_factor = implied_df

    # enforce exact divisibility using effective dims
    if (H_eff % args.down_factor) != 0 or (W_eff % args.down_factor) != 0:
        raise ValueError(
            f"H,W must be divisible by down_factor={args.down_factor}. "
            f"Got H={H_eff}, W={W_eff}. (Set dataset res_h/res_w or choose a different down_factor.)"
        )

    latent_h = H_eff // args.down_factor
    latent_w = W_eff // args.down_factor

    # --- If resuming weights-only and ckpt lacks hyperparams, infer base_ch/z_channels from ckpt ---
    if args.resume_weights_only is not None:
        ckpt = torch.load(args.resume_weights_only, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)

        # base_ch from encoder.conv_in out channels
        if "encoder.conv_in.weight" in sd:
            inferred_base_ch = int(sd["encoder.conv_in.weight"].shape[0])
            if args.base_ch != inferred_base_ch:
                print(f">> Overriding --base_ch={args.base_ch} -> {inferred_base_ch} (from ckpt encoder.conv_in.weight)", flush=True)
                args.base_ch = inferred_base_ch

        # z_channels from quant_conv (in channels)
        if "quant_conv.weight" in sd:
            inferred_z = int(sd["quant_conv.weight"].shape[1])
            if args.embed_dim != int(sd["quant_conv.weight"].shape[0]):
                print(f">> WARNING: embed_dim arg {args.embed_dim} != ckpt quant_conv out {int(sd['quant_conv.weight'].shape[0])}", flush=True)
            # Note: in your script you tie z_channels=args.embed_dim; that is wrong for this ckpt.
            # ckpt expects z_channels=256 and embed_dim=256, so it's OK here.

    ddconfig = dict(
        double_z=False,
        z_channels=args.embed_dim,
        resolution=max(H_eff, W_eff),
        in_channels=C_eff,
        out_ch=C_eff,
        ch=args.base_ch,
        ch_mult=ch_mult,
        num_res_blocks=args.num_res_blocks,
        attn_resolutions=[min(latent_h, latent_w)],  # safe for rectangular
        dropout=args.dropout,
    )

    use_sched = (args.lr_schedule != "none")
    model = LitVQVAE(
        ddconfig=ddconfig,
        n_embed=args.n_embed,
        embed_dim=args.embed_dim,
        learning_rate=args.learning_rate,
        beta=args.beta,
        vq_loss_weight=args.vq_loss_weight,
        recon_l2_weight=0.0,
        grad_loss_weight=args.grad_loss_weight,
        l2_normalize_codebook=False,
        legacy_beta_bug=False,
        output_clamp=False,
        image_key="image",
        no_quant=args.no_quant,
        warmup_steps=args.warmup_steps if use_sched else 0,
        total_steps=total_steps if use_sched else 0,
        min_lr=args.min_lr,
        usage_entropy_weight=args.usage_entropy_weight,
        dead_code_reset_every=args.dead_code_reset_every,
        dead_code_reset_threshold=args.dead_code_reset_threshold,
        dead_code_reset_warmup_steps=args.dead_code_reset_warmup_steps,
        max_dead_code_resets=args.max_dead_code_resets,
    )

    if args.resume_weights_only is not None:
        ckpt = torch.load(args.resume_weights_only, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f">> Loaded weights-only from {args.resume_weights_only}", flush=True)
        print(f">> Missing keys: {len(missing)}  Unexpected keys: {len(unexpected)}", flush=True)

    tb = TensorBoardLogger(save_dir=os.path.join(args.out_dir, "tb"), name="", version="")

    resume_ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.out_dir, "checkpoints"),
        filename="resume-{epoch:04d}-{step:08d}",
        monitor="step",
        mode="max",
        save_top_k=3,
        save_last=True,
        every_n_train_steps=args.ckpt_every_n_train_steps,
        enable_version_counter=False,
    )

    best_ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.out_dir, "checkpoints"),
        filename="best-valreconl1",
        monitor="val/recon_l1",
        mode="min",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )

    lrm = LearningRateMonitor(logging_interval="step")
    code_cb = RunningCodeUsageCallback(every_steps=args.code_usage_every, window_steps=20)

    # Save PNG epochs (1-based -> 0-based)
    save_png_epochs = []
    if args.save_png:
        for p in str(args.save_png).split(","):
            p = p.strip()
            if not p:
                continue
            iv = int(p)
            if iv > 0:
                save_png_epochs.append(iv - 1)

    printer = PrintValMetrics()

    callbacks: List[Callback] = [resume_ckpt_cb, best_ckpt_cb, lrm, code_cb, printer]

    if save_png_epochs:
        callbacks.append(
            SaveReconCallback(
                val_dataloader=dl_va,
                epoch_list=save_png_epochs,
                out_dir=args.out_dir,
                max_channels=args.viz_channels,
                sample_idx=args.viz_sample_idx,
                viz_vmin=args.viz_vmin,
                viz_vmax=args.viz_vmax,
                err_vmax=args.viz_err_vmax,
            )
        )

    use_ddp = (int(args.devices) > 1) or (int(args.num_nodes) > 1)
    strategy = "ddp" if use_ddp else "auto"

    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        max_epochs=args.max_epochs,
        logger=tb,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        val_check_interval=args.val_check_interval,
        enable_progress_bar=not args.disable_pbar,
    )

    if args.dry_run:
        # pick a deterministic single sample from VAL dataset
        ds = dl_va.dataset
        idx = max(0, min(int(args.viz_sample_idx), len(ds) - 1))
        ex = ds[idx]  # dict like {"image": ...}
        batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in ex.items()}

        x = model.get_input(batch).to("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(x.device)

        with torch.no_grad():
            x_rec, qloss, info = model(x)
        inds = info[-1]
        print("DRY RUN OK",
              "x", tuple(x.shape),
              "x_rec", tuple(x_rec.shape),
              "qloss", float(qloss),
              "inds", (tuple(inds.shape) if inds is not None else None),
              flush=True)
        if args.dry_run_png:
            def to_color(x_hw, vmin=-1.0, vmax=1.0, cmap_name="viridis"):
                a = (x_hw - vmin) / (vmax - vmin + 1e-12)
                a = np.clip(a, 0.0, 1.0)
                cmap = cm.get_cmap(cmap_name)
                rgb = (cmap(a)[..., :3] * 255.0).astype(np.uint8)
                return Image.fromarray(rgb)

            gt = x[0,0].detach().cpu().numpy()
            rc = x_rec[0,0].detach().cpu().numpy()
            er = np.abs(gt-rc)

            gt_im = to_color(gt, vmin=args.viz_vmin, vmax=args.viz_vmax)
            rc_im = to_color(rc, vmin=args.viz_vmin, vmax=args.viz_vmax)

            # error as grayscale
            er_n = np.clip(er / (args.viz_err_vmax + 1e-12), 0.0, 1.0)
            er_im = Image.fromarray((er_n*255).astype(np.uint8)).convert("RGB")

            w,h = gt_im.size
            grid = Image.new("RGB", (w*3, h))
            grid.paste(gt_im, (0,0))
            grid.paste(rc_im, (w,0))
            grid.paste(er_im, (2*w,0))
            os.makedirs(os.path.dirname(args.dry_run_png) or ".", exist_ok=True)
            grid.save(args.dry_run_png)
            print(f">> Saved dry-run PNG: {args.dry_run_png}", flush=True)
        return

    if trainer.is_global_zero:
        print(">> resume_ckpt =", args.resume_ckpt, flush=True)

    trainer.fit(model, train_dataloaders=dl_tr, val_dataloaders=dl_va, ckpt_path=args.resume_ckpt)
    print(">> Training complete. Checkpoints:", os.path.join(args.out_dir, "checkpoints"), flush=True)
    print(">> Best val/recon_l1 checkpoint:", best_ckpt_cb.best_model_path, flush=True)
    print(">> Best val/recon_l1 value:", best_ckpt_cb.best_model_score, flush=True)


if __name__ == "__main__":
    main()
