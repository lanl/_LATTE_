#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot rollout RMSE vs timestep for one or more rollout folders, for *any dataset* where you have:
- Stage-A rollout outputs: *_pred.npz files (pred_tokens: (T,Hq,Wq), src_tokens_npz, optionally run_dir/time_start/time_count)
- GT real-valued frames stored as NPZs in --gt_frames_dir, with a known key (e.g. av_density)
- One or more VQ-VAE checkpoints (LitVQVAE) to decode tokens -> fields

Key features:
- Supports MULTIPLE rollout dirs and MULTIPLE vq ckpts:
    --rollout_dirs dirA dirB ...
    --vq_ckpts     vqA  vqB  ...   (either 1 ckpt reused for all, or one-per-rollout_dir)
- Robust GT path mapping:
    --gt_frames_dir /path/to/gt_npzs
    --gt_replace "test_=" "_tokens.npz=.npz"   (apply replacements to the basename of src_tokens_npz)
- Robust shape/orientation handling:
    If decoded frames are (H,W)=(200,560) but GT is (560,200), we transpose GT ONCE consistently.
- Computes RMSE in normalized [-1,1] space (matches your VQ training normalization).
- Optional sanity baselines:
    - "copy_prev" baseline: RMSE between GT[t] and GT[t-1]
    - "gt_dec_vs_gt" sanity: RMSE between decode(GT_tokens) and GT(real) if you provide tokens (not required)

Outputs:
- rmse_curves.npz
- rmse_plot.png
- viz_panel.png (optional single-trajectory visualization at chosen timestep)
"""

from __future__ import annotations

import os, json, argparse
from typing import Optional, Any, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.lit_vqvae import LitVQVAE


# =============================================================================
# Small utils
# =============================================================================

def _as_str(x) -> str:
    if isinstance(x, np.ndarray):
        x = x.item()
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    return str(x)

def _get_nested_attr(obj: Any, path: str):
    cur = obj
    for p in path.split("."):
        if cur is None or not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur

def match_hw(x, ref_hw: Tuple[int,int]):
    """
    Match the last two dims of x to ref_hw, transposing last two dims if swapped.
    Works for np.ndarray or torch.Tensor.
    """
    H_ref, W_ref = int(ref_hw[0]), int(ref_hw[1])
    H, W = int(x.shape[-2]), int(x.shape[-1])
    if (H, W) == (H_ref, W_ref):
        return x
    if (H, W) == (W_ref, H_ref):
        if isinstance(x, np.ndarray):
            return x.swapaxes(-2, -1)
        else:
            return x.transpose(-2, -1)
    raise RuntimeError(f"Cannot match HW: got {(H,W)} expected {ref_hw} (or swapped)")

def apply_replacements(name: str, repls: List[Tuple[str,str]]) -> str:
    out = name
    for a,b in repls:
        out = out.replace(a, b)
    return out


# =============================================================================
# GT loading + normalization
# =============================================================================

def load_gt_npz_as_tchw(path: str, key: Optional[str]) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    if key is None:
        # try common keys
        for k in ("frames", "gt", "images", "x", "data", "av_density"):
            if k in z:
                key = k
                break
    if key is None or key not in z:
        raise RuntimeError(f"{path} does not contain a GT array. Keys={list(z.keys())}. Provide --gt_key.")
    arr = np.asarray(z[key])

    # Normalize to (T,C,H,W)
    if arr.ndim == 3:
        # (T,H,W)
        arr = arr[:, None, :, :]
    elif arr.ndim == 4:
        # could be (T,H,W,C) or already (T,C,H,W)
        if arr.shape[-1] <= 16 and arr.shape[1] > 16 and arr.shape[2] > 16:
            arr = arr.transpose(0, 3, 1, 2)
        # else assume already (T,C,H,W)
    else:
        raise RuntimeError(f"Unsupported GT shape {arr.shape} in {path}")
    return arr.astype(np.float32)

def load_minmax_json(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r") as f:
        mm = json.load(f)

    if isinstance(mm, dict):
        if "min" in mm and "max" in mm:
            vmin, vmax = mm["min"], mm["max"]
        elif "ch_min" in mm and "ch_max" in mm:
            vmin, vmax = mm["ch_min"], mm["ch_max"]
        elif len(mm) == 1 and isinstance(next(iter(mm.values())), dict):
            mm2 = next(iter(mm.values()))
            if "min" in mm2 and "max" in mm2:
                vmin, vmax = mm2["min"], mm2["max"]
            elif "ch_min" in mm2 and "ch_max" in mm2:
                vmin, vmax = mm2["ch_min"], mm2["ch_max"]
            else:
                raise RuntimeError(f"Unrecognized nested minmax format in {path}. Keys={list(mm2.keys())}")
        else:
            raise RuntimeError(f"Unrecognized minmax json format in {path}. Keys={list(mm.keys())}")
    else:
        raise RuntimeError(f"minmax json must be a dict, got {type(mm)}")

    vmin = np.asarray(vmin, dtype=np.float32).reshape(-1)
    vmax = np.asarray(vmax, dtype=np.float32).reshape(-1)
    return vmin, vmax

def denormalize_from_m11(x_tchw: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    """
    Inverse of normalize_to_m11. Takes normalized [-1,1] -> original units using per-channel min/max.
    x_tchw: (T,C,H,W)
    """
    vmin = np.asarray(vmin, dtype=np.float32).reshape(-1)
    vmax = np.asarray(vmax, dtype=np.float32).reshape(-1)

    C = x_tchw.shape[1]
    if vmin.size == 1 and C > 1:
        vmin = np.repeat(vmin, C)
        vmax = np.repeat(vmax, C)

    if vmin.size != C or vmax.size != C:
        raise RuntimeError(f"min/max have C={vmin.size} but x has C={C}")

    den = (vmax - vmin)
    den = np.where(np.abs(den) < 1e-12, 1.0, den)

    x01 = (x_tchw + 1.0) * 0.5
    x = x01 * den[None, :, None, None] + vmin[None, :, None, None]
    return x.astype(np.float32)

def normalize_to_m11(x_tchw: np.ndarray, vmin: np.ndarray, vmax: np.ndarray, clamp: bool=True) -> Tuple[np.ndarray, float]:
    vmin = np.asarray(vmin, dtype=np.float32).reshape(-1)
    vmax = np.asarray(vmax, dtype=np.float32).reshape(-1)

    C = int(x_tchw.shape[1])
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
        x11c = np.clip(x11, -1.0, 1.0)
        clip_frac = float(np.mean(x11 != x11c))
        x11 = x11c
    else:
        clip_frac = 0.0
    return x11.astype(np.float32), clip_frac


# =============================================================================
# VQ decode (generic, works even if no decode_code method exists)
# =============================================================================

def _find_codebook_weight(vq: nn.Module) -> Optional[torch.Tensor]:
    candidates = [
        "quantize.embedding.weight",
        "model.quantize.embedding.weight",
        "quantizer.embedding.weight",
        "model.quantizer.embedding.weight",
        "vq.quantize.embedding.weight",
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

    raise RuntimeError("Could not find a decode path on LitVQVAE.")

@torch.inference_mode()
def decode_codes_batch_to_fields(vq: LitVQVAE, codes_bhw: torch.Tensor) -> torch.Tensor:
    """
    codes_bhw: (B,Hq,Wq) int64
    returns:   (B,C,H,W) float (typically [-1,1])
    """
    for fn_path in ("decode_code", "model.decode_code"):
        fn = _get_nested_attr(vq, fn_path)
        if callable(fn):
            out = fn(codes_bhw)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if getattr(vq, "output_clamp", False):
                out = out.clamp(-1, 1)
            return out

    w = _find_codebook_weight(vq)
    if w is None:
        raise RuntimeError("Could not locate codebook embedding weight on LitVQVAE.")
    B, Hq, Wq = codes_bhw.shape
    z = w[codes_bhw.reshape(-1)]
    z = z.view(B, Hq, Wq, -1).permute(0, 3, 1, 2).contiguous()
    x = _vq_decode_latents(vq, z)
    if getattr(vq, "output_clamp", False):
        x = x.clamp(-1, 1)
    return x


# =============================================================================
# VQ building/loading (supports both “hyper_parameters” style and fallback args)
# =============================================================================

def build_vq_from_ckpt_or_args(vq_ckpt: str, device: torch.device,
                              *,
                              fallback_res: int,
                              fallback_model_channels: int,
                              fallback_n_embed: int,
                              fallback_embed_dim: int,
                              use_ema_copy: bool = False) -> LitVQVAE:
    ck = torch.load(vq_ckpt, map_location="cpu")
    sd = ck.get("state_dict", ck)
    hparams = ck.get("hyper_parameters", {}) or {}

    # Preferred: hyper_parameters include ddconfig etc.
    if "ddconfig" in hparams and "n_embed" in hparams and "embed_dim" in hparams:
        ddconfig = hparams["ddconfig"]
        n_embed = int(hparams.get("n_embed"))
        embed_dim = int(hparams.get("embed_dim"))
        image_key = str(hparams.get("image_key", "image"))

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
    else:
        # Fallback: construct a known-good ddconfig for common LATTE VQ checkpoints.
        ch_mult = [1, 2, 2]
        downs = len(ch_mult) - 1
        latent_factor = 2 ** downs
        latent_h = int(fallback_res) // latent_factor

        ddconfig = dict(
            double_z=False,
            z_channels=int(fallback_embed_dim),
            resolution=int(fallback_res),
            in_channels=int(fallback_model_channels),
            out_ch=int(fallback_model_channels),
            ch=128,
            ch_mult=list(ch_mult),
            num_res_blocks=3,
            attn_resolutions=[int(latent_h)],
            dropout=0.0,
        )

        vq = LitVQVAE(
            ddconfig=ddconfig,
            n_embed=int(fallback_n_embed),
            embed_dim=int(fallback_embed_dim),
            learning_rate=2e-4,
            beta=0.1,
            vq_loss_weight=5.0,
            recon_l2_weight=0.0,
            grad_loss_weight=0.0,
            l2_normalize_codebook=False,
            legacy_beta_bug=False,
            output_clamp=True,
            image_key="image",
            no_quant=False,
            warmup_steps=0,
            total_steps=0,
            min_lr=1e-6,
        )

    missing, unexpected = vq.load_state_dict(sd, strict=False)
    print(f">> Loaded VQ: {vq_ckpt}\n   missing={len(missing)} unexpected={len(unexpected)}")

    # Optional: copy EMA weights if present as model_ema.* (best-effort)
    if use_ema_copy:
        model_keys = set(vq.state_dict().keys())
        ema_sd = {}
        for k, v in sd.items():
            if k.startswith("model_ema."):
                base_k = k.replace("model_ema.", "", 1)
                if base_k in model_keys:
                    ema_sd[base_k] = v
        if ema_sd:
            m2, u2 = vq.load_state_dict(ema_sd, strict=False)
            print(f">> VQ copied EMA->base: loaded={len(ema_sd)} missing={len(m2)} unexpected={len(u2)}")
        else:
            print(">> VQ EMA keys not found; using raw model weights.")

    return vq.to(device).eval()


# =============================================================================
# Rollout index loading
# =============================================================================

def load_rollout_index(rollout_dir: str) -> List[str]:
    idx = os.path.join(rollout_dir, "rollouts_index.json")
    if os.path.isfile(idx):
        with open(idx, "r") as f:
            j = json.load(f)
        return list(j["outputs"])
    return sorted([os.path.join(rollout_dir, x) for x in os.listdir(rollout_dir) if x.endswith("_pred.npz")])

def map_by_src(files: List[str]) -> Dict[str, str]:
    m = {}
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        src = _as_str(z["src_tokens_npz"]) if "src_tokens_npz" in z else fp
        m[src] = fp
    return m


# =============================================================================
# Visualization panel
# =============================================================================

def save_last_timestep_panel(
    out_path: str,
    gt_chw: np.ndarray,
    preds_kchw: np.ndarray,
    labels: List[str],
    channel_names: Optional[List[str]] = None,
    per_channel_scale: bool = True,
):
    """
    Saves a panel image with correct pixel aspect (no stretching).

    Row 0: GT
    Rows 1..K: predictions
    Cols 0..C-1: channels
    """
    assert gt_chw.ndim == 3
    C, H, W = gt_chw.shape
    K = preds_kchw.shape[0]
    assert preds_kchw.shape[1:] == (C, H, W), f"preds_kchw {preds_kchw.shape} != {(K,C,H,W)}"

    if channel_names is None:
        channel_names = [f"ch{c}" for c in range(C)]
    else:
        assert len(channel_names) == C

    # Color limits
    if per_channel_scale:
        vmins = np.zeros((C,), dtype=np.float32)
        vmaxs = np.zeros((C,), dtype=np.float32)
        for c in range(C):
            vals = np.concatenate([gt_chw[c].reshape(-1), preds_kchw[:, c].reshape(-1)], axis=0)
            vmins[c] = float(np.min(vals))
            vmaxs[c] = float(np.max(vals))
            if vmaxs[c] - vmins[c] < 1e-6:
                vmaxs[c] = vmins[c] + 1e-6
    else:
        vmins = np.full((C,), -1.0, dtype=np.float32)
        vmaxs = np.full((C,),  1.0, dtype=np.float32)

    # Layout
    nrows = 1 + K
    ncols = C

    # --- critical: size the figure to preserve H/W ---
    # base "height per image row" in inches:
    row_h = 2.6
    # width per image column should scale with W/H:
    col_w = row_h * (W / max(1, H))

    # add extra room for left-side row labels
    left_margin_in = 1.8
    fig_w = left_margin_in + col_w * ncols
    fig_h = row_h * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    row_names = ["GT"] + labels

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])

            img = gt_chw[c] if r == 0 else preds_kchw[r - 1, c]
            ax.imshow(img, vmin=vmins[c], vmax=vmaxs[c], interpolation="nearest", origin="lower")

            # --- critical: preserve pixel aspect ---
            ax.set_aspect("equal", adjustable="box")

            if r == 0:
                ax.set_title(channel_names[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(row_names[r], rotation=0, labelpad=40, va="center", fontsize=10)

            for s in ax.spines.values():
                s.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--rollout_dirs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--vq_ckpts", nargs="+", required=True,
                    help="Either 1 ckpt (reused for all rollout_dirs) or one-per-rollout_dir.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vq_decode_bs", type=int, default=16)
    ap.add_argument("--use_vq_ema_copy", action="store_true")

    # Fallback VQ construction params (only used if ckpt lacks hyper_parameters['ddconfig'])
    ap.add_argument("--fallback_res", type=int, default=560)
    ap.add_argument("--fallback_model_channels", type=int, default=1)
    ap.add_argument("--fallback_n_embed", type=int, default=2048)
    ap.add_argument("--fallback_embed_dim", type=int, default=256)

    # GT loading
    ap.add_argument("--gt_frames_dir", required=True,
                    help="Directory containing GT NPZs (real-valued) for each trajectory.")
    ap.add_argument("--gt_key", default=None, help="Key inside GT NPZ (e.g. av_density).")
    ap.add_argument("--minmax_json", default=None,
                    help="Min/max JSON used during VQ training to normalize GT to [-1,1]. If omitted, assumes GT already normalized.")
    ap.add_argument("--gt_already_normalized", action="store_true")

    # Name mapping from src_tokens_npz basename -> GT basename
    ap.add_argument("--gt_replace", nargs="*", default=[],
                    help='Replacements applied to src_tokens_npz basename, each as "FROM=TO". '
                         'Example: --gt_replace "test_=" "_tokens.npz=.npz"')

    # Viz
    ap.add_argument("--viz_index", type=int, default=0)
    ap.add_argument("--viz_timestep", type=int, default=-1)
    ap.add_argument("--channel_names", default=None)
    ap.add_argument("--out_rollout_npz", type=str, default=None,
                help="Where to save full decoded rollout (DENORMALIZED) as .npz. "
                     "If not set, derives from --out_png by replacing .png with _rollout_preds_unnorm.npz")

    # Sanity
    ap.add_argument("--add_copy_prev_baseline", action="store_true",
                    help="Also plot GT copy-prev baseline RMSE (GT[t] vs GT[t-1]) as a dashed line.")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print(">> device:", device)

    # labels
    labels = args.labels
    if not labels or len(labels) != len(args.rollout_dirs):
        labels = [os.path.basename(d.rstrip("/")) for d in args.rollout_dirs]

    # parse replacements
    repls: List[Tuple[str,str]] = []
    for item in args.gt_replace:
        if "=" not in item:
            raise RuntimeError(f"--gt_replace item must be 'FROM=TO', got: {item}")
        a,b = item.split("=", 1)
        repls.append((a,b))

    # Load rollout indices
    dir_files = []
    for d in args.rollout_dirs:
        files = load_rollout_index(d)
        if not files:
            raise RuntimeError(f"No rollout files found in {d}")
        dir_files.append(files)
        print(f">> {d}: {len(files)} runs")

    maps = [map_by_src(files) for files in dir_files]
    common_src = sorted(set(maps[0].keys()).intersection(*[set(m.keys()) for m in maps[1:]]))
    if not common_src:
        raise RuntimeError("No common runs across rollout_dirs (src_tokens_npz mismatch).")
    print(f">> Common runs across all dirs: {len(common_src)}")

    # Inspect first rollout for token shape
    z0 = np.load(maps[0][common_src[0]], allow_pickle=True)
    pred0 = np.asarray(z0["pred_tokens"])
    if pred0.ndim != 3:
        raise RuntimeError(f"pred_tokens must be (T,Hq,Wq), got {pred0.shape}")
    T, Hq, Wq = pred0.shape
    print(f">> Pred token shape: T={T}, Hq={Hq}, Wq={Wq}")

    # Load minmax (optional)
    vmin = vmax = None
    if (args.minmax_json is not None) and (not args.gt_already_normalized):
        vmin, vmax = load_minmax_json(args.minmax_json)
        print(f">> Loaded minmax: {args.minmax_json}")

    # VQ ckpt assignment
    if len(args.vq_ckpts) == 1:
        vq_for_dir = [args.vq_ckpts[0] for _ in args.rollout_dirs]
    elif len(args.vq_ckpts) == len(args.rollout_dirs):
        vq_for_dir = list(args.vq_ckpts)
    else:
        raise RuntimeError("--vq_ckpts must have length 1 or match --rollout_dirs length.")

    # Build VQs (one per unique ckpt)
    vq_cache: Dict[str, LitVQVAE] = {}
    for ckpt in sorted(set(vq_for_dir)):
        vq_cache[ckpt] = build_vq_from_ckpt_or_args(
            ckpt, device=device,
            fallback_res=args.fallback_res,
            fallback_model_channels=args.fallback_model_channels,
            fallback_n_embed=args.fallback_n_embed,
            fallback_embed_dim=args.fallback_embed_dim,
            use_ema_copy=bool(args.use_vq_ema_copy),
        )

    # Choose viz trajectory
    viz_i = max(0, min(int(args.viz_index), len(common_src)-1))
    viz_src = common_src[viz_i]
    viz_t = int(args.viz_timestep)
    if viz_t < 0:
        viz_t = T + viz_t
    if not (0 <= viz_t < T):
        raise RuntimeError(f"--viz_timestep out of range: resolved to {viz_t}, but T={T}")

    # Storage
    K = len(args.rollout_dirs)
    N = len(common_src)
    rmse_runs = np.zeros((K, N, T), dtype=np.float64)
    copy_prev_rmse = np.zeros((N, T), dtype=np.float64) if args.add_copy_prev_baseline else None

    viz_gt_chw = None
    viz_preds_kchw = None
    viz_pred_rollout_norm = None   # (K,T,C,H,W) in [-1,1]
    viz_gt_rollout_norm = None     # (T,C,H,W) in [-1,1]
    viz_gt_path = None
    viz_hdwd = None                # (H,W)

    # Main loop over trajectories
    for i, src in enumerate(common_src):
        # Determine GT filename from src_tokens_npz basename
        src_base = os.path.basename(src)
        gt_base = apply_replacements(src_base, repls)
        gt_path = os.path.join(args.gt_frames_dir, gt_base)
        if i == viz_i:
            viz_gt_path = gt_path
        if not os.path.isfile(gt_path):
            raise RuntimeError(
                f"GT file not found for src={src_base}\n"
                f"  expected: {gt_path}\n"
                f"Fix mapping with --gt_replace, or put GT in --gt_frames_dir."
            )

        # Load GT (T,C,H,W)
        gt = load_gt_npz_as_tchw(gt_path, key=args.gt_key)

        if gt.shape[0] != T:
            raise RuntimeError(f"GT T={gt.shape[0]} but pred T={T} for {gt_path}")

        # Normalize GT to [-1,1] if requested
        if (vmin is not None) and (vmax is not None) and (not args.gt_already_normalized):
            gt, clip_frac = normalize_to_m11(gt, vmin, vmax, clamp=True)
            if i == 0:
                print(f">> GT normalized: {gt.shape} clip_frac: {clip_frac}")

        # ---- enforce denormalized rollout NPZ requirements (only if saving) ----
        if args.out_rollout_npz is not None:
            if args.gt_already_normalized:
                raise RuntimeError(
                    "Requested --out_rollout_npz, but --gt_already_normalized was set.\n"
                    "For DENORMALIZED output, remove --gt_already_normalized and provide --minmax_json."
                )
            if args.minmax_json is None:
                raise RuntimeError(
                    "Requested --out_rollout_npz, but --minmax_json was not provided.\n"
                    "For DENORMALIZED output, provide --minmax_json so we can invert normalization."
                )

        # Baseline: copy previous frame RMSE in [-1,1]
        if copy_prev_rmse is not None:
            # define copy_prev[0] = 0
            copy_prev_rmse[i, 0] = 0.0
            dif = (gt[1:] - gt[:-1]).astype(np.float32)
            mse = np.mean(dif * dif, axis=(1,2,3))
            copy_prev_rmse[i, 1:] = np.sqrt(mse + 1e-12)

        # Prepare viz buffers
        if i == viz_i:
            # we will fill after we know decoded HW
            viz_preds_kchw = np.zeros((K, gt.shape[1], 1, 1), dtype=np.float32)  # resized later

        # For each rollout dir/model
        for k in range(K):
            rollout_fp = maps[k][src]
            z = np.load(rollout_fp, allow_pickle=True)
            pred_tokens = np.asarray(z["pred_tokens"]).astype(np.int64)
            if pred_tokens.shape != (T, Hq, Wq):
                raise RuntimeError(f"{rollout_fp}: pred_tokens shape {pred_tokens.shape} != {(T,Hq,Wq)}")

            vq_ckpt = vq_for_dir[k]
            vq = vq_cache[vq_ckpt]

            bs = max(1, int(args.vq_decode_bs))
            rmse_t = np.zeros((T,), dtype=np.float64)

            with torch.inference_mode():
                # decode one small batch first to establish decoded HW
                s0, e0 = 0, min(bs, T)
                codes0 = torch.from_numpy(pred_tokens[s0:e0]).to(device, non_blocking=True)
                pred0_x = decode_codes_batch_to_fields(vq, codes0).float()  # (B,C,Hd,Wd)
                Hd, Wd = int(pred0_x.shape[-2]), int(pred0_x.shape[-1])

                # Make a GT torch tensor on device, matched HW
                gt_t = torch.from_numpy(gt).to(device, non_blocking=True).float()  # (T,C,H,W)
                gt_t = match_hw(gt_t, (Hd, Wd))

                # Capture viz GT rollout in normalized space after HW matching
                if i == viz_i and k == 0:
                    viz_gt_rollout_norm = gt_t.detach().cpu().numpy().astype(np.float32)  # (T,C,Hd,Wd)
                    viz_hdwd = (Hd, Wd)

                # If viz traj, build viz_gt_chw (matched HW)
                if i == viz_i and k == 0:
                    viz_gt = gt_t[viz_t].detach().cpu().numpy()  # (C,Hd,Wd)
                    viz_gt_chw = viz_gt
                    viz_preds_kchw = np.zeros((K, viz_gt_chw.shape[0], viz_gt_chw.shape[1], viz_gt_chw.shape[2]), dtype=np.float32)

                # Decode + RMSE in batches
                for s in range(0, T, bs):
                    e = min(T, s + bs)
                    codes_bhw = torch.from_numpy(pred_tokens[s:e]).to(device, non_blocking=True)
                    pred_x = decode_codes_batch_to_fields(vq, codes_bhw).float()  # (B,C,Hd,Wd)

                    # Capture full viz predicted rollout (normalized)
                    if i == viz_i:
                        if viz_pred_rollout_norm is None:
                            # allocate once, after we know (C,Hd,Wd)
                            C = int(pred_x.shape[1])
                            viz_pred_rollout_norm = np.zeros((K, T, C, Hd, Wd), dtype=np.float32)
                        viz_pred_rollout_norm[k, s:e] = pred_x.detach().cpu().numpy().astype(np.float32)
                    gt_x = gt_t[s:e]  # already matched
                    mse = torch.mean((pred_x - gt_x) ** 2, dim=(1,2,3))
                    rmse = torch.sqrt(mse + 1e-12)
                    rmse_t[s:e] = rmse.detach().cpu().numpy().astype(np.float64)

                # Viz decode at viz_t
                if i == viz_i:
                    codes_1 = torch.from_numpy(pred_tokens[viz_t:viz_t+1]).to(device, non_blocking=True)
                    pred_1 = decode_codes_batch_to_fields(vq, codes_1).float()[0]  # (C,Hd,Wd)
                    viz_preds_kchw[k] = pred_1.detach().cpu().numpy()

            rmse_runs[k, i] = rmse_t

        if (i % 10) == 0 or (i == N-1):
            print(f">> [{i+1:03d}/{N}] done: {src_base}", flush=True)

    # ---- save denormalized full decoded rollout for viz trajectory (if requested) ----
    if args.out_rollout_npz is not None:
        if viz_pred_rollout_norm is None or viz_gt_rollout_norm is None:
            raise RuntimeError("Internal error: viz rollout buffers not populated.")

        # Denormalize using minmax_json (required by enforcement)
        vmin2, vmax2 = load_minmax_json(args.minmax_json)

        # preds: (K,T,C,H,W) -> denorm
        K2, T2, C2, H2, W2 = viz_pred_rollout_norm.shape
        preds_ktchw = viz_pred_rollout_norm.reshape(K2 * T2, C2, H2, W2)
        preds_ktchw_unnorm = denormalize_from_m11(preds_ktchw, vmin2, vmax2).reshape(K2, T2, C2, H2, W2)

        # gt: (T,C,H,W) -> denorm
        gt_tchw_unnorm = denormalize_from_m11(viz_gt_rollout_norm, vmin2, vmax2)

        out_roll = args.out_rollout_npz
        os.makedirs(os.path.dirname(out_roll) or ".", exist_ok=True)

        np.savez_compressed(
            out_roll,
            labels=np.asarray(labels),
            rollout_dirs=np.asarray(args.rollout_dirs),
            vq_ckpts=np.asarray(vq_for_dir),
            viz_index=np.int64(viz_i),
            viz_src=np.asarray(viz_src),
            gt_path=np.asarray(viz_gt_path),
            pred_rollout=preds_ktchw_unnorm.astype(np.float32),  # (K,T,C,H,W) ORIGINAL units
            gt_rollout=gt_tchw_unnorm.astype(np.float32),        # (T,C,H,W) ORIGINAL units
            minmax_json=np.asarray(args.minmax_json),
            hw=np.asarray([H2, W2], dtype=np.int64),
        )
        print(f">> Saved DENORMALIZED rollout NPZ (viz trajectory): {out_roll}", flush=True)
        print(f">> GT path used for that NPZ: {viz_gt_path}", flush=True)
        
    # Aggregate means
    rmse_mean = rmse_runs.mean(axis=1)  # (K,T)
    copy_prev_mean = copy_prev_rmse.mean(axis=0) if copy_prev_rmse is not None else None

    # Save npz
    out_npz = os.path.join(args.out_dir, "rmse_curves.npz")
    np.savez(
        out_npz,
        labels=np.asarray(labels),
        rollout_dirs=np.asarray(args.rollout_dirs),
        vq_ckpts=np.asarray(vq_for_dir),
        common_src=np.asarray(common_src),
        rmse_runs=rmse_runs,
        rmse_mean=rmse_mean,
        copy_prev_mean=copy_prev_mean,
        T=T, Hq=Hq, Wq=Wq,
        gt_frames_dir=args.gt_frames_dir,
        gt_key=args.gt_key,
        minmax_json=args.minmax_json,
        gt_replace=np.asarray(args.gt_replace, dtype=object),
    )
    print(f">> Saved: {out_npz}")

    # Plot
    xs = np.arange(0, T, dtype=np.int64)
    plt.figure(figsize=(9, 4.8))
    for k in range(K):
        plt.plot(xs, rmse_mean[k], label=labels[k])
    if copy_prev_mean is not None:
        plt.plot(xs, copy_prev_mean, linestyle="--", label="GT copy_prev baseline")

    plt.xlabel("Timestep")
    plt.ylabel("RMSE vs GT in [-1,1] space")
    plt.title("Average rollout RMSE (normalized) over test trajectories")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png = os.path.join(args.out_dir, "rmse_plot.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f">> Saved plot: {out_png}")

    # Viz panel
    if (viz_gt_chw is not None) and (viz_preds_kchw is not None):
        if args.channel_names is None:
            ch_names = [f"ch{c}" for c in range(viz_gt_chw.shape[0])]
        else:
            ch_names = [s.strip() for s in args.channel_names.split(",")]
        out_panel = os.path.join(args.out_dir, f"viz_traj{viz_i:04d}_t{viz_t:03d}.png")
        save_last_timestep_panel(
            out_panel,
            gt_chw=viz_gt_chw,
            preds_kchw=viz_preds_kchw,
            labels=labels,
            channel_names=ch_names,
            per_channel_scale=True,
        )
        print(f">> Saved viz panel: {out_panel}")

if __name__ == "__main__":
    main()
