#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json, argparse, time
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.utils.data as tud
from contextlib import nullcontext

from src.models.lit_vqvae import LitVQVAE
from src.data.registry import build_dataset_from_registry


# -----------------------------
# Helpers
# -----------------------------

# -----------------------------
# VQ decode helpers (robust)
# -----------------------------
def _get_nested_attr(obj, path: str):
    cur = obj
    for p in path.split("."):
        if cur is None or not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur

def _find_codebook_weight(vq: torch.nn.Module) -> Optional[torch.Tensor]:
    # Common in your repo: quantize.embedding.weight
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
    # fallback: embedding module
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

def _vq_decode_latents(vq: torch.nn.Module, z_bchw: torch.Tensor) -> torch.Tensor:
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

@torch.inference_mode()
def decode_codes_to_x_m11(vq: LitVQVAE, inds_bhw: torch.Tensor) -> torch.Tensor:
    """
    inds_bhw: (B,Hq,Wq) int64/long
    returns:  (B,C,H,W) float in VQ output space (typically [-1,1])
    """
    device = next(vq.parameters()).device
    inds_bhw = inds_bhw.to(device=device, dtype=torch.long)

    # If model exposes decode_code, prefer it
    for fn_path in ("decode_code", "model.decode_code"):
        fn = _get_nested_attr(vq, fn_path)
        if callable(fn):
            out = fn(inds_bhw)
            if isinstance(out, (tuple, list)):
                out = out[0]
            x = out
            if getattr(vq, "output_clamp", False):
                x = x.clamp(-1, 1)
            return x

    w = _find_codebook_weight(vq)
    if w is None:
        raise RuntimeError("Could not locate VQ codebook embedding weight for manual decode.")

    B, Hq, Wq = inds_bhw.shape
    z = w[inds_bhw.reshape(-1)]                          # (B*Hq*Wq, embed_dim)
    z = z.view(B, Hq, Wq, -1).permute(0, 3, 1, 2)        # (B, embed_dim, Hq, Wq)
    x = _vq_decode_latents(vq, z)
    if getattr(vq, "output_clamp", False):
        x = x.clamp(-1, 1)
    return x


def parse_overrides(s: Optional[str]) -> Dict[str, int]:
    """
    "CE-CRP=1000,NS-Gauss=5000" -> {"CE-CRP":1000, "NS-Gauss":5000}
    """
    if s is None or str(s).strip() == "":
        return {}
    out: Dict[str, int] = {}
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad override '{p}'. Expected NAME=N.")
        k, v = p.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def infer_codebook_from_state_dict(sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """
    Robustly infer (n_embed, embed_dim) from the checkpoint state_dict.
    Works even if hyperparams are not saved.
    """
    candidates = [
        "quantize.embedding.weight",   # VectorQuantizer2 common
        "quantize.embed.weight",       # some variants
        "quantize.embedding",          # occasionally
    ]
    for k in candidates:
        if k in sd and hasattr(sd[k], "shape") and len(sd[k].shape) == 2:
            w = sd[k]
            return int(w.shape[0]), int(w.shape[1])

    # Fallback: search for any param name that ends with "embedding.weight"
    for k, v in sd.items():
        if str(k).endswith("embedding.weight") and hasattr(v, "shape") and len(v.shape) == 2:
            return int(v.shape[0]), int(v.shape[1])

    raise RuntimeError("Could not infer (n_embed, embed_dim) from checkpoint state_dict.")


def copy_ema_to_base_if_present(model: torch.nn.Module, state_dict: dict):
    """
    Optional: if ckpt has model_ema.* keys, copy them into base model weights.
    """
    model_keys = set(model.state_dict().keys())
    ema_sd = {}
    for k, v in state_dict.items():
        if k.startswith("model_ema."):
            base_k = k.replace("model_ema.", "", 1)
            if base_k in model_keys:
                ema_sd[base_k] = v
    if not ema_sd:
        print(">> EMA keys not found/mappable; using raw model weights.", flush=True)
        return
    model.load_state_dict(ema_sd, strict=False)
    print(f">> Copied EMA->base. loaded={len(ema_sd)}", flush=True)


def load_vqvae_from_ckpt(ckpt_path: str, ddconfig_fallback: dict, use_ema_copy: bool = False) -> LitVQVAE:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    hp = ckpt.get("hyper_parameters", {})

    # Prefer ddconfig saved in ckpt; else use fallback built from dataset+flags
    ddconfig = hp.get("ddconfig", None)
    if ddconfig is None:
        ddconfig = ddconfig_fallback

    # Infer codebook from weights if not present
    n_embed, embed_dim = infer_codebook_from_state_dict(sd)
    n_embed = int(hp.get("n_embed", n_embed))
    embed_dim = int(hp.get("embed_dim", embed_dim))

    model = LitVQVAE(
        ddconfig=ddconfig,
        n_embed=n_embed,
        embed_dim=embed_dim,
        learning_rate=float(hp.get("learning_rate", 2e-4)),
        beta=float(hp.get("beta", 0.1)),
        vq_loss_weight=float(hp.get("vq_loss_weight", 5.0)),
        recon_l2_weight=float(hp.get("recon_l2_weight", 0.0)),
        grad_loss_weight=float(hp.get("grad_loss_weight", 0.0)),
        l2_normalize_codebook=bool(hp.get("l2_normalize_codebook", False)),
        legacy_beta_bug=bool(hp.get("legacy_beta_bug", False)),
        output_clamp=bool(hp.get("output_clamp", True)),
        image_key=str(hp.get("image_key", "image")),
        no_quant=False,        # MUST be False for token export
        warmup_steps=0,
        total_steps=0,
        min_lr=float(hp.get("min_lr", 1e-6)),
    )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f">> loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    if use_ema_copy:
        copy_ema_to_base_if_present(model, sd)

    model.eval()
    return model


def batch_image_to_bchw(image: torch.Tensor) -> torch.Tensor:
    """
    Accepts batched image as either:
      [B,H,W,C] or [B,C,H,W]

    Returns:
      [B,C,H,W]
    """
    if image.dim() != 4:
        raise RuntimeError(f"Expected batched 4D image tensor, got {tuple(image.shape)}")

    b, a, c_or_h, d = image.shape

    # BCHW if second dim looks like channel count.
    if a <= 64 and c_or_h >= 16 and d >= 16:
        return image.contiguous().float()

    # Otherwise assume BHWC.
    return image.permute(0, 3, 1, 2).contiguous().float()


def scalarize_bchw(x_bchw: torch.Tensor) -> Tuple[torch.Tensor, int, int, int, int]:
    """
    x_bchw: [B,C,H,W]

    Returns:
      x_scalar: [B*C,1,H,W]
      B,C,H,W
    """
    B, C, H, W = x_bchw.shape
    x_scalar = x_bchw.reshape(B * C, 1, H, W).contiguous()
    return x_scalar, B, C, H, W


def infer_model_in_channels_from_ckpt(ckpt_path: str) -> Optional[int]:
    """
    Best-effort inference of model input channels from checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    hp = ckpt.get("hyper_parameters", {})

    ddconfig = hp.get("ddconfig", None)
    if isinstance(ddconfig, dict) and "in_channels" in ddconfig:
        return int(ddconfig["in_channels"])

    # Encoder input conv usually has shape [ch, in_channels, k, k]
    for k in ["encoder.conv_in.weight", "model.encoder.conv_in.weight"]:
        if k in sd and hasattr(sd[k], "shape") and len(sd[k].shape) == 4:
            return int(sd[k].shape[1])

    return None


def resolve_export_mode(requested_mode: str, model_in_channels: int, dataset_channels: int) -> str:
    """
    Returns one of:
      scalar_channels
      native
    """
    if requested_mode == "scalar_channels":
        if model_in_channels != 1:
            raise ValueError(
                f"--export_mode scalar_channels requires model_in_channels=1, "
                f"but checkpoint appears to have model_in_channels={model_in_channels}"
            )
        return "scalar_channels"

    if requested_mode == "native":
        if model_in_channels != dataset_channels:
            raise ValueError(
                f"--export_mode native requires model_in_channels == dataset_channels, "
                f"but got model_in_channels={model_in_channels}, dataset_channels={dataset_channels}"
            )
        return "native"

    if requested_mode != "auto":
        raise ValueError(f"Unknown export_mode={requested_mode}")

    # Auto mode.
    if model_in_channels == 1:
        return "scalar_channels"

    if model_in_channels == dataset_channels:
        return "native"

    raise ValueError(
        f"Could not auto-resolve export mode: model_in_channels={model_in_channels}, "
        f"dataset_channels={dataset_channels}. Use --export_mode explicitly or check model/dataset mismatch."
    )

@torch.inference_mode()
def encode_batch_to_indices(
    model: LitVQVAE,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    *,
    export_mode: str,
    do_roundtrip_check: bool = False,
):
    """
    export_mode:
      scalar_channels:
        input image [B,H,W,C] or [B,C,H,W]
        encode each channel independently using single-channel VQ-VAE
        returns tokens [B,C,Hq,Wq]

      native:
        input image is passed through model.get_input(batch)
        returns tokens [B,Hq,Wq]

    Returns:
      inds_cpu_int16
      Hq, Wq
      x_cpu_fp16
      rt_match
    """
    if export_mode == "scalar_channels":
        image = batch["image"]
        if not torch.is_tensor(image):
            raise RuntimeError("batch['image'] must be a torch.Tensor")

        x_bchw = batch_image_to_bchw(image).clamp(-1.0, 1.0)
        x_scalar, B, C, H, W = scalarize_bchw(x_bchw)

        x = x_scalar.to(device, non_blocking=True)
        x_cpu = x_bchw.detach().to("cpu", dtype=torch.float16)

        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)

        # Use fp32 for deterministic/stable export.
        z, _, info = model.encode(x.float())

        inds = info[-1]
        if inds is None:
            raise RuntimeError("model.encode() returned no indices. Is no_quant=False?")

        inds = torch.as_tensor(inds, device=z.device, dtype=torch.long)

        Hq = int(z.shape[-2])
        Wq = int(z.shape[-1])

        expected = B * C * Hq * Wq
        if inds.numel() != expected:
            raise RuntimeError(
                f"Unexpected inds size: inds.numel()={inds.numel()} expected={expected} "
                f"(B={B}, C={C}, Hq={Hq}, Wq={Wq})"
            )

        inds = inds.view(B, C, Hq, Wq)

        rt_match = None
        if do_roundtrip_check:
            print(
                f">> encode debug scalar_channels: x_scalar={tuple(x.shape)} "
                f"z={tuple(z.shape)} tokens={tuple(inds.shape)}",
                flush=True,
            )

            # Check first frame, first channel.
            inds0 = inds[0, 0].unsqueeze(0).to(device=device, dtype=torch.long)
            x0 = x_bchw[0, 0].unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)

            xrec0 = decode_codes_to_x_m11(model, inds0).to(dtype=torch.float32)

            z2, _, info2 = model.encode(xrec0.float())
            inds2 = torch.as_tensor(info2[-1], device=device, dtype=torch.long).view(1, Hq, Wq)

            rt_match = (inds2 == inds0).float().mean().item()

            mse = torch.mean((xrec0 - x0) ** 2).item()
            maxabs = torch.max(torch.abs(xrec0 - x0)).item()

            print(f">> ROUNDTRIP token match first sample/channel: {rt_match:.6f}", flush=True)
            print(f">> DEBUG decode(tokens[0,0]) vs x[0,0]: MSE={mse:.6e} maxabs={maxabs:.6e}", flush=True)

        return inds.to("cpu", dtype=torch.int16), Hq, Wq, x_cpu, rt_match

    if export_mode == "native":
        x = model.get_input(batch).to(device, non_blocking=True)
        x_cpu = x.detach().to("cpu", dtype=torch.float16)

        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)

        z, _, info = model.encode(x.float())

        inds = info[-1]
        if inds is None:
            raise RuntimeError("model.encode() returned no indices. Is no_quant=False?")

        inds = torch.as_tensor(inds, device=z.device, dtype=torch.long)

        B = int(x.shape[0])
        Hq = int(z.shape[-2])
        Wq = int(z.shape[-1])

        expected = B * Hq * Wq
        if inds.numel() != expected:
            raise RuntimeError(
                f"Unexpected inds size: inds.numel()={inds.numel()} expected={expected} "
                f"(B={B}, Hq={Hq}, Wq={Wq})"
            )

        inds = inds.view(B, Hq, Wq)

        rt_match = None
        if do_roundtrip_check:
            print(
                f">> encode debug native: x={tuple(x.shape)} z={tuple(z.shape)} "
                f"tokens={tuple(inds.shape)}",
                flush=True,
            )

            inds0 = inds[0].unsqueeze(0).to(device=device, dtype=torch.long)
            x0 = x[0:1].to(device=device, dtype=torch.float32)

            xrec0 = decode_codes_to_x_m11(model, inds0).to(dtype=torch.float32)

            z2, _, info2 = model.encode(xrec0.float())
            inds2 = torch.as_tensor(info2[-1], device=device, dtype=torch.long).view(1, Hq, Wq)

            rt_match = (inds2 == inds0).float().mean().item()

            mse = torch.mean((xrec0 - x0) ** 2).item()
            maxabs = torch.max(torch.abs(xrec0 - x0)).item()

            print(f">> ROUNDTRIP token match first sample: {rt_match:.6f}", flush=True)
            print(f">> DEBUG decode(tokens[0]) vs x[0]: MSE={mse:.6e} maxabs={maxabs:.6e}", flush=True)

        return inds.to("cpu", dtype=torch.int16), Hq, Wq, x_cpu, rt_match

    raise ValueError(f"Unsupported export_mode={export_mode}")

def try_get_traj_ids(ds) -> Optional[List[str]]:
    """
    Best-effort: if dataset has a list of source files, derive nice IDs.
    Otherwise return None and we’ll use traj indices.
    """
    if hasattr(ds, "files") and isinstance(ds.files, list) and len(ds.files) > 0:
        ids = []
        for fp in ds.files:
            b = os.path.basename(str(fp))
            ids.append(os.path.splitext(b)[0])
        return ids
    return None

def df_to_ch_mult(df: int):
    if df == 4:   return [1, 2, 2]          # /4
    if df == 8:   return [1, 2, 2, 2]       # /8
    if df == 16:  return [1, 2, 2, 2, 2]    # /16
    raise ValueError(f"Unsupported down_factor={df}")

def _infer_chw(x: torch.Tensor):
    # x is a single example: HWC or CHW
    if x.dim() != 3:
        raise RuntimeError(f"Expected 3D tensor but got {tuple(x.shape)}")
    a, b, c = x.shape
    if a <= 64 and b >= 16 and c >= 16:
        return int(a), int(b), int(c)  # CHW
    return int(c), int(a), int(b)      # HWC

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # VQ-VAE checkpoint
    ap.add_argument("--ckpt", default="${LATTE_WORK_ROOT}/vqvae_example/checkpoints/last.ckpt")
    ap.add_argument("--use_ema_copy", action="store_true", default=False)

    # Dataset registry selection
    ap.add_argument("--datasets_json", default='configs/datasets_poseidon_pdegym_v2.json')
    ap.add_argument("--dataset", default="ExampleHDF5Fields", help="Dataset name inside registry JSON.")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])

    # Output
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--one_file", action="store_true",
                    help="If set, write a single tokens npz for the whole split (instead of per-trajectory).")

    # Optional overrides pushed into dataset kwargs
    ap.add_argument("--time_start", type=int, default=None)
    ap.add_argument("--time_count", type=int, default=None)
    ap.add_argument("--res_h", type=int, default=None)
    ap.add_argument("--res_w", type=int, default=None)
    ap.add_argument("--model_channels", type=int, default=None)
    ap.add_argument("--traj_count", type=int, default=None)  # clamp, if dataset supports it

    # Performance
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument(
        "--export_mode",
        type=str,
        default="auto",
        choices=["auto", "scalar_channels", "native"],
        help=(
            "Token export mode. "
            "'scalar_channels' encodes each physical channel independently with a single-channel VQ-VAE. "
            "'native' encodes the full dataset image using a multichannel VQ-VAE. "
            "'auto' chooses based on checkpoint input channels."
        ),
    )

    # Limits
    ap.add_argument("--limit_traj", type=int, default=None,
                    help="Export only first N trajectories (after any traj_start already in split).")
    ap.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Start frame index within the frame-level dataset for one_file export.",
    )
    ap.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to export in one_file mode.",
    )
    ap.add_argument(
        "--out_name_suffix",
        type=str,
        default="",
        help="Suffix inserted before _tokens.npz in one_file mode, e.g. _shard000.",
    )


    ap.add_argument("--down_factor", type=int, default=8, choices=[4, 8, 16],
                help="Must match training down_factor.")
    ap.add_argument("--base_ch", type=int, default=128,
                    help="Must match training base_ch (ddconfig['ch']).")
    ap.add_argument("--num_res_blocks", type=int, default=3,
                    help="Must match training num_res_blocks.")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="Must match training dropout.")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- load dataset registry
    with open(args.datasets_json, "r") as f:
        reg = json.load(f)

    extra: Dict[str, Any] = {}
    for k in ["time_start", "time_count", "res_h", "res_w", "model_channels", "traj_count"]:
        v = getattr(args, k, None)
        if v is not None:
            extra[k] = v
    if args.traj_count is not None:
        extra["traj_count"] = int(args.traj_count)

    # ----- instantiate dataset
    ds = build_dataset_from_registry(reg, name=args.dataset, split=args.split, extra_kwargs=extra)

    # Determine trajectory structure
    time_count = getattr(ds, "time_count", None)
    if time_count is None:
        if args.time_count is None:
            raise RuntimeError(
                "Dataset does not expose ds.time_count and you did not pass --time_count.\n"
                "Fix: ensure your dataset class stores self.time_count, or pass --time_count."
            )
        time_count = int(args.time_count)
    else:
        time_count = int(time_count)

    if len(ds) % time_count != 0:
        raise RuntimeError(f"len(ds)={len(ds)} is not divisible by time_count={time_count}. Cannot split into trajectories.")

    n_traj = len(ds) // time_count
    if args.limit_traj is not None:
        n_traj = min(n_traj, int(args.limit_traj))

    traj_ids = try_get_traj_ids(ds)
    if traj_ids is not None:
        traj_ids = traj_ids[:n_traj]

    print(f">> dataset={args.dataset} split={args.split} len={len(ds)} time_count={time_count} n_traj={n_traj}", flush=True)

    # --- build ddconfig fallback from (dataset sample + CLI flags) ---
    # Load ckpt once to infer embed_dim for ddconfig['z_channels']
    ckpt0 = torch.load(args.ckpt, map_location="cpu")
    sd0 = ckpt0.get("state_dict", ckpt0)
    _, embed_dim0 = infer_codebook_from_state_dict(sd0)

    # Peek dataset sample shape
    ex0 = ds[0]
    x0 = ex0["image"]
    if not torch.is_tensor(x0):
        raise RuntimeError("Dataset must return torch.Tensor at key 'image'")
    C, H, W = _infer_chw(x0)

    model_in_channels = infer_model_in_channels_from_ckpt(args.ckpt)
    if model_in_channels is None:
        raise RuntimeError("Could not infer model input channels from checkpoint.")

    resolved_export_mode = resolve_export_mode(
        args.export_mode,
        model_in_channels=model_in_channels,
        dataset_channels=C,
    )

    print(
        f">> export_mode requested={args.export_mode} resolved={resolved_export_mode} "
        f"model_in_channels={model_in_channels} dataset_channels={C}",
        flush=True,
    )

    # Effective geometry (apply overrides if user passed them)
    if resolved_export_mode == "scalar_channels":
        C_eff = 1
    else:
        C_eff = int(args.model_channels) if args.model_channels is not None else C
    H_eff = int(args.res_h) if args.res_h is not None else H
    W_eff = int(args.res_w) if args.res_w is not None else W

    # Must match training
    ch_mult = df_to_ch_mult(args.down_factor)

    if (H_eff % args.down_factor) != 0 or (W_eff % args.down_factor) != 0:
        raise ValueError(f"H,W must be divisible by down_factor={args.down_factor}. Got H={H_eff}, W={W_eff}")

    latent_h = H_eff // args.down_factor
    latent_w = W_eff // args.down_factor

    ddconfig_fallback = dict(
        double_z=False,
        z_channels=int(embed_dim0),
        resolution=max(H_eff, W_eff),
        in_channels=int(C_eff),
        out_ch=int(C_eff),
        ch=int(args.base_ch),
        ch_mult=list(ch_mult),
        num_res_blocks=int(args.num_res_blocks),
        attn_resolutions=[int(min(latent_h, latent_w))],
        dropout=float(args.dropout),
    )

    # Now load model using fallback ddconfig if ckpt doesn't store it
    model = load_vqvae_from_ckpt(args.ckpt, ddconfig_fallback=ddconfig_fallback, use_ema_copy=args.use_ema_copy)
    model = model.to(device).eval().to(memory_format=torch.channels_last)

    torch.backends.cudnn.benchmark = True

    # ----- export
    if args.one_file:
        suffix = str(args.out_name_suffix)
        out_path = os.path.join(args.out_dir, f"{args.dataset}_{args.split}{suffix}_tokens.npz")

        # IMPORTANT: support safe chunked one_file export.
        start_frame = int(args.start_frame)
        if start_frame < 0:
            raise ValueError(f"--start_frame must be >= 0, got {start_frame}")
        if start_frame >= len(ds):
            raise ValueError(f"--start_frame={start_frame} >= len(ds)={len(ds)}")

        end_frame = len(ds)

        # Legacy trajectory limit: first N trajectories, unless start_frame also used.
        if args.limit_traj is not None:
            end_frame = min(end_frame, int(args.limit_traj) * int(time_count))

        if args.max_frames is not None:
            end_frame = min(end_frame, start_frame + int(args.max_frames))

        if end_frame <= start_frame:
            raise ValueError(
                f"Empty export slice: start_frame={start_frame}, end_frame={end_frame}, len(ds)={len(ds)}"
            )

        frame_indices = list(range(start_frame, end_frame))
        ds = tud.Subset(ds, frame_indices)

        print(
            f">> one_file: exporting frame slice [{start_frame}, {end_frame}) "
            f"num_frames={len(frame_indices)} original_len={len(frame_indices) if False else 'see above'}",
            flush=True,
        )

        dl = tud.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(args.num_workers > 0),
        )

        toks: List[torch.Tensor] = []
        Hq = Wq = None
        t0 = time.time()

        # Store a few x samples from batch 0 in CPU for debug MSE
        x_debug_np = None
        did_check = False

        for batch_i, batch in enumerate(dl):
            do_check = (not did_check)  # debug only on first batch

            # Encode -> inds (and roundtrip check only on batch 0)
            inds, Hq, Wq, x_cpu, rt_match = encode_batch_to_indices(
                model,
                batch,
                device,
                export_mode=resolved_export_mode,
                do_roundtrip_check=do_check,
            )
            
            toks.append(inds)

            if do_check:
                did_check = True
                # x_cpu is what get_input() returned, moved to CPU float16
                x_debug_np = x_cpu[:4].numpy()  # (<=4,C,H,W)

                # Print encoder-visible range from batch 0
                x0_dbg = x_cpu[0].float()
                print(f">> x range (what encoder sees): [{x0_dbg.min().item():.6f}, {x0_dbg.max().item():.6f}]", flush=True)

                print(f">> ROUNDTRIP token match (first batch): {rt_match:.6f}", flush=True)

            # progress so it never looks stuck
            if (batch_i + 1) % 20 == 0:
                print(f">> progress: batch {batch_i+1}", flush=True)

        tokens = torch.cat(toks, dim=0).numpy()  # (Nframes,Hq,Wq)
        t1 = time.time()

        # --- debug: MSE between decode(tokens[0]) and x_m11[0] from batch0
        if x_debug_np is not None:
            if resolved_export_mode == "scalar_channels":
                # tokens[0]: [C,Hq,Wq], x_debug_np[0]: [C,H,W]
                debug_ch = 0
                x0 = torch.from_numpy(x_debug_np[0, debug_ch]).unsqueeze(0).unsqueeze(0)
                x0 = x0.to(dtype=torch.float32)

                inds0 = torch.from_numpy(tokens[0, debug_ch]).unsqueeze(0).to(torch.long)
                inds0 = inds0.to(device)

                with torch.inference_mode():
                    xrec0 = decode_codes_to_x_m11(model, inds0).detach().to("cpu", dtype=torch.float32)

                mse = torch.mean((xrec0 - x0) ** 2).item()
                maxabs = torch.max(torch.abs(xrec0 - x0)).item()
                print(
                    f">> DEBUG decode(tokens[0,{debug_ch}]) vs x_m11[0,{debug_ch}]: "
                    f"MSE={mse:.6e} maxabs={maxabs:.6e}",
                    flush=True,
                )

            else:
                # native mode: tokens[0]: [Hq,Wq], x_debug_np[0]: [C,H,W]
                x0 = torch.from_numpy(x_debug_np[0]).unsqueeze(0)
                x0 = x0.to(dtype=torch.float32)

                inds0 = torch.from_numpy(tokens[0]).unsqueeze(0).to(torch.long)
                inds0 = inds0.to(device)

                with torch.inference_mode():
                    xrec0 = decode_codes_to_x_m11(model, inds0).detach().to("cpu", dtype=torch.float32)

                mse = torch.mean((xrec0 - x0) ** 2).item()
                maxabs = torch.max(torch.abs(xrec0 - x0)).item()
                print(
                    f">> DEBUG decode(tokens[0]) vs x_m11[0]: "
                    f"MSE={mse:.6e} maxabs={maxabs:.6e}",
                    flush=True,
                )

        # Infer codebook size from checkpoint state_dict (robust)
        ckpt = torch.load(args.ckpt, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        n_embed, embed_dim = infer_codebook_from_state_dict(sd)

        np.savez_compressed(
            out_path,
            tokens=tokens,
            Hq=np.int32(Hq),
            Wq=np.int32(Wq),
            n_embed=np.int32(n_embed),
            embed_dim=np.int32(embed_dim),
            dataset=np.array(args.dataset),
            split=np.array(args.split),
            time_count=np.int32(time_count),
            x_m11=x_debug_np,  # only first few samples from batch 0
            export_mode=np.array(resolved_export_mode),
            dataset_channels=np.int32(C),
            model_in_channels=np.int32(model_in_channels),
        )

        print(f">> wrote {out_path} tokens={tokens.shape} ({t1-t0:.2f}s)", flush=True)
        return


    # Per-trajectory outputs
    # NOTE: this assumes the dataset is laid out as (traj-major, then time).
    for ti in range(n_traj):
        traj_name = traj_ids[ti] if traj_ids is not None else f"traj{ti:06d}"
        out_path = os.path.join(args.out_dir, f"{args.split}_{traj_name}_tokens.npz")

        # indices for this trajectory
        start = ti * time_count
        idxs = list(range(start, start + time_count))
        ds_sub = tud.Subset(ds, idxs)

        dl = tud.DataLoader(
            ds_sub,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(args.num_workers > 0),
        )

        t0 = time.time()
        toks = []
        Hq = Wq = None
        did_check = False
        for batch_i, batch in enumerate(dl):
            do_check = (not did_check) and (ti == 0)   # first batch, first traj only
            inds, Hq, Wq, x_cpu, rt_match = encode_batch_to_indices(
                model,
                batch,
                device,
                export_mode=resolved_export_mode,
                do_roundtrip_check=do_check,
            )
            toks.append(inds)
            if do_check:
                did_check = True
                print(f">> ROUNDTRIP token match: {rt_match:.6f}", flush=True)
        tokens = torch.cat(toks, dim=0).numpy()  # (T,Hq,Wq)
        t1 = time.time()

        ckpt = torch.load(args.ckpt, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        n_embed, embed_dim = infer_codebook_from_state_dict(sd)

        np.savez_compressed(
            out_path,
            tokens=tokens,
            Hq=np.int32(Hq),
            Wq=np.int32(Wq),
            n_embed=np.int32(n_embed),
            embed_dim=np.int32(embed_dim),
            dataset=np.array(args.dataset),
            split=np.array(args.split),
            traj_idx=np.int32(ti),
            time_count=np.int32(time_count),
            export_mode=np.array(resolved_export_mode),
            dataset_channels=np.int32(C),
            model_in_channels=np.int32(model_in_channels),
        )

        if (ti + 1) % 25 == 0 or (ti + 1) == n_traj:
            print(f"[{ti+1}/{n_traj}] wrote {out_path} tokens={tokens.shape} encode={t1-t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()


