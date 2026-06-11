#!/usr/bin/env python3
import json, numpy as np, torch
from src.models.lit_vqvae import LitVQVAE

def load_vq_any(vq_ckpt: str, device):
    ck = torch.load(vq_ckpt, map_location="cpu")
    sd = ck.get("state_dict", ck)
    hp = ck.get("hyper_parameters", {}) or {}
    ddconfig = hp["ddconfig"]
    vq = LitVQVAE(
        ddconfig=ddconfig,
        n_embed=int(hp["n_embed"]),
        embed_dim=int(hp["embed_dim"]),
        learning_rate=float(hp.get("learning_rate", 2e-4)),
        beta=float(hp.get("beta", 0.25)),
        vq_loss_weight=float(hp.get("vq_loss_weight", 1.0)),
        recon_l2_weight=float(hp.get("recon_l2_weight", 0.0)),
        grad_loss_weight=float(hp.get("grad_loss_weight", 0.0)),
        l2_normalize_codebook=bool(hp.get("l2_normalize_codebook", False)),
        legacy_beta_bug=bool(hp.get("legacy_beta_bug", False)),
        output_clamp=bool(hp.get("output_clamp", True)),
        image_key=str(hp.get("image_key", "image")),
        no_quant=bool(hp.get("no_quant", False)),
        warmup_steps=int(hp.get("warmup_steps", 0)),
        total_steps=int(hp.get("total_steps", 0)),
        min_lr=float(hp.get("min_lr", 1e-6)),
    )
    vq.load_state_dict(sd, strict=False)
    return vq.to(device).eval()

@torch.inference_mode()
def encode_inds(vq, x_bchw, Hq: int, Wq: int):
    _, _, info = vq.encode(x_bchw)
    inds = info[-1]
    if inds is None:
        raise RuntimeError("encode() returned info[-1]=None (no indices).")

    # inds can be (B,Hq,Wq) OR flattened (B*Hq*Wq) OR (B,Hq*Wq)
    if not torch.is_tensor(inds):
        raise RuntimeError(f"info[-1] is not a Tensor: {type(inds)}")

    B = x_bchw.shape[0]
    if inds.ndim == 3:
        pass  # (B,Hq,Wq)
    elif inds.ndim == 2 and inds.shape[0] == B and inds.shape[1] == Hq * Wq:
        inds = inds.view(B, Hq, Wq)
    elif inds.ndim == 1 and inds.numel() == B * Hq * Wq:
        inds = inds.view(B, Hq, Wq)
    else:
        raise RuntimeError(f"Unexpected inds shape {tuple(inds.shape)}; expected (B,Hq,Wq) or flattenable to it.")

    return inds[0].detach().cpu().numpy().astype(np.int64)  # (Hq,Wq)

def to_m11(a_hw, vmin, vmax, clamp=True):
    den = (vmax - vmin)
    if abs(den) < 1e-12:
        den = 1.0
    a01 = (a_hw - vmin) / den
    if clamp:
        a01 = np.clip(a01, 0.0, 1.0)
    return (a01 * 2.0 - 1.0).astype(np.float32)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens_npz", required=True)
    ap.add_argument("--gt_npz", required=True)
    ap.add_argument("--gt_key", default="av_density")
    ap.add_argument("--vq_ckpt", required=True)
    ap.add_argument("--minmax_json", required=True)
    ap.add_argument("--t", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    ztok = np.load(args.tokens_npz, allow_pickle=True)
    tokens = ztok["tokens"].astype(np.int64)
    T, Hq, Wq = tokens.shape
    tok_t = tokens[args.t]

    zgt = np.load(args.gt_npz, allow_pickle=True)
    a = np.asarray(zgt[args.gt_key][args.t], dtype=np.float32)  # likely (560,200)
    print("GT raw shape:", a.shape, "range:", float(a.min()), float(a.max()))

    with open(args.minmax_json, "r") as f:
        mm = json.load(f)
    vmin_p = float(mm["min"][0]) if isinstance(mm["min"], list) else float(mm["min"])
    vmax_p = float(mm["max"][0]) if isinstance(mm["max"], list) else float(mm["max"])
    print("p999 minmax:", vmin_p, vmax_p)

    # also compute true global min/max from this file (cheap-ish: whole array)
    a_all = np.asarray(zgt[args.gt_key], dtype=np.float32)
    vmin_g = float(a_all.min()); vmax_g = float(a_all.max())
    print("file global minmax:", vmin_g, vmax_g)

    vq = load_vq_any(args.vq_ckpt, device=device)

    # try combinations
    variants = []
    for name_ori, a_hw in [("raw", a), ("T", a.T)]:
        for name_mm, (mn, mx) in [("p999", (vmin_p, vmax_p)), ("global", (vmin_g, vmax_g))]:
            x11 = to_m11(a_hw, mn, mx, clamp=True)  # (H,W)
            x_bchw = torch.from_numpy(x11[None, None]).to(device)  # (1,1,H,W)
            inds = encode_inds(vq, x_bchw, Hq=Hq, Wq=Wq)
            if inds.shape != (Hq, Wq):
                variants.append((name_ori, name_mm, -1.0, inds.shape))
                continue
            match = float((inds == tok_t).mean())
            variants.append((name_ori, name_mm, match, inds.shape))

    for v in variants:
        print(f"variant ori={v[0]:>3s} mm={v[1]:>6s}  match={v[2]:.6f}  inds_shape={v[3]}")

if __name__ == "__main__":
    main()

