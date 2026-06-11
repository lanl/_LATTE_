#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import time
import copy
import types
import argparse
import importlib.util
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.transformer_registry import build_transformer_splits_from_registry


# -------------------------
# DDP helpers
# -------------------------
def ddp_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def ddp_setup():
    if not ddp_enabled():
        return 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def ddp_cleanup():
    if ddp_enabled() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def is_rank0(rank: int) -> bool:
    return rank == 0


def rank_print(rank: int, *args, **kwargs):
    if is_rank0(rank):
        print(*args, **kwargs)


def reduce_scalar(x, device):
    if not torch.is_tensor(x):
        x = torch.tensor(float(x), device=device)
    else:
        x = x.to(device)
    if ddp_enabled():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x = x / dist.get_world_size()
    return x


# -------------------------
# model helpers
# -------------------------
def attach_forward(model):
    """
    rollout.py EncDecGPT has encode/decode but no forward().
    This adds forward() so DDP can track gradient-bearing calls.
    """
    def _forward(self, enc_tokens, dec_in, dec_tgt=None, deck_tokens=None):
        mem = self.encode(enc_tokens, deck_tokens=deck_tokens)
        logits = self.decode(dec_in, mem)
        if dec_tgt is None:
            return logits
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            dec_tgt.reshape(-1),
            ignore_index=self.pad_id,
        )
        return logits, loss

    model.forward = types.MethodType(_forward, model)
    return model


def load_rollout_module(path: str):
    spec = importlib.util.spec_from_file_location("latte_rollout_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import rollout module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_minmax_json(path: str):
    with open(path, "r") as f:
        mm = json.load(f)
    vmin = float(mm.get("min", 0.0))
    vmax = float(mm.get("p_hi", mm.get("max")))
    return vmin, vmax


def make_r_weights(n_r: int, r_min: float, r_max: float, device, dtype):
    return torch.linspace(float(r_min), float(r_max), int(n_r), device=device, dtype=dtype)


def filter_dataset_by_timestep(ds, start_t: int, end_t: int):
    """
    Keep samples with start_t <= t < end_t.
    For K-step rollout, caller should use end_t <= T-K.
    """
    if not hasattr(ds, "pairs"):
        raise RuntimeError("Dataset does not expose .pairs; cannot timestep-filter.")
    pairs = np.asarray(ds.pairs)
    ts = pairs[:, 1]
    keep = np.nonzero((ts >= int(start_t)) & (ts < int(end_t)))[0]
    if keep.size == 0:
        raise RuntimeError(f"No dataset pairs found for {start_t} <= t < {end_t}")
    return Subset(ds, keep.tolist())


def ce_loss_for_batch(model, enc, din, dtgt, deck_tokens=None):
    _, loss = model(enc, din, dtgt, deck_tokens=deck_tokens)
    return loss


# -------------------------
# token/sequence helpers
# -------------------------
@torch.no_grad()
def sample_sequences(model, enc, deck_tokens, *, bos_id: int, dec_block: int, temperature: float, top_k: int):
    """
    Sample one rowwise decoder sequence for each enc row.
    Returns seq without BOS, shape (B, dec_block).
    """
    m = unwrap(model)
    was_training = m.training
    m.eval()

    B = enc.size(0)
    mem = m.encode(enc, deck_tokens=deck_tokens)
    out = torch.full((B, 1), int(bos_id), device=enc.device, dtype=torch.long)

    temp = float(temperature)
    greedy = temp <= 0.0
    if greedy:
        temp = 1.0

    for _ in range(int(dec_block)):
        ctx = out[:, -m.dec_block_size:] if out.size(1) > m.dec_block_size else out
        logits = m.decode(ctx, mem)[:, -1, :] / temp

        if top_k and top_k > 0:
            k = min(int(top_k), logits.size(-1))
            vals, idx = torch.topk(logits, k=k, dim=-1)
            filt = torch.full_like(logits, float("-inf"))
            filt.scatter_(1, idx, vals)
            logits = filt

        if greedy:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)

        out = torch.cat([out, nxt], dim=1)

    if was_training:
        m.train()

    return out[:, 1:].contiguous()


def score_sequence_token_logprobs(model, enc, deck_tokens, seq, *, bos_id: int):
    """
    Gradient-bearing scoring of a sampled sequence.
    Returns token logprobs, shape (B, L).
    """
    B = seq.size(0)
    bos = torch.full((B, 1), int(bos_id), device=seq.device, dtype=torch.long)
    dec_in = torch.cat([bos, seq[:, :-1]], dim=1)

    logits, _ = model(enc, dec_in, seq, deck_tokens=deck_tokens)
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, seq.unsqueeze(-1)).squeeze(-1)


def extract_code_grid_from_seq(seq, *, n_embed: int, Hq: int, Wq: int):
    """
    Extract first Hq*Wq valid code tokens from rowwise decoder sequence.
    Pads with last seen code if too short.
    Returns torch long (B,Hq,Wq).
    """
    seq_np = seq.detach().cpu().numpy()
    B = seq_np.shape[0]
    need = int(Hq) * int(Wq)
    out = np.zeros((B, need), dtype=np.int64)

    for b in range(B):
        codes = []
        for tok in seq_np[b]:
            tok = int(tok)
            if 0 <= tok < int(n_embed):
                codes.append(tok)
                if len(codes) >= need:
                    break
        if len(codes) == 0:
            codes = [0] * need
        elif len(codes) < need:
            codes.extend([codes[-1]] * (need - len(codes)))
        out[b] = np.asarray(codes[:need], dtype=np.int64)

    return torch.from_numpy(out.reshape(B, int(Hq), int(Wq))).to(seq.device)


def code_grid_to_enc(codes_bhw: torch.Tensor) -> torch.Tensor:
    """
    codes_bhw: (B,Hq,Wq) -> enc tokens (B,Hq*Wq)
    """
    return codes_bhw.reshape(codes_bhw.size(0), -1).long()


@torch.no_grad()
def decode_codes_to_fields_torch(rollout_mod, vq, codes_bhw):
    """
    Decode VQ code grid to normalized field, shape (B,C,H,W).
    """
    w = rollout_mod._find_codebook_weight(vq)
    if w is None:
        raise RuntimeError("Could not locate VQ codebook embedding weight.")

    B, Hq, Wq = codes_bhw.shape
    z = w[codes_bhw.reshape(-1)]
    z = z.view(B, Hq, Wq, -1).permute(0, 3, 1, 2).contiguous()

    x = rollout_mod._vq_decode_latents(vq, z)
    if getattr(vq, "output_clamp", False):
        x = x.clamp(-1, 1)
    return x.float()


@torch.no_grad()
def r_weighted_mass_from_codes(rollout_mod, vq, codes_bhw, *, vmin: float, vmax: float, r_weights):
    """
    r-weighted cylindrical mass proxy from VQ codes.
    """
    x_norm = decode_codes_to_fields_torch(rollout_mod, vq, codes_bhw)
    x_phys = ((x_norm + 1.0) * 0.5) * (float(vmax) - float(vmin)) + float(vmin)
    x_phys = x_phys.clamp_min(0.0)

    if x_phys.shape[1] != 1:
        raise RuntimeError(f"Expected single-channel VQ decode, got C={x_phys.shape[1]}")
    if x_phys.shape[-1] != r_weights.numel():
        raise RuntimeError(f"Decoded R={x_phys.shape[-1]} but r_weights has {r_weights.numel()}")

    return (x_phys[:, 0] * r_weights[None, None, :]).sum(dim=(-2, -1))


# -------------------------
# short-rollout GRPO
# -------------------------
def rank_advantages(rewards: torch.Tensor, mode: str) -> torch.Tensor:
    """
    rewards: (B,G), larger is better.
    """
    B, G = rewards.shape
    mode = str(mode).lower()

    if mode == "rank":
        order = torch.argsort(rewards, dim=1)  # worst -> best
        template = torch.linspace(-1.0, 1.0, G, device=rewards.device, dtype=rewards.dtype)
        adv = torch.empty_like(rewards)
        adv.scatter_(1, order, template[None, :].expand(B, G))
        return adv

    if mode == "winner":
        best = torch.argmax(rewards, dim=1, keepdim=True)
        adv = torch.full_like(rewards, fill_value=-1.0 / max(1, G - 1))
        adv.scatter_(1, best, 1.0)
        return adv

    if mode == "reward_norm":
        adv = rewards - rewards.mean(dim=1, keepdim=True)
        adv = adv / (rewards.std(dim=1, keepdim=True, unbiased=False) + 1e-8)
        return adv

    raise ValueError(f"Unknown advantage_type={mode}")


def reduce_tokens(tok_values: torch.Tensor, seq: torch.Tensor, *, n_embed: int, reduction: str, code_only: bool):
    """
    tok_values: (N,L), e.g. token logprobs or token KL
    seq:        (N,L), sampled tokens
    """
    if bool(code_only):
        mask = ((seq >= 0) & (seq < int(n_embed))).to(tok_values.dtype)
    else:
        mask = torch.ones_like(tok_values, dtype=tok_values.dtype)

    denom = mask.sum(dim=1).clamp_min(1.0)

    reduction = str(reduction).lower()
    if reduction == "sum":
        return (tok_values * mask).sum(dim=1), denom
    if reduction == "mean":
        return (tok_values * mask).sum(dim=1) / denom, denom

    raise ValueError(f"Unknown reduction={reduction}")


def grpo_mass_short_rollout_loss(
    *,
    rollout_mod,
    model,
    ref_model,
    vq,
    enc,
    deck_tokens,
    n_embed: int,
    Hq: int,
    Wq: int,
    bos_id: int,
    dec_block: int,
    rollout_k: int,
    num_generations: int,
    temperature: float,
    top_k: int,
    vmin: float,
    vmax: float,
    r_weights,
    beta: float,
    logp_reduction: str,
    kl_reduction: str,
    advantage_type: str,
    code_only: bool,
):
    """
    Short-rollout GRPO.

    Input:
      enc is GT x_t tokens, shape (B,Hq*Wq)

    For each example and generation:
      autoregressively sample K predicted frames
      reward = -mean relative mass jump across predicted rollout:
          h=1: |M_pred1 - M_start| / |M_start|
          h>1: |M_pred_h - M_pred_{h-1}| / |M_pred_{h-1}|
    """
    B = enc.size(0)
    G = int(num_generations)
    K = int(rollout_k)
    device = enc.device

    # Repeat initial state by candidate group.
    enc_cur = enc.repeat_interleave(G, dim=0)
    deck_rep = None if deck_tokens is None else deck_tokens.repeat_interleave(G, dim=0)

    # Starting mass from initial input codes.
    start_codes = enc_cur.reshape(B * G, Hq, Wq)
    with torch.no_grad():
        prev_mass = r_weighted_mass_from_codes(
            rollout_mod, vq, start_codes, vmin=vmin, vmax=vmax, r_weights=r_weights
        )

    all_seq: List[torch.Tensor] = []
    all_enc_for_score: List[torch.Tensor] = []
    all_deck_for_score: List[torch.Tensor | None] = []
    rel_jumps: List[torch.Tensor] = []

    # Sample K-step rollout with no-grad. Then later rescore the fixed sampled sequences with grad.
    with torch.no_grad():
        for h in range(K):
            seq_h = sample_sequences(
                model,
                enc_cur,
                deck_rep,
                bos_id=bos_id,
                dec_block=dec_block,
                temperature=temperature,
                top_k=top_k,
            )

            codes_h = extract_code_grid_from_seq(seq_h, n_embed=n_embed, Hq=Hq, Wq=Wq)
            mass_h = r_weighted_mass_from_codes(
                rollout_mod, vq, codes_h, vmin=vmin, vmax=vmax, r_weights=r_weights
            )

            jump_h = torch.abs(mass_h - prev_mass) / (torch.abs(prev_mass) + 1e-8)
            rel_jumps.append(jump_h)

            all_seq.append(seq_h)
            all_enc_for_score.append(enc_cur)
            all_deck_for_score.append(deck_rep)

            # Feed sampled predicted codes back as next encoder input.
            enc_cur = code_grid_to_enc(codes_h)
            prev_mass = mass_h

    # Reward per rollout candidate.
    # rel_jumps: K tensors of shape (B*G,)
    jumps = torch.stack(rel_jumps, dim=1)  # (B*G,K)
    reward_rep = -jumps.mean(dim=1)       # (B*G,)
    rewards = reward_rep.view(B, G)

    # Score the sampled K-step rollout under active model and ref model.
    rollout_logp_parts = []
    rollout_kl_parts = []
    code_count_parts = []

    for h in range(K):
        enc_h = all_enc_for_score[h]
        seq_h = all_seq[h]
        deck_h = all_deck_for_score[h]

        tok_logp_h = score_sequence_token_logprobs(
            model, enc_h, deck_h, seq_h, bos_id=bos_id
        )

        with torch.no_grad():
            ref_tok_logp_h = score_sequence_token_logprobs(
                ref_model, enc_h, deck_h, seq_h, bos_id=bos_id
            )
            delta = ref_tok_logp_h - tok_logp_h.detach()
            kl_tok_h = torch.exp(delta) - delta - 1.0

        seq_logp_h, denom_h = reduce_tokens(
            tok_logp_h,
            seq_h,
            n_embed=n_embed,
            reduction=logp_reduction,
            code_only=code_only,
        )
        seq_kl_h, _ = reduce_tokens(
            kl_tok_h,
            seq_h,
            n_embed=n_embed,
            reduction=kl_reduction,
            code_only=code_only,
        )

        rollout_logp_parts.append(seq_logp_h)
        rollout_kl_parts.append(seq_kl_h)
        code_count_parts.append(denom_h)

    rollout_logp = torch.stack(rollout_logp_parts, dim=1).sum(dim=1).view(B, G)
    rollout_kl = torch.stack(rollout_kl_parts, dim=1).sum(dim=1).view(B, G)
    code_count = torch.stack(code_count_parts, dim=1).sum(dim=1).view(B, G)

    adv = rank_advantages(rewards, advantage_type)

    policy_loss = -(adv.detach() * rollout_logp).mean()
    kl_loss = rollout_kl.mean()
    loss = policy_loss + float(beta) * kl_loss

    best_idx = torch.argmax(rewards, dim=1, keepdim=True)
    worst_idx = torch.argmin(rewards, dim=1, keepdim=True)

    best_logp = rollout_logp.gather(1, best_idx).mean()
    worst_logp = rollout_logp.gather(1, worst_idx).mean()
    best_reward = rewards.gather(1, best_idx).mean()
    worst_reward = rewards.gather(1, worst_idx).mean()

    stats = {
        "grpo_loss": loss.detach(),
        "policy_loss": policy_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "reward_mean": rewards.mean().detach(),
        "reward_best": best_reward.detach(),
        "reward_worst": worst_reward.detach(),
        "mass_jump_mean": jumps.mean().detach(),
        "mass_jump_max": jumps.max().detach(),
        "logp_mean": rollout_logp.mean().detach(),
        "best_logp": best_logp.detach(),
        "worst_logp": worst_logp.detach(),
        "best_minus_worst_logp": (best_logp - worst_logp).detach(),
        "adv_abs_mean": adv.abs().mean().detach(),
        "code_token_count_mean": code_count.mean().detach(),
    }

    return loss, stats


def save_ckpt(path, model, opt, step, args, extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    base = unwrap(model)
    torch.save(
        {
            "model": base.state_dict(),
            "optimizer": opt.state_dict(),
            "step": int(step),
            "cfg": vars(args),
            **extra,
        },
        tmp,
    )
    os.replace(tmp, path)


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--datasets_json", required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--gpt_ckpt", required=True)
    ap.add_argument("--vq_ckpt", required=True)
    ap.add_argument("--minmax_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rollout_py", default="scripts/rollout.py")

    ap.add_argument("--no_deck_tokens", action="store_true")
    ap.add_argument("--batch_size", type=int, default=1, help="Per-GPU batch size.")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_pairs", type=int, default=None)

    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--save_every", type=int, default=250)
    ap.add_argument("--log_every", type=int, default=1)

    ap.add_argument("--grpo_start_t", type=int, default=50)
    ap.add_argument("--grpo_end_t", type=int, default=97)
    ap.add_argument("--rollout_k", type=int, default=3)
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)

    ap.add_argument("--lambda_grpo", type=float, default=1e-4)
    ap.add_argument("--grpo_beta", type=float, default=3e-4)
    ap.add_argument("--grpo_logp_reduction", type=str, default="sum", choices=["mean", "sum"])
    ap.add_argument("--grpo_kl_reduction", type=str, default="sum", choices=["mean", "sum"])
    ap.add_argument("--grpo_advantage_type", type=str, default="rank", choices=["reward_norm", "rank", "winner"])
    ap.add_argument("--grpo_code_only", action="store_true")

    ap.add_argument("--r_min", type=float, default=0.0)
    ap.add_argument("--r_max", type=float, default=10.0)

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use_vq_ema_copy", action="store_true")

    args = ap.parse_args()

    if args.num_generations < 2:
        raise ValueError("--num_generations must be >= 2 for GRPO.")
    if args.rollout_k < 1:
        raise ValueError("--rollout_k must be >= 1.")

    rank, world_size, local_rank = ddp_setup()

    try:
        torch.manual_seed(args.seed + rank)
        np.random.seed(args.seed + rank)

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}" if ddp_enabled() else args.device)
        else:
            device = torch.device("cpu")

        if is_rank0(rank):
            os.makedirs(args.out_dir, exist_ok=True)

        rollout_mod = load_rollout_module(args.rollout_py)

        ck = torch.load(args.gpt_ckpt, map_location="cpu")
        n_embed = int(ck["n_embed"])
        Hq = int(ck["Hq"])
        Wq = int(ck["Wq"])
        dec_block = int(ck["dec_block"])
        bos_id = int(ck["bos_id"])
        eos_id = int(ck["eos_id"])
        row_id = int(ck["row_id"])
        pad_id = int(ck["pad_id"])
        deck_len = int(ck.get("deck_len", 0))

        rank_print(rank, f">> ckpt n_embed={n_embed} Hq={Hq} Wq={Wq} dec_block={dec_block} deck_len={deck_len}")
        rank_print(rank, f">> specials bos={bos_id} eos={eos_id} row={row_id} pad={pad_id}")
        rank_print(rank, f">> DDP world_size={world_size} rank={rank} local_rank={local_rank}")

        if deck_len != 0:
            raise RuntimeError("This short-rollout GRPO script assumes nodeck checkpoint with deck_len=0.")

        model, _ = rollout_mod.load_gpt_any(args.gpt_ckpt, device=device, Hq=Hq, Wq=Wq, n_embed=n_embed)
        model = attach_forward(model)
        model.train()

        ref_model = attach_forward(copy.deepcopy(model).to(device).eval())
        for p in ref_model.parameters():
            p.requires_grad_(False)

        if ddp_enabled():
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )

        vq = rollout_mod.load_vq_any(args.vq_ckpt, device=device, use_ema_copy=args.use_vq_ema_copy)
        vq.eval()
        for p in vq.parameters():
            p.requires_grad_(False)

        vmin, vmax = load_minmax_json(args.minmax_json)
        rank_print(rank, f">> minmax: vmin={vmin} vmax/p_hi={vmax}")

        decoded_R = int(Wq) * 16
        r_weights = make_r_weights(decoded_R, args.r_min, args.r_max, device=device, dtype=torch.float32)
        rank_print(rank, f">> r_weights: n={decoded_R} min={args.r_min} max={args.r_max}")

        with open(args.datasets_json, "r") as f:
            reg = json.load(f)

        extra_train = {"seed": args.seed, "return_timestep": True}
        extra_val = {"seed": args.seed}

        if args.max_pairs is not None:
            extra_train["max_pairs"] = int(args.max_pairs)

        if args.no_deck_tokens:
            extra_train["use_deck"] = False
            extra_val["use_deck"] = False

        ds_tr, _, _ = build_transformer_splits_from_registry(
            reg,
            name=args.dataset_name,
            extra_train_kwargs=extra_train,
            extra_val_kwargs=extra_val,
        )

        ds_tr = filter_dataset_by_timestep(ds_tr, args.grpo_start_t, args.grpo_end_t)

        sampler = DistributedSampler(
            ds_tr,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        ) if ddp_enabled() else None

        dl = DataLoader(
            ds_tr,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=(args.num_workers > 0),
        )

        opt = AdamW(unwrap(model).parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

        rank_print(rank, f">> filtered dataset len={len(ds_tr)} batch_size_per_gpu={args.batch_size}")
        rank_print(rank, f">> effective selected examples per optimizer step = batch({args.batch_size}) * world({world_size}) * accum({args.grad_accum})")
        rank_print(rank, f">> GRPO start timesteps: {args.grpo_start_t} <= t < {args.grpo_end_t}")
        rank_print(rank, f">> short rollout K={args.rollout_k} num_generations={args.num_generations}")
        rank_print(rank, f">> lambda_grpo={args.lambda_grpo} beta={args.grpo_beta}")
        rank_print(rank, f">> logp_reduction={args.grpo_logp_reduction} kl_reduction={args.grpo_kl_reduction} advantage_type={args.grpo_advantage_type} code_only={args.grpo_code_only}")
        rank_print(rank, f">> output: {args.out_dir}")

        step = 0
        micro = 0
        epoch = 0
        opt.zero_grad(set_to_none=True)
        t0 = time.time()

        while step < args.max_steps:
            if sampler is not None:
                sampler.set_epoch(epoch)

            for batch in dl:
                deck, enc, din, dtgt, ts = batch

                enc = enc.to(device, non_blocking=True)
                din = din.to(device, non_blocking=True)
                dtgt = dtgt.to(device, non_blocking=True)
                deck_in = None

                ce = ce_loss_for_batch(model, enc, din, dtgt, deck_tokens=deck_in)

                grpo, grpo_stats = grpo_mass_short_rollout_loss(
                    rollout_mod=rollout_mod,
                    model=model,
                    ref_model=ref_model,
                    vq=vq,
                    enc=enc,
                    deck_tokens=None,
                    n_embed=n_embed,
                    Hq=Hq,
                    Wq=Wq,
                    bos_id=bos_id,
                    dec_block=dec_block,
                    rollout_k=args.rollout_k,
                    num_generations=args.num_generations,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    vmin=vmin,
                    vmax=vmax,
                    r_weights=r_weights,
                    beta=args.grpo_beta,
                    logp_reduction=args.grpo_logp_reduction,
                    kl_reduction=args.grpo_kl_reduction,
                    advantage_type=args.grpo_advantage_type,
                    code_only=args.grpo_code_only,
                )

                loss_full = ce + float(args.lambda_grpo) * grpo
                loss = loss_full / int(args.grad_accum)
                loss.backward()
                micro += 1

                if micro % int(args.grad_accum) == 0:
                    if args.grad_clip and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), float(args.grad_clip))

                    opt.step()
                    opt.zero_grad(set_to_none=True)

                    if step % args.log_every == 0:
                        ce_r = reduce_scalar(ce.detach(), device)
                        loss_r = reduce_scalar(loss_full.detach(), device)
                        grpo_r = reduce_scalar(grpo_stats["grpo_loss"], device)
                        policy_r = reduce_scalar(grpo_stats["policy_loss"], device)
                        kl_r = reduce_scalar(grpo_stats["kl_loss"], device)
                        rew_r = reduce_scalar(grpo_stats["reward_mean"], device)
                        rew_best_r = reduce_scalar(grpo_stats["reward_best"], device)
                        rew_worst_r = reduce_scalar(grpo_stats["reward_worst"], device)
                        jump_mean_r = reduce_scalar(grpo_stats["mass_jump_mean"], device)
                        jump_max_r = reduce_scalar(grpo_stats["mass_jump_max"], device)
                        logp_r = reduce_scalar(grpo_stats["logp_mean"], device)
                        bmw_logp_r = reduce_scalar(grpo_stats["best_minus_worst_logp"], device)
                        adv_abs_r = reduce_scalar(grpo_stats["adv_abs_mean"], device)
                        code_n_r = reduce_scalar(grpo_stats["code_token_count_mean"], device)

                        if is_rank0(rank):
                            dt = (time.time() - t0) / 60.0
                            print(
                                f"[{step:06d}] "
                                f"ce={float(ce_r.cpu()):.5f} "
                                f"loss={float(loss_r.cpu()):.5f} "
                                f"elapsed_min={dt:.1f}"
                                f" grpo={float(grpo_r.cpu()):.5f}"
                                f" policy={float(policy_r.cpu()):.5f}"
                                f" kl={float(kl_r.cpu()):.5e}"
                                f" rew={float(rew_r.cpu()):.5e}"
                                f" rew_best={float(rew_best_r.cpu()):.5e}"
                                f" rew_worst={float(rew_worst_r.cpu()):.5e}"
                                f" jump_mean={float(jump_mean_r.cpu()):.5e}"
                                f" jump_max={float(jump_max_r.cpu()):.5e}"
                                f" logp={float(logp_r.cpu()):.5e}"
                                f" best_minus_worst_logp={float(bmw_logp_r.cpu()):.5e}"
                                f" adv_abs={float(adv_abs_r.cpu()):.5e}"
                                f" code_n={float(code_n_r.cpu()):.1f}",
                                flush=True,
                            )

                    if step > 0 and step % args.save_every == 0:
                        if ddp_enabled():
                            dist.barrier()
                        if is_rank0(rank):
                            extra = dict(
                                dataset_name=str(args.dataset_name),
                                datasets_json=str(args.datasets_json),
                                vocab_size=int(ck["vocab_size"]),
                                n_embed=int(n_embed),
                                Hq=int(Hq),
                                Wq=int(Wq),
                                dec_block=int(dec_block),
                                bos_id=int(bos_id),
                                eos_id=int(eos_id),
                                row_id=int(row_id),
                                pad_id=int(pad_id),
                                deck_len=int(deck_len),
                                world_size=int(world_size),
                                grpo_reward="short_rollout_r_weighted_mass_jump",
                                rollout_k=int(args.rollout_k),
                                vmin=float(vmin),
                                vmax=float(vmax),
                                r_min=float(args.r_min),
                                r_max=float(args.r_max),
                            )
                            path = os.path.join(args.out_dir, f"gpt_grpo_step{step}.pt")
                            save_ckpt(path, model, opt, step, args, extra)
                            print(f">> saved {path}", flush=True)
                        if ddp_enabled():
                            dist.barrier()

                    step += 1
                    if step >= args.max_steps:
                        break

            epoch += 1
            if step >= args.max_steps:
                break

        if ddp_enabled():
            dist.barrier()

        if is_rank0(rank):
            extra = dict(
                dataset_name=str(args.dataset_name),
                datasets_json=str(args.datasets_json),
                vocab_size=int(ck["vocab_size"]),
                n_embed=int(n_embed),
                Hq=int(Hq),
                Wq=int(Wq),
                dec_block=int(dec_block),
                bos_id=int(bos_id),
                eos_id=int(eos_id),
                row_id=int(row_id),
                pad_id=int(pad_id),
                deck_len=int(deck_len),
                world_size=int(world_size),
                grpo_reward="short_rollout_r_weighted_mass_jump",
                rollout_k=int(args.rollout_k),
                vmin=float(vmin),
                vmax=float(vmax),
                r_min=float(args.r_min),
                r_max=float(args.r_max),
            )
            path = os.path.join(args.out_dir, "gpt_grpo_final.pt")
            save_ckpt(path, model, opt, step, args, extra)
            print(f">> saved final {path}", flush=True)

    finally:
        ddp_cleanup()


if __name__ == "__main__":
    main()
