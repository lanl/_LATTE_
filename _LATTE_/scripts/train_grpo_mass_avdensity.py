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

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.transformer_registry import build_transformer_splits_from_registry


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


def is_rank0(rank: int) -> bool:
    return rank == 0


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def rank_print(rank: int, *args, **kwargs):
    if is_rank0(rank):
        print(*args, **kwargs)


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


@torch.no_grad()
def sample_sequences(model, enc, deck_tokens, *, bos_id: int, dec_block: int, temperature: float, top_k: int):
    """
    Sampling is no-grad. It may call unwrap(model).encode/decode because the sampled
    tokens are actions, not differentiable paths.
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
    Gradient-bearing scoring of sampled sequence. For active model this goes through
    DDP forward. For ref_model, which is not DDP-wrapped, it also works.
    """
    B = seq.size(0)
    bos = torch.full((B, 1), int(bos_id), device=seq.device, dtype=torch.long)
    dec_in = torch.cat([bos, seq[:, :-1]], dim=1)

    logits, _ = model(enc, dec_in, seq, deck_tokens=deck_tokens)
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, seq.unsqueeze(-1)).squeeze(-1)


def extract_code_grid_from_seq(seq, *, n_embed: int, Hq: int, Wq: int):
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


def extract_code_grid_from_dtgt(dtgt, *, n_embed: int, Hq: int, Wq: int):
    return extract_code_grid_from_seq(dtgt, n_embed=n_embed, Hq=Hq, Wq=Wq)


@torch.no_grad()
def decode_codes_to_fields_torch(rollout_mod, vq, codes_bhw):
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
    x_norm = decode_codes_to_fields_torch(rollout_mod, vq, codes_bhw)

    x_phys = ((x_norm + 1.0) * 0.5) * (float(vmax) - float(vmin)) + float(vmin)
    x_phys = x_phys.clamp_min(0.0)

    if x_phys.shape[1] != 1:
        raise RuntimeError(f"Expected single-channel VQ decode, got C={x_phys.shape[1]}")
    if x_phys.shape[-1] != r_weights.numel():
        raise RuntimeError(f"Decoded R={x_phys.shape[-1]} but r_weights has {r_weights.numel()}")

    return (x_phys[:, 0] * r_weights[None, None, :]).sum(dim=(-2, -1))


def grpo_mass_loss(
    *,
    rollout_mod,
    model,
    ref_model,
    vq,
    enc,
    dtgt,
    deck_tokens,
    n_embed: int,
    Hq: int,
    Wq: int,
    bos_id: int,
    dec_block: int,
    num_generations: int,
    temperature: float,
    top_k: int,
    vmin: float,
    vmax: float,
    r_weights,
    mass_tol: float,
    beta: float,
    reward_l1_weight: float = 0.0,
    reward_mse_weight: float = 0.0,
    logp_reduction: str = "mean",
    kl_reduction: str = "mean",
    advantage_type: str = "reward_norm",
    code_only: bool = False,
):
    B = enc.size(0)

    gt_codes = extract_code_grid_from_dtgt(dtgt, n_embed=n_embed, Hq=Hq, Wq=Wq)
    with torch.no_grad():
        mass_gt = r_weighted_mass_from_codes(
            rollout_mod, vq, gt_codes, vmin=vmin, vmax=vmax, r_weights=r_weights
        )

    G = int(num_generations)
    enc_rep = enc.repeat_interleave(G, dim=0)
    deck_rep = None if deck_tokens is None else deck_tokens.repeat_interleave(G, dim=0)

    with torch.no_grad():
        seq_rep = sample_sequences(
            model,
            enc_rep,
            deck_rep,
            bos_id=bos_id,
            dec_block=dec_block,
            temperature=temperature,
            top_k=top_k,
        )

    tok_logp_rep = score_sequence_token_logprobs(
        model, enc_rep, deck_rep, seq_rep, bos_id=bos_id
    )

    with torch.no_grad():
        ref_tok_logp_rep = score_sequence_token_logprobs(
            ref_model, enc_rep, deck_rep, seq_rep, bos_id=bos_id
        )

        pred_codes_rep = extract_code_grid_from_seq(seq_rep, n_embed=n_embed, Hq=Hq, Wq=Wq)
        mass_pred_rep = r_weighted_mass_from_codes(
            rollout_mod, vq, pred_codes_rep, vmin=vmin, vmax=vmax, r_weights=r_weights
        )

        mass_gt_rep = mass_gt.repeat_interleave(G)
        rel_err_rep = torch.abs(mass_pred_rep - mass_gt_rep) / (torch.abs(mass_gt_rep) + 1e-8)
        mass_penalty_rep = torch.clamp(rel_err_rep - float(mass_tol), min=0.0)

        field_l1_rep = torch.zeros_like(rel_err_rep)
        field_mse_rep = torch.zeros_like(rel_err_rep)

        # Optional decoded-field reward term.
        # Uses dimensionless physical-space error: (pred_phys - gt_phys) / (vmax - vmin).
        if float(reward_l1_weight) > 0.0 or float(reward_mse_weight) > 0.0:
            gt_codes_rep = gt_codes.repeat_interleave(G, dim=0)

            x_pred_norm = decode_codes_to_fields_torch(rollout_mod, vq, pred_codes_rep)
            x_gt_norm = decode_codes_to_fields_torch(rollout_mod, vq, gt_codes_rep)

            x_pred_phys = ((x_pred_norm + 1.0) * 0.5) * (float(vmax) - float(vmin)) + float(vmin)
            x_gt_phys = ((x_gt_norm + 1.0) * 0.5) * (float(vmax) - float(vmin)) + float(vmin)

            scale = max(float(vmax) - float(vmin), 1e-8)
            diff = (x_pred_phys - x_gt_phys) / scale

            field_l1_rep = diff.abs().mean(dim=(1, 2, 3))
            field_mse_rep = (diff * diff).mean(dim=(1, 2, 3))

        reward_rep = (
            -mass_penalty_rep
            - float(reward_l1_weight) * field_l1_rep
            - float(reward_mse_weight) * field_mse_rep
        )

        delta = ref_tok_logp_rep - tok_logp_rep.detach()
        kl_tok_rep = torch.exp(delta) - delta - 1.0

    logp_reduction = str(logp_reduction).lower()
    kl_reduction = str(kl_reduction).lower()
    advantage_type = str(advantage_type).lower()

    if bool(code_only):
        # Reward is computed only from decoded VQ codes, so apply GRPO only to code-token positions.
        # This excludes ROW/EOS/PAD/etc. and avoids pushing probability on tokens unrelated to mass.
        token_mask = ((seq_rep >= 0) & (seq_rep < int(n_embed))).to(tok_logp_rep.dtype)
    else:
        token_mask = torch.ones_like(tok_logp_rep, dtype=tok_logp_rep.dtype)

    denom = token_mask.sum(dim=1).clamp_min(1.0)

    if logp_reduction == "sum":
        # Sequence/code log-probability: log p(y|x) = sum_t log p(y_t|...)
        logp = (tok_logp_rep * token_mask).sum(dim=1).view(B, G)
    elif logp_reduction == "mean":
        logp = ((tok_logp_rep * token_mask).sum(dim=1) / denom).view(B, G)
    else:
        raise ValueError(f"Unknown logp_reduction={logp_reduction}. Use 'mean' or 'sum'.")

    if kl_reduction == "sum":
        kl = (kl_tok_rep * token_mask).sum(dim=1).view(B, G)
    elif kl_reduction == "mean":
        kl = ((kl_tok_rep * token_mask).sum(dim=1) / denom).view(B, G)
    else:
        raise ValueError(f"Unknown kl_reduction={kl_reduction}. Use 'mean' or 'sum'.")

    rewards = reward_rep.view(B, G)
    rels = rel_err_rep.view(B, G)

    if advantage_type == "reward_norm":
        # Old behavior: normalize reward magnitudes within each group.
        adv = rewards - rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True, unbiased=False)
        adv = adv / (std + 1e-8)

    elif advantage_type == "rank":
        # Rank-based group advantage.
        # Worst reward gets -1, best reward gets +1, evenly spaced in between.
        # This is more robust when reward magnitudes are tiny/noisy.
        order = torch.argsort(rewards, dim=1)  # ascending: worst -> best
        template = torch.linspace(-1.0, 1.0, G, device=rewards.device, dtype=rewards.dtype)
        adv = torch.empty_like(rewards)
        adv.scatter_(1, order, template[None, :].expand(B, G))

    elif advantage_type == "winner":
        # Stronger variant: best gets +1, all others get -1/(G-1), so group mean is zero.
        best = torch.argmax(rewards, dim=1, keepdim=True)
        adv = torch.full_like(rewards, fill_value=-1.0 / max(1, G - 1))
        adv.scatter_(1, best, 1.0)

    else:
        raise ValueError(
            f"Unknown advantage_type={advantage_type}. "
            "Use 'reward_norm', 'rank', or 'winner'."
        )

    policy_loss = -(adv.detach() * logp).mean()
    kl_loss = kl.mean()
    loss = policy_loss + float(beta) * kl_loss

    best_idx = torch.argmax(rewards, dim=1, keepdim=True)
    worst_idx = torch.argmin(rewards, dim=1, keepdim=True)
    best_logp = logp.gather(1, best_idx).mean()
    worst_logp = logp.gather(1, worst_idx).mean()
    best_minus_worst_logp = best_logp - worst_logp

    stats = {
        "grpo_loss": loss.detach(),
        "policy_loss": policy_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "reward_mean": rewards.mean().detach(),
        "rel_err_mean": rels.mean().detach(),
        "rel_err_min": rels.min().detach(),
        "rel_err_max": rels.max().detach(),
        "field_l1_mean": field_l1_rep.view(B, G).mean().detach(),
        "field_mse_mean": field_mse_rep.view(B, G).mean().detach(),
        "policy_loss": policy_loss.detach(),
        "logp_mean": logp.mean().detach(),
        "logp_abs_mean": logp.abs().mean().detach(),
        "adv_abs_mean": adv.abs().mean().detach(),
        "reward_best_mean": rewards.max(dim=1).values.mean().detach(),
        "reward_worst_mean": rewards.min(dim=1).values.mean().detach(),
        "best_logp_mean": best_logp.detach(),
        "worst_logp_mean": worst_logp.detach(),
        "best_minus_worst_logp": best_minus_worst_logp.detach(),
        "code_token_count_mean": denom.view(B, G).mean().detach(),
    }
    return loss, stats



def compute_grpo_loss_on_fixed_sequences(
    *,
    rollout_mod,
    model,
    ref_model,
    vq,
    enc,
    dtgt,
    seq_rep,
    deck_tokens,
    n_embed: int,
    Hq: int,
    Wq: int,
    bos_id: int,
    num_generations: int,
    vmin: float,
    vmax: float,
    r_weights,
    mass_tol: float,
    beta: float,
    reward_l1_weight: float = 0.0,
    reward_mse_weight: float = 0.0,
    logp_reduction: str = "sum",
    kl_reduction: str = "sum",
    advantage_type: str = "rank",
    code_only: bool = True,
):
    B = enc.size(0)
    G = int(num_generations)

    enc_rep = enc.repeat_interleave(G, dim=0)
    deck_rep = None if deck_tokens is None else deck_tokens.repeat_interleave(G, dim=0)

    gt_codes = extract_code_grid_from_dtgt(dtgt, n_embed=n_embed, Hq=Hq, Wq=Wq)
    with torch.no_grad():
        mass_gt = r_weighted_mass_from_codes(
            rollout_mod, vq, gt_codes, vmin=vmin, vmax=vmax, r_weights=r_weights
        )

        pred_codes_rep = extract_code_grid_from_seq(seq_rep, n_embed=n_embed, Hq=Hq, Wq=Wq)
        mass_pred_rep = r_weighted_mass_from_codes(
            rollout_mod, vq, pred_codes_rep, vmin=vmin, vmax=vmax, r_weights=r_weights
        )

        mass_gt_rep = mass_gt.repeat_interleave(G)
        rel_err_rep = torch.abs(mass_pred_rep - mass_gt_rep) / (torch.abs(mass_gt_rep) + 1e-8)
        mass_penalty_rep = torch.clamp(rel_err_rep - float(mass_tol), min=0.0)

        field_l1_rep = torch.zeros_like(rel_err_rep)
        field_mse_rep = torch.zeros_like(rel_err_rep)

        if float(reward_l1_weight) > 0.0 or float(reward_mse_weight) > 0.0:
            gt_codes_rep = gt_codes.repeat_interleave(G, dim=0)
            x_pred_norm = decode_codes_to_fields_torch(rollout_mod, vq, pred_codes_rep)
            x_gt_norm = decode_codes_to_fields_torch(rollout_mod, vq, gt_codes_rep)

            x_pred_phys = ((x_pred_norm + 1.0) * 0.5) * (float(vmax) - float(vmin)) + float(vmin)
            x_gt_phys = ((x_gt_norm + 1.0) * 0.5) * (float(vmax) - float(vmin)) + float(vmin)

            scale = max(float(vmax) - float(vmin), 1e-8)
            diff = (x_pred_phys - x_gt_phys) / scale
            field_l1_rep = diff.abs().mean(dim=(1, 2, 3))
            field_mse_rep = (diff * diff).mean(dim=(1, 2, 3))

        reward_rep = (
            -mass_penalty_rep
            - float(reward_l1_weight) * field_l1_rep
            - float(reward_mse_weight) * field_mse_rep
        )

    tok_logp_rep = score_sequence_token_logprobs(
        model, enc_rep, deck_rep, seq_rep, bos_id=bos_id
    )

    with torch.no_grad():
        ref_tok_logp_rep = score_sequence_token_logprobs(
            ref_model, enc_rep, deck_rep, seq_rep, bos_id=bos_id
        )
        delta = ref_tok_logp_rep - tok_logp_rep.detach()
        kl_tok_rep = torch.exp(delta) - delta - 1.0

    if bool(code_only):
        token_mask = ((seq_rep >= 0) & (seq_rep < int(n_embed))).to(tok_logp_rep.dtype)
    else:
        token_mask = torch.ones_like(tok_logp_rep, dtype=tok_logp_rep.dtype)

    denom = token_mask.sum(dim=1).clamp_min(1.0)

    logp_reduction = str(logp_reduction).lower()
    kl_reduction = str(kl_reduction).lower()
    advantage_type = str(advantage_type).lower()

    if logp_reduction == "sum":
        logp = (tok_logp_rep * token_mask).sum(dim=1).view(B, G)
    elif logp_reduction == "mean":
        logp = ((tok_logp_rep * token_mask).sum(dim=1) / denom).view(B, G)
    else:
        raise ValueError(logp_reduction)

    if kl_reduction == "sum":
        kl = (kl_tok_rep * token_mask).sum(dim=1).view(B, G)
    elif kl_reduction == "mean":
        kl = ((kl_tok_rep * token_mask).sum(dim=1) / denom).view(B, G)
    else:
        raise ValueError(kl_reduction)

    rewards = reward_rep.view(B, G)
    rels = rel_err_rep.view(B, G)

    if advantage_type == "rank":
        order = torch.argsort(rewards, dim=1)
        template = torch.linspace(-1.0, 1.0, G, device=rewards.device, dtype=rewards.dtype)
        adv = torch.empty_like(rewards)
        adv.scatter_(1, order, template[None, :].expand(B, G))
    elif advantage_type == "winner":
        best = torch.argmax(rewards, dim=1, keepdim=True)
        adv = torch.full_like(rewards, fill_value=-1.0 / max(1, G - 1))
        adv.scatter_(1, best, 1.0)
    elif advantage_type == "reward_norm":
        adv = rewards - rewards.mean(dim=1, keepdim=True)
        adv = adv / (rewards.std(dim=1, keepdim=True, unbiased=False) + 1e-8)
    else:
        raise ValueError(advantage_type)

    policy_loss = -(adv.detach() * logp).mean()
    kl_loss = kl.mean()
    loss = policy_loss + float(beta) * kl_loss

    best_idx = torch.argmax(rewards, dim=1, keepdim=True)
    worst_idx = torch.argmin(rewards, dim=1, keepdim=True)
    best_logp = logp.gather(1, best_idx).mean()
    worst_logp = logp.gather(1, worst_idx).mean()

    stats = {
        "loss": loss.detach(),
        "policy_loss": policy_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "best_logp": best_logp.detach(),
        "worst_logp": worst_logp.detach(),
        "best_minus_worst_logp": (best_logp - worst_logp).detach(),
        "reward_best": rewards.max(dim=1).values.mean().detach(),
        "reward_worst": rewards.min(dim=1).values.mean().detach(),
        "rel_mean": rels.mean().detach(),
        "rel_max": rels.max().detach(),
        "code_n": denom.view(B, G).mean().detach(),
    }
    return loss, stats


def reduce_scalar(x, device):
    if not torch.is_tensor(x):
        x = torch.tensor(float(x), device=device)
    else:
        x = x.to(device)
    if ddp_enabled():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x = x / dist.get_world_size()
    return x


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
    ap.add_argument("--grpo_end_t", type=int, default=100)
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--lambda_grpo", type=float, default=0.05)
    ap.add_argument("--grpo_beta", type=float, default=0.02)
    ap.add_argument("--mass_tol", type=float, default=0.0)
    ap.add_argument("--reward_l1_weight", type=float, default=0.0,
                    help="Optional decoded-field L1 reward weight. Default 0 keeps current behavior.")
    ap.add_argument("--reward_mse_weight", type=float, default=0.0,
                    help="Optional decoded-field MSE reward weight. Default 0 keeps current behavior.")

    ap.add_argument("--grpo_logp_reduction", type=str, default="mean", choices=["mean", "sum"],
                    help="Use mean or sum token log-prob for GRPO. 'sum' gives true sequence log-prob.")
    ap.add_argument("--grpo_kl_reduction", type=str, default="mean", choices=["mean", "sum"],
                    help="Use mean or sum token KL for the reference penalty.")
    ap.add_argument("--grpo_advantage_type", type=str, default="reward_norm",
                    choices=["reward_norm", "rank", "winner"],
                    help="Group advantage type. 'rank' is robust when reward magnitudes are tiny.")
    ap.add_argument("--grpo_code_only", action="store_true",
                    help="Apply GRPO logp/KL only on VQ code tokens 0 <= token < n_embed, not ROW/EOS/etc.")

    ap.add_argument("--r_min", type=float, default=0.0)
    ap.add_argument("--r_max", type=float, default=10.0)

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use_vq_ema_copy", action="store_true")
    ap.add_argument("--debug_one_update_direction", action="store_true",
                    help="Run one fixed-candidate GRPO update and report best-vs-worst logp before/after, then exit.")

    args = ap.parse_args()

    if args.num_generations < 2:
        raise ValueError("--num_generations should be >= 2 for group-relative advantages.")

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
            raise RuntimeError("This GRPO script assumes nodeck checkpoint with deck_len=0.")

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

        if args.debug_one_update_direction:
            if ddp_enabled():
                raise RuntimeError("--debug_one_update_direction should be run on 1 GPU, not DDP.")

            batch = next(iter(dl))
            deck, enc, din, dtgt, ts = batch
            enc = enc.to(device, non_blocking=True)
            dtgt = dtgt.to(device, non_blocking=True)
            deck_in = None

            G = int(args.num_generations)
            enc_rep = enc.repeat_interleave(G, dim=0)

            with torch.no_grad():
                seq_rep = sample_sequences(
                    model,
                    enc_rep,
                    None,
                    bos_id=bos_id,
                    dec_block=dec_block,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )

            model.train()
            loss_before, stats_before = compute_grpo_loss_on_fixed_sequences(
                rollout_mod=rollout_mod,
                model=model,
                ref_model=ref_model,
                vq=vq,
                enc=enc,
                dtgt=dtgt,
                seq_rep=seq_rep,
                deck_tokens=deck_in,
                n_embed=n_embed,
                Hq=Hq,
                Wq=Wq,
                bos_id=bos_id,
                num_generations=G,
                vmin=vmin,
                vmax=vmax,
                r_weights=r_weights,
                mass_tol=args.mass_tol,
                beta=args.grpo_beta,
                reward_l1_weight=args.reward_l1_weight,
                reward_mse_weight=args.reward_mse_weight,
                logp_reduction=args.grpo_logp_reduction,
                kl_reduction=args.grpo_kl_reduction,
                advantage_type=args.grpo_advantage_type,
                code_only=args.grpo_code_only,
            )

            opt.zero_grad(set_to_none=True)
            loss_before.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), float(args.grad_clip))
            opt.step()
            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                loss_after, stats_after = compute_grpo_loss_on_fixed_sequences(
                    rollout_mod=rollout_mod,
                    model=model,
                    ref_model=ref_model,
                    vq=vq,
                    enc=enc,
                    dtgt=dtgt,
                    seq_rep=seq_rep,
                    deck_tokens=deck_in,
                    n_embed=n_embed,
                    Hq=Hq,
                    Wq=Wq,
                    bos_id=bos_id,
                    num_generations=G,
                    vmin=vmin,
                    vmax=vmax,
                    r_weights=r_weights,
                    mass_tol=args.mass_tol,
                    beta=args.grpo_beta,
                    reward_l1_weight=args.reward_l1_weight,
                    reward_mse_weight=args.reward_mse_weight,
                    logp_reduction=args.grpo_logp_reduction,
                    kl_reduction=args.grpo_kl_reduction,
                    advantage_type=args.grpo_advantage_type,
                    code_only=args.grpo_code_only,
                )

            print("============================================================", flush=True)
            print("DEBUG ONE UPDATE DIRECTION", flush=True)
            print(f"reward_best             before={float(stats_before['reward_best'].cpu()):.6e} after={float(stats_after['reward_best'].cpu()):.6e}", flush=True)
            print(f"reward_worst            before={float(stats_before['reward_worst'].cpu()):.6e} after={float(stats_after['reward_worst'].cpu()):.6e}", flush=True)
            print(f"best_logp               before={float(stats_before['best_logp'].cpu()):.6e} after={float(stats_after['best_logp'].cpu()):.6e}", flush=True)
            print(f"worst_logp              before={float(stats_before['worst_logp'].cpu()):.6e} after={float(stats_after['worst_logp'].cpu()):.6e}", flush=True)
            print(f"best_minus_worst_logp   before={float(stats_before['best_minus_worst_logp'].cpu()):.6e} after={float(stats_after['best_minus_worst_logp'].cpu()):.6e}", flush=True)
            print(f"delta_best_minus_worst  {float((stats_after['best_minus_worst_logp'] - stats_before['best_minus_worst_logp']).cpu()):.6e}", flush=True)
            print(f"policy_loss             before={float(stats_before['policy_loss'].cpu()):.6e} after={float(stats_after['policy_loss'].cpu()):.6e}", flush=True)
            print(f"kl_loss                 before={float(stats_before['kl_loss'].cpu()):.6e} after={float(stats_after['kl_loss'].cpu()):.6e}", flush=True)
            print(f"code_n                  {float(stats_before['code_n'].cpu()):.1f}", flush=True)
            print("============================================================", flush=True)
            return

        rank_print(rank, f">> filtered dataset len={len(ds_tr)} batch_size_per_gpu={args.batch_size}")
        rank_print(rank, f">> effective selected examples per optimizer step = batch({args.batch_size}) * world({world_size}) * accum({args.grad_accum})")
        rank_print(rank, f">> GRPO timesteps: {args.grpo_start_t} <= t < {args.grpo_end_t}")
        rank_print(rank, f">> num_generations={args.num_generations} lambda_grpo={args.lambda_grpo} beta={args.grpo_beta} mass_tol={args.mass_tol}")
        rank_print(rank, f">> reward_l1_weight={args.reward_l1_weight} reward_mse_weight={args.reward_mse_weight}")
        rank_print(rank, f">> grpo_logp_reduction={args.grpo_logp_reduction} grpo_kl_reduction={args.grpo_kl_reduction} advantage_type={args.grpo_advantage_type} code_only={args.grpo_code_only}")
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

                grpo, grpo_stats = grpo_mass_loss(
                    rollout_mod=rollout_mod,
                    model=model,
                    ref_model=ref_model,
                    vq=vq,
                    enc=enc,
                    dtgt=dtgt,
                    deck_tokens=None,
                    n_embed=n_embed,
                    Hq=Hq,
                    Wq=Wq,
                    bos_id=bos_id,
                    dec_block=dec_block,
                    num_generations=args.num_generations,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    vmin=vmin,
                    vmax=vmax,
                    r_weights=r_weights,
                    mass_tol=args.mass_tol,
                    beta=args.grpo_beta,
                    reward_l1_weight=args.reward_l1_weight,
                    reward_mse_weight=args.reward_mse_weight,
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
                        rew_r = reduce_scalar(grpo_stats["reward_mean"], device)
                        rel_mean_r = reduce_scalar(grpo_stats["rel_err_mean"], device)
                        rel_max_r = reduce_scalar(grpo_stats["rel_err_max"], device)
                        kl_r = reduce_scalar(grpo_stats["kl_loss"], device)
                        field_l1_r = reduce_scalar(grpo_stats["field_l1_mean"], device)
                        field_mse_r = reduce_scalar(grpo_stats["field_mse_mean"], device)
                        policy_r = reduce_scalar(grpo_stats["policy_loss"], device)
                        logp_r = reduce_scalar(grpo_stats["logp_mean"], device)
                        adv_abs_r = reduce_scalar(grpo_stats["adv_abs_mean"], device)
                        rew_best_r = reduce_scalar(grpo_stats["reward_best_mean"], device)
                        rew_worst_r = reduce_scalar(grpo_stats["reward_worst_mean"], device)
                        best_logp_r = reduce_scalar(grpo_stats["best_logp_mean"], device)
                        worst_logp_r = reduce_scalar(grpo_stats["worst_logp_mean"], device)
                        bmw_logp_r = reduce_scalar(grpo_stats["best_minus_worst_logp"], device)
                        code_n_r = reduce_scalar(grpo_stats["code_token_count_mean"], device)

                        if is_rank0(rank):
                            dt = (time.time() - t0) / 60.0
                            print(
                                f"[{step:06d}] "
                                f"ce={float(ce_r.cpu()):.5f} "
                                f"loss={float(loss_r.cpu()):.5f} "
                                f"elapsed_min={dt:.1f}"
                                f" grpo={float(grpo_r.cpu()):.5f}"
                                f" rew={float(rew_r.cpu()):.5e}"
                                f" rel_mean={float(rel_mean_r.cpu()):.5e}"
                                f" rel_max={float(rel_max_r.cpu()):.5e}"
                                f" kl={float(kl_r.cpu()):.5e}"
                                f" field_l1={float(field_l1_r.cpu()):.5e}"
                                f" field_mse={float(field_mse_r.cpu()):.5e}"
                                f" policy={float(policy_r.cpu()):.5e}"
                                f" logp={float(logp_r.cpu()):.5e}"
                                f" adv_abs={float(adv_abs_r.cpu()):.5e}"
                                f" rew_best={float(rew_best_r.cpu()):.5e}"
                                f" rew_worst={float(rew_worst_r.cpu()):.5e}"
                                f" best_logp={float(best_logp_r.cpu()):.5e}"
                                f" worst_logp={float(worst_logp_r.cpu()):.5e}"
                                f" best_minus_worst_logp={float(bmw_logp_r.cpu()):.5e}"
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
                                grpo_mass_reward="r_weighted_avdensity_gt_t1_deadband",
                                mass_tol=float(args.mass_tol),
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
                grpo_mass_reward="r_weighted_avdensity_gt_t1_deadband",
                mass_tol=float(args.mass_tol),
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
