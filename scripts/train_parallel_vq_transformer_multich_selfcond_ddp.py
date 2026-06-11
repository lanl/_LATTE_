#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import math
import time
import argparse
import random
from functools import partial
from typing import Dict, Any

import numpy as np

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.registry import build_dataset_from_registry
from src.data.datasets.vq_token_parallel_multich_twostep import collate_parallel_vq_multich_twostep

from scripts.custom.train_parallel_vq_transformer_multich_ddp import ParallelVQTransformerMultiCh


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ddp_is_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def ddp_setup():
    if not ddp_is_enabled():
        return 0, 1, 0

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def ddp_cleanup():
    if ddp_is_enabled() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    return rank == 0


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def lr_at_step(base_lr, step, warmup_steps, total_steps, min_lr_ratio=0.2):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)

    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos)


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        targets.reshape(-1),
        ignore_index=int(pad_id),
    )


def save_ckpt(path, model, opt, scaler, step, best_val, args, extra):
    tmp = path + ".tmp"
    state = unwrap_model(model).state_dict()

    torch.save(
        {
            "model": state,
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": int(step),
            "best_val": float(best_val),
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
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--init_from", required=True, help="Clean pretrained multich checkpoint.")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--max_pairs", type=int, default=None)

    ap.add_argument("--batch_size", type=int, default=1, help="Per-GPU batch size.")
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--max_steps", type=int, default=40000)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=5000)

    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--min_lr_ratio", type=float, default=0.2)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--n_layer", type=int, default=22)
    ap.add_argument("--n_head", type=int, default=20)
    ap.add_argument("--n_embd", type=int, default=1280)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--no_dataset_emb", action="store_true")
    ap.add_argument("--no_channel_emb", action="store_true")
    ap.add_argument("--min_max_channels", type=int, default=8)

    ap.add_argument("--rollout_loss_weight", type=float, default=0.25)
    ap.add_argument(
        "--rollout_prob",
        type=float,
        default=1.0,
        help="Probability of using model-predicted t+1 as input for t+2 loss. Otherwise uses real t+1.",
    )

    ap.add_argument("--eval_batches", type=int, default=100)
    ap.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])

    ap.add_argument("--seed", type=int, default=1337)

    args = ap.parse_args()

    rank, world_size, local_rank = ddp_setup()

    try:
        os.makedirs(args.out_dir, exist_ok=True)
        seed_all(args.seed + rank)

        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        with open(args.datasets_json, "r") as f:
            reg = json.load(f)

        extra_train: Dict[str, Any] = {"seed": args.seed}
        extra_val: Dict[str, Any] = {"seed": args.seed}

        if args.max_pairs is not None:
            extra_train["max_pairs"] = int(args.max_pairs)

        ds_tr = build_dataset_from_registry(
            reg,
            name=args.dataset_name,
            split="train",
            extra_kwargs=extra_train,
        )
        ds_va = build_dataset_from_registry(
            reg,
            name=args.dataset_name,
            split="val",
            extra_kwargs=extra_val,
        )

        pad_id = int(ds_tr.pad_id)
        collate_fn = partial(collate_parallel_vq_multich_twostep, pad_id=pad_id)

        tr_sampler = DistributedSampler(
            ds_tr,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        ) if ddp_is_enabled() else None

        dl_tr = DataLoader(
            ds_tr,
            batch_size=args.batch_size,
            sampler=tr_sampler,
            shuffle=(tr_sampler is None),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=(args.num_workers > 0),
        )

        dl_va = None
        if is_rank0(rank):
            dl_va = DataLoader(
                ds_va,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
                collate_fn=collate_fn,
                persistent_workers=(args.num_workers > 0),
            )

        model = ParallelVQTransformerMultiCh(
            vocab_size=int(ds_tr.vocab_size),
            pad_id=int(ds_tr.pad_id),
            max_Hq=max(int(ds_tr.max_Hq), int(ds_va.max_Hq)),
            max_Wq=max(int(ds_tr.max_Wq), int(ds_va.max_Wq)),
            num_datasets=max(int(ds_tr.num_datasets), int(ds_va.num_datasets)),
            max_channels=max(
                int(ds_tr.max_channels),
                int(ds_va.max_channels),
                int(args.min_max_channels),
            ),
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            use_dataset_emb=not args.no_dataset_emb,
            use_channel_emb=not args.no_channel_emb,
        ).to(device)

        raw_model = model

        if is_rank0(rank):
            n_params = sum(p.numel() for p in raw_model.parameters())
            print(f">> params={n_params:,} ({n_params / 1e6:.2f}M)", flush=True)
            print(
                f">> train_len={len(ds_tr)} val_len={len(ds_va)} "
                f"vocab={ds_tr.vocab_size} n_embed={ds_tr.n_embed} "
                f"max_Hq={raw_model.max_Hq} max_Wq={raw_model.max_Wq} "
                f"num_datasets={raw_model.num_datasets} "
                f"max_channels={raw_model.max_channels} pad_id={pad_id}",
                flush=True,
            )
            print(
                f">> self-conditioning: rollout_loss_weight={args.rollout_loss_weight} "
                f"rollout_prob={args.rollout_prob}",
                flush=True,
            )

        # Load clean pretrained weights.
        init_ckpt = torch.load(args.init_from, map_location="cpu")
        missing, unexpected = raw_model.load_state_dict(init_ckpt["model"], strict=False)
        if is_rank0(rank):
            print(
                f">> init_from={args.init_from} missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )

        opt = torch.optim.AdamW(
            raw_model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )

        use_amp = torch.cuda.is_available() and args.precision != "fp32"
        amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and args.precision == "fp16"))

        step = 0
        best_val = float("inf")

        if args.resume is not None:
            ckpt = torch.load(args.resume, map_location="cpu")
            missing, unexpected = raw_model.load_state_dict(ckpt["model"], strict=False)

            if ckpt.get("optimizer") is not None:
                opt.load_state_dict(ckpt["optimizer"])
            if ckpt.get("scaler") is not None and scaler is not None:
                scaler.load_state_dict(ckpt["scaler"])

            step = int(ckpt.get("step", 0))
            best_val = float(ckpt.get("best_val", best_val))

            if is_rank0(rank):
                print(
                    f">> resume={args.resume} step={step} best_val={best_val} "
                    f"missing={len(missing)} unexpected={len(unexpected)}",
                    flush=True,
                )

        if ddp_is_enabled():
            model = torch.nn.parallel.DistributedDataParallel(
                raw_model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
        else:
            model = raw_model

        def move_batch(batch):
            return {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }

        def model_forward(tokens_in, batch):
            return model(
                tokens_in,
                batch["rows"],
                batch["cols"],
                batch["valid"],
                dataset_id=batch["dataset_id"],
                channels=batch["channels"],
            )

        @torch.no_grad()
        def evaluate():
            if not is_rank0(rank):
                return None

            unwrap_model(model).eval()

            clean_total = 0.0
            rollout_total = 0.0
            combined_total = 0.0
            n = 0

            for batch in dl_va:
                batch = move_batch(batch)

                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    logits1 = unwrap_model(model)(
                        batch["tokens_t"],
                        batch["rows"],
                        batch["cols"],
                        batch["valid"],
                        dataset_id=batch["dataset_id"],
                        channels=batch["channels"],
                    )
                    loss_clean = compute_loss(logits1, batch["tokens_t1"], pad_id=pad_id)

                    pred_t1 = torch.argmax(logits1[:, :, : int(ds_tr.n_embed)].float(), dim=-1)
                    pred_t1 = torch.where(batch["valid"], pred_t1, torch.full_like(pred_t1, pad_id))

                    logits2 = unwrap_model(model)(
                        pred_t1,
                        batch["rows"],
                        batch["cols"],
                        batch["valid"],
                        dataset_id=batch["dataset_id"],
                        channels=batch["channels"],
                    )
                    loss_rollout = compute_loss(logits2, batch["tokens_t2"], pad_id=pad_id)

                    combined = loss_clean + float(args.rollout_loss_weight) * loss_rollout

                clean_total += float(loss_clean.item())
                rollout_total += float(loss_rollout.item())
                combined_total += float(combined.item())
                n += 1

                if n >= args.eval_batches:
                    break

            unwrap_model(model).train()

            return {
                "clean": clean_total / max(1, n),
                "rollout": rollout_total / max(1, n),
                "combined": combined_total / max(1, n),
            }

        def make_extra():
            m = unwrap_model(model)
            return dict(
                dataset_name=str(args.dataset_name),
                datasets_json=str(args.datasets_json),
                vocab_size=int(ds_tr.vocab_size),
                n_embed=int(ds_tr.n_embed),
                pad_id=int(ds_tr.pad_id),
                bos_id=int(ds_tr.bos_id),
                eos_id=int(ds_tr.eos_id),
                row_id=int(ds_tr.row_id),
                max_Hq=int(m.max_Hq),
                max_Wq=int(m.max_Wq),
                num_datasets=int(m.num_datasets),
                max_channels=int(m.max_channels),
                model_type="parallel_vq_multichannel_selfcond_twostep",
                rollout_loss_weight=float(args.rollout_loss_weight),
                rollout_prob=float(args.rollout_prob),
            )

        model.train()
        t0 = time.time()
        epoch = 0

        while step < args.max_steps:
            if tr_sampler is not None:
                tr_sampler.set_epoch(epoch)

            for batch in dl_tr:
                batch = move_batch(batch)

                lr = lr_at_step(
                    args.lr,
                    step,
                    args.warmup_steps,
                    args.max_steps,
                    min_lr_ratio=args.min_lr_ratio,
                )
                for pg in opt.param_groups:
                    pg["lr"] = lr

                opt.zero_grad(set_to_none=True)

                # First forward: real t -> real t+1.
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    logits1 = model_forward(batch["tokens_t"], batch)
                    loss_clean = compute_loss(logits1, batch["tokens_t1"], pad_id=pad_id)

                # Make predicted t+1 input, detached.
                with torch.no_grad():
                    pred_t1 = torch.argmax(logits1[:, :, : int(ds_tr.n_embed)].float(), dim=-1)
                    pred_t1 = torch.where(batch["valid"], pred_t1, torch.full_like(pred_t1, pad_id))

                    if float(args.rollout_prob) < 1.0:
                        use_pred = torch.rand((pred_t1.size(0), 1), device=device) < float(args.rollout_prob)
                        rollout_input = torch.where(use_pred, pred_t1, batch["tokens_t1"])
                    else:
                        rollout_input = pred_t1

                # Backward clean first to avoid holding both graphs at once.
                scaler.scale(loss_clean).backward()

                loss_rollout_value = torch.tensor(0.0, device=device)

                if float(args.rollout_loss_weight) > 0.0:
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        logits2 = model_forward(rollout_input, batch)
                        loss_rollout = compute_loss(logits2, batch["tokens_t2"], pad_id=pad_id)
                        weighted_rollout = float(args.rollout_loss_weight) * loss_rollout

                    loss_rollout_value = loss_rollout.detach()
                    scaler.scale(weighted_rollout).backward()

                scaler.unscale_(opt)

                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(opt)
                scaler.update()

                if (step % args.eval_every) == 0:
                    if ddp_is_enabled():
                        dist.barrier()

                    val = evaluate()

                    if ddp_is_enabled():
                        dist.barrier()

                    if is_rank0(rank):
                        dt = (time.time() - t0) / 60.0
                        combined_val = float(val["combined"])

                        print(
                            f"[{step:>7}] lr={lr:.3e} "
                            f"train_clean={float(loss_clean.item()):.4f} "
                            f"train_rollout={float(loss_rollout_value.item()):.4f} "
                            f"val_clean={val['clean']:.4f} "
                            f"val_rollout={val['rollout']:.4f} "
                            f"val_combined={val['combined']:.4f} "
                            f"({dt:.1f} min)",
                            flush=True,
                        )

                        if combined_val < best_val:
                            best_val = combined_val
                            save_ckpt(
                                os.path.join(args.out_dir, "parallel_best.pt"),
                                model,
                                opt,
                                scaler,
                                step,
                                best_val,
                                args,
                                make_extra(),
                            )
                            print(">> saved parallel_best.pt", flush=True)

                if (step % args.save_every) == 0 and step > 0:
                    if ddp_is_enabled():
                        dist.barrier()

                    if is_rank0(rank):
                        save_ckpt(
                            os.path.join(args.out_dir, f"parallel_step{step}.pt"),
                            model,
                            opt,
                            scaler,
                            step,
                            best_val,
                            args,
                            make_extra(),
                        )
                        print(f">> saved parallel_step{step}.pt", flush=True)

                    if ddp_is_enabled():
                        dist.barrier()

                step += 1

                if step >= args.max_steps:
                    break

            epoch += 1

        if ddp_is_enabled():
            dist.barrier()

        if is_rank0(rank):
            save_ckpt(
                os.path.join(args.out_dir, "parallel_final.pt"),
                model,
                opt,
                scaler,
                step,
                best_val,
                args,
                make_extra(),
            )
            print(">> training complete", flush=True)

    finally:
        ddp_cleanup()


if __name__ == "__main__":
    main()
