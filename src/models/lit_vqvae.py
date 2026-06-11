import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math
from copy import deepcopy

from src.modules.diffusionmodules.model import Encoder, Decoder
from src.modules.vqvae.quantize import VectorQuantizer2


class LitVQVAE(L.LightningModule):
    """
    Minimal VQ-VAE:
      x -> Encoder -> z (B, z_channels, Hz, Wz)
         -> 1x1 quant_conv -> (B, embed_dim, Hz, Wz)
         -> VectorQuantizer2 -> z_q + qloss
         -> 1x1 post_quant_conv -> (B, z_channels, Hz, Wz)
         -> Decoder -> x_rec (B, C, H, W)

    Loss:
      recon = L1(x_rec, x)  (optionally + small L2)
      total = recon + beta * qloss + grad_w * grad_loss
    """

    def __init__(
        self,
        ddconfig: dict,
        n_embed: int,
        embed_dim: int,
        learning_rate: float = 2e-4,
        beta: float = 0.25,              # commitment weight inside quantizer
        vq_loss_weight: float = 1.0,     # multiplier on qloss
        recon_l2_weight: float = 0.0,    # optional small stabilizer
        grad_loss_weight: float = 0.0,   # optional edge/gradient preservation
        l2_normalize_codebook: bool = False,
        legacy_beta_bug: bool = False,    # keep quantizer legacy behavior unless you want to fix
        output_clamp: bool = True,       # clamp x_rec into [-1,1] before loss
        image_key: str = "image",
        usage_entropy_weight: float = 0.03,
        dead_code_reset_every: int = 1000,
        dead_code_reset_threshold: float = 1.0,
        dead_code_reset_warmup_steps: int = 2000,
        max_dead_code_resets: int = 512,
        no_quant: bool = False,
        warmup_steps: int = 0,
        total_steps: int = 0,
        min_lr: float = 1e-6,
    ):
        super().__init__()

        ddconfig = deepcopy(ddconfig)
        for k in ("ch_mult", "attn_resolutions"):
            if k in ddconfig and ddconfig[k] is not None:
                ddconfig[k] = list(ddconfig[k])

        self.save_hyperparameters() 


        self.image_key = image_key
        self.learning_rate = float(learning_rate)

        # Encoder/Decoder from your diffusionmodules/model.py
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        # Quantizer expects channels = embed_dim
        z_channels = int(ddconfig["z_channels"])
        self.quant_conv = nn.Conv2d(z_channels, embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, kernel_size=1)

        # Plain VQ.
        self.quantize = VectorQuantizer2(
            n_e=n_embed,
            e_dim=embed_dim,
            beta=beta,
            remap=None,
            sane_index_shape=False,
            legacy=legacy_beta_bug,
            l2_normalize=l2_normalize_codebook,
        )

        self.vq_loss_weight = float(vq_loss_weight)
        self.recon_l2_weight = float(recon_l2_weight)
        self.grad_loss_weight = float(grad_loss_weight)
        self.output_clamp = bool(output_clamp)

        self.usage_entropy_weight = float(usage_entropy_weight)
        self.dead_code_reset_every = int(dead_code_reset_every)
        self.dead_code_reset_threshold = float(dead_code_reset_threshold)
        self.dead_code_reset_warmup_steps = int(dead_code_reset_warmup_steps)
        self.max_dead_code_resets = int(max_dead_code_resets)

        self.no_quant = bool(no_quant)

        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)

    # --------------------
    # IO helper: your dataset yields HWC in [-1,1]
    # --------------------
    def get_input(self, batch):
        x = batch[self.image_key]
        # batch["image"] is (B,H,W,C) in your dataset
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x.float()

    # --------------------
    # forward pieces
    # --------------------
    def encode_prequant(self, x):
        h = self.encoder(x)              # (B, z_channels, Hz, Wz)
        h = self.quant_conv(h)           # (B, embed_dim, Hz, Wz)
        return h


    def encode(self, x):
        h = self.encode_prequant(x)

        if self.no_quant:
            qloss = torch.zeros((), device=h.device, dtype=h.dtype)
            info = (None, None, None)
            return h, qloss, info

        z_q, qloss, info = self.quantize(h)
        return z_q, qloss, info


    def decode(self, z_q):
        z_q = self.post_quant_conv(z_q)  # (B, z_channels, Hz, Wz)
        x_rec = self.decoder(z_q)        # (B, C, H, W)
        return x_rec

    def forward(self, x):
        z_q, qloss, info = self.encode(x)
        x_rec = self.decode(z_q)
        return x_rec, qloss, info

    # --------------------
    # optional gradient loss
    # --------------------
    @staticmethod
    def grad_loss(x_rec, x):
        # x_rec, x: (B,C,H,W)
        dx_rec = x_rec[..., :, 1:] - x_rec[..., :, :-1]
        dx = x[..., :, 1:] - x[..., :, :-1]
        dy_rec = x_rec[..., 1:, :] - x_rec[..., :-1, :]
        dy = x[..., 1:, :] - x[..., :-1, :]
        return F.l1_loss(dx_rec, dx) + F.l1_loss(dy_rec, dy)

    # --------------------
    # training/validation
    # --------------------
    @staticmethod
    def codebook_usage_entropy_loss(inds: torch.Tensor, n_embed: int, eps: float = 1e-8):
        # inds: (N,) long in [0, n_embed)
        counts = torch.bincount(inds, minlength=n_embed).float()
        p = counts / (counts.sum() + eps)
        entropy = -(p * (p + eps).log()).sum()
        # maximize entropy => minimize negative entropy
        return -entropy, entropy

    def _ddp_global_counts(self, inds_1d: torch.Tensor) -> torch.Tensor:
        """
        inds_1d: (N,) long on GPU
        returns counts: (n_embed,) float on GPU, summed across all ranks
        """
        n_embed = self.quantize.n_e
        if self.trainer is None or self.trainer.world_size == 1:
            return torch.bincount(inds_1d, minlength=n_embed).float()

        gathered = self.all_gather(inds_1d)   # (world_size, N)
        all_inds = gathered.reshape(-1)
        return torch.bincount(all_inds, minlength=n_embed).float()

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch)

        # Keep pre-quantized encoder latents around for proper dead-code reset.
        h = self.encode_prequant(x)

        if self.no_quant:
            z_q = h
            qloss = torch.zeros((), device=h.device, dtype=h.dtype)
            info = (None, None, None)
        else:
            z_q, qloss, info = self.quantize(h)

        x_rec = self.decode(z_q)

        if self.output_clamp:
            x_rec = x_rec.clamp(-1, 1)

        recon_l1 = F.l1_loss(x_rec, x)
        loss = recon_l1

        recon_l2 = None
        if self.recon_l2_weight > 0:
            recon_l2 = F.mse_loss(x_rec, x)
            loss = loss + self.recon_l2_weight * recon_l2

        loss = loss + self.vq_loss_weight * qloss

        gl = None
        if self.grad_loss_weight > 0:
            gl = self.grad_loss(x_rec, x)
            loss = loss + self.grad_loss_weight * gl

        # codebook stats ONLY when quant is active
        if (not self.no_quant) and (info is not None) and (len(info) > 0) and (info[-1] is not None):
            inds = info[-1].reshape(-1).long()
            counts = self._ddp_global_counts(inds)
            p = counts / (counts.sum() + 1e-8)
            entropy = -(p * (p + 1e-8).log()).sum()
            perplexity = torch.exp(entropy)
            unique_global = (counts > 0).sum()

            if self.usage_entropy_weight > 0:
                loss = loss - self.usage_entropy_weight * entropy

        if (
            self.dead_code_reset_every > 0
            and self.global_step >= self.dead_code_reset_warmup_steps
            and (self.global_step % self.dead_code_reset_every == 0)
        ):
            dead = (counts <= self.dead_code_reset_threshold).nonzero(as_tuple=False).squeeze(1)

            if dead.numel() > 0:
                # Do not reset thousands of codes at once.
                if self.max_dead_code_resets > 0 and dead.numel() > self.max_dead_code_resets:
                    perm = torch.randperm(dead.numel(), device=dead.device)
                    dead = dead[perm[: self.max_dead_code_resets]]

                # Reset from pre-quantized encoder outputs, not from z_q.
                h_flat = h.detach().permute(0, 2, 3, 1).reshape(-1, h.shape[1])
                rand_idx = torch.randint(
                    0,
                    h_flat.shape[0],
                    (dead.numel(),),
                    device=h_flat.device,
                )

                with torch.no_grad():
                    self.quantize.embedding.weight.data[dead] = h_flat[rand_idx]

                if self.trainer is None or self.trainer.is_global_zero:
                    self.log(
                        "train/dead_codes_reset",
                        float(dead.numel()),
                        on_step=True,
                        on_epoch=False,
                        logger=True,
                    )

            if self.trainer is None or self.trainer.is_global_zero:
                self.log("train/codebook_unique_global", unique_global.float(), on_step=True, on_epoch=False)
                self.log("train/code_entropy_global", entropy, on_step=True, on_epoch=False)
                self.log("train/code_perplexity_global", perplexity, on_step=True, on_epoch=False)

        # logs
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/recon_l1", recon_l1, on_step=True, on_epoch=True)
        self.log("train/qloss", qloss, on_step=True, on_epoch=True)
        if recon_l2 is not None:
            self.log("train/recon_l2", recon_l2, on_step=True, on_epoch=True)
        if gl is not None:
            self.log("train/grad_loss", gl, on_step=True, on_epoch=True)

        return loss


    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch)
        x_rec, qloss, info = self(x)
        if self.output_clamp:
            x_rec = x_rec.clamp(-1, 1)

        recon_l1 = F.l1_loss(x_rec, x)
        recon_l2 = F.mse_loss(x_rec, x)
        val_loss = recon_l1 + self.vq_loss_weight * qloss

        self.log("val/loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/recon_l1", recon_l1, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/recon_l2", recon_l2, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/qloss", qloss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return val_loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))

        # No scheduler if total_steps not provided (keeps current fixed-LR behavior)
        if self.total_steps <= 0:
            return opt

        base_lr = self.learning_rate
        min_lr = self.min_lr
        warmup = max(0, self.warmup_steps)
        total = max(1, self.total_steps)

        # LambdaLR multiplies the optimizer's base LR by this factor
        min_factor = min_lr / base_lr

        def lr_lambda(step: int):
            if warmup > 0 and step < warmup:
                return float(step) / float(max(1, warmup))  # 0 -> 1
            # cosine decay from 1 -> min_factor
            progress = float(step - warmup) / float(max(1, total - warmup))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
            },
        }

