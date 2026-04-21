"""DropTok model: a small ViT encoder with progressive attention-based
token dropping, plus a cross-attention decoder that reconstructs frozen
DINOv2 features at masked positions.

This file implements the original proposal (Banerjee, Bayiz, Athavale
2026) stripped to its essentials. The encoder processes DINOv2 patch
tokens; at chosen layers it drops tokens using a relevance score derived
from self-attention, and the decoder cross-attends from learnable
positional queries at masked grid locations to the surviving token set.

Three methods are exposed:

- ``gated``  -- the variable-length proposal method. Relevance scores
  are turned into binary keep/drop decisions via a learnable
  (alpha, tau) sigmoid gate and a Straight-Through Estimator. Dropped
  tokens are physically excluded from later attention via
  ``key_padding_mask``. A ``lambda_comp * N_surviving`` penalty
  pressures the model toward fewer tokens.

- ``progressive_learned`` -- the simpler fixed-K variant used for
  matched-K comparisons. After each drop layer, keep the top-K tokens
  by relational relevance.

- ``progressive_random`` -- control; random top-K at each drop stage.
  Same architecture and parameter count as progressive_learned.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

METHODS = {"gated", "progressive_learned", "progressive_random"}


@dataclass
class DropTokConfig:
    # selection method (see module docstring)
    method: str = "gated"

    # patch grid / teacher
    grid_h: int = 16
    grid_w: int = 16
    d_teacher: int = 768           # DINOv2-base patch feature dim
    image_size: int = 224

    # encoder
    d_model: int = 256
    n_enc_layers: int = 4
    n_heads: int = 4
    enc_ffn: int = 512
    mask_ratio: float = 0.75       # fraction of INPUT tokens replaced with [MASK]

    # drop stages (1-indexed layer after which to drop)
    drop_after_layers: Tuple[int, ...] = (2, 3)

    # ---- fixed-K variants only ----
    # tokens kept after each drop stage; len must == len(drop_after_layers)
    K_total: int = 64
    drop_schedule: Optional[List[int]] = None   # if None: derived from K_total

    # ---- decoder ----
    decoder_layers: int = 2
    decoder_heads: int = 4
    decoder_ffn: int = 512

    # ---- gated (STE) method only ----
    # lambda * mean(N_surviving) added to loss. 0 = no compression pressure.
    lambda_comp: float = 0.0
    # gate_i = sigmoid(alpha * (r_i - tau)). Initialised to produce gates
    # near 1.0 on the log-space relevance scale (see _relational_relevance).
    gate_alpha_init: float = 2.0
    gate_tau_init: float = -7.0
    # When True, training adds Logistic(0,1) noise for Gumbel-sigmoid
    # sampling. False (default) = deterministic STE, no train/eval mismatch.
    gate_gumbel: bool = False
    gate_temp: float = 1.0

    # ---- reconstruction target ----
    # "dino":  predict DINOv2 features, cosine loss. Cheapest when running
    #          from a pre-computed teacher cache with no augmentation.
    # "pixel": predict normalised RGB patch, MSE loss. Preferred when
    #          training with data augmentation (pixels actually vary per
    #          augmented view) -- see README.
    recon_target: str = "dino"


def default_drop_schedule(K: int, n_stages: int = 2, N_total: int = 256) -> List[int]:
    """Geometric schedule from N_total down to K over n_stages drops."""
    ratio = (K / N_total) ** (1.0 / n_stages)
    sched = []
    cur = N_total
    for _ in range(n_stages):
        cur = max(K, int(round(cur * ratio)))
        sched.append(cur)
    sched[-1] = K
    return sched


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _softplus_inverse(y: float) -> float:
    return math.log(max(math.expm1(y), 1e-6))


def _topk_keep(scores: torch.Tensor, n_keep: int) -> torch.Tensor:
    """Return sorted indices of top-n_keep entries along last dim."""
    if n_keep >= scores.size(-1):
        return (torch.arange(scores.size(-1), device=scores.device)
                .unsqueeze(0).expand(scores.size(0), -1))
    _, idx = scores.topk(n_keep, dim=-1)
    idx, _ = idx.sort(dim=-1)
    return idx


def _random_keep(B: int, N: int, n_keep: int, device) -> torch.Tensor:
    return _topk_keep(torch.rand(B, N, device=device), n_keep)


def _gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def _gather_positions(pos: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return pos.gather(1, idx)


def _relational_relevance(attn: torch.Tensor, log_space: bool = False) -> torch.Tensor:
    """r_i = max_{j != i} attn[j, i]. attn: (B, N, N) averaged over heads.

    log_space=True returns log(r + eps), which approximates the
    proposal's raw-logit scale r_i = max_j q_j.k_i / sqrt(d). The raw
    post-softmax weights are in [0, 1] and tiny for large N (~1/N at
    init), which saturates a sigmoid gate; log-space gives a wider
    dynamic range.
    """
    B, N, _ = attn.shape
    eye = torch.eye(N, device=attn.device, dtype=torch.bool).unsqueeze(0)
    masked = attn.masked_fill(eye, float("-inf"))
    r = masked.max(dim=1).values
    return torch.log(r.clamp_min(1e-8)) if log_space else r


def _ensure_min_keep(alive: torch.Tensor, logit: torch.Tensor,
                     min_keep: int = 1) -> torch.Tensor:
    """Force >= min_keep tokens alive per sample so attention never sees
    an all-padded query row."""
    n_alive = alive.sum(dim=1, keepdim=True)
    need = n_alive < min_keep
    if not need.any():
        return alive
    topk_idx = logit.topk(min_keep, dim=1).indices
    rescue = torch.zeros_like(alive)
    rescue.scatter_(1, topk_idx, True)
    return alive | (rescue & need)


# --------------------------------------------------------------------------- #
# Transformer block (optionally returns averaged attention weights)
# --------------------------------------------------------------------------- #

class Block(nn.Module):
    def __init__(self, d_model: int, nhead: int, ffn: int, dropout: float = 0.0):
        super().__init__()
        self.nhead = nhead
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn),
            nn.GELU(),
            nn.Linear(ffn, d_model),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False,
                key_padding_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y = self.norm1(x)
        y, attn = self.attn(y, y, y, key_padding_mask=key_padding_mask,
                            need_weights=return_attn, average_attn_weights=True)
        x = x + y
        x = x + self.ffn(self.norm2(x))
        return x, attn


# --------------------------------------------------------------------------- #
# Decoder
# --------------------------------------------------------------------------- #

class MaskedFeatureDecoder(nn.Module):
    """Cross-attention decoder: learnable positional queries at MASKED
    grid positions cross-attend to the surviving encoder tokens and
    predict the target at each masked position.

    Output dim is configurable via ``d_out`` so the same decoder can
    predict DINO features (d_teacher dims, cosine loss) or raw RGB
    patches (3*patch_h*patch_w dims, MSE loss).
    """

    def __init__(self, grid_h: int, grid_w: int, d_model: int,
                 d_out: int, num_layers: int = 2, nhead: int = 4,
                 dim_feedforward: int = 512, dropout: float = 0.0):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.d_out = d_out
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.pos = nn.Parameter(torch.randn(grid_h * grid_w, d_model) * 0.02)
        self.out = nn.Linear(d_model, d_out)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor,
                patch_mask: torch.Tensor) -> torch.Tensor:
        """tokens: (B, K, D). token_mask: (B, K) True=valid.
        patch_mask: (B, H, W) True = MASKED position to reconstruct.
        Returns (sum_b M_b, d_out), rows concatenated per batch element
        in patch_mask iteration order.
        """
        B = tokens.size(0)
        H, W = self.grid_h, self.grid_w
        device = tokens.device

        mask_flat = patch_mask.reshape(B, H * W)
        counts = mask_flat.sum(-1)
        Mmax = int(counts.max().item())
        if Mmax == 0:
            return tokens.new_zeros((0, self.d_out))

        pos_all = self.pos.unsqueeze(0).expand(B, -1, -1)
        q = tokens.new_zeros(B, Mmax, tokens.size(-1))
        q_mask = torch.zeros(B, Mmax, dtype=torch.bool, device=device)
        for b in range(B):
            idx = mask_flat[b].nonzero(as_tuple=False).squeeze(-1)
            m_b = idx.numel()
            if m_b == 0:
                continue
            q[b, :m_b] = pos_all[b, idx]
            q_mask[b, :m_b] = True

        y = self.decoder(
            tgt=q, memory=tokens,
            tgt_key_padding_mask=~q_mask,
            memory_key_padding_mask=~token_mask,
        )
        pred = self.out(y)

        parts = []
        for b in range(B):
            m_b = int(counts[b].item())
            if m_b == 0:
                continue
            parts.append(pred[b, :m_b])
        return torch.cat(parts, dim=0)


# --------------------------------------------------------------------------- #
# Top-level model
# --------------------------------------------------------------------------- #

class DropTok(nn.Module):
    def __init__(self, cfg: DropTokConfig):
        super().__init__()
        assert cfg.method in METHODS, f"unknown method {cfg.method}"
        self.cfg = cfg
        N = cfg.grid_h * cfg.grid_w

        # ---- encoder input ----
        self.in_proj = nn.Linear(cfg.d_teacher, cfg.d_model)
        self.mask_token = nn.Parameter(torch.randn(cfg.d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(N, cfg.d_model) * 0.02)

        # ---- encoder blocks ----
        self.blocks = nn.ModuleList([
            Block(cfg.d_model, cfg.n_heads, cfg.enc_ffn)
            for _ in range(cfg.n_enc_layers)
        ])

        # ---- gated-method parameters ----
        if cfg.method == "gated":
            n_stages = len(cfg.drop_after_layers)
            raw_alpha = _softplus_inverse(cfg.gate_alpha_init)
            self.gate_alpha_raw = nn.Parameter(torch.full((n_stages,), raw_alpha))
            self.gate_tau = nn.Parameter(torch.full((n_stages,), cfg.gate_tau_init))
        else:
            self.register_parameter("gate_alpha_raw", None)
            self.register_parameter("gate_tau", None)

        # ---- decoder ----
        self.patch_h = cfg.image_size // cfg.grid_h
        self.patch_w = cfg.image_size // cfg.grid_w
        d_out = (3 * self.patch_h * self.patch_w
                 if cfg.recon_target == "pixel" else cfg.d_teacher)
        self.decoder = MaskedFeatureDecoder(
            grid_h=cfg.grid_h, grid_w=cfg.grid_w,
            d_model=cfg.d_model, d_out=d_out,
            num_layers=cfg.decoder_layers, nhead=cfg.decoder_heads,
            dim_feedforward=cfg.decoder_ffn,
        )

        # ---- fixed-K drop schedule ----
        if cfg.method.startswith("progressive"):
            self.drop_sched = (cfg.drop_schedule
                               or default_drop_schedule(
                                   cfg.K_total, len(cfg.drop_after_layers), N))
            assert len(self.drop_sched) == len(cfg.drop_after_layers), (
                f"drop_schedule {self.drop_sched} must match drop_after_layers "
                f"{cfg.drop_after_layers}"
            )
        else:
            self.drop_sched = None

    # ------------------------------------------------------------------ #
    def _build_input(self, teacher: torch.Tensor, patch_mask: torch.Tensor
                     ) -> torch.Tensor:
        """(B,H,W,d_teacher) + (B,H,W) mask -> (B, N, d_model)."""
        B, H, W, _ = teacher.shape
        N = H * W
        flat = teacher.reshape(B, N, -1)
        m = patch_mask.reshape(B, N).unsqueeze(-1).float()
        x = self.in_proj(flat)
        x = x * (1.0 - m) + self.mask_token[None, None, :] * m
        return x + self.pos_embed[None, :, :]

    # ------------------------------------------------------------------ #
    def forward(self, teacher: torch.Tensor, patch_mask: torch.Tensor) -> dict:
        cfg = self.cfg
        B, H, W, _ = teacher.shape
        N = H * W
        device = teacher.device

        x = self._build_input(teacher, patch_mask)
        positions = (torch.arange(N, device=device)
                     .unsqueeze(0).expand(B, -1).clone())

        drop_after = set(cfg.drop_after_layers)
        drop_step = 0
        stage_masks: List[torch.Tensor] = []

        # gated: maintain alive mask over the ORIGINAL N positions.
        alive: Optional[torch.Tensor] = (
            torch.ones(B, N, dtype=torch.bool, device=device)
            if cfg.method == "gated" else None
        )

        for i, block in enumerate(self.blocks, start=1):
            want_attn = (i in drop_after) and (
                cfg.method in ("progressive_learned", "gated"))
            kp = (~alive) if cfg.method == "gated" else None
            x, attn = block(x, return_attn=want_attn, key_padding_mask=kp)

            if i not in drop_after:
                continue

            if cfg.method == "gated":
                # relevance on log(max softmax) scale -- approximates
                # proposal's pre-softmax logit.
                r = _relational_relevance(attn, log_space=True)          # (B, N)
                alpha_i = F.softplus(self.gate_alpha_raw[drop_step]) + 1e-3
                tau_i = self.gate_tau[drop_step]
                logit = alpha_i * (r - tau_i)
                logit = logit.masked_fill(~alive, -1e9)   # no resurrection

                if self.training and cfg.gate_gumbel:
                    u = torch.rand_like(logit).clamp(1e-6, 1 - 1e-6)
                    gumbel = torch.log(u) - torch.log1p(-u)
                    y_soft = torch.sigmoid((logit + gumbel)
                                           / max(float(cfg.gate_temp), 1e-3))
                else:
                    y_soft = torch.sigmoid(logit)
                y_hard = (y_soft > 0.5).float()
                # Straight-Through: forward = hard, backward = soft.
                m = y_hard + (y_soft - y_soft.detach())

                x = x * m.unsqueeze(-1)                   # zero dropped features
                alive = alive & (y_hard > 0.5)
                alive = _ensure_min_keep(alive, logit, min_keep=1)
                stage_masks.append(m)
                drop_step += 1
                continue

            # --- fixed-K drop ---
            n_keep = self.drop_sched[drop_step]
            if cfg.method == "progressive_learned":
                r = _relational_relevance(attn)
                idx = _topk_keep(r, n_keep)
            else:   # progressive_random
                idx = _random_keep(B, x.size(1), n_keep, device)
            x = _gather_tokens(x, idx)
            positions = _gather_positions(positions, idx)
            drop_step += 1

        # ---- decode ----
        if cfg.method == "gated":
            pred = self.decoder(x, alive, patch_mask)
            final_m = stage_masks[-1] if stage_masks else torch.ones(
                B, N, device=device)
            out = {
                "pred": pred,
                "alive": alive,
                "stage_masks": stage_masks,
                "n_surviving_soft": final_m.sum(dim=1).mean(),
                "n_surviving_hard": alive.float().sum(dim=1).mean(),
                "n_surviving_per_image": alive.float().sum(dim=1),
            }
        else:
            token_mask = torch.ones(B, x.size(1), dtype=torch.bool, device=device)
            pred = self.decoder(x, token_mask, patch_mask)
            out = {
                "pred": pred,
                "final_positions": positions,
                "n_surviving_hard": torch.tensor(
                    float(x.size(1)), device=device),
            }
        return out

    # ------------------------------------------------------------------ #
    def _build_target(self, teacher: torch.Tensor, patch_mask: torch.Tensor,
                      img: Optional[torch.Tensor]) -> torch.Tensor:
        """Per-masked-position target, batched in the same order the
        decoder produces its predictions."""
        cfg = self.cfg
        B, H, W, Dt = teacher.shape
        mflat = patch_mask.reshape(B, H * W)

        if cfg.recon_target == "dino":
            flat = teacher.reshape(B, H * W, Dt)
            parts = [flat[b, mflat[b].nonzero(as_tuple=False).squeeze(-1)]
                     for b in range(B) if mflat[b].any()]
            return (torch.cat(parts, dim=0) if parts
                    else flat.new_zeros((0, Dt)))

        assert img is not None, "recon_target='pixel' requires img"
        ph, pw = self.patch_h, self.patch_w
        patches = img.reshape(B, 3, H, ph, W, pw)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(B, H * W, 3 * ph * pw)
        parts = [patches[b, mflat[b].nonzero(as_tuple=False).squeeze(-1)]
                 for b in range(B) if mflat[b].any()]
        return (torch.cat(parts, dim=0) if parts
                else patches.new_zeros((0, 3 * ph * pw)))

    # ------------------------------------------------------------------ #
    def compute_loss(self, teacher: torch.Tensor, patch_mask: torch.Tensor,
                     img: Optional[torch.Tensor] = None):
        cfg = self.cfg
        out = self.forward(teacher, patch_mask)
        pred = out["pred"]
        target = self._build_target(teacher, patch_mask, img)

        if pred.numel() == 0 or target.numel() == 0:
            return pred.new_zeros(()), out, target

        if cfg.recon_target == "dino":
            recon = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
        else:
            recon = F.mse_loss(pred, target)

        loss = recon
        if cfg.method == "gated" and cfg.lambda_comp > 0.0:
            loss = loss + cfg.lambda_comp * out["n_surviving_soft"]
        return loss, out, target
