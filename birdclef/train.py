"""
Training logic: Stage 1 (Perch2 KD) and Stage 2 (supervised + SWA).

run_kd_stage(cfg, model, loader, device)
    → trains PretrainModel for cfg.pretrain_epochs
    → saves pretrained_backbone.pth
    → returns final cosine similarity

run_supervised_stage(cfg, model, loaders, device)
    → trains BirdCLEFModel for cfg.num_epochs
    → runs composite validation after each epoch
    → maintains SWA from cfg.swa_start_epoch
    → saves best_model.pth + swa_model.pth
    → returns history DataFrame

Waveform mixup (constant lambda=0.5) is applied in the training loop to the
raw waveforms before GPU mel conversion, consistent with all experiments from
exp16 onwards.

AMP ordering:
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)              ← must precede clip_grad_norm_
    clip_grad_norm_(model.parameters(), …)
    scaler.step(optimizer)
    scaler.update()
"""

from __future__ import annotations

import copy
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .model import BirdCLEFModel, PretrainModel, build_optimizer_with_llrd
from .transforms import MelTransform
from .utils import safe_copy_bn_buffers
from .validate import (LB_PROXY_SPECIES, save_oof_predictions, save_oof_unlabeled_sc,
                       validate_composite, validate_unlabeled_sc)


# ── Loss functions ────────────────────────────────────────────────────────────

class FocalLossBCE(nn.Module):
    """
    Sigmoid focal loss for multi-label classification.

    FL(p_t) = alpha_t * (1 - p_t)^gamma * BCE(logit, target)

    where p_t = sigmoid(logit) for positives, 1 - sigmoid(logit) for negatives,
    and alpha_t = alpha for positives, (1 - alpha) for negatives.

    Setting gamma=0 recovers plain BCE (weighted by alpha).
    Setting alpha=0.5 removes the positive/negative asymmetry (symmetric weighting).

    The modulating factor (1 - p_t)^gamma down-weights well-classified examples,
    which is particularly valuable with background augmentation: if the model
    correctly detects a real species in the background noise at high confidence,
    the focal term attenuates the gradient compared to plain BCE.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p   = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # p_t: probability of the true class
        p_t         = p * targets + (1.0 - p) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma
        # alpha_t: per-element alpha weighting
        alpha_t     = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * focal_weight * bce).mean()


class SoftAUCLoss(nn.Module):
    """
    Soft AUC loss — directly optimises the ranking metric (ROC-AUC).
    From BirdCLEF 2025 4th place solution (Dylan Liu, 0.918 public LB).

    Core idea: for every (positive, negative) prediction pair in the batch,
    penalise cases where the positive score does not exceed the negative score.
    Soft labels are supported: the per-element weight is (label - 0.5) for
    positives and (0.5 - label) for negatives, so uncertain labels contribute
    proportionally less.

    Accepts raw logits; sigmoid is applied internally.
    Uses F.softplus for numerical stability (avoids log+exp overflow at large
    |diff| values — tip from competition host Tom Denton).

    Important: CV score is typically *lower* with this loss than with BCE/Focal,
    but LB generalisation is much better — this is expected behaviour, not
    overfitting. The loss is structurally resistant to shortcut learning.

    bce_weight: weight of an auxiliary BCE calibration term (default 0.0).
        Without this, species that never appear as positives in training batches
        (e.g. Insecta sonotypes with 0 focal clips and sparse SC pseudo-labels)
        receive zero gradient from the ranking loss and collapse to near-zero
        predictions.  A value of 0.1 anchors their scale via negative supervision
        without meaningfully reducing ensemble diversity (Spearman shift <0.03).
    """

    def __init__(self, margin: float = 1.0, bce_weight: float = 0.0):
        super().__init__()
        self.margin     = margin
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds      = torch.sigmoid(logits)
        pos_mask   = targets > 0.5
        neg_mask   = targets < 0.5
        pos_preds  = preds[pos_mask]
        neg_preds  = preds[neg_mask]
        pos_labels = targets[pos_mask]
        neg_labels = targets[neg_mask]
        if pos_preds.numel() == 0 or neg_preds.numel() == 0:
            auc_loss = logits.sum() * 0.0   # zero with grad; avoids None grad issues
        else:
            pos_weights = pos_labels - 0.5                                # (N_pos,)
            neg_weights = 0.5 - neg_labels                                # (N_neg,)
            diff        = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0) # (N_pos, N_neg)
            loss_matrix = F.softplus(-diff * self.margin)                 # numerically stable
            weighted    = loss_matrix * pos_weights.unsqueeze(1) * neg_weights.unsqueeze(0)
            auc_loss    = weighted.mean()
        if self.bce_weight > 0.0:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
            return auc_loss + self.bce_weight * bce_loss
        return auc_loss


class LogitSoftAUCLoss(nn.Module):
    """
    Logit-space soft AUC loss — identical to the BirdCLEF 2025 4th place
    implementation (Dylan Liu).

    Key difference from SoftAUCLoss above: takes raw logits directly with NO
    sigmoid applied internally.  The pairwise diff is computed in logit space
    (unbounded), so loss → 0 for well-separated pairs.  SoftAUCLoss applies
    sigmoid first, producing probability diffs ∈ (−1, 1) with a floor of
    ~0.313/pair even for a perfect model.

    pos_weight / neg_weight: global multipliers on soft label weights.
    Default 1.0 matches 4th place.  Increasing pos_weight emphasises
    recall; increasing neg_weight emphasises precision.

    No BCE auxiliary term — 4th place used pure ranking loss.  If scale
    collapse is observed, add --soft_auc_bce_weight on the existing
    SoftAUCLoss instead, which already has a calibration term.
    """

    def __init__(self, margin: float = 1.0,
                 pos_weight: float = 1.0, neg_weight: float = 1.0,
                 bce_weight: float = 0.0):
        super().__init__()
        self.margin     = margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_mask   = targets > 0.5
        neg_mask   = targets < 0.5
        pos_logits = logits[pos_mask]
        neg_logits = logits[neg_mask]
        pos_labels = targets[pos_mask]
        neg_labels = targets[neg_mask]
        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            auc_loss = logits.sum() * 0.0   # zero with grad
        else:
            pos_weights = self.pos_weight * (pos_labels - 0.5)              # (N_pos,)
            neg_weights = self.neg_weight * (0.5 - neg_labels)              # (N_neg,)
            diff        = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0) # logit diff, unbounded
            # F.softplus is numerically stable; avoids float16 overflow at |diff|>11
            # that torch.log(1+torch.exp(-diff)) produces under AMP
            loss_matrix = F.softplus(-diff * self.margin)
            weighted    = loss_matrix * pos_weights.unsqueeze(1) * neg_weights.unsqueeze(0)
            auc_loss    = weighted.mean()
        if self.bce_weight > 0.0:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
            return auc_loss + self.bce_weight * bce_loss
        return auc_loss


class SoftCELoss(nn.Module):
    """
    Soft cross-entropy loss for multi-label / soft-label targets.

    Normalises the label vector to a probability distribution per sample, then
    computes -sum(p * log_softmax(logit), dim=1).mean().

    With one-hot targets this is exactly standard cross-entropy.
    With multi-hot or soft targets (mixup, SC pseudo-labels) the loss is a
    weighted mixture of per-class CE terms.

    Key property: log_softmax creates inter-class competition — pushing one
    class logit up forces all others down.  Unlike BCE (independent sigmoids),
    this pressure propagates through the attention head and forces it to find
    time steps where one species dominates rather than relying on positional
    shortcuts.

    At inference sigmoid is still used as usual; the train/infer asymmetry is
    intentional (Salman Ahmed, BirdCLEF+ 2026 discussion, 0.937 single model).
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        t = targets / targets.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return -(t * log_p).sum(dim=1).mean()


class SmoothAPLoss(nn.Module):
    """
    Smooth Average Precision loss for multi-label classification.

    Differentiable approximation to per-class AP (Brown et al., ECCV 2020),
    adapted for soft labels. Loss = 1 - mean_class_AP over active classes.

    Compared to SoftAUCLoss: AP is position-weighted (high-precision retrievals
    count more), while SoftAUCLoss is uniform pairwise. Optimising AP vs AUC
    gives structurally different gradient paths — useful as an ensemble diversity
    source when combined with a SoftAUCLoss model.

    Without a calibration term, SmoothAP can collapse: all logits drift upward
    since the ranking objective has no absolute-scale constraint. bce_weight adds
    a BCE term that anchors prediction scale and prevents this.

    Memory: O(B² × C). At B=32, C=234: ~9.7 MB per batch. Safe for typical
    GPU budgets; reduce batch_size if OOM.

    Args:
        tau:        sigmoid temperature (default 0.01). With BF16, tau < 0.05 can
                    cause the soft-rank sum to overflow, producing negative loss.
                    Use tau >= 0.1 when use_bf16=True.
        bce_weight: weight of the auxiliary BCE calibration term (default 0.0).
                    A value of 0.1 is a good starting point.
    """

    def __init__(self, tau: float = 0.01, bce_weight: float = 0.0):
        super().__init__()
        self.tau = tau
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C = logits.shape
        # (B, B, C) pairwise score differences
        diff  = logits.unsqueeze(1) - logits.unsqueeze(0)
        # Soft indicator: above[i,j,c] ≈ 1 if score_i > score_j for class c
        above = torch.sigmoid(diff / self.tau)
        # Soft rank of each sample (rank 1 = highest score, rank B = lowest)
        soft_rank = (B + 1) - above.sum(dim=1)                           # (B, C)
        # Soft count of positives ranked above each sample
        n_soft_above = (above * targets.unsqueeze(0)).sum(dim=1)         # (B, C)
        # Precision at each rank position (soft)
        prec_at = n_soft_above / soft_rank.clamp(min=1e-8)               # (B, C)
        # Weight by positive label confidence (pos_weight = label - 0.5 clamped to 0)
        pos_weight   = (targets - 0.5).clamp(min=0.0)                    # (B, C)
        total_weight = pos_weight.sum(dim=0)                              # (C,)
        # Per-class AP as weighted average of precision at each positive's rank
        class_ap = (prec_at * pos_weight).sum(dim=0) / total_weight.clamp(min=1e-8)
        active = total_weight > 0
        if not active.any():
            return logits.sum() * 0.0
        ap_loss = 1.0 - class_ap[active].mean()
        if self.bce_weight > 0.0:
            ap_loss = ap_loss + self.bce_weight * F.binary_cross_entropy_with_logits(
                logits, targets
            )
        return ap_loss


# ── Shared scheduler builder ──────────────────────────────────────────────────

def _build_scheduler(optimizer: optim.Optimizer, cfg: Config, n_epochs: int,
                     warmup: int) -> optim.lr_scheduler.LRScheduler:
    """
    Linear warmup then either:
      'cosine'    — single CosineAnnealingLR over remaining epochs.
                    eta_min=0 for ViT; 1e-5 for CNN.
      'cosine_wr' — CosineAnnealingWarmRestarts(T_0=cosine_restart_period).
                    Restarts LR to peak every T_0 epochs; eta_min=1e-6.
                    Matches 1st-place BirdCLEF 2025 sc_pl schedule (5e-4→1e-6,
                    restart every 5 epochs, same params as supervised training).
    """
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)

    if getattr(cfg, 'scheduler', 'cosine') == 'cosine_wr':
        cosine_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.cosine_restart_period, T_mult=1, eta_min=1e-6)
    else:
        eta_min = 0.0 if cfg.is_vit else 1e-5
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, n_epochs - warmup), eta_min=eta_min)

    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup])


# ── Stage 1: Perch2 knowledge distillation ───────────────────────────────────

def run_kd_stage(
    cfg: Config,
    pretrain_model: PretrainModel,
    distill_loader: DataLoader,
    mel_transform: MelTransform,
    device: torch.device,
    output_dir: Path,
) -> float:
    """
    Train pretrain_model with cosine distillation loss against Perch2 embeddings.

    Loss = 1 − mean cosine similarity (target embeddings are pre-L2-normalised).
    No SpecAugment: Perch2 embeddings were computed on unaugmented audio —
    augmenting the student input makes alignment harder without clear benefit.

    Saves pretrained_backbone.pth to output_dir.
    Returns final epoch mean cosine similarity.
    """
    pretrain_model.to(device)

    pretrain_optim = optim.AdamW(
        pretrain_model.parameters(),
        lr=cfg.pretrain_lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = _build_scheduler(
        pretrain_optim, cfg, cfg.pretrain_epochs, cfg.pretrain_warmup)
    # BF16 has FP32 dynamic range → no gradient scaling needed.
    # GradScaler must be disabled with BF16 (it is a no-op in newer PyTorch but
    # disabling explicitly avoids version-specific surprises).
    _amp_dtype = torch.bfloat16 if cfg.use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and not cfg.use_bf16)

    history = []
    stage_start = time.time()

    print('\n' + '=' * 60)
    print('Stage 1: Perch2 embedding distillation')
    print('=' * 60)

    for epoch in range(1, cfg.pretrain_epochs + 1):
        pretrain_model.train()
        total_loss = 0.0
        total      = 0
        ep_start   = time.time()

        for waves, target_embs in tqdm(distill_loader,
                                        desc=f'KD {epoch:02d}/{cfg.pretrain_epochs}',
                                        leave=False):
            waves, target_embs = waves.to(device), target_embs.to(device)

            with torch.no_grad():
                specs = mel_transform(waves, augment=False)

            pretrain_optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg.use_amp, dtype=_amp_dtype):
                student_embs = pretrain_model(specs)   # L2-normalised (B, 1536)
                loss = 1.0 - (student_embs * target_embs).sum(dim=1).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(pretrain_optim)
            torch.nn.utils.clip_grad_norm_(pretrain_model.parameters(), cfg.grad_clip)
            scaler.step(pretrain_optim)
            scaler.update()

            total_loss += loss.item() * len(waves)
            total      += len(waves)

        mean_loss    = total_loss / total
        cos_sim      = 1.0 - mean_loss
        current_lr   = pretrain_optim.param_groups[0]['lr']
        scheduler.step()
        ep_time = time.time() - ep_start

        history.append({
            'epoch': epoch, 'kd_loss': round(mean_loss, 6),
            'cos_sim': round(cos_sim, 4), 'lr': current_lr,
            'epoch_time_sec': round(ep_time, 1),
        })
        print(f'KD Epoch {epoch:02d}/{cfg.pretrain_epochs}  '
              f'loss={mean_loss:.4f}  cos_sim={cos_sim:.4f}  '
              f'lr={current_lr:.2e}  time={ep_time/60:.1f}min')

    total_time = time.time() - stage_start
    print(f'Stage 1 complete — total time: {total_time/3600:.2f} h')

    # Save backbone weights (projection head discarded)
    backbone_path = output_dir / 'pretrained_backbone.pth'
    torch.save(pretrain_model.backbone.state_dict(), backbone_path)
    print(f'Saved backbone → {backbone_path}')

    pd.DataFrame(history).to_csv(output_dir / 'kd_history.csv', index=False)

    return float(history[-1]['cos_sim'])


# ── Stage 2: Supervised training + SWA ───────────────────────────────────────

def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    mel_transform: MelTransform,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    cfg: Config,
    device: torch.device,
    amp_dtype: torch.dtype = torch.float16,
    nocall_augmenter=None,
) -> float:
    model.train()
    total_loss = 0.0
    total      = 0

    for waves, labels in tqdm(loader, desc='Train', leave=False):
        if nocall_augmenter is not None:
            waves = nocall_augmenter(waves)   # CPU, before GPU transfer
        waves, labels = waves.to(device), labels.to(device)

        # Waveform-domain mixup (constant lambda=0.5, union labels)
        if random.random() < cfg.mixup_prob:
            perm = torch.randperm(waves.size(0), device=device)
            if cfg.mixup_min_overlap > 0.0:
                # Only mix pairs whose non-zero audio regions overlap sufficiently.
                # With left padding, audio occupies [audio_start, T); overlap region
                # is [max(s_i, s_j), T). Pairs with insufficient overlap keep their
                # original waves and labels (no contamination from silent regions).
                T_s = waves.size(1)
                has_audio  = waves.abs() > 1e-6           # (B, T_samples)
                any_audio  = has_audio.any(dim=1)          # (B,)
                audio_start = has_audio.float().argmax(dim=1)  # first non-zero index
                audio_start[~any_audio] = T_s              # silent clips: no overlap possible
                overlap_len = (T_s - torch.maximum(audio_start, audio_start[perm])).clamp(min=0).float()
                mix_mask = (overlap_len / T_s) >= cfg.mixup_min_overlap  # (B,)
                waves  = torch.where(mix_mask.unsqueeze(1),
                                     0.5 * waves + 0.5 * waves[perm], waves)
                labels = torch.where(mix_mask.unsqueeze(1),
                                     torch.maximum(labels, labels[perm]), labels)
            else:
                waves  = 0.5 * waves + 0.5 * waves[perm]
                labels = torch.maximum(labels, labels[perm])

        with torch.no_grad():
            specs = mel_transform(waves, augment=True)
            if cfg.cons_weight > 0.0:
                specs2 = mel_transform(waves, augment=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=cfg.use_amp, dtype=amp_dtype):
            _need_att = cfg.att_entropy_weight > 0.0 or cfg.att_edge_weight > 0.0
            if _need_att:
                clipwise_logits, _, frame_logits, att_weights = model(specs, return_att=True)
            else:
                clipwise_logits, _, frame_logits = model(specs)
            loss = criterion(clipwise_logits, labels)
            if cfg.dual_loss_weight > 0.0:
                frame_max = frame_logits.max(dim=1).values   # (B, num_classes)
                loss = loss + cfg.dual_loss_weight * criterion(frame_max, labels)
            if _need_att:
                # att_weights: (B, T, num_classes), softmax-normalised over T
                T = att_weights.size(1)
                if cfg.att_entropy_weight > 0.0:
                    H = -(att_weights * (att_weights + 1e-8).log()).sum(dim=1)  # (B, C)
                    loss = loss + cfg.att_entropy_weight * (H / math.log(T)).mean()
                if cfg.att_edge_weight > 0.0:
                    loss = loss + cfg.att_edge_weight * (
                        att_weights[:, 0, :] + att_weights[:, -1, :]).mean()
            if cfg.cons_weight > 0.0:
                # Augmentation consistency: second forward pass with a fresh SpecAugment
                # draw on the same waveforms.  Stop-gradient on the first view so the
                # consistency signal flows only through the second branch and cannot
                # collapse both views toward zero.
                p1 = torch.sigmoid(clipwise_logits).detach()
                clipwise_logits2, _, _ = model(specs2)
                p2 = torch.sigmoid(clipwise_logits2)
                loss = loss + cfg.cons_weight * F.mse_loss(p2, p1)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(labels)
        total      += len(labels)

    return total_loss / total


def run_supervised_stage(
    cfg: Config,
    model: BirdCLEFModel,
    train_loader: DataLoader,
    val_sc_loader: DataLoader,
    val_focal_loader: DataLoader,
    mel_transform: MelTransform,
    sc_val_df: pd.DataFrame,
    focal_val_df: pd.DataFrame,
    device: torch.device,
    output_dir: Path,
    nocall_augmenter=None,
    proxy_class_indices=None,
    val_unlabeled_sc_loader: Optional[DataLoader] = None,
    unlabeled_sc_val_manifest: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, float, float, float]:
    """
    Run the supervised training stage with SWA.

    Args:
      proxy_class_indices: optional set of int class indices for LB_PROXY_SPECIES,
        used to compute val_lb_proxy each epoch.  Build it in train.py as:
        ``{species2idx[s] for s in LB_PROXY_SPECIES if s in species2idx}``

    Returns (history_df, best_val_auc, swa_val_auc, best_sc_auc).
    """
    model.to(device)
    num_classes = model.num_classes

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Staged unfreeze: when freeze_epochs > 0, the backbone is frozen for the
    # first N epochs so the randomly-initialised SED head can warm up before
    # backbone weights start moving.
    #
    # Both param groups are registered with the optimizer from the start so the
    # scheduler decays them in tandem.  The backbone simply receives no gradient
    # updates while requires_grad=False — its LR slot follows the same warmup +
    # cosine curve as the head, so the ratio backbone_lr/head_lr stays constant
    # throughout training.  At the unfreeze epoch we just flip requires_grad;
    # no optimizer or scheduler state changes are needed.
    #
    # Note: not combined with LLRD — orthogonal use-cases.
    _freeze_active = cfg.freeze_epochs > 0 and not cfg.use_llrd
    _unfreeze_lr   = cfg.unfreeze_lr if cfg.unfreeze_lr > 0 else cfg.lr

    if cfg.use_llrd:
        param_groups = build_optimizer_with_llrd(model, cfg)
        optimizer = optim.AdamW(param_groups)
    elif _freeze_active:
        for p in model.backbone.parameters():
            p.requires_grad = False
        _head_params = (
            list(model.fc.parameters()) +
            list(model.att_fc.parameters()) +
            list(model.bn.parameters())
        )
        # Both groups in the optimizer from the start so the scheduler tracks both.
        # Backbone receives no updates while frozen (no grad), but its LR decays
        # in step with the head, preserving the ratio at every epoch.
        optimizer = optim.AdamW([
            {'params': _head_params,                    'lr': cfg.lr},
            {'params': list(model.backbone.parameters()), 'lr': _unfreeze_lr},
        ], weight_decay=cfg.weight_decay)
        print(f'Backbone frozen for first {cfg.freeze_epochs} epoch(s); '
              f'will unfreeze at lr={_unfreeze_lr:.2e}  (head lr={cfg.lr:.2e})')
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler  = _build_scheduler(optimizer, cfg, cfg.num_epochs, cfg.warmup_epochs)
    _amp_dtype = torch.bfloat16 if cfg.use_bf16 else torch.float16
    scaler     = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and not cfg.use_bf16)
    if cfg.use_smooth_ap_loss:
        criterion = SmoothAPLoss(tau=cfg.smooth_ap_tau, bce_weight=cfg.smooth_ap_bce_weight)
        print(f'Loss: SmoothAP  tau={cfg.smooth_ap_tau}  bce_weight={cfg.smooth_ap_bce_weight}')
    elif cfg.use_logit_auc_loss:
        criterion = LogitSoftAUCLoss(margin=cfg.logit_auc_margin,
                                     pos_weight=cfg.logit_auc_pos_weight,
                                     neg_weight=cfg.logit_auc_neg_weight,
                                     bce_weight=cfg.logit_auc_bce_weight)
        print(f'Loss: LogitSoftAUC  margin={cfg.logit_auc_margin}  '
              f'pos_weight={cfg.logit_auc_pos_weight}  neg_weight={cfg.logit_auc_neg_weight}  '
              f'bce_weight={cfg.logit_auc_bce_weight}')
    elif cfg.use_soft_auc_loss:
        criterion = SoftAUCLoss(margin=cfg.soft_auc_margin, bce_weight=cfg.soft_auc_bce_weight)
        print(f'Loss: SoftAUC  margin={cfg.soft_auc_margin}  bce_weight={cfg.soft_auc_bce_weight}')
    elif cfg.use_ce_loss:
        criterion = SoftCELoss()
        print('Loss: SoftCE  (inter-class competition; sigmoid at inference)')
    elif cfg.use_focal_loss:
        criterion = FocalLossBCE(gamma=cfg.focal_gamma, alpha=cfg.focal_alpha)
        print(f'Loss: FocalBCE  gamma={cfg.focal_gamma}  alpha={cfg.focal_alpha}')
    else:
        criterion = nn.BCEWithLogitsLoss()
        print('Loss: BCEWithLogitsLoss')
    if nocall_augmenter is not None:
        print(f'Background aug: prob={cfg.bg_aug_prob}  '
              f'SNR=[{cfg.bg_aug_snr_min_db}, {cfg.bg_aug_snr_max_db}] dB')
    swa_model = AveragedModel(model)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_auc = 0.0
    best_epoch   = 0
    best_state:  Optional[Dict] = None
    best_sc_auc   = 0.0
    best_sc_epoch = 0
    best_sc_state: Optional[Dict] = None
    history:     List[Dict]     = []
    per_class_rows: List[Dict]  = []
    train_start = time.time()

    idx2species = {i: sp for i, sp in enumerate(
        [''] * num_classes)}   # placeholder; actual names populated in train.py
    # The training script passes per-class names via per_class_rows.

    for epoch in range(1, cfg.num_epochs + 1):
        epoch_start = time.time()
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        train_loss = _train_one_epoch(
            model, train_loader, mel_transform, optimizer, criterion, scaler, cfg, device,
            amp_dtype=_amp_dtype, nocall_augmenter=nocall_augmenter)

        (composite_auc, sc_auc, focal_auc, lb_proxy,
         sc_class_aucs, focal_class_aucs,
         _, _, _, _,
         att_vs_mean_gap, att_entropy) = validate_composite(
            model, val_sc_loader, val_focal_loader, device, num_classes, cfg.use_amp,
            proxy_class_indices=proxy_class_indices)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        if _freeze_active and epoch == cfg.freeze_epochs:
            for p in model.backbone.parameters():
                p.requires_grad = True
            backbone_lr = optimizer.param_groups[1]['lr']
            print(f'Backbone unfrozen after epoch {epoch}  '
                  f'(backbone lr={backbone_lr:.2e}, head lr={current_lr:.2e})')

        if epoch >= cfg.swa_start_epoch:
            swa_model.update_parameters(model)

        epoch_time = time.time() - epoch_start

        row = {
            'epoch':                      epoch,
            'train_loss':                 round(train_loss,    6),
            'val_sc_auc':                 round(sc_auc,        4),
            'val_focal_auc':              round(focal_auc,     4),
            'val_auc':                    round(composite_auc, 4),
            'val_lb_proxy':               round(lb_proxy, 4) if lb_proxy == lb_proxy else float('nan'),
            'val_att_entropy':            round(att_entropy, 4)     if att_entropy     == att_entropy     else float('nan'),
            'val_att_vs_mean_gap':        round(att_vs_mean_gap, 4) if att_vs_mean_gap == att_vs_mean_gap else float('nan'),
            'lr':                         current_lr,
            'epoch_time_sec':             round(epoch_time, 1),
        }
        history.append(row)

        for c_idx, auc_val in sc_class_aucs.items():
            per_class_rows.append({
                'epoch': epoch, 'class_idx': c_idx,
                'split': 'soundscape', 'auc': round(auc_val, 4),
            })
        for c_idx, auc_val in focal_class_aucs.items():
            per_class_rows.append({
                'epoch': epoch, 'class_idx': c_idx,
                'split': 'focal', 'auc': round(auc_val, 4),
            })

        improved_comp = '✓C' if composite_auc > best_val_auc else '  '
        improved_sc   = '✓S' if sc_auc > best_sc_auc else '  '
        if composite_auc > best_val_auc:
            best_val_auc = composite_auc
            best_epoch   = epoch
            best_state   = copy.deepcopy(model.state_dict())
            torch.save(best_state, output_dir / 'best_model.pth')
        if sc_auc > best_sc_auc:
            best_sc_auc   = sc_auc
            best_sc_epoch = epoch
            best_sc_state = copy.deepcopy(model.state_dict())
            torch.save(best_sc_state, output_dir / 'best_sc_model.pth')

        print(f'Epoch {epoch:02d}/{cfg.num_epochs} {improved_comp} {improved_sc}  '
              f'loss={train_loss:.4f}  sc={sc_auc:.4f}  composite={composite_auc:.4f}  '
              f'lr={current_lr:.2e}  time={epoch_time/60:.1f}min')

    total_time = time.time() - train_start
    print(f'\nBest composite AUC: {best_val_auc:.4f}  (epoch {best_epoch})')
    print(f'Best SC AUC:        {best_sc_auc:.4f}  (epoch {best_sc_epoch})')
    print(f'Total training time: {total_time/3600:.2f} h')

    # ── SWA finalisation ──────────────────────────────────────────────────────
    print('\nCopying BN running stats from trained model to SWA model...')
    safe_copy_bn_buffers(swa_model, model, verbose=True)

    (composite_swa, sc_swa, focal_swa, lb_proxy_swa,
     _, _, _, _, _, _, _, _) = validate_composite(
        swa_model, val_sc_loader, val_focal_loader, device, num_classes, cfg.use_amp,
        proxy_class_indices=proxy_class_indices)

    proxy_swa_str = f'  LB-proxy: {lb_proxy_swa:.4f}' if lb_proxy_swa == lb_proxy_swa else ''
    print(f'SWA composite AUC:   {composite_swa:.4f}{proxy_swa_str}')
    print(f'Best per-epoch AUC:  {best_val_auc:.4f}  (epoch {best_epoch})')

    if val_unlabeled_sc_loader is not None:
        _, unlabeled_swa_consistency, _, _ = validate_unlabeled_sc(
            swa_model, val_unlabeled_sc_loader, device, num_classes, cfg.use_amp)
        print(f'SWA unlabeled SC PL-consistency: {unlabeled_swa_consistency:.4f}')

    torch.save(swa_model.module.state_dict(), output_dir / 'swa_model.pth')
    print('Saved → swa_model.pth')

    # ── OOF predictions ───────────────────────────────────────────────────────
    print('\nSaving OOF predictions...')
    save_oof_predictions(
        swa_model, val_sc_loader, val_focal_loader,
        sc_val_df, focal_val_df, device, num_classes, cfg.use_amp,
        output_dir, tag='swa')
    if val_unlabeled_sc_loader is not None and unlabeled_sc_val_manifest is not None:
        save_oof_unlabeled_sc(
            swa_model, val_unlabeled_sc_loader, unlabeled_sc_val_manifest,
            device, num_classes, cfg.use_amp, output_dir, tag='swa')

    if best_state is not None:
        model.load_state_dict(best_state)
    save_oof_predictions(
        model, val_sc_loader, val_focal_loader,
        sc_val_df, focal_val_df, device, num_classes, cfg.use_amp,
        output_dir, tag='best')
    if val_unlabeled_sc_loader is not None and unlabeled_sc_val_manifest is not None:
        save_oof_unlabeled_sc(
            model, val_unlabeled_sc_loader, unlabeled_sc_val_manifest,
            device, num_classes, cfg.use_amp, output_dir, tag='best')

    if best_sc_state is not None:
        model.load_state_dict(best_sc_state)
    save_oof_predictions(
        model, val_sc_loader, val_focal_loader,
        sc_val_df, focal_val_df, device, num_classes, cfg.use_amp,
        output_dir, tag='best_sc')
    if val_unlabeled_sc_loader is not None and unlabeled_sc_val_manifest is not None:
        save_oof_unlabeled_sc(
            model, val_unlabeled_sc_loader, unlabeled_sc_val_manifest,
            device, num_classes, cfg.use_amp, output_dir, tag='best_sc')

    # ── Logging ───────────────────────────────────────────────────────────────
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(output_dir / 'epoch_history.csv', index=False)
    pd.DataFrame(per_class_rows).to_csv(output_dir / 'per_class_auc.csv', index=False)

    return hist_df, best_val_auc, float(composite_swa), best_sc_auc
