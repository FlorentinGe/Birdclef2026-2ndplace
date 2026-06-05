"""
Validation: composite AUC and OOF prediction saving.

validate_composite() implements the two-part composite metric:
  composite = (sc_mean * n_sc + focal_mean * n_focal) / (n_sc + n_focal)

Per-class AUC is computed with a per-class loop (not sklearn average='macro')
to correctly handle constant columns (all-zero ground truth for a class in the
val set).  This exactly matches the competition metric which skips classes with
no true positive labels.

LB_PROXY_SPECIES is a set of 11 ebird codes whose per-class AUC correlates most
strongly with the public leaderboard (Pearson r=0.69 over 20 experiments,
vs r=0.16 for the full 15-species soundscape mean).  Pass proxy_class_indices
(a set of integer class indices for these species) to validate_composite to
have lb_proxy computed automatically each epoch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set, Tuple

# Species whose composite AUC correlates most with LB (derived April 2026).
# Positive correlators: Solitary Cacique, Purplish Jay, Chestnut-vented Conebill,
# Black-tailed Tityra, Mato Grosso Snouted Tree Frog, Yellow-headed Caracara,
# Rusty-margined Flycatcher, Short-tailed Nighthawk, Hooded Capuchin,
# Bahia Dwarf Frog, Barred Forest-Falcon.
LB_PROXY_SPECIES: list[str] = [
    'sobcac1', 'purjay1', 'chvcon1', 'blttit1', '24321',
    'yehcar1', 'rumfly1', 'shtnig1', '516975', '23154', 'baffal1',
]

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm


# ── Per-loader AUC computation ────────────────────────────────────────────────

@torch.inference_mode()
def _compute_auc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_amp: bool,
    desc: str = 'Val',
    compute_diagnostics: bool = False,
) -> Tuple[Dict[int, float], np.ndarray, np.ndarray, Optional[Dict[int, float]], float]:
    """
    Run model on `loader`, compute per-class AUC.

    When compute_diagnostics=True:
      - also scores using clipwise_logits (frame mean, no attention)
      - computes mean Shannon entropy of att_weights over (samples × classes)
      These are Diagnostics 1 and 2 for attention quality monitoring.

    Returns:
      class_aucs      : dict class_index → AUC using att_clipwise (≥1 positive only)
      all_probs       : (N, num_classes) float32 probabilities (att_clipwise)
      all_labels      : (N, num_classes) float32 ground-truth
      mean_class_aucs : same dict but using clipwise_logits (mean); None if not compute_diagnostics
      att_entropy     : mean Shannon entropy of att_weights (nats); nan if not compute_diagnostics
    """
    model.eval()
    all_probs:      list[np.ndarray] = []
    all_probs_mean: list[np.ndarray] = []
    all_labels:     list[np.ndarray] = []
    entropy_sum = 0.0
    entropy_n   = 0

    for imgs, label_vecs in tqdm(loader, desc=desc, leave=False):
        imgs = imgs.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            if compute_diagnostics:
                clipwise_logits, att_clipwise, _, att_weights = model(imgs, return_att=True)
                # att_weights: (B, T, num_classes), softmax-normalised over T
                w = att_weights.float()
                H = -(w * (w + 1e-8).log()).sum(dim=1)   # (B, num_classes)
                entropy_sum += H.mean().item() * imgs.size(0)
                entropy_n   += imgs.size(0)
                all_probs_mean.append(
                    torch.sigmoid(clipwise_logits).float().cpu().numpy())
            else:
                _, att_clipwise, _ = model(imgs)
        probs = torch.sigmoid(att_clipwise).float().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(label_vecs.numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    class_aucs: Dict[int, float] = {}
    for c in range(num_classes):
        if all_labels[:, c].sum() == 0:
            continue
        try:
            class_aucs[c] = float(roc_auc_score(all_labels[:, c], all_probs[:, c]))
        except ValueError:
            pass

    mean_class_aucs: Optional[Dict[int, float]] = None
    if compute_diagnostics and all_probs_mean:
        all_probs_mean_arr = np.concatenate(all_probs_mean, axis=0)
        mean_class_aucs = {}
        for c in range(num_classes):
            if all_labels[:, c].sum() == 0:
                continue
            try:
                mean_class_aucs[c] = float(
                    roc_auc_score(all_labels[:, c], all_probs_mean_arr[:, c]))
            except ValueError:
                pass

    att_entropy = entropy_sum / entropy_n if entropy_n > 0 else float('nan')
    return class_aucs, all_probs, all_labels, mean_class_aucs, att_entropy


# ── Composite validation ──────────────────────────────────────────────────────

@torch.inference_mode()
def validate_composite(
    model: nn.Module,
    val_sc_loader: DataLoader,
    val_focal_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_amp: bool,
    proxy_class_indices: Optional[Set[int]] = None,
) -> Tuple[float, float, float, float,
           Dict[int, float], Dict[int, float],
           np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           float, float]:
    """
    Compute two-part composite AUC plus attention quality diagnostics.

    Args:
      proxy_class_indices: optional set of integer class indices corresponding
        to LB_PROXY_SPECIES.  When provided, lb_proxy is the mean AUC over
        those classes (pooling soundscape and focal splits).  Pass the result
        of ``{species2idx[sp] for sp in LB_PROXY_SPECIES if sp in species2idx}``
        from the training script.

    Returns:
      (composite, sc_mean, focal_mean, lb_proxy,
       sc_class_aucs, focal_class_aucs,
       sc_probs, sc_labels, focal_probs, focal_labels,
       att_vs_mean_gap, att_entropy)

      lb_proxy is NaN when proxy_class_indices is None or no proxy species
      appear in either val set.

      att_vs_mean_gap: SC AUC(att_clipwise) − SC AUC(clipwise_mean) averaged
        over classes present in SC val; positive = attention adds value.
      att_entropy: mean Shannon entropy (nats) of att_weights over SC val
        samples and classes; low = selective, high (→ log T) = decorative.
    """
    # SC: full diagnostics (Diagnostic 1 + 2)
    (sc_class_aucs, sc_probs, sc_labels,
     sc_mean_class_aucs, att_entropy) = _compute_auc(
        model, val_sc_loader, device, num_classes, use_amp, 'Val-SC  ',
        compute_diagnostics=True)
    # Focal: no diagnostics needed (focal AUC is not used for attention analysis)
    (focal_class_aucs, focal_probs, focal_labels,
     _, _) = _compute_auc(
        model, val_focal_loader, device, num_classes, use_amp, 'Val-Focal')

    sc_aucs    = list(sc_class_aucs.values())
    focal_aucs = list(focal_class_aucs.values())
    sc_mean    = float(np.mean(sc_aucs))    if sc_aucs    else 0.0
    focal_mean = float(np.mean(focal_aucs)) if focal_aucs else 0.0
    n_sc, n_focal = len(sc_aucs), len(focal_aucs)
    composite = (
        (sc_mean * n_sc + focal_mean * n_focal) / (n_sc + n_focal)
        if (n_sc + n_focal) > 0 else 0.0
    )

    # LB proxy: mean AUC over the 11 high-correlation species (sc + focal pooled)
    lb_proxy = float('nan')
    if proxy_class_indices:
        proxy_aucs = []
        for c in proxy_class_indices:
            if c in sc_class_aucs:
                proxy_aucs.append(sc_class_aucs[c])
            if c in focal_class_aucs:
                proxy_aucs.append(focal_class_aucs[c])
        if proxy_aucs:
            lb_proxy = float(np.mean(proxy_aucs))

    # Attention diagnostics: gap = SC AUC(att) − SC AUC(mean), per class then averaged
    att_vs_mean_gap = float('nan')
    if sc_mean_class_aucs is not None:
        common = sorted(set(sc_class_aucs) & set(sc_mean_class_aucs))
        if common:
            att_vs_mean_gap = float(np.mean(
                [sc_class_aucs[c] - sc_mean_class_aucs[c] for c in common]))

    proxy_str = f'  LB-proxy: {lb_proxy:.4f}' if not np.isnan(lb_proxy) else ''
    diag_str  = (f'  att_gap={att_vs_mean_gap:+.4f}  att_H={att_entropy:.3f}'
                 if not np.isnan(att_entropy) else '')
    print(f'  SC    AUC: {sc_mean:.4f} ({n_sc} cls)  '
          f'Focal AUC: {focal_mean:.4f} ({n_focal} cls)  '
          f'Composite: {composite:.4f}{proxy_str}{diag_str}')

    return (composite, sc_mean, focal_mean, lb_proxy,
            sc_class_aucs, focal_class_aucs,
            sc_probs, sc_labels, focal_probs, focal_labels,
            att_vs_mean_gap, att_entropy)


# ── OOF prediction saving ─────────────────────────────────────────────────────

@torch.inference_mode()
def save_oof_predictions(
    model: nn.Module,
    val_sc_loader: DataLoader,
    val_focal_loader: DataLoader,
    sc_val_df: pd.DataFrame,
    focal_val_df: pd.DataFrame,
    device: torch.device,
    num_classes: int,
    use_amp: bool,
    output_dir: Path,
    tag: str,
) -> None:
    """
    Run model on val sets and save OOF predictions as compressed .npz files.

    Output files:
      oof_{tag}_sc.npz    — probs, labels, filenames, start_sec
      oof_{tag}_focal.npz — probs, labels, primary_labels, filenames
    """
    _, sc_probs, sc_labels, _, _ = _compute_auc(
        model, val_sc_loader, device, num_classes, use_amp, f'OOF-{tag}-SC')
    np.savez_compressed(
        output_dir / f'oof_{tag}_sc.npz',
        probs=sc_probs,
        labels=sc_labels,
        filenames=sc_val_df['filename'].values,
        start_sec=sc_val_df['start_sec'].values,
    )

    _, focal_probs, focal_labels, _, _ = _compute_auc(
        model, val_focal_loader, device, num_classes, use_amp, f'OOF-{tag}-Focal')
    np.savez_compressed(
        output_dir / f'oof_{tag}_focal.npz',
        probs=focal_probs,
        labels=focal_labels,
        primary_labels=focal_val_df['primary_label'].values,
        filenames=focal_val_df['filename'].values,
    )
    print(f'OOF saved → oof_{tag}_sc.npz  ({sc_probs.shape[0]} chunks), '
          f'oof_{tag}_focal.npz  ({focal_probs.shape[0]} clips)')


# ── Unlabeled soundscape validation ──────────────────────────────────────────

@torch.inference_mode()
def validate_unlabeled_sc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_amp: bool,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Run model on unlabeled soundscape val set and compute self-consistency metrics.

    Since there are no ground-truth labels, the ensemble pseudo-labels (returned
    by SoundscapeUnlabeledValDataset) serve as the reference:

      consistency_score — mean per-chunk cosine similarity between model sigmoid
        outputs and ensemble pseudo-labels; measures alignment with PL targets.
      mean_max_prob     — mean over chunks of max sigmoid(logit) per chunk.

    Returns (mean_max_prob, consistency_score, all_probs, all_pseudo_labels).
    """
    model.eval()
    all_probs:  list[np.ndarray] = []
    all_pseudo: list[np.ndarray] = []

    for imgs, pseudo_labels in tqdm(loader, desc='Val-UnlabSC', leave=False):
        imgs = imgs.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            _, att_clipwise, _ = model(imgs)
        probs = torch.sigmoid(att_clipwise).float().cpu().numpy()
        all_probs.append(probs)
        all_pseudo.append(pseudo_labels.numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)
    all_pseudo = np.concatenate(all_pseudo, axis=0)

    mean_max_prob = float(all_probs.max(axis=1).mean())

    norms_pred   = np.linalg.norm(all_probs,  axis=1, keepdims=True) + 1e-8
    norms_pseudo = np.linalg.norm(all_pseudo, axis=1, keepdims=True) + 1e-8
    cos_sim = ((all_probs / norms_pred) * (all_pseudo / norms_pseudo)).sum(axis=1)
    consistency_score = float(cos_sim.mean())

    print(f'  UnlabSC  mean_max_prob={mean_max_prob:.4f}  '
          f'PL-consistency={consistency_score:.4f}  ({len(all_probs)} chunks)')

    return mean_max_prob, consistency_score, all_probs, all_pseudo


def save_oof_unlabeled_sc(
    model: nn.Module,
    loader: DataLoader,
    manifest_df: pd.DataFrame,
    device: torch.device,
    num_classes: int,
    use_amp: bool,
    output_dir: Path,
    tag: str,
) -> None:
    """
    Save OOF predictions for the held-out unlabeled soundscape split.

    Output: oof_{tag}_unlabeled_sc.npz
      probs         — (N, num_classes) model predictions
      pseudo_labels — (N, num_classes) ensemble pseudo-labels (reference)
      filenames     — (N,) soundscape filenames
      start_sec     — (N,) chunk start times in seconds
    """
    _, _, probs, pseudo_labels = validate_unlabeled_sc(
        model, loader, device, num_classes, use_amp)
    np.savez_compressed(
        output_dir / f'oof_{tag}_unlabeled_sc.npz',
        probs=probs,
        pseudo_labels=pseudo_labels,
        filenames=manifest_df['filename'].values,
        start_sec=manifest_df['start_sec'].values,
    )
    print(f'OOF unlabeled SC saved → oof_{tag}_unlabeled_sc.npz  '
          f'({probs.shape[0]} chunks)')
