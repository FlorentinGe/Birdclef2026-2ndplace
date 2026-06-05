#!/usr/bin/env python3
"""
BirdCLEF+ 2026 — training entry point.

Usage examples:

  # Supervised training from ImageNet pretrained weights:
  python train.py --backbone eca_nfnet_l0 --stage supervised

  # KD pretraining (saves pretrained_backbone.pth), then supervised:
  python train.py --backbone eca_nfnet_l0 --stage kd \\
      --perch_embed_dirs /path/to/perch/0-3500 /path/to/perch/3500-7000
  python train.py --backbone eca_nfnet_l0 --stage supervised \\
      --checkpoint pretrained_backbone.pth  # load KD backbone

  # ViT with LLRD:
  python train.py --backbone vit_base_patch16_224.dino --stage supervised \\
      --num_epochs 40 --lr 1e-4 --weight_decay 0.05 --swa_start_epoch 22

  # Focal pseudo-label fine-tuning (after generate_focal_pl.py):
  python train.py --backbone eca_nfnet_l0 --stage focal_pl \\
      --focal_pl_csv runs/focal_pl/focal_pl.csv \\
      --checkpoint   runs/eca_nfnet_l0_supervised/swa_model.pth \\
      --output_dir   runs/eca_nfnet_l0_focal_pl

  # Override paths for local run (not Kaggle):
  python train.py --backbone eca_nfnet_l0 --stage supervised \\
      --config configs/local_paths.yaml --output_dir ./runs/exp_local

Config loading order (later wins):
  1. Dataclass defaults
  2. configs/backbone/<backbone>.yaml
  3. configs/stage/<stage>.yaml
  4. --config <path>.yaml (optional run-level overrides)
  5. Explicit CLI flags
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from birdclef.config import load_config
from birdclef.datasets import (
    AISpecialistSampler,
    BirdDataset,
    BirdDatasetWithPL,
    ExternalNoiseAugmenter,
    FocalValDataset,
    NocallAugmenter,
    NoisePadder,
    PerchDistillDataset,
    SoundscapeMixupDataset,
    SoundscapePLDataset,
    SoundscapeSubstitutionDataset,
    SoundscapeTrainDataset,
    SoundscapeUnlabeledValDataset,
    SoundscapeValDataset,
)
from birdclef.model import BirdCLEFModel, PretrainModel
from birdclef.train import run_kd_stage, run_supervised_stage
from birdclef.validate import LB_PROXY_SPECIES
from birdclef.transforms import MelTransform
from birdclef.utils import (
    build_vocabulary,
    build_wav_path_map,
    encode_multilabel,
    make_sc_label_vec,
    normalise_sc_df,
    resolve_audio_paths,
    resolve_wav_paths,
    seed_everything,
    stratified_soundscape_split,
    stratified_unlabeled_sc_split,
)


def _make_sample_weights(
    focal_df: pd.DataFrame,
    n_sc: int,
    thresh: int,
    cap: float,
    sc_weights: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Compute per-sample weights for WeightedRandomSampler.

    Focal clips from rare species (primary_label count < thresh) are upsampled
    proportionally: weight = min(cap, thresh / count).  All other focal clips
    and all soundscape clips get weight 1.0 unless sc_weights is provided.

    The sampler draws with replacement so epoch length is preserved.  cap limits
    the maximum boost to avoid rare species dominating the batch (exp12 lesson).

    Args:
        focal_df:   training focal clips DataFrame (primary_label column required)
        n_sc:       number of soundscape training clips
        thresh:     species count threshold; 0 disables focal upsampling
        cap:        maximum weight multiplier for focal clips
        sc_weights: optional (n_sc,) float32 array of per-chunk SC weights.
                    When None, all SC clips get weight 1.0.

    Returns:
        1-D float tensor of length len(focal_df) + n_sc.
    """
    counts = focal_df['primary_label'].value_counts()
    if thresh > 0:
        # Vectorised: map each clip's primary_label to its species count,
        # then compute weight = clip(thresh / count, 1.0, cap).
        clip_counts  = focal_df['primary_label'].map(counts).fillna(1).astype(float)
        raw_weights  = (thresh / clip_counts).clip(upper=cap)
        focal_w_arr  = np.where(clip_counts < thresh, raw_weights, 1.0)
        focal_w      = focal_w_arr.tolist()
        n_rare = int((focal_w_arr > 1.0).sum())
        print(f'Rare-species upsampling: thresh={thresh}, cap={cap}× '
              f'→ {n_rare:,} / {len(focal_df):,} focal clips upsampled')
    else:
        focal_w = [1.0] * len(focal_df)

    if sc_weights is not None:
        assert len(sc_weights) == n_sc, (
            f'sc_weights length {len(sc_weights)} != n_sc {n_sc}')
        sc_w = sc_weights.tolist()
    else:
        sc_w = [1.0] * n_sc
    return torch.tensor(focal_w + sc_w, dtype=torch.float32)


def _build_sc_hard_label_mask(
    manifest: pd.DataFrame,
    labels_csv: Path,
    species2idx: dict,
    num_classes: int,
) -> np.ndarray:
    """
    Build a (M, num_classes) bool mask from train_soundscapes_labels.csv.

    For each (filename, start_sec) row in the labels CSV that matches a row in
    manifest, sets True for every species listed in primary_label.  Unmatched
    entries remain False (pseudo-label used as-is).
    """
    def _ts_to_sec(s: str) -> int:
        h, m, sec = s.split(':')
        return int(h) * 3600 + int(m) * 60 + int(sec)

    labels_df = pd.read_csv(labels_csv)
    labels_df['start_sec_int'] = labels_df['start'].apply(_ts_to_sec)

    lookup: dict = {}
    for _, row in labels_df.iterrows():
        key = (row['filename'], int(row['start_sec_int']))
        for sp in str(row['primary_label']).split(';'):
            sp = sp.strip()
            if sp and sp in species2idx:
                lookup.setdefault(key, []).append(species2idx[sp])

    mask = np.zeros((len(manifest), num_classes), dtype=bool)
    for i, (fname, start) in enumerate(
            zip(manifest['filename'], manifest['start_sec'])):
        key = (fname, int(start))
        if key in lookup:
            for cidx in lookup[key]:
                mask[i, cidx] = True
    return mask


def _build_nocall_augmenter(
    cfg,
    base_dir: Path,
    soundscape_dir: Path,
    sc_val_fileset: set,
) -> NocallAugmenter:
    """
    Build a NocallAugmenter from the competition pseudo-label files.

    Nocall segments are those present in train_soundscape_pseudolabels.csv
    (Perch's full probability matrix, one row per 5-second chunk) but absent
    from pseudo_labels.csv (Perch's thresholded output).  The absence means
    Perch did not confidently assign any species label — the closest available
    approximation to a bird-free segment.

    Val soundscape files are excluded to prevent audio leakage.
    """
    pseudo_labels_path = base_dir / 'pseudo_labels.csv'
    pseudo_probs_path  = base_dir / 'train_soundscape_pseudolabels.csv'

    # Load only the row_id column of the large probability matrix
    all_ids_df = pd.read_csv(pseudo_probs_path, usecols=['row_id'])

    # Build set of labeled segment IDs from pseudo_labels.csv
    pl = pd.read_csv(pseudo_labels_path)

    def _timestr_to_sec(t: str) -> int:
        h, m, s = t.split(':')
        return int(h) * 3600 + int(m) * 60 + int(s)

    pl['end_sec'] = pl['end'].apply(_timestr_to_sec)
    pl['stem']    = pl['filename'].str.replace('.ogg', '', regex=False)
    labeled_ids   = set(pl['stem'] + '_' + pl['end_sec'].astype(str))

    # Nocall = in full matrix but not in thresholded output
    nocall_mask = ~all_ids_df['row_id'].isin(labeled_ids)
    nocall_df   = all_ids_df[nocall_mask].copy()

    # Parse row_id → filename + start_sec
    # row_id format: {stem}_{end_sec}  e.g. BC2026_Train_0001_S08_20250606_030007_5
    split         = nocall_df['row_id'].str.rsplit('_', n=1)
    nocall_df['stem']      = split.str[0]
    nocall_df['end_sec']   = split.str[1].astype(int)
    nocall_df['start_sec'] = nocall_df['end_sec'] - cfg.chunk_duration
    nocall_df['filename']  = nocall_df['stem'] + '.ogg'

    # Exclude val soundscape files (no audio leakage)
    nocall_df = nocall_df[
        ~nocall_df['filename'].isin(sc_val_fileset)
    ][['filename', 'start_sec']].reset_index(drop=True)

    print(f'Nocall background pool: {len(nocall_df):,} segments '
          f'from {nocall_df["filename"].nunique():,} files  '
          f'(val files excluded)')

    return NocallAugmenter(nocall_df, soundscape_dir, cfg)


def build_perch_caps(
    species2idx: dict,
    sp2taxclass: dict,
    default_cap: float,
) -> np.ndarray:
    """
    Build a per-class Perch cap array for BirdDatasetWithPL.

    Perch is a bird-sound model, so its pseudo-labels are reliable for Aves
    but noisy for other taxonomic classes:
      Aves      → default_cap  (cfg.pl_perch_max, e.g. 0.5)
      Amphibia  → min(default_cap, 0.3)  (Perch has partial frog coverage)
      Insecta   → min(default_cap, 0.3)  (Perch has limited insect coverage)
      Mammalia  → 0.0           (Perch is a bird model; mammal labels are noise)
      unknown   → default_cap

    Args:
        species2idx:  mapping species_code → class index.
        sp2taxclass:  mapping species_code → class_name from taxonomy.csv.
        default_cap:  cfg.pl_perch_max (the Aves cap).

    Returns:
        Float32 ndarray of shape (num_classes,).
    """
    num_classes = len(species2idx)
    caps = np.full(num_classes, default_cap, dtype=np.float32)

    cap_by_taxclass = {
        'Aves':     default_cap,
        'Amphibia': min(default_cap, 0.5),#0.3
        'Insecta':  min(default_cap, 0.5),#0.3
        'Mammalia': min(default_cap, 0.5),#0.0
    }

    counts = {k: 0 for k in cap_by_taxclass}
    counts['unknown'] = 0
    for sp, idx in species2idx.items():
        taxclass = sp2taxclass.get(str(sp), '')
        cap = cap_by_taxclass.get(taxclass, default_cap)
        caps[idx] = cap
        counts[taxclass if taxclass in cap_by_taxclass else 'unknown'] += 1

    print('Perch class-conditional caps:')
    for cls, cap in cap_by_taxclass.items():
        print(f'  {cls:<12} cap={cap:.2f}  ({counts[cls]} species)')
    if counts['unknown']:
        print(f'  {"unknown":<12} cap={default_cap:.2f}  ({counts["unknown"]} species)')

    return caps


def _run_ai_specialist(
    cfg, extra_args, train_df, sc_df, soundscape_dir, species2idx, idx2species,
    num_classes, sp2taxclass, proxy_class_indices, mel_transform, device, output_dir,
):
    """
    Amphibia + Insecta specialist model (1st-place 2025 recipe).

    Vocabulary: competition 234 species (indices 0–233) + XC extra species (234+).
    Training data:
      - Competition focal clips filtered to Amphibia/Insecta primary labels
      - Competition labeled soundscape windows — Amphibia/Insecta labels only (Aves zeroed)
      - XC extra focal data (birdclef2025_extra_species_data: grasshoppers + frogs)
    Validation: same soundscape/focal splits as other stages; AUC computed for all
      classes with positives — Aves AUC will be near-random (no Aves training signal),
      AI species AUC is what matters.
    OOF unlabeled SC: saved in competition 234-column format (first num_classes columns
      of the extended prediction), compatible with submission_analysis_v4.py.
    """
    if not cfg.ai_xc_species_csv or not cfg.ai_xc_species_dir:
        raise ValueError(
            '--ai_xc_species_csv and --ai_xc_species_dir are required for stage=ai_specialist')

    base_dir = Path(cfg.base_dir)
    xc_csv_path = Path(cfg.ai_xc_species_csv)
    xc_dir      = Path(cfg.ai_xc_species_dir)

    # ── Identify AI species in competition vocabulary ──────────────────────────
    _tax_path = base_dir / 'taxonomy.csv'
    if not _tax_path.exists():
        raise FileNotFoundError(f'taxonomy.csv not found at {_tax_path}')
    tax = pd.read_csv(_tax_path)
    ai_labels = frozenset(
        tax[tax['class_name'].isin(['Amphibia', 'Insecta'])]['primary_label'])
    ai_indices_set = frozenset(species2idx[sp] for sp in ai_labels if sp in species2idx)
    print(f'AI species in competition vocab: {len(ai_indices_set)} '
          f'(Amphibia+Insecta from taxonomy.csv)')

    # ── Load XC extra species and build extended vocabulary ────────────────────
    xc_df = pd.read_csv(xc_csv_path)
    xc_df = xc_df[xc_df['primary_label'].notna()].reset_index(drop=True)

    # XC-only species: not already in competition vocabulary
    xc_only_labels = sorted(set(xc_df['primary_label'].unique()) - set(species2idx))
    xc2idx_ext     = {sp: num_classes + i for i, sp in enumerate(xc_only_labels)}
    full_s2idx     = {**species2idx, **xc2idx_ext}
    extended_num_classes = num_classes + len(xc_only_labels)

    # Save extended label map: first num_classes entries are competition vocab (unchanged)
    ext_idx2species = {**idx2species, **{v: k for k, v in xc2idx_ext.items()}}
    np.save(output_dir / 'label_map.npy', ext_idx2species)
    print(f'Extended vocabulary: {num_classes} competition + {len(xc_only_labels)} '
          f'XC-only = {extended_num_classes} total')
    print(f'XC extra data: {len(xc_df)} clips, {xc_df["primary_label"].nunique()} species')

    # ── Soundscape split (same seed/fracs → identical val files as other stages) ──
    sc_train_fileset, sc_val_fileset = stratified_soundscape_split(
        sc_df, cfg.soundscape_val_frac, cfg.seed)
    sc_train_df = sc_df[sc_df['filename'].isin(sc_train_fileset)].reset_index(drop=True)
    sc_val_df   = sc_df[sc_df['filename'].isin(sc_val_fileset)].reset_index(drop=True)
    print(f'Soundscape files → train: {len(sc_train_fileset)}, val: {len(sc_val_fileset)}')

    # ── Focal split (same seed/frac) ──────────────────────────────────────────
    rng_focal      = np.random.default_rng(cfg.seed)
    focal_val_mask = rng_focal.random(len(train_df)) < cfg.focal_val_frac
    focal_val_df   = train_df[focal_val_mask].reset_index(drop=True)
    train_df_fit   = train_df[~focal_val_mask].reset_index(drop=True)
    print(f'Focal clips → train: {len(train_df_fit):,}, val: {len(focal_val_df):,}')

    # ── Build training datasets ────────────────────────────────────────────────

    # 1. Competition AI focal clips
    ai_focal_df = train_df_fit[
        train_df_fit['primary_label'].isin(ai_labels)
    ].copy().reset_index(drop=True)
    ai_focal_df['label'] = ai_focal_df['primary_label'].map(species2idx)
    ai_focal_df['secondary_label_vec'] = ai_focal_df['secondary_labels'].apply(
        lambda x: encode_multilabel(x, extended_num_classes, species2idx))
    focal_ai_ds = BirdDataset(ai_focal_df, cfg, extended_num_classes, species2idx)
    print(f'AI focal train: {len(ai_focal_df)} clips, '
          f'{ai_focal_df["primary_label"].nunique()} species')

    # 2. Labeled soundscape windows — AI species labels only
    #    Zero out any Aves labels so the specialist does not train on Aves detection.
    def _ai_sc_label_vec(label_str, n, s2i, ai_idx):
        vec = np.zeros(n, dtype=np.float32)
        for sp in str(label_str).split(';'):
            sp = sp.strip()
            idx = s2i.get(sp)
            if idx is not None and idx in ai_idx:
                vec[idx] = 1.0
        return vec

    sc_train_df_ai = sc_train_df.copy()
    sc_train_df_ai['label_vec'] = sc_train_df_ai['primary_label'].apply(
        lambda x: _ai_sc_label_vec(x, extended_num_classes, species2idx, ai_indices_set))
    sc_ai_ds = SoundscapeTrainDataset(sc_train_df_ai, soundscape_dir, cfg)
    n_sc_ai_positive = int(
        (sc_train_df_ai['label_vec'].apply(lambda v: v.sum() > 0)).sum())
    print(f'AI soundscape train: {len(sc_train_df_ai)} windows, '
          f'{n_sc_ai_positive} with ≥1 AI label')

    # 2b. Soundscape pseudo-labels (sc_pl_dir) — AI species only.
    #     Preds are (M, num_classes=234); pad to extended_num_classes and zero
    #     non-AI columns so the specialist head is not trained on Aves signal.
    #     Excluded: labeled SC val files, unlabeled SC holdout (sc_pl_val_files.txt),
    #     and labeled SC training files (already served via sc_ai_ds every epoch).
    sc_pl_train_ds    = None
    _pl_conf_weights  = None     # set below when PL is loaded successfully
    if cfg.sc_pl_dir:
        _sc_pl_path = Path(cfg.sc_pl_dir)
        _sc_pl_csv  = _sc_pl_path / 'sc_pl.csv'
        _sc_pl_npy  = _sc_pl_path / 'sc_pl_preds_ensemble.npy'
        _val_txt    = _sc_pl_path / 'sc_pl_val_files.txt'
        if _sc_pl_csv.exists() and _sc_pl_npy.exists():
            _manifest  = pd.read_csv(_sc_pl_csv)
            _ens_preds = np.load(_sc_pl_npy)          # (M, 234)
            # Exclude labeled SC val files, unlabeled SC holdout, and labeled SC
            # training files (sc_train_fileset) — those are already in sc_ai_ds.
            _exclude = sc_val_fileset | sc_train_fileset
            if _val_txt.exists():
                _exclude |= set(_val_txt.read_text().splitlines())
            _keep      = ~_manifest['filename'].isin(_exclude)
            _manifest  = _manifest[_keep].reset_index(drop=True)
            _ens_preds = _ens_preds[_keep.values]
            # Build AI mask over the 234 competition columns
            _ai_mask = np.zeros(num_classes, dtype=np.float32)
            for _i in ai_indices_set:
                if _i < num_classes:
                    _ai_mask[_i] = 1.0
            # Zero non-AI columns and pad to extended_num_classes
            _preds_ext = np.zeros(
                (len(_manifest), extended_num_classes), dtype=np.float32)
            _preds_ext[:, :num_classes] = _ens_preds * _ai_mask[None, :]
            # File-level confidence weights (1st-place recipe): for each file,
            # sum the per-class maximum prediction across all its chunks.
            # Chunks from high-confidence files are sampled more often.
            _file_w: dict = {}
            for _fn, _grp in _manifest.groupby('filename'):
                _cidx = _grp.index.values
                _file_w[_fn] = float(_preds_ext[_cidx].max(axis=0).sum())
            _pl_conf_weights = np.array(
                [_file_w[_manifest.loc[i, 'filename']] for i in range(len(_manifest))],
                dtype=np.float32,
            )
            sc_pl_train_ds = SoundscapePLDataset(
                _manifest, _preds_ext, soundscape_dir, cfg)
            print(f'SC PL train (AI-masked): {len(_manifest):,} chunks from '
                  f'{_manifest["filename"].nunique():,} files  '
                  f'(conf mean={_pl_conf_weights.mean():.3f})')
        else:
            _missing = ([_sc_pl_csv.name] if not _sc_pl_csv.exists() else []) + \
                       ([_sc_pl_npy.name] if not _sc_pl_npy.exists() else [])
            print(f'SC PL: skipping ({", ".join(_missing)} not found in {cfg.sc_pl_dir})')

    # 3. XC extra focal clips
    xc_df['file_path'] = xc_df['filepath'].apply(lambda p: str(xc_dir / p))
    xc_df['label']     = xc_df['primary_label'].map(full_s2idx)
    # Drop rows where label is None (species not in full_s2idx — shouldn't happen)
    xc_df = xc_df[xc_df['label'].notna()].copy()
    xc_df['label'] = xc_df['label'].astype(int)
    # No secondary_label_vec column → BirdDataset.get() returns None → skipped safely
    xc_train_ds = BirdDataset(xc_df, cfg, extended_num_classes, full_s2idx)
    print(f'XC extra train: {len(xc_df)} clips, '
          f'{xc_df["primary_label"].nunique()} species')

    # When the backbone is frozen for the full run, XC extra data provides no useful
    # signal and actively harms training: every XC sample has label=0 for all 63
    # competition AI class indices, so BCE pushes those logits toward zero in every
    # XC batch.  With 17K XC samples vs ~2K competition samples, this overwhelms the
    # positive gradient from competition data and causes catastrophic forgetting of
    # Pantanal Amphibia/Insecta patterns (observed as SC AUC peaking at ep4 then
    # collapsing).  The 1st-place backbone already encodes XC Insecta/Amphibia
    # features; the frozen head only needs competition-specific label supervision.
    # Exclude XC extra when (a) backbone is frozen for the full run, or (b) use_xc_extra=False.
    # Rationale for (b): XC clips carry label=0 for every competition class index.  With ~17K
    # XC samples vs ~650 competition clips the effective gradient ratio is ~33:1 negative vs
    # positive for competition Insecta/Amphibia classes, collapsing the SC AUC after a brief
    # early peak.  When the backbone is already XC-pretrained the XC features are already
    # encoded; re-exposing them only pollutes the competition-class head.
    exclude_xc = (cfg.freeze_epochs >= cfg.num_epochs) or (not cfg.use_xc_extra)

    n_focal = len(focal_ai_ds)
    n_sc    = len(sc_ai_ds)
    n_xc    = len(xc_train_ds) if not exclude_xc else 0

    if exclude_xc:
        reason = (f'freeze_epochs={cfg.freeze_epochs} >= num_epochs={cfg.num_epochs}'
                  if cfg.freeze_epochs >= cfg.num_epochs else 'use_xc_extra=False')
        print(f'XC extra excluded from training ({reason})')

    if sc_pl_train_ds is not None and _pl_conf_weights is not None:
        # AISpecialistSampler: each epoch draws ~n_base_draws items from the base
        # (focal + labeled SC + XC, weighted [3,2,1] to match exp85 composition)
        # plus an equal count from the PL pool (file-level confidence-weighted,
        # without replacement).  set_epoch() is called by run_supervised_stage.
        base_ds = ConcatDataset(
            [focal_ai_ds, sc_ai_ds] + ([] if exclude_xc else [xc_train_ds]))
        combined_train_ds = ConcatDataset([base_ds, sc_pl_train_ds])
        sampler = AISpecialistSampler(
            n_focal=n_focal, w_focal=3.0,
            n_sc=n_sc,       w_sc=2.0,
            n_xc=n_xc,       w_xc=1.0,
            pl_conf_weights=_pl_conf_weights,
            base_seed=cfg.seed,
        )
        _n_base_draws = sampler._n_base_draws
        _desc = (f'AI focal×3: {n_focal}, AI SC×2: {n_sc}'
                 + (f', XC×1: {n_xc}' if n_xc else '')
                 + f' | PL conf-weighted: {len(sc_pl_train_ds):,}'
                 + f' | epoch: {len(sampler):,} steps ({_n_base_draws:,} base + {_n_base_draws:,} PL)')
    else:
        # No PL: original WeightedRandomSampler
        _datasets = [focal_ai_ds, sc_ai_ds]
        _weights  = [3.0] * n_focal + [2.0] * n_sc
        _desc     = f'AI focal×3: {n_focal}, AI SC×2: {n_sc}'
        if not exclude_xc:
            _datasets.append(xc_train_ds)
            _weights += [1.0] * n_xc
            _desc    += f', XC extra×1: {n_xc}'
        combined_train_ds = ConcatDataset(_datasets)
        sample_weights    = torch.tensor(_weights, dtype=torch.float32)
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(combined_train_ds), replacement=True)

    train_loader = DataLoader(
        combined_train_ds, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    print(f'Train batches: {len(train_loader):,}  ({_desc})')

    # ── Validation datasets (same splits, extended_num_classes label vecs) ─────
    sc_val_df['label_vec'] = sc_val_df['primary_label'].apply(
        lambda x: make_sc_label_vec(x, extended_num_classes, species2idx))
    focal_val_df['label'] = focal_val_df['primary_label'].map(species2idx)
    focal_val_df['secondary_label_vec'] = focal_val_df['secondary_labels'].apply(
        lambda x: encode_multilabel(x, extended_num_classes, species2idx))

    val_sc_ds    = SoundscapeValDataset(sc_val_df, soundscape_dir, cfg)
    val_focal_ds = FocalValDataset(focal_val_df, cfg, extended_num_classes)

    val_sc_loader = DataLoader(
        val_sc_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True)
    val_focal_loader = DataLoader(
        val_focal_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    # backbone: tf_efficientnet_b0 (1st-place specialist used EffNet-B0)
    # Always pretrained=False; 1st-place backbone loaded via --checkpoint.
    # The 1st-place checkpoint has backbone.* keys (shape-matched) and head.*
    # keys (not in our namespace → silently skipped by the existing loader).
    load_pretrained = extra_args.checkpoint is None
    model = BirdCLEFModel(cfg, extended_num_classes, pretrained=load_pretrained)

    if extra_args.checkpoint is not None:
        ckpt_path = Path(extra_args.checkpoint)
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if any(k.startswith('backbone.') for k in state.keys()):
            current_state = model.state_dict()
            compatible = {k: v for k, v in state.items()
                          if k in current_state and v.shape == current_state[k].shape}
            current_state.update(compatible)
            model.load_state_dict(current_state)
            n_bb = sum(1 for k in compatible if k.startswith('backbone.'))
            n_hd = len(compatible) - n_bb
            print(f'Loaded checkpoint {ckpt_path.name}  '
                  f'({n_bb} backbone + {n_hd} head tensors matched, '
                  f'{len(state) - len(compatible)} skipped)')
        else:
            missing, unexpected = model.backbone.load_state_dict(state, strict=False)
            print(f'Loaded backbone-only checkpoint {ckpt_path.name}')
            if missing:
                print(f'  Missing ({len(missing)}): {missing[:4]}...')
            if unexpected:
                print(f'  Unexpected ({len(unexpected)}): {unexpected[:4]}...')

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {cfg.backbone}  |  Classes: {extended_num_classes}  '
          f'|  Params: {total_params:,}')

    # ── Train ─────────────────────────────────────────────────────────────────
    hist_df, best_auc, swa_auc, best_sc_auc = run_supervised_stage(
        cfg, model, train_loader, val_sc_loader, val_focal_loader,
        mel_transform, sc_val_df, focal_val_df, device, output_dir,
        nocall_augmenter=None,
        proxy_class_indices=proxy_class_indices,
        val_unlabeled_sc_loader=None,
        unlabeled_sc_val_manifest=None)

    # ── OOF on unlabeled soundscape holdout (competition 234-column format) ───
    # Requires --sc_pl_dir pointing to a directory with sc_pl.csv and
    # sc_pl_val_files.txt (written by the first sc_pl run).
    if not cfg.sc_pl_dir:
        print('\nSkipping unlabeled SC OOF — provide --sc_pl_dir to generate it.')
        return

    sc_pl_path    = Path(cfg.sc_pl_dir)
    val_txt       = sc_pl_path / 'sc_pl_val_files.txt'
    sc_pl_csv     = sc_pl_path / 'sc_pl.csv'
    if not sc_pl_csv.exists():
        print(f'WARNING: {sc_pl_csv} not found — skipping unlabeled SC OOF.')
        return
    if not val_txt.exists():
        print(f'WARNING: {val_txt} not found — run stage=sc_pl first to create it.')
        return

    manifest   = pd.read_csv(sc_pl_csv)
    val_files  = set(val_txt.read_text().splitlines())
    val_mask   = manifest['filename'].isin(val_files & set(manifest['filename'].unique()))
    uls_manifest = manifest[val_mask].reset_index(drop=True)
    print(f'\nUnlabeled SC holdout: {len(uls_manifest):,} chunks from '
          f'{uls_manifest["filename"].nunique():,} files')

    # Dummy ensemble_preds (zeros) — we only need audio loading, not PL consistency.
    dummy_preds = np.zeros((len(uls_manifest), num_classes), dtype=np.float32)
    uls_ds = SoundscapeUnlabeledValDataset(uls_manifest, dummy_preds, soundscape_dir, cfg)
    uls_loader = DataLoader(
        uls_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True)

    for tag, ckpt_name in [('best', 'best_model.pth'),
                            ('best_sc', 'best_sc_model.pth'),
                            ('swa', 'swa_model.pth')]:
        ckpt_p = output_dir / ckpt_name
        if not ckpt_p.exists():
            print(f'  {ckpt_name} not found — skipping {tag}')
            continue

        infer_model = BirdCLEFModel(cfg, extended_num_classes, pretrained=False)
        infer_state = torch.load(ckpt_p, map_location='cpu', weights_only=False)
        infer_model.load_state_dict(infer_state)
        infer_model.to(device).eval()

        all_probs: list = []
        with torch.inference_mode():
            for mels, _ in uls_loader:
                mels = mels.to(device)
                with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                    _, att_clipwise, _ = infer_model(mels)
                # Slice to competition 234 columns only — compatible with submission_analysis_v4.py
                probs = torch.sigmoid(att_clipwise[:, :num_classes]).float().cpu().numpy()
                all_probs.append(probs)

        all_probs_arr = np.concatenate(all_probs, axis=0)   # (N, 234)
        np.savez_compressed(
            output_dir / f'oof_{tag}_unlabeled_sc.npz',
            probs=all_probs_arr,
            filenames=uls_manifest['filename'].values,
            start_sec=uls_manifest['start_sec'].values,
        )
        print(f'OOF unlabeled SC → oof_{tag}_unlabeled_sc.npz  '
              f'({all_probs_arr.shape[0]} chunks × {all_probs_arr.shape[1]} competition classes)')
        del infer_model

    # ── Run summary ───────────────────────────────────────────────────────────
    run_summary = {
        'backbone':              cfg.backbone,
        'stage':                 cfg.stage,
        'competition_classes':   num_classes,
        'extended_num_classes':  extended_num_classes,
        'xc_extra_species':      len(xc_only_labels),
        'num_epochs':            cfg.num_epochs,
        'best_composite_auc':    round(best_auc, 4),
        'swa_composite_auc':     round(swa_auc,  4),
        'best_sc_auc':           round(best_sc_auc, 4),
        'checkpoint_loaded':     str(extra_args.checkpoint),
    }
    import json
    with open(output_dir / 'run_summary.json', 'w') as f:
        json.dump(run_summary, f, indent=2)
    cfg.save_json(output_dir / 'run_config.json')


def main():
    # ── Config ────────────────────────────────────────────────────────────────
    # load_config() handles the CLI + YAML merging described in config.py.
    # We add one extra CLI arg here: --checkpoint (backbone weights to warm-start from).
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--checkpoint', default=None,
                        help='Path to backbone .pth to load before Stage 2 training '
                             '(e.g. pretrained_backbone.pth from KD stage)')
    extra_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining   # pass remaining args to load_config

    cfg = load_config()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(cfg.seed)

    print(f'Backbone : {cfg.backbone}')
    print(f'Stage    : {cfg.stage}')
    print(f'Device   : {device}  |  AMP: {cfg.use_amp}')
    print(f'Output   : {output_dir}')

    # ── Data loading ──────────────────────────────────────────────────────────
    base_dir  = Path(cfg.base_dir)
    train_df  = pd.read_csv(base_dir / 'train.csv')

    # Taxonomy: primary_label → class_name (Aves / Amphibia / Insecta / Mammalia …)
    _tax_path = base_dir / 'taxonomy.csv'
    if _tax_path.exists():
        _tax = pd.read_csv(_tax_path)
        sp2taxclass: dict = _tax.set_index('primary_label')['class_name'].to_dict()
    else:
        sp2taxclass = {}
    sc_df_raw = pd.read_csv(base_dir / 'train_soundscapes_labels.csv')

    # Resolve focal audio paths.
    # Default: OGG originals at base_dir/train_audio/{primary_label}/{stem}.ogg
    # To use WAV shards instead, set train_audio_dir explicitly or pass
    # --train_audio_dir to point at one of the ttahara shard directories after
    # building a per-label shard mapping via build_wav_path_map + resolve_wav_paths.
    audio_dir = cfg.train_audio_path()
    if cfg.train_audio_dir is None and cfg.train_audio_wav_prefix != \
            '/kaggle/input/datasets/ttahara/birdclef2026-train-audio-wav-':
        # Non-default wav prefix was set explicitly — use legacy WAV shard mapping
        pl2wav_dir = build_wav_path_map(cfg.train_audio_wav_prefix)
        train_df['file_path'] = resolve_wav_paths(train_df, pl2wav_dir)
        print(f'Audio source: WAV shards ({cfg.train_audio_wav_prefix}*)')
    else:
        train_df['file_path'] = resolve_audio_paths(train_df, audio_dir)
        print(f'Audio source: {audio_dir}')

    sc_df = normalise_sc_df(sc_df_raw)
    soundscape_dir = cfg.soundscape_path()

    # ── Soundscape split ──────────────────────────────────────────────────────
    sc_train_fileset, sc_val_fileset = stratified_soundscape_split(
        sc_df, cfg.soundscape_val_frac, cfg.seed)
    sc_train_df = sc_df[sc_df['filename'].isin(sc_train_fileset)].reset_index(drop=True)
    sc_val_df   = sc_df[sc_df['filename'].isin(sc_val_fileset)].reset_index(drop=True)
    print(f'Soundscape files → train: {len(sc_train_fileset)}, val: {len(sc_val_fileset)}')

    # ── Vocabulary ────────────────────────────────────────────────────────────
    all_species, species2idx, idx2species = build_vocabulary(train_df, sc_df)
    num_classes = len(all_species)
    np.save(output_dir / 'label_map.npy', idx2species)
    print(f'Vocabulary: {num_classes} species')

    # LB proxy species indices (subset of 11 that are present in this vocabulary)
    proxy_class_indices = {species2idx[s] for s in LB_PROXY_SPECIES if s in species2idx}
    n_proxy = len(proxy_class_indices)
    print(f'LB proxy species: {n_proxy}/{len(LB_PROXY_SPECIES)} present in vocabulary')

    # ── Label encoding ────────────────────────────────────────────────────────
    train_df['label'] = train_df['primary_label'].map(species2idx)
    train_df['secondary_label_vec'] = train_df['secondary_labels'].apply(
        lambda x: encode_multilabel(x, num_classes, species2idx))
    sc_train_df['label_vec'] = sc_train_df['primary_label'].apply(
        lambda x: make_sc_label_vec(x, num_classes, species2idx))
    sc_val_df['label_vec']   = sc_val_df['primary_label'].apply(
        lambda x: make_sc_label_vec(x, num_classes, species2idx))

    # ── Focal val split ───────────────────────────────────────────────────────
    rng_focal      = np.random.default_rng(cfg.seed)
    focal_val_mask = rng_focal.random(len(train_df)) < cfg.focal_val_frac
    focal_val_df   = train_df[focal_val_mask].reset_index(drop=True)
    train_df_fit   = train_df[~focal_val_mask].reset_index(drop=True)
    print(f'Focal clips → train: {len(train_df_fit):,}, val: {len(focal_val_df):,}')

    # ── MelTransform (GPU-side) ───────────────────────────────────────────────
    mel_transform = MelTransform(cfg).to(device)

    # ── Stage 1: KD pretraining ───────────────────────────────────────────────
    if cfg.stage == 'kd':
        if not cfg.perch_embed_dirs:
            raise ValueError(
                '--perch_embed_dirs is required for stage=kd. '
                'Provide one or more directories containing '
                'perch_train_arrays.npz + perch_train_meta.parquet.')

        # Load Perch embeddings
        emb_list, meta_list = [], []
        for d in cfg.perch_embed_dirs:
            d = Path(d)
            data = np.load(d / 'perch_train_arrays.npz')
            meta = pd.read_parquet(d / 'perch_train_meta.parquet')
            if data['embeddings'].shape[0] > 0:
                emb_list.append(data['embeddings'].astype(np.float32))
                meta_list.append(meta)

        perch_embeddings = np.concatenate(emb_list, axis=0)
        perch_meta       = pd.concat(meta_list, ignore_index=True)

        # L2-normalise once
        norms = np.linalg.norm(perch_embeddings, axis=1, keepdims=True)
        perch_embeddings = perch_embeddings / np.maximum(norms, 1e-8)

        # Parse start_sec: row_id = "<stem>_<end_sec>", start = end - CHUNK_DURATION
        perch_meta['start_sec'] = (
            perch_meta['row_id'].str.rsplit('_', n=1).str[1].astype(int)
            - cfg.chunk_duration
        )

        # Exclude val soundscape files from KD pretraining (no audio leakage)
        pretrain_mask    = ~perch_meta['filename'].isin(sc_val_fileset)
        perch_train_meta = perch_meta[pretrain_mask].reset_index(drop=True)
        perch_train_embs = perch_embeddings[pretrain_mask.values]
        print(f'Perch embeddings → total: {len(perch_meta):,}, '
              f'after val exclusion: {len(perch_train_meta):,}')

        distill_ds = PerchDistillDataset(
            perch_train_meta, perch_train_embs, soundscape_dir, cfg)
        distill_loader = DataLoader(
            distill_ds, batch_size=cfg.pretrain_batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

        pretrain_model = PretrainModel(cfg, pretrained=True)
        run_kd_stage(cfg, pretrain_model, distill_loader, mel_transform,
                     device, output_dir)
        return

    # ── Stage: AI specialist (Amphibia + Insecta) ─────────────────────────────
    if cfg.stage == 'ai_specialist':
        _run_ai_specialist(
            cfg, extra_args, train_df, sc_df, soundscape_dir, species2idx, idx2species,
            num_classes, sp2taxclass, proxy_class_indices, mel_transform, device, output_dir)
        return

    # ── Noise padding ─────────────────────────────────────────────────────────
    # Separate from bg_aug: fills zero-padded regions of short clips with
    # low-amplitude noise so the SED attention head cannot use silent frames
    # as a padding indicator.  Does not touch the actual bird-call signal.
    noise_padder = None
    if cfg.noise_padding:
        if not cfg.noise_dir:
            raise ValueError('--noise_padding requires --noise_dir to be set.')
        noise_padder = NoisePadder(Path(cfg.noise_dir), cfg.sr)
        print(f'Noise padding enabled (noise_dir={cfg.noise_dir})')

    # ── Stage 2+: Supervised / pseudo-label fine-tuning ──────────────────────

    # Build focal training dataset — BirdDataset for supervised and sc_pl,
    # BirdDatasetWithPL only for focal_pl stage.
    # sc_pl always uses hard labels: the 1st-place MixUp is
    #   mixed_label = 0.5 * hard_focal + 0.5 * sc_pseudo
    # Applying BirdDatasetWithPL here would double-dilute hard labels to 15%
    # weight (0.5 × 0.3) instead of 50%; focal PL knowledge is already in the
    # checkpoint loaded via --checkpoint.
    focal_pl_train_df = None   # set below if focal_pl_csv is used
    if cfg.stage == 'focal_pl':
        if not cfg.focal_pl_csv:
            raise ValueError(
                '--focal_pl_csv is required for stage=focal_pl. '
                'Generate it first with generate_focal_pl.py.')
        focal_pl_csv = Path(cfg.focal_pl_csv)
        focal_pl_df  = pd.read_csv(focal_pl_csv)

        # Raw CNN ensemble probabilities (no threshold, values in [0, 1])
        raw_ensemble = np.load(focal_pl_csv.parent / 'focal_pl_raw_ensemble.npy')

        # Optional Perch continuous soft labels (z/4 clipped to [0,1])
        perch_soft = None
        perch_caps = None
        if cfg.pl_perch_max > 0.0:
            perch_path = focal_pl_csv.parent / 'focal_pl_preds_perch_continuous.npy'
            if perch_path.exists():
                perch_soft = np.load(perch_path)
                perch_caps = build_perch_caps(species2idx, sp2taxclass, cfg.pl_perch_max)
            else:
                print(f'WARNING: pl_perch_max={cfg.pl_perch_max} but '
                      f'{perch_path.name} not found — Perch disabled.')

        # Attach label index and secondary_label_vec from train_df so the
        # dataset can build original hard labels without re-reading the CSV.
        focal_pl_df = focal_pl_df.merge(
            train_df[['filename', 'label', 'secondary_label_vec']],
            on='filename', how='left')

        # Exclude the held-out focal val clips (same filenames as focal_val_df)
        val_filenames = set(focal_val_df['filename'].values)
        focal_pl_train_df = focal_pl_df[
            ~focal_pl_df['filename'].isin(val_filenames)
        ].reset_index(drop=True)
        print(f'Focal PL clips → train: {len(focal_pl_train_df):,}, '
              f'val: {len(focal_val_df):,}')
        print(f'PL formula: alpha={cfg.pl_pseudo_alpha}, '
              f'th={cfg.pl_pseudo_th}, power={cfg.pl_pseudo_power}')
        focal_train_ds = BirdDatasetWithPL(
            focal_pl_train_df, raw_ensemble, cfg, num_classes, perch_soft,
            perch_caps=perch_caps, noise_padder=noise_padder)
    else:
        focal_train_ds = BirdDataset(train_df_fit, cfg, num_classes, species2idx,
                                     noise_padder=noise_padder)

    sc_only_rarity_w = None   # (M_sc_only,) rarity weights; set in sc_pl substitution path

    # ── Stage 4: soundscape pseudo-label fine-tuning ─────────────────────────
    if cfg.stage == 'sc_pl':
        if not cfg.sc_pl_dir:
            raise ValueError(
                '--sc_pl_dir is required for stage=sc_pl. '
                'Run blend_sc_pl.py first to produce sc_pl_preds_ensemble.npy.')
        sc_pl_dir  = Path(cfg.sc_pl_dir)
        manifest   = pd.read_csv(sc_pl_dir / 'sc_pl.csv')
        ens_preds  = np.load(sc_pl_dir / 'sc_pl_preds_ensemble.npy')

        # Exclude labeled SC val files (no audio leakage into validation)
        keep_mask = ~manifest['filename'].isin(sc_val_fileset)
        # Optionally exclude ALL labeled soundscape files from PL training so that
        # accurate PLs for those chunks cannot inflate the labeled-SC val AUC.
        if cfg.sc_pl_exclude_labelled:
            keep_mask &= ~manifest['filename'].isin(sc_train_fileset)
            print(f'SC PL: excluding {(~keep_mask).sum():,} labeled-soundscape chunks '
                  f'({len(sc_train_fileset):,} train files) from PL training '
                  f'(--sc_pl_exclude_labelled)')
        ens_preds = ens_preds[keep_mask.values]
        manifest  = manifest[keep_mask].reset_index(drop=True)

        # Fixed unlabeled SC validation split: load from sc_pl_val_files.txt when it
        # exists so the holdout is stable across sc_pl rounds regardless of changes to
        # ensemble prediction density (which would change density stratification strata).
        # The file is written on first run (typically round1) and reused thereafter.
        val_files_path = sc_pl_dir / 'sc_pl_val_files.txt'
        if val_files_path.exists():
            saved_val = set(val_files_path.read_text().splitlines())
            sc_unlabeled_val_fileset   = saved_val & set(manifest['filename'].unique())
            sc_unlabeled_train_fileset = set(manifest['filename'].unique()) - sc_unlabeled_val_fileset
            n_dropped = len(saved_val) - len(sc_unlabeled_val_fileset)
            msg = f' ({n_dropped} saved files not in current manifest)' if n_dropped else ''
            print(f'Unlabeled SC val split: loaded {len(sc_unlabeled_val_fileset):,} files '
                  f'from {val_files_path.name}{msg}')
        else:
            sc_unlabeled_train_fileset, sc_unlabeled_val_fileset = \
                stratified_unlabeled_sc_split(manifest, ens_preds,
                                              val_frac=0.10, seed=cfg.seed)
            val_files_path.write_text('\n'.join(sorted(sc_unlabeled_val_fileset)))
            print(f'Unlabeled SC val split: computed {len(sc_unlabeled_val_fileset):,} files, '
                  f'saved to {val_files_path.name}')

        val_uls_mask              = manifest['filename'].isin(sc_unlabeled_val_fileset)
        unlabeled_sc_val_manifest = manifest[val_uls_mask].reset_index(drop=True)
        unlabeled_sc_val_preds    = ens_preds[val_uls_mask.values]

        train_uls_mask = ~val_uls_mask
        manifest       = manifest[train_uls_mask].reset_index(drop=True)
        ens_preds      = ens_preds[train_uls_mask.values]

        print(f'Soundscape PL: {len(manifest):,} chunks  '
              f'({manifest["filename"].nunique():,} train files)')
        print(f'Unlabeled SC holdout: {len(sc_unlabeled_val_fileset):,} val files  '
              f'({len(unlabeled_sc_val_manifest):,} chunks)')
        print(f'PL power transform: prob^{cfg.pl_sc_pseudo_power}  '
              f'[1st-place recipe; round 1=1.0, round 2≈1.54, round 3≈1.82]')

        hard_positive_mask = None
        if cfg.sc_pl_hard_labels:
            labels_csv_path = base_dir / 'train_soundscapes_labels.csv'
            hard_positive_mask = _build_sc_hard_label_mask(
                manifest, labels_csv_path, species2idx, num_classes)
            n_pos     = int(hard_positive_mask.sum())
            n_chunks  = int(hard_positive_mask.any(axis=1).sum())
            n_species = int(hard_positive_mask.any(axis=0).sum())
            print(f'SC hard labels: {n_pos:,} overrides across {n_chunks:,} chunks '
                  f'/ {n_species:,} species (--sc_pl_hard_labels)')

        if cfg.sc_chunk_level_weights:
            # Chunk-level: each chunk weighted by its own max class probability.
            # Directly selects high-confidence individual chunks rather than
            # high-confidence files (which may contain many low-confidence chunks).
            chunk_weights = ens_preds.max(axis=1).astype(np.float32)
            print(f'SC PL sampling: chunk-level weights  '
                  f'(mean={chunk_weights.mean():.3f}, >0.5={(chunk_weights>0.5).mean():.1%})')
        else:
            # File-level (1st-place recipe): weight = sum of per-class max prob across
            # all chunks in the file. Higher-confidence files are sampled more often.
            file_weight_map: dict = {}
            for filename, grp in manifest.groupby('filename'):
                chunk_idx = grp.index.values           # 0-based after reset_index
                file_max  = ens_preds[chunk_idx].max(axis=0)   # (C,) max per class
                file_weight_map[filename] = float(file_max.sum())

            chunk_weights = np.array(
                [file_weight_map[manifest.loc[i, 'filename']] for i in range(len(manifest))],
                dtype=np.float32,
            )
            print(f'SC PL sampling: file-level weights  '
                  f'(mean={chunk_weights.mean():.3f})')
        chunk_weights = np.maximum(chunk_weights, 1e-6)

        # ── SC PL rare-species rarity boost ──────────────────────────────────
        # Per-chunk weight based on argmax-species count in the manifest.
        # Chunks whose dominant species has fewer than sc_pl_upsample_thresh
        # appearances get weight min(cap, thresh/count).  Applied to chunk_weights
        # (MixUp path) and to the WeightedRandomSampler for sc_only_pl_ds
        # (substitution path) so that rare non-Aves species appear in enough
        # batches for SoftAUCLoss to compute ranking gradients.
        sc_top_cls = ens_preds.argmax(axis=1)   # (M,) — dominant species per chunk
        sc_argmax_counts = np.bincount(sc_top_cls, minlength=num_classes).astype(np.float32)
        if cfg.sc_pl_upsample_thresh > 0:
            sc_count_per_chunk = sc_argmax_counts[sc_top_cls]
            sc_rarity_w = np.where(
                sc_count_per_chunk < cfg.sc_pl_upsample_thresh,
                np.minimum(cfg.sc_pl_upsample_cap,
                           cfg.sc_pl_upsample_thresh / np.maximum(sc_count_per_chunk, 1.0)),
                1.0,
            ).astype(np.float32)
            n_boosted = int((sc_rarity_w > 1.0).sum())
            n_species_boosted = int(
                (sc_argmax_counts < cfg.sc_pl_upsample_thresh).sum())
            print(f'SC PL rare-species boost: {n_boosted:,}/{len(manifest):,} chunks '
                  f'({n_species_boosted} species) '
                  f'thresh={cfg.sc_pl_upsample_thresh} cap={cfg.sc_pl_upsample_cap}×')
            chunk_weights = chunk_weights * sc_rarity_w
        else:
            sc_rarity_w = None

        sc_pl_train_ds = SoundscapePLDataset(manifest, ens_preds, soundscape_dir, cfg,
                                              noise_padder=noise_padder,
                                              hard_positive_mask=hard_positive_mask)

        # Labeled soundscape training set (same split as all other stages).
        # Kept separate from pseudo-labeled data so ground-truth labels are never
        # blended with pseudo-labels. Appended via ConcatDataset so the rare-species
        # sampler accounts for labeled-SC slots (important for Insecta/Amphibia that
        # only appear in soundscapes and have no pseudo-label signal in round 2+).
        sc_train_ds = SoundscapeTrainDataset(sc_train_df, soundscape_dir, cfg,
                                             noise_padder=noise_padder)

        # SC PL focal wrapper: either species-matched substitution or blind 50/50 MixUp.
        # Epoch length = len(focal_train_ds); rare-species sampler still applies.
        if cfg.sc_pl_sub_prob > 0.0:
            sc_focal_ds = SoundscapeSubstitutionDataset(
                focal_train_ds, sc_pl_train_ds, cfg.sc_pl_sub_prob)
            print(f'SC PL mode: species-matched substitution (sub_prob={cfg.sc_pl_sub_prob})')

            # SC-only species (Insecta, Amphibia, Mammalia) have no focal clips
            # so are never substitution targets — their PL chunks would go unused.
            # In the MixUp path they were randomly sampled as the SC component.
            # Restore that coverage by adding their PL chunks as direct training items.
            focal_species = set(int(x) for x in focal_train_ds.df['label'].unique())
            top_cls       = ens_preds.argmax(axis=1)           # (M,)
            sc_only_mask  = ~np.isin(top_cls, list(focal_species))
            if sc_only_mask.any():
                sc_only_pl_ds = SoundscapePLDataset(
                    manifest[sc_only_mask].reset_index(drop=True),
                    ens_preds[sc_only_mask],
                    soundscape_dir, cfg, noise_padder=noise_padder,
                    hard_positive_mask=(hard_positive_mask[sc_only_mask]
                                        if hard_positive_mask is not None else None))
                if sc_rarity_w is not None:
                    sc_only_rarity_w = sc_rarity_w[sc_only_mask]
                print(f'SC-only PL chunks added directly: {int(sc_only_mask.sum()):,} '
                      f'({int((~np.isin(np.unique(top_cls[sc_only_mask]), list(focal_species))).sum())} species)')
                combined_train_ds = ConcatDataset([sc_focal_ds, sc_train_ds, sc_only_pl_ds])
            else:
                combined_train_ds = ConcatDataset([sc_focal_ds, sc_train_ds])
        else:
            sc_focal_ds = SoundscapeMixupDataset(focal_train_ds, sc_pl_train_ds, chunk_weights)
            print('SC PL mode: blind 50/50 MixUp')
            combined_train_ds = ConcatDataset([sc_focal_ds, sc_train_ds])
    else:
        sc_train_ds               = SoundscapeTrainDataset(sc_train_df, soundscape_dir, cfg,
                                                           noise_padder=noise_padder)
        combined_train_ds         = ConcatDataset([focal_train_ds, sc_train_ds])
        unlabeled_sc_val_manifest = None
        unlabeled_sc_val_preds    = None

    # _make_sample_weights needs the focal DataFrame that matches focal_train_ds.
    _focal_df_for_weights = (
        focal_pl_train_df if cfg.stage == 'focal_pl' else train_df_fit)
    # Derive _n_sc from combined_train_ds so it stays correct when the substitution
    # branch adds SC-only PL chunks as a third ConcatDataset component.
    _n_sc = len(combined_train_ds) - len(_focal_df_for_weights)

    # Build sc_weights for WeightedRandomSampler: labeled-SC chunks get 1.0;
    # sc_only_pl_ds chunks get rarity-boost weights when sc_pl_upsample_thresh > 0.
    _sc_sampler_weights = None
    if sc_only_rarity_w is not None:
        _sc_sampler_weights = np.concatenate([
            np.ones(len(sc_train_ds), dtype=np.float32),
            sc_only_rarity_w,
        ])

    if cfg.rare_upsample_thresh > 0 or _sc_sampler_weights is not None:
        sample_weights = _make_sample_weights(
            _focal_df_for_weights, _n_sc,
            cfg.rare_upsample_thresh, cfg.rare_upsample_cap,
            sc_weights=_sc_sampler_weights)
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(combined_train_ds), replacement=True)
        train_loader = DataLoader(
            combined_train_ds, batch_size=cfg.batch_size, sampler=sampler,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(
            combined_train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    val_sc_ds    = SoundscapeValDataset(sc_val_df, soundscape_dir, cfg)
    val_focal_ds = FocalValDataset(focal_val_df, cfg, num_classes)

    val_sc_loader = DataLoader(
        val_sc_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True)
    val_focal_loader = DataLoader(
        val_focal_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True)

    if cfg.stage == 'sc_pl':
        print(f'Train batches: {len(train_loader):,}  '
              f'(focal×sc_pl MixUp: {len(focal_train_ds):,} focal + '
              f'{len(sc_pl_train_ds):,} sc_pl pool + '
              f'{len(sc_train_ds):,} labeled sc)')
    else:
        print(f'Train batches: {len(train_loader):,}  '
              f'(focal: {len(focal_train_ds):,}, sc: {len(sc_train_ds):,})')

    # Unlabeled SC val loader (sc_pl stage only)
    if unlabeled_sc_val_manifest is not None:
        val_unlabeled_sc_ds = SoundscapeUnlabeledValDataset(
            unlabeled_sc_val_manifest, unlabeled_sc_val_preds, soundscape_dir, cfg)
        val_unlabeled_sc_loader = DataLoader(
            val_unlabeled_sc_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)
        print(f'Unlabeled SC val loader: {len(val_unlabeled_sc_loader):,} batches')
    else:
        val_unlabeled_sc_loader = None

    # ── Background augmentation ───────────────────────────────────────────────
    nocall_augmenter = None
    if cfg.bg_aug_prob > 0.0:
        if cfg.noise_dir:
            # External verified-abiotic noise (e.g. ESC-50 filtered).
            # Preferred over competition nocall segments for focal loss training:
            # ESC-50 files are truly bird-free; competition nocall segments are
            # only Perch-thresholded and may contain quiet calls that focal loss
            # will amplify as hard negatives.
            nocall_augmenter = ExternalNoiseAugmenter(
                Path(cfg.noise_dir), cfg)
        else:
            nocall_augmenter = _build_nocall_augmenter(
                cfg, base_dir, soundscape_dir, sc_val_fileset)

    # ── Model ─────────────────────────────────────────────────────────────────
    load_pretrained = extra_args.checkpoint is None  # ImageNet pretrained if no KD checkpoint
    model = BirdCLEFModel(cfg, num_classes, pretrained=load_pretrained)

    if extra_args.checkpoint is not None:
        ckpt_path = Path(extra_args.checkpoint)
        state = torch.load(ckpt_path, map_location='cpu')
        # Remap PaSST final-norm keys if checkpoint was saved before _CaptureNorm:
        #   backbone._net.norm.{weight,bias} → backbone._net.norm.norm.{weight,bias}
        # No-op for CNN/ViT checkpoints (no backbone._net.norm key) or for new
        # PaSST checkpoints already saved in _CaptureNorm format.
        _OLD = '_net.norm.'
        _NEW = '_net.norm.norm.'
        state = {
            (k.replace(_OLD, _NEW, 1) if _OLD in k and _NEW not in k else k): v
            for k, v in state.items()
        }
        # Auto-detect checkpoint format:
        #   Full BirdCLEFModel (our swa_model.pth / best_model.pth):
        #       keys start with 'backbone.' / 'fc.' / 'att_fc.' / 'bn.'
        #   Backbone-only (KD stage output, 1st-place weights):
        #       keys are raw backbone parameter names (no 'backbone.' prefix)
        #
        # Loading a full-model checkpoint into model.backbone directly fails:
        # no keys match (backbone starts from scratch) and any BN buffer in the
        # backbone that shares the short name 'bn.*' with BirdCLEFModel's SED-head
        # BN triggers a shape-mismatch RuntimeError even under strict=False.
        # The fix below pre-filters by shape so every copy is guaranteed safe.
        if any(k.startswith('backbone.') for k in state.keys()):
            # Full BirdCLEFModel checkpoint.  Load all tensors whose key and
            # shape match the current model — backbone + head if num_classes
            # is identical, backbone only if the head dimension changed.
            current_state = model.state_dict()
            compatible = {k: v for k, v in state.items()
                          if k in current_state and v.shape == current_state[k].shape}
            current_state.update(compatible)
            model.load_state_dict(current_state)
            n_bb  = sum(1 for k in compatible if k.startswith('backbone.'))
            n_hd  = len(compatible) - n_bb
            n_tot = len(state)
            print(f'Loaded full model from {ckpt_path}  '
                  f'({n_bb} backbone + {n_hd} head tensors / {n_tot} total)')
            skipped = [k for k in state if k not in compatible]
            if skipped:
                print(f'  Skipped {len(skipped)} tensors (shape mismatch — '
                      f'num_classes changed?): {skipped[:4]}'
                      f'{"..." if len(skipped) > 4 else ""}')
        else:
            # Backbone-only checkpoint (KD output, external pretrained weights).
            missing, unexpected = model.backbone.load_state_dict(state, strict=False)
            print(f'Loaded backbone from {ckpt_path}')
            if missing:
                print(f'  Missing ({len(missing)}): {missing[:4]}'
                      f'{"..." if len(missing) > 4 else ""}')
            if unexpected:
                print(f'  Unexpected ({len(unexpected)}): {unexpected[:4]}'
                      f'{"..." if len(unexpected) > 4 else ""}')

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {cfg.backbone}  |  Classes: {num_classes}  |  Params: {total_params:,}')

    # ── Train ─────────────────────────────────────────────────────────────────
    hist_df, best_auc, swa_auc, best_sc_auc = run_supervised_stage(
        cfg, model, train_loader, val_sc_loader, val_focal_loader,
        mel_transform, sc_val_df, focal_val_df, device, output_dir,
        nocall_augmenter=nocall_augmenter,
        proxy_class_indices=proxy_class_indices,
        val_unlabeled_sc_loader=val_unlabeled_sc_loader,
        unlabeled_sc_val_manifest=unlabeled_sc_val_manifest)

    # ── Save run summary ──────────────────────────────────────────────────────
    run_summary = {
        'backbone':           cfg.backbone,
        'stage':              cfg.stage,
        'num_classes':        num_classes,
        'num_epochs':         cfg.num_epochs,
        'best_epoch':         int(hist_df.loc[hist_df['val_auc'].idxmax(), 'epoch']),
        'best_composite_auc': round(best_auc, 4),
        'swa_composite_auc':  round(swa_auc,  4),
        'best_sc_epoch':      int(hist_df.loc[hist_df['val_sc_auc'].idxmax(), 'epoch']),
        'best_sc_auc':        round(best_sc_auc, 4),
        'use_amp':            cfg.use_amp,
        'checkpoint_loaded':  str(extra_args.checkpoint),
    }
    with open(output_dir / 'run_summary.json', 'w') as f:
        json.dump(run_summary, f, indent=2)
    cfg.save_json(output_dir / 'run_config.json')

    print('\nSaved: epoch_history.csv, per_class_auc.csv')
    print(f'       best_model.pth, best_sc_model.pth, swa_model.pth, label_map.npy')
    print(f'       oof_swa_sc.npz, oof_swa_focal.npz')
    print(f'       oof_best_sc.npz, oof_best_focal.npz')
    print(f'       oof_best_sc_sc.npz, oof_best_sc_focal.npz')
    if val_unlabeled_sc_loader is not None:
        print(f'       oof_swa_unlabeled_sc.npz, oof_best_unlabeled_sc.npz, oof_best_sc_unlabeled_sc.npz')
    print(f'       run_summary.json, run_config.json')


if __name__ == '__main__':
    main()
