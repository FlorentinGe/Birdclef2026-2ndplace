#!/usr/bin/env python3
"""
Recover OOF predictions from a saved checkpoint when the SWA finalisation
step failed (e.g. exp41 PaSST — only best_model.pth was saved).

Reconstructs the same train/val split used during training (same seed,
same val fractions) and runs inference with the provided checkpoint.

Usage:
    python extract_oof_from_checkpoint.py \
        --backbone passt_base \
        --checkpoint /path/to/best_model.pth \
        --output_dir /path/to/exp41 \
        [--config /path/to/run_level.yaml]   # optional extra overrides

The script writes:
    oof_best_sc.npz
    oof_best_focal.npz

to the output_dir.  Run_config.json / run_summary.json are NOT written here
(they are produced by train.py); this script is recovery-only.

Note: the --tag flag (default "best") controls the output filename prefix.
Pass --tag swa if you have an swa_model.pth to evaluate instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from birdclef.config import load_config
from birdclef.datasets import FocalValDataset, SoundscapeValDataset
from birdclef.model import BirdCLEFModel
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
)
from birdclef.validate import save_oof_predictions


def main():
    # ── Extra args not in load_config ────────────────────────────────────────
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--checkpoint', required=True,
                        help='Path to .pth state dict (model weights only)')
    parser.add_argument('--tag', default='best',
                        help='Output filename prefix: oof_{tag}_sc.npz  (default: best)')
    extra, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    cfg = load_config()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(cfg.seed)

    print(f'Backbone   : {cfg.backbone}')
    print(f'Checkpoint : {extra.checkpoint}')
    print(f'Output dir : {output_dir}')
    print(f'Tag        : {extra.tag}')
    print(f'Device     : {device}  |  AMP: {cfg.use_amp}')

    # ── Data (identical to train.py) ─────────────────────────────────────────
    base_dir = Path(cfg.base_dir)
    train_df = pd.read_csv(base_dir / 'train.csv')
    sc_df_raw = pd.read_csv(base_dir / 'train_soundscapes_labels.csv')

    audio_dir = cfg.train_audio_path()
    if cfg.train_audio_dir is None and cfg.train_audio_wav_prefix != \
            '/kaggle/input/datasets/ttahara/birdclef2026-train-audio-wav-':
        pl2wav_dir = build_wav_path_map(cfg.train_audio_wav_prefix)
        train_df['file_path'] = resolve_wav_paths(train_df, pl2wav_dir)
    else:
        train_df['file_path'] = resolve_audio_paths(train_df, audio_dir)

    sc_df = normalise_sc_df(sc_df_raw)
    soundscape_dir = cfg.soundscape_path()

    # Same split as training (same seed, same fractions)
    _, sc_val_fileset = stratified_soundscape_split(
        sc_df, cfg.soundscape_val_frac, cfg.seed)
    sc_val_df = sc_df[sc_df['filename'].isin(sc_val_fileset)].reset_index(drop=True)

    all_species, species2idx, _ = build_vocabulary(train_df, sc_df)
    num_classes = len(all_species)
    print(f'Vocabulary : {num_classes} species')

    train_df['label'] = train_df['primary_label'].map(species2idx)
    train_df['secondary_label_vec'] = train_df['secondary_labels'].apply(
        lambda x: encode_multilabel(x, num_classes, species2idx))
    sc_val_df['label_vec'] = sc_val_df['primary_label'].apply(
        lambda x: make_sc_label_vec(x, num_classes, species2idx))

    rng_focal = np.random.default_rng(cfg.seed)
    focal_val_mask = rng_focal.random(len(train_df)) < cfg.focal_val_frac
    focal_val_df = train_df[focal_val_mask].reset_index(drop=True)
    print(f'Val split  : {len(sc_val_fileset)} sc files, {len(focal_val_df):,} focal clips')

    val_sc_ds    = SoundscapeValDataset(sc_val_df, soundscape_dir, cfg)
    val_focal_ds = FocalValDataset(focal_val_df, cfg, num_classes)

    val_sc_loader = DataLoader(
        val_sc_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True)
    val_focal_loader = DataLoader(
        val_focal_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BirdCLEFModel(cfg, num_classes, pretrained=False)
    state = torch.load(extra.checkpoint, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.to(device)
    print(f'Loaded weights from {extra.checkpoint}')

    # ── OOF ──────────────────────────────────────────────────────────────────
    save_oof_predictions(
        model, val_sc_loader, val_focal_loader,
        sc_val_df, focal_val_df, device, num_classes, cfg.use_amp,
        output_dir, tag=extra.tag)

    print('\nDone.')


if __name__ == '__main__':
    main()
