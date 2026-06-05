#!/usr/bin/env python3
"""
Generate focal pseudo-labels by ensembling trained BirdCLEFModel checkpoints.

For each focal clip in train.csv the script:
  1. Loads the full audio file and splits it into non-overlapping windows of
     cfg.duration seconds (pad the last window if it is ≥ 50 % full).
  2. Runs each checkpoint on every window, applies sigmoid, averages across
     windows → per-model clip-level probabilities.
  3. Averages predictions across all PyTorch ensemble members.
  4. Zeros entries below --threshold.
  5. Optionally adds Perch-discovered species via per-clip z-score normalisation
     (z > --perch_z_thresh → soft label at --perch_soft_value). Perch is applied
     as a separate max() after the ensemble, not averaged in, because its absolute
     sigmoid probabilities are dominated by a regional prior and are not comparable
     to the PyTorch models' calibrated outputs.
  6. Takes max(hard_label_vec, soft_preds):
       - primary label stays at 1.0
       - secondary labels from train.csv stay at 1.0
       - newly discovered species get their raw sigmoid value

Each model is loaded from its run directory, which must contain:
  swa_model.pth   (falls back to best_model.pth if absent)
  run_config.json (written by train.py; carries backbone, sr, duration, …)
  label_map.npy   (idx2species dict; must be identical across all models)

Outputs written to --out_dir:
  focal_pl_vecs.npy  — (N, num_classes) float32 soft label vectors,
                        row order matches focal_pl.csv
  focal_pl.csv       — filename, file_path, primary_label, vec_idx

Usage
-----
python generate_focal_pl.py \\
    --model_dirs  runs/eca_nfnet_l0_supervised \\
                  runs/regnety_032_supervised \\
                  runs/tf_efficientnetv2_s_supervised \\
                  runs/vit_base_patch16_224.dino_supervised \\
    --base_dir    /path/to/birdclef-2026 \\
    --out_dir     runs/focal_pl \\
    [--perch_npz  /path/to/perch_focal_arrays.npz] \\
    [--threshold  0.1] \\
    [--batch_size 32] \\
    [--device     cuda]

Perch npz format (if provided)
-------------------------------
The .npz must have two keys:
  filenames : (N,) str array — matches the 'filename' column in train.csv,
              e.g. "aldfly/XC1234.ogg"
  logits    : (N, num_classes) float32 — raw logits mapped to the competition
              vocabulary (output of the Perch focal-clip extraction script).
              Sigmoid is applied internally; do NOT pass probabilities here.

Clips with no matching entry in the Perch npz receive zero Perch contribution
and are still covered by the PyTorch model ensemble.

Notes
-----
- All model run directories must share the same vocabulary (label_map.npy).
  The script asserts this before running any inference.
- Per-model prediction arrays are saved as focal_pl_preds_<model_name>.npy
  for debugging. Delete them after use to save disk space.
- The produced focal_pl.csv + focal_pl_vecs.npy are consumed by
  BirdDatasetWithPL in datasets.py (to be wired into train.py stage focal_pl).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from birdclef.config import Config
from birdclef.model import BirdCLEFModel
from birdclef.transforms import MelTransform
from birdclef.utils import (
    build_wav_path_map,
    encode_multilabel,
    resolve_audio_paths,
    resolve_wav_paths,
)


# ── Config loading ────────────────────────────────────────────────────────────

def _load_cfg_from_json(run_dir: Path) -> Config:
    """
    Rebuild a Config from the run_config.json saved by train.py.

    Also handles the old notebook format used by exp18 and earlier, where the
    config was saved as {"cfg": {"MODEL_NAME": "eca_nfnet_l0", "SR": "32000", ...}}.
    That format uses uppercase keys and stores all values as strings.
    """
    # Mapping from old notebook uppercase keys → Config attribute names.
    # Only the fields that affect inference are listed; others are ignored.
    _OLD_KEY_MAP = {
        'MODEL_NAME':      'backbone',
        'SR':              'sr',
        'DURATION':        'duration',
        'CHUNK_DURATION':  'chunk_duration',
        'N_FFT':           'n_fft',
        'HOP_LENGTH':      'hop_length',
        'N_MELS':          'n_mels',
        'FMIN':            'fmin',
        'FMAX':            'fmax',
        'DROP_PATH_RATE':  'drop_path_rate',
        'USE_AMP':         'use_amp',
        'BATCH_SIZE':      'batch_size',
        'NUM_WORKERS':     'num_workers',
        'FREQ_MASK':       'freq_mask',
        'TIME_MASK':       'time_mask',
    }

    def _coerce(v):
        """Convert a string value from the old notebook format to the right type."""
        if isinstance(v, str):
            if v.lower() == 'true':  return True
            if v.lower() == 'false': return False
            try: return int(v)
            except ValueError: pass
            try: return float(v)
            except ValueError: pass
        return v

    with open(run_dir / 'run_config.json') as f:
        d = json.load(f)

    cfg = Config()

    # ── Old notebook format: {"cfg": {"MODEL_NAME": ..., "SR": ...}, ...} ──
    if 'cfg' in d and isinstance(d['cfg'], dict) and 'MODEL_NAME' in d['cfg']:
        old = d['cfg']
        for old_key, new_key in _OLD_KEY_MAP.items():
            if old_key in old:
                try:
                    setattr(cfg, new_key, _coerce(old[old_key]))
                except AttributeError:
                    pass
        # The old format never saved imagenet_norm, use_bf16, in_chans, use_gem,
        # or other fields added later.  Overlay the backbone YAML to recover them —
        # this matches what train.py does and is the authoritative source for these
        # per-backbone defaults (e.g. imagenet_norm: true for ViT DINO).
        _CONFIGS_DIR = Path(__file__).parent / 'configs'
        backbone_yaml = _CONFIGS_DIR / 'backbone' / f'{cfg.backbone}.yaml'
        if backbone_yaml.exists():
            try:
                import yaml
                with open(backbone_yaml) as _f:
                    yaml_vals = yaml.safe_load(_f) or {}
                # Only apply fields relevant to inference; skip training-only ones
                # (num_epochs, lr, swa_start_epoch, etc.) so they don't shadow the
                # experiment's actual settings.
                _INFERENCE_FIELDS = {
                    'imagenet_norm', 'use_bf16', 'use_amp', 'in_chans',
                    'use_gem', 'gem_p_init', 'drop_path_rate',
                }
                for k, v in yaml_vals.items():
                    if k in _INFERENCE_FIELDS:
                        try:
                            setattr(cfg, k, v)
                        except AttributeError:
                            pass
            except ImportError:
                pass   # pyyaml not installed; backbone YAML skipped
        return cfg

    # ── Current train.py format: flat dict with lowercase keys ──
    for k, v in d.items():
        try:
            setattr(cfg, k, v)
        except AttributeError:
            pass   # skip read-only properties (img_size, img_size_chunk, is_vit)
    return cfg


# ── Per-clip inference ────────────────────────────────────────────────────────

@torch.inference_mode()
def _predict_one_clip(
    model: BirdCLEFModel,
    mel_transform: MelTransform,
    audio_path: str,
    target_samples: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """
    Load the full audio file, split into non-overlapping windows of
    target_samples, run the model on each window, return the mean
    sigmoid probability vector.

    Window rules:
      - Clip shorter than target_samples: padded to target_samples (1 window).
      - Clip longer than target_samples: non-overlapping complete windows only;
        the trailing partial window is included if it is ≥ 50 % of target_samples
        (padded to full length), dropped otherwise.

    Returns a (num_classes,) float32 array.
    """
    try:
        wave, _ = soundfile.read(audio_path, dtype='float32', always_2d=False)
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
    except Exception:
        return np.zeros(model.num_classes, dtype=np.float32)

    absmax = np.abs(wave).max()
    if absmax > 0:
        wave /= absmax

    n = len(wave)
    windows: List[torch.Tensor] = []

    if n < target_samples:
        padded = np.pad(wave, (0, target_samples - n))
        windows.append(torch.from_numpy(padded))
    else:
        # Complete non-overlapping windows
        n_complete = n // target_samples
        for k in range(n_complete):
            start = k * target_samples
            windows.append(torch.from_numpy(wave[start : start + target_samples].copy()))
        # Trailing partial window — include if ≥ 50 % full
        remainder = wave[n_complete * target_samples :]
        if len(remainder) >= target_samples // 2:
            padded = np.pad(remainder.copy(), (0, target_samples - len(remainder)))
            windows.append(torch.from_numpy(padded))

    all_probs: List[np.ndarray] = []
    for i in range(0, len(windows), batch_size):
        batch = torch.stack(windows[i : i + batch_size]).to(device)  # (B, T)
        specs  = mel_transform(batch, augment=False)                  # (B, C, H, W)
        _, att_clip, _ = model(specs)
        all_probs.append(torch.sigmoid(att_clip).float().cpu().numpy())

    return np.concatenate(all_probs, axis=0).mean(axis=0)   # (num_classes,)


# ── Full-dataset inference for one model ─────────────────────────────────────

def _predict_all_clips(
    model_dir: Path,
    file_paths: List[str],
    num_classes: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """
    Load one model checkpoint and run it over all focal clips.

    Returns a (N, num_classes) float32 array of sigmoid probabilities.
    The model and MelTransform are deleted after use to free GPU memory.
    """
    cfg = _load_cfg_from_json(model_dir)
    print(f'  backbone      : {cfg.backbone}')
    print(f'  duration      : {cfg.duration}s  '
          f'imagenet_norm={cfg.imagenet_norm}  use_amp={cfg.use_amp}')

    # Prefer SWA checkpoint; fall back to best-epoch if absent
    ckpt = model_dir / 'swa_model.pth'
    if not ckpt.exists():
        ckpt = model_dir / 'best_model.pth'
        print(f'  WARNING: swa_model.pth not found — using best_model.pth')
    print(f'  checkpoint    : {ckpt.name}')

    model = BirdCLEFModel(cfg, num_classes, pretrained=False)
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)

    mel_transform  = MelTransform(cfg).to(device)
    target_samples = cfg.sr * cfg.duration
    N              = len(file_paths)
    preds          = np.zeros((N, num_classes), dtype=np.float32)

    for i, fp in enumerate(tqdm(file_paths, desc='  clips', leave=False)):
        preds[i] = _predict_one_clip(
            model, mel_transform, fp, target_samples, device, batch_size)

    # Summary
    max_per_clip = preds.max(axis=1)
    print(f'  mean max prob : {max_per_clip.mean():.4f}  '
          f'p10={np.percentile(max_per_clip, 10):.4f}  '
          f'p90={np.percentile(max_per_clip, 90):.4f}')

    del model, mel_transform
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return preds


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    # Required
    parser.add_argument('--model_dirs', nargs='+', required=True,
                        help='Run directories: each must contain swa_model.pth, '
                             'run_config.json, label_map.npy')
    parser.add_argument('--base_dir', required=True,
                        help='Competition root directory (contains train.csv)')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for focal_pl.csv and focal_pl_vecs.npy')

    # Audio source (mutually exclusive with wav_prefix)
    parser.add_argument('--train_audio_dir', default=None,
                        help='OGG train_audio directory (default: base_dir/train_audio). '
                             'Ignored when --wav_prefix is set.')
    parser.add_argument('--wav_prefix', default=None,
                        help='ttahara WAV shard prefix '
                             '(e.g. /path/birdclef2026-train-audio-wav-). '
                             'When set, all 4 shards (00–03) are searched for each file.')

    # Perch
    parser.add_argument('--perch_npz', default=None,
                        help='Optional: .npz with keys "filenames" and "logits" '
                             'from a separate Perch focal-clip extraction run.')
    parser.add_argument('--perch_z_thresh', type=float, default=2.0,
                        help='Per-clip z-score threshold for Perch discovery '
                             '(default: 2.0). Perch logits are z-score normalised '
                             'per clip to remove the regional prior; only species '
                             'whose logit exceeds this many std above the clip mean '
                             'are treated as discovered. Lower = more recalls, '
                             'higher = fewer false positives.')
    parser.add_argument('--perch_soft_value', type=float, default=0.3,
                        help='Soft label value assigned to Perch-discovered species '
                             '(default: 0.3). Applied via max() after the PyTorch '
                             'ensemble, so it only adds species the ensemble missed.')

    # Inference settings
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Zero out ensemble probs below this value (default: 0.1)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Window batch size per model forward pass (default: 32)')
    parser.add_argument('--device', default=None,
                        help='Torch device string (default: cuda if available, else cpu)')

    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    device    = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    out_dir   = Path(args.out_dir)
    base_dir  = Path(args.base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dirs = [Path(d) for d in args.model_dirs]

    print('=' * 60)
    print('Focal pseudo-label generation')
    print('=' * 60)
    print(f'Device     : {device}')
    print(f'Models     : {len(model_dirs)}')
    print(f'Threshold  : {args.threshold}  (PyTorch ensemble)')
    if args.perch_npz:
        print(f'Perch      : z>{args.perch_z_thresh}  soft={args.perch_soft_value}')
    print(f'Batch size : {args.batch_size}')
    print(f'Output     : {out_dir}')

    # ── Vocabulary ───────────────────────────────────────────────────────────
    ref_label_map = model_dirs[0] / 'label_map.npy'
    if not ref_label_map.exists():
        raise FileNotFoundError(f'label_map.npy not found in {model_dirs[0]}')

    idx2species: Dict[int, str] = np.load(ref_label_map, allow_pickle=True).item()

    for md in model_dirs[1:]:
        other = np.load(md / 'label_map.npy', allow_pickle=True).item()
        if other != idx2species:
            raise ValueError(
                f'Vocabulary mismatch:\n  ref: {model_dirs[0]}\n  got: {md}\n'
                'All model directories must share the same label_map.npy.')

    num_classes = len(idx2species)
    species2idx = {v: k for k, v in idx2species.items()}
    print(f'Vocabulary : {num_classes} species')

    # ── Load train.csv + resolve audio paths ─────────────────────────────────
    train_df = pd.read_csv(base_dir / 'train.csv')

    if args.wav_prefix:
        pl2wav_dir = build_wav_path_map(args.wav_prefix)
        train_df['file_path'] = resolve_wav_paths(train_df, pl2wav_dir)
        print(f'Audio      : WAV shards  ({args.wav_prefix}*)')
    else:
        audio_dir = Path(args.train_audio_dir) if args.train_audio_dir \
                    else base_dir / 'train_audio'
        train_df['file_path'] = resolve_audio_paths(train_df, audio_dir)
        print(f'Audio      : {audio_dir}')

    N          = len(train_df)
    file_paths = train_df['file_path'].tolist()
    print(f'Focal clips: {N:,}')

    # ── Build hard label vectors ──────────────────────────────────────────────
    # primary label = 1.0 (always hard), secondary labels from train.csv = 1.0
    print('\nBuilding hard label vectors from train.csv secondary_labels ...')
    hard_vecs = np.zeros((N, num_classes), dtype=np.float32)
    for i, row in enumerate(train_df.itertuples(index=False)):
        primary_idx = species2idx.get(row.primary_label)
        if primary_idx is None:
            print(f'  WARNING: primary_label "{row.primary_label}" not in vocabulary '
                  f'(row {i}) — skipping hard label')
        else:
            hard_vecs[i, primary_idx] = 1.0
        sec = encode_multilabel(
            getattr(row, 'secondary_labels', None), num_classes, species2idx)
        hard_vecs[i] = np.maximum(hard_vecs[i], sec)

    hard_positives_per_clip = (hard_vecs > 0).sum(axis=1)
    print(f'  mean hard positives per clip: {hard_positives_per_clip.mean():.2f}')

    # ── Per-model inference ───────────────────────────────────────────────────
    model_preds: List[np.ndarray] = []

    for md in model_dirs:
        print(f'\nRunning model: {md.name}')
        preds = _predict_all_clips(md, file_paths, num_classes, device, args.batch_size)

        # Save per-model predictions for debugging
        per_model_path = out_dir / f'focal_pl_preds_{md.name}.npy'
        np.save(per_model_path, preds)
        print(f'  saved → {per_model_path.name}')

        model_preds.append(preds)

    # ── Optional: Perch discovery (NOT averaged into ensemble) ───────────────
    # Perch logits have a strong regional prior for Pantanal species — the
    # absolute sigmoid probabilities are meaningless (median ~0.68 for all 209
    # mapped classes on any tropical recording).  Instead we z-score normalise
    # each clip's logit vector and treat species with z > perch_z_thresh as
    # "discovered".  Perch is applied as a separate max() AFTER the PyTorch
    # ensemble so it can only ADD species the ensemble missed, never inflate
    # probabilities of species the ensemble already found.
    PERCH_SENTINEL = -20.0   # value written for unmapped species
    perch_discovery = np.zeros((N, num_classes), dtype=np.float32)

    if args.perch_npz:
        perch_path = Path(args.perch_npz)
        print(f'\nLoading Perch logits: {perch_path}')
        perch_data      = np.load(perch_path, allow_pickle=True)
        perch_filenames = perch_data['filenames']
        perch_logits    = perch_data['logits'].astype(np.float32)  # (M, num_classes)

        if perch_logits.shape[1] != num_classes:
            raise ValueError(
                f'Perch logits have {perch_logits.shape[1]} classes '
                f'but vocabulary has {num_classes}. '
                'Re-run the Perch focal extraction with the current vocabulary.')

        perch_fn_to_row: Dict[str, int] = {
            str(fn): i for i, fn in enumerate(perch_filenames)}

        n_matched = 0
        for i, fn in enumerate(train_df['filename']):
            row_idx = perch_fn_to_row.get(str(fn))
            if row_idx is None:
                continue
            raw  = perch_logits[row_idx]              # (num_classes,)
            mask = raw != PERCH_SENTINEL              # mapped species only
            if mask.sum() < 2:
                continue
            # Per-clip z-score to remove regional prior
            mu  = raw[mask].mean()
            sig = raw[mask].std()
            sig = max(sig, 0.1)                       # floor: avoid division by ~0
            z   = np.where(mask, (raw - mu) / sig, -999.0)
            # Species above threshold contribute a fixed soft label value
            perch_discovery[i] = np.where(
                z > args.perch_z_thresh, args.perch_soft_value, 0.0)
            n_matched += 1

        n_unmatched = N - n_matched
        print(f'  matched  : {n_matched:,} / {N:,} clips')
        print(f'  z-thresh : {args.perch_z_thresh}  soft-value : {args.perch_soft_value}')
        n_disc = (perch_discovery > 0).any(axis=1).sum()
        mean_disc = (perch_discovery > 0).sum(axis=1).mean()
        print(f'  clips with ≥1 discovered species : {n_disc:,}  '
              f'({100*n_disc/N:.1f}%)')
        print(f'  mean discovered per clip         : {mean_disc:.2f}')
        if n_unmatched:
            print(f'  unmatched (no Perch contribution): {n_unmatched:,}')

        perch_save_path = out_dir / 'focal_pl_preds_perch.npy'
        np.save(perch_save_path, perch_discovery)
        print(f'  saved → {perch_save_path.name}')

    # ── Ensemble + threshold (PyTorch models only) ────────────────────────────
    n_members = len(model_preds)
    print(f'\nEnsembling {n_members} PyTorch member(s) (equal weights) ...')
    ensemble = np.mean(model_preds, axis=0)   # (N, num_classes) float32

    # Zero out below threshold; keep values as soft labels above threshold
    soft = np.where(ensemble >= args.threshold, ensemble, 0.0).astype(np.float32)

    # Perch adds species the ensemble missed (max, so it never lowers a prob)
    soft = np.maximum(soft, perch_discovery)

    # Hard labels always win; soft labels only fill positions that were 0
    label_vecs = np.maximum(hard_vecs, soft)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    newly_added  = (soft > 0) & (hard_vecs == 0)
    n_enriched   = newly_added.any(axis=1).sum()
    mean_new     = newly_added.sum(axis=1).mean()
    print(f'\nDiagnostics:')
    print(f'  Clips enriched with ≥1 new PL species : '
          f'{n_enriched:,} / {N:,}  ({100 * n_enriched / N:.1f}%)')
    print(f'  Mean new PL species per clip           : {mean_new:.2f}')
    print(f'  Mean total positives per clip          : '
          f'{(label_vecs > 0).sum(axis=1).mean():.2f}')

    # Per-class enrichment summary (top 10 most-added species)
    new_per_class  = newly_added.sum(axis=0)
    top10_idx      = new_per_class.argsort()[::-1][:10]
    print(f'\n  Top 10 most-discovered PL species:')
    for rank, ci in enumerate(top10_idx, 1):
        print(f'    {rank:2d}. {idx2species[ci]:<12s}  +{new_per_class[ci]:,} clips')

    # ── Save outputs ──────────────────────────────────────────────────────────
    vecs_path = out_dir / 'focal_pl_vecs.npy'
    np.save(vecs_path, label_vecs)

    csv_path = out_dir / 'focal_pl.csv'
    manifest = train_df[['filename', 'file_path', 'primary_label']].copy()
    manifest['vec_idx'] = np.arange(N, dtype=np.int32)
    manifest.to_csv(csv_path, index=False)

    print(f'\nOutputs:')
    print(f'  {vecs_path}  '
          f'shape={label_vecs.shape}  {label_vecs.nbytes / 1e6:.1f} MB')
    print(f'  {csv_path}')
    print(f'\nTo consume in training:')
    print(f'  vecs = np.load("{vecs_path}")')
    print(f'  df   = pd.read_csv("{csv_path}")')
    print(f'  # merge df with train_df on filename; index vecs by vec_idx column')
    print(f'  # or pass --focal_pl_csv {csv_path} to train.py once wired up')


if __name__ == '__main__':
    main()
