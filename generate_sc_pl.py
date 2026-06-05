#!/usr/bin/env python3
"""
Generate soundscape pseudo-labels by running trained BirdCLEFModel checkpoints
on the unlabelled soundscape pool.

Design
------
Unlike generate_focal_pl.py (which outputs a single blended label array), this
script saves **raw per-model sigmoid probabilities only** — no ensembling, no
power transform, no blending with hard labels.  All of that happens at training
time inside SoundscapePLDataset, so weights, power, and alpha can be changed
without re-running inference.

For Perch predictions the script maps the competition-provided
`train_soundscape_pseudolabels.csv` to the training vocabulary and saves a
separate array.  Because Perch's raw probabilities are dominated by a regional
prior, they are saved as-is and treated as a separate teacher at training time
(same z-score normalisation used for focal PL applies there).

Outputs written to --out_dir
-----------------------------
  sc_pl.csv                    — chunk manifest: filename, start_sec, end_sec
                                 Row i ↔ row i in every prediction array.
  sc_pl_preds_{model_name}.npy — (M, num_classes) float32 sigmoid probs per model
  sc_pl_preds_perch.npy        — (M, num_classes) float32 from Perch CSV
                                 (only written when --perch_csv is provided)

Usage
-----
  python generate_sc_pl.py \\
      --model_dirs runs/exp37b runs/exp38 runs/exp41 \\
      --base_dir   /path/to/birdclef-2026 \\
      --soundscape_dir /path/to/train_soundscapes \\
      --out_dir    runs/sc_pl_round1 \\
      [--perch_csv /path/to/train_soundscape_pseudolabels.csv] \\
      [--exclude_labelled] \\
      [--batch_size 32] \\
      [--device     cuda]

Resumable: if sc_pl_preds_{model_name}.npy already exists in --out_dir the
model is skipped.  The manifest sc_pl.csv is always re-verified for consistency.

Training consumption
--------------------
At training start (in train.py / SoundscapePLDataset):
  1. Load all sc_pl_preds_*.npy arrays.
  2. Compute ensemble mean (with per-model weights if desired).
  3. Apply power transform: p_t = p^power.
  4. Blend with labelled-soundscape hard labels where available.
  5. Use as soft targets for SoundscapePLDataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile
import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from birdclef.model import BirdCLEFModel
from birdclef.transforms import MelTransform
from generate_focal_pl import _load_cfg_from_json


# ── PaSST key remapping ───────────────────────────────────────────────────────

def _remap_passt_keys(state: dict) -> dict:
    """
    Remap checkpoint keys saved before _CaptureNorm was introduced.
      old: backbone._net.norm.{weight,bias}
      new: backbone._net.norm.norm.{weight,bias}
    No-op for CNN/ViT checkpoints (no _net.norm key).
    """
    OLD, NEW = '_net.norm.', '_net.norm.norm.'
    return {
        (k.replace(OLD, NEW, 1) if OLD in k and NEW not in k else k): v
        for k, v in state.items()
    }


# ── Manifest building ─────────────────────────────────────────────────────────

def _build_manifest(
    soundscape_dir: Path,
    chunk_duration: int,
    exclude_labelled: bool,
    labelled_set: Optional[set],   # {(filename, start_sec)} with hard labels
) -> pd.DataFrame:
    """
    Scan all OGG files in soundscape_dir and enumerate non-overlapping
    chunk_duration-second chunks.

    Files are read with soundfile.info() (no audio loaded) to get duration.
    Chunks are included only if they are fully within the file duration.

    Args:
        exclude_labelled: when True, omit chunks that have hard labels in
            train_soundscapes_labels.csv (i.e. (filename, start_sec) in
            labelled_set).  Default is False — include all chunks so the
            training side can decide how to blend hard and soft labels.
    """
    rows: List[dict] = []
    ogg_files = sorted(soundscape_dir.glob('*.ogg'))
    if not ogg_files:
        raise FileNotFoundError(f'No .ogg files found in {soundscape_dir}')

    for fpath in tqdm(ogg_files, desc='Building manifest', leave=False):
        try:
            info       = soundfile.info(str(fpath))
            duration_s = info.frames / info.samplerate
        except Exception as e:
            print(f'  WARNING: cannot read {fpath.name}: {e}')
            continue

        n_chunks = int(duration_s // chunk_duration)
        for k in range(n_chunks):
            start = k * chunk_duration
            end   = start + chunk_duration
            if exclude_labelled and labelled_set is not None:
                if (fpath.name, start) in labelled_set:
                    continue
            rows.append({
                'filename':  fpath.name,
                'start_sec': start,
                'end_sec':   end,
            })

    if not rows:
        raise RuntimeError(
            'Manifest is empty. Check --soundscape_dir and --exclude_labelled.')

    return pd.DataFrame(rows, dtype=object).astype(
        {'start_sec': int, 'end_sec': int})


# ── Per-model inference ───────────────────────────────────────────────────────

def _load_model(
    model_dir: Path,
    num_classes: int,
    device: torch.device,
    checkpoint: str = 'swa',
) -> tuple[BirdCLEFModel, MelTransform, 'Config']:
    """
    Load one BirdCLEFModel from its run directory.

    checkpoint='swa' (default): tries swa_model.pth, falls back to best_model.pth.
    checkpoint='best': loads best_model.pth directly.
    checkpoint='best_sc': loads best_sc_model.pth directly.
    Applies the PaSST _CaptureNorm key remap for exp41-style checkpoints.
    """
    cfg = _load_cfg_from_json(model_dir)

    if checkpoint == 'best':
        ckpt = model_dir / 'best_model.pth'
        if not ckpt.exists():
            raise FileNotFoundError(f'best_model.pth not found in {model_dir}')
    elif checkpoint == 'best_sc':
        ckpt = model_dir / 'best_sc_model.pth'
        if not ckpt.exists():
            raise FileNotFoundError(f'best_sc_model.pth not found in {model_dir}')
    else:
        ckpt = model_dir / 'swa_model.pth'
        if not ckpt.exists():
            ckpt = model_dir / 'best_model.pth'
            print(f'    WARNING: swa_model.pth not found — using best_model.pth')
    print(f'    backbone      : {cfg.backbone}')
    print(f'    checkpoint    : {ckpt.name}')
    print(f'    hop_length    : {cfg.hop_length}')
    print(f'    duration      : {cfg.duration}s  (training clip)')
    print(f'    img_size      : {cfg.img_size}  (training)')
    print(f'    img_size_chunk: {cfg.img_size_chunk}  (5s inference)')

    model = BirdCLEFModel(cfg, num_classes, pretrained=False)
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    state = _remap_passt_keys(state)
    model.load_state_dict(state)
    model.eval().to(device)

    mel_transform = MelTransform(cfg).to(device)
    return model, mel_transform, cfg


@torch.inference_mode()
def _predict_all_chunks(
    model_dir: Path,
    manifest: pd.DataFrame,
    soundscape_dir: Path,
    num_classes: int,
    device: torch.device,
    batch_size: int,
    checkpoint: str = 'swa',
    tta: bool = False,
    tta_shift_s: float = 1.25,
) -> np.ndarray:
    """
    Run one model on every chunk in the manifest.

    Audio is loaded file-by-file (one soundfile.read per file), then all
    chunks for that file are batched into a single forward pass.

    When tta=True, each chunk is also evaluated at ±tta_shift_s circular rolls
    and the three sigmoid outputs are averaged before storing.

    Returns (M, num_classes) float32 array of sigmoid probabilities.
    """
    model, mel_transform, cfg = _load_model(
        model_dir, num_classes, device, checkpoint=checkpoint)
    chunk_len = cfg.sr * cfg.chunk_duration
    tta_shift = int(tta_shift_s * cfg.sr) if tta else 0

    M     = len(manifest)
    preds = np.zeros((M, num_classes), dtype=np.float32)

    # Group consecutive rows by filename (manifest is sorted by filename)
    files_in_order = manifest['filename'].unique()

    for filename in tqdm(files_in_order, desc='  files', leave=False):
        fpath = soundscape_dir / filename
        grp   = manifest[manifest['filename'] == filename]

        try:
            wave, orig_sr = soundfile.read(
                str(fpath), dtype='float32', always_2d=False)
            if wave.ndim > 1:
                wave = wave.mean(axis=1)
            if orig_sr != cfg.sr:
                wave = torchaudio.functional.resample(
                    torch.from_numpy(wave), orig_sr, cfg.sr).numpy()
            absmax = np.abs(wave).max()
            if absmax > 0:
                wave /= absmax
        except Exception as e:
            print(f'    WARNING: failed to load {filename}: {e}')
            continue

        # Extract chunks for this file
        global_indices: List[int] = []
        chunk_tensors:  List[torch.Tensor] = []

        for row_idx, row in grp.iterrows():
            start_sample = int(row['start_sec'] * cfg.sr)
            end_sample   = start_sample + chunk_len
            chunk        = wave[start_sample:end_sample]
            if len(chunk) < chunk_len:
                chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
            global_indices.append(row_idx)
            chunk_tensors.append(torch.from_numpy(chunk.copy()))

        if not chunk_tensors:
            continue

        # Batch inference over all chunks for this file
        for b_start in range(0, len(chunk_tensors), batch_size):
            b_chunks  = chunk_tensors[b_start : b_start + batch_size]
            b_indices = global_indices[b_start : b_start + batch_size]

            if tta:
                # Interleave original + two circular rolls; reshape after forward pass.
                views: List[torch.Tensor] = []
                for ct in b_chunks:
                    arr = ct.numpy()
                    views.append(ct)
                    views.append(torch.from_numpy(np.roll(arr,  tta_shift).copy()))
                    views.append(torch.from_numpy(np.roll(arr, -tta_shift).copy()))
                b_waves = torch.stack(views).to(device)          # (B*3, chunk_len)
                specs   = mel_transform(b_waves, augment=False)
                _, att_clip, _ = model(specs)
                # (B*3, C) → (B, 3, C) → mean over TTA dim
                probs = (torch.sigmoid(att_clip)
                         .float().cpu().numpy()
                         .reshape(len(b_chunks), 3, num_classes)
                         .mean(axis=1))
            else:
                b_waves = torch.stack(b_chunks).to(device)
                specs   = mel_transform(b_waves, augment=False)
                _, att_clip, _ = model(specs)
                probs = torch.sigmoid(att_clip).float().cpu().numpy()

            for k, g_idx in enumerate(b_indices):
                preds[g_idx] = probs[k]

    # Summary stats
    max_per_chunk = preds.max(axis=1)
    print(f'    mean max-prob : {max_per_chunk.mean():.4f}  '
          f'p10={np.percentile(max_per_chunk, 10):.4f}  '
          f'p90={np.percentile(max_per_chunk, 90):.4f}')

    del model, mel_transform
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return preds


# ── Perch predictions from competition CSV ────────────────────────────────────

def _parse_perch_predictions(
    perch_csv: Path,
    manifest: pd.DataFrame,
    species2idx: Dict[str, int],
    num_classes: int,
) -> np.ndarray:
    """
    Map the competition's train_soundscape_pseudolabels.csv to our vocabulary.

    The CSV has one row per 5s chunk identified by
        row_id = {stem}_{end_sec}   (e.g. BC2026_Train_0001_S08_…_5)
    and one column per Perch species (ebird codes).  Values are Perch
    probabilities (already sigmoid-applied).

    Chunks not present in the CSV receive a zero vector.

    Returns (M, num_classes) float32.
    """
    print(f'  Loading Perch CSV: {perch_csv.name}')
    perch_df = pd.read_csv(perch_csv)

    # Build set of Perch columns that are in our vocabulary
    perch_species_cols = [c for c in perch_df.columns if c != 'row_id']
    col_to_idx: Dict[str, int] = {}
    for col in perch_species_cols:
        idx = species2idx.get(col)
        if idx is not None:
            col_to_idx[col] = idx

    n_mapped = len(col_to_idx)
    print(f'  Perch columns mapped to vocabulary: {n_mapped} / {len(perch_species_cols)}')

    # Index Perch rows by row_id for fast lookup
    perch_df = perch_df.set_index('row_id')

    M     = len(manifest)
    preds = np.zeros((M, num_classes), dtype=np.float32)
    n_matched = 0

    for row_idx, row in manifest.iterrows():
        stem    = Path(row['filename']).stem
        end_sec = int(row['end_sec'])
        row_id  = f'{stem}_{end_sec}'

        if row_id not in perch_df.index:
            continue

        perch_row = perch_df.loc[row_id]
        for col, our_idx in col_to_idx.items():
            val = perch_row.get(col, 0.0)
            if pd.notna(val):
                preds[row_idx, our_idx] = float(val)
        n_matched += 1

    pct_matched = 100.0 * n_matched / M if M > 0 else 0.0
    print(f'  Chunks matched in Perch CSV: {n_matched:,} / {M:,}  ({pct_matched:.1f}%)')
    if pct_matched < 50.0:
        print('  WARNING: fewer than 50% of chunks matched. Check that --perch_csv '
              'covers the same soundscape files as --soundscape_dir.')

    # Summary of Perch confidence
    max_per_chunk = preds.max(axis=1)
    has_signal    = max_per_chunk > 0
    if has_signal.any():
        print(f'  Chunks with ≥1 non-zero Perch prediction : '
              f'{has_signal.sum():,}  ({100*has_signal.mean():.1f}%)')
        print(f'  Mean max-prob (matched chunks) : '
              f'{max_per_chunk[has_signal].mean():.4f}')

    return preds


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    # Required
    parser.add_argument('--model_dirs', nargs='+', required=True,
                        help='Run directories: each must contain a model '
                             'checkpoint, run_config.json, and label_map.npy')
    parser.add_argument('--base_dir', required=True,
                        help='Competition root (contains train_soundscapes_labels.csv)')
    parser.add_argument('--soundscape_dir', required=True,
                        help='Directory of OGG soundscape files')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for manifest and prediction arrays')

    # Perch
    parser.add_argument('--perch_csv', default=None,
                        help='Path to train_soundscape_pseudolabels.csv (competition '
                             'Perch predictions).  When provided, sc_pl_preds_perch.npy '
                             'is written.  Note: Perch raw probabilities have a strong '
                             'regional prior — apply z-score normalisation at training '
                             'time (same as focal PL).')

    # Manifest options
    parser.add_argument('--exclude_labelled', action='store_true',
                        help='Omit chunks that have hard labels in '
                             'train_soundscapes_labels.csv.  Default: include all '
                             'chunks so the training side can blend hard and soft labels.')

    # Checkpoint selection
    parser.add_argument('--checkpoints', nargs='*',
                        choices=['swa', 'best', 'best_sc'],
                        default=None,
                        metavar='{swa,best,best_sc}',
                        help='Checkpoint to load per model dir: "swa", "best", or '
                             '"best_sc". Must match the number of --model_dirs entries. '
                             'Default (omitted): prefer swa_model.pth, fall back '
                             'to best_model.pth. Use "best_sc" for models where the '
                             'best_sc_model.pth checkpoint is preferred (e.g. PaSST '
                             'models where best_sc gives higher soundscape AUC).')

    # TTA
    parser.add_argument('--tta', action='store_true',
                        help='Enable test-time augmentation: average predictions '
                             'over original + circular roll ±tta_shift_s. '
                             'Recommended for all models when generating PL offline.')
    parser.add_argument('--tta_shift_s', type=float, default=1.25,
                        help='Circular-roll shift in seconds for TTA (default: 1.25)')

    # Inference
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Chunk batch size per forward pass (default: 32). '
                             'With --tta each batch expands to 3× internally.')
    parser.add_argument('--device', default=None,
                        help='Torch device (default: cuda if available, else cpu)')

    args = parser.parse_args()

    if args.checkpoints is not None and len(args.checkpoints) != len(args.model_dirs):
        parser.error(
            f'--checkpoints has {len(args.checkpoints)} entries but '
            f'--model_dirs has {len(args.model_dirs)}. They must match.')

    # ── Setup ─────────────────────────────────────────────────────────────────
    device          = torch.device(
        args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    out_dir         = Path(args.out_dir)
    base_dir        = Path(args.base_dir)
    soundscape_dir  = Path(args.soundscape_dir)
    model_dirs      = [Path(d) for d in args.model_dirs]
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_types = (
        args.checkpoints
        if args.checkpoints is not None
        else ['swa'] * len(model_dirs)
    )

    print('=' * 60)
    print('Soundscape pseudo-label generation')
    print('=' * 60)
    print(f'Device         : {device}')
    print(f'Models         : {len(model_dirs)}')
    print(f'Checkpoints    : {args.checkpoints or "auto (swa → best fallback)"} → {checkpoint_types}')
    print(f'TTA            : {args.tta}' + (f'  (shift ±{args.tta_shift_s}s)' if args.tta else ''))
    print(f'Soundscape dir : {soundscape_dir}')
    print(f'Exclude labelled: {args.exclude_labelled}')
    print(f'Output         : {out_dir}')

    # ── Vocabulary ────────────────────────────────────────────────────────────
    ref_lm_path = model_dirs[0] / 'label_map.npy'
    if not ref_lm_path.exists():
        raise FileNotFoundError(f'label_map.npy not found in {model_dirs[0]}')

    idx2species: Dict[int, str] = np.load(ref_lm_path, allow_pickle=True).item()
    for md in model_dirs[1:]:
        other = np.load(md / 'label_map.npy', allow_pickle=True).item()
        if other != idx2species:
            raise ValueError(
                f'Vocabulary mismatch between {model_dirs[0]} and {md}. '
                'All models must share the same label_map.npy.')

    num_classes = len(idx2species)
    species2idx = {v: k for k, v in idx2species.items()}
    print(f'Vocabulary     : {num_classes} species')

    # ── Labelled chunk set (for manifest filtering) ───────────────────────────
    labelled_set: Optional[set] = None
    if args.exclude_labelled:
        sc_labels_path = base_dir / 'train_soundscapes_labels.csv'
        if not sc_labels_path.exists():
            raise FileNotFoundError(
                f'--exclude_labelled requires {sc_labels_path}')
        sc_labels_df = pd.read_csv(sc_labels_path)
        # Normalise start time from "HH:MM:SS" to integer seconds
        def _to_sec(t: str) -> int:
            h, m, s = str(t).split(':')
            return int(h) * 3600 + int(m) * 60 + int(s)
        sc_labels_df['start_sec_int'] = sc_labels_df['start'].apply(_to_sec)
        labelled_set = set(
            zip(sc_labels_df['filename'], sc_labels_df['start_sec_int']))
        print(f'Labelled chunks (to exclude): {len(labelled_set):,}')

    # ── Validate model audio configs ──────────────────────────────────────────
    MANIFEST_CHUNK_DUR = 5  # competition format: always 5s chunks
    for md in model_dirs:
        mcfg = _load_cfg_from_json(md)
        if mcfg.sr != 32000:
            raise ValueError(
                f'{md.name}: sr={mcfg.sr}, expected 32000. '
                'All models must share sr=32000.')
        if mcfg.chunk_duration != MANIFEST_CHUNK_DUR:
            raise ValueError(
                f'{md.name}: chunk_duration={mcfg.chunk_duration}, '
                f'expected {MANIFEST_CHUNK_DUR}. '
                'Soundscape PL manifest is always 5s chunks.')

    # ── Manifest ─────────────────────────────────────────────────────────────
    manifest_path = out_dir / 'sc_pl.csv'
    chunk_dur     = MANIFEST_CHUNK_DUR
    print(f'Chunk duration : {chunk_dur}s')

    print('\nBuilding chunk manifest ...')
    manifest = _build_manifest(soundscape_dir, chunk_dur, args.exclude_labelled,
                                labelled_set)
    M = len(manifest)
    print(f'Total chunks   : {M:,}  '
          f'({manifest["filename"].nunique():,} files)')

    # Save / verify manifest
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        if len(existing) != M:
            raise RuntimeError(
                f'Existing manifest has {len(existing)} rows but current scan '
                f'found {M}. Delete {manifest_path} and re-run to rebuild.')
        print(f'Manifest already exists and matches ({M:,} rows).')
    else:
        manifest.to_csv(manifest_path, index=False)
        print(f'Saved manifest → {manifest_path.name}')

    # ── Per-model inference ───────────────────────────────────────────────────
    for md, ckpt_type in zip(model_dirs, checkpoint_types):
        name     = md.name
        out_path = out_dir / f'sc_pl_preds_{name}.npy'

        if out_path.exists():
            saved_shape = np.load(out_path, mmap_mode='r').shape
            if saved_shape == (M, num_classes):
                print(f'\n[SKIP] {name}: {out_path.name} already exists '
                      f'with correct shape {saved_shape}')
                continue
            else:
                print(f'\n[REDO] {name}: shape mismatch '
                      f'{saved_shape} ≠ ({M}, {num_classes})')

        print(f'\nRunning model: {name}')
        preds = _predict_all_chunks(
            md, manifest, soundscape_dir, num_classes, device, args.batch_size,
            checkpoint=ckpt_type,
            tta=args.tta,
            tta_shift_s=args.tta_shift_s,
        )
        np.save(out_path, preds)
        print(f'  Saved → {out_path.name}  '
              f'{preds.nbytes / 1e6:.0f} MB')

    # ── Perch predictions ─────────────────────────────────────────────────────
    if args.perch_csv:
        perch_out = out_dir / 'sc_pl_preds_perch.npy'
        if perch_out.exists():
            saved_shape = np.load(perch_out, mmap_mode='r').shape
            if saved_shape == (M, num_classes):
                print(f'\n[SKIP] Perch: {perch_out.name} already exists')
            else:
                print(f'\n[REDO] Perch: shape mismatch {saved_shape}')
                perch_preds = _parse_perch_predictions(
                    Path(args.perch_csv), manifest, species2idx, num_classes)
                np.save(perch_out, perch_preds)
                print(f'  Saved → {perch_out.name}  '
                      f'{perch_preds.nbytes / 1e6:.0f} MB')
        else:
            print('\nParsing Perch predictions ...')
            perch_preds = _parse_perch_predictions(
                Path(args.perch_csv), manifest, species2idx, num_classes)
            np.save(perch_out, perch_preds)
            print(f'  Saved → {perch_out.name}  '
                  f'{perch_preds.nbytes / 1e6:.0f} MB')

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('Output files:')
    for p in sorted(out_dir.iterdir()):
        size_mb = p.stat().st_size / 1e6
        print(f'  {p.name:<40s}  {size_mb:.1f} MB')

    print(f'\nManifest: {M:,} chunks  ({manifest["filename"].nunique():,} files)')
    print('\nNext step: pass to SoundscapePLDataset at training time.')
    print('  Ensemble weights, power transform, and alpha are set in train.py —')
    print('  no need to re-run this script to change them.')


if __name__ == '__main__':
    main()
