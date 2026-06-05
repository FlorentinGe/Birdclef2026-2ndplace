#!/usr/bin/env python3
"""
Generate soundscape pseudo-labels using overlapping 20s-window inference.

Designed for BirdCLEFModel checkpoints trained with TRAIN_DURATION=20 (e.g. exp85).
Falls back to standard non-overlapping 5s inference when cfg.duration == cfg.chunk_duration.

20s inference path
------------------
For each soundscape file:
  1. Pad the waveform symmetrically by (win_len - step_len) // 2 samples so each
     5s target chunk is centred in its 20s window.
  2. Extract n_chunks overlapping 20s windows with 5s stride.
  3. Forward all windows in batches → frame_logits (N, T_train, C).
  4. Sigmoid + 1st-place overlap-average-max reconstruction → (N, C).

TTA (--tta): each 20s window is independently circularly rolled by ±tta_shift_s
before the forward pass; three (N, C) arrays are blended 1:2:2.

Output format matches generate_sc_pl.py:
  sc_pl.csv                    — chunk manifest (filename, start_sec, end_sec)
  sc_pl_preds_{model_name}.npy — (M, num_classes) float32 sigmoid probabilities
  sc_pl_preds_perch.npy        — (M, num_classes) Perch probabilities (optional)

Usage
-----
  python generate_sc_pl_20s.py \\
      --model_dirs runs/exp85 \\
      --base_dir   /path/to/birdclef-2026 \\
      --soundscape_dir /path/to/train_soundscapes \\
      --out_dir    runs/sc_pl_exp85 \\
      [--perch_csv /path/to/train_soundscape_pseudolabels.csv] \\
      [--exclude_labelled] \\
      [--checkpoints best_sc] \\
      [--batch_size 8]   # 20s windows are ~4x larger than 5s — use smaller batches \\
      [--tta] \\
      [--device cuda]

Resumable: if sc_pl_preds_{model_name}.npy already exists with the correct shape
the model is skipped.  Delete the file to force a re-run.
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
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from birdclef.model import BirdCLEFModel
from birdclef.transforms import MelTransform
from generate_focal_pl import _load_cfg_from_json


# ── PaSST key remapping ───────────────────────────────────────────────────────

def _remap_passt_keys(state: dict) -> dict:
    OLD, NEW = '_net.norm.', '_net.norm.norm.'
    return {
        (k.replace(OLD, NEW, 1) if OLD in k and NEW not in k else k): v
        for k, v in state.items()
    }


# ── Overlap-average-max reconstruction ───────────────────────────────────────

def _overlap_average_max(
    frame_probs: np.ndarray,   # (n_windows, T_train, C)
    T_train: int,
    T_chunk: int,
) -> np.ndarray:
    """
    1st-place overlap-average-max reconstruction.

    Places each window's frame predictions in a shared per-frame timeline,
    averages overlapping regions, trims symmetric edge padding, then max-pools
    within each 5s chunk.

    Returns (n_windows, C) float32 probabilities.
    """
    n      = frame_probs.shape[0]
    C      = frame_probs.shape[2]
    step   = T_chunk
    ss_len = T_train + step * (n - 1)

    timeline = np.zeros((ss_len, C), dtype=np.float32)
    count    = np.zeros((ss_len, 1), dtype=np.float32)
    for i in range(n):
        s = i * step
        timeline[s : s + T_train] += frame_probs[i]
        count   [s : s + T_train] += 1.0
    timeline /= count

    pad   = (T_train - step) // 2
    extra = (T_train - step) % 2
    timeline = timeline[pad : ss_len - pad - extra]    # (n * step, C)
    return timeline.reshape(n, step, C).max(axis=1)    # (n, C)


# ── Manifest building ─────────────────────────────────────────────────────────

def _build_manifest(
    soundscape_dir: Path,
    chunk_duration: int,
    exclude_labelled: bool,
    labelled_set: Optional[set],
) -> pd.DataFrame:
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
            rows.append({'filename': fpath.name, 'start_sec': start, 'end_sec': end})

    if not rows:
        raise RuntimeError(
            'Manifest is empty. Check --soundscape_dir and --exclude_labelled.')
    return pd.DataFrame(rows, dtype=object).astype({'start_sec': int, 'end_sec': int})


# ── Perch CSV → array ─────────────────────────────────────────────────────────

def _parse_perch_predictions(
    perch_csv: Path,
    manifest: pd.DataFrame,
    species2idx: Dict[str, int],
    num_classes: int,
) -> np.ndarray:
    print(f'  Loading Perch CSV: {perch_csv.name}')
    perch_df           = pd.read_csv(perch_csv)
    perch_species_cols = [c for c in perch_df.columns if c != 'row_id']
    col_to_idx         = {c: species2idx[c] for c in perch_species_cols if c in species2idx}
    print(f'  Perch columns mapped: {len(col_to_idx)} / {len(perch_species_cols)}')

    perch_df  = perch_df.set_index('row_id')
    M         = len(manifest)
    preds     = np.zeros((M, num_classes), dtype=np.float32)
    n_matched = 0

    for row_idx, row in manifest.iterrows():
        row_id = f'{Path(row["filename"]).stem}_{int(row["end_sec"])}'
        if row_id not in perch_df.index:
            continue
        perch_row = perch_df.loc[row_id]
        for col, our_idx in col_to_idx.items():
            val = perch_row.get(col, 0.0)
            if pd.notna(val):
                preds[row_idx, our_idx] = float(val)
        n_matched += 1

    pct = 100.0 * n_matched / M if M > 0 else 0.0
    print(f'  Chunks matched: {n_matched:,} / {M:,}  ({pct:.1f}%)')
    if pct < 50.0:
        print('  WARNING: fewer than 50% of chunks matched — check --perch_csv coverage.')

    max_per_chunk = preds.max(axis=1)
    has_signal    = max_per_chunk > 0
    if has_signal.any():
        print(f'  Chunks with signal: {has_signal.sum():,}  '
              f'mean max-prob={max_per_chunk[has_signal].mean():.4f}')
    return preds


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model(
    model_dir: Path,
    num_classes: int,
    device: torch.device,
    checkpoint: str = 'swa',
):
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
            print('    WARNING: swa_model.pth not found — using best_model.pth')

    use_20s = cfg.duration > cfg.chunk_duration
    print(f'    backbone       : {cfg.backbone}')
    print(f'    checkpoint     : {ckpt.name}')
    print(f'    hop_length     : {cfg.hop_length}')
    print(f'    training window: {cfg.duration}s')
    print(f'    inference mode : '
          f'{"20s overlapping windows" if use_20s else "5s non-overlapping chunks"}')

    model = BirdCLEFModel(cfg, num_classes, pretrained=False)
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    state = _remap_passt_keys(state)
    model.load_state_dict(state)
    model.eval().to(device)

    mel_transform = MelTransform(cfg).to(device)
    return model, mel_transform, cfg


# ── Per-model inference ───────────────────────────────────────────────────────

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

    20s mode (cfg.duration > cfg.chunk_duration):
        batch_size is in 20s windows (~4× larger than 5s chunks).
        Collect all windows per file first, then reconstruct — _overlap_average_max
        needs the full (n_windows, T_train, C) array, not per-batch slices.
        Recommended: batch_size 8–16 on GPU, 4–8 on CPU.

    5s mode (cfg.duration == cfg.chunk_duration):
        Non-overlapping 5s chunks, uses att_clipwise (identical to generate_sc_pl.py).

    Returns (M, num_classes) float32 sigmoid probabilities.
    """
    model, mel_transform, cfg = _load_model(model_dir, num_classes, device, checkpoint)

    chunk_len = cfg.sr * cfg.chunk_duration
    use_20s   = cfg.duration > cfg.chunk_duration
    win_len   = cfg.sr * cfg.duration                       # 640 000 for 20s @ 32 kHz
    sym_pad   = (win_len - chunk_len) // 2 if use_20s else 0  # 240 000 for 20s/5s
    tta_shift = int(tta_shift_s * cfg.sr) if tta else 0

    M     = len(manifest)
    preds = np.zeros((M, num_classes), dtype=np.float32)

    def _run_20s_windows(wins: List[torch.Tensor]) -> np.ndarray:
        """Batch-forward 20s windows → (n_chunks, C) via overlap reconstruction."""
        parts: List[np.ndarray] = []
        for b in range(0, len(wins), batch_size):
            b_wins = torch.stack(wins[b : b + batch_size]).to(device)
            specs  = mel_transform(b_wins, augment=False)
            _, _, frame_logits = model(specs)               # (B, T_train, C)
            parts.append(torch.sigmoid(frame_logits).float().cpu().numpy())
        fp      = np.concatenate(parts, axis=0)             # (N, T_train, C)
        T_train = fp.shape[1]
        T_chunk = max(1, T_train * cfg.chunk_duration // cfg.duration)
        return _overlap_average_max(fp, T_train, T_chunk)   # (N, C)

    for filename in tqdm(manifest['filename'].unique(), desc='  files', leave=False):
        fpath = soundscape_dir / filename
        grp   = manifest[manifest['filename'] == filename]
        g_idx = list(grp.index)
        n     = len(grp)

        try:
            wave, orig_sr = soundfile.read(str(fpath), dtype='float32', always_2d=False)
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

        if use_20s:
            flat   = torch.from_numpy(wave)
            needed = sym_pad + (n - 1) * chunk_len + win_len
            flat   = F.pad(flat, (sym_pad, max(0, needed - sym_pad - flat.shape[0])))
            wins   = [flat[i * chunk_len : i * chunk_len + win_len] for i in range(n)]

            if tta:
                p_o = _run_20s_windows(wins)
                p_f = _run_20s_windows([torch.roll(w,  tta_shift) for w in wins])
                p_b = _run_20s_windows([torch.roll(w, -tta_shift) for w in wins])
                probs = (p_o + 2.0 * p_f + 2.0 * p_b) / 5.0
            else:
                probs = _run_20s_windows(wins)

            for k, gi in enumerate(g_idx):
                preds[gi] = probs[k]

        else:
            # Standard 5s non-overlapping path (identical to generate_sc_pl.py)
            chunk_tensors: List[torch.Tensor] = []
            for _, row in grp.iterrows():
                s     = int(row['start_sec'] * cfg.sr)
                chunk = wave[s : s + chunk_len]
                if len(chunk) < chunk_len:
                    chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
                chunk_tensors.append(torch.from_numpy(chunk.copy()))

            for b in range(0, len(chunk_tensors), batch_size):
                b_chunks  = chunk_tensors[b : b + batch_size]
                b_indices = g_idx[b : b + batch_size]

                if tta:
                    views: List[torch.Tensor] = []
                    for ct in b_chunks:
                        arr = ct.numpy()
                        views += [ct,
                                  torch.from_numpy(np.roll(arr,  tta_shift).copy()),
                                  torch.from_numpy(np.roll(arr, -tta_shift).copy())]
                    b_waves = torch.stack(views).to(device)
                    specs   = mel_transform(b_waves, augment=False)
                    _, att_clip, _ = model(specs)
                    probs = (torch.sigmoid(att_clip).float().cpu().numpy()
                             .reshape(len(b_chunks), 3, num_classes).mean(axis=1))
                else:
                    b_waves = torch.stack(b_chunks).to(device)
                    specs   = mel_transform(b_waves, augment=False)
                    _, att_clip, _ = model(specs)
                    probs = torch.sigmoid(att_clip).float().cpu().numpy()

                for k, gi in enumerate(b_indices):
                    preds[gi] = probs[k]

    max_per_chunk = preds.max(axis=1)
    print(f'    mean max-prob : {max_per_chunk.mean():.4f}  '
          f'p10={np.percentile(max_per_chunk, 10):.4f}  '
          f'p90={np.percentile(max_per_chunk, 90):.4f}')

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
                        help='Run directories: each must contain a checkpoint, '
                             'run_config.json, and label_map.npy')
    parser.add_argument('--base_dir', required=True,
                        help='Competition root (contains train_soundscapes_labels.csv)')
    parser.add_argument('--soundscape_dir', required=True,
                        help='Directory of OGG soundscape files')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for manifest and prediction arrays')

    # Perch
    parser.add_argument('--perch_csv', default=None,
                        help='Path to train_soundscape_pseudolabels.csv')

    # Manifest options
    parser.add_argument('--exclude_labelled', action='store_true',
                        help='Omit chunks that have hard labels in '
                             'train_soundscapes_labels.csv')

    # Checkpoint selection
    parser.add_argument('--checkpoints', nargs='*',
                        choices=['swa', 'best', 'best_sc'], default=None,
                        metavar='{swa,best,best_sc}',
                        help='Checkpoint per model dir. Must match --model_dirs count. '
                             'Default: prefer swa_model.pth, fall back to best_model.pth.')

    # TTA
    parser.add_argument('--tta', action='store_true',
                        help='Average predictions over original + ±tta_shift_s rolls (1:2:2).')
    parser.add_argument('--tta_shift_s', type=float, default=1.25)

    # Inference
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Windows per forward pass. 20s windows are ~4× larger '
                             'than 5s chunks; default 8 is conservative for GPU.')
    parser.add_argument('--device', default=None,
                        help='Torch device (default: cuda if available, else cpu)')

    args = parser.parse_args()

    if args.checkpoints is not None and len(args.checkpoints) != len(args.model_dirs):
        parser.error(f'--checkpoints has {len(args.checkpoints)} entries but '
                     f'--model_dirs has {len(args.model_dirs)}.')

    device         = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    out_dir        = Path(args.out_dir)
    base_dir       = Path(args.base_dir)
    soundscape_dir = Path(args.soundscape_dir)
    model_dirs     = [Path(d) for d in args.model_dirs]
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_types = (args.checkpoints if args.checkpoints is not None
                        else ['swa'] * len(model_dirs))

    print('=' * 60)
    print('Soundscape PL generation  (20s overlapping-window inference)')
    print('=' * 60)
    print(f'Device         : {device}')
    print(f'Models         : {len(model_dirs)}')
    print(f'Checkpoints    : {checkpoint_types}')
    print(f'TTA            : {args.tta}' + (f'  (±{args.tta_shift_s}s)' if args.tta else ''))
    print(f'Batch size     : {args.batch_size}  (20s windows)')
    print(f'Soundscape dir : {soundscape_dir}')
    print(f'Output         : {out_dir}')

    # Vocabulary — all models must share label_map.npy
    ref_lm = model_dirs[0] / 'label_map.npy'
    if not ref_lm.exists():
        raise FileNotFoundError(f'label_map.npy not found in {model_dirs[0]}')
    idx2species: Dict[int, str] = np.load(ref_lm, allow_pickle=True).item()
    for md in model_dirs[1:]:
        other = np.load(md / 'label_map.npy', allow_pickle=True).item()
        if other != idx2species:
            raise ValueError(f'Vocabulary mismatch: {model_dirs[0].name} vs {md.name}')
    num_classes = len(idx2species)
    species2idx = {v: k for k, v in idx2species.items()}
    print(f'Vocabulary     : {num_classes} species')

    # Validate audio configs
    MANIFEST_CHUNK_DUR = 5
    for md in model_dirs:
        mcfg = _load_cfg_from_json(md)
        if mcfg.sr != 32000:
            raise ValueError(f'{md.name}: sr={mcfg.sr}, expected 32000')
        if mcfg.chunk_duration != MANIFEST_CHUNK_DUR:
            raise ValueError(f'{md.name}: chunk_duration={mcfg.chunk_duration}, '
                             f'expected {MANIFEST_CHUNK_DUR}')

    # Labelled chunk set
    labelled_set: Optional[set] = None
    if args.exclude_labelled:
        sc_labels_path = base_dir / 'train_soundscapes_labels.csv'
        if not sc_labels_path.exists():
            raise FileNotFoundError(f'--exclude_labelled requires {sc_labels_path}')
        sc_labels_df = pd.read_csv(sc_labels_path)
        def _to_sec(t: str) -> int:
            h, m, s = str(t).split(':')
            return int(h) * 3600 + int(m) * 60 + int(s)
        sc_labels_df['start_sec_int'] = sc_labels_df['start'].apply(_to_sec)
        labelled_set = set(zip(sc_labels_df['filename'], sc_labels_df['start_sec_int']))
        print(f'Labelled chunks (excluded): {len(labelled_set):,}')

    # Manifest
    manifest_path = out_dir / 'sc_pl.csv'
    print('\nBuilding chunk manifest ...')
    manifest = _build_manifest(soundscape_dir, MANIFEST_CHUNK_DUR,
                               args.exclude_labelled, labelled_set)
    M = len(manifest)
    print(f'Total chunks   : {M:,}  ({manifest["filename"].nunique():,} files)')

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

    # Per-model inference
    for md, ckpt_type in zip(model_dirs, checkpoint_types):
        name     = md.name
        out_path = out_dir / f'sc_pl_preds_{name}.npy'

        if out_path.exists():
            saved_shape = np.load(out_path, mmap_mode='r').shape
            if saved_shape == (M, num_classes):
                print(f'\n[SKIP] {name}: {out_path.name} already exists '
                      f'with correct shape {saved_shape}')
                continue
            print(f'\n[REDO] {name}: shape mismatch {saved_shape} ≠ ({M}, {num_classes})')

        print(f'\nRunning model: {name}')
        preds = _predict_all_chunks(
            md, manifest, soundscape_dir, num_classes, device, args.batch_size,
            checkpoint=ckpt_type, tta=args.tta, tta_shift_s=args.tta_shift_s,
        )
        np.save(out_path, preds)
        print(f'  Saved → {out_path.name}  {preds.nbytes / 1e6:.0f} MB')

    # Perch predictions
    if args.perch_csv:
        perch_out = out_dir / 'sc_pl_preds_perch.npy'
        if perch_out.exists():
            saved_shape = np.load(perch_out, mmap_mode='r').shape
            if saved_shape == (M, num_classes):
                print(f'\n[SKIP] Perch: already exists with correct shape')
            else:
                print(f'\n[REDO] Perch: shape mismatch {saved_shape}')
                perch_preds = _parse_perch_predictions(
                    Path(args.perch_csv), manifest, species2idx, num_classes)
                np.save(perch_out, perch_preds)
                print(f'  Saved → {perch_out.name}  {perch_preds.nbytes / 1e6:.0f} MB')
        else:
            print('\nParsing Perch predictions ...')
            perch_preds = _parse_perch_predictions(
                Path(args.perch_csv), manifest, species2idx, num_classes)
            np.save(perch_out, perch_preds)
            print(f'  Saved → {perch_out.name}  {perch_preds.nbytes / 1e6:.0f} MB')

    print('\n' + '=' * 60)
    print('Output files:')
    for p in sorted(out_dir.iterdir()):
        print(f'  {p.name:<40s}  {p.stat().st_size / 1e6:.1f} MB')
    print(f'\nManifest: {M:,} chunks  ({manifest["filename"].nunique():,} files)')
    print('\nNext step: pass sc_pl_preds_*.npy to SoundscapePLDataset at training time.')


if __name__ == '__main__':
    main()
