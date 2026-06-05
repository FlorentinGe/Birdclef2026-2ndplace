#!/usr/bin/env python3
"""
Extract Perch v2 logits for focal clips using perch-hoplite.

Install
-------
    pip install 'perch-hoplite[tf-cuda]'   # GPU — RTX 4000, CUDA 12.x / CuDNN 9.x
    pip install 'perch-hoplite[tf]'         # CPU-only fallback

Why this resolves the CuDNN conflict
-------------------------------------
The Kaggle Perch binary was built against TF 2.12–2.15 (CuDNN 8.x). perch-hoplite
requires TF ≥ 2.20, which supports CuDNN 9.x natively. The model is still a TF
SavedModel internally — there is no JAX path for Perch inference — but upgrading
TF to 2.20 is what makes it compatible with a modern CUDA 12 environment.

Model weights
-------------
By default the model is fetched via kagglehub on first run and cached locally
(~/.cache/kagglehub/). Set KAGGLE_USERNAME and KAGGLE_KEY environment variables
or place ~/.kaggle/kaggle.json before running without --model_path.

On Kaggle (model already mounted), pass the SavedModel directory directly:
    --model_path /kaggle/input/models/google/bird-vocalization-classifier/ \\
                 tensorflow2/perch_v2_gpu/1

Output
------
Writes <out_dir>/perch_focal_arrays.npz with:
    filenames : (N,) str  — train.csv 'filename' values, e.g. "aldfly/XC1234.ogg"
    logits    : (N, num_classes) float32 — raw logits in our competition vocabulary

Pass to generate_focal_pl.py with:
    --perch_npz <out_dir>/perch_focal_arrays.npz

Usage
-----
    python extract_perch_focal.py \\
        --label_map   runs/eca_nfnet_l0_supervised/label_map.npy \\
        --base_dir    /path/to/birdclef-2026 \\
        --out_dir     runs/focal_pl \\
        [--model_name   perch_v2] \\
        [--model_path   /path/to/perch_v2_gpu/1] \\
        [--train_audio_dir /path/to/train_audio] \\
        [--wav_prefix      /path/to/birdclef2026-train-audio-wav-]
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile

sys.path.insert(0, str(Path(__file__).parent))
from birdclef.utils import build_wav_path_map, resolve_audio_paths, resolve_wav_paths

SR          = 32_000
NO_LOGIT    = -20.0   # logit for species with no Perch match; sigmoid(-20) ≈ 0
MAX_CLIP_S  = 60      # cap audio at 60 s before passing to Perch (12 × 5 s windows).
                      # Prevents huge intermediate tensors from very long recordings
                      # and fixes TF retracing on every unique clip length.


# ── Model-directory fix ───────────────────────────────────────────────────────

def _fix_duplicate_class_csvs(model_path: str) -> None:
    """
    Deduplicate entries in any class-list CSV files under model_path.

    perch_v2 ships with duplicate rows in perch_v2_ebird_classes.csv.
    Newer perch-hoplite raises ValueError during model load when it encounters
    these duplicates.  This function fixes the CSV in place (first occurrence
    wins) so TaxonomyModelTF.from_config succeeds.  It is a no-op if the files
    are already clean or if model_path is not a local directory.
    """
    import glob as _glob
    for csv_path in sorted(_glob.glob(
            str(Path(model_path) / '**' / '*.csv'), recursive=True)):
        try:
            with open(csv_path, 'r') as f:
                lines = f.readlines()
            seen: set = set()
            deduped = []
            n_dups = 0
            for line in lines:
                key = line.strip()
                if key and key in seen:
                    n_dups += 1
                else:
                    if key:
                        seen.add(key)
                    deduped.append(line)
            if n_dups:
                with open(csv_path, 'w') as f:
                    f.writelines(deduped)
                print(f'  Fixed {n_dups} duplicate entries in '
                      f'{Path(csv_path).name}')
        except Exception:
            pass


# ── Audio loading ─────────────────────────────────────────────────────────────

def _load_focal_clip(path: str) -> Optional[np.ndarray]:
    """
    Load a focal clip as a mono float32 waveform at SR.
    Returns None on any read error.

    Resampling is done with simple linear interpolation — good enough for
    Perch inference on competition audio which is uniformly 32 kHz.
    """
    try:
        wave, sr = soundfile.read(path, dtype='float32', always_2d=False)
    except Exception as e:
        print(f'  WARNING: could not load {path}: {e}')
        return None
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr != SR:
        old_len = len(wave)
        new_len = int(old_len * SR / sr)
        wave = np.interp(
            np.linspace(0, old_len - 1, new_len),
            np.arange(old_len),
            wave,
        ).astype(np.float32)
    # Cap to MAX_CLIP_S using a random window so model.embed() sees a bounded
    # tensor (prevents giant intermediate buffers from long recordings) and we
    # don't systematically bias toward the recording's opening.
    # Short clips (the 89% under 60 s) pass through unchanged.
    max_samples = MAX_CLIP_S * SR
    if len(wave) > max_samples:
        start = random.randint(0, len(wave) - max_samples)
        wave  = wave[start : start + max_samples]
    return wave


# ── Vocabulary mapping ────────────────────────────────────────────────────────

def _get_perch_classes(model, model_path: Optional[str] = None) -> tuple:
    """
    Return (logit_key, perch_class_list) for this Perch model.

    perch_v2 ships with a CSV that has duplicate entries.  Newer perch-hoplite
    versions raise ValueError: duplicate entries in class list both when
    accessing model.class_list[key] AND when accessing cl.classes on the
    returned object.  All such accesses are wrapped so we fall through to the
    CSV-direct fallback transparently.

    Fallback order:
      1. model.class_list API (any key that returns a non-empty, valid list).
      2. Load the CSV from the model's own assets/ directory (deduped).
      3. Load the CSV from the kagglehub cache directory (deduped).

    Returns
    -------
    logit_key    : str — key to use when indexing outputs.logits
    class_labels : list[str] — ordered Perch class labels (normalised lowercase)
    """
    def _safe_classes(cl):
        """Return cl.classes as a list, or None if inaccessible/empty."""
        try:
            if cl is None or not hasattr(cl, 'classes'):
                return None
            classes = cl.classes
            return classes if len(classes) > 0 else None
        except (AttributeError, ValueError):
            return None

    def _cl_get(key: str):
        """Safe get from class_list regardless of whether it has .get()."""
        try:
            return model.class_list[key]
        except (KeyError, TypeError, ValueError):
            return None

    # Try known key names in order of likelihood
    for key in ('label', 'ebird2021', 'ebird', 'species'):
        classes = _safe_classes(_cl_get(key))
        if classes is not None:
            return key, [lbl.strip().lower() for lbl in classes]

    # All keys empty — try any non-empty key
    try:
        items = model.class_list.items()
    except (AttributeError, ValueError):
        items = []
    for key, cl in items:
        classes = _safe_classes(cl)
        if classes is not None:
            print(f'  Perch class_list: using key {key!r}')
            return key, [lbl.strip().lower() for lbl in classes]

    # Last resort: load the CSV directly, searching:
    #   1. the model's own assets/ directory (works with --model_path)
    #   2. the kagglehub cache (works with auto-download)
    import glob as _glob
    patterns = []
    if model_path:
        patterns += [
            str(Path(model_path) / '**' / '*ebird*classes*.csv'),
            str(Path(model_path) / '**' / '*classes*.csv'),
        ]
    patterns += [
        str(Path.home() / '.cache' / 'kagglehub' / '**' / '*ebird*classes*.csv'),
        str(Path.home() / '.cache' / 'kagglehub' / '**' / '*classes*.csv'),
    ]
    for pat in patterns:
        for csv_path in sorted(_glob.glob(pat, recursive=True)):
            try:
                csv_df  = pd.read_csv(csv_path, header=None)
                labels  = csv_df.iloc[:, 0].str.strip().str.lower()
                labels  = labels.drop_duplicates().tolist()
                if len(labels) > 100:
                    print(f'  Perch class_list empty; loaded {len(labels)} classes '
                          f'from {csv_path} (deduped)')
                    return 'label', labels
            except Exception:
                continue

    raise RuntimeError(
        'Cannot resolve Perch class list: model.class_list is empty/broken '
        'and no assets CSV was found. '
        'Pass --model_path pointing to the SavedModel directory so assets/ '
        'is reachable, or ensure the model is cached under ~/.cache/kagglehub.')


def build_vocab_mapping(
    model,
    base_dir: Path,
    idx2species: Dict[int, str],
    model_path: Optional[str] = None,
) -> tuple:
    """
    Map Perch's output class indices to our competition vocabulary.

    Matching order per species:
      1. Direct label match (works if Perch uses ebird codes or scientific names
         that happen to match our codes exactly).
      2. Scientific name match via taxonomy.csv (primary fallback).
      3. Genus proxy: max-pool over all Perch classes that share the genus,
         derived from scientific names in the Perch class list.
      4. No match → logit stays at NO_LOGIT at inference time.

    Returns
    -------
    logit_key         : str   — key for outputs.logits (e.g. 'label')
    mapped_pos        : (M,) int32 — our vocabulary positions with a Perch match
    mapped_perch_idx  : (M,) int32 — corresponding Perch class indices
    proxy_map         : dict {our_pos: perch_idx_array} for genus-proxy species
    """
    num_classes = len(idx2species)
    species2idx = {v: k for k, v in idx2species.items()}

    # Perch class list: scientific names (e.g. "turdus merula") or ebird codes.
    # _get_perch_classes handles the perch_v2 duplicate-CSV bug transparently.
    logit_key, perch_classes = _get_perch_classes(model, model_path)
    perch_label_to_idx = {lbl: i for i, lbl in enumerate(perch_classes)}

    # taxonomy.csv: ebird code → scientific name (lowercase)
    taxonomy = pd.read_csv(base_dir / 'taxonomy.csv')
    sci_col  = next(
        (c for c in taxonomy.columns
         if 'scientific' in c.lower() or c.lower() == 'species'),
        None,
    )
    if sci_col is None:
        raise RuntimeError(
            f'Cannot find scientific-name column in taxonomy.csv. '
            f'Columns: {taxonomy.columns.tolist()}')
    taxon_lookup: Dict[str, str] = (
        taxonomy[['primary_label', sci_col]].dropna()
        .assign(**{sci_col: lambda df: df[sci_col].str.strip().str.lower()})
        .set_index('primary_label')[sci_col]
        .to_dict()
    )

    # Per-species matching
    our_to_perch: Dict[str, int] = {}
    n_direct = n_sciname = 0
    for sp in species2idx:
        if sp.lower() in perch_label_to_idx:               # direct
            our_to_perch[sp] = perch_label_to_idx[sp.lower()]
            n_direct += 1
        else:
            sci = taxon_lookup.get(sp, '').lower()          # scientific name
            if sci and sci in perch_label_to_idx:
                our_to_perch[sp] = perch_label_to_idx[sci]
                n_sciname += 1

    matched          = sorted(our_to_perch, key=lambda s: species2idx[s])
    mapped_pos       = np.array([species2idx[s] for s in matched],  dtype=np.int32)
    mapped_perch_idx = np.array([our_to_perch[s] for s in matched], dtype=np.int32)

    unmapped = [s for s in species2idx if s not in our_to_perch]

    # Genus proxies — genus extracted from Perch class label (works for
    # scientific names; ebird codes don't encode genus, so these proxies rely
    # on the scientific-name entries in the Perch class list).
    def _genus(label: str) -> str:
        parts = label.strip().split()
        return parts[0] if len(parts) > 1 else ''

    genus_to_perch: Dict[str, List[int]] = {}
    for lbl, pidx in perch_label_to_idx.items():
        g = _genus(lbl)
        if g:
            genus_to_perch.setdefault(g, []).append(pidx)

    proxy_map: Dict[int, np.ndarray] = {}
    for sp in unmapped:
        sci = taxon_lookup.get(sp, '').lower()
        g   = _genus(sci)
        if g and g in genus_to_perch:
            proxy_map[species2idx[sp]] = np.array(genus_to_perch[g], dtype=np.int32)

    n_proxy = len(proxy_map)
    n_none  = len(unmapped) - n_proxy
    print(f'Perch mapping  : {n_direct} direct  '
          f'+ {n_sciname} sci-name  '
          f'+ {n_proxy} genus proxy  '
          f'+ {n_none} unmapped (logit={NO_LOGIT})')

    return logit_key, mapped_pos, mapped_perch_idx, proxy_map


# ── Per-clip inference ────────────────────────────────────────────────────────

def infer_focal_clip(
    model,
    wave: np.ndarray,
    num_classes: int,
    mapped_pos: np.ndarray,
    mapped_perch_idx: np.ndarray,
    proxy_map: Dict[int, np.ndarray],
    logit_key: str = 'label',
) -> np.ndarray:
    """
    Run Perch on a full audio clip and return a (num_classes,) mean logit vector.

    perch-hoplite's embed() handles 5-second windowing and per-window peak
    normalisation (to 0.25) internally — no manual framing needed.

    outputs.logits[logit_key] shape: [num_frames, num_perch_classes].
    We average across frames to get a single clip-level prediction.
    """
    outputs = model.embed(wave)
    # Use the key discovered at build_vocab_mapping time; fall back to first
    # available key if needed (guards against model version differences).
    if logit_key in outputs.logits:
        raw_logits = np.asarray(outputs.logits[logit_key], dtype=np.float32)
    else:
        fallback_key = next(iter(outputs.logits))
        raw_logits   = np.asarray(outputs.logits[fallback_key], dtype=np.float32)
    mean_logits = raw_logits.mean(axis=0) if raw_logits.ndim > 1 else raw_logits

    clip_logits                 = np.full(num_classes, NO_LOGIT, dtype=np.float32)
    clip_logits[mapped_pos]     = mean_logits[mapped_perch_idx]
    for pos, pidx_arr in proxy_map.items():
        clip_logits[pos] = mean_logits[pidx_arr].max()

    return clip_logits


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    parser.add_argument('--label_map', required=True,
                        help='Path to label_map.npy from any trained run directory.')
    parser.add_argument('--base_dir', required=True,
                        help='Competition root (contains train.csv, taxonomy.csv)')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory; writes perch_focal_arrays.npz')

    # Model source (mutually exclusive in practice; --model_path takes priority)
    parser.add_argument('--model_name',
                        default='google/bird-vocalization-classifier/tensorflow2/perch_v2_gpu/1',
                        help='Kaggle model handle for auto-download via kagglehub. '
                             'Ignored when --model_path is set.')
    parser.add_argument('--model_path', default=None,
                        help='Local path to a Perch SavedModel directory. '
                             'Use when the model is already mounted (e.g. on Kaggle) '
                             'or cached. Bypasses kagglehub download.')

    # Audio source
    parser.add_argument('--train_audio_dir', default=None,
                        help='OGG train_audio directory (default: base_dir/train_audio)')
    parser.add_argument('--wav_prefix', default=None,
                        help='ttahara WAV shard prefix '
                             '(e.g. /path/birdclef2026-train-audio-wav-)')

    # GPU control
    parser.add_argument('--force_cpu', action='store_true',
                        help='Hide GPUs from TensorFlow (CUDA_VISIBLE_DEVICES=""). '
                             'Use when the system CuDNN version is older than the one '
                             'TF was compiled against (e.g. runtime 9.1 vs compiled 9.3). '
                             'Inference is still fast enough on CPU for focal clips.')

    args    = parser.parse_args()

    # Must be set before TF initialises — cannot be changed after first TF import.
    if args.force_cpu:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print('GPU disabled for TensorFlow (--force_cpu).')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Enable TF memory growth so TF doesn't claim all VRAM upfront and can
    # release blocks back when tensors go out of scope.  Must be called before
    # the first TF GPU operation (i.e. before model load).
    try:
        import tensorflow as _tf
        for _gpu in _tf.config.list_physical_devices('GPU'):
            _tf.config.experimental.set_memory_growth(_gpu, True)
    except Exception:
        pass   # CPU-only or TF not yet importable — will be caught at model load

    # ── Load model ────────────────────────────────────────────────────────────
    from ml_collections import config_dict
    from perch_hoplite.zoo.taxonomy_model_tf import TaxonomyModelTF

    if args.model_path:
        model_path = str(args.model_path)
        print(f'Loading Perch from local path: {model_path}')
    else:
        try:
            import kagglehub
        except ImportError:
            raise ImportError(
                'kagglehub is required for auto-download. '
                'Install with: pip install kagglehub\n'
                'Or supply --model_path to skip the download.')
        print(f'Downloading Perch via kagglehub ({args.model_name}) ...')
        print('  Set KAGGLE_USERNAME + KAGGLE_KEY if not already configured.')
        model_path = str(kagglehub.model_download(args.model_name))
        print(f'  Cached at: {model_path}')

    _fix_duplicate_class_csvs(model_path)
    cfg = config_dict.ConfigDict({
        'model_path':    model_path,
        'sample_rate':   SR,
        'window_size_s': 5.0,
        'hop_size_s':    5.0,
        'target_peak':   0.25,
        'tfhub_version': 0,
        'tfhub_path':    '',
    })
    model = TaxonomyModelTF.from_config(cfg)

    # ── Vocabulary ────────────────────────────────────────────────────────────
    label_map_path = Path(args.label_map)
    if not label_map_path.exists():
        raise FileNotFoundError(f'label_map.npy not found: {label_map_path}')
    idx2species: Dict[int, str] = np.load(label_map_path, allow_pickle=True).item()
    num_classes = len(idx2species)
    print(f'Vocabulary     : {num_classes} species  (from {label_map_path})')

    logit_key, mapped_pos, mapped_perch_idx, proxy_map = build_vocab_mapping(
        model, Path(args.base_dir), idx2species, model_path=model_path)

    # ── Audio paths ───────────────────────────────────────────────────────────
    train_df = pd.read_csv(Path(args.base_dir) / 'train.csv')

    if args.wav_prefix:
        pl2wav_dir = build_wav_path_map(args.wav_prefix)
        train_df['file_path'] = resolve_wav_paths(train_df, pl2wav_dir)
        print(f'Audio          : WAV shards  ({args.wav_prefix}*)')
    else:
        audio_dir = Path(args.train_audio_dir) if args.train_audio_dir \
                    else Path(args.base_dir) / 'train_audio'
        train_df['file_path'] = resolve_audio_paths(train_df, audio_dir)
        print(f'Audio          : {audio_dir}')

    N = len(train_df)
    print(f'Focal clips    : {N:,}')
    print()

    # ── Inference ─────────────────────────────────────────────────────────────
    out_filenames = train_df['filename'].values
    out_logits    = np.full((N, num_classes), NO_LOGIT, dtype=np.float32)
    n_errors      = 0

    for i, row in enumerate(train_df.itertuples(index=False)):
        wave = _load_focal_clip(row.file_path)
        if wave is None:
            n_errors += 1
            continue

        try:
            out_logits[i] = infer_focal_clip(
                model, wave, num_classes,
                mapped_pos, mapped_perch_idx, proxy_map,
                logit_key=logit_key,
            )
        except Exception as e:
            print(f'  WARNING: inference failed for {row.file_path}: {e}')
            n_errors += 1

        if (i + 1) % 100 == 0:
            gc.collect()
        if (i + 1) % 500 == 0:
            print(f'  {i + 1:,} / {N:,} clips processed')

    if n_errors:
        print(f'WARNING: {n_errors} clips skipped (load or inference error)')

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = out_dir / 'perch_focal_arrays.npz'
    np.savez_compressed(out_path, filenames=out_filenames, logits=out_logits)

    print(f'\nSaved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)')
    print(f'  filenames : {out_filenames.shape}')
    print(f'  logits    : {out_logits.shape}  dtype={out_logits.dtype}')
    print(f'\nNext step:')
    print(f'  python generate_focal_pl.py --perch_npz {out_path} ...')


if __name__ == '__main__':
    main()
