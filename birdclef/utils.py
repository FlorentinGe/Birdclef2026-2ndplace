"""
Miscellaneous helpers: seeding, vocab building, soundscape split.
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch


# ── Reproducibility ───────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # benchmark=True selects fastest cuDNN kernel per op at startup.
    # On new GPU architectures (sm_89+, sm_120/Blackwell) the benchmarked winner
    # can be a numerically broken kernel in early driver releases, causing silent
    # corruption (wrong gradients, degrading AUC). Set False for safety.
    # deterministic=True without benchmark=False is also self-contradictory:
    # benchmark overrides determinism for most ops — don't set both True.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = False


# ── Label parsing ─────────────────────────────────────────────────────────────

def parse_secondary_labels(raw) -> List[str]:
    """
    Parse the secondary_labels column from train.csv.
    Handles: "['rufhor2', 'whtdov']", "[]", NaN, and empty strings.
    """
    if not isinstance(raw, str) or raw.strip() in ('', '[]'):
        return []
    raw = raw.strip().lstrip('[').rstrip(']')
    return [s.strip().strip("'").strip('"') for s in raw.split(',') if s.strip()]


# ── Soundscape label normalisation ───────────────────────────────────────────

def _union_labels(series) -> str:
    """Merge multiple primary_label values (';'-separated) into a single string."""
    all_labels: Set[str] = set()
    for s in series:
        all_labels.update(str(s).split(';'))
    return ';'.join(sorted(all_labels))


def normalise_sc_df(sc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse train_soundscapes_labels rows to one row per (filename, start, end),
    merging all primary_label values into a ';'-joined union.
    Adds start_sec (integer seconds).
    """
    sc_df = (
        sc_df.groupby(['filename', 'start', 'end'], sort=False)['primary_label']
        .apply(_union_labels)
        .reset_index()
    )
    sc_df['start_sec'] = sc_df['start'].apply(
        lambda t: sum(int(x) * s for x, s in zip(str(t).split(':'), [3600, 60, 1]))
    )
    return sc_df


# ── Soundscape train/val split ────────────────────────────────────────────────

def stratified_soundscape_split(
    sc_df: pd.DataFrame,
    val_frac: float,
    seed: int,
) -> Tuple[Set[str], Set[str]]:
    """
    Split soundscape files into train/val sets.

    Rules:
    - Files that are the only source of a species are forced into the train set
      (guarantees every species seen in training).
    - The remaining files are randomly split.
    - If any species ends up with no train files after the split, one val file
      covering that species is rescued into the train set (greedy: pick the file
      that covers the most species to minimise information loss).

    Returns (train_fileset, val_fileset).
    """
    sc_files = list(sc_df['filename'].unique())

    file_species: Dict[str, Set[str]] = {}
    for fn in sc_files:
        sp_set: Set[str] = set()
        for lab in sc_df[sc_df['filename'] == fn]['primary_label']:
            sp_set.update(lab.split(';'))
        file_species[fn] = sp_set

    species_files: Dict[str, Set[str]] = defaultdict(set)
    for fn, sp_set in file_species.items():
        for sp in sp_set:
            species_files[sp].add(fn)

    # Force-train files that are the only carrier of a species
    forced_train: Set[str] = set()
    for sp, files in species_files.items():
        if len(files) == 1:
            forced_train.update(files)

    remaining = [f for f in sc_files if f not in forced_train]
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)

    n_val = max(2, int(len(sc_files) * val_frac))
    n_val = min(n_val, len(remaining))

    val_fileset   = set(remaining[:n_val])
    train_fileset = forced_train | set(remaining[n_val:])

    # Rescue pass: move one val file to train for any species with no train coverage
    for sp, files in species_files.items():
        if not files & train_fileset:
            rescue = max(files & val_fileset, key=lambda f: len(file_species[f]))
            val_fileset.discard(rescue)
            train_fileset.add(rescue)

    return train_fileset, val_fileset


# ── Unlabeled soundscape val split ───────────────────────────────────────────

def stratified_unlabeled_sc_split(
    manifest: pd.DataFrame,
    ensemble_preds: np.ndarray,
    val_frac: float = 0.10,
    seed: int = 42,
) -> Tuple[Set[str], Set[str]]:
    """
    Stratified 10% file-level holdout for unlabeled soundscape OOF tracking.

    Strata: site × hour_bucket × label_density_quartile
      site         — parsed from filename: BC2026_Train_NNNN_S{site}_...ogg
      hour_bucket  — night (20–23 and 0–5), morning (6–11), afternoon (12–19)
      density_q    — per-file sum of per-class max probs, cut at global q25/q50/q75

    The split is file-level: all chunks of a file land in the same partition.
    Files in singleton strata are forced into the training set.
    When val_frac * stratum_size rounds to 0, the stratum is skipped (all train).

    Returns (train_fileset, val_fileset).
    """
    records: List[dict] = []
    for fn, grp in manifest.groupby('filename'):
        positions = grp.index.values        # 0-based, aligned with ensemble_preds
        parts = str(fn).replace('.ogg', '').split('_')
        site  = parts[3] if len(parts) > 3 else 'S00'
        try:
            hour = int(parts[5][:2]) if len(parts) > 5 else 0
        except (ValueError, IndexError):
            hour = 0
        if hour >= 20 or hour < 6:
            hour_bucket = 'night'
        elif hour < 12:
            hour_bucket = 'morning'
        else:
            hour_bucket = 'afternoon'
        density = float(ensemble_preds[positions].max(axis=0).sum())
        records.append({'filename': fn, 'site': site,
                        'hour_bucket': hour_bucket, 'density': density})

    file_df = pd.DataFrame(records)
    q25, q50, q75 = np.percentile(file_df['density'], [25, 50, 75])

    def _dq(d: float) -> str:
        if d <= q25: return 'q1'
        if d <= q50: return 'q2'
        if d <= q75: return 'q3'
        return 'q4'

    file_df['density_q'] = file_df['density'].apply(_dq)
    file_df['stratum']   = (file_df['site'] + '_' +
                            file_df['hour_bucket'] + '_' +
                            file_df['density_q'])

    rng = np.random.default_rng(seed)
    val_fileset:   Set[str] = set()
    train_fileset: Set[str] = set()

    for _stratum, grp in file_df.groupby('stratum'):
        files = grp['filename'].tolist()
        rng.shuffle(files)
        if len(files) < 2:
            train_fileset.update(files)
            continue
        n_val = max(1, round(len(files) * val_frac))
        n_val = min(n_val, len(files) - 1)   # keep ≥1 file in train
        val_fileset.update(files[:n_val])
        train_fileset.update(files[n_val:])

    return train_fileset, val_fileset


# ── Vocabulary building ───────────────────────────────────────────────────────

def build_vocabulary(
    train_df: pd.DataFrame,
    sc_df: pd.DataFrame,
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    Build sorted species vocabulary from train_audio primary labels + all species
    appearing in labelled soundscapes.

    Returns (all_species, species2idx, idx2species).
    """
    train_species = set(train_df['primary_label'].dropna().unique())
    sc_species: Set[str] = set()
    for lab in sc_df['primary_label']:
        sc_species.update(str(lab).split(';'))

    all_species = sorted(train_species | sc_species)
    species2idx = {sp: i for i, sp in enumerate(all_species)}
    idx2species = {i: sp for sp, i in species2idx.items()}
    return all_species, species2idx, idx2species


def encode_multilabel(label_str, num_classes: int, species2idx: Dict[str, int]) -> np.ndarray:
    """
    Encode the secondary_labels string from train.csv as a multi-hot float32 vector.
    Handles: "['rufhor2', 'whtdov']", "[]", NaN, and empty strings.
    """
    vec = np.zeros(num_classes, dtype=np.float32)
    if not isinstance(label_str, str) or not label_str.strip():
        return vec
    for sp in label_str.replace("'", '').strip('[]').split(','):
        sp = sp.strip()
        if sp in species2idx:
            vec[species2idx[sp]] = 1.0
    return vec


def make_sc_label_vec(label_str: str, num_classes: int, species2idx: Dict[str, int]) -> np.ndarray:
    """
    Encode a soundscape primary_label string (';'-separated) as a multi-hot vector.
    """
    vec = np.zeros(num_classes, dtype=np.float32)
    for sp in str(label_str).split(';'):
        sp = sp.strip()
        if sp in species2idx:
            vec[species2idx[sp]] = 1.0
    return vec


# ── WAV file path mapping ─────────────────────────────────────────────────────

def resolve_audio_paths(train_df: pd.DataFrame, train_audio_dir: Path) -> pd.Series:
    """
    Resolve focal clip file paths.

    Works for both OGG (default) and WAV (shard-converted) layouts:
      - OGG: train_audio_dir = base_dir/train_audio
             filename in train.csv = "{primary_label}/{stem}.ogg"
             → train_audio_dir / filename

      - WAV shards: train_audio_dir = one of the 4 shard dirs
             filename in train.csv = "{primary_label}/{stem}.ogg"
             → train_audio_dir / primary_label / stem.wav
             (WAV shards preserve the primary_label subdirectory layout)

    For the WAV shard case, call this function once per shard (after building
    a per-primary_label mapping), or use the legacy build_wav_path_map /
    resolve_wav_paths functions below.
    """
    return pd.Series(
        [str(train_audio_dir / fn) for fn in train_df['filename']],
        index=train_df.index,
    )


def build_wav_path_map(train_audio_wav_prefix: str) -> Dict[str, Path]:
    """
    Walk the 4 sharded WAV directories (<prefix>-00 … <prefix>-03) and build a
    mapping from primary_label (directory name) to the shard Path that contains it.

    Returns pl2wav_dir: Dict[primary_label, Path].
    Only needed when using the ttahara WAV dataset (not OGG originals).
    """
    pl2wav_dir: Dict[str, Path] = {}
    for i in range(4):
        shard = Path(f'{train_audio_wav_prefix}{i:02d}')
        if not shard.exists():
            continue
        for species_dir in sorted(shard.iterdir()):
            if species_dir.is_dir():
                pl2wav_dir[species_dir.name] = shard
    return pl2wav_dir


def resolve_wav_paths(train_df: pd.DataFrame, pl2wav_dir: Dict[str, Path]) -> pd.Series:
    """
    Return a Series of resolved .wav file paths for each row in train_df.
    Only needed when using the ttahara WAV dataset (not OGG originals).
    """
    return pd.Series([
        str(pl2wav_dir[pl] / f'{fn.rsplit(".", 1)[0]}.wav')
        for fn, pl in train_df[['filename', 'primary_label']].values
    ], index=train_df.index)


# ── SWA BN finalisation ───────────────────────────────────────────────────────

def safe_copy_bn_buffers(swa_model, trained_model, verbose: bool = True) -> None:
    """
    Copy BatchNorm running stats (running_mean, running_var, num_batches_tracked)
    from trained_model into swa_model.module in-place.

    This avoids update_bn(), which causes NaN running stats with some backbone +
    SWA combinations (EfficientNetV2-S, NFNet-L0 SED head).  The copy is safe
    because the last training epoch's BN stats are already well-calibrated and
    SWA only averages weight parameters, not running stats.

    Validates the result and prints a warning if any NaN/Inf buffers remain.
    """
    trained_buffers = dict(trained_model.named_buffers())
    copied = 0
    for name, buf in swa_model.module.named_buffers():
        if name in trained_buffers:
            buf.copy_(trained_buffers[name])
            copied += 1

    nan_bufs = [
        (n, b) for n, b in swa_model.module.named_buffers()
        if torch.isnan(b).any() or torch.isinf(b).any()
    ]
    if verbose:
        if nan_bufs:
            print(f'WARNING: {len(nan_bufs)} SWA buffers contain NaN/Inf after copy: '
                  f'{[n for n, _ in nan_bufs]}')
        else:
            print(f'SWA BN buffers copied OK ({copied} buffers, no NaN/Inf)')
