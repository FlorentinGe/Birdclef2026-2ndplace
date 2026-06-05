#!/usr/bin/env python3
"""
Extract Perch v2 pseudo-labels for soundscape files using overlapping inference.

Background
----------
Perch internally uses 5-second analysis windows.  When we request hop_size_s=2.5
(via TaxonomyModelTF config), it slides those windows every 2.5 seconds, giving
each species two chances to be centred inside a window for each 5-second chunk.
This halves the "call at boundary" miss rate vs. the default non-overlapping pass.

For each non-overlapping 5s chunk in the manifest we max-pool over every Perch
frame whose window overlaps the chunk:

    chunk k  covers [k·chunk_dur,  (k+1)·chunk_dur]
    frame i  covers [i·hop_size_s,  i·hop_size_s + 5.0]
    overlap: i·hop_size_s < (k+1)·chunk_dur  AND  i·hop_size_s + 5 > k·chunk_dur

With hop_size_s=2.5 each interior chunk receives 2-3 overlapping Perch frames.

Model loading
-------------
Always uses TaxonomyModelTF.from_config so hop_size_s is respected.
Pass --model_path for a locally cached SavedModel; omit to auto-download via
kagglehub (requires KAGGLE_USERNAME + KAGGLE_KEY or ~/.kaggle/kaggle.json).

On Kaggle:
    --model_path /kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_gpu/1

Output
------
Writes to --out_dir:
  sc_pl.csv              — chunk manifest: filename, start_sec, end_sec
                           Row order is shared with generate_sc_pl.py. If
                           sc_pl.csv already exists in --out_dir its row order
                           is preserved so all prediction arrays stay aligned.
  sc_pl_preds_perch.npy  — (M, C) float32 sigmoid probabilities
                           Drop-in replacement for the --perch_csv path in
                           generate_sc_pl.py.

Usage
-----
    python extract_perch_soundscape.py \\
        --label_map      runs/exp37b/label_map.npy \\
        --base_dir       /path/to/birdclef-2026 \\
        --soundscape_dir /path/to/train_soundscapes \\
        --out_dir        runs/sc_pl_round1 \\
        [--model_path    /path/to/perch_v2_gpu/1] \\
        [--hop_size_s    2.5] \\
        [--chunk_dur     5] \\
        [--force_cpu]
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile
from tqdm import tqdm

SR          = 32_000
NO_LOGIT    = -20.0   # sentinel for unmatched species; sigmoid(-20) ≈ 0
WINDOW_S    = 5.0     # Perch always analyses 5-second windows internally
CHUNK_DUR_S = 5       # non-overlapping output chunk duration (seconds)


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


# ── Manifest ──────────────────────────────────────────────────────────────────

def _build_manifest(soundscape_dir: Path, chunk_dur: int) -> pd.DataFrame:
    """
    Enumerate non-overlapping chunk_dur-second chunks for all OGG files.
    Only full chunks are included (trailing partial chunks are dropped).
    """
    rows: List[dict] = []
    ogg_files = sorted(soundscape_dir.glob('*.ogg'))
    if not ogg_files:
        raise FileNotFoundError(f'No .ogg files found in {soundscape_dir}')

    for fpath in tqdm(ogg_files, desc='Building manifest', leave=False):
        try:
            info = soundfile.info(str(fpath))
            dur  = info.frames / info.samplerate
        except Exception as e:
            print(f'  WARNING: cannot read {fpath.name}: {e}')
            continue
        for k in range(int(dur // chunk_dur)):
            rows.append({
                'filename':  fpath.name,
                'start_sec': k * chunk_dur,
                'end_sec':   (k + 1) * chunk_dur,
            })

    if not rows:
        raise RuntimeError('Manifest is empty — check --soundscape_dir.')
    return pd.DataFrame(rows).astype({'start_sec': int, 'end_sec': int})


# ── Audio loading ─────────────────────────────────────────────────────────────

def _load_soundscape(path: Path) -> Optional[np.ndarray]:
    """Load a soundscape as mono float32 at SR, peak-normalised to ±1."""
    try:
        wave, sr = soundfile.read(str(path), dtype='float32', always_2d=False)
    except Exception as e:
        print(f'  WARNING: could not load {path.name}: {e}')
        return None
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr != SR:
        old_n = len(wave)
        new_n = int(old_n * SR / sr)
        wave  = np.interp(
            np.linspace(0, old_n - 1, new_n), np.arange(old_n), wave
        ).astype(np.float32)
    absmax = np.abs(wave).max()
    if absmax > 0:
        wave /= absmax
    return wave


# ── Vocabulary mapping (kept in sync with extract_perch_focal.py) ─────────────

def _get_perch_classes(model, model_path: Optional[str] = None) -> tuple:
    def _safe_classes(cl):
        try:
            if cl is None or not hasattr(cl, 'classes'):
                return None
            classes = cl.classes
            return classes if len(classes) > 0 else None
        except (AttributeError, ValueError):
            return None

    def _cl_get(key):
        try:
            return model.class_list[key]
        except (KeyError, TypeError, ValueError):
            return None

    for key in ('label', 'ebird2021', 'ebird', 'species'):
        classes = _safe_classes(_cl_get(key))
        if classes is not None:
            return key, [lbl.strip().lower() for lbl in classes]

    try:
        items = model.class_list.items()
    except (AttributeError, ValueError):
        items = []
    for key, cl in items:
        classes = _safe_classes(cl)
        if classes is not None:
            print(f'  Perch class_list: using key {key!r}')
            return key, [lbl.strip().lower() for lbl in classes]

    import glob as _glob

    # Known scientific-name columns used by BirdClassifier-style SavedModels.
    _SCI_COLS = ('inat2024_fsd50k', 'scientific_name', 'species_name')

    def _labels_from_csv(csv_path: str) -> Optional[List[str]]:
        """Return ordered list of class labels from a CSV, or None if unusable."""
        try:
            # Try with header (labels.csv has a header row)
            df = pd.read_csv(csv_path)
            for col in _SCI_COLS:
                if col in df.columns:
                    labels = df[col].str.strip().str.lower().dropna().tolist()
                    if len(labels) > 100:
                        print(f'  Loaded {len(labels)} Perch classes from '
                              f'{Path(csv_path).name} (col={col!r})')
                        return labels
            # First column may be scientific names without a recognised header
            labels = df.iloc[:, 0].str.strip().str.lower().dropna().tolist()
            if len(labels) > 100:
                print(f'  Loaded {len(labels)} Perch classes from {Path(csv_path).name}')
                return labels
        except Exception:
            pass
        try:
            # Fallback: headerless CSV (*ebird*classes*.csv style)
            df = pd.read_csv(csv_path, header=None)
            labels = df.iloc[:, 0].str.strip().str.lower().drop_duplicates().tolist()
            if len(labels) > 100:
                print(f'  Loaded {len(labels)} Perch classes from {Path(csv_path).name}')
                return labels
        except Exception:
            pass
        return None

    patterns = []
    if model_path:
        patterns += [
            str(Path(model_path) / 'assets' / 'labels.csv'),         # BirdClassifier
            str(Path(model_path) / '**' / '*ebird*classes*.csv'),
            str(Path(model_path) / '**' / '*classes*.csv'),
        ]
    patterns += [
        str(Path.home() / '.cache' / 'kagglehub' / '**' / '*ebird*classes*.csv'),
        str(Path.home() / '.cache' / 'kagglehub' / '**' / '*classes*.csv'),
    ]
    for pat in patterns:
        for csv_path in sorted(_glob.glob(pat, recursive=True)):
            labels = _labels_from_csv(csv_path)
            if labels is not None:
                return 'label', labels

    raise RuntimeError(
        'Cannot resolve Perch class list. Re-download the model or supply '
        '--model_path so assets/ is reachable.')


def _build_vocab_mapping(
    model,
    base_dir: Path,
    idx2species: Dict[int, str],
    model_path: Optional[str] = None,
) -> tuple:
    """
    Map Perch output indices → competition vocabulary.

    Matching order:
      1. Scientific name (primary — covers BirdClassifier iNat2024 labels and
         standard Perch EBIRD labels whose classes happen to be scientific names)
      2. Direct label code (fallback — covers standard Perch EBIRD ebird-code lists)
      3. Genus proxy (Aves + Amphibia + Insecta only; sonotypes excluded)

    Returns (logit_key, mapped_pos, mapped_perch_idx, proxy_map).
    """
    # Taxa for which genus proxies are acoustically meaningful
    _PROXY_TAXA = {'Aves', 'Amphibia', 'Insecta'}

    species2idx = {v: k for k, v in idx2species.items()}

    logit_key, perch_classes = _get_perch_classes(model, model_path)
    perch_label_to_idx = {lbl: i for i, lbl in enumerate(perch_classes)}

    taxonomy = pd.read_csv(base_dir / 'taxonomy.csv')
    sci_col  = next(
        (c for c in taxonomy.columns
         if 'scientific' in c.lower() or c.lower() == 'species'),
        None,
    )
    if sci_col is None:
        raise RuntimeError(
            f'No scientific-name column in taxonomy.csv. '
            f'Columns: {taxonomy.columns.tolist()}')

    taxon_lookup: Dict[str, str] = (
        taxonomy[['primary_label', sci_col]].dropna()
        .assign(**{sci_col: lambda df: df[sci_col].str.strip().str.lower()})
        .set_index('primary_label')[sci_col].to_dict()
    )
    class_name_map: Dict[str, str] = (
        taxonomy[['primary_label', 'class_name']].dropna()
        .set_index('primary_label')['class_name'].to_dict()
        if 'class_name' in taxonomy.columns else {}
    )

    # ── Load labels.csv for scientific-name matching ─────────────────────────
    # _get_perch_classes returns ebird codes when model.class_list['ebird2021']
    # exists (perch_v2_ebird_classes.csv is found first).  labels.csv carries
    # the parallel iNat2024 scientific names at the same row indices, so we
    # load it independently to build a sci-name → output-index map.
    _SCI_COLS = ('inat2024_fsd50k', 'scientific_name', 'species_name')
    sci_to_perch_idx: Dict[str, int] = {}
    if model_path:
        # Try model_path/assets/labels.csv; also parent dir in case model_path
        # was resolved to a subdirectory (e.g. saved_model.pb lives one level down).
        _lcsv_candidates = [
            Path(model_path) / 'assets' / 'labels.csv',
            Path(model_path).parent / 'assets' / 'labels.csv',
        ]
        _lcsv = next((p for p in _lcsv_candidates if p.exists()), None)
        print(f'  labels.csv search: {[str(p) for p in _lcsv_candidates]}')
        print(f'  labels.csv found : {_lcsv}')
        if _lcsv is not None:
            try:
                _ldf = pd.read_csv(_lcsv)
                _sci_col = next((c for c in _SCI_COLS if c in _ldf.columns), None)
                if _sci_col:
                    sci_to_perch_idx = {
                        str(v).strip().lower(): i
                        for i, v in enumerate(_ldf[_sci_col])
                        if str(v).strip()
                    }
                    print(f'  labels.csv       : {len(sci_to_perch_idx)} scientific names '
                          f'(col={_sci_col!r})')
                else:
                    print(f'  WARNING: labels.csv has no recognised sci-name column. '
                          f'Columns: {_ldf.columns.tolist()}')
            except Exception as _e:
                print(f'  WARNING: could not read labels.csv: {_e}')

    our_to_perch: Dict[str, int] = {}
    n_sciname = n_direct = 0
    for sp in species2idx:
        sci = taxon_lookup.get(sp, '').lower()
        # 1. Scientific name via labels.csv (iNat2024) — covers ~200 species
        if sci and sci in sci_to_perch_idx:
            our_to_perch[sp] = sci_to_perch_idx[sci]
            n_sciname += 1
        # 2. Scientific name via perch class list (if classes happen to be sci-names)
        elif sci and sci in perch_label_to_idx:
            our_to_perch[sp] = perch_label_to_idx[sci]
            n_sciname += 1
        # 3. Direct ebird-code match via perch class list
        elif sp.lower() in perch_label_to_idx:
            our_to_perch[sp] = perch_label_to_idx[sp.lower()]
            n_direct += 1

    matched          = sorted(our_to_perch, key=lambda s: species2idx[s])
    mapped_pos       = np.array([species2idx[s] for s in matched], dtype=np.int32)
    mapped_perch_idx = np.array([our_to_perch[s] for s in matched], dtype=np.int32)
    unmapped         = [s for s in species2idx if s not in our_to_perch]

    def _genus(label: str) -> str:
        parts = label.strip().split()
        return parts[0] if len(parts) > 1 else ''

    # Build genus → [perch indices] from scientific names.
    # Ebird codes (e.g. "rufhor2") have no spaces so yield no genus information;
    # sci_to_perch_idx must be the source when available.
    genus_to_perch: Dict[str, List[int]] = {}
    _genus_src = sci_to_perch_idx if sci_to_perch_idx else perch_label_to_idx
    for lbl, pidx in _genus_src.items():
        g = _genus(lbl)
        if g:
            genus_to_perch.setdefault(g, []).append(pidx)

    proxy_map: Dict[int, np.ndarray] = {}
    for sp in unmapped:
        # Sonotypes are artificial codes — no acoustic genus proxy makes sense
        if 'son' in sp:
            continue
        # Proxy is only meaningful within acoustically coherent taxa
        if class_name_map.get(sp) not in _PROXY_TAXA:
            continue
        g = _genus(taxon_lookup.get(sp, '').lower())
        if g and g in genus_to_perch:
            proxy_map[species2idx[sp]] = np.array(genus_to_perch[g], dtype=np.int32)

    n_proxy = len(proxy_map)
    n_none  = len(unmapped) - n_proxy
    print(f'Perch mapping  : {n_sciname} sci-name  '
          f'+ {n_direct} direct  '
          f'+ {n_proxy} genus proxy  '
          f'+ {n_none} unmapped (logit={NO_LOGIT})')
    return logit_key, mapped_pos, mapped_perch_idx, proxy_map


# ── Per-file inference ────────────────────────────────────────────────────────

def _frames_to_chunks(
    frame_logits: np.ndarray,       # (num_frames, num_perch_classes)
    hop_size_s: float,
    chunk_dur: int,
    file_dur_s: float,
    mapped_pos: np.ndarray,
    mapped_perch_idx: np.ndarray,
    proxy_map: Dict[int, np.ndarray],
    num_classes: int,
) -> np.ndarray:
    """
    Aggregate overlapping Perch frames into non-overlapping chunk_dur-s chunks.

    For chunk k = [k*chunk_dur, (k+1)*chunk_dur] we collect every frame whose
    5-second window overlaps the chunk and max-pool their logits, then map to
    the competition vocabulary and apply sigmoid.

    Returns (n_chunks, num_classes) float32 probabilities.
    """
    n_chunks = int(file_dur_s // chunk_dur)
    n_frames = len(frame_logits)

    chunk_probs = np.zeros((n_chunks, num_classes), dtype=np.float32)

    for k in range(n_chunks):
        c_start = k * chunk_dur
        c_end   = c_start + chunk_dur

        # Frame i = [i*hop, i*hop + WINDOW_S]; overlaps chunk if:
        #   i*hop < c_end  AND  i*hop + WINDOW_S > c_start
        overlapping = [
            i for i in range(n_frames)
            if i * hop_size_s < c_end and i * hop_size_s + WINDOW_S > c_start
        ]
        if not overlapping:
            continue  # no Perch frame covers this chunk; probs stay 0

        pooled = frame_logits[overlapping].max(axis=0)   # (num_perch_classes,)

        comp_logits = np.full(num_classes, NO_LOGIT, dtype=np.float32)
        comp_logits[mapped_pos] = pooled[mapped_perch_idx]
        for pos, pidx_arr in proxy_map.items():
            comp_logits[pos] = pooled[pidx_arr].max()

        # Numerically-stable sigmoid
        chunk_probs[k] = 1.0 / (1.0 + np.exp(-comp_logits.clip(-88.0, 88.0)))

    return chunk_probs


def _extract_sig_logits(out, logit_key: str) -> np.ndarray:
    """
    Pull a (n_perch_classes,) float32 array from a signature call output.
    Handles three formats:
      - dict of tensors (TF1-style serving_default)
      - object with .logits dict (perch_hoplite InferenceOutputs)
      - raw tensor
    """
    if isinstance(out, dict):
        for k in (logit_key, f'{logit_key}_logits', 'label_logits', 'logits', 'output_0'):
            if k in out:
                return np.asarray(out[k], dtype=np.float32).squeeze(0)
        # Last resort: first value
        return np.asarray(next(iter(out.values())), dtype=np.float32).squeeze(0)
    if hasattr(out, 'logits'):
        d   = out.logits
        lk  = logit_key if logit_key in d else next(iter(d))
        return np.asarray(d[lk], dtype=np.float32).squeeze(0)
    return np.asarray(out, dtype=np.float32).squeeze(0)


def _build_perch_infer_fn(model, logit_key: str):
    """
    Return a callable  infer_fn(wave, hop_size_s) → (n_frames, n_perch_classes) float32.

    Strategy:
      1. Try model.embed() on a 5-second dummy clip.  This works when the
         SavedModel exposes 'infer_tf' as a TF2 concrete function (typical for
         perch_v2 downloaded via kagglehub).
      2. If embed() fails, fall back to calling the SavedModel's serving
         signature directly with manual windowing.  This handles TF1-format
         SavedModels that only expose inference via model.model.signatures.
    """
    import tensorflow as _tf

    # ── Attempt 1: perch_hoplite embed() ─────────────────────────────────────
    _dummy = np.zeros(SR * 5, dtype=np.float32)
    try:
        _out = model.embed(_dummy)
        _lk  = logit_key if hasattr(_out, 'logits') and logit_key in _out.logits \
               else (next(iter(_out.logits)) if hasattr(_out, 'logits') else None)
        if _lk is not None:
            print('  Inference backend: model.embed()  (TF2 concrete function)')

            def _embed_fn(wave: np.ndarray, hop_size_s: float) -> np.ndarray:
                out = model.embed(wave)
                return np.asarray(out.logits[_lk], dtype=np.float32)

            return _embed_fn
    except Exception:
        pass

    # ── Attempt 2: SavedModel signature ──────────────────────────────────────
    raw  = model.model
    sigs = getattr(raw, 'signatures', {})
    sig_key = next(
        (k for k in ('serving_default', 'infer_tf', 'predict') if k in sigs),
        next(iter(sigs), None),
    )
    if sig_key is None:
        raise RuntimeError(
            f"Perch model has no embed() and no usable signature.\n"
            f"signatures keys: {list(sigs.keys()) if hasattr(sigs, 'keys') else sigs}"
        )

    sig_fn = sigs[sig_key]

    # Determine the input tensor name from the signature spec
    try:
        in_specs = sig_fn.structured_input_signature[1]   # OrderedDict of TensorSpec
        in_key   = next(iter(in_specs))
    except Exception:
        in_key = 'input_0'

    # Print output keys once for diagnostics
    try:
        _test_out = sig_fn(**{in_key: _tf.zeros([1, SR * 5], dtype=_tf.float32)})
        _out_keys = list(_test_out.keys()) if isinstance(_test_out, dict) else str(type(_test_out))
    except Exception as _e:
        _out_keys = f'(probe failed: {_e})'
    print(f'  Inference backend: signatures[{sig_key!r}]  '
          f'input={in_key!r}  outputs={_out_keys}')

    def _sig_fn(wave: np.ndarray, hop_size_s: float) -> np.ndarray:
        window_samples = int(WINDOW_S * SR)
        step_samples   = int(hop_size_s * SR)
        n_frames = max(1, (len(wave) - window_samples) // step_samples + 1)

        needed = (n_frames - 1) * step_samples + window_samples
        if len(wave) < needed:
            wave_pad = np.pad(wave, (0, needed - len(wave))).astype(np.float32)
        else:
            wave_pad = wave.astype(np.float32)

        rows = []
        for i in range(n_frames):
            start   = i * step_samples
            clip    = wave_pad[start : start + window_samples]
            audio_t = _tf.constant(clip[np.newaxis], dtype=_tf.float32)
            out     = sig_fn(**{in_key: audio_t})
            rows.append(_extract_sig_logits(out, logit_key))

        return np.stack(rows, axis=0)   # (n_frames, n_perch_classes)

    return _sig_fn


def _infer_file(
    infer_fn,
    wave: np.ndarray,
    hop_size_s: float,
    chunk_dur: int,
    mapped_pos: np.ndarray,
    mapped_perch_idx: np.ndarray,
    proxy_map: Dict[int, np.ndarray],
    num_classes: int,
) -> np.ndarray:
    """Run Perch on one soundscape; return (n_chunks, num_classes) probabilities."""
    file_dur_s = len(wave) / SR
    raw_logits = infer_fn(wave, hop_size_s)   # (n_frames, n_perch_classes)

    if raw_logits.ndim == 1:
        raw_logits = raw_logits[np.newaxis]

    return _frames_to_chunks(
        raw_logits, hop_size_s, chunk_dur, file_dur_s,
        mapped_pos, mapped_perch_idx, proxy_map, num_classes,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    parser.add_argument('--label_map', required=True,
                        help='label_map.npy from any trained run directory.')
    parser.add_argument('--base_dir', required=True,
                        help='Competition root (contains taxonomy.csv).')
    parser.add_argument('--soundscape_dir', required=True,
                        help='Directory of OGG soundscape files.')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory; writes sc_pl.csv + sc_pl_preds_perch.npy.')

    parser.add_argument('--model_name',
                        default='google/bird-vocalization-classifier/tensorflow2/perch_v2_gpu/1',
                        help='Kaggle model handle for auto-download via kagglehub. '
                             'Ignored when --model_path is set.')
    parser.add_argument('--model_path', default=None,
                        help='Local SavedModel directory. Recommended on Kaggle or '
                             'when the model is already cached.')

    parser.add_argument('--hop_size_s', type=float, default=2.5,
                        help='Perch window stride in seconds (default: 2.5). '
                             '2.5 → 2× coverage per chunk (recommended). '
                             '1.25 → 4× coverage at ~4× compute cost.')
    parser.add_argument('--chunk_dur', type=int, default=CHUNK_DUR_S,
                        help='Output chunk duration in seconds (default: 5).')

    parser.add_argument('--force_cpu', action='store_true',
                        help='Hide GPUs from TensorFlow (CUDA_VISIBLE_DEVICES=""). '
                             'Use when the installed CuDNN version is incompatible.')

    args = parser.parse_args()

    if args.force_cpu:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print('GPU disabled (--force_cpu).')

    out_dir        = Path(args.out_dir)
    soundscape_dir = Path(args.soundscape_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Enable TF memory growth before any TF GPU operation
    try:
        import tensorflow as _tf
        for _gpu in _tf.config.list_physical_devices('GPU'):
            _tf.config.experimental.set_memory_growth(_gpu, True)
    except Exception:
        pass

    # ── Load model ────────────────────────────────────────────────────────────
    from ml_collections import config_dict
    from perch_hoplite.zoo.taxonomy_model_tf import TaxonomyModelTF

    if args.model_path:
        model_path = str(args.model_path)
        print(f'Loading Perch from: {model_path}')
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

    # ── Locate the actual saved_model.pb ────────────────────────────────────
    # kagglehub sometimes returns a parent directory; tf.saved_model.load then
    # loads a stub _UserObject that lacks 'infer_tf'.  Descend until we find
    # the directory that directly contains saved_model.pb.
    _model_dir = Path(model_path)
    if not (_model_dir / 'saved_model.pb').exists():
        _candidates = sorted(_model_dir.rglob('saved_model.pb'))
        if _candidates:
            model_path = str(_candidates[0].parent)
            print(f'  SavedModel located at: {model_path}')
        else:
            print(f'  WARNING: saved_model.pb not found under {model_path}')

    _fix_duplicate_class_csvs(model_path)
    cfg = config_dict.ConfigDict({
        'model_path':    model_path,
        'sample_rate':   SR,
        'window_size_s': WINDOW_S,
        'hop_size_s':    args.hop_size_s,
        'target_peak':   0.25,
    })
    model = TaxonomyModelTF.from_config(cfg)
    print(f'Perch loaded   (hop_size_s={args.hop_size_s}s → '
          f'~{int(WINDOW_S / args.hop_size_s)} overlapping frames per {args.chunk_dur}s chunk)')

    # ── Vocabulary ────────────────────────────────────────────────────────────
    label_map_path = Path(args.label_map)
    if not label_map_path.exists():
        raise FileNotFoundError(f'label_map.npy not found: {label_map_path}')
    idx2species: Dict[int, str] = np.load(label_map_path, allow_pickle=True).item()
    num_classes = len(idx2species)
    print(f'Vocabulary     : {num_classes} species  (from {label_map_path.name})')

    logit_key, mapped_pos, mapped_perch_idx, proxy_map = _build_vocab_mapping(
        model, Path(args.base_dir), idx2species, model_path=model_path)

    # ── Build inference callable ──────────────────────────────────────────────
    # Abstracts over TF2 concrete-function models (model.embed works) and
    # TF1-format models (inference via model.model.signatures only).
    infer_fn = _build_perch_infer_fn(model, logit_key)

    # ── Manifest ─────────────────────────────────────────────────────────────
    manifest_path = out_dir / 'sc_pl.csv'
    if manifest_path.exists():
        print(f'Using existing manifest: {manifest_path}')
        manifest = pd.read_csv(manifest_path).astype({'start_sec': int, 'end_sec': int})
        missing  = [f for f in manifest['filename'].unique()
                    if not (soundscape_dir / f).exists()]
        if missing:
            print(f'  WARNING: {len(missing)} manifest file(s) not found in '
                  f'soundscape_dir (e.g. {missing[0]})')
    else:
        print('Building chunk manifest ...')
        manifest = _build_manifest(soundscape_dir, args.chunk_dur)
        manifest.to_csv(manifest_path, index=False)
        print(f'  Saved {len(manifest):,} chunks  '
              f'({manifest["filename"].nunique():,} files) → {manifest_path.name}')

    M = len(manifest)
    print(f'Total chunks   : {M:,}  ({manifest["filename"].nunique():,} files)\n')

    # ── Skip if already done ──────────────────────────────────────────────────
    out_path = out_dir / 'sc_pl_preds_perch.npy'
    if out_path.exists():
        existing = np.load(out_path, mmap_mode='r').shape
        if existing == (M, num_classes):
            print(f'[SKIP] {out_path.name} already exists with correct shape {existing}.')
            print('Delete it to force re-extraction.')
            return
        print(f'[REDO] {out_path.name}: shape mismatch {existing} ≠ ({M}, {num_classes})')

    # ── Build filename → manifest-row index lookup ────────────────────────────
    file_to_row_indices: Dict[str, List[int]] = {}
    for row_idx, row in manifest.iterrows():
        file_to_row_indices.setdefault(row['filename'], []).append(row_idx)

    # ── Inference loop ────────────────────────────────────────────────────────
    all_files = manifest['filename'].unique()
    preds     = np.zeros((M, num_classes), dtype=np.float32)
    n_errors  = 0

    for filename in tqdm(all_files, desc='Soundscapes'):
        wave = _load_soundscape(soundscape_dir / filename)
        if wave is None:
            n_errors += 1
            continue

        try:
            file_probs = _infer_file(
                infer_fn, wave,
                args.hop_size_s, args.chunk_dur,
                mapped_pos, mapped_perch_idx, proxy_map, num_classes,
            )
        except Exception as e:
            print(f'  WARNING: inference failed for {filename}: {e}')
            n_errors += 1
            continue

        row_indices   = file_to_row_indices[filename]
        n_file_chunks = len(row_indices)
        n_produced    = len(file_probs)

        if n_produced != n_file_chunks:
            # Duration mismatch between manifest scan and current audio read
            n_copy = min(n_produced, n_file_chunks)
            for k in range(n_copy):
                preds[row_indices[k]] = file_probs[k]
            if n_produced < n_file_chunks:
                print(f'  WARNING: {filename}: manifest expects {n_file_chunks} chunks, '
                      f'Perch produced {n_produced} — trailing rows left as 0')
        else:
            for k, row_idx in enumerate(row_indices):
                preds[row_idx] = file_probs[k]

        gc.collect()

    if n_errors:
        print(f'\nWARNING: {n_errors} file(s) skipped (load or inference error)')

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(out_path, preds)
    print(f'\nSaved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)')
    print(f'  shape : {preds.shape}  dtype={preds.dtype}')

    max_per_chunk = preds.max(axis=1)
    has_signal    = max_per_chunk > 0
    print(f'  chunks with any prediction > 0 : '
          f'{has_signal.sum():,} / {M:,}  ({100*has_signal.mean():.1f}%)')
    if has_signal.any():
        print(f'  mean max-prob (non-zero chunks): {max_per_chunk[has_signal].mean():.4f}')

    print(f'\nManifest : {manifest_path}')
    print(f'Preds    : {out_path}')
    print('\nBoth files are compatible with generate_sc_pl.py out_dir layout.')
    print('Load sc_pl_preds_perch.npy alongside sc_pl_preds_<model>.npy arrays')
    print('in SoundscapePLDataset for ensemble soft-label training.')


if __name__ == '__main__':
    main()
