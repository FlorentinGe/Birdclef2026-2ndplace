#!/usr/bin/env python3
"""
Blend per-model soundscape pseudo-label arrays into a single ensemble array.

Reads the sc_pl_preds_<name>.npy files produced by generate_sc_pl.py and
writes one sc_pl_preds_ensemble.npy to the same directory (or --out_path).

Default ensembling strategy (Option B)
---------------------------------------
Weighted mean over the four CNN models with weights that correct for the
near-duplicate exp41/exp43 pair (cosine similarity 0.86):

    weight  model
    ------  -----
    1.0     exp38
    0.5     exp41
    0.5     exp43
    1.0     exp47

This is equivalent to treating [exp38, mean(exp41,exp43), exp47] as three
equally-weighted independent members.

Optional rank-gates (--gates, Option C)
----------------------------------------
After computing the weighted mean, applies one or more models as
multiplicative rank-gates in sequence:

    ensemble *= (0.5 + 0.5 * rank_normalize(gate_model))

where rank_normalize maps each column to fractional ranks in [0, 1].
Gates can only attenuate, never amplify (multiplier floor = 0.5 per gate).
Multiple gates stack multiplicatively.

Use this for two kinds of models:
  - Perch: not calibrated to CNN sigmoid scale, so never averaged in directly.
  - Overconfident CNN models (e.g. exp54): saturated sigmoid outputs make them
    unsuitable for the weighted mean, but their ordinal signal is still useful
    as a gate.  Rank-normalising strips the saturation.

Override model weights
-----------------------
Use --model_weights to specify models and their relative weights explicitly:

    python blend_sc_pl.py --sc_pl_dir runs/sc_pl_round1 \\
        --model_weights exp38:1.0 exp41:0.5 exp43:0.5 exp47:1.0

Usage
-----
    # Default (Option B)
    python blend_sc_pl.py --sc_pl_dir Datasets/sc_pl_round1

    # Perch rank-gate only (Option C, round 1)
    python blend_sc_pl.py --sc_pl_dir Datasets/sc_pl_round1 --gates perch

    # Perch + overconfident CNN as second gate (round 2)
    python blend_sc_pl.py --sc_pl_dir Datasets/sc_pl_round2 \\
        --model_weights exp53:1.0 exp59b:1.0 exp61:1.0 exp62:1.0 \\
        --gates perch exp54

    # Write to a specific path
    python blend_sc_pl.py --sc_pl_dir Datasets/sc_pl_round1 \\
        --out_path Datasets/sc_pl_round1/sc_pl_preds_ensemble.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Default model weights (Option B): corrects for exp41/exp43 near-duplication.
DEFAULT_MODEL_WEIGHTS: list[tuple[str, float]] = [
    ('exp38', 1.0),
    ('exp41', 0.5),
    ('exp43', 0.5),
    ('exp47', 1.0),
]


def _parse_model_weights(specs: list[str]) -> list[tuple[str, float]]:
    result = []
    for spec in specs:
        if ':' not in spec:
            raise argparse.ArgumentTypeError(
                f'--model_weights entries must be name:weight (got "{spec}")')
        name, w = spec.rsplit(':', 1)
        try:
            weight = float(w)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f'Weight must be a float (got "{w}" in "{spec}")')
        if weight < 0:
            raise argparse.ArgumentTypeError(
                f'Weight must be non-negative (got {weight})')
        result.append((name, weight))
    return result


def _rank_normalize(arr: np.ndarray) -> np.ndarray:
    """Per-column fractional rank in [0, 1].  Ties get the average rank."""
    M, C = arr.shape
    ranks = np.empty_like(arr)
    for c in range(C):
        col = arr[:, c]
        order = col.argsort()
        r = np.empty(M, dtype=np.float32)
        r[order] = np.arange(M, dtype=np.float32)
        # Resolve ties: average rank
        unique_vals, inverse, counts = np.unique(col, return_inverse=True,
                                                  return_counts=True)
        cum_r = np.zeros(len(unique_vals), dtype=np.float32)
        running = 0
        for i, cnt in enumerate(counts):
            cum_r[i] = running + (cnt - 1) / 2.0
            running += cnt
        r = cum_r[inverse]
        ranks[:, c] = r / (M - 1)
    return ranks


def blend(
    sc_pl_dir: Path,
    model_weights: list[tuple[str, float]],
    gates: list[str],
    out_path: Path,
) -> None:
    # ── Load and validate arrays ──────────────────────────────────────────────
    print('Loading model prediction arrays ...')
    arrays: list[np.ndarray] = []
    weights: list[float] = []
    ref_shape: tuple | None = None

    for name, w in model_weights:
        if w == 0.0:
            print(f'  [SKIP] {name}: weight is 0.0')
            continue
        path = sc_pl_dir / f'sc_pl_preds_{name}.npy'
        if not path.exists():
            raise FileNotFoundError(
                f'sc_pl_preds_{name}.npy not found in {sc_pl_dir}. '
                'Run generate_sc_pl.py first or adjust --model_weights.')
        arr = np.load(path)
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            raise ValueError(
                f'Shape mismatch: {name} is {arr.shape}, expected {ref_shape}')
        arrays.append(arr)
        weights.append(w)
        print(f'  {name:<12s}  weight={w:.2f}  shape={arr.shape}  '
              f'mean={arr.mean():.4f}')

    if not arrays:
        raise RuntimeError('No model arrays were loaded. Check --model_weights.')

    # ── Weighted mean (Option B) ──────────────────────────────────────────────
    total_weight = sum(weights)
    print(f'\nComputing weighted mean (total weight = {total_weight:.2f}) ...')
    ensemble = np.zeros(ref_shape, dtype=np.float32)
    for arr, w in zip(arrays, weights):
        ensemble += (w / total_weight) * arr

    print(f'  Ensemble  mean={ensemble.mean():.4f}  std={ensemble.std():.4f}  '
          f'% > 0.5: {(ensemble > 0.5).mean()*100:.2f}%')
    print(f'  Avg classes > 0.5 per chunk: '
          f'{(ensemble > 0.5).sum(axis=1).mean():.2f}')

    # ── Rank-gates (Option C) ─────────────────────────────────────────────────
    for gate_name in gates:
        gate_path = sc_pl_dir / f'sc_pl_preds_{gate_name}.npy'
        if not gate_path.exists():
            raise FileNotFoundError(
                f'--gates {gate_name} requires sc_pl_preds_{gate_name}.npy '
                f'in {sc_pl_dir}')
        print(f'\nApplying {gate_name} rank-gate ...')
        gate_arr = np.load(gate_path)
        if gate_arr.shape != ref_shape:
            raise ValueError(
                f'{gate_name} shape {gate_arr.shape} does not match '
                f'model shape {ref_shape}')

        print('  Computing per-class fractional ranks (this may take ~30s) ...')
        gate_rank = _rank_normalize(gate_arr)
        ensemble = ensemble * (0.5 + 0.5 * gate_rank)  # multiplier in [0.5, 1.0]

        print(f'  Post-gate  mean={ensemble.mean():.4f}  std={ensemble.std():.4f}  '
              f'% > 0.5: {(ensemble > 0.5).mean()*100:.2f}%')
        print(f'  Avg classes > 0.5 per chunk: '
              f'{(ensemble > 0.5).sum(axis=1).mean():.2f}')

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(out_path, ensemble)
    print(f'\nSaved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)')


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    parser.add_argument(
        '--sc_pl_dir', required=True,
        help='Directory containing sc_pl_preds_<name>.npy files '
             '(output of generate_sc_pl.py)')
    parser.add_argument(
        '--model_weights', nargs='+', default=None,
        metavar='NAME:WEIGHT',
        help='Model names and relative weights as name:weight pairs. '
             'Default: exp38:1.0 exp41:0.5 exp43:0.5 exp47:1.0')
    parser.add_argument(
        '--gates', nargs='*', default=[],
        metavar='NAME',
        help='Model names to apply as rank-gates after the weighted mean '
             '(Option C).  Each gate multiplies the ensemble by '
             '(0.5 + 0.5 * rank_normalize(gate_model)), attenuating chunks '
             'and classes where that model assigns a low relative rank.  '
             'Gates are applied in the order listed.  Use "perch" for the '
             'Perch model; use an overconfident CNN name (e.g. exp54) to '
             'include its ordinal signal without letting saturated sigmoid '
             'values pollute the weighted mean.  '
             'Requires sc_pl_preds_<name>.npy in --sc_pl_dir for each name.')
    parser.add_argument(
        '--out_path', default=None,
        help='Output .npy path. Default: <sc_pl_dir>/sc_pl_preds_ensemble.npy')

    args = parser.parse_args()

    sc_pl_dir = Path(args.sc_pl_dir)
    if not sc_pl_dir.is_dir():
        parser.error(f'--sc_pl_dir does not exist: {sc_pl_dir}')

    model_weights = (
        _parse_model_weights(args.model_weights)
        if args.model_weights is not None
        else DEFAULT_MODEL_WEIGHTS
    )

    out_path = Path(args.out_path) if args.out_path else (
        sc_pl_dir / 'sc_pl_preds_ensemble.npy')

    print('=' * 60)
    print('Soundscape pseudo-label blending')
    print('=' * 60)
    print(f'sc_pl_dir   : {sc_pl_dir}')
    print(f'Models      : {[(n, w) for n, w in model_weights]}')
    print(f'Gates       : {args.gates if args.gates else "(none)"}')
    print(f'Output      : {out_path}')
    print()

    blend(sc_pl_dir, model_weights, args.gates, out_path)

    print('\nDone.')


if __name__ == '__main__':
    main()
