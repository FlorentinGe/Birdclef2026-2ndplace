"""
Attention quality visualization — Diagnostic 3.

Usage from a notebook (mel tensor already computed):

    import sys; sys.path.insert(0, '/path/to/Codebase')
    from attention_viz import plot_attention_grid

    model.eval()
    fig, data = plot_attention_grid(model, mel_tensor, idx2species, top_k=8,
                                    title='exp53 ep14')
    fig.savefig('attention.png', dpi=150, bbox_inches='tight')

    # Raw arrays for numerical inspection:
    # data['top_att']    : (top_k, T) attention weights for top-k species
    # data['att_full']   : (T, num_classes) full attention matrix
    # data['entropies']  : (top_k,) per-class Shannon entropy (nats)
    # data['top_species']: list of species codes
    # data['top_probs']  : (top_k,) predicted probabilities
    # data['H_max']      : log(T) — uniform-distribution ceiling
    # data['probs']      : (num_classes,) all class probabilities

Usage from CLI (loads a raw audio file):

    python attention_viz.py \\
        --checkpoint experiments/exp53/best_sc_model.pth \\
        --config     experiments/exp53/run_config.json \\
        --species    experiments/exp53/label_map.npy \\
        --audio      /path/to/soundscape.ogg \\
        [--out attention.png] [--top_k 8] [--chunk_idx 0]

    Saves both attention.png AND attention.npz (same stem).

Interpretation guide
--------------------
  Sparse heat map (bright isolated bands on dark background)
      → attention is temporally selective; the head contributes real temporal
        localisation on top of the simple frame mean.
        att_vs_mean_gap > 0.01 expected.

  Flat heat map (uniform colour across all T)
      → attention is decorative; entropy ≈ log(T); the head adds nothing beyond
        clipwise_logits (mean).  att_vs_mean_gap ≈ 0.

  Edge-focused heat map (bright only at first/last time steps)
      → positional shortcut: the model attends to clip boundaries rather than
        call content.  Low entropy but NOT genuine temporal localisation.
        Common cause: zero-padding during training always at the same edge,
        so the model learns the padding boundary as a discriminative cue.
        Fix: randomise padding position in load_audio_chunk (random_pad=True).
        Note: switching to CE loss does NOT fix this — empirically it makes
        attention flatter while leaving edge bias unchanged or slightly worse.

Per-class entropy H (shown on y-axis labels):
    H = -Σ_t w_t log(w_t)  [nats]
    H_max = log(T_sed_frames);  e.g. log(4) ≈ 1.39 for T=4, log(40) ≈ 3.69
    H/H_max < 0.5 AND peak NOT at frame 0 or T-1 → genuine temporal localisation.
    H/H_max < 0.5 AND peak at frame 0 or T-1   → edge-position shortcut.
    H/H_max → 1.0                               → flat/decorative attention.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

__all__ = ['plot_attention_grid', 'run_attention']


# ── Core forward pass ─────────────────────────────────────────────────────────

def run_attention(
    model: nn.Module,
    mel: torch.Tensor,
    idx2species: Dict[int, str],
    top_k: int = 8,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Run one forward pass and return raw attention data for numerical inspection.

    Args:
        model       : BirdCLEFModel in eval mode; must support return_att=True.
        mel         : mel spectrogram — any of (n_mels, T), (C, n_mels, T),
                      or (1, C, n_mels, T).
        idx2species : {class_idx: species_code_or_name}
        top_k       : number of top predicted classes to extract
        device      : inference device; inferred from mel if not given

    Returns dict with keys:
        probs        (num_classes,)  — sigmoid probabilities for all classes
        att_full     (T, num_classes)— full attention weight matrix
        top_indices  (top_k,)        — class indices of top-k predictions
        top_species  list[str]       — species codes for top-k
        top_probs    (top_k,)        — probabilities for top-k
        top_att      (top_k, T)      — attention weights for top-k classes
        entropies    (top_k,)        — Shannon entropy per top-k class (nats)
        H_max        float           — log(T), uniform-distribution ceiling
        T            int             — number of SED time steps
        edge_bias    (top_k,)        — max(att[0], att[-1]) / att.max() per class;
                                       > 0.5 flags likely edge-position shortcut
    """
    if device is None:
        device = mel.device if hasattr(mel, 'device') else torch.device('cpu')

    model = model.to(device).eval()

    x = mel
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(device)

    with torch.inference_mode():
        _, att_clipwise, _, att_weights = model(x, return_att=True)
        probs = torch.sigmoid(att_clipwise)[0].cpu().float().numpy()  # (C,)
        att   = att_weights[0].cpu().float().numpy()                  # (T, C)

    T = att.shape[0]
    top_idx  = np.argsort(probs)[::-1][:top_k]
    top_spc  = [idx2species.get(int(i), str(i)) for i in top_idx]
    top_prob = probs[top_idx]
    top_att  = att[:, top_idx].T                           # (top_k, T)

    w = att[:, top_idx] + 1e-8                             # (T, top_k)
    H = -(w * np.log(w)).sum(axis=0)                       # (top_k,)
    H_max = float(np.log(T)) if T > 1 else 1.0

    # Edge-bias score: how dominant are the first/last frames vs the peak?
    att_max   = top_att.max(axis=1)                        # (top_k,)
    edge_vals = np.maximum(top_att[:, 0], top_att[:, -1])  # (top_k,)
    edge_bias = np.where(att_max > 0, edge_vals / att_max, 0.0)

    return {
        'probs':       probs,
        'att_full':    att,
        'top_indices': top_idx,
        'top_species': top_spc,
        'top_probs':   top_prob,
        'top_att':     top_att,
        'entropies':   H,
        'H_max':       H_max,
        'T':           T,
        'edge_bias':   edge_bias,
    }


# ── Figure ────────────────────────────────────────────────────────────────────

def plot_attention_grid(
    model: nn.Module,
    mel: torch.Tensor,
    idx2species: Dict[int, str],
    top_k: int = 8,
    title: str = '',
    device: Optional[torch.device] = None,
) -> Tuple['matplotlib.figure.Figure', Dict]:
    """
    Plot mel spectrogram + per-class attention heat map for one clip.

    Returns (fig, data) where data is the dict from run_attention().
    The data dict lets callers inspect raw arrays without re-running the model.
    """
    import matplotlib.pyplot as plt

    data    = run_attention(model, mel, idx2species, top_k=top_k, device=device)
    top_att = data['top_att']      # (top_k, T)
    top_spc = data['top_species']
    top_prob= data['top_probs']
    H       = data['entropies']
    H_max   = data['H_max']
    T       = data['T']
    edge_b  = data['edge_bias']

    if mel.dim() == 2:
        mel_disp = mel.cpu().numpy()
    elif mel.dim() == 3:
        mel_disp = mel[0].cpu().numpy()
    else:
        mel_disp = mel[0, 0].cpu().numpy()

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, max(4, top_k * 0.65 + 2)),
        gridspec_kw={'width_ratios': [3, 2]},
    )

    ax_mel = axes[0]
    ax_mel.imshow(mel_disp, origin='lower', aspect='auto',
                  cmap='inferno', interpolation='nearest')
    ax_mel.set_xlabel('Time frame (mel)')
    ax_mel.set_ylabel('Mel bin')
    ax_mel.set_title(f'{title}  —  mel spectrogram' if title else 'Mel spectrogram')

    ax_att = axes[1]
    im = ax_att.imshow(top_att, aspect='auto', cmap='hot',
                       interpolation='nearest', vmin=0.0)
    ax_att.set_xlabel(f'SED time step  (T={T},  H_max={H_max:.2f})')
    ax_att.set_yticks(range(top_k))
    # Flag edge-biased rows with a ⚠ marker
    labels = [
        f'{"⚠ " if eb > 0.5 else ""}{sp}  p={p:.2f}  H={h:.2f}  eb={eb:.2f}'
        for sp, p, h, eb in zip(top_spc, top_prob, H, edge_b)
    ]
    ax_att.set_yticklabels(labels, fontsize=8)
    ax_att.set_title('Attention weights  (top classes)')
    fig.colorbar(im, ax=ax_att, fraction=0.04, pad=0.02, label='att weight')
    fig.tight_layout()
    return fig, data


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli_main() -> None:
    parser = argparse.ArgumentParser(
        description='BirdCLEF attention visualization (Diagnostic 3)')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config',     required=True,
                        help='run_config.json from the experiment directory')
    parser.add_argument('--species',    required=True,
                        help='label_map.npy (saved by train.py) or a .txt with one ebird code per line')
    parser.add_argument('--audio',      required=True,
                        help='audio file (.ogg / .wav / .flac) to visualize')
    parser.add_argument('--out',        default='attention.png')
    parser.add_argument('--top_k',      type=int, default=8)
    parser.add_argument('--chunk_idx',  type=int, default=0,
                        help='which 20 s chunk to visualize (0-indexed)')
    args = parser.parse_args()

    import json, sys, dataclasses
    import torchaudio
    import torchaudio.transforms as AT
    sys.path.insert(0, str(Path(__file__).parent))
    from birdclef.config import Config
    from birdclef.model import BirdCLEFModel

    with open(args.config) as f:
        cfg_raw = json.load(f)
    known = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in cfg_raw.items() if k in known})

    sp_path = Path(args.species)
    if sp_path.suffix == '.npy':
        idx2species = np.load(sp_path, allow_pickle=True).item()
    else:
        with open(sp_path) as f:
            species_list = [ln.strip() for ln in f if ln.strip()]
        idx2species = dict(enumerate(species_list))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = BirdCLEFModel(cfg, num_classes=len(idx2species), pretrained=False)
    ckpt   = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    model  = model.to(device).eval()

    waveform, sr = torchaudio.load(args.audio)
    if sr != cfg.sr:
        waveform = AT.Resample(sr, cfg.sr)(waveform)
    waveform = waveform.mean(dim=0)

    chunk_samples = cfg.duration * cfg.sr
    start  = args.chunk_idx * chunk_samples
    chunk  = waveform[start: start + chunk_samples]
    if len(chunk) < chunk_samples:
        chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - len(chunk)))
    chunk = chunk.unsqueeze(0).to(device)

    mel_fn = AT.MelSpectrogram(
        sample_rate=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=cfg.fmin,
        f_max=cfg.fmax,
    ).to(device)
    mel = (mel_fn(chunk) + 1e-6).log()
    if cfg.in_chans == 3:
        mel = mel.repeat(3, 1, 1)

    stem  = Path(args.audio).stem
    title = f'{stem}  chunk={args.chunk_idx}'
    fig, data = plot_attention_grid(
        model, mel, idx2species,
        top_k=args.top_k, title=title, device=device,
    )

    out_png = Path(args.out)
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f'Saved → {out_png}')

    # Save companion .npz with all raw arrays for numerical inspection
    out_npz = out_png.with_suffix('.npz')
    np.savez_compressed(
        out_npz,
        probs        = data['probs'],
        att_full     = data['att_full'],
        top_indices  = data['top_indices'],
        top_species  = np.array(data['top_species'], dtype=object),
        top_probs    = data['top_probs'],
        top_att      = data['top_att'],
        entropies    = data['entropies'],
        edge_bias    = data['edge_bias'],
        H_max        = np.array(data['H_max']),
        T            = np.array(data['T']),
    )
    print(f'Saved → {out_npz}')

    # Print summary table to stdout
    print(f'\n{"species":<16} {"prob":>6} {"H":>6} {"H/Hmax":>7} {"edge_bias":>9}  note')
    print('-' * 60)
    for sp, p, h, eb in zip(
            data['top_species'], data['top_probs'],
            data['entropies'],   data['edge_bias']):
        ratio = h / data['H_max']
        note  = '⚠ edge' if eb > 0.5 else ('flat' if ratio > 0.85 else 'ok')
        print(f'{sp:<16} {p:>6.3f} {h:>6.3f} {ratio:>7.3f} {eb:>9.3f}  {note}')


if __name__ == '__main__':
    _cli_main()
