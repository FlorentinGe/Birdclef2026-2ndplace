"""
Dataset classes for all training stages.

All datasets that produce waveforms return raw float32 tensors — the GPU-side
MelTransform in the training loop handles mel conversion and augmentation.

Val datasets and FocalValDataset return pre-computed mel spectrograms (CPU)
because they are not augmented and can be converted once per sample rather than
every epoch.

Dataset      | Output           | Used in
-------------|------------------|------------------------------------------
BirdDataset  | (wave, labels)   | Stage 2 training (focal clips)
SoundscapeTrainDataset | (wave, labels) | Stage 2 training (labelled soundscapes)
SoundscapeValDataset | (mel, labels) | Validation (soundscape AUC)
FocalValDataset | (mel, labels) | Validation (focal AUC)
PerchDistillDataset | (wave, emb) | Stage 1 KD pretraining
SoundscapePLDataset | (wave, labels) | Stage 4 training (soundscape pseudo-labels)
SoundscapeUnlabeledValDataset | (mel, pseudo_labels) | Stage 4 val (unlabeled SC OOF tracking)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import soundfile
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, Sampler

from .config import Config
from .transforms import mel_to_spectrogram_cpu


# ── Audio loading ─────────────────────────────────────────────────────────────

def load_audio_chunk(path: str, sr: int, duration: int,
                     pad_type: str = "right",
                     noise_padder: Optional['NoisePadder'] = None) -> np.ndarray:
    """
    Load a random `duration`-second chunk from a WAV file using soundfile.

    soundfile.read with random start is ~2× faster than librosa for random seeks
    into large files.  Falls back to zeros on any error (corrupt / missing file).

    pad_type controls how short clips (< duration seconds) are padded:
      "left"   — padding before the audio, audio at the right end of the buffer.
                 Use for supervised focal training so that MixUp always overlaps:
                 two left-padded clips always share audio content in the rightmost
                 region regardless of their individual lengths.  Matches the 1st
                 place 2025 recipe for Stage 1 (supervised learning).
      "random" — audio placed at a uniformly random position, padding fills the rest.
                 Use for self-training (pseudo-label) stages so that the model
                 cannot rely on any fixed-position shortcut.  Matches the 1st
                 place recipe for Stage 2+ (self-training).
      "right"  — audio at the left end, padding appended at the right (legacy).
                 Use for validation to keep evaluation reproducible.

    noise_padder: when provided, padded regions are filled with low-amplitude
                  environmental noise instead of zeros, preventing the model from
                  learning "zero frames = padding" attention shortcuts.  Has no
                  effect on clips that fill the full duration without padding, and
                  is never applied during validation (noise_padder=None by default).
    """
    target_len = sr * duration
    try:
        info  = soundfile.info(path)
        total = info.frames
        start = random.randint(0, max(0, total - target_len))
        wave, _ = soundfile.read(
            path, frames=target_len, start=start, dtype='float32', always_2d=False)
    except Exception:
        wave = np.zeros(target_len, dtype=np.float32)
    pad_left = pad_right = 0
    if len(wave) < target_len:
        pad_needed = target_len - len(wave)
        if pad_type == "left":
            pad_left = pad_needed
        elif pad_type == "random":
            pad_left  = random.randint(0, pad_needed)
            pad_right = pad_needed - pad_left
        else:  # "right"
            pad_right = pad_needed
        wave = np.pad(wave, (pad_left, pad_right))
    absmax = np.abs(wave).max()
    if absmax > 0:
        wave /= absmax
    if noise_padder is not None and (pad_left > 0 or pad_right > 0):
        if pad_left > 0:
            wave[:pad_left] = noise_padder.sample(pad_left)
        if pad_right > 0:
            wave[-pad_right:] = noise_padder.sample(pad_right)
    return wave


def load_soundscape_chunk(
    filepath: Path,
    start_sec: float,
    sr: int,
    chunk_len: int,
) -> torch.Tensor:
    """
    Load a fixed-length chunk starting at start_sec from a soundscape file.
    Returns a 1-D float32 tensor of length chunk_len, normalised to max=1.
    """
    start_sam = int(start_sec * sr)
    try:
        waveform, orig_sr = torchaudio.load(
            str(filepath), frame_offset=start_sam, num_frames=chunk_len)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if orig_sr != sr:
            waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
        waveform = waveform.squeeze(0)
    except Exception:
        waveform = torch.zeros(chunk_len)
    if waveform.shape[0] < chunk_len:
        waveform = F.pad(waveform, (0, chunk_len - waveform.shape[0]))
    waveform = waveform / (waveform.abs().max() + 1e-6)
    return waveform


def load_soundscape_window(
    filepath: Path,
    start_sec: float,
    sr: int,
    chunk_samples: int,
    window_samples: int,
    noise_padder: Optional['NoisePadder'] = None,
) -> torch.Tensor:
    """
    Load a window_samples-length clip from a soundscape file, placing the
    chunk_samples labeled region at a uniformly random offset within it.

    Eliminates the structural 5s→20s zero-padding that causes the SED
    attention head to attend to silent frames instead of real bird audio.
    The model receives real soundscape context on both sides of the labeled
    chunk and must learn temporal localisation from spectral content alone.

    The window start is:
        window_start = max(0, chunk_start − random_offset)
    so the window is always within the file.  Noise/zero padding is applied
    only at file-boundary edge cases (start or end of recording).

    When window_samples == chunk_samples (e.g. ViT with duration=chunk_duration),
    pad_budget=0 and this is identical to load_soundscape_chunk.
    """
    chunk_start    = int(start_sec * sr)
    pad_budget     = window_samples - chunk_samples          # 15 * sr typically
    offset         = random.randint(0, pad_budget)
    window_start   = max(0, chunk_start - offset)

    try:
        waveform, orig_sr = torchaudio.load(
            str(filepath), frame_offset=window_start, num_frames=window_samples)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)
        if orig_sr != sr:
            waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
    except Exception:
        waveform = torch.zeros(window_samples)

    # Padding only fires at file edges (short read near end of recording)
    if waveform.shape[0] < window_samples:
        shortfall = window_samples - waveform.shape[0]
        if noise_padder is not None:
            waveform = torch.cat([waveform, noise_padder.sample_tensor(shortfall)])
        else:
            waveform = F.pad(waveform, (0, shortfall))

    absmax = waveform.abs().max().item()
    if absmax > 0:
        waveform = waveform / absmax
    return waveform


# ── Training datasets (return waveforms) ─────────────────────────────────────

class BirdDataset(Dataset):
    """
    Focal clip training dataset.

    Returns (waveform, label_vec) where waveform is a (T_samples,) float32 tensor
    and label_vec is a (num_classes,) float32 one-hot + secondary labels vector.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
        num_classes: int,
        species2idx: Dict[str, int],
        noise_padder: Optional['NoisePadder'] = None,
    ):
        self.df           = df.reset_index(drop=True)
        self.cfg          = cfg
        self.num_classes  = num_classes
        self.species2idx  = species2idx
        self.noise_padder = noise_padder

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row  = self.df.iloc[idx]
        wave = load_audio_chunk(row['file_path'], self.cfg.sr, self.cfg.duration,
                                pad_type=self.cfg.focal_pad_type,
                                noise_padder=self.noise_padder)

        label_vec = np.zeros(self.num_classes, dtype=np.float32)
        label_vec[int(row['label'])] = 1.0
        sec = row.get('secondary_label_vec')
        if sec is not None:
            label_vec = np.maximum(label_vec, sec)

        return torch.from_numpy(wave), torch.tensor(label_vec, dtype=torch.float32)


class BirdDatasetWithPL(Dataset):
    """
    Focal clip training dataset with pseudo-label power-transform + blend.

    Applies the 5th-place self-distillation label formula per sample:
        pseudo_t = clip(pseudo*(pseudo>th) + pseudo**power, 0, 1)
        label    = alpha*pseudo_t + (1-alpha)*original_hard_labels
        label    = max(label, perch_soft * perch_caps[c])  [if perch_soft given]

    Args:
        df:           focal_pl DataFrame; must have columns
                      file_path, vec_idx, label (int), secondary_label_vec.
        raw_ensemble: (N, num_classes) float32 — raw CNN ensemble probabilities,
                      no thresholding applied (focal_pl_raw_ensemble.npy).
        cfg:          Config — pl_pseudo_th, pl_pseudo_power, pl_pseudo_alpha,
                      pl_perch_max are read from here.
        num_classes:  total number of classes.
        perch_soft:   optional (N, num_classes) float32 — continuous Perch
                      z-score soft labels (focal_pl_preds_perch_continuous.npy).
        perch_caps:   optional (num_classes,) float32 — per-class Perch cap,
                      overrides cfg.pl_perch_max when provided.  Build with
                      build_perch_caps() in train.py to apply class-conditional
                      caps (e.g. 0.0 for Mammalia, 0.3 for Amphibia/Insecta,
                      cfg.pl_perch_max for Aves).  When None, cfg.pl_perch_max
                      is used as a uniform cap for all classes.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        raw_ensemble: np.ndarray,
        cfg: 'Config',
        num_classes: int,
        perch_soft: Optional[np.ndarray] = None,
        perch_caps: Optional[np.ndarray] = None,
        noise_padder: Optional['NoisePadder'] = None,
    ):
        self.df           = df.reset_index(drop=True)
        self.raw_ensemble = raw_ensemble      # (N, C) float32
        self.cfg          = cfg
        self.num_classes  = num_classes
        self.perch_soft   = perch_soft        # (N, C) float32 or None
        self.noise_padder = noise_padder
        # perch_caps: (C,) float32, or None → fall back to scalar cfg.pl_perch_max
        if perch_caps is not None:
            self.perch_caps = perch_caps.astype(np.float32)
        elif cfg.pl_perch_max > 0.0 and perch_soft is not None:
            self.perch_caps = np.full(num_classes, cfg.pl_perch_max, dtype=np.float32)
        else:
            self.perch_caps = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row  = self.df.iloc[idx]
        wave = load_audio_chunk(row['file_path'], self.cfg.sr, self.cfg.duration,
                                pad_type="random", noise_padder=self.noise_padder)

        # ── Original hard labels (primary + secondary from train.csv) ──────────
        original = np.zeros(self.num_classes, dtype=np.float32)
        original[int(row['label'])] = 1.0
        sec = row.get('secondary_label_vec')
        if sec is not None:
            original = np.maximum(original, sec)

        # ── Power-transform + blend ────────────────────────────────────────────
        pseudo   = self.raw_ensemble[int(row['vec_idx'])]   # (C,) float32
        th       = self.cfg.pl_pseudo_th
        power    = self.cfg.pl_pseudo_power
        alpha    = self.cfg.pl_pseudo_alpha
        pseudo_t = pseudo * (pseudo > th) + pseudo ** power
        pseudo_t = np.clip(pseudo_t, 0.0, 1.0)
        label    = alpha * pseudo_t + (1.0 - alpha) * original

        # ── Optional Perch discovery (continuous z-score, max-only) ───────────
        if self.perch_soft is not None and self.perch_caps is not None:
            perch = self.perch_soft[int(row['vec_idx'])] * self.perch_caps
            label = np.maximum(label, perch)

        label = np.clip(label, 0.0, 1.0)
        return torch.from_numpy(wave), torch.tensor(label, dtype=torch.float32)


class SoundscapeTrainDataset(Dataset):
    """
    Labelled soundscape training dataset.

    Loads a DURATION-second window from the soundscape file with the labeled
    CHUNK_DURATION-second region placed at a random offset within it, using
    real surrounding audio rather than zero-padding.  The SED attention head
    must localise the labeled chunk by spectral content rather than by
    attending to a silent/padded majority of the clip.

    For ViT (DURATION == CHUNK_DURATION == 5s), the window equals the chunk
    and no context extension occurs.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: Path,
        cfg: Config,
        noise_padder: Optional['NoisePadder'] = None,
    ):
        self.df           = df.reset_index(drop=True)
        self.audio_dir    = audio_dir
        self.cfg          = cfg
        self.chunk_len    = cfg.sr * cfg.chunk_duration
        self.full_len     = cfg.sr * cfg.duration
        self.noise_padder = noise_padder

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row      = self.df.iloc[idx]
        filepath = self.audio_dir / row['filename']
        if self.cfg.sc_use_window:
            waveform = load_soundscape_window(
                filepath, row['start_sec'], self.cfg.sr,
                self.chunk_len, self.full_len,
                noise_padder=self.noise_padder)
        else:
            waveform = load_soundscape_chunk(filepath, row['start_sec'], self.cfg.sr, self.chunk_len)
            pad_total = self.full_len - self.chunk_len
            if pad_total > 0:
                pad_left = random.randint(0, pad_total)
                waveform = F.pad(waveform, (pad_left, pad_total - pad_left))
        label_vec = torch.tensor(row['label_vec'], dtype=torch.float32)
        return waveform, label_vec


# ── Validation datasets (return mel spectrograms) ────────────────────────────

class SoundscapeValDataset(Dataset):
    """
    Soundscape validation dataset.

    Returns (mel, label_vec) where mel is (3, n_mels, T_chunk) float32.
    Mel conversion and optional ImageNet normalisation are done on CPU in
    __getitem__ using lazy-initialised torchaudio transforms (DataLoader
    worker-safe pattern).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: Path,
        cfg: Config,
    ):
        self.df        = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.cfg       = cfg
        self.chunk_len = cfg.sr * cfg.chunk_duration

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row       = self.df.iloc[idx]
        filepath  = self.audio_dir / row['filename']
        start_sam = int(row['start_sec'] * self.cfg.sr)
        try:
            waveform, orig_sr = torchaudio.load(
                str(filepath), frame_offset=start_sam, num_frames=self.chunk_len)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if orig_sr != self.cfg.sr:
                waveform = torchaudio.functional.resample(waveform, orig_sr, self.cfg.sr)
        except Exception:
            waveform = torch.zeros(1, self.chunk_len)
        if waveform.shape[-1] < self.chunk_len:
            waveform = F.pad(waveform, (0, self.chunk_len - waveform.shape[-1]))

        mel = mel_to_spectrogram_cpu(
            waveform.squeeze(0), self.cfg,
            imagenet_norm=self.cfg.imagenet_norm, in_chans=self.cfg.in_chans)
        return mel, torch.tensor(row['label_vec'], dtype=torch.float32)


class FocalValDataset(Dataset):
    """
    Focal clip validation dataset.

    Returns (mel, label_vec) where mel is (3, n_mels, T_duration) float32.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
        num_classes: int,
    ):
        self.df          = df.reset_index(drop=True)
        self.cfg         = cfg
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row  = self.df.iloc[idx]
        wave = load_audio_chunk(row['file_path'], self.cfg.sr, self.cfg.duration)

        mel = mel_to_spectrogram_cpu(
            torch.from_numpy(wave), self.cfg,
            imagenet_norm=self.cfg.imagenet_norm, in_chans=self.cfg.in_chans)

        label_vec = np.zeros(self.num_classes, dtype=np.float32)
        label_vec[int(row['label'])] = 1.0
        return mel, torch.tensor(label_vec, dtype=torch.float32)


class SoundscapeUnlabeledValDataset(Dataset):
    """
    Unlabeled soundscape validation dataset for OOF tracking in Stage 4.

    Mirrors SoundscapeValDataset but uses ensemble pseudo-labels as the label
    vector (raw probabilities, not power-transformed) so that the training
    script can measure model-vs-PL consistency across epochs.

    Args:
        manifest_df:    sc_pl.csv subset for the held-out val files; row i must
                        correspond to ensemble_preds[i].
        ensemble_preds: (M, num_classes) float32 from blend_sc_pl.py.
        audio_dir:      directory of soundscape OGG files.
        cfg:            Config — sr, chunk_duration, imagenet_norm, in_chans.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        ensemble_preds: np.ndarray,
        audio_dir: Path,
        cfg: Config,
    ):
        self.manifest       = manifest_df.reset_index(drop=True)
        self.ensemble_preds = ensemble_preds
        self.audio_dir      = audio_dir
        self.cfg            = cfg
        self.chunk_len      = cfg.sr * cfg.chunk_duration

        if len(self.manifest) != len(self.ensemble_preds):
            raise ValueError(
                f'manifest has {len(self.manifest)} rows but '
                f'ensemble_preds has {len(self.ensemble_preds)} rows.')

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row       = self.manifest.iloc[idx]
        filepath  = self.audio_dir / row['filename']
        start_sam = int(row['start_sec'] * self.cfg.sr)
        try:
            waveform, orig_sr = torchaudio.load(
                str(filepath), frame_offset=start_sam, num_frames=self.chunk_len)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if orig_sr != self.cfg.sr:
                waveform = torchaudio.functional.resample(waveform, orig_sr, self.cfg.sr)
        except Exception:
            waveform = torch.zeros(1, self.chunk_len)
        if waveform.shape[-1] < self.chunk_len:
            waveform = F.pad(waveform, (0, self.chunk_len - waveform.shape[-1]))

        mel = mel_to_spectrogram_cpu(
            waveform.squeeze(0), self.cfg,
            imagenet_norm=self.cfg.imagenet_norm, in_chans=self.cfg.in_chans)
        pseudo_label = torch.tensor(self.ensemble_preds[idx], dtype=torch.float32)
        return mel, pseudo_label


# ── Background augmentation ───────────────────────────────────────────────────

class NocallAugmenter:
    """
    Per-batch background augmentation using nocall soundscape segments.

    Applied on CPU in the training loop before GPU transfer.  Randomly picks a
    5-second nocall chunk, tiles it to the training clip length, and mixes it in
    at a randomly sampled SNR in [snr_min_db, snr_max_db].

    Labels are NOT modified — nocall segments contain no recognisable bird calls,
    so there is no positive-label signal to add and no punishment-for-detection
    problem that affects plain BCE with arbitrary backgrounds.

    Usage:
        augmenter = NocallAugmenter(nocall_meta, soundscape_dir, cfg)
        # in training loop, before .to(device):
        waves = augmenter(waves)   # (B, T_samples) CPU tensor → same shape
    """

    def __init__(
        self,
        nocall_meta: pd.DataFrame,     # columns: filename, start_sec
        soundscape_dir: Path,
        cfg: Config,
    ):
        self.meta          = nocall_meta.reset_index(drop=True)
        self.soundscape_dir = soundscape_dir
        self.sr            = cfg.sr
        self.chunk_len     = cfg.sr * cfg.chunk_duration   # 5 s of samples
        self.target_len    = cfg.sr * cfg.duration         # 20 s of samples
        self.prob          = cfg.bg_aug_prob
        self.snr_min       = cfg.bg_aug_snr_min_db
        self.snr_max       = cfg.bg_aug_snr_max_db

    def _load_bg(self) -> torch.Tensor:
        """Load a random nocall segment and tile it to target_len."""
        row      = self.meta.iloc[random.randint(0, len(self.meta) - 1)]
        filepath = self.soundscape_dir / row['filename']
        chunk    = load_soundscape_chunk(
            filepath, float(row['start_sec']), self.sr, self.chunk_len)
        # Tile 5 s → 20 s (ceiling division then slice)
        repeats = -(-self.target_len // len(chunk))
        return chunk.repeat(repeats)[:self.target_len]

    def __call__(self, waves: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waves: (B, T_samples) CPU float32 tensor

        Returns:
            Augmented tensor of the same shape.
        """
        waves = waves.clone()
        for i in range(waves.shape[0]):
            if random.random() >= self.prob:
                continue
            bg     = self._load_bg()
            fg_rms = waves[i].pow(2).mean().sqrt().item()
            bg_rms = bg.pow(2).mean().sqrt().item()
            if fg_rms < 1e-6 or bg_rms < 1e-6:
                continue
            snr_db = random.uniform(self.snr_min, self.snr_max)
            # Scale background so its RMS is fg_rms * 10^(-SNR/20)
            scale        = fg_rms / bg_rms * (10.0 ** (-snr_db / 20.0))
            waves[i]     = waves[i] + bg * scale
            # Re-normalise to max=1 to keep the same dynamic range as before
            absmax = waves[i].abs().max().item()
            if absmax > 0:
                waves[i] = waves[i] / absmax
        return waves


class ExternalNoiseAugmenter:
    """
    Background augmentation from a directory of pre-filtered external audio files
    (e.g. ESC-50 with bird/frog/insect categories removed).

    Unlike NocallAugmenter (which relies on Perch thresholding to find
    "quiet" segments), these files are verified abiotic — safe for use with
    focal loss because there is no risk of amplifying contradictory label noise.

    Files shorter than the target length are tiled (repeated), then a random
    window is taken.  Longer files get a random start offset, so the full pool
    is explored over training.

    Usage:
        augmenter = ExternalNoiseAugmenter(Path('data/esc50_filtered'), cfg)
        waves = augmenter(waves)   # (B, T_samples) CPU tensor → same shape
    """

    def __init__(self, noise_dir: Path, cfg: Config, prob: float = None):
        self.files = (
            sorted(noise_dir.glob('**/*.wav')) +
            sorted(noise_dir.glob('**/*.ogg'))
        )
        if not self.files:
            raise FileNotFoundError(f'No audio files found in {noise_dir}')
        self.sr         = cfg.sr
        self.target_len = cfg.sr * cfg.duration
        self.prob       = prob if prob is not None else cfg.bg_aug_prob
        self.snr_min    = cfg.bg_aug_snr_min_db
        self.snr_max    = cfg.bg_aug_snr_max_db
        print(f'External noise pool: {len(self.files)} files  ({noise_dir})')

    def _load_bg(self) -> torch.Tensor:
        """Load a random file, tile if needed, return a target_len window."""
        path = random.choice(self.files)
        try:
            wave, sr = soundfile.read(str(path), dtype='float32', always_2d=False)
            if wave.ndim > 1:
                wave = wave.mean(axis=1)
            if sr != self.sr:
                wave = torchaudio.functional.resample(
                    torch.from_numpy(wave), sr, self.sr).numpy()
        except Exception:
            wave = np.zeros(self.target_len, dtype=np.float32)

        # Tile short files to at least target_len
        if len(wave) < self.target_len:
            repeats = -(-self.target_len // len(wave))   # ceil division
            wave    = np.tile(wave, repeats)

        start = random.randint(0, len(wave) - self.target_len)
        return torch.from_numpy(wave[start : start + self.target_len].copy())

    def __call__(self, waves: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waves: (B, T_samples) CPU float32 tensor

        Returns:
            Augmented tensor of the same shape.
        """
        waves = waves.clone()
        for i in range(waves.shape[0]):
            if random.random() >= self.prob:
                continue
            bg     = self._load_bg()
            fg_rms = waves[i].pow(2).mean().sqrt().item()
            bg_rms = bg.pow(2).mean().sqrt().item()
            if fg_rms < 1e-6 or bg_rms < 1e-6:
                continue
            snr_db   = random.uniform(self.snr_min, self.snr_max)
            scale    = fg_rms / bg_rms * (10.0 ** (-snr_db / 20.0))
            waves[i] = waves[i] + bg * scale
            absmax   = waves[i].abs().max().item()
            if absmax > 0:
                waves[i] /= absmax
        return waves


# ── Noise padding ────────────────────────────────────────────────────────────

class NoisePadder:
    """
    Fills zero-padded regions of short training clips with low-amplitude
    environmental noise drawn from the same external noise bank used by
    ExternalNoiseAugmenter.

    Zero-padding creates a uniform feature for every padded frame (all-zero
    mel bins → identical CNN activations), which the SED attention head learns
    to exploit as a positional shortcut — attending to silent padding rather than
    actual bird calls.  Replacing zeros with varied noise breaks this attractor:
    padded frames now produce distinct, non-repeating features so the model must
    attend to spectral content rather than frame identity.

    Unlike ExternalNoiseAugmenter (which mixes noise into the entire clip at
    SNR 6–24 dB and therefore perturbs real signal), NoisePadder only writes into
    the gap that was already empty — the actual bird-call audio is untouched.

    Design choices to prevent the model overfitting to the noise itself:
      - Amplitude is re-sampled per call from [amp_min, amp_max] (default ~30–40 dB
        below a normalised peak), so no fixed amplitude level becomes a padding cue.
      - A random contiguous crop of a randomly chosen file is drawn each call,
        maximising spectral variety across training steps.
      - Files are loaded and normalised lazily, then cached per DataLoader worker;
        no shared cross-process state, no start-up overhead.

    Never applied during validation or inference (noise_padder=None by default
    in all val/test code paths).
    """

    def __init__(
        self,
        noise_dir: Path,
        sr: int,
        amp_min: float = 0.003,
        amp_max: float = 0.015,
    ):
        self.files = (sorted(noise_dir.glob('**/*.wav')) +
                      sorted(noise_dir.glob('**/*.ogg')))
        if not self.files:
            raise FileNotFoundError(f'NoisePadder: no audio files found in {noise_dir}')
        self.sr      = sr
        self.amp_min = amp_min
        self.amp_max = amp_max
        self._cache: dict = {}   # str(path) → np.ndarray, populated lazily per worker
        print(f'NoisePadder: {len(self.files)} files in {noise_dir}  '
              f'amp=[{amp_min}, {amp_max}]')

    def _load(self, path: Path) -> np.ndarray:
        key = str(path)
        if key not in self._cache:
            try:
                wave, file_sr = soundfile.read(str(path), dtype='float32', always_2d=False)
                if wave.ndim > 1:
                    wave = wave.mean(axis=1)
                if file_sr != self.sr:
                    wave = torchaudio.functional.resample(
                        torch.from_numpy(wave), file_sr, self.sr).numpy()
                absmax = np.abs(wave).max()
                if absmax > 0:
                    wave = wave / absmax
            except Exception:
                wave = np.zeros(self.sr, dtype=np.float32)
            self._cache[key] = wave.astype(np.float32)
        return self._cache[key]

    def sample(self, n_samples: int) -> np.ndarray:
        """Return `n_samples` of noise at a randomised amplitude (numpy float32)."""
        if n_samples <= 0:
            return np.zeros(0, dtype=np.float32)
        wave = self._load(random.choice(self.files))
        if len(wave) < n_samples:
            wave = np.tile(wave, -(-n_samples // len(wave)))   # ceil tile
        start = random.randint(0, len(wave) - n_samples)
        return wave[start:start + n_samples] * random.uniform(self.amp_min, self.amp_max)

    def sample_tensor(self, n_samples: int) -> torch.Tensor:
        """Return `n_samples` of noise as a float32 torch Tensor."""
        return torch.from_numpy(self.sample(n_samples).copy())


# ── Soundscape pseudo-label dataset (Stage 4) ────────────────────────────────

class SoundscapePLDataset(Dataset):
    """
    Soundscape pseudo-label pool for Stage 4 (1st-place Noisy Student recipe).

    Stores pre-blended ensemble predictions and serves (waveform, label) pairs
    that are consumed by SoundscapeMixupDataset — NOT used directly in a
    DataLoader.

    Label transform (1st-place recipe):
        label = clip(prob ** power, 0, 1)

    where power = cfg.pl_sc_pseudo_power.  Iteration-dependent tuning:
        Round 1: power = 1.0  (identity — use raw ensemble probs)
        Round 2: power ≈ 1.54 (= 1/0.65, suppress noise from round-1 PL)
        Round 3: power ≈ 1.82 (= 1/0.55)
        Round 4: power ≈ 1.67 (= 1/0.60)

    Power > 1 shrinks low-confidence values toward 0 while leaving high-
    confidence values relatively intact, preventing noisy pseudo-labels from
    accumulating across iterations.

    Args:
        manifest_df:    sc_pl.csv DataFrame; row i ↔ row i in ensemble_preds.
        ensemble_preds: (M, num_classes) float32 from blend_sc_pl.py.
        audio_dir:      directory of soundscape OGG files.
        cfg:            Config — pl_sc_pseudo_power, sr, chunk_duration,
                        duration are read from here.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        ensemble_preds: np.ndarray,
        audio_dir: Path,
        cfg: Config,
        noise_padder: Optional['NoisePadder'] = None,
        hard_positive_mask: Optional[np.ndarray] = None,
    ):
        self.manifest           = manifest_df.reset_index(drop=True)
        self.ensemble_preds     = ensemble_preds   # (M, C) float32
        self.audio_dir          = audio_dir
        self.cfg                = cfg
        self.chunk_len          = cfg.sr * cfg.chunk_duration
        self.full_len           = cfg.sr * cfg.duration
        self.noise_padder       = noise_padder
        self.hard_positive_mask = hard_positive_mask  # (M, C) bool or None

        if len(self.manifest) != len(self.ensemble_preds):
            raise ValueError(
                f'Manifest has {len(self.manifest)} rows but ensemble_preds '
                f'has {len(self.ensemble_preds)} rows. '
                'They must match (both from the same blend_sc_pl.py run).')

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row      = self.manifest.iloc[idx]
        filepath = self.audio_dir / row['filename']
        if self.cfg.sc_use_window:
            waveform = load_soundscape_window(
                filepath, int(row['start_sec']), self.cfg.sr,
                self.chunk_len, self.full_len,
                noise_padder=self.noise_padder)
        else:
            waveform = load_soundscape_chunk(filepath, int(row['start_sec']), self.cfg.sr, self.chunk_len)
            pad_total = self.full_len - self.chunk_len
            if pad_total > 0:
                pad_left = random.randint(0, pad_total)
                waveform = F.pad(waveform, (pad_left, pad_total - pad_left))

        raw   = self.ensemble_preds[idx].copy()
        clip_th = self.cfg.pl_sc_clip_th
        if clip_th is not None:
            raw = np.where(raw >= clip_th, 1.0, raw)
        label = np.clip(raw ** self.cfg.pl_sc_pseudo_power, 0.0, 1.0)
        if self.hard_positive_mask is not None:
            label[self.hard_positive_mask[idx]] = 1.0
        return waveform, torch.tensor(label, dtype=torch.float32)


class SoundscapeMixupDataset(Dataset):
    """
    Mixes focal clips with pseudo-labeled soundscape chunks at a fixed 0.5/0.5
    weight, following the 1st-place Noisy Student recipe.

    For every focal clip, a random soundscape chunk is drawn (using
    chunk_weights for file-level weighted sampling) and the two are mixed in the
    waveform domain:
        mixed_wave  = 0.5 * focal_wave + 0.5 * sc_wave
        mixed_label = clip(0.5 * focal_label + 0.5 * sc_label, 0, 1)

    Epoch length equals len(focal_ds) so the learning rate schedule and rare-
    species sampler remain anchored to the focal dataset, consistent with
    supervised training.

    Args:
        focal_ds:      BirdDataset or BirdDatasetWithPL.
        sc_pl_ds:      SoundscapePLDataset — the pseudo-label pool.
        chunk_weights: (M,) float32 — per-chunk sampling probability weights.
                       Build with _sc_chunk_weights() in train.py.
                       Weights proportional to sum-of-max-class-probs per file
                       (1st-place sampler: higher-confidence files sampled more).
    """

    def __init__(
        self,
        focal_ds: Dataset,
        sc_pl_ds: SoundscapePLDataset,
        chunk_weights: np.ndarray,
    ):
        self.focal_ds  = focal_ds
        self.sc_pl_ds  = sc_pl_ds
        w = chunk_weights.astype(np.float64)
        self._sc_probs = w / w.sum()   # normalised for np.random.choice

    def __len__(self) -> int:
        return len(self.focal_ds)

    def __getitem__(self, idx: int):
        focal_wave, focal_label = self.focal_ds[idx]

        sc_idx = int(np.random.choice(len(self.sc_pl_ds), p=self._sc_probs))
        sc_wave, sc_label = self.sc_pl_ds[sc_idx]

        mixed_wave  = 0.5 * focal_wave + 0.5 * sc_wave
        mixed_label = torch.clamp(0.5 * focal_label + 0.5 * sc_label, 0.0, 1.0)
        return mixed_wave, mixed_label


class SoundscapeSubstitutionDataset(Dataset):
    """
    Species-matched substitution strategy (Ali Ozan Memetoglu, 2nd place BirdCLEF 2025).

    For each focal XC clip with primary species S, with probability sub_prob the
    clip is *replaced* by a soundscape pseudo chunk whose top predicted species
    (argmax of ensemble_preds) equals S.  When no SC chunk maps to S, or when
    the random draw fails, the focal clip is returned unchanged.

    The SC chunk carries its own power-transformed soft labels (consistent with
    SoundscapePLDataset) — the focal hard label is discarded on substituted samples.

    Args:
        focal_ds:  BirdDataset — must expose .df with an integer 'label' column.
        sc_pl_ds:  SoundscapePLDataset — pseudo-label pool.
        sub_prob:  Probability in [0, 1] of substituting each focal clip.
    """

    def __init__(
        self,
        focal_ds: Dataset,
        sc_pl_ds: 'SoundscapePLDataset',
        sub_prob: float,
    ):
        self.focal_ds = focal_ds
        self.sc_pl_ds = sc_pl_ds
        self.sub_prob = sub_prob

        # Build per-species index: class_idx → list[sc_pl_ds row indices]
        raw     = sc_pl_ds.ensemble_preds    # (M, C) float32
        top_cls = raw.argmax(axis=1)         # (M,) int
        species_index: dict = {}
        for i, c in enumerate(top_cls.tolist()):
            species_index.setdefault(c, []).append(i)
        self._species_index = species_index

        print(f'SoundscapeSubstitutionDataset: {len(species_index)} species with SC chunks '
              f'({len(sc_pl_ds):,} total chunks, sub_prob={sub_prob})')

    def __len__(self) -> int:
        return len(self.focal_ds)

    def __getitem__(self, idx: int):
        if self.sub_prob > 0.0 and random.random() < self.sub_prob:
            primary_cls = int(self.focal_ds.df.iloc[idx]['label'])
            candidates  = self._species_index.get(primary_cls)
            if candidates:
                sc_idx = candidates[random.randint(0, len(candidates) - 1)]
                return self.sc_pl_ds[sc_idx]

        return self.focal_ds[idx]


# ── KD distillation dataset ───────────────────────────────────────────────────

class PerchDistillDataset(Dataset):
    """
    Stage 1: returns (waveform_5s, perch_embedding) pairs for KD pretraining.

    Audio is loaded from soundscape files at positions encoded in the Perch
    metadata CSV/parquet.  Embeddings are pre-L2-normalised 1536-dim float32
    arrays (normalised once at load time in the training script).

    Raw waveforms are returned (no mel conversion here) — the GPU MelTransform
    in the training loop handles this, consistent with Stage 2.
    """

    def __init__(
        self,
        meta_df: pd.DataFrame,
        embeddings: np.ndarray,
        audio_dir: Path,
        cfg: Config,
    ):
        self.meta       = meta_df.reset_index(drop=True)
        self.embeddings = embeddings          # (N, 1536), L2-normalised float32
        self.audio_dir  = audio_dir
        self.cfg        = cfg
        self.chunk_len  = cfg.sr * cfg.chunk_duration

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        row    = self.meta.iloc[idx]
        target = torch.tensor(self.embeddings[idx], dtype=torch.float32)

        filepath = self.audio_dir / row['filename']
        waveform = load_soundscape_chunk(
            filepath, int(row['start_sec']), self.cfg.sr, self.chunk_len)

        return waveform, target


# ── AI-specialist per-epoch sampler ──────────────────────────────────────────

class AISpecialistSampler(Sampler):
    """
    Per-epoch sampler for the ai_specialist stage when sc_pl_dir is provided.

    Each epoch produces two equal-sized draws that are shuffled together:

    1. Base draw  — weighted sampling with replacement from
       [focal_ai | sc_ai | xc_extra] using per-source weights [w_focal, w_sc,
       w_xc].  n_base_draws = int(n_focal*w_focal + n_sc*w_sc + n_xc*w_xc),
       reproducing the exp85 epoch composition.

    2. PL draw    — confidence-weighted sampling WITHOUT replacement from the
       PL pool (indices offset by n_base in the ConcatDataset).  Confidence
       weight per chunk = file-level weight: sum over classes of the per-class
       maximum prediction across all chunks in the file (1st-place 2025 recipe).
       Falls back to with-replacement if the pool is smaller than n_base_draws.

    Call set_epoch(ep) before each epoch to reseed the per-epoch RNG.
    Total epoch length = n_base_draws * 2.
    """

    def __init__(
        self,
        n_focal: int, w_focal: float,
        n_sc:    int, w_sc:    float,
        n_xc:    int, w_xc:    float,
        pl_conf_weights: np.ndarray,
        base_seed: int = 42,
    ):
        base_w = np.concatenate([
            np.full(n_focal, w_focal, dtype=np.float64),
            np.full(n_sc,    w_sc,    dtype=np.float64),
            np.full(n_xc,    w_xc,    dtype=np.float64),
        ])
        self._n_base       = n_focal + n_sc + n_xc
        self._base_probs   = base_w / base_w.sum()
        self._n_base_draws = max(1, int(base_w.sum()))

        w_pl = np.asarray(pl_conf_weights, dtype=np.float64)
        w_pl = np.maximum(w_pl, 1e-6)
        self._pl_probs = w_pl / w_pl.sum()
        self._n_pl     = len(w_pl)

        self._base_seed = base_seed
        self._ep        = 0

    def set_epoch(self, ep: int) -> None:
        self._ep = ep

    def __len__(self) -> int:
        return self._n_base_draws * 2

    def __iter__(self):
        rng = np.random.default_rng(self._base_seed + self._ep)

        base_idx = rng.choice(
            self._n_base, size=self._n_base_draws,
            replace=True, p=self._base_probs,
        ).tolist()

        n_pl_draws  = min(self._n_base_draws, self._n_pl)
        pl_local    = rng.choice(
            self._n_pl, size=n_pl_draws,
            replace=self._n_base_draws > self._n_pl,
            p=self._pl_probs,
        )
        pl_idx = (self._n_base + pl_local).tolist()

        combined = base_idx + pl_idx
        rng.shuffle(combined)
        return iter(combined)
