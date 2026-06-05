"""
GPU-side mel spectrogram conversion and augmentation.

Usage:
    mel_xf = MelTransform(cfg).to(device)

    # training forward pass
    specs = mel_xf(waves, augment=True)   # (B, 3, n_mels, T)

    # validation / KD distillation — no augment, no ImageNet norm
    specs = mel_xf(waves, augment=False)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from .config import Config


class FreqMixStyle(nn.Module):
    """
    Frequency-domain MixStyle for recording-condition generalisation.

    Mixes per-frequency-bin statistics (mean and std computed over the time axis)
    between randomly paired samples in the batch.  This simulates variation in
    microphone frequency response and environmental acoustics without altering
    the temporal structure of the signal.

    Applied to (B, 3, F, T) mel spectrograms during training only.
    The Beta distribution concentration parameter alpha controls mixing strength:
    lower alpha → lambda near 0 or 1 (stronger domain shift);
    higher alpha → lambda near 0.5 (mild blending).
    """

    def __init__(self, alpha: float = 0.3, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps   = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        if B < 2:
            return x

        # Per-sample per-freq-bin statistics over time → (B, 3, F, 1)
        mu  = x.mean(dim=3, keepdim=True)
        sig = (x.var(dim=3, keepdim=True) + self.eps).sqrt()

        # Instance-normalise
        x_norm = (x - mu) / sig

        # Sample mixing coefficient λ from Beta(alpha, alpha)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B,))
        lam = lam.to(x.device).view(B, 1, 1, 1)

        # Randomly pair each sample with another in the batch
        perm = torch.randperm(B, device=x.device)

        # Mix statistics
        mu_mix  = lam * mu  + (1.0 - lam) * mu[perm]
        sig_mix = lam * sig + (1.0 - lam) * sig[perm]

        return x_norm * sig_mix + mu_mix


class MelTransform(nn.Module):
    """
    Waveform → 3-channel mel-spectrogram.

    Processing chain:
      1. MelSpectrogram   (n_fft, hop_length, n_mels, fmin, fmax)
      2. AmplitudeToDB    (top_db=80)
      3. Per-sample min-max normalisation → [0, 1]
      4. Unsqueeze + repeat × 3  → (B, 3, n_mels, T)
      5. SpecAugment (FrequencyMasking + TimeMasking) — training only
      6. ImageNet normalisation — only when cfg.imagenet_norm is True
         (e.g. ViT-DINO backbones expect ImageNet-normalised inputs)

    Notes:
    - All transforms are nn.Modules registered as children so .to(device) moves
      their parameters correctly.
    - imagenet_norm constants are registered as buffers so they follow .to().
    - When augment=False (val / inference / KD stage), SpecAugment is skipped.
    """

    _IN_MEAN = [0.485, 0.456, 0.406]
    _IN_STD  = [0.229, 0.224, 0.225]

    def __init__(self, cfg: Config):
        super().__init__()
        self.imagenet_norm = cfg.imagenet_norm
        self.in_chans      = cfg.in_chans
        self._mel_w        = cfg.mel_w     # None = no resize
        self._n_mels       = cfg.n_mels

        self.mel_tf    = T.MelSpectrogram(
            sample_rate=cfg.sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
        )
        self.to_db     = T.AmplitudeToDB(top_db=80)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=cfg.freq_mask)
        self.time_mask = T.TimeMasking(time_mask_param=cfg.time_mask)

        self._apply_freq_mixstyle = cfg.freq_mixstyle
        if cfg.freq_mixstyle:
            self.freq_mixstyle_aug = FreqMixStyle(alpha=cfg.freq_mixstyle_alpha)

        if cfg.imagenet_norm:
            # (1, 3, 1, 1) — broadcast over (B, 3, H, W)
            self.register_buffer(
                'in_mean',
                torch.tensor(self._IN_MEAN).view(1, 3, 1, 1),
            )
            self.register_buffer(
                'in_std',
                torch.tensor(self._IN_STD).view(1, 3, 1, 1),
            )

    def forward(self, waves: torch.Tensor, augment: bool = False) -> torch.Tensor:
        """
        Args:
            waves:   (B, T_samples) float32 waveform tensor on the same device.
            augment: Apply SpecAugment when True (training only).

        Returns:
            (B, 3, n_mels, T_frames) float32 spectrogram tensor.
        """
        mel = self.mel_tf(waves)                              # (B, n_mels, T)
        mel = self.to_db(mel)

        # Per-sample [0, 1] normalisation
        mn = mel.flatten(1).min(dim=1).values[:, None, None]
        mx = mel.flatten(1).max(dim=1).values[:, None, None]
        mel = (mel - mn) / (mx - mn + 1e-6)

        mel = mel.unsqueeze(1).repeat(1, self.in_chans, 1, 1) # (B, C, n_mels, T)

        if self._mel_w is not None:
            mel = F.interpolate(
                mel, size=(self._n_mels, self._mel_w),
                mode='bilinear', align_corners=False,
            )

        if augment:
            if self._apply_freq_mixstyle:
                mel = self.freq_mixstyle_aug(mel)
            mel = self.freq_mask(mel)
            mel = self.time_mask(mel)

        if self.imagenet_norm:
            mel = (mel - self.in_mean) / self.in_std

        return mel


# ── Single-sample helpers (CPU, used inside val/focal Datasets) ───────────────

def mel_to_spectrogram_cpu(
    wave: torch.Tensor,
    cfg: Config,
    imagenet_norm: bool = False,
    in_chans: int = 3,
) -> torch.Tensor:
    """
    Convert a 1-D waveform to a (3, n_mels, T) mel spectrogram on CPU.
    Used in val/focal Dataset.__getitem__ where GPU transforms are unavailable.

    These transforms are created fresh each call — they are lightweight objects
    and this function is only called from Dataset workers.
    """
    mel_tf = T.MelSpectrogram(
        sample_rate=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        n_mels=cfg.n_mels, f_min=cfg.fmin, f_max=cfg.fmax,
    )
    to_db = T.AmplitudeToDB(top_db=80)

    mel = mel_tf(wave)
    mel = to_db(mel)
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)
    mel = mel.unsqueeze(0).repeat(in_chans, 1, 1)             # (C, n_mels, T)

    if cfg.mel_w is not None:
        mel = F.interpolate(
            mel.unsqueeze(0), size=(cfg.n_mels, cfg.mel_w),
            mode='bilinear', align_corners=False,
        ).squeeze(0)

    if imagenet_norm:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        mel = (mel - mean) / std

    return mel
