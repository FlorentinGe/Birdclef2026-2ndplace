"""
Config dataclass + YAML/CLI loading.

Loading order (later entries win):
  1. Dataclass defaults
  2. configs/backbone/<backbone>.yaml   (backbone-specific overrides)
  3. configs/stage/<stage>.yaml         (stage-specific overrides)
  4. --config <path>.yaml               (run-level overrides, optional)
  5. Explicit CLI flags
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────────
    base_dir: str = '/workspace/Birdclef/datasets/birdclef-2026'
    output_dir: str = '/workspace/Birdclef/experiments'
    # Override only if focal clips live outside base_dir/train_audio (default: None)
    # Set to the 4-shard WAV prefix to use the ttahara WAV dataset instead of OGG:
    #   /kaggle/input/datasets/ttahara/birdclef2026-train-audio-wav-
    train_audio_dir: Optional[str] = None
    # Legacy: kept for backward compat; ignored when train_audio_dir is set
    train_audio_wav_prefix: str = (
        '/kaggle/input/datasets/ttahara/birdclef2026-train-audio-wav-'
    )
    # Override only if soundscapes live elsewhere (default: base_dir/train_soundscapes)
    soundscape_dir: Optional[str] = None
    # KD stage: list of directories containing perch_train_arrays.npz + .parquet
    #perch_embed_dirs: List[str] = field(default_factory=list)
    perch_embed_dirs=[
        base_dir + '/perch embeddings/results 0-3500',
        base_dir + '/perch embeddings/results 3500-7000',
        base_dir + '/perch embeddings/results 7000:10658',
    ]
    # Pseudo-label stages
    focal_pl_csv: Optional[str] = None   # CSV with columns: file_path, label_vec_path
    sc_pl_csv: Optional[str] = None      # CSV with columns: filename, start_sec, label_vec_path
    sc_pl_dir: Optional[str] = None      # Dir containing sc_pl.csv + sc_pl_preds_ensemble.npy
                                         # (output of blend_sc_pl.py)
    # AI specialist stage (Amphibia + Insecta specialist model)
    ai_xc_species_csv: Optional[str] = None  # Path to birdclef2025_extra_species_data.csv
    ai_xc_species_dir: Optional[str] = None  # Path to birdclef2025_extra_species_data/data/
    use_xc_extra: bool = True                # Include XC extra focal data in ai_specialist
                                             # training.  Set False when backbone is already
                                             # XC-pretrained: XC clips carry label=0 for all
                                             # competition class indices, creating a ~33:1
                                             # negative-gradient ratio that overwhelms the
                                             # competition Insecta/Amphibia positive signal.

    # ── Audio ──────────────────────────────────────────────────────────────────
    sr: int = 32000
    duration: int = 20          # training clip length (seconds)
    chunk_duration: int = 5     # soundscape / inference chunk length (seconds)
    n_fft: int = 4096
    hop_length: int = 1252
    n_mels: int = 224
    fmin: int = 0
    fmax: int = 16000

    # ── Model ──────────────────────────────────────────────────────────────────
    backbone: str = 'hgnetv2_b4.ssld_stage2_ft_in1k'
    drop_path_rate: float = 0.15
    use_gem: bool = False        # GeM pooling over freq axis (replaces mean); learnable p
    gem_p_init: float = 3.0     # initial p value (1=mean, ∞=max); optimised during training
    # ViT-specific
    imagenet_norm: bool = False  # apply ImageNet mean/std after [0,1] mel normalisation
    use_llrd: bool = False       # layer-wise LR decay (ViT only)
    llrd_decay: float = 0.75     # per-layer multiplier; SED head=1.0×, patch_embed≈0.024×

    # ── Training (Stage 2 / supervised) ───────────────────────────────────────
    stage: str = 'supervised'
    batch_size: int = 32
    num_epochs: int = 20
    lr: float = 5e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    swa_start_epoch: int = 11
    grad_clip: float = 1.0
    seed: int = 42
    num_workers: int = 4
    use_amp: bool = True
    use_bf16: bool = False    # BF16 autocast instead of FP16 (Ampere/Ada GPUs only).

    # ── LR scheduler ──────────────────────────────────────────────────────────
    # 'cosine'    : linear warmup + single CosineAnnealingLR (default; all stages)
    # 'cosine_wr' : linear warmup + CosineAnnealingWarmRestarts (1st-place sc_pl recipe)
    #               Restarts every cosine_restart_period epochs with peak LR reset to cfg.lr.
    #               Use with lr=5e-4, cosine_restart_period=5, num_epochs=30 to match
    #               the 1st-place BirdCLEF 2025 self-training schedule.
    scheduler: str = 'cosine'
    cosine_restart_period: int = 5   # T_0 for CosineAnnealingWarmRestarts
                              # Eliminates ScaledStdConv NaN on NFNet — use_amp must also
                              # be True.  GradScaler is disabled automatically with BF16
                              # (BF16 has FP32 dynamic range so gradient scaling is
                              # unnecessary).

    # ── SpecAugment ────────────────────────────────────────────────────────────
    freq_mask: int = 30
    time_mask: int = 80   # scaled to 20 for ViT (5s → 128 time frames)

    # ── FreqMixStyle ───────────────────────────────────────────────────────────
    freq_mixstyle: bool = False      # mix per-freq-bin statistics between batch samples
    freq_mixstyle_alpha: float = 0.3 # Beta dist concentration; lower = stronger mixing

    # ── Mixup ──────────────────────────────────────────────────────────────────
    mixup_prob: float = 0.5
    mixup_min_overlap: float = 0.0   # min fraction of T_samples that must be non-zero in
                                     # both samples; 0.0=disabled (always mix). With left
                                     # padding this filters out pairs where audio regions
                                     # don't overlap at all (e.g. 0.1 = 10% of 640k frames).
    focal_pad_type: str = "left"     # padding side for short focal clips in BirdDataset:
                                     #   "left"   — silence at left, audio at right; MixUp pairs
                                     #              always overlap in the audio region (1st-place recipe)
                                     #   "random" — audio at a uniformly random position (no fixed shortcut)
                                     #   "right"  — audio at left, silence at right (exp53 behaviour)

    # ── Validation ─────────────────────────────────────────────────────────────
    focal_val_frac: float = 0.1
    soundscape_val_frac: float = 0.15

    # ── KD stage (Stage 1) ────────────────────────────────────────────────────
    pretrain_epochs: int = 7
    pretrain_lr: float = 5e-4
    pretrain_batch_size: int = 64
    pretrain_warmup: int = 1
    perch_emb_dim: int = 1536
    proj_mlp: bool = True         # MLP projection head (True) vs linear (False)
                                  # MLP gives backbone more representational freedom;
                                  # linear forces the backbone itself to encode Perch2 space

    # ── Stage 2 staged unfreeze ───────────────────────────────────────────────
    freeze_epochs: int = 0        # freeze backbone for first N Stage 2 epochs (0=disabled)
                                  # recommended: 2–3 when loading a KD checkpoint
    unfreeze_lr: float = 0.0      # backbone LR after unfreeze (0.0 = use cfg.lr)

    # ── Rare-species upsampling ────────────────────────────────────────────────
    rare_upsample_thresh: int = 0     # 0 = disabled; upsample focal clips for species
                                      # with fewer than this many training clips
    rare_upsample_cap: float = 5.0    # maximum per-sample weight multiplier for rare species
                                      # e.g. thresh=20, cap=5 → a species with 4 clips gets
                                      # weight min(5, 20/4) = 5.0×
    sc_pl_upsample_thresh: int = 0    # 0 = disabled; upsample sc_pl chunks whose argmax
                                      # species has fewer than this many chunks in the manifest.
                                      # Mirrors rare_upsample_thresh for SC pseudo-label data.
                                      # Addresses batch starvation in SoftAUCLoss: species
                                      # with few SC chunks rarely appear as positives in a
                                      # batch, producing near-zero ranking gradient.
                                      # Applied to chunk_weights (MixUp) and WeightedRandomSampler
                                      # sc_only_pl_ds weights (substitution path).
    sc_pl_upsample_cap: float = 5.0   # max weight multiplier for rare SC PL chunks

    # ── Input channels ────────────────────────────────────────────────────────
    in_chans: int = 3              # mel spectrogram channels fed to backbone
                                    # 3 = standard RGB-style (all CNN/ViT backbones)
                                    # 1 = PaSST (native single-channel audio spectrograms)

    # ── Focal loss ────────────────────────────────────────────────────────────
    use_focal_loss: bool = False    # replace BCEWithLogitsLoss with sigmoid focal BCE
    focal_gamma: float = 2.0       # modulating exponent; 0 = plain BCE; 2 = standard focal
    focal_alpha: float = 0.25      # per-class pos weight; 0.25 = RetinaNet default;
                                    # 0.5 = symmetric (equivalent to no alpha weighting)

    # ── CE loss ───────────────────────────────────────────────────────────────
    # Soft cross-entropy: normalises the multi-hot label vector to a probability
    # distribution, then applies -sum(p * log_softmax(logit)).
    # Creates inter-class competition via softmax (unlike BCE which treats each
    # class independently).  At inference sigmoid is still used — the
    # train/infer asymmetry forces the attention head to find time steps where
    # one species dominates rather than falling back on positional shortcuts.
    # Salman Ahmed (33rd place, 0.937 single model, BirdCLEF+ 2026 discussion).
    use_ce_loss: bool = False

    # ── Soft AUC loss ─────────────────────────────────────────────────────────
    # Directly optimises ROC-AUC via a pairwise ranking loss.
    # BirdCLEF 2025 4th place (Dylan Liu): +0.051 LB on single tf_efficientnetv2_b0.
    # Takes priority over use_focal_loss when both are True.
    # Note: CV score will be *lower* than BCE/Focal — this is expected, not a bug.
    use_soft_auc_loss: bool = False
    soft_auc_margin: float = 1.0    # margin scaling in softplus(-diff * margin)
    soft_auc_bce_weight: float = 0.0  # auxiliary BCE calibration term weight (default 0.0).
                                       # Set to ~0.1 to prevent near-zero collapse for species
                                       # with sparse sc_pl positives (Insecta sonotypes, Mammalia).

    # ── Logit-space soft AUC loss ─────────────────────────────────────────────
    # Identical to BirdCLEF 2025 4th place (Dylan Liu) — logits fed directly
    # without sigmoid.  Loss → 0 for well-separated pairs (no probability floor).
    # Takes priority over use_soft_auc_loss and use_focal_loss when True.
    use_logit_auc_loss: bool = False
    logit_auc_margin: float = 1.0
    logit_auc_pos_weight: float = 1.0
    logit_auc_neg_weight: float = 1.0
    logit_auc_bce_weight: float = 0.0  # auxiliary BCE calibration term; same batch-starvation
                                        # rationale as soft_auc_bce_weight — set ~0.1 for sc_pl

    # ── Smooth-AP loss ────────────────────────────────────────────────────────
    # Differentiable approximation to Average Precision (Brown et al., ECCV 2020).
    # Optimises the PR curve (AP) rather than the ROC curve (AUC). Position-weighted
    # pairwise loss — high-confidence positives ranked first contribute more than
    # lower-ranked ones, unlike SoftAUCLoss's uniform pairwise gradient.
    # Takes priority over all other loss flags when True.
    # Memory: O(B² × C) — at B=32, C=234 uses ~9.7 MB; reduce batch_size if OOM.
    use_smooth_ap_loss: bool = False
    smooth_ap_tau: float = 0.01        # sigmoid temperature for rank smoothing; use >=0.1 with BF16
    smooth_ap_bce_weight: float = 0.0  # auxiliary BCE term weight; 0.1 recommended to prevent scale collapse

    # ── Background augmentation ───────────────────────────────────────────────
    bg_aug_prob: float = 0.0        # per-sample probability of mixing in a bg segment; 0=disabled
    bg_aug_snr_min_db: float = 6.0  # minimum SNR of background relative to foreground (dB)
                                    # 6 dB = bg is 2× quieter than fg in RMS; safe lower bound
    bg_aug_snr_max_db: float = 24.0 # maximum SNR; uniformly sampled in [min, max] each call
                                    # 24 dB = bg is ~16× quieter; barely audible
    noise_dir: Optional[str] = None # Path to external noise directory (e.g. ESC-50 filtered).
                                    # When set, ExternalNoiseAugmenter is used instead of
                                    # NocallAugmenter.  The dir must contain *.wav or *.ogg files.
    noise_padding: bool = False     # Replace zero-padding of short clips with low-amplitude noise
                                    # from noise_dir (requires noise_dir to be set).
                                    # Fixes the silent-pad attention shortcut without touching the
                                    # actual bird-call signal.  Amplitude drawn per sample from
                                    # [0.003, 0.015] so no fixed level becomes a padding cue.
                                    # Never applied to validation or inference.
    sc_use_window: bool = True      # When True (default): SoundscapeTrainDataset and
                                    # SoundscapePLDataset load a DURATION-second window from the
                                    # actual soundscape file with the labeled CHUNK_DURATION chunk
                                    # placed at a random offset, using real surrounding audio.
                                    # When False: loads the CHUNK_DURATION chunk and zero-pads to
                                    # DURATION with a random offset (exp53 behaviour).

    # ── Dual loss ─────────────────────────────────────────────────────────────
    dual_loss_weight: float = 0.0   # weight for frame-max BCE loss alongside clip loss;
                                    # 0.0 = disabled; 0.5 = equal weighting (clip + 0.5*frame_max)

    # ── Augmentation consistency regularisation ───────────────────────────────
    cons_weight: float = 0.0        # weight for augmentation consistency loss; 0.0 = disabled.
                                    # Runs a second forward pass with a fresh SpecAugment draw on
                                    # the same waveforms and penalises MSE between the two sigmoid
                                    # output vectors (stop-gradient on the first view).
                                    # Pushes the model toward augmentation-invariant representations
                                    # and is structurally orthogonal to all label-based losses.
                                    # Adds one model forward pass per batch — reduce batch_size by
                                    # ~15% when enabling to stay within GPU memory budget.
                                    # Recommended starting value: 0.1.

    # ── Attention regularisation ──────────────────────────────────────────────
    att_entropy_weight: float = 0.0  # penalise flat/uniform attention; loss += w*(H/H_max).mean()
                                     # where H=-sum(att*log(att)) over T, H_max=log(T).
                                     # Positive values push the model toward focused peaks.
    att_edge_weight: float = 0.0     # penalise attention mass at t=0 and t=last (zero-pad edges);
                                     # loss += w*(att[:,0,:]+att[:,-1,:]).mean(). Forces the model
                                     # to ignore padding boundaries as discriminative cues.

    # ── Pseudo-label stages ────────────────────────────────────────────────────
    # Power-transform + blend formula (5th place recipe):
    #   pseudo_t = clip(pseudo*(pseudo>th) + pseudo**power, 0, 1)
    #   label    = alpha*pseudo_t + (1-alpha)*original_hard_labels
    # Then optionally: label = max(label, perch_soft * perch_max)
    pl_pseudo_th: float    = 0.3   # threshold: entries below this are zeroed before blend
    pl_pseudo_power: float = 2.0   # exponent added to all entries (concentrates mass)
    pl_pseudo_alpha: float = 0.7   # pseudo-label weight in blend; 1-alpha = original weight
    pl_perch_max: float    = 0.0   # cap for Perch continuous discovery; 0.0 = Perch disabled
                                   # e.g. 0.5 → Perch at z=4 can push a label up to 0.5
    # Soundscape PL stage params (1st-place Noisy Student recipe).
    # Formula: label = prob ** power  (no threshold, no alpha blend).
    # Tune power per iteration on LB:
    #   Round 1: 1.0 (identity — raw ensemble probs)
    #   Round 2: 1/0.65 ≈ 1.54
    #   Round 3: 1/0.55 ≈ 1.82
    #   Round 4: 1/0.60 ≈ 1.67
    pl_sc_pseudo_power: float = 1.0
    pl_sc_clip_th: Optional[float] = None  # Hard-cap SC PL labels >= this value to 1.0 before
                                            # the power transform. Brings high-confidence entries
                                            # to pos_weight=0.5 (matching focal clips) so that
                                            # SoftAUCLoss treats them as clean positives.
                                            # Recommended value: 0.7 (round 3 data).
    sc_chunk_level_weights: bool = False  # When True, weight sc_pl chunks by their own
                                          # max-class probability instead of file-level
                                          # aggregate. Directly promotes high-confidence
                                          # individual chunks (ablation vs 1st-place recipe).
    sc_pl_exclude_labelled: bool = False  # When True, remove all labeled soundscape files
                                          # from sc_pl training (both sc_train and sc_val
                                          # splits). Prevents pseudo-label leakage where
                                          # accurate PLs for labeled chunks inflate val AUC.
    sc_pl_hard_labels: bool = False       # When True, override pseudo-labels with hard 1.0
                                          # targets for (filename, start_sec, species) triples
                                          # found in train_soundscapes_labels.csv.  Applies
                                          # only to the sc_pl training set; pseudo-labels for
                                          # unlabeled species in those chunks are unchanged.
    sc_pl_sub_prob: float = 0.0           # Species-matched substitution probability
                                          # (Ali Ozan Memetoglu, 2nd place BirdCLEF 2025):
                                          # with this probability, a focal XC clip is
                                          # replaced by a soundscape pseudo chunk whose
                                          # top predicted species matches the focal clip's
                                          # primary label. 0.0 = use MixUp (default).

    # ── Image resize ──────────────────────────────────────────────────────────
    # When set, MelTransform resizes every mel to (n_mels, mel_w) before the
    # model. Use this to normalise extreme aspect ratios from low hop_length
    # values (e.g. hop_length=64 → raw T≈10000 for 20 s; set mel_w=512).
    # None = no resize (raw T from hop_length).
    mel_w: Optional[int] = None

    # ── Stem stride ───────────────────────────────────────────────────────────
    # Controls the stride of the first stride-2 conv in the backbone stem.
    # Default 2 = standard 32× total spatial downsampling (original behaviour).
    # Set to 1 to halve total downsampling to 16×, doubling the spatial
    # dimensions of every feature map (e.g. [8,16] → [16,32] for [256,512]
    # input with EfficientNet-B0).  Pretrained weights transfer without any
    # re-initialisation — Conv2d stride is runtime behaviour, not stored in
    # the weight tensor.
    # Supported: EfficientNet/V2 (conv_stem), ECA-NFNet (stem.conv4),
    #            RegNetY/X (stem.conv).  ViT/PaSST: no effect (patch size
    #            governs their spatial resolution instead).
    stem_stride: int = 2

    # ── Derived (read-only helpers, not serialised) ────────────────────────────
    @property
    def img_size(self):
        """(n_mels, time_frames) for the training duration."""
        if self.mel_w is not None:
            return (self.n_mels, self.mel_w)
        t = math.floor(self.sr * self.duration / self.hop_length) + 1
        return (self.n_mels, t)

    @property
    def img_size_chunk(self):
        """(n_mels, time_frames) for one inference/val chunk."""
        if self.mel_w is not None:
            return (self.n_mels, self.mel_w)
        t = math.floor(self.sr * self.chunk_duration / self.hop_length) + 1
        return (self.n_mels, t)

    @property
    def is_vit(self) -> bool:
        return self.backbone.startswith('vit_')

    def train_audio_path(self) -> Path:
        """
        Path to the directory containing focal training audio files.
        Defaults to base_dir/train_audio (OGG originals).
        Set train_audio_dir to use the converted WAV shards instead.
        """
        if self.train_audio_dir:
            return Path(self.train_audio_dir)
        return Path(self.base_dir) / 'train_audio'

    def soundscape_path(self) -> Path:
        if self.soundscape_dir:
            return Path(self.soundscape_dir)
        return Path(self.base_dir) / 'train_soundscapes'

    def to_dict(self):
        d = asdict(self)
        d['img_size'] = list(self.img_size)
        return d

    def save_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ── YAML helpers ───────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    if not _HAS_YAML:
        raise ImportError('pyyaml is required for config loading: pip install pyyaml')
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _apply_dict(cfg: Config, overrides: dict) -> None:
    """Apply a dict of overrides to cfg in place."""
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            raise ValueError(f'Unknown config key: {k!r}')


# ── CLI + config loading entry point ──────────────────────────────────────────

_CONFIGS_DIR = Path(__file__).parent.parent / 'configs'


def load_config(argv=None) -> Config:
    """
    Parse CLI arguments and return a fully-resolved Config.

    Usage examples:
        python train.py --backbone eca_nfnet_l0 --stage kd
        python train.py --backbone vit_base_patch16_224.dino --stage supervised \\
                        --num_epochs 40 --output_dir /data/runs/exp23
        python train.py --backbone eca_nfnet_l0 --stage kd \\
                        --config configs/local_paths.yaml
    """
    parser = argparse.ArgumentParser(description='BirdCLEF+ 2026 Training')

    # ── Required / main ───────────────────────────────────────────────────────
    parser.add_argument('--backbone', required=True,
                        help='timm model name; must match a configs/backbone/*.yaml')
    parser.add_argument('--stage', default='supervised',
                        choices=['supervised', 'kd', 'focal_pl', 'sc_pl', 'ai_specialist'],
                        help='Training stage')
    parser.add_argument('--config', default=None,
                        help='Additional YAML file to merge (run-level overrides)')

    # ── Paths ─────────────────────────────────────────────────────────────────
    parser.add_argument('--base_dir',         default=None)
    parser.add_argument('--output_dir',       default=None)
    parser.add_argument('--train_audio_dir',  default=None,
                        help='Override focal audio directory (default: base_dir/train_audio)')
    parser.add_argument('--train_audio_wav_prefix', default=None,
                        help='Legacy: WAV shard prefix; ignored when train_audio_dir is set')
    parser.add_argument('--soundscape_dir',   default=None)
    parser.add_argument('--perch_embed_dirs', nargs='+', default=None,
                        help='One or more directories with perch embeddings (kd stage)')
    parser.add_argument('--focal_pl_csv', default=None)
    parser.add_argument('--sc_pl_csv',    default=None)
    parser.add_argument('--sc_pl_dir',    default=None,
                        help='Directory containing sc_pl.csv and sc_pl_preds_ensemble.npy '
                             '(output of blend_sc_pl.py). Required for stage=sc_pl. '
                             'Also used by stage=ai_specialist to locate the unlabeled SC '
                             'holdout manifest for OOF generation (sc_pl_val_files.txt).')
    parser.add_argument('--use_xc_extra', type=lambda x: x.lower() != 'false', default=None,
                        help='Include XC extra focal data in ai_specialist training (default True). '
                             'Set false when using an XC-pretrained backbone: XC clips have '
                             'label=0 for all competition class indices, producing a ~33:1 '
                             'negative-gradient ratio that collapses competition Insecta signal.')
    parser.add_argument('--ai_xc_species_csv', default=None,
                        help='Path to birdclef2025_extra_species_data.csv '
                             '(required for stage=ai_specialist)')
    parser.add_argument('--ai_xc_species_dir', default=None,
                        help='Path to birdclef2025_extra_species_data/data/ directory '
                             'containing OGG files (required for stage=ai_specialist)')

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument('--num_epochs',       type=int,   default=None)
    parser.add_argument('--batch_size',       type=int,   default=None)
    parser.add_argument('--lr',               type=float, default=None)
    parser.add_argument('--weight_decay',     type=float, default=None)
    parser.add_argument('--warmup_epochs',    type=int,   default=None)
    parser.add_argument('--swa_start_epoch',  type=int,   default=None)
    parser.add_argument('--grad_clip',        type=float, default=None)
    parser.add_argument('--seed',             type=int,   default=None)
    parser.add_argument('--num_workers',      type=int,   default=None)
    parser.add_argument('--use_amp',          type=lambda x: x.lower() != 'false',
                        default=None)
    parser.add_argument('--use_bf16',         type=lambda x: x.lower() != 'false',
                        default=None,
                        help='BF16 autocast (Ampere/Ada only). Requires --use_amp true. '
                             'Fixes NFNet NaN without FP32 fallback.')

    # ── KD stage ──────────────────────────────────────────────────────────────
    parser.add_argument('--pretrain_epochs',      type=int,   default=None)
    parser.add_argument('--pretrain_lr',          type=float, default=None)
    parser.add_argument('--pretrain_batch_size',  type=int,   default=None)
    parser.add_argument('--proj_mlp',  type=lambda x: x.lower() != 'false', default=None,
                        help='MLP projection head for KD (default True); pass false to use linear')

    # ── Staged unfreeze ───────────────────────────────────────────────────────
    parser.add_argument('--freeze_epochs', type=int,   default=None,
                        help='Freeze backbone for first N Stage 2 epochs (recommended: 2–3 with KD checkpoint)')
    parser.add_argument('--unfreeze_lr',   type=float, default=None,
                        help='Backbone LR after unfreeze (default: cfg.lr)')

    # ── Rare-species upsampling ───────────────────────────────────────────────
    parser.add_argument('--rare_upsample_thresh', type=int,   default=None,
                        help='Upsample focal clips for species with fewer than N clips (0=off)')
    parser.add_argument('--rare_upsample_cap',    type=float, default=None,
                        help='Max weight multiplier for rare species (default 5.0)')
    parser.add_argument('--sc_pl_upsample_thresh', type=int, default=None,
                        help='Upsample sc_pl chunks whose argmax species has fewer than N '
                             'chunks in the manifest (0=off). Mirrors rare_upsample_thresh '
                             'for SC pseudo-label data; fixes SoftAUC batch starvation for '
                             'rare non-Aves species.')
    parser.add_argument('--sc_pl_upsample_cap', type=float, default=None,
                        help='Max weight multiplier for rare SC PL chunks (default 5.0)')

    # ── Normalisation / LLRD ─────────────────────────────────────────────────
    parser.add_argument('--imagenet_norm', type=lambda x: x.lower() != 'false', default=None,
                        help='Apply ImageNet mean/std after [0,1] mel normalisation (default False). '
                             'Enable for CLIP-pretrained backbones (e.g. convnext_base.clip_*).')
    parser.add_argument('--use_llrd', type=lambda x: x.lower() != 'false', default=None,
                        help='Layer-wise LR decay (ViT, ConvNext, EfficientNet; default False)')
    parser.add_argument('--llrd_decay', type=float, default=None,
                        help='Per-layer LR decay multiplier for LLRD (default 0.75). '
                             'Lower = steeper decay; 0.65 protects early layers more aggressively.')

    # ── GeM pooling ───────────────────────────────────────────────────────────
    parser.add_argument('--use_gem',    type=lambda x: x.lower() != 'false', default=None,
                        help='GeM pooling over freq axis (replaces mean); default False')
    parser.add_argument('--gem_p_init', type=float, default=None,
                        help='Initial p value for GeM (default 3.0)')

    # ── Input channels ────────────────────────────────────────────────────────
    parser.add_argument('--in_chans', type=int, default=None,
                        help='Mel spectrogram channels (3=standard, 1=PaSST)')

    # ── Focal loss ────────────────────────────────────────────────────────────
    parser.add_argument('--use_focal_loss', type=lambda x: x.lower() != 'false', default=None,
                        help='Use sigmoid focal BCE instead of plain BCE (default False)')
    parser.add_argument('--focal_gamma',    type=float, default=None,
                        help='Focal modulating exponent (default 2.0)')
    parser.add_argument('--focal_alpha',    type=float, default=None,
                        help='Per-class pos weight for focal loss (default 0.25; 0.5=no weighting)')

    # ── CE loss ───────────────────────────────────────────────────────────────
    parser.add_argument('--use_ce_loss', type=lambda x: x.lower() != 'false', default=None,
                        help='Soft cross-entropy loss — inter-class competition via softmax '
                             'at train time, sigmoid at inference (default False)')

    # ── Soft AUC loss ─────────────────────────────────────────────────────────
    parser.add_argument('--use_soft_auc_loss', type=lambda x: x.lower() != 'false', default=None,
                        help='Use SoftAUCLoss (pairwise ranking, directly optimises AUC). '
                             'Takes priority over --use_focal_loss. CV score will be lower '
                             'than BCE/Focal — expected behaviour, not a bug. (default False)')
    parser.add_argument('--soft_auc_margin', type=float, default=None,
                        help='Margin scaling in SoftAUCLoss softplus(-diff * margin) (default 1.0)')
    parser.add_argument('--soft_auc_bce_weight', type=float, default=None,
                        help='Weight of auxiliary BCE calibration term added to SoftAUC loss '
                             '(default 0.0). Set to ~0.1 to prevent near-zero prediction collapse '
                             'for species with sparse sc_pl positives (Insecta sonotypes, Mammalia). '
                             'Does not meaningfully reduce ensemble diversity (Spearman shift <0.03).')

    # ── Logit-space soft AUC loss ─────────────────────────────────────────────
    parser.add_argument('--use_logit_auc_loss', type=lambda x: x.lower() != 'false', default=None,
                        help='Use LogitSoftAUCLoss (4th-place 2025 implementation, no sigmoid). '
                             'Takes priority over use_soft_auc_loss. (default False)')
    parser.add_argument('--logit_auc_margin', type=float, default=None,
                        help='Margin scaling in LogitSoftAUCLoss log(1+exp(-diff*margin)) (default 1.0)')
    parser.add_argument('--logit_auc_pos_weight', type=float, default=None,
                        help='Global multiplier on positive soft weights (default 1.0)')
    parser.add_argument('--logit_auc_neg_weight', type=float, default=None,
                        help='Global multiplier on negative soft weights (default 1.0)')
    parser.add_argument('--logit_auc_bce_weight', type=float, default=None,
                        help='Weight of auxiliary BCE calibration term for LogitSoftAUCLoss '
                             '(default 0.0). Set ~0.1 for sc_pl stage to prevent logit collapse '
                             'for rare Insecta/Amphibia/Mammalia species with sparse sc_pl chunks.')

    # ── Smooth-AP loss ────────────────────────────────────────────────────────
    parser.add_argument('--use_smooth_ap_loss', type=lambda x: x.lower() != 'false', default=None,
                        help='Use SmoothAPLoss (differentiable AP, optimises PR curve). '
                             'Takes priority over all other loss flags. Position-weighted '
                             'pairwise loss — structurally different from SoftAUCLoss. '
                             'Memory: ~9.7 MB at B=32, C=234; reduce batch_size if OOM. (default False)')
    parser.add_argument('--smooth_ap_tau', type=float, default=None,
                        help='Sigmoid temperature for SmoothAP rank smoothing (default 0.01; '
                             'smaller = sharper rank approximation; use >=0.1 with BF16)')
    parser.add_argument('--smooth_ap_bce_weight', type=float, default=None,
                        help='Weight of auxiliary BCE calibration term added to SmoothAP loss '
                             '(default 0.0). Set to ~0.1 to prevent prediction scale collapse '
                             'when training with SmoothAP alone.')

    # ── Background augmentation ───────────────────────────────────────────────
    parser.add_argument('--bg_aug_prob',       type=float, default=None,
                        help='Per-sample prob of background aug (0=disabled)')
    parser.add_argument('--bg_aug_snr_min_db', type=float, default=None,
                        help='Min background SNR in dB (default 6.0)')
    parser.add_argument('--bg_aug_snr_max_db', type=float, default=None,
                        help='Max background SNR in dB (default 24.0)')
    parser.add_argument('--noise_dir',         default=None,
                        help='Directory of external noise files (*.wav/*.ogg). '
                             'When set, ExternalNoiseAugmenter is used instead of '
                             'NocallAugmenter (competition nocall segments).')
    parser.add_argument('--noise_padding',    type=lambda x: x.lower() != 'false',
                        default=None,
                        help='Replace zero-padding with low-amplitude noise from noise_dir '
                             '(default False). Requires --noise_dir. '
                             'Fixes silent-pad attention shortcuts without touching real signal.')
    parser.add_argument('--focal_pad_type',  default=None,
                        choices=['left', 'random', 'right'],
                        help='Padding side for short focal clips in BirdDataset: '
                             '"left" (default, 1st-place recipe; MixUp always overlaps), '
                             '"random" (no fixed-position shortcut), '
                             '"right" (exp53 legacy behaviour).')
    parser.add_argument('--sc_use_window',   type=lambda x: x.lower() != 'false',
                        default=None,
                        help='Load a real DURATION-second window around each soundscape chunk '
                             'instead of zero-padding (default True). '
                             'Set false to reproduce exp53 zero-pad behaviour.')

    # ── Dual loss ─────────────────────────────────────────────────────────────
    parser.add_argument('--dual_loss_weight', type=float, default=None,
                        help='Weight for frame-max BCE alongside clip BCE (0=off, 0.5=equal)')

    # ── Augmentation consistency regularisation ───────────────────────────────
    parser.add_argument('--cons_weight', type=float, default=None,
                        help='Weight for augmentation consistency loss (0=off, recommended 0.1). '
                             'Runs a second forward pass with a fresh SpecAugment draw and '
                             'penalises MSE between sigmoid outputs (stop-grad on first view). '
                             'Adds one forward pass per batch — reduce batch_size ~15%% to '
                             'stay within GPU memory budget.')

    # ── Attention regularisation ──────────────────────────────────────────────
    parser.add_argument('--att_entropy_weight', type=float, default=None,
                        help='Penalise flat attention H/H_max; 0=disabled (default 0.0)')
    parser.add_argument('--att_edge_weight',    type=float, default=None,
                        help='Penalise attention at t=0 and t=last; 0=disabled (default 0.0)')
    parser.add_argument('--mixup_min_overlap',  type=float, default=None,
                        help='Min audio overlap fraction for MixUp pairs; 0=disabled (default 0.0)')

    # ── FreqMixStyle ──────────────────────────────────────────────────────────
    parser.add_argument('--freq_mixstyle',       type=lambda x: x.lower() != 'false',
                        default=None, help='Freq-MixStyle augmentation (default False)')
    parser.add_argument('--freq_mixstyle_alpha', type=float, default=None,
                        help='Beta dist alpha for FreqMixStyle (default 0.3)')

    # ── Audio ─────────────────────────────────────────────────────────────────
    parser.add_argument('--time_mask', type=int, default=None,
                        help='SpecAugment time mask width in frames (default 80). '
                             'Scale proportionally when reducing duration: '
                             'e.g. 40 for 5s+hop=512, 20 for 5s+hop=1252.')
    parser.add_argument('--duration', type=int, default=None,
                        help='Training clip length in seconds (default 20). '
                             'Use 5 or 10 with low hop_length to keep the '
                             'spectrogram aspect ratio manageable and eliminate '
                             'the train/inference temporal gap.')
    parser.add_argument('--hop_length', type=int, default=None,
                        help='STFT hop length in samples (default 1252; try 320–512 '
                             'per 4th-place 2025 finding). Affects img_size and '
                             'img_size_chunk — change together with n_fft if needed.')
    parser.add_argument('--n_fft', type=int, default=None,
                        help='FFT window size in samples (default 4096). '
                             '4th-place 2025 used 2048 across all variants.')
    parser.add_argument('--n_mels', type=int, default=None,
                        help='Number of mel filterbank bins (default 224). '
                             '4th-place 2025 used 256 (v1) or 192 (v2).')
    parser.add_argument('--fmin', type=int, default=None,
                        help='Minimum frequency for mel filterbank in Hz (default 0). '
                             '4th-place 2025 used 40–60 Hz to suppress low-freq noise.')
    parser.add_argument('--fmax', type=int, default=None,
                        help='Maximum frequency for mel filterbank in Hz (default 16000).')
    parser.add_argument('--mel_w', type=int, default=None,
                        help='Resize mel time axis to this width after computation '
                             '(default None = no resize). Use to normalise extreme '
                             'aspect ratios, e.g. --mel_w 512 with --hop_length 64.')
    parser.add_argument('--stem_stride', type=int, default=None, choices=[1, 2],
                        help='Stem conv stride (default 2 = standard 32× downsampling). '
                             'Set to 1 to halve total downsampling (16×), doubling the '
                             'spatial feature map in both dims (e.g. [8,16]→[16,32] for '
                             '[256,512] input). Weights transfer cleanly. Supported: '
                             'EfficientNet/V2, ECA-NFNet, RegNetY/X.')
    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument('--scheduler', default=None,
                        choices=['cosine', 'cosine_wr'],
                        help="LR scheduler: 'cosine' (default, single cycle) or "
                             "'cosine_wr' (warm restarts, 1st-place sc_pl recipe)")
    parser.add_argument('--cosine_restart_period', type=int, default=None,
                        help='Restart period T_0 for cosine_wr (default 5)')
    parser.add_argument('--drop_path_rate',   type=float, default=None)
    parser.add_argument('--focal_val_frac',   type=float, default=None)
    parser.add_argument('--pl_pseudo_th',     type=float, default=None,
                        help='PL power-transform threshold (default 0.3)')
    parser.add_argument('--pl_pseudo_power',  type=float, default=None,
                        help='PL power exponent (default 2.0)')
    parser.add_argument('--pl_pseudo_alpha',  type=float, default=None,
                        help='PL blend weight for pseudo labels (default 0.7)')
    parser.add_argument('--pl_perch_max',     type=float, default=None,
                        help='Perch discovery cap; 0 = disabled (default 0.0)')
    parser.add_argument('--pl_sc_pseudo_power', type=float, default=None,
                        help='Soundscape PL power exponent (default 1.0 for round 1; '
                             'set to ~1.54/1.82/1.67 for rounds 2/3/4)')
    parser.add_argument('--pl_sc_clip_th', type=float, default=None,
                        help='Hard-cap SC PL labels >= this value to 1.0 before the power '
                             'transform (recommended 0.7 for round 3). Equalises SoftAUCLoss '
                             'pos_weight between high-confidence SC chunks and focal clips. '
                             'No-op when omitted.')
    parser.add_argument('--sc_chunk_level_weights',
                        type=lambda x: x.lower() != 'false', default=None,
                        help='Weight sc_pl chunks by per-chunk max probability instead of '
                             'file-level aggregate (default False = 1st-place file-level recipe)')
    parser.add_argument('--sc_pl_exclude_labelled',
                        type=lambda x: x.lower() != 'false', default=None,
                        help='Exclude all labeled soundscape files from sc_pl training data. '
                             'Prevents pseudo-label leakage where accurate PLs for labeled '
                             'chunks inflate the labeled-SC validation AUC. (default False)')
    parser.add_argument('--sc_pl_hard_labels',
                        type=lambda x: x.lower() != 'false', default=None,
                        help='Override pseudo-labels with hard 1.0 targets for any '
                             '(filename, start_sec, species) triple in '
                             'train_soundscapes_labels.csv.  All 75 species with ground-truth '
                             'labels in labeled soundscapes benefit; pseudo-labels for '
                             'co-occurring unlabeled species are unchanged. (default False)')
    parser.add_argument('--sc_pl_sub_prob', type=float, default=None,
                        help='Species-matched substitution probability (Ali Ozan, 2nd place): '
                             'with this probability a focal XC clip is replaced by a soundscape '
                             'pseudo chunk whose top predicted species matches the focal primary '
                             'label. 0.0 = MixUp (default).')

    args = parser.parse_args(argv)

    # ── Build config ──────────────────────────────────────────────────────────
    cfg = Config()
    cfg.backbone = args.backbone
    cfg.stage    = args.stage

    # 1. Backbone YAML
    backbone_yaml = _CONFIGS_DIR / 'backbone' / f'{args.backbone}.yaml'
    if backbone_yaml.exists():
        _apply_dict(cfg, _load_yaml(backbone_yaml))

    # 2. Stage YAML
    stage_yaml = _CONFIGS_DIR / 'stage' / f'{args.stage}.yaml'
    if stage_yaml.exists():
        _apply_dict(cfg, _load_yaml(stage_yaml))

    # 3. Run-level YAML override
    if args.config:
        _apply_dict(cfg, _load_yaml(Path(args.config)))

    # 4. Explicit CLI overrides (only non-None values)
    cli_overrides = {
        'base_dir':               args.base_dir,
        'output_dir':             args.output_dir,
        'train_audio_dir':        args.train_audio_dir,
        'train_audio_wav_prefix': args.train_audio_wav_prefix,
        'soundscape_dir':         args.soundscape_dir,
        'perch_embed_dirs':       args.perch_embed_dirs,
        'focal_pl_csv':           args.focal_pl_csv,
        'sc_pl_csv':              args.sc_pl_csv,
        'sc_pl_dir':              args.sc_pl_dir,
        'use_xc_extra':           args.use_xc_extra,
        'ai_xc_species_csv':      args.ai_xc_species_csv,
        'ai_xc_species_dir':      args.ai_xc_species_dir,
        'num_epochs':             args.num_epochs,
        'batch_size':             args.batch_size,
        'lr':                     args.lr,
        'weight_decay':           args.weight_decay,
        'warmup_epochs':          args.warmup_epochs,
        'swa_start_epoch':        args.swa_start_epoch,
        'grad_clip':              args.grad_clip,
        'seed':                   args.seed,
        'num_workers':            args.num_workers,
        'use_amp':                args.use_amp,
        'use_bf16':               args.use_bf16,
        'pretrain_epochs':        args.pretrain_epochs,
        'pretrain_lr':            args.pretrain_lr,
        'pretrain_batch_size':    args.pretrain_batch_size,
        'proj_mlp':               args.proj_mlp,
        'freeze_epochs':          args.freeze_epochs,
        'unfreeze_lr':            args.unfreeze_lr,
        'rare_upsample_thresh':    args.rare_upsample_thresh,
        'rare_upsample_cap':       args.rare_upsample_cap,
        'sc_pl_upsample_thresh':   args.sc_pl_upsample_thresh,
        'sc_pl_upsample_cap':      args.sc_pl_upsample_cap,
        'scheduler':              args.scheduler,
        'cosine_restart_period':  args.cosine_restart_period,
        'drop_path_rate':         args.drop_path_rate,
        'focal_val_frac':         args.focal_val_frac,
        'pl_pseudo_th':           args.pl_pseudo_th,
        'pl_pseudo_power':        args.pl_pseudo_power,
        'pl_pseudo_alpha':        args.pl_pseudo_alpha,
        'pl_perch_max':           args.pl_perch_max,
        'pl_sc_pseudo_power':         args.pl_sc_pseudo_power,
        'pl_sc_clip_th':              args.pl_sc_clip_th,
        'sc_chunk_level_weights':     args.sc_chunk_level_weights,
        'sc_pl_exclude_labelled':     args.sc_pl_exclude_labelled,
        'sc_pl_hard_labels':          args.sc_pl_hard_labels,
        'sc_pl_sub_prob':             args.sc_pl_sub_prob,
        'imagenet_norm':              args.imagenet_norm,
        'use_llrd':               args.use_llrd,
        'llrd_decay':             args.llrd_decay,
        'use_gem':                args.use_gem,
        'gem_p_init':             args.gem_p_init,
        'dual_loss_weight':       args.dual_loss_weight,
        'cons_weight':            args.cons_weight,
        'att_entropy_weight':     args.att_entropy_weight,
        'att_edge_weight':        args.att_edge_weight,
        'mixup_min_overlap':      args.mixup_min_overlap,
        'in_chans':               args.in_chans,
        'use_focal_loss':         args.use_focal_loss,
        'focal_gamma':            args.focal_gamma,
        'focal_alpha':            args.focal_alpha,
        'use_ce_loss':            args.use_ce_loss,
        'use_soft_auc_loss':      args.use_soft_auc_loss,
        'soft_auc_margin':        args.soft_auc_margin,
        'soft_auc_bce_weight':    args.soft_auc_bce_weight,
        'use_logit_auc_loss':     args.use_logit_auc_loss,
        'logit_auc_margin':       args.logit_auc_margin,
        'logit_auc_pos_weight':   args.logit_auc_pos_weight,
        'logit_auc_neg_weight':   args.logit_auc_neg_weight,
        'logit_auc_bce_weight':   args.logit_auc_bce_weight,
        'use_smooth_ap_loss':     args.use_smooth_ap_loss,
        'smooth_ap_tau':          args.smooth_ap_tau,
        'smooth_ap_bce_weight':   args.smooth_ap_bce_weight,
        'bg_aug_prob':            args.bg_aug_prob,
        'bg_aug_snr_min_db':      args.bg_aug_snr_min_db,
        'bg_aug_snr_max_db':      args.bg_aug_snr_max_db,
        'noise_dir':              args.noise_dir,
        'noise_padding':          args.noise_padding,
        'focal_pad_type':         args.focal_pad_type,
        'sc_use_window':          args.sc_use_window,
        'freq_mixstyle':          args.freq_mixstyle,
        'freq_mixstyle_alpha':    args.freq_mixstyle_alpha,
        'time_mask':              args.time_mask,
        'duration':               args.duration,
        'hop_length':             args.hop_length,
        'n_fft':                  args.n_fft,
        'n_mels':                 args.n_mels,
        'fmin':                   args.fmin,
        'fmax':                   args.fmax,
        'mel_w':                  args.mel_w,
        'stem_stride':            args.stem_stride,
    }
    for k, v in cli_overrides.items():
        if v is not None:
            setattr(cfg, k, v)

    # Default output_dir if still unset
    if cfg.output_dir == '/kaggle/working':
        cfg.output_dir = f'./runs/{cfg.backbone}_{cfg.stage}'

    return cfg
