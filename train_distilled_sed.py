#!/usr/bin/env python3
"""
BirdCLEF+ 2026 — Distilled SED v2: Local Training Script

Adapted from notebooks/BC2026 Distilled-SED v2 Training.py for running outside Kaggle.

Key differences from the Kaggle version:
  - Reads OGG files directly from train_audio/ and train_soundscapes/ (no Tucker .pt cache)
  - Focal metadata sourced from audio_cache_meta.csv; fold assignment identical to Kaggle notebook
  - SC metadata sourced from soundscape_cache_meta.csv (Tucker's index); same window set
  - No pip install block — install deps via requirements.txt
  - No online ONNX Perch teacher — cache-only distillation

Perch embedding cache key format (unchanged from Kaggle version):
    focal : f"{cache_file}:::{start_sec}"
              cache_file from audio_cache_meta.csv, e.g. "audio/audio_000000.pt"
    SC    : f"{cache_file}:::{start_sec}"
              cache_file from soundscape_cache_meta.csv,
              e.g. "soundscape/sc_BC2026_Train_0001_S08_20250606_030007.pt"

Required files in PERCH_CACHE_DIR (Datasets/Perch embeddings for SED/ by default):
    perch_focal_cache.npz
    perch_sc_cache.npz
    audio_cache_meta.csv        — from Tucker's waveform-cache Kaggle dataset
    soundscape_cache_meta.csv   — from Tucker's waveform-cache Kaggle dataset
"""

import json
import os
import gc
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import timm
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

warnings.filterwarnings("ignore")

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    print("WARNING: onnxruntime not installed — ONNX export disabled")

# =============================================================
# S1 — Configuration
# =============================================================

_REPO_ROOT = Path(__file__).resolve().parent.parent

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
# TF32: free throughput on Ampere+; negligible accuracy difference for this task
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Speed toggles ─────────────────────────────────────────
# BF16: preferred over FP16 on Ampere+ (no overflow, no GradScaler needed).
USE_BF16          = True   # bfloat16 AMP (RTX 3090/4090/A100/H100)
# Channels-last: NHWC layout — 10-30% faster for conv-heavy backbones (EfficientNet, NFNet).
USE_CHANNELS_LAST = True
# torch.compile: ~20-40% faster after the first epoch warm-up.
# Disable if you hit graph-break warnings or need to debug.
USE_COMPILE       = False

_AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

# ── A/B toggles ───────────────────────────────────────────
USE_SOFT_AUC     = True    # SoftAUC + BCE blend
USE_SWA          = True    # Stochastic Weight Averaging
USE_HOP64_RESIZE = False   # Dense mel then bilinear-resize
USE_MORE_SC      = False   # SC share → 0.2, EPOCHS → 30

BCE_WEIGHT      = 0.25
SWA_START_EPOCH = 25

# ── Local paths ───────────────────────────────────────────
COMP_DIR = Path("/workspace/Birdclef/datasets/birdclef-2026")
TRAIN_AUDIO_DIR = COMP_DIR / "train_audio"
SOUNDSCAPE_DIR  = COMP_DIR / "train_soundscapes"

# Contains perch_focal_cache.npz, perch_sc_cache.npz, audio_cache_meta.csv
PERCH_CACHE_DIR = Path("/workspace/Birdclef/datasets/Perch embeddings for SED")

# XC-pretrained backbone .pth (or None for ImageNet defaults)
_xc_default = None
XC_WEIGHTS_PATH= Path("/workspace/Birdclef/datasets/extracted_backbones/tf_efficientnetv2_s_in21k_pretrain_from_bigXCV2Ext_swa.ckpt")

OUT_DIR = Path("/workspace/Birdclef/experiments/exp86b")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Backbone & distillation mode ──────────────────────────
BACKBONE_MODE = "no_distill"
BACKBONE_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"

_ALPHA_MAP = {
    "distill_sg":   1.0,
    "distill_nosg": 0.2,
    "no_distill":   0.0,
    "frozen":       0.0,
}
USE_PERCH_DISTILL = BACKBONE_MODE in ("distill_sg", "distill_nosg")
ALPHA_DISTILL     = _ALPHA_MAP[BACKBONE_MODE]
USE_STOP_GRAD     = BACKBONE_MODE == "distill_sg"
FREEZE_BACKBONE   = BACKBONE_MODE == "frozen"

# ── Dataset & mel ─────────────────────────────────────────
NUM_CLASSES    = 234
SR             = 32_000
TRAIN_DURATION = 5
VAL_DURATION   = 5
TRAIN_SAMPLES  = SR * TRAIN_DURATION
VAL_SAMPLES    = SR * VAL_DURATION
N_FOLDS        = 5
FOLDS          = [0, 1, 2, 3, 4]

N_FFT      = 2048
HOP_LENGTH = 512
N_MELS     = 128
FMIN       = 20
FMAX       = 16000

# ── Model ─────────────────────────────────────────────────
HIDDEN_DIM      = 512
PERCH_EMBED_DIM = 1536
DROP_PATH_RATE  = 0.1

# ── Training ──────────────────────────────────────────────
EPOCHS        = 35
BATCH         = 64
LR            = 5e-4
MIN_LR        = 1e-6
WD            = 1e-4
WARMUP_EPOCHS = 2
# soundfile partial-reads are cheap per worker; 4-8 is safe on a 4090 workstation.
NUM_WORKERS   = 4

# ── Augmentation ──────────────────────────────────────────
MIN_SAMPLE             = 20
AUG_PROB               = 0.5
AUG_GAIN_DB_RANGE      = (-6.0, 6.0)
AUG_NOISE_SNR_DB_RANGE = (10.0, 30.0)

USE_FOCAL_MIXUP      = True
MIXUP_PROB           = 0.5
MIXUP_ALPHA          = 0.4
MIXUP_HARD           = True

USE_FOCAL_SC_MIXUP   = True
FOCAL_SC_MIXUP_PROB  = 0.5
FOCAL_SC_MIXUP_ALPHA = 0.4

FREQ_MASK_PARAM = 10
TIME_MASK_PARAM = 10
NUM_FREQ_MASKS  = 1
NUM_TIME_MASKS  = 2

# ── Source weights ────────────────────────────────────────
USE_FOCAL           = True
USE_FOCAL_SECONDARY = True
USE_LABELED_SC      = True
SHARES         = {"focal": 0.8, "sc": 0.2} if USE_MORE_SC else {"focal": 0.9, "sc": 0.1}
SOURCE_WEIGHTS = {"focal": 1.0, "focal_missing": 0.0, "sc": 1.0}

# ── Derived: Perch cache availability ─────────────────────
USE_PERCH_CACHE = (
    USE_PERCH_DISTILL
    and (PERCH_CACHE_DIR / "perch_focal_cache.npz").exists()
    and (PERCH_CACHE_DIR / "perch_sc_cache.npz").exists()
)

print(f"Device  : {device}" + (f"  GPU: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else ""))
print(f"Backbone: {BACKBONE_NAME}  mode={BACKBONE_MODE}")
print(f"Distill : {'ON' if USE_PERCH_DISTILL else 'OFF'}  "
      f"alpha={ALPHA_DISTILL}  stop_grad={USE_STOP_GRAD}  frozen={FREEZE_BACKBONE}")
print(f"Perch cache: {'YES' if USE_PERCH_CACHE else 'NO'}")
print(f"XC weights : {XC_WEIGHTS_PATH}")
print(f"Folds: {FOLDS}  Epochs: {EPOCHS}  Batch: {BATCH}")
print(f"A/B: SoftAUC={USE_SOFT_AUC}(w={BCE_WEIGHT})  SWA={USE_SWA}(ep{SWA_START_EPOCH}+)")

# =============================================================
# S2 — Load Data
# =============================================================

# ── Label ordering ────────────────────────────────────────
sample_sub     = pd.read_csv(COMP_DIR / "sample_submission.csv")
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
LABEL2IDX      = {label: idx for idx, label in enumerate(PRIMARY_LABELS)}
taxonomy       = pd.read_csv(COMP_DIR / "taxonomy.csv")
label_to_taxon = dict(zip(
    taxonomy["primary_label"].astype(str),
    taxonomy["class_name"].astype(str),
))
TAXON_MASKS = {
    t: np.array([i for i, l in enumerate(PRIMARY_LABELS) if label_to_taxon.get(l, "") == t])
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]
}

# ── Focal metadata — Tucker's audio_cache_meta.csv ────────
# Mirrors the Kaggle notebook exactly:
#   audio_cache_meta ← audio_cache_meta.csv (one row per original file)
#   merged with train.csv secondary_labels
#   filtered to species in LABEL2IDX
# Each record carries cache_file for Perch embedding lookup.
_focal_meta_csv = PERCH_CACHE_DIR / "audio_cache_meta.csv"
if not _focal_meta_csv.exists():
    raise FileNotFoundError(
        f"audio_cache_meta.csv not found in {PERCH_CACHE_DIR}. "
        "Download it from Tucker's waveform-cache Kaggle dataset."
    )

audio_cache_meta = pd.read_csv(_focal_meta_csv)
_train_csv       = pd.read_csv(COMP_DIR / "train.csv")
audio_cache_meta = audio_cache_meta.merge(
    _train_csv[["filename", "secondary_labels"]], on="filename", how="left"
)
audio_cache_meta = audio_cache_meta[
    audio_cache_meta["primary_label"].isin(LABEL2IDX)
].reset_index(drop=True)

# Resolve local OGG paths
audio_cache_meta["file_path"] = audio_cache_meta["filename"].apply(
    lambda fn: str(TRAIN_AUDIO_DIR / fn)
)
_exists = audio_cache_meta["file_path"].apply(lambda p: Path(p).exists())
if not _exists.all():
    print(f"Warning: {(~_exists).sum()} focal OGG files not found on disk — dropping")
    audio_cache_meta = audio_cache_meta[_exists].reset_index(drop=True)

print(f"Focal audio cache: {len(audio_cache_meta)} entries")

# ── Soundscape metadata ────────────────────────────────────
# soundscape_cache_meta.csv mirrors Tucker's Kaggle dataset:
# one row per 5-second window with filename, start_sec, site, cache_file, label_list.
sc_cache_meta = pd.read_csv(PERCH_CACHE_DIR / "soundscape_cache_meta.csv")
sc_cache_meta["label_list"] = sc_cache_meta["label_list"].apply(
    lambda x: x.split(";") if isinstance(x, str) and x.strip() else []
)
print(f"Soundscape cache: {len(sc_cache_meta)} windows, "
      f"{sc_cache_meta['filename'].nunique()} files")

# ── Soundscape label matrix ────────────────────────────────
Y_SC = np.zeros((len(sc_cache_meta), NUM_CLASSES), dtype=np.float32)
for i, row in sc_cache_meta.iterrows():
    for lbl in row["label_list"]:
        lbl = lbl.strip()
        if lbl in LABEL2IDX:
            Y_SC[i, LABEL2IDX[lbl]] = 1.0

labeled_sc_mask = Y_SC.sum(axis=1) > 0
print(f"Soundscape labels: {labeled_sc_mask.sum()}/{len(Y_SC)} windows labeled, "
      f"{int(Y_SC.sum())} positives")

sc_sites        = sc_cache_meta["site"].values
non_s22_mask_sc = sc_sites != "S22"
print(f"S22: {(~non_s22_mask_sc).sum()}, non-S22: {non_s22_mask_sc.sum()}")

# ── Fold assignment — focal ───────────────────────────────
# Identical to Kaggle notebook: deduplicate on original_idx, StratifiedKFold
# by primary_label, merge fold back into full metadata.
audio_for_split = audio_cache_meta.drop_duplicates("original_idx").reset_index(drop=True)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
audio_for_split["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(audio_for_split, audio_for_split["primary_label"])):
    audio_for_split.loc[val_idx, "fold"] = fold
audio_cache_meta = audio_cache_meta.merge(
    audio_for_split[["original_idx", "fold"]], on="original_idx", how="left"
)
print(f"Focal fold distribution:\n{audio_cache_meta['fold'].value_counts().sort_index()}")

# ── Fold assignment — soundscape ──────────────────────────
sc_files = sc_cache_meta[["filename", "site"]].drop_duplicates().reset_index(drop=True)
gkf = GroupKFold(n_splits=N_FOLDS)
sc_files["fold"] = -1
for fold, (_, val_idx) in enumerate(gkf.split(sc_files, groups=sc_files["filename"])):
    sc_files.loc[sc_files.index[val_idx], "fold"] = fold
file_to_fold  = dict(zip(sc_files["filename"], sc_files["fold"]))
sc_cache_meta["fold"] = sc_cache_meta["filename"].map(file_to_fold).fillna(-1).astype(int)

# ── Upsample rare focal species ───────────────────────────
counts = audio_cache_meta["primary_label"].value_counts()
rare   = counts[counts < MIN_SAMPLE].index
extra  = []
for sp in rare:
    sp_rows  = audio_cache_meta[audio_cache_meta["primary_label"] == sp]
    n_copies = int(np.ceil(MIN_SAMPLE / len(sp_rows))) - 1
    for _ in range(n_copies):
        extra.append(sp_rows)
n_before = len(audio_cache_meta)
if extra:
    audio_cache_meta = pd.concat([audio_cache_meta] + extra, ignore_index=True)
print(f"Upsampled {len(rare)} rare species: {n_before} → {len(audio_cache_meta)}")

# ── Preload soundscape waveforms ──────────────────────────
# Workers inherit this dict via fork — no SC file I/O in the hot path.
def _load_ogg(path: Path) -> np.ndarray | None:
    """Load OGG/WAV → float32 mono numpy at SR."""
    try:
        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        return wav.squeeze(0).numpy().astype(np.float32)
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return None


SC_WAV_ALL: dict[str, np.ndarray] = {}
for _fn in sc_cache_meta["filename"].unique():
    _p = SOUNDSCAPE_DIR / _fn
    if _p.exists():
        _wav = _load_ogg(_p)
        if _wav is not None:
            SC_WAV_ALL[_fn] = _wav

_sc_ram_mb = sum(v.nbytes for v in SC_WAV_ALL.values()) / 1e6
print(f"Preloaded {len(SC_WAV_ALL)}/{sc_cache_meta['filename'].nunique()} soundscape files "
      f"({_sc_ram_mb:.0f} MB)")

# ── Secondary labels ──────────────────────────────────────
focal_secondary_labels: dict = {}
if USE_FOCAL_SECONDARY:
    for _, row in audio_cache_meta.drop_duplicates("original_idx").iterrows():
        sec = row.get("secondary_labels", "")
        if pd.isna(sec) or sec in ("", "[]"):
            continue
        try:
            sec_list = eval(sec) if isinstance(sec, str) else []
        except Exception:
            continue
        valid = [s for s in sec_list if s in LABEL2IDX]
        if valid:
            focal_secondary_labels[int(row["original_idx"])] = valid
    print(f"Secondary labels: {len(focal_secondary_labels)} files")

# ── Perch embedding caches ────────────────────────────────
# Key format matches Tucker's notebook exactly (cache_file:::start_sec).
FOCAL_EMB_LOOKUP: dict = {}
SC_EMB_LOOKUP:    dict = {}
if USE_PERCH_CACHE:
    _fc = np.load(PERCH_CACHE_DIR / "perch_focal_cache.npz", allow_pickle=True)
    FOCAL_EMB_LOOKUP = dict(zip(_fc["keys"], _fc["embeddings"]))
    del _fc
    _sc = np.load(PERCH_CACHE_DIR / "perch_sc_cache.npz", allow_pickle=True)
    SC_EMB_LOOKUP = dict(zip(_sc["keys"], _sc["embeddings"]))
    del _sc
    print(f"Perch cache: {len(FOCAL_EMB_LOOKUP)} focal  {len(SC_EMB_LOOKUP)} SC windows")

print("OK Data loaded")

# =============================================================
# S3 — Model Architecture
# =============================================================

def compute_macro_auc(y_true, y_pred, mask=None, class_mask=None):
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    if class_mask is not None:
        y_true, y_pred = y_true[:, class_mask], y_pred[:, class_mask]
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        try:
            aucs.append(roc_auc_score(col, y_pred[:, c]))
        except ValueError:
            continue
    return (np.mean(aucs) if aucs else float("nan")), len(aucs)


def full_eval(y_true, y_pred, ns22, tm):
    r = {}
    a, n = compute_macro_auc(y_true, y_pred)
    r["macro_auc_all"], r["n_all"] = round(a, 4), n
    a, n = compute_macro_auc(y_true, y_pred, mask=ns22)
    r["non_s22_macro"], r["n_ns22"] = round(a, 4), n
    for t, cm in tm.items():
        a, _ = compute_macro_auc(y_true, y_pred, mask=ns22, class_mask=cm)
        r[f"non_s22_{t}"] = round(a, 4)
    return r


class SoftAUCLoss(nn.Module):
    """Pairwise AUC surrogate (squared hinge). O(B²·C)."""
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        p    = torch.sigmoid(logits)
        diff = p.unsqueeze(1) - p.unsqueeze(0)
        pair = labels.unsqueeze(1) * (1.0 - labels.unsqueeze(0))
        loss = pair * torch.clamp(self.margin - diff, min=0.0) ** 2
        denom = pair.sum(dim=(0, 1)).clamp(min=1.0)
        return (loss.sum(dim=(0, 1)) / denom).mean()


class MelSpecTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
        if USE_HOP64_RESIZE:
            self._mel_hr = torchaudio.transforms.MelSpectrogram(
                sample_rate=SR, n_fft=N_FFT, hop_length=64,
                n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
            )
            self._T_std = TRAIN_SAMPLES // HOP_LENGTH + 1
        else:
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
            )

    def forward(self, x):
        if USE_HOP64_RESIZE:
            mel = self.db_transform(self._mel_hr(x))
            return F.interpolate(mel, size=(N_MELS, self._T_std),
                                 mode="bilinear", align_corners=False)
        return self.db_transform(self.mel_spec(x))


class SpecAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=FREQ_MASK_PARAM)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=TIME_MASK_PARAM)

    def forward(self, mel):
        for _ in range(NUM_FREQ_MASKS):
            mel = self.freq_mask(mel)
        for _ in range(NUM_TIME_MASKS):
            mel = self.time_mask(mel)
        return mel


class GeMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)


class DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=PERCH_EMBED_DIM):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)

    def forward(self, feature_map):
        return self.proj(feature_map.mean(dim=[2, 3]))


class BirdSEDModel(nn.Module):
    def __init__(self, backbone_name=BACKBONE_NAME, num_classes=NUM_CLASSES,
                 drop_path_rate=DROP_PATH_RATE, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )

        if XC_WEIGHTS_PATH is not None and Path(XC_WEIGHTS_PATH).exists():
            ckpt  = torch.load(str(XC_WEIGHTS_PATH), map_location="cpu", weights_only=False)
            state = ckpt.get("backbone", ckpt.get("model", ckpt))
            for key in ("conv_stem.weight", "stem.conv1.weight", "stem.conv0.weight"):
                if key in state and state[key].shape[1] == 3:
                    state[key] = state[key].mean(dim=1, keepdim=True)
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            print(f"XC weights: {len(missing)} missing  {len(unexpected)} unexpected")
        elif XC_WEIGHTS_PATH is not None:
            print(f"Warning: XC weights not found at {XC_WEIGHTS_PATH} — using ImageNet")

        if FREEZE_BACKBONE:
            self.backbone.requires_grad_(False)
            print("Backbone frozen")

        with torch.no_grad():
            n_tf  = TRAIN_SAMPLES // HOP_LENGTH + 1
            dummy = torch.randn(1, 1, N_MELS, n_tf)
            feat  = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
        print(f"Backbone: {backbone_name}  dim={self.backbone_dim}")

        self.gem_freq = GeMFreqPool(p_init=3.0)
        self.dense    = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.att.weight); self.att.bias.data.fill_(0.)
        nn.init.xavier_uniform_(self.cla.weight); self.cla.bias.data.fill_(0.)

        if USE_PERCH_DISTILL:
            self.distill_head = DistillHead(self.backbone_dim, PERCH_EMBED_DIM)

    def _sed_head(self, h):
        h = self.gem_freq(h)
        h = h.permute(0, 2, 1)
        h = self.dense(h)
        h = h.permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h)), dim=-1)
        fw       = self.cla(h)
        clip     = torch.sum(norm_att * fw, dim=2)
        return clip, fw.permute(0, 2, 1)

    def forward(self, x, return_framewise=False, return_distill=False):
        h = self.backbone(x)

        distill_emb = None
        if return_distill and hasattr(self, "distill_head"):
            distill_emb = self.distill_head(h)

        h_cls = h.detach() if USE_STOP_GRAD else h
        clip_logits, fw = self._sed_head(h_cls)

        if return_framewise and return_distill:
            return clip_logits, fw, distill_emb
        if return_framewise:
            return clip_logits, fw
        if return_distill:
            return clip_logits, distill_emb
        return clip_logits


def make_model():
    return BirdSEDModel(BACKBONE_NAME).to(device)


print("OK Model definitions ready")

# =============================================================
# S4 — Data Pipeline
# =============================================================

def extract_chunk_np(waveform: np.ndarray, start_sample: int, n_samples: int) -> np.ndarray:
    total = len(waveform)
    if total <= n_samples:
        return np.pad(waveform, (n_samples - total, 0))
    end = start_sample + n_samples
    if end > total:
        start_sample = max(0, total - n_samples)
    return waveform[start_sample:start_sample + n_samples]


def apply_aug(w: np.ndarray) -> np.ndarray:
    if np.random.random() < AUG_PROB:
        w = w * (10 ** (np.random.uniform(*AUG_GAIN_DB_RANGE) / 20))
    if np.random.random() < AUG_PROB:
        sp = (w ** 2).mean()
        if sp > 1e-10:
            snr   = np.random.uniform(*AUG_NOISE_SNR_DB_RANGE)
            noise = np.random.randn(*w.shape).astype(w.dtype)
            w     = w + noise * np.sqrt(sp / (10 ** (snr / 10)))
    return w


def _zero_emb() -> np.ndarray:
    return np.zeros(PERCH_EMBED_DIM, dtype=np.float32)


# SC MixUp pool: labeled windows whose audio is preloaded
SC_MIXUP_POOL: list[dict] = []
for _i, _row in sc_cache_meta.iterrows():
    if Y_SC[_i].sum() > 0 and _row["filename"] in SC_WAV_ALL:
        SC_MIXUP_POOL.append({
            "filename":   _row["filename"],
            "cache_file": _row["cache_file"],
            "start_sec":  int(_row["start_sec"]),
            "label_idx":  _i,
            "fold":       int(_row.get("fold", -1)),
        })
print(f"SC MixUp pool: {len(SC_MIXUP_POOL)} labeled windows")


class FocalDS(Dataset):
    """
    Focal recording dataset. Each item is a 6-tuple:
        (waveform, label, weight, mask, source_tag, perch_emb)

    Uses soundfile.read() with a start offset so only the 5-second chunk is decoded
    (vs torchaudio.load which decodes the full file). File length is read from the
    n_samples column in audio_cache_meta.csv, avoiding a soundfile.info() syscall.
    Perch key: f"{cache_file}:::{start_sec}"  (Tucker's original format)
    """
    def __init__(self, df, l2i, secondary_lookup=None, fold_k=None, aug=False):
        self.records          = df.reset_index(drop=True).to_dict("records")
        self.l2i              = l2i
        self.aug              = aug
        self.secondary_lookup = secondary_lookup
        self.fold_k           = fold_k
        self._eligible_sc     = (
            [r for r in SC_MIXUP_POOL if r["fold"] != fold_k]
            if fold_k is not None else SC_MIXUP_POOL
        )

    def __len__(self):
        return len(self.records)

    def _load_chunk(self, r: dict):
        """Returns (chunk_np, label_np, perch_emb_np) or (None, None, None).

        soundfile.read decodes only TRAIN_SAMPLES frames starting at start_sec*SR.
        For a 20-second OGG this is ~4× less decode work than loading the full file.
        """
        n_windows = max(1, int(r["n_samples"]) // TRAIN_SAMPLES)
        start_sec = int(np.random.randint(0, n_windows)) * 5 if self.aug else 0

        try:
            ch, file_sr = soundfile.read(
                r["file_path"],
                frames=TRAIN_SAMPLES,
                start=start_sec * SR,
                dtype="float32",
                always_2d=False,
            )
            if ch.ndim == 2:
                ch = ch.mean(axis=1)
        except Exception:
            return None, None, None

        if len(ch) < TRAIN_SAMPLES:
            ch = np.pad(ch, (TRAIN_SAMPLES - len(ch), 0))  # left-pad short clips
        if file_sr != SR:
            ch = torchaudio.functional.resample(
                torch.from_numpy(ch), file_sr, SR
            ).numpy()

        lb = np.zeros(NUM_CLASSES, dtype=np.float32)
        pl = str(r["primary_label"])
        if pl in self.l2i:
            lb[self.l2i[pl]] = 1.0
        if self.secondary_lookup is not None and "original_idx" in r:
            for s in self.secondary_lookup.get(int(r["original_idx"]), []):
                if s in self.l2i:
                    lb[self.l2i[s]] = 1.0

        cache_key = f"{r['cache_file']}:::{start_sec}" if USE_PERCH_CACHE else None
        emb = FOCAL_EMB_LOOKUP.get(cache_key) if cache_key else None
        if emb is None:
            emb = _zero_emb()

        return ch, lb, emb

    def __getitem__(self, i):
        r1    = self.records[i]
        ch1, lb1, emb1 = self._load_chunk(r1)
        _ones = torch.ones(NUM_CLASSES)

        if ch1 is None:
            return (torch.zeros(1, TRAIN_SAMPLES), torch.zeros(NUM_CLASSES),
                    _ones, _ones, "focal_missing", torch.zeros(PERCH_EMBED_DIM))

        # Focal–Focal MixUp
        if USE_FOCAL_MIXUP and self.aug and np.random.random() < MIXUP_PROB:
            ch2 = None
            for _ in range(3):
                ch2, lb2, emb2 = self._load_chunk(
                    self.records[np.random.randint(len(self.records))]
                )
                if ch2 is not None:
                    break
            if ch2 is not None:
                lam  = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                ch_m = (lam * ch1 + (1 - lam) * ch2).astype(np.float32)
                if self.aug:
                    ch_m = apply_aug(ch_m)
                lb_m  = np.maximum(lb1, lb2) if MIXUP_HARD else lam * lb1 + (1 - lam) * lb2
                emb_m = (lam * emb1 + (1 - lam) * emb2).astype(np.float32)
                return (torch.from_numpy(ch_m).unsqueeze(0), torch.from_numpy(lb_m),
                        _ones, _ones, "focal", torch.from_numpy(emb_m))

        # Focal–Soundscape MixUp
        if (USE_FOCAL_SC_MIXUP and self.aug and self._eligible_sc
                and np.random.random() < FOCAL_SC_MIXUP_PROB):
            sc_row = self._eligible_sc[np.random.randint(len(self._eligible_sc))]
            sc_wav = SC_WAV_ALL.get(sc_row["filename"])
            if sc_wav is not None and len(sc_wav) >= TRAIN_SAMPLES:
                sc_chunk = extract_chunk_np(sc_wav, sc_row["start_sec"] * SR, TRAIN_SAMPLES)
                lam      = np.random.beta(FOCAL_SC_MIXUP_ALPHA, FOCAL_SC_MIXUP_ALPHA)
                ch_m     = (lam * ch1 + (1 - lam) * sc_chunk).astype(np.float32)
                if self.aug:
                    ch_m = apply_aug(ch_m)
                lb_sc = Y_SC[sc_row["label_idx"]].astype(np.float32)
                lb_m  = np.maximum(lb1, lb_sc) if MIXUP_HARD else lam * lb1 + (1 - lam) * lb_sc
                sc_key = f"{sc_row['cache_file']}:::{sc_row['start_sec']}"
                emb_sc = SC_EMB_LOOKUP.get(sc_key, _zero_emb())
                emb_m  = (lam * emb1 + (1 - lam) * emb_sc).astype(np.float32)
                return (torch.from_numpy(ch_m).unsqueeze(0), torch.from_numpy(lb_m),
                        _ones, _ones, "focal", torch.from_numpy(emb_m))

        if self.aug:
            ch1 = apply_aug(ch1)
        return (torch.from_numpy(ch1.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(lb1), _ones, _ones, "focal",
                torch.from_numpy(emb1.astype(np.float32)))


class ScDS(Dataset):
    """Labeled soundscape windows. Same 6-tuple as FocalDS."""
    def __init__(self, Y, sc_df, aug=False):
        self.Y    = Y
        self.aug  = aug
        self.rows = sc_df.reset_index(drop=True).to_dict("records")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row      = self.rows[i]
        fn       = row.get("filename")
        wav_full = SC_WAV_ALL.get(fn) if fn else None
        _ones    = torch.ones(NUM_CLASSES)

        if wav_full is None:
            return (torch.zeros(1, TRAIN_SAMPLES),
                    torch.from_numpy(self.Y[i].astype(np.float32)),
                    _ones, _ones, "sc", torch.zeros(PERCH_EMBED_DIM))

        start_sec = int(row["start_sec"])
        chunk     = extract_chunk_np(wav_full, start_sec * SR, TRAIN_SAMPLES)
        if self.aug:
            chunk = apply_aug(chunk)
        sc_key = f"{row['cache_file']}:::{start_sec}"
        emb    = SC_EMB_LOOKUP.get(sc_key, _zero_emb())
        return (torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(self.Y[i].astype(np.float32)),
                _ones, _ones, "sc",
                torch.from_numpy(emb))


class MixSamp(torch.utils.data.Sampler):
    """Multi-source batch sampler. Call set_epoch(ep) before each epoch."""
    def __init__(self, sizes, names, shares, bs, nst, base_seed=42):
        self.sizes, self.names, self.bs, self.nst = sizes, names, bs, nst
        self.base_seed = base_seed
        per_src = [max(1, int(round(bs * shares.get(n, 0.0)))) for n in names]
        total   = sum(per_src)
        if total != bs:
            per_src[int(np.argmax(per_src))] += (bs - total)
        self.per_src = per_src
        self.offsets = [0]
        for s in sizes[:-1]:
            self.offsets.append(self.offsets[-1] + s)
        self.rng = np.random.default_rng(base_seed)

    def set_epoch(self, ep: int):
        self.rng = np.random.default_rng(self.base_seed + ep)

    def __len__(self):
        return self.nst

    def __iter__(self):
        for _ in range(self.nst):
            batch = []
            for off, size, n in zip(self.offsets, self.sizes, self.per_src):
                if n > 0 and size > 0:
                    batch.extend([off + int(k) for k in self.rng.integers(0, size, size=n)])
            self.rng.shuffle(batch)
            yield batch


def collate_m(batch):
    return (
        torch.stack([b[0] for b in batch]),
        torch.stack([b[1] for b in batch]),
        torch.stack([b[2] for b in batch]),
        torch.stack([b[3] for b in batch]),
        [b[4] for b in batch],
        torch.stack([b[5] for b in batch]),
    )


def mk_sw(sr):
    return torch.tensor([SOURCE_WEIGHTS.get(s, 0.0) for s in sr], dtype=torch.float32)


print("OK Data pipeline ready")

# =============================================================
# S5 — Training Loop
# =============================================================

def _load_val_waveforms(val_sc_df: pd.DataFrame) -> list[torch.Tensor]:
    wavs = []
    for _, row in val_sc_df.iterrows():
        fn       = row["filename"]
        wav_full = SC_WAV_ALL.get(fn)
        if wav_full is not None:
            chunk = extract_chunk_np(wav_full, int(row["start_sec"]) * SR, VAL_SAMPLES)
            wavs.append(torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0))
        else:
            wavs.append(torch.zeros(1, VAL_SAMPLES))
    return wavs


def _mel_norm_batch(mel: torch.Tensor) -> torch.Tensor:
    B    = mel.size(0)
    flat = mel.reshape(B, -1)
    mean = flat.mean(dim=1).view(B, 1, 1, 1)
    std  = flat.std(dim=1).view(B, 1, 1, 1)
    return (mel - mean) / (std + 1e-6)


def _predict_from_waveforms(model, mel_transform, wav_list, batch_size=64):
    model.eval()
    preds_clip, preds_fmax, preds_blend = [], [], []
    with torch.no_grad():
        for s in range(0, len(wav_list), batch_size):
            batch = torch.stack(wav_list[s:s + batch_size]).to(device)
            mel   = mel_transform(batch)
            mel   = _mel_norm_batch(mel)
            if USE_CHANNELS_LAST:
                mel = mel.to(memory_format=torch.channels_last)
            with autocast(dtype=_AMP_DTYPE):
                clip_logits, framewise = model(mel, return_framewise=True)
                frame_max = framewise.max(dim=1).values
            p_clip  = torch.sigmoid(clip_logits).float().cpu().numpy()
            p_fmax  = torch.sigmoid(frame_max).float().cpu().numpy()
            p_blend = 0.5 * p_clip + 0.5 * p_fmax
            preds_clip.append(p_clip)
            preds_fmax.append(p_fmax)
            preds_blend.append(p_blend)
    return {
        "clip":  np.concatenate(preds_clip),
        "fmax":  np.concatenate(preds_fmax),
        "blend": np.concatenate(preds_blend),
    }


def _swa_update_bn(swa_model, loader, mel_transform, n_batches: int = 100):
    for mod in swa_model.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            mod.reset_running_stats()
            mod.num_batches_tracked.zero_()
    swa_model.train()
    with torch.no_grad():
        for i, (wav, *_) in enumerate(loader):
            if i >= n_batches:
                break
            mel = mel_transform(wav.to(device))
            mel = _mel_norm_batch(mel)
            swa_model(mel)
    swa_model.eval()


def train_fold(fold_k: int):
    vm        = sc_cache_meta["fold"].values == fold_k
    Y_val     = Y_SC[vm]
    ns22_val  = non_s22_mask_sc[vm]
    val_sc_df = sc_cache_meta[vm].reset_index(drop=True)
    val_wavs  = _load_val_waveforms(val_sc_df)

    # ── Datasets ──────────────────────────────────────────
    datasets, names, sizes = [], [], []
    if USE_FOCAL:
        fds = FocalDS(
            audio_cache_meta[audio_cache_meta["fold"] != fold_k],
            LABEL2IDX, secondary_lookup=focal_secondary_labels,
            fold_k=fold_k, aug=True,
        )
        datasets.append(fds); names.append("focal"); sizes.append(len(fds))
    if USE_LABELED_SC:
        labeled_train_mask = (~vm) & labeled_sc_mask
        sc_train_df = sc_cache_meta[labeled_train_mask].reset_index(drop=True)
        sds = ScDS(Y_SC[labeled_train_mask], sc_train_df, aug=True)
        datasets.append(sds); names.append("sc"); sizes.append(len(sds))

    mds = ConcatDataset(datasets)
    nst = max(100, int(sum(sizes) / BATCH))
    smp = MixSamp(sizes, names, SHARES, BATCH, nst, base_seed=42)

    tl = DataLoader(
        mds, batch_sampler=smp, collate_fn=collate_m,
        num_workers=NUM_WORKERS,
        pin_memory=(NUM_WORKERS > 0),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=(4 if NUM_WORKERS > 0 else None),
    )
    print(f"  Streams: {dict(zip(names, sizes))}  steps/ep: {nst}")

    m_raw = make_model()
    if USE_CHANNELS_LAST:
        m_raw = m_raw.to(memory_format=torch.channels_last)
    mel_transform = MelSpecTransform().to(device)
    spec_augment  = SpecAugment().to(device)

    soft_auc_criterion = SoftAUCLoss(margin=1.0).to(device) if USE_SOFT_AUC else None
    # SWA wraps the raw model so state_dict() and BN update work without compiled graph
    swa_model = AveragedModel(m_raw) if USE_SWA else None
    # Compile after SWA wrapping; shared parameters so update_parameters(m) still works
    m = torch.compile(m_raw, mode="reduce-overhead") if USE_COMPILE else m_raw

    trainable = [p for p in m_raw.parameters() if p.requires_grad]
    opt       = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)
    # BF16 doesn't overflow, so GradScaler is a no-op (enabled=False)
    scaler    = GradScaler(enabled=not USE_BF16)

    warmup_steps = nst * WARMUP_EPOCHS
    total_steps  = nst * EPOCHS
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1 / 25, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps - warmup_steps, eta_min=MIN_LR
    )
    sch = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps]
    )

    history = {
        "ep": [], "train_loss": [], "cls_loss": [], "dist_loss": [],
        "macro": [], "ns22_macro": [],
        "ns22_Aves": [], "ns22_Amphibia": [], "ns22_Insecta": [], "ns22_Mammalia": [],
        "val_preds": [],
    }
    best_ns22, best_state_ns22 = -1.0, None
    best_macro, best_state_macro = -1.0, None
    swa_state: dict | None = None

    for ep in range(EPOCHS):
        smp.set_epoch(ep)
        m.train()
        el, el_cls, el_dist, nb = 0.0, 0.0, 0.0, 0
        t0 = time.time()

        for wav, lb, wt, mk, sr, batch_perch_emb in tl:
            wav, lb, wt, mk = (wav.to(device), lb.to(device),
                               wt.to(device), mk.to(device))
            sw = mk_sw(sr).to(device)

            with torch.no_grad():
                mel = mel_transform(wav)
                mel = _mel_norm_batch(mel)
                if USE_CHANNELS_LAST:
                    mel = mel.to(memory_format=torch.channels_last)
                mel = spec_augment(mel)

            with autocast(dtype=_AMP_DTYPE):
                if USE_PERCH_DISTILL:
                    clip_logits, framewise, distill_emb = m(
                        mel, return_framewise=True, return_distill=True
                    )
                else:
                    clip_logits, framewise = m(mel, return_framewise=True)

                frame_max_logits = framewise.max(dim=1).values

                bce_clip  = F.binary_cross_entropy_with_logits(
                    clip_logits, lb, reduction="none"
                )
                bce_frame = F.binary_cross_entropy_with_logits(
                    frame_max_logits, lb, reduction="none"
                )
                bce      = 0.5 * bce_clip + 0.5 * bce_frame
                ps       = (bce * wt * mk).sum(1) / (mk.sum(1) + 1e-8)
                bce_loss = (ps * sw).mean()
                if USE_SOFT_AUC:
                    sauc_loss = soft_auc_criterion(clip_logits, lb)
                    cls_loss  = BCE_WEIGHT * bce_loss + (1.0 - BCE_WEIGHT) * sauc_loss
                else:
                    cls_loss = bce_loss

                if USE_PERCH_DISTILL:
                    perch_emb    = batch_perch_emb.to(device)
                    distill_loss = F.mse_loss(distill_emb, perch_emb)
                    loss         = cls_loss + ALPHA_DISTILL * distill_loss
                else:
                    distill_loss = torch.tensor(0.0, device=device)
                    loss         = cls_loss

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()

            el += loss.item(); el_cls += cls_loss.item()
            el_dist += distill_loss.item(); nb += 1

        val_preds_d = _predict_from_waveforms(m, mel_transform, val_wavs)
        r = full_eval(Y_val, val_preds_d["blend"], ns22_val, TAXON_MASKS)
        for mode in ["clip", "fmax", "blend"]:
            r[f"ns22_{mode}"] = full_eval(
                Y_val, val_preds_d[mode], ns22_val, TAXON_MASKS
            )["non_s22_macro"]

        history["ep"].append(ep)
        history["train_loss"].append(round(el / nb, 5))
        history["cls_loss"].append(round(el_cls / nb, 5))
        history["dist_loss"].append(round(el_dist / nb, 5))
        history["macro"].append(r["macro_auc_all"])
        history["ns22_macro"].append(r["non_s22_macro"])
        for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]:
            history[f"ns22_{t}"].append(r[f"non_s22_{t}"])
        history["val_preds"].append(val_preds_d["blend"].astype(np.float32))

        tag = ""
        if r["non_s22_macro"] > best_ns22:
            best_ns22       = r["non_s22_macro"]
            best_state_ns22 = {k: v.cpu().clone() for k, v in m_raw.state_dict().items()}
            tag += " *ns22"
        if r["macro_auc_all"] > best_macro:
            best_macro       = r["macro_auc_all"]
            best_state_macro = {k: v.cpu().clone() for k, v in m_raw.state_dict().items()}
            tag += " *macro"

        dist_str = f" dist={el_dist/nb:.4f}" if USE_PERCH_DISTILL else ""
        print(
            f"  Ep{ep:02d}: loss={el/nb:.4f} cls={el_cls/nb:.4f}{dist_str} "
            f"lr={opt.param_groups[0]['lr']:.1e} | "
            f"macro={r['macro_auc_all']:.4f}  ns22={r['ns22_blend']:.4f} | "
            f"Av={r['non_s22_Aves']:.4f} Am={r['non_s22_Amphibia']:.4f} "
            f"In={r['non_s22_Insecta']:.4f} Ma={r['non_s22_Mammalia']:.4f} "
            f"[{time.time() - t0:.0f}s]{tag}"
        )

        if USE_SWA and ep >= SWA_START_EPOCH:
            swa_model.update_parameters(m)

    if USE_SWA and swa_model.n_averaged > 0:
        _swa_update_bn(swa_model, tl, mel_transform)
        swa_preds = _predict_from_waveforms(swa_model, mel_transform, val_wavs)
        r_swa = full_eval(Y_val, swa_preds["blend"], ns22_val, TAXON_MASKS)
        print(f"  SWA  : ns22={r_swa['non_s22_macro']:.4f}  macro={r_swa['macro_auc_all']:.4f}  "
              f"(averaged over {int(swa_model.n_averaged)} checkpoints)")
        swa_flat  = {k: v.cpu().clone() for k, v in swa_model.module.state_dict().items()}
        swa_state = swa_flat
        if r_swa["non_s22_macro"] > best_ns22:
            best_ns22       = r_swa["non_s22_macro"]
            best_state_ns22 = swa_flat
        if r_swa["macro_auc_all"] > best_macro:
            best_macro       = r_swa["macro_auc_all"]
            best_state_macro = swa_flat

    del m, m_raw, mel_transform, spec_augment
    torch.cuda.empty_cache(); gc.collect()
    return best_state_ns22, best_state_macro, swa_state, history


print("OK Training loop ready")

# =============================================================
# S6 — Fold Loop + ONNX Export
# =============================================================

oof_ns22 = np.full((len(sc_cache_meta), NUM_CLASSES), np.nan, dtype=np.float32)
all_hist: dict = {}

for fold_k in FOLDS:
    print(f"\n{'='*60}\nFOLD {fold_k}\n{'='*60}")
    vm          = sc_cache_meta["fold"].values == fold_k
    val_sc_df_k = sc_cache_meta[vm].reset_index(drop=True)

    best_ns22_state, best_macro_state, swa_state, hist = train_fold(fold_k)
    all_hist[fold_k] = hist

    # ── Save training history ──────────────────────────────
    _metrics = {k: v for k, v in hist.items() if k != "val_preds"}
    pd.DataFrame(_metrics).to_csv(OUT_DIR / f"fold{fold_k}_history.csv", index=False)

    if hist["val_preds"]:
        np.savez_compressed(
            OUT_DIR / f"fold{fold_k}_val_preds.npz",
            preds=np.stack(hist["val_preds"]),  # (epochs, N_windows, NUM_CLASSES)
            y_true=Y_SC[vm],
            ns22=non_s22_mask_sc[vm],
        )

    _best_ep_ns22  = int(np.argmax(hist["ns22_macro"])) if hist["ns22_macro"] else -1
    _best_ep_macro = int(np.argmax(hist["macro"]))      if hist["macro"]      else -1
    _fold_summary  = {
        "fold":             fold_k,
        "epochs_run":       len(hist["ep"]),
        "best_ns22_auc":    max(hist["ns22_macro"],  default=None),
        "best_ns22_epoch":  _best_ep_ns22,
        "best_macro_auc":   max(hist["macro"],       default=None),
        "best_macro_epoch": _best_ep_macro,
        "final_ns22_Aves":      hist["ns22_Aves"][-1]     if hist["ns22_Aves"]     else None,
        "final_ns22_Amphibia":  hist["ns22_Amphibia"][-1] if hist["ns22_Amphibia"] else None,
        "final_ns22_Insecta":   hist["ns22_Insecta"][-1]  if hist["ns22_Insecta"]  else None,
        "final_ns22_Mammalia":  hist["ns22_Mammalia"][-1] if hist["ns22_Mammalia"] else None,
    }
    with open(OUT_DIR / f"fold{fold_k}_summary.json", "w") as _f:
        json.dump(_fold_summary, _f, indent=2)

    print(f"  Saved: fold{fold_k}_history.csv, fold{fold_k}_val_preds.npz, "
          f"fold{fold_k}_summary.json")

    mel_tf     = MelSpecTransform().to(device)
    val_wavs_k = _load_val_waveforms(val_sc_df_k)

    for tag, state in [("ns22", best_ns22_state), ("macro", best_macro_state), ("swa", swa_state)]:
        if state is not None:
            torch.save(state, OUT_DIR / f"fold{fold_k}_best_{tag}.pt")

    if best_macro_state is None:
        del mel_tf; gc.collect(); continue

    # OOF predictions from best-macro checkpoint
    m = make_model()
    m.load_state_dict(best_macro_state, strict=False)
    oof_ns22[vm] = _predict_from_waveforms(m, mel_tf, val_wavs_k)["blend"]

    # ── ONNX Export ───────────────────────────────────────
    if not HAS_ORT:
        print("  ONNX: skipped (onnxruntime not installed)")
        del m, mel_tf; gc.collect(); continue

    m.eval()
    N_FRAMES_EXPORT = VAL_SAMPLES // HOP_LENGTH + 1

    class SEDExportWrapper(nn.Module):
        def __init__(self, backbone_name, num_classes, backbone_dim,
                     hidden_dim=HIDDEN_DIM):
            super().__init__()
            self.backbone    = timm.create_model(
                backbone_name, pretrained=False, in_chans=1,
                num_classes=0, global_pool="", drop_path_rate=DROP_PATH_RATE,
            )
            self.gem_freq    = GeMFreqPool(p_init=3.0)
            self.dense_drop1 = nn.Dropout(0.25)
            self.dense_conv  = nn.Conv1d(backbone_dim, hidden_dim, kernel_size=1)
            self.dense_relu  = nn.ReLU(inplace=True)
            self.dense_drop2 = nn.Dropout(0.5)
            self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)
            self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

        def forward(self, mel):
            h = self.backbone(mel)
            h = self.gem_freq(h)
            h = self.dense_drop1(h)
            h = self.dense_conv(h)
            h = self.dense_relu(h)
            h = self.dense_drop2(h)
            norm_att  = torch.softmax(torch.tanh(self.att(h)), dim=-1)
            framewise = self.cla(h)
            clip      = torch.sum(norm_att * framewise, dim=2)
            return clip, framewise.permute(0, 2, 1)

    def _remap_state(export_model, trained_state: dict):
        remap = {}
        for k, v in trained_state.items():
            if k.startswith("distill_head."):
                continue
            if k == "dense.1.weight":
                remap["dense_conv.weight"] = v.unsqueeze(-1)
            elif k == "dense.1.bias":
                remap["dense_conv.bias"] = v
            else:
                remap[k] = v
        missing, unexpected = export_model.load_state_dict(remap, strict=False)
        if missing:
            print(f"  ONNX remap — missing keys: {missing}")

    export_m = SEDExportWrapper(BACKBONE_NAME, NUM_CLASSES, m.backbone_dim).to(device)
    dummy    = torch.randn(1, 1, N_MELS, N_FRAMES_EXPORT).to(device)

    for ckpt_tag, ckpt_state in [
        ("macro", best_macro_state),
        ("ns22",  best_ns22_state),
        ("swa",   swa_state),
    ]:
        if ckpt_state is None:
            print(f"  ONNX: fold{fold_k}_{ckpt_tag} skipped (no checkpoint)")
            continue
        _remap_state(export_m, ckpt_state)
        export_m.eval()
        onnx_path = OUT_DIR / f"sed_fold{fold_k}_{ckpt_tag}.onnx"
        torch.onnx.export(
            export_m, dummy, str(onnx_path),
            input_names=["mel"],
            output_names=["clip_logits", "framewise_logits"],
            dynamic_axes={
                "mel":              {0: "batch"},
                "clip_logits":      {0: "batch"},
                "framewise_logits": {0: "batch"},
            },
            opset_version=17,
        )
        _sess     = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        _onnx_out = _sess.run(None, {"mel": dummy.cpu().numpy()})
        with torch.no_grad():
            _ref_clip, _ = export_m(dummy)
        _diff = np.abs(_ref_clip.cpu().numpy() - _onnx_out[0]).max()
        _status = "OK" if _diff < 1e-2 else "WARN: large diff"
        if _diff >= 1e-2:
            print(f"  ONNX {ckpt_tag}: max|diff|={_diff:.3e} — export may be unreliable")
        del _sess
        print(f"  ONNX: {onnx_path.name}  "
              f"({onnx_path.stat().st_size / 1e6:.1f} MB)  max|diff|={_diff:.1e}  [{_status}]")

    del m, export_m, mel_tf
    gc.collect()

# ── OOF summary ───────────────────────────────────────────
has = ~np.isnan(oof_ns22[:, 0])
if has.sum() > 0:
    r_all = full_eval(Y_SC[has], oof_ns22[has], non_s22_mask_sc[has], TAXON_MASKS)
    print("\n" + "=" * 60)
    print("OOF RESULTS (best-macro checkpoints)")
    print("=" * 60)
    print(f"  macro AUC (all):     {r_all['macro_auc_all']:.4f}")
    print(f"  macro AUC (non-S22): {r_all['non_s22_macro']:.4f}")
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]:
        print(f"    {t:<12}: {r_all.get(f'non_s22_{t}', float('nan')):.4f}")

print("\nDone. Outputs in:", OUT_DIR)
