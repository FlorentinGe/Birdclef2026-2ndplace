# %% [code]
# %% [code]
# %% [code]
# # 🦜 BirdCLEF+ 2026 — CPU Inference & Submission Notebook
#
# **Inputs required (attach as Kaggle datasets):**
# - Competition data: `/kaggle/input/birdclef-2026/`
# - Saved checkpoint from training notebook: `/kaggle/input/<your-dataset>/swa_model.pth`
# - Saved label map from training notebook: `/kaggle/input/<your-dataset>/label_map.npy`
#
# **Output:** `submission.csv` — one row per 5-second chunk, one probability column per species.
#
# ---
#
# **Submission format:**
# Each `row_id` is `{soundscape_stem}_{end_second}`. There is one column per species (234 total).
# Each cell contains the predicted **probability** that the species was present in that 5-second window.
#
# **CPU speed strategy (target: < 30 min for 600 files):**
# - `torchaudio` for fast audio loading & resampling (C++ backend)
# - `torchaudio.transforms.MelSpectrogram` for vectorised mel computation over all chunks at once
# - All 12 chunks from each 1-min file batched into a single forward pass
# - Note: TorchScript export is not used — the SED model's dynamic feature-size
#   detection in `__init__` is incompatible with TorchScript tracing. Plain
#   `torch.inference_mode()` with `model.eval()` is fast enough on CPU.

# %% [markdown]
# ## 1. Imports

# %% [code]
import math
import re as _re
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from tqdm.auto import tqdm

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

try:
    import openvino as ov
    _OV_AVAILABLE = True
except ImportError:
    _OV_AVAILABLE = False

warnings.filterwarnings('ignore')
print(f'PyTorch      : {torch.__version__}')
print(f'Torchaudio   : {torchaudio.__version__}')
print(f'OnnxRuntime  : {ort.__version__ if _ORT_AVAILABLE else "not installed"}')
print(f'OpenVINO     : {ov.__version__ if _OV_AVAILABLE else "not installed"}')
print(f'Device       : CPU (required by competition rules)')

# %% [markdown]
# ## 2. Configuration

# %% [code]
class CFG:
    # ── Competition data ───────────────────────────────────────────────────
    BASE_DIR       = Path('/kaggle/input/competitions/birdclef-2026')
    TEST_DIR       = BASE_DIR / 'test_soundscapes'
    SAMPLE_SUB     = BASE_DIR / 'sample_submission.csv'

    # ── Saved artefacts from training notebook ─────────────────────────────
    # Update the dataset slug to match your uploaded dataset name.
    ARTEFACT_DIR   = Path('/kaggle/input/datasets/tennogh/birdnet-model-zoo/exp88')
    SOUNDSCAPE_DIR = Path('/kaggle/input/competitions/birdclef-2026/train_soundscapes')
    CHECKPOINT     = ARTEFACT_DIR / 'swa_model.pth'    # PyTorch fallback
    ONNX_MODEL     = ARTEFACT_DIR / 'swa_model.onnx'   # preferred; set to None to force PyTorch
    LABEL_MAP      = ARTEFACT_DIR / 'label_map.npy'

    # ── OnnxRuntime thread count ───────────────────────────────────────────
    # Kaggle CPU notebooks have 4 cores; using all of them for ORT intra-op
    # parallelism gives the best single-session throughput.
    ORT_THREADS    = 4

    OUTPUT_DIR     = Path('/kaggle/working')

    # ── Audio — must match training config exactly ─────────────────────────
    SR             = 32000
    CHUNK_DURATION = 5        # seconds per inference chunk
    N_FFT          = 2048
    HOP_LENGTH     = 512
    N_MELS         = 128
    FMIN           = 20
    FMAX           = 16000

    # ── Model — must match training config exactly ─────────────────────────
    MODEL_NAME     = 'efficientnet_b0'
    DROP_PATH_RATE = 0.15
    USE_GEM        = False    # set True if trained with --use_gem true
    GEM_P_INIT     = 3.0     # must match gem_p_init used during training
    IMAGENET_NORM  = False    # set True for ViT models trained with imagenet_norm=True
    STEM_STRIDE    = 2       # 1 if trained with --stem_stride 1, else 2 (default)

    # ── Temperature scaling ────────────────────────────────────────────────
    TAXONOMY_CSV       = BASE_DIR / 'taxonomy.csv'
    TAXON_TEMPERATURES = {'Aves': 1.0, 'Insecta': 1.0, 'Amphibia': 1.0}

    # ── Post-processing ────────────────────────────────────────────────────
    # Temporal TTA: re-run inference with the full waveform circularly rolled
    # by ±TTA_SHIFT samples, then average with the unshifted predictions.
    # Weight: (1 * orig + 2 * roll_fwd + 2 * roll_bwd) / 5
    # ±40,000 samples = ±1.25s at SR=32000. Zero training cost.
    TEMPORAL_TTA       = False
    TTA_SHIFT          = 40_000   # samples

    # Temporal smoothing: [0.20, 0.60, 0.20] kernel. Disabled — 0 to -0.001 LB.
    TEMPORAL_SMOOTHING = False

    # Delta-shift smoothing: gentle temporal smoothing across consecutive chunks.
    # new[t] = (1 - alpha) * old[t] + 0.5 * alpha * (old[t-1] + old[t+1])
    # OOF isolation tests: +0.003 to +0.007 AUC.
    DELTA_SMOOTH       = False
    DELTA_SMOOTH_ALPHA = 0.15

    # Rank-aware scaling: multiply each chunk's predictions by file_max^power
    # per species. Suppresses species the model is never confident about in
    # a given file. OOF isolation tests: +0.002 to +0.017 AUC.
    RANK_AWARE         = False
    RANK_AWARE_POWER   = 0.4

    # Confidence scaling: multiply each chunk by the per-species top-K mean
    # across all chunks of the file. Validated: +0.018–0.027 LB.
    CONF_SCALE         = False
    CONF_SCALE_TOP_K   = 2

    # Diel prior: class-level likelihood ratios from training soundscapes.
    # Disabled — causes -0.006 LB (double-counts soundscape training signal).
    DIEL_PRIOR         = False

    # ── Inference ─────────────────────────────────────────────────────────
    DEVICE     = 'cpu'    # competition constraint
    BATCH_SIZE = 12       # all chunks of one 1-min file in one pass

    # ── I/O prefetch ──────────────────────────────────────────────────────
    # Number of files to load ahead of the main thread in a background thread.
    # torchaudio.load + OGG decode releases the GIL, so this overlaps disk
    # I/O and audio decoding with ONNX inference at no extra CPU contention.
    # 2 is enough to hide latency; more wastes RAM (each file ~7 MB as float32).
    PREFETCH   = 2

    # ── Mel resize ────────────────────────────────────────────────────────
    # Set to an int (e.g. 512) to bilinear-resize the time axis of the mel
    # spectrogram before feeding the model. Must match the value used during
    # training. None = no resize (natural chunk length).
    MEL_W          = None

    # ── Derived ────────────────────────────────────────────────────────────
    @classmethod
    def chunk_frames(cls):
        """Number of mel time frames for one CHUNK_DURATION clip (after optional resize)."""
        natural = math.floor(cls.SR * cls.CHUNK_DURATION / cls.HOP_LENGTH) + 1
        return cls.MEL_W if cls.MEL_W is not None else natural

CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'Chunk frames : {CFG.chunk_frames()}  (mel T for {CFG.CHUNK_DURATION}s at hop={CFG.HOP_LENGTH})')

# %% [markdown]
# ## 3. Load Submission Template & Label Map

# %% [code]
sample_sub      = pd.read_csv(CFG.SAMPLE_SUB)
SPECIES_COLS    = [c for c in sample_sub.columns if c != 'row_id']
NUM_SUB_SPECIES = len(SPECIES_COLS)
print(f'Submission species : {NUM_SUB_SPECIES}')
print(f'Example row_id     : {sample_sub["row_id"].iloc[0]}')
sample_sub.head(3)

# %% [code]
# label_map.npy is saved as a dict {int -> species_str} via np.save.
# np.load returns a 0-d object array; .item() extracts the dict.
label_map   = np.load(CFG.LABEL_MAP, allow_pickle=True).item()
NUM_CLASSES = len(label_map)
idx2species = label_map    # {int: str}

UNIFORM_PRIOR = 1.0 / NUM_SUB_SPECIES

# model output index -> submission column index (-1 = not in submission)
model_to_sub = np.full(NUM_CLASSES, -1, dtype=np.int32)
sub_col_map  = {sp: j for j, sp in enumerate(SPECIES_COLS)}
for model_idx, sp in idx2species.items():
    if sp in sub_col_map:
        model_to_sub[model_idx] = sub_col_map[sp]

n_mapped = (model_to_sub >= 0).sum()
print(f'Model classes                        : {NUM_CLASSES}')
print(f'Mapped to submission columns         : {n_mapped}')
print(f'Submission columns on uniform prior  : {NUM_SUB_SPECIES - n_mapped}')

# %% [markdown]
# ## 4. Load Model
#
# Tries to load an ONNX model via OnnxRuntime (faster on CPU).
# Falls back to the inline PyTorch BirdCLEFModel if the ONNX file is absent
# or onnxruntime is not installed.
#
# Inline PyTorch class matches `Codebase/birdclef/model.py::BirdCLEFModel`
# and supports CNN + ViT backbones and optional GeM pooling.
# Must stay in sync with the codebase version.

# %% [code]
_VIT_PREFIXES = ('vit_', 'deit_', 'beit_', 'eva_')


def _is_vit(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _VIT_PREFIXES)


def _patch_stem_stride(backbone: nn.Module, backbone_name: str, stride: int) -> None:
    """Patch the first stride-2 stem conv. Must be called before the dummy forward pass."""
    if stride == 2:
        return
    s = (stride, stride)
    if hasattr(backbone, 'conv_stem'):
        backbone.conv_stem.stride = s
    elif hasattr(backbone, 'stem') and hasattr(backbone.stem, 'conv4'):
        backbone.stem.conv4.stride = s
    elif hasattr(backbone, 'stem') and hasattr(backbone.stem, 'conv'):
        backbone.stem.conv.stride = s
    else:
        print(f'WARNING: stem_stride={stride} has no effect on {backbone_name!r}')


class GeMPool1d(nn.Module):
    """GeM pooling over dim=2 of (B, C, H, W). p loaded from checkpoint."""
    def __init__(self, p_init: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)


class BirdCLEFModel(nn.Module):
    """
    SED architecture matching Codebase/birdclef/model.py::BirdCLEFModel.
    Only used as PyTorch fallback when ONNX is unavailable.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        n_mels: int,
        chunk_frames: int,
        drop_path_rate: float = 0.0,
        use_gem: bool = False,
        gem_p_init: float = 3.0,
        stem_stride: int = 2,
    ):
        super().__init__()
        self._is_vit  = _is_vit(model_name)
        self.num_classes = num_classes
        self._use_gem = use_gem
        if use_gem:
            self.gem_pool = GeMPool1d(p_init=gem_p_init)

        if self._is_vit:
            self.backbone = timm.create_model(
                model_name, pretrained=False, num_classes=0, global_pool='',
                in_chans=3, img_size=(n_mels, chunk_frames),
                drop_path_rate=drop_path_rate, dynamic_img_size=True,
            )
            with torch.no_grad():
                dummy      = torch.zeros(1, 3, n_mels, chunk_frames)
                tokens     = self.backbone(dummy)[:, 1:, :]
                n_features = tokens.shape[2]
                self.h_patches = n_mels // self.backbone.patch_embed.patch_size[0]
        else:
            self.backbone = timm.create_model(
                model_name, pretrained=False, num_classes=0, global_pool='',
                in_chans=3, drop_path_rate=drop_path_rate,
            )
            _patch_stem_stride(self.backbone, model_name, stem_stride)
            with torch.no_grad():
                dummy      = torch.zeros(1, 3, n_mels, chunk_frames)
                n_features = self.backbone(dummy).shape[1]

        self.fc      = nn.Linear(n_features, num_classes)
        self.att_fc  = nn.Linear(n_features, num_classes)
        self.bn      = nn.BatchNorm1d(n_features)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor):
        if self._is_vit:
            tokens = self.backbone(x)[:, 1:, :]              # (B, N, D)
            B, N, D = tokens.shape
            w_patches = N // self.h_patches
            feat = tokens.reshape(B, self.h_patches, w_patches, D).permute(0, 3, 1, 2)
            feat = self.gem_pool(feat) if self._use_gem else feat.mean(dim=2)
            feat = feat.permute(0, 2, 1)                     # (B, T, D)
        else:
            feat = self.backbone(x)                          # (B, C, H, W)
            feat = self.gem_pool(feat) if self._use_gem else feat.mean(dim=2)
            feat = feat.permute(0, 2, 1)                     # (B, T, C)

        B, T, C = feat.shape
        feat            = self.bn(feat.reshape(B * T, C)).reshape(B, T, C)
        feat            = self.dropout(feat)
        frame_logits    = self.fc(feat)
        att_weights     = torch.softmax(self.att_fc(feat), dim=1)
        clipwise_logits = frame_logits.mean(dim=1)
        att_clipwise    = (frame_logits * att_weights).sum(dim=1)
        return clipwise_logits, att_clipwise, frame_logits


# ── Backend selection: OpenVINO > OnnxRuntime > PyTorch ──────────────────────
_use_ov      = False
_use_onnx    = False
_ov_compiled = None
_ov_output   = None
_ort_session = None
model        = None

_onnx_path = CFG.ONNX_MODEL
if _OV_AVAILABLE and _onnx_path is not None and Path(_onnx_path).exists():
    try:
        _ov_core = ov.Core()
        _ov_core.set_property("CPU", {"PERFORMANCE_HINT": "LATENCY"})
        # Some ONNX exports set BatchNormalization training_mode=1 (opset-17 artefact).
        # OV rejects it; patch to inference mode (training_mode=0, 1 output) in memory.
        try:
            import onnx as _onnx_lib, io as _ov_io
            _proto   = _onnx_lib.load(str(_onnx_path))
            _patched = False
            for _nd in _proto.graph.node:
                if _nd.op_type == 'BatchNormalization':
                    for _at in _nd.attribute:
                        if _at.name == 'training_mode' and _at.i != 0:
                            _at.i    = 0
                            _patched = True
                    while len(_nd.output) > 1:
                        _nd.output.pop()
                        _patched = True
            if _patched:
                print('  Patched BatchNormalization training_mode → 0')
            _buf = _ov_io.BytesIO()
            _onnx_lib.save(_proto, _buf)
            _ov_nfnet = _ov_core.compile_model(_ov_core.read_model(_buf.getvalue()), "CPU")
        except ImportError:
            # onnx package not available — load directly (may still fail if BN is training-mode)
            _ov_nfnet = _ov_core.compile_model(_ov_core.read_model(str(_onnx_path)), "CPU")
        _use_ov_nfnet = True
        print(f'Backend       : OpenVINO  ({_onnx_path.name})')
    except Exception as _ov_exc:
        print(f'OpenVINO load failed ({_ov_exc}) — trying OnnxRuntime')

if not _use_ov_nfnet and _ORT_AVAILABLE and _onnx_path is not None and Path(_onnx_path).exists():
    _sess_opts = ort.SessionOptions()
    _sess_opts.intra_op_num_threads      = CFG.ORT_THREADS
    _sess_opts.graph_optimization_level  = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _ort_session = ort.InferenceSession(
        str(_onnx_path),
        sess_options=_sess_opts,
        providers=['CPUExecutionProvider'],
    )
    _use_onnx = True
    print(f'Backend       : OnnxRuntime  ({_onnx_path.name})')
    print(f'ORT threads   : {CFG.ORT_THREADS}')

if not _use_ov_nfnet and not _use_onnx:
    if not _ORT_AVAILABLE:
        print('Backend       : PyTorch  (onnxruntime not installed)')
    elif _onnx_path is None:
        print('Backend       : PyTorch  (CFG.ONNX_MODEL is None)')
    else:
        print(f'Backend       : PyTorch  ({_onnx_path.name} not found)')
    model = BirdCLEFModel(
        model_name     = CFG.MODEL_NAME,
        num_classes    = NUM_CLASSES,
        n_mels         = CFG.N_MELS,
        chunk_frames   = CFG.chunk_frames(),
        drop_path_rate = CFG.DROP_PATH_RATE,
        use_gem        = CFG.USE_GEM,
        gem_p_init     = CFG.GEM_P_INIT,
    )
    model.load_state_dict(torch.load(CFG.CHECKPOINT, map_location='cpu'))
    model.eval()

print(f'Model classes : {NUM_CLASSES}')
print(f'GeM pooling   : {CFG.USE_GEM}')

# %% [markdown]
# ## 4b. Temperature Scaling Setup
#
# Builds a per-species temperature vector in model-output order.
# Applied as `sigmoid(logits / temp)` in `_forward_probs`.
# All values currently 1.0 (no scaling). Edit CFG.TAXON_TEMPERATURES to tune.

# %% [code]
taxonomy  = pd.read_csv(CFG.TAXONOMY_CSV)
taxon_map = taxonomy.set_index('primary_label')['class_name'].to_dict()
taxon_map = {str(k): v for k, v in taxon_map.items()}

model_temp = torch.ones(NUM_CLASSES)
for model_idx, sp in idx2species.items():
    cls = taxon_map.get(str(sp), 'Aves')
    model_temp[model_idx] = CFG.TAXON_TEMPERATURES.get(cls, 1.0)

temp_counts = Counter(model_temp.tolist())
print('Temperature vector built:')
for t, n in sorted(temp_counts.items()):
    print(f'  T={t:.2f}  →  {n} species')

# %% [markdown]
# ## 4c. Diel Prior Setup
#
# Class-level likelihood ratios LR(class | hour) derived from 739 labelled
# 5-second chunks across 66 training soundscape files.
# Disabled by default — causes -0.006 LB (double-counts soundscape training signal).

# %% [code]
_DIEL_LR = {
    'Amphibia': {0: 0.776, 1: 1.118, 2: 1.274, 3: 0.333, 4: 0.441,
                 6: 0.100, 7: 0.100, 18: 1.256, 19: 1.289,
                 20: 1.299, 21: 1.300, 22: 1.297, 23: 1.300},
    'Aves':     {0: 1.281, 1: 1.440, 2: 0.942, 3: 0.746, 4: 1.741,
                 6: 3.447, 7: 1.647, 18: 3.348, 19: 0.141,
                 20: 0.102, 21: 0.100, 22: 0.100, 23: 1.549},
    'Insecta':  {0: 0.100, 1: 0.100, 2: 0.100, 3: 3.271, 4: 2.908,
                 6: 4.347, 7: 4.332, 18: 0.169, 19: 2.908,
                 20: 0.100, 21: 0.100, 22: 0.100, 23: 0.100},
}
_HOUR_RE = _re.compile(r'_(\d{8})_(\d{6})')

_diel_log_lr_by_hour = {}
for _h in range(24):
    _log_lr = np.zeros(NUM_CLASSES, dtype=np.float32)
    for _model_idx, _sp in idx2species.items():
        _cls = taxon_map.get(str(_sp))
        if _cls in _DIEL_LR:
            _lr = _DIEL_LR[_cls].get(_h, 1.0)
            _log_lr[_model_idx] = np.log(_lr)
    _diel_log_lr_by_hour[_h] = _log_lr


def apply_diel_prior(probs: np.ndarray, filename: str) -> np.ndarray:
    """
    Adjust (N_chunks, NUM_CLASSES) probabilities by class-level diel LR.
    Extracts hour from filename, applies log-LR shift in logit space.
    Returns probs unchanged if hour cannot be parsed.
    """
    m = _HOUR_RE.search(filename)
    if m is None:
        return probs
    hour   = int(m.group(2)[:2])
    log_lr = _diel_log_lr_by_hour.get(hour, _diel_log_lr_by_hour[0] * 0)
    if not np.any(log_lr):
        return probs
    eps    = 1e-6
    logits = np.log(np.clip(probs, eps, 1 - eps) / np.clip(1 - probs, eps, 1 - eps))
    return (1.0 / (1.0 + np.exp(-(logits + log_lr[np.newaxis, :])))).astype(np.float32)

_test_hour  = 6
_test_probs = np.full((1, NUM_CLASSES), 0.3, dtype=np.float32)
_adjusted   = apply_diel_prior(_test_probs, f'BC2026_Test_0001_S05_20250101_{_test_hour:02d}0000.ogg')
print(f'Diel prior sanity check at hour {_test_hour}:')
for _sp, _cls in [('rufhor2', 'Aves'), ('22985', 'Amphibia'), ('1161364', 'Insecta')]:
    _idx = next((i for i, s in idx2species.items() if str(s) == _sp), None)
    if _idx is not None:
        print(f'  {_sp} ({_cls}): {_test_probs[0, _idx]:.3f} → {_adjusted[0, _idx]:.3f}  '
              f'(LR={_DIEL_LR.get(_cls, {}).get(_test_hour, 1.0):.3f})')
print('Diel prior setup complete.')

# %% [markdown]
# ## 5. Audio Utilities

# %% [code]
mel_transform   = T.MelSpectrogram(
    sample_rate = CFG.SR,
    n_fft       = CFG.N_FFT,
    hop_length  = CFG.HOP_LENGTH,
    n_mels      = CFG.N_MELS,
    f_min       = CFG.FMIN,
    f_max       = CFG.FMAX,
)
amplitude_to_db = T.AmplitudeToDB(top_db=80)

# ImageNet normalisation constants — only applied when CFG.IMAGENET_NORM is True
# (ViT models trained with imagenet_norm=True in the training config).
_IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IN_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_and_chunk(filepath: Path) -> tuple[torch.Tensor, int]:
    """
    Load a soundscape, resample if needed, split into non-overlapping
    CHUNK_DURATION-second chunks.

    Returns:
        chunks   : (N_chunks, chunk_len) float32 tensor
        n_chunks : number of chunks
    """
    waveform, orig_sr = torchaudio.load(str(filepath))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sr != CFG.SR:
        waveform = torchaudio.functional.resample(waveform, orig_sr, CFG.SR)

    chunk_len = CFG.SR * CFG.CHUNK_DURATION
    total_len = waveform.shape[-1]
    remainder = total_len % chunk_len
    if remainder:
        waveform = F.pad(waveform, (0, chunk_len - remainder))

    chunks = waveform.squeeze(0).reshape(-1, chunk_len)
    return chunks, chunks.shape[0]


def chunks_to_mel_batch(chunks: torch.Tensor) -> torch.Tensor:
    """
    Convert (N_chunks, chunk_len) → (N_chunks, 3, N_MELS, T_frames).
    Vectorised; no Python loop per chunk.
    Applies ImageNet normalisation when CFG.IMAGENET_NORM is True.
    """
    mel = mel_transform(chunks)                           # (N, n_mels, T)
    mel = amplitude_to_db(mel)
    mn  = mel.flatten(1).min(dim=1).values[:, None, None]
    mx  = mel.flatten(1).max(dim=1).values[:, None, None]
    mel = (mel - mn) / (mx - mn + 1e-6)
    mel = mel.unsqueeze(1).repeat(1, 3, 1, 1)            # (N, 3, n_mels, T)
    if CFG.MEL_W is not None:
        mel = F.interpolate(mel, size=(CFG.N_MELS, CFG.MEL_W), mode='bilinear', align_corners=False)
    if CFG.IMAGENET_NORM:
        mel = (mel - _IN_MEAN) / _IN_STD
    return mel

# %% [markdown]
# ## 6. Inference Loop

# %% [code]
_model_temp_np = None   # lazily set to model_temp.numpy() on first ORT call


def _forward_probs(batch: torch.Tensor) -> np.ndarray:
    """
    Forward pass → probabilities (N_chunks, NUM_CLASSES).
    Uses OpenVINO > OnnxRuntime > PyTorch fallback.
    Temperature scaling is applied in the same dtype as the backend output.
    """
    global _model_temp_np
    if _use_ov_nfnet:
        if _model_temp_np is None:
            _model_temp_np = model_temp.numpy()
        logits = list(_ov_nfnet({"mel_spec": batch.numpy()}).values())[0]
        return (1.0 / (1.0 + np.exp(-logits / _model_temp_np))).astype(np.float32)
    elif _use_onnx:
        if _model_temp_np is None:
            _model_temp_np = model_temp.numpy()
        logits = _ort_session.run(None, {'mel_spec': batch.numpy()})[0]  # (N, C) float32
        return (1.0 / (1.0 + np.exp(-logits / _model_temp_np))).astype(np.float32)
    else:
        with torch.inference_mode():
            _, att_clipwise, _ = model(batch)
            return torch.sigmoid(att_clipwise / model_temp).numpy()


def _roll_chunks(chunks: torch.Tensor, shift: int) -> torch.Tensor:
    """Circularly roll the full waveform by `shift` samples, then re-chunk."""
    flat     = chunks.reshape(-1)
    flat     = torch.roll(flat, shifts=shift)
    n_chunks = chunks.shape[0]
    return flat.reshape(n_chunks, -1)


def _process_chunks(filepath: Path, chunks: torch.Tensor, n_chunks: int) -> pd.DataFrame:
    """
    Compute-only part of inference: mel → ONNX → post-processing → DataFrame.
    Separated from I/O so the main loop can overlap loading with inference.

    Post-processing order:
      1. Temporal TTA        — average over ±1.25s circular rolls (1:2:2 weights)
      2. Diel prior          — class-level LR in logit space (disabled: -0.006 LB)
      3. Temporal smoothing  — [0.20, 0.60, 0.20] kernel (disabled: 0 to -0.001 LB)
      4. Delta-shift smooth  — gentle [0.075, 0.85, 0.075] kernel (+0.003 to +0.007)
      5. Rank-aware scaling  — multiply by file_max^power per species (+0.002 to +0.017)
      6. Confidence scaling  — multiply by per-species top-K mean (+0.018–0.027 LB)
    """
    stem  = filepath.stem

    probs = _forward_probs(chunks_to_mel_batch(chunks))   # (N, NUM_CLASSES)

    if CFG.TEMPORAL_TTA:
        probs_fwd = _forward_probs(chunks_to_mel_batch(_roll_chunks(chunks, +CFG.TTA_SHIFT)))
        probs_bwd = _forward_probs(chunks_to_mel_batch(_roll_chunks(chunks, -CFG.TTA_SHIFT)))
        probs = (probs + 2.0 * probs_fwd + 2.0 * probs_bwd) / 5.0

    if CFG.DIEL_PRIOR:
        probs = apply_diel_prior(probs, filepath.name)

    if CFG.TEMPORAL_SMOOTHING:
        padded = np.concatenate([probs[:1], probs, probs[-1:]], axis=0)
        probs  = 0.20 * padded[:-2] + 0.60 * padded[1:-1] + 0.20 * padded[2:]

    if CFG.DELTA_SMOOTH:
        a     = CFG.DELTA_SMOOTH_ALPHA
        prev  = np.concatenate([probs[:1],  probs[:-1]], axis=0)
        nxt   = np.concatenate([probs[1:],  probs[-1:]], axis=0)
        probs = (1 - a) * probs + 0.5 * a * (prev + nxt)

    if CFG.RANK_AWARE:
        file_max = probs.max(axis=0)
        scale    = np.power(np.clip(file_max, 1e-9, 1.0), CFG.RANK_AWARE_POWER)
        probs    = probs * scale[None, :]

    if CFG.CONF_SCALE:
        top_k_mean = np.sort(probs, axis=0)[-CFG.CONF_SCALE_TOP_K:].mean(axis=0)
        probs      = probs * top_k_mean[None, :]

    out = np.full((n_chunks, NUM_SUB_SPECIES), UNIFORM_PRIOR, dtype=np.float32)
    for model_idx in range(NUM_CLASSES):
        sub_idx = model_to_sub[model_idx]
        if sub_idx >= 0:
            out[:, sub_idx] = probs[:, model_idx]

    row_ids = [f'{stem}_{(i + 1) * CFG.CHUNK_DURATION}' for i in range(n_chunks)]
    df = pd.DataFrame(out, columns=SPECIES_COLS)
    df.insert(0, 'row_id', row_ids)
    return df


# ── Main inference loop ────────────────────────────────────────────────────
test_files = sorted(CFG.TEST_DIR.glob('*.ogg'))
print(f'Test files found: {len(test_files)}')

if len(test_files) == 0:
    print('Hidden test not mounted. Dry-run on first 4 train soundscapes.')
    test_files = sorted(CFG.SOUNDSCAPE_DIR.glob('*.ogg'))[:5]

all_preds = []
t0        = time.time()

# Prefetch pipeline: a thread pool keeps PREFETCH load_and_chunk futures in
# flight while the main thread runs mel + ONNX + post-processing.
# torchaudio.load and OGG decoding release the GIL, so this overlaps disk
# I/O with compute without competing with ORT's CPU threads.
with ThreadPoolExecutor(max_workers=CFG.PREFETCH) as pool:
    # pending is a list of (filepath, future); we append to it while iterating,
    # which is valid in Python — the for-loop will see newly appended items.
    pending  = []
    files    = list(test_files)
    n_files  = len(files)
    next_idx = 0

    # Seed: submit first PREFETCH files
    for f in files[:CFG.PREFETCH]:
        pending.append((f, pool.submit(load_and_chunk, f)))
    next_idx = CFG.PREFETCH

    for filepath, future in tqdm(pending, total=n_files, desc='Inference'):
        # Submit the next file immediately so it loads while we do inference
        if next_idx < n_files:
            pending.append((files[next_idx], pool.submit(load_and_chunk, files[next_idx])))
            next_idx += 1

        chunks, n_chunks = future.result()
        all_preds.append(_process_chunks(filepath, chunks, n_chunks))

elapsed  = time.time() - t0
n_passes = 3 if CFG.TEMPORAL_TTA else 1
print(f'\nDone in {elapsed / 60:.1f} min  ({elapsed / len(test_files):.2f} s/file, {n_passes} pass(es)/file)')

# %% [markdown]
# ## 7. Build & Validate Submission

# %% [code]
preds = pd.concat(all_preds, ignore_index=True)

submission = (
    sample_sub[['row_id']]
    .merge(preds, on='row_id', how='left')
)
for sp in SPECIES_COLS:
    submission[sp] = submission[sp].fillna(UNIFORM_PRIOR)

assert list(submission.columns) == list(sample_sub.columns), \
    'Column mismatch with sample submission!'
assert submission.shape[0] == sample_sub.shape[0], \
    f'Row count mismatch: got {submission.shape[0]}, expected {sample_sub.shape[0]}'
assert submission[SPECIES_COLS].isnull().sum().sum() == 0, \
    'NaN values found in submission!'
assert ((submission[SPECIES_COLS] >= 0) & (submission[SPECIES_COLS] <= 1)).all().all(), \
    'Probabilities outside [0, 1]!'

print(f'Submission shape  : {submission.shape}')
print(f'Rows              : {submission.shape[0]}')
print(f'Species columns   : {submission.shape[1] - 1}')
print(f'All checks passed ✓')
submission.head(3)

# %% [code]
out_path = CFG.OUTPUT_DIR / 'submission.csv'
submission.to_csv(out_path, index=False)
print(f'Saved → {out_path}')

# %% [markdown]
# ## 8. Probability Distribution Check

# %% [code]
import matplotlib.pyplot as plt

mean_probs = submission[SPECIES_COLS].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(range(len(mean_probs)), mean_probs.values, color='steelblue', width=1.0)
ax.axhline(UNIFORM_PRIOR, color='red', linestyle='--', linewidth=1,
           label=f'Uniform prior ({UNIFORM_PRIOR:.4f})')
ax.set_title('Mean predicted probability per species (sorted)')
ax.set_xlabel('Species rank')
ax.set_ylabel('Mean probability')
ax.legend()
plt.tight_layout()
plt.show()

print('Top 5 most predicted species:')
print(mean_probs.head())
print('\nBottom 5 least predicted species:')
print(mean_probs.tail())

# %% [markdown]
# ## 9. Free Memory for Ensembling

# %% [code]
import gc

# Keep `preds` (raw per-model predictions) for ensemble combination.
# Delete the model session, audio pipeline, and intermediate buffers.
exp68c_preds = preds

if _use_ov_nfnet and _ov_nfnet is not None:
    del _ov_nfnet
elif _use_onnx and _ort_session is not None:
    del _ort_session
elif model is not None:
    del model
del mel_transform, amplitude_to_db
del all_preds, preds, submission
del model_temp, _model_temp_np, _diel_log_lr_by_hour
gc.collect()
print('exp68c_preds ready for ensembling.')

